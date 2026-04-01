"""KnowledgeBuilder — triple-write pipeline: Vault + Memory + Knowledge Graph.

Takes FetchResult objects from ResearchAgent and persists the content via
three complementary MCP tool calls:

1. **Vault**: Full document stored for long-term retrieval.
2. **Memory**: Chunked semantic memories for RAG.
3. **Graph**: Entities and relations extracted via LLM.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple

from jarvis.evolution.research_agent import FetchResult
from jarvis.memory.consolidation import ContentDeduplicator
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

__all__ = [
    "BuildResult",
    "KnowledgeBuilder",
    "_is_usable_content",
    "_score_source_confidence",
    "_parse_llm_json",
]


@dataclass
class BuildResult:
    """Outcome of building knowledge from a single FetchResult."""

    vault_path: str = ""
    chunks_created: int = 0
    entities_created: int = 0
    relations_created: int = 0
    claims_extracted: int = 0
    errors: List[str] = field(default_factory=list)


# Patterns that indicate garbage entities extracted from PDF metadata,
# dictionary/navigation pages, or generic web boilerplate.
_GARBAGE_PATTERNS = re.compile(
    r"^("
    r"PDF-[\d.]+"  # PDF version strings
    r"|Linearized.*"  # PDF linearization marker
    r"|\d+ 0 obj"  # PDF object references
    r"|trailer"  # PDF trailer
    r"|MediaBox"  # PDF page dimensions
    r"|XObject|Font|Page|Root"  # PDF internal types
    r"|Rechtschreibung"  # Duden dictionary metadata
    r"|Grammatik"  # Duden dictionary metadata
    r"|Synonyme"  # Duden dictionary metadata
    r"|Worttrennung"  # Duden dictionary metadata
    r"|Aussprache"  # Duden dictionary metadata
    r"|Betonung"  # Duden dictionary metadata
    r"|Cookie"  # Cookie consent noise
    r"|Datenschutz"  # Privacy policy noise
    r"|Impressum"  # Imprint noise
    r"|Newsletter"  # Newsletter signup noise
    r"|Inhaltsverzeichnis"  # Table of contents
    r"|Breadcrumb"  # Navigation noise
    r"|Login|Logout"  # Auth elements
    r"|Warenkorb|Cart"  # Shopping cart noise
    r"|Hauptmenü"  # Wikipedia/site navigation
    r"|Themenportale"  # Wikipedia navigation
    r"|Zufälliger Artikel"  # Wikipedia navigation
    r"|Autorenportal"  # Wikipedia navigation
    r"|Letzte Änderungen"  # Wikipedia navigation
    r"|Kontakt"  # Generic site element
    r"|Hilfe"  # Generic site element
    r"|Wiktionary|Wikibooks|Wikiquote|Wikisource|Wikiversity|Wikivoyage|Commons"  # Wikimedia siblings
    r"|Wikimedia Foundation"  # Wikimedia meta
    r"|Wikipedia"  # Wikipedia itself (not useful as entity)
    r"|Catalog|Pages|Stream|Metadata"  # PDF internal structure objects
    r"|FlateDecode|DeviceRGB|DeviceCMYK|ObjStm"  # PDF codec/colorspace internals
    r"|ArialMT|TimesNewRomanPSMT|Helvetica"  # Embedded font names
    r"|Image"  # Generic PDF image reference
    r"|Uniform Resource Locator"  # URL as entity — too generic
    r"|Administrator|Löschkriterien"  # Wikipedia admin/deletion noise
    r"|XRef"  # PDF cross-reference table
    r"|Objekt \d+"  # PDF object references (e.g. "Objekt 4759")
    r"|Root \d+"  # PDF root object references (e.g. "Root 4760")
    r"|Info \d+"  # PDF info object references (e.g. "Info 1562")
    r"|obj \d+"  # PDF object IDs
    r"|endobj|endstream|xref"  # PDF structure markers
    r"|Type1|TrueType|CIDFont"  # PDF font type names
    r")$",
    re.IGNORECASE,
)

# Entity names that are too generic to be useful
_TOO_GENERIC = {
    "grundlage",
    "methode",
    "anwendung",
    "beispiel",
    "definition",
    "ergebnis",
    "information",
    "inhalt",
    "thema",
    "uebersicht",
    "seite",
    "kapitel",
    "abschnitt",
    "tabelle",
    "abbildung",
    "quelle",
    "link",
    "download",
    "suche",
    "startseite",
    "artikel",
    "image",
    "catalog",
    "pages",
    "stream",
    "metadata",
    "xref",
    "root",
    "info",
    "objekt",
    "endobj",
    "trailer",
}


def _is_valid_entity(entity: dict) -> bool:
    """Reject garbage entities from PDF metadata, dictionaries, navigation."""
    name = entity.get("name", "").strip()
    if not name or len(name) < 2:
        return False
    # Reject PDF/web boilerplate patterns
    if _GARBAGE_PATTERNS.match(name):
        return False
    # Reject too-generic single words
    if name.lower() in _TOO_GENERIC:
        return False
    # Reject pure version numbers or object IDs
    if re.match(r"^[\d.]+$", name):
        return False
    # Reject names that are just section markers (§1, §2, etc.)
    if re.match(r"^§\d+$", name):
        return False
    return True


# Compiled pattern for detecting PDF structural artifacts in source text.
# Applied per-line: if >30% of lines match, the source is rejected.
_PDF_ARTIFACT_LINE_RE = re.compile(
    r"(\d+ \d+ obj"
    r"|endobj|endstream|xref|trailer"
    r"|stream\s*$"
    r"|/Type\b|/Filter\b|/Length\b|/Pages\b|/Root\b"
    r"|FlateDecode|MediaBox|DeviceRGB|DeviceCMYK"
    r"|%%EOF)",
)


def _is_usable_content(text: str, min_chars: int = 200) -> tuple[bool, str]:
    """Check whether fetched text is worth indexing.

    Returns:
        (usable, reason) — reason is 'ok' on success or a short tag on rejection.
    """
    cleaned = " ".join(text.split())
    if len(cleaned) < min_chars:
        return False, "too_short"

    lines = text.splitlines()
    non_empty = [l for l in lines if l.strip()]
    if not non_empty:
        return False, "too_short"

    garbage_count = sum(1 for l in non_empty if _PDF_ARTIFACT_LINE_RE.search(l))
    ratio = garbage_count / len(non_empty)
    if ratio > 0.3:
        return False, f"pdf_artifacts_{ratio:.0%}"

    return True, "ok"


# ── Source Confidence Scoring ────────────────────────────────────────

_TRUSTED_DOMAINS: dict[str, float] = {
    ".gov.de": 0.9,
    ".bund.de": 0.9,
    ".europa.eu": 0.9,
    "gesetze-im-internet.de": 0.9,
    "dejure.org": 0.9,
    "bafin.de": 0.9,
    "bundesbank.de": 0.9,
    "bsi.bund.de": 0.9,
    "owasp.org": 0.8,
    "wikipedia.org": 0.7,
    "arxiv.org": 0.7,
    "springer.com": 0.7,
    "nature.com": 0.7,
    "heise.de": 0.7,
    "golem.de": 0.7,
}
_LOW_TRUST_SIGNALS: list[str] = ["blog", "medium.com", "reddit.com", "forum", "quora.com"]
_DEFAULT_CONFIDENCE: float = 0.5


def _score_source_confidence(url: str) -> float:
    """Derive confidence from the source URL domain.

    Trusted government/academic sources score higher. Blogs and forums
    score lower. Unknown domains get a neutral 0.5.
    """
    if not url:
        return _DEFAULT_CONFIDENCE

    url_lower = url.lower()

    # Check trusted domains (longest suffix match first for specificity)
    for domain, score in sorted(_TRUSTED_DOMAINS.items(), key=lambda x: -len(x[0])):
        if domain in url_lower:
            return score

    # Check low-trust signals
    for signal in _LOW_TRUST_SIGNALS:
        if signal in url_lower:
            return 0.3

    return _DEFAULT_CONFIDENCE


# ── LLM JSON Parsing with Fallback ──────────────────────────────────


def _parse_llm_json(raw: str, fallback_content: str, url: str) -> dict:
    """Parse LLM response with graceful degradation.

    4-tier strategy:
    1. Direct json.loads
    2. Extract ```json ... ``` markdown block
    3. Regex extraction of individual fields
    4. Fallback defaults using original content
    """
    # Tier 1: direct parse
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "summary" in data:
            return _validate_parsed(data)
    except (json.JSONDecodeError, ValueError):
        pass

    # Tier 2: markdown code block
    md_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
    if md_match:
        try:
            data = json.loads(md_match.group(1))
            if isinstance(data, dict) and "summary" in data:
                return _validate_parsed(data)
        except (json.JSONDecodeError, ValueError):
            pass

    # Tier 3: regex extraction of individual fields
    summary_m = re.search(r'"summary"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
    type_m = re.search(r'"memory_type"\s*:\s*"(\w+)"', raw)
    useful_m = re.search(r'"is_useful"\s*:\s*(true|false)', raw, re.IGNORECASE)
    tags_m = re.search(r'"tags"\s*:\s*\[(.*?)\]', raw)

    if summary_m:
        tags: list[str] = []
        if tags_m:
            tags = re.findall(r'"([^"]+)"', tags_m.group(1))
        return _validate_parsed(
            {
                "summary": summary_m.group(1),
                "memory_type": type_m.group(1) if type_m else "semantic",
                "tags": tags,
                "is_useful": (useful_m.group(1).lower() == "true") if useful_m else True,
            }
        )

    # Tier 4: fallback
    return {
        "summary": fallback_content[:800],
        "memory_type": "semantic",
        "tags": [],
        "is_useful": True,
    }


def _validate_parsed(data: dict) -> dict:
    """Ensure parsed dict has all required fields with correct types."""
    return {
        "summary": str(data.get("summary", ""))[:3000],
        "memory_type": str(data.get("memory_type", "semantic"))
        if data.get("memory_type") in ("semantic", "procedural", "episodic")
        else "semantic",
        "tags": [str(t) for t in data.get("tags", []) if isinstance(t, str)][:10],
        "is_useful": bool(data.get("is_useful", True)),
    }


_ENTITY_EXTRACTION_PROMPT = """\
Analysiere den folgenden Text und extrahiere Entitaeten und Beziehungen.

Regeln:
- Maximal 10 Entitaeten, maximal 10 Beziehungen.
- Entitaets-Typen: person, law, concept, organization, product, event
- Beziehungs-Typen: regelt, teil_von, gehoert_zu, definiert, referenziert

Antworte NUR mit validem JSON in diesem Format:
{{
  "entities": [
    {{"name": "...", "type": "...", "attributes": {{...}}}}
  ],
  "relations": [
    {{"source": "...", "relation": "...", "target": "..."}}
  ]
}}

Text:
{text}
"""


class KnowledgeBuilder:
    """Builds structured knowledge from fetched web content.

    Triple-write pipeline:
    1. vault_save — full document into Vault
    2. chunk_text + save_to_memory — semantic chunks
    3. extract_entities + add_entity / add_relation — knowledge graph

    When ``skip_entity_extraction=True``, step 3 is queued instead of
    skipped.  Call :meth:`drain_entity_queue` later (when the GPU is
    free) to process all pending entity extractions.
    """

    def __init__(
        self,
        mcp_client: Any,
        llm_fn: Optional[Callable] = None,
        goal_slug: str = "",
        knowledge_validator: Any = None,
        goal_index: Any = None,
        entity_llm_fn: Optional[Callable] = None,
        memory_manager: Any = None,
    ) -> None:
        self._mcp = mcp_client
        self._llm_fn = llm_fn
        # Entity extraction uses a smaller/faster model (e.g. qwen3:8b)
        # to avoid blocking the GPU for 10+ minutes per document.
        # Falls back to the main llm_fn if not provided.
        self._entity_llm_fn = entity_llm_fn or llm_fn
        self._goal_slug = goal_slug
        self._validator = knowledge_validator
        self._goal_index = goal_index
        self._entity_queue: List[str] = []  # Deferred texts for entity extraction
        self._memory_manager = memory_manager
        self._identity_dedup = ContentDeduplicator(similarity_threshold=0.85)
        self._identity_seen_hashes: set[str] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def build(
        self,
        fetch_result: FetchResult,
        *,
        skip_entity_extraction: bool = False,
        min_content_chars: int = 200,
        already_summarized: bool = False,
    ) -> BuildResult:
        """Run the triple-write pipeline for a single FetchResult.

        Args:
            skip_entity_extraction: If True, skip the LLM-based entity
                extraction (step 3). Useful for cron-triggered builds where
                the GPU should not be blocked for 2-10 minutes per document.
                Vault save + memory chunking still run normally.
            min_content_chars: Minimum character count for the quality gate.
                Default 200 for web-fetched content. ATL synthesis notes
                use a lower threshold (100) since they are pre-processed.
        """
        result = BuildResult()

        if fetch_result.error or not fetch_result.text:
            result.errors.append(fetch_result.error or "Empty text in FetchResult")
            return result

        # Content quality gate: reject PDF artifacts and too-short text
        usable, reason = _is_usable_content(fetch_result.text, min_chars=min_content_chars)
        if not usable:
            log.info(
                "content_rejected",
                url=fetch_result.url[:80],
                reason=reason,
            )
            result.errors.append(f"Content rejected: {reason}")
            return result

        # 1. Vault save
        try:
            vault_resp = await self._mcp.call_tool(
                "vault_save",
                {
                    "title": fetch_result.title or fetch_result.url,
                    "content": fetch_result.text,
                    "tags": [self._goal_slug, fetch_result.source_type],
                    "folder": f"wissen/{self._goal_slug}",
                    "sources": fetch_result.url,
                },
            )
            result.vault_path = f"wissen/{self._goal_slug}/{fetch_result.title or fetch_result.url}"
        except Exception as exc:
            result.errors.append(f"vault_save failed: {exc}")

        # 2. Chunking + memory
        chunks = self.chunk_text(fetch_result.text)
        for i, chunk in enumerate(chunks):
            try:
                await self._mcp.call_tool(
                    "save_to_memory",
                    {
                        "content": chunk,
                        "tier": "semantic",
                        "source_path": f"wissen/{self._goal_slug}/{fetch_result.url}#chunk{i}",
                    },
                )
                result.chunks_created += 1
                # Also write to goal-scoped index
                if self._goal_index:
                    try:
                        self._goal_index.add_chunk(chunk, source_url=fetch_result.url)
                    except Exception:
                        log.debug("goal_index_add_chunk_failed", exc_info=True)
            except Exception as exc:
                result.errors.append(f"save_to_memory chunk {i} failed: {exc}")

        # 3. Entity extraction + graph
        if self._llm_fn is not None and skip_entity_extraction:
            # Queue for later processing instead of blocking GPU now
            self._entity_queue.append(fetch_result.text[:4000])
            log.debug(
                "entity_extraction_queued",
                url=fetch_result.url[:50],
                queue_size=len(self._entity_queue),
            )
        elif self._llm_fn is not None:
            entities, relations = await self.extract_entities(fetch_result.text)
            for entity in entities:
                try:
                    attrs = dict(entity.get("attributes", {}))
                    attrs["domain"] = self._goal_slug
                    await self._mcp.call_tool(
                        "add_entity",
                        {
                            "name": entity["name"],
                            "entity_type": entity["type"],
                            "attributes": json.dumps(attrs, ensure_ascii=False),
                            "source_file": fetch_result.url,
                        },
                    )
                    result.entities_created += 1
                    # Also write to goal-scoped index
                    if self._goal_index:
                        try:
                            self._goal_index.add_entity(
                                entity["name"], entity["type"], attrs, fetch_result.url
                            )
                        except Exception:
                            log.debug("goal_index_add_entity_failed", exc_info=True)
                except Exception as exc:
                    result.errors.append(f"add_entity failed: {exc}")

            for rel in relations:
                try:
                    await self._mcp.call_tool(
                        "add_relation",
                        {
                            "source_name": rel["source"],
                            "relation_type": rel["relation"],
                            "target_name": rel["target"],
                            "attributes": json.dumps({}, ensure_ascii=False),
                        },
                    )
                    result.relations_created += 1
                    # Also write to goal-scoped index
                    if self._goal_index:
                        try:
                            self._goal_index.add_relation(
                                rel["source"], rel["relation"], rel["target"]
                            )
                        except Exception:
                            log.debug("goal_index_add_relation_failed", exc_info=True)
                except Exception as exc:
                    result.errors.append(f"add_relation failed: {exc}")

        # 4. Claims: extract and track factual claims for validation
        if self._validator:
            try:
                claims = await self._validator.extract_claims(
                    text=fetch_result.text[:3000],
                    source_url=fetch_result.url,
                    goal_slug=self._goal_slug,
                )
                result.claims_extracted = len(claims)
                log.info(
                    "knowledge_claims_tracked",
                    url=fetch_result.url[:50],
                    claims=len(claims),
                )
            except Exception:
                log.debug("knowledge_claims_extraction_failed", exc_info=True)

        # 5. Identity Memory: summarize + classify + store
        try:
            await self._summarize_for_identity(
                text=fetch_result.text,
                url=fetch_result.url,
                already_summarized=already_summarized,
            )
        except Exception:
            log.debug("identity_summarize_failed", exc_info=True)

        return result

    @property
    def entity_queue_size(self) -> int:
        """Number of texts waiting for entity extraction."""
        return len(self._entity_queue)

    async def drain_entity_queue(self, max_items: int = 5) -> int:
        """Process queued entity extractions (call when GPU is free).

        Args:
            max_items: Max items to process in one batch.

        Returns:
            Number of items successfully processed.
        """
        if not self._llm_fn or not self._entity_queue:
            return 0

        processed = 0
        while self._entity_queue and processed < max_items:
            text = self._entity_queue.pop(0)
            try:
                entities, relations = await self.extract_entities(text)
                for entity in entities:
                    try:
                        attrs = dict(entity.get("attributes", {}))
                        attrs["domain"] = self._goal_slug
                        await self._mcp.call_tool(
                            "add_entity",
                            {
                                "name": entity["name"],
                                "entity_type": entity["type"],
                                "attributes": json.dumps(attrs, ensure_ascii=False),
                                "source_file": "",
                            },
                        )
                        if self._goal_index:
                            try:
                                self._goal_index.add_entity(
                                    entity["name"], entity["type"], attrs, ""
                                )
                            except Exception:
                                pass
                    except Exception:
                        pass
                for rel in relations:
                    try:
                        await self._mcp.call_tool(
                            "add_relation",
                            {
                                "source_name": rel["source"],
                                "relation_type": rel["relation"],
                                "target_name": rel["target"],
                                "attributes": json.dumps({}, ensure_ascii=False),
                            },
                        )
                        if self._goal_index:
                            try:
                                self._goal_index.add_relation(
                                    rel["source"], rel["relation"], rel["target"]
                                )
                            except Exception:
                                pass
                    except Exception:
                        pass
                processed += 1
                log.info(
                    "entity_queue_drained",
                    entities=len(entities),
                    relations=len(relations),
                    remaining=len(self._entity_queue),
                )
            except Exception:
                log.debug("entity_queue_drain_failed", exc_info=True)
                break  # LLM likely still busy, stop draining
        return processed

    # ------------------------------------------------------------------
    # Identity Memory Summarization (Step 5)
    # ------------------------------------------------------------------

    _IDENTITY_SUMMARY_PROMPT = (
        "Du bist ein Wissenskurator. Analysiere diesen Text und erstelle "
        "einen strukturierten Wissensbaustein.\n\n"
        "Themenbereich: {goal_slug}\n"
        "Quelle: {url}\n\n"
        "Text:\n{text}\n\n"
        "Antworte NUR mit validem JSON:\n"
        "{{\n"
        '  "summary": "Praegnante Zusammenfassung in 3-8 Saetzen. Nur Fakten, kein Fuelltext.",\n'
        '  "memory_type": "semantic|procedural|episodic",\n'
        '  "tags": ["tag1", "tag2", "tag3"],\n'
        '  "is_useful": true/false\n'
        "}}\n\n"
        "Regeln:\n"
        '- memory_type "semantic" = Fakten, Wissen, Definitionen\n'
        '- memory_type "procedural" = Anleitungen, Prozesse, How-To\n'
        '- memory_type "episodic" = Ereignisse, Nachrichten, zeitgebunden\n'
        "- is_useful: false wenn der Text keine verwertbaren Informationen enthaelt\n"
        "- tags: 2-5 thematische Schlagwoerter, deutsch\n"
        "- summary: Deutsch, sachlich, nur Kernaussagen\n"
    )

    async def _summarize_for_identity(
        self,
        text: str,
        url: str,
        *,
        already_summarized: bool = False,
    ) -> None:
        """Summarize content and store in Identity Memory (Step 5 of pipeline).

        When already_summarized=True (ATL synthesis), skips the LLM call and
        stores the text directly with URL-based confidence.
        """
        if self._memory_manager is None:
            return

        min_chars = 100 if already_summarized else 200
        usable, _reason = _is_usable_content(text, min_chars=min_chars)
        if not usable:
            return

        confidence = _score_source_confidence(url)

        if already_summarized:
            summary = text
            memory_type = "semantic"
            tags = [self._goal_slug] if self._goal_slug else []
        elif self._llm_fn is not None:
            prompt = self._IDENTITY_SUMMARY_PROMPT.format(
                goal_slug=self._goal_slug or "allgemein",
                url=url[:200],
                text=text[:4000],
            )
            try:
                raw = await self._llm_fn(prompt)
            except Exception:
                log.debug("identity_summary_llm_failed", exc_info=True)
                return

            parsed = _parse_llm_json(raw, text, url)
            if not parsed.get("is_useful", True):
                log.debug("identity_summary_not_useful", url=url[:80])
                return

            summary = parsed["summary"]
            memory_type = parsed["memory_type"]
            tags = parsed["tags"]
            if self._goal_slug:
                tags = [self._goal_slug] + [t for t in tags if t != self._goal_slug]
        else:
            return

        # Dedup check
        content_hash = self._identity_dedup.content_hash(summary)
        if content_hash in self._identity_seen_hashes:
            log.debug("identity_dedup_skipped", url=url[:80])
            return
        self._identity_seen_hashes.add(content_hash)

        # Store in Identity Memory
        self._memory_manager.sync_document_to_identity(
            summary=summary,
            memory_type=memory_type,
            confidence=confidence,
            tags=tags,
        )
        log.info(
            "identity_memory_stored",
            url=url[:80],
            memory_type=memory_type,
            confidence=confidence,
            tags=tags[:3],
            summary_len=len(summary),
        )

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    @staticmethod
    def chunk_text(text: str, max_tokens: int = 512, overlap_tokens: int = 64) -> List[str]:
        """Split text into overlapping word-based chunks.

        Parameters use 'tokens' in name but operate on words as a proxy.
        Each chunk has at most *max_tokens* words, with *overlap_tokens*
        words carried over from the previous chunk.
        """
        words = text.split()
        if len(words) <= max_tokens:
            return [text]

        chunks: List[str] = []
        start = 0
        while start < len(words):
            end = start + max_tokens
            chunk_words = words[start:end]
            chunks.append(" ".join(chunk_words))
            start += max_tokens - overlap_tokens

        return chunks

    # ------------------------------------------------------------------
    # Entity extraction
    # ------------------------------------------------------------------

    async def extract_entities(self, text: str) -> Tuple[List[dict], List[dict]]:
        """Ask the LLM to extract entities and relations from *text*.

        Returns (entities, relations). Falls back to empty lists if the
        LLM does not produce valid JSON.
        """
        if self._entity_llm_fn is None:
            return [], []

        prompt = _ENTITY_EXTRACTION_PROMPT.format(text=text[:4000])

        try:
            raw = await self._entity_llm_fn(prompt)
        except Exception as exc:
            log.warning("LLM call failed during entity extraction: %s", exc)
            return [], []

        # Try to extract JSON from the response
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Try regex extraction of JSON block
            match = re.search(r"\{[\s\S]*\}", raw)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    return [], []
            else:
                return [], []

        entities = data.get("entities", [])
        relations = data.get("relations", [])

        # Validate structure
        valid_entities = [
            e for e in entities if isinstance(e, dict) and "name" in e and "type" in e
        ]
        valid_relations = [
            r
            for r in relations
            if isinstance(r, dict) and "source" in r and "relation" in r and "target" in r
        ]

        # Content filter: reject garbage entities from PDF metadata,
        # dictionary pages, navigation elements, etc.
        valid_entities = [e for e in valid_entities if _is_valid_entity(e)]
        # Filter relations whose source or target was rejected
        entity_names = {e["name"] for e in valid_entities}
        valid_relations = [
            r
            for r in valid_relations
            if r["source"] in entity_names and r["target"] in entity_names
        ]

        return valid_entities[:10], valid_relations[:10]
