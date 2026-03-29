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
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

__all__ = ["BuildResult", "KnowledgeBuilder"]


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
    """

    def __init__(
        self,
        mcp_client: Any,
        llm_fn: Optional[Callable] = None,
        goal_slug: str = "",
        knowledge_validator: Any = None,
        goal_index: Any = None,
    ) -> None:
        self._mcp = mcp_client
        self._llm_fn = llm_fn
        self._goal_slug = goal_slug
        self._validator = knowledge_validator
        self._goal_index = goal_index

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def build(self, fetch_result: FetchResult) -> BuildResult:
        """Run the triple-write pipeline for a single FetchResult."""
        result = BuildResult()

        if fetch_result.error or not fetch_result.text:
            result.errors.append(fetch_result.error or "Empty text in FetchResult")
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
        if self._llm_fn is not None:
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

        return result

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
        if self._llm_fn is None:
            return [], []

        prompt = _ENTITY_EXTRACTION_PROMPT.format(text=text[:4000])

        try:
            raw = await self._llm_fn(prompt)
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
