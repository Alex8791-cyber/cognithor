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
    errors: List[str] = field(default_factory=list)


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
    ) -> None:
        self._mcp = mcp_client
        self._llm_fn = llm_fn
        self._goal_slug = goal_slug

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def build(self, fetch_result: FetchResult) -> BuildResult:
        """Run the triple-write pipeline for a single FetchResult."""
        result = BuildResult()

        if fetch_result.error or not fetch_result.text:
            result.errors.append(
                fetch_result.error or "Empty text in FetchResult"
            )
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
                except Exception as exc:
                    result.errors.append(f"add_relation failed: {exc}")

        return result

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    @staticmethod
    def chunk_text(
        text: str, max_tokens: int = 512, overlap_tokens: int = 64
    ) -> List[str]:
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

    async def extract_entities(
        self, text: str
    ) -> Tuple[List[dict], List[dict]]:
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
            e for e in entities
            if isinstance(e, dict) and "name" in e and "type" in e
        ]
        valid_relations = [
            r for r in relations
            if isinstance(r, dict)
            and "source" in r
            and "relation" in r
            and "target" in r
        ]

        return valid_entities[:10], valid_relations[:10]
