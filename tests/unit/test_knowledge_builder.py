"""Tests for KnowledgeBuilder — triple-write pipeline (Vault + Memory + Graph)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from unittest.mock import AsyncMock

import pytest


@dataclass
class _MockToolResult:
    content: str = ""
    is_error: bool = False


_LLM_ENTITY_JSON = json.dumps(
    {
        "entities": [
            {
                "name": "VVG",
                "type": "law",
                "attributes": {"full_name": "Versicherungsvertragsgesetz"},
            },
            {
                "name": "Widerrufsrecht",
                "type": "concept",
                "attributes": {},
            },
        ],
        "relations": [
            {
                "source": "VVG",
                "relation": "regelt",
                "target": "Widerrufsrecht",
            }
        ],
    }
)


def _make_mcp() -> AsyncMock:
    mcp = AsyncMock()
    mcp.call_tool = AsyncMock(return_value=_MockToolResult(content="OK"))
    return mcp


async def _mock_llm(prompt: str) -> str:
    return _LLM_ENTITY_JSON


def _make_fetch_result(**kwargs):
    from jarvis.evolution.research_agent import FetchResult

    defaults = {
        "url": "https://example.com/vvg",
        "text": "Das Versicherungsvertragsgesetz regelt das Widerrufsrecht.",
        "title": "VVG Uebersicht",
        "source_type": "article",
        "error": "",
    }
    defaults.update(kwargs)
    return FetchResult(**defaults)


class TestKnowledgeBuilder:
    @pytest.mark.asyncio
    async def test_build_from_fetch_result(self):
        from jarvis.evolution.knowledge_builder import BuildResult, KnowledgeBuilder

        mcp = _make_mcp()
        kb = KnowledgeBuilder(mcp_client=mcp, llm_fn=_mock_llm, goal_slug="vvg-recht")
        fr = _make_fetch_result()

        result = await kb.build(fr)

        assert isinstance(result, BuildResult)
        assert result.vault_path != ""
        assert result.chunks_created > 0
        assert result.entities_created >= 1
        assert result.relations_created >= 1

        # Verify MCP calls
        call_names = [c.args[0] for c in mcp.call_tool.call_args_list]
        assert "vault_save" in call_names
        assert "save_to_memory" in call_names
        assert "add_entity" in call_names
        assert "add_relation" in call_names

    @pytest.mark.asyncio
    async def test_chunking(self):
        from jarvis.evolution.knowledge_builder import KnowledgeBuilder

        kb = KnowledgeBuilder(mcp_client=_make_mcp(), goal_slug="test")
        long_text = " ".join(["word"] * 2000)

        chunks = kb.chunk_text(long_text)

        assert len(chunks) > 1
        for chunk in chunks:
            word_count = len(chunk.split())
            assert word_count <= 600, f"Chunk has {word_count} words, expected <=600"

    @pytest.mark.asyncio
    async def test_chunking_short_text(self):
        from jarvis.evolution.knowledge_builder import KnowledgeBuilder

        kb = KnowledgeBuilder(mcp_client=_make_mcp(), goal_slug="test")
        short_text = "This is a short text with only a few words."

        chunks = kb.chunk_text(short_text)

        assert len(chunks) == 1
        assert chunks[0] == short_text

    @pytest.mark.asyncio
    async def test_entity_extraction(self):
        from jarvis.evolution.knowledge_builder import KnowledgeBuilder

        kb = KnowledgeBuilder(
            mcp_client=_make_mcp(), llm_fn=_mock_llm, goal_slug="test"
        )

        entities, relations = await kb.extract_entities("Some legal text about VVG.")

        assert len(entities) == 2
        assert entities[0]["name"] == "VVG"
        assert entities[0]["type"] == "law"
        assert len(relations) == 1
        assert relations[0]["source"] == "VVG"
        assert relations[0]["relation"] == "regelt"
        assert relations[0]["target"] == "Widerrufsrecht"

    @pytest.mark.asyncio
    async def test_entity_extraction_llm_failure(self):
        from jarvis.evolution.knowledge_builder import KnowledgeBuilder

        async def bad_llm(prompt: str) -> str:
            return "This is not JSON at all, sorry."

        kb = KnowledgeBuilder(
            mcp_client=_make_mcp(), llm_fn=bad_llm, goal_slug="test"
        )

        entities, relations = await kb.extract_entities("Some text.")

        assert entities == []
        assert relations == []

    @pytest.mark.asyncio
    async def test_vault_folder_uses_goal_slug(self):
        from jarvis.evolution.knowledge_builder import KnowledgeBuilder

        mcp = _make_mcp()
        kb = KnowledgeBuilder(
            mcp_client=mcp, llm_fn=_mock_llm, goal_slug="versicherung"
        )
        fr = _make_fetch_result()

        await kb.build(fr)

        # Find the vault_save call
        vault_calls = [
            c for c in mcp.call_tool.call_args_list if c.args[0] == "vault_save"
        ]
        assert len(vault_calls) == 1
        kwargs = vault_calls[0].args[1]
        assert "versicherung" in kwargs["folder"]

    @pytest.mark.asyncio
    async def test_memory_uses_semantic_tier(self):
        from jarvis.evolution.knowledge_builder import KnowledgeBuilder

        mcp = _make_mcp()
        kb = KnowledgeBuilder(
            mcp_client=mcp, llm_fn=_mock_llm, goal_slug="test-slug"
        )
        fr = _make_fetch_result()

        await kb.build(fr)

        memory_calls = [
            c for c in mcp.call_tool.call_args_list if c.args[0] == "save_to_memory"
        ]
        assert len(memory_calls) >= 1
        for call in memory_calls:
            kwargs = call.args[1]
            assert kwargs["tier"] == "semantic"

    @pytest.mark.asyncio
    async def test_build_result_accumulates(self):
        from jarvis.evolution.knowledge_builder import KnowledgeBuilder

        mcp = _make_mcp()
        kb = KnowledgeBuilder(mcp_client=mcp, llm_fn=_mock_llm, goal_slug="multi")

        results = []
        for i in range(3):
            fr = _make_fetch_result(
                url=f"https://example.com/page{i}",
                text=f"Content for page {i} about legal matters.",
            )
            results.append(await kb.build(fr))

        total_chunks = sum(r.chunks_created for r in results)
        assert total_chunks >= 3

    def test_build_result_dataclass(self):
        from jarvis.evolution.knowledge_builder import BuildResult

        br = BuildResult()

        assert br.vault_path == ""
        assert br.chunks_created == 0
        assert br.entities_created == 0
        assert br.relations_created == 0
        assert br.errors == []
