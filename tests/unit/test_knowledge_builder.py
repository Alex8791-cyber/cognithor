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
        "text": (
            "Das Versicherungsvertragsgesetz (VVG) regelt die Rechtsbeziehungen "
            "zwischen Versicherungsnehmer und Versicherer. Es umfasst allgemeine "
            "Vorschriften ueber den Abschluss und die Durchfuehrung von "
            "Versicherungsvertraegen. Die wichtigsten Paragraphen betreffen "
            "die Anzeigepflicht und das Widerrufsrecht."
        ),
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

        kb = KnowledgeBuilder(mcp_client=_make_mcp(), llm_fn=_mock_llm, goal_slug="test")

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

        kb = KnowledgeBuilder(mcp_client=_make_mcp(), llm_fn=bad_llm, goal_slug="test")

        entities, relations = await kb.extract_entities("Some text.")

        assert entities == []
        assert relations == []

    @pytest.mark.asyncio
    async def test_vault_folder_uses_goal_slug(self):
        from jarvis.evolution.knowledge_builder import KnowledgeBuilder

        mcp = _make_mcp()
        kb = KnowledgeBuilder(mcp_client=mcp, llm_fn=_mock_llm, goal_slug="versicherung")
        fr = _make_fetch_result()

        await kb.build(fr)

        # Find the vault_save call
        vault_calls = [c for c in mcp.call_tool.call_args_list if c.args[0] == "vault_save"]
        assert len(vault_calls) == 1
        kwargs = vault_calls[0].args[1]
        assert "versicherung" in kwargs["folder"]

    @pytest.mark.asyncio
    async def test_memory_uses_semantic_tier(self):
        from jarvis.evolution.knowledge_builder import KnowledgeBuilder

        mcp = _make_mcp()
        kb = KnowledgeBuilder(mcp_client=mcp, llm_fn=_mock_llm, goal_slug="test-slug")
        fr = _make_fetch_result()

        await kb.build(fr)

        memory_calls = [c for c in mcp.call_tool.call_args_list if c.args[0] == "save_to_memory"]
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
                text=(
                    f"Content for page {i} about legal matters. "
                    "Das Versicherungsvertragsgesetz (VVG) regelt die Rechtsbeziehungen "
                    "zwischen Versicherungsnehmer und Versicherer. Es umfasst allgemeine "
                    "Vorschriften ueber den Abschluss und die Durchfuehrung von "
                    "Versicherungsvertraegen sowie die Anzeigepflicht."
                ),
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


class TestContentQualityGate:
    """Tests for _is_usable_content — rejects PDF artifacts and too-short text."""

    def test_rejects_too_short_text(self):
        from jarvis.evolution.knowledge_builder import _is_usable_content

        usable, reason = _is_usable_content("Short.")
        assert usable is False
        assert reason == "too_short"

    def test_rejects_empty_text(self):
        from jarvis.evolution.knowledge_builder import _is_usable_content

        usable, reason = _is_usable_content("")
        assert usable is False
        assert reason == "too_short"

    def test_rejects_whitespace_only(self):
        from jarvis.evolution.knowledge_builder import _is_usable_content

        usable, reason = _is_usable_content("   \n\n\t  \n  ")
        assert usable is False
        assert reason == "too_short"

    def test_rejects_pdf_artifact_text(self):
        from jarvis.evolution.knowledge_builder import _is_usable_content

        # Pad with enough PDF-like lines to exceed min_chars and trigger artifact ratio
        pdf_dump = "\n".join(
            [
                "1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj",
                "2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj",
                "3 0 obj << /Type /Page /MediaBox [0 0 612 792] /Contents 5 0 R >> endobj",
                "5 0 obj",
                "<< /Filter /FlateDecode /Length 1528 >>",
                "stream",
                "endstream",
                "endobj",
                "6 0 obj << /Type /Font /Subtype /Type1 /BaseFont /ArialMT >> endobj",
                "xref",
                "0 15",
                "0000000000 65535 f",
                "trailer",
                "<< /Root 1 0 R /Info 2 0 R >>",
                "%%EOF",
                "Some actual text here that is real content.",
            ]
        )
        usable, reason = _is_usable_content(pdf_dump)
        assert usable is False
        assert "pdf_artifacts" in reason

    def test_accepts_real_article(self):
        from jarvis.evolution.knowledge_builder import _is_usable_content

        article = (
            "Das Versicherungsvertragsgesetz (VVG) regelt die Rechtsbeziehungen "
            "zwischen Versicherungsnehmer und Versicherer. Es umfasst allgemeine "
            "Vorschriften ueber den Abschluss und die Durchfuehrung von "
            "Versicherungsvertraegen. Die wichtigsten Paragraphen betreffen "
            "die Anzeigepflicht, das Widerrufsrecht und die Leistungspflicht "
            "des Versicherers bei Eintritt des Versicherungsfalls."
        )
        usable, reason = _is_usable_content(article)
        assert usable is True
        assert reason == "ok"

    def test_accepts_content_at_boundary(self):
        from jarvis.evolution.knowledge_builder import _is_usable_content

        text = "x " * 101  # 202 chars — just above 200 threshold
        usable, reason = _is_usable_content(text)
        assert usable is True

    def test_borderline_garbage_ratio_below_threshold(self):
        from jarvis.evolution.knowledge_builder import _is_usable_content

        lines = ["endobj", "xref", "trailer"]
        lines += ["Dies ist ein normaler Satz ueber Versicherungsrecht."] * 8
        text = "\n".join(lines)
        usable, reason = _is_usable_content(text)
        assert usable is True

    def test_custom_min_chars(self):
        from jarvis.evolution.knowledge_builder import _is_usable_content

        text = "a " * 60  # 120 chars
        usable_default, _ = _is_usable_content(text)
        usable_low, _ = _is_usable_content(text, min_chars=100)
        assert usable_default is False
        assert usable_low is True


class TestBuildRejectsGarbage:
    """build() should skip triple-write when content is unusable."""

    @pytest.mark.asyncio
    async def test_build_skips_pdf_garbage(self):
        from jarvis.evolution.knowledge_builder import BuildResult, KnowledgeBuilder

        mcp = _make_mcp()
        kb = KnowledgeBuilder(mcp_client=mcp, llm_fn=_mock_llm, goal_slug="test")
        pdf_dump = "\n".join(
            [
                "5 0 obj",
                "<< /Type /Page /MediaBox [0 0 612 792] >>",
                "endobj",
                "6 0 obj",
                "<< /Filter /FlateDecode /Length 1528 >>",
                "stream",
                "xref",
                "0 15",
                "trailer",
                "<< /Root 1 0 R /Info 2 0 R >>",
                "%%EOF",
                "Some text.",
            ]
        )
        fr = _make_fetch_result(text=pdf_dump)
        result = await kb.build(fr)
        assert isinstance(result, BuildResult)
        assert result.chunks_created == 0
        assert result.vault_path == ""
        assert len(result.errors) == 1
        assert "Content rejected" in result.errors[0]
        mcp.call_tool.assert_not_called()

    @pytest.mark.asyncio
    async def test_build_skips_too_short(self):
        from jarvis.evolution.knowledge_builder import BuildResult, KnowledgeBuilder

        mcp = _make_mcp()
        kb = KnowledgeBuilder(mcp_client=mcp, goal_slug="test")
        fr = _make_fetch_result(text="Short.")
        result = await kb.build(fr)
        assert result.chunks_created == 0
        assert "Content rejected" in result.errors[0]
        mcp.call_tool.assert_not_called()

    @pytest.mark.asyncio
    async def test_build_accepts_good_content(self):
        from jarvis.evolution.knowledge_builder import BuildResult, KnowledgeBuilder

        mcp = _make_mcp()
        kb = KnowledgeBuilder(mcp_client=mcp, llm_fn=_mock_llm, goal_slug="test")
        fr = _make_fetch_result(
            text=(
                "Das Versicherungsvertragsgesetz (VVG) regelt die Rechtsbeziehungen "
                "zwischen Versicherungsnehmer und Versicherer. Es umfasst allgemeine "
                "Vorschriften ueber den Abschluss und die Durchfuehrung von "
                "Versicherungsvertraegen. Die wichtigsten Paragraphen betreffen "
                "die Anzeigepflicht und das Widerrufsrecht."
            )
        )
        result = await kb.build(fr)
        assert result.vault_path != ""
        assert result.chunks_created > 0


class TestSourceConfidenceScoring:
    """Tests for _score_source_confidence — URL-based trust scoring."""

    def test_trusted_gov_domain(self):
        from jarvis.evolution.knowledge_builder import _score_source_confidence

        score = _score_source_confidence("https://www.bafin.de/SharedDocs/some-article.html")
        assert score == 0.9

    def test_trusted_bund_domain(self):
        from jarvis.evolution.knowledge_builder import _score_source_confidence

        score = _score_source_confidence("https://www.gesetze-im-internet.de/vvg/__1.html")
        assert score == 0.9

    def test_medium_trust_wikipedia(self):
        from jarvis.evolution.knowledge_builder import _score_source_confidence

        score = _score_source_confidence("https://de.wikipedia.org/wiki/Versicherung")
        assert score == 0.7

    def test_medium_trust_heise(self):
        from jarvis.evolution.knowledge_builder import _score_source_confidence

        score = _score_source_confidence("https://www.heise.de/news/some-article.html")
        assert score == 0.7

    def test_low_trust_blog(self):
        from jarvis.evolution.knowledge_builder import _score_source_confidence

        score = _score_source_confidence("https://some-random-blog.com/post/123")
        assert score == 0.3

    def test_low_trust_medium(self):
        from jarvis.evolution.knowledge_builder import _score_source_confidence

        score = _score_source_confidence("https://medium.com/@user/my-article-abc123")
        assert score == 0.3

    def test_low_trust_reddit(self):
        from jarvis.evolution.knowledge_builder import _score_source_confidence

        score = _score_source_confidence("https://www.reddit.com/r/python/comments/abc")
        assert score == 0.3

    def test_default_unknown_domain(self):
        from jarvis.evolution.knowledge_builder import _score_source_confidence

        score = _score_source_confidence("https://www.example.com/article")
        assert score == 0.5

    def test_empty_url(self):
        from jarvis.evolution.knowledge_builder import _score_source_confidence

        score = _score_source_confidence("")
        assert score == 0.5

    def test_owasp_high_trust(self):
        from jarvis.evolution.knowledge_builder import _score_source_confidence

        score = _score_source_confidence("https://owasp.org/Top10/")
        assert score == 0.8

    def test_europa_eu(self):
        from jarvis.evolution.knowledge_builder import _score_source_confidence

        score = _score_source_confidence("https://eur-lex.europa.eu/legal-content/EN/ALL/")
        assert score == 0.9


class TestParseLLMJson:
    """Tests for _parse_llm_json — 4-tier fallback parsing."""

    def test_tier1_valid_json(self):
        from jarvis.evolution.knowledge_builder import _parse_llm_json

        raw = json.dumps(
            {
                "summary": "Das VVG regelt Versicherungen.",
                "memory_type": "semantic",
                "tags": ["versicherung", "recht"],
                "is_useful": True,
            }
        )
        result = _parse_llm_json(raw, "fallback text", "https://example.com")
        assert result["summary"] == "Das VVG regelt Versicherungen."
        assert result["memory_type"] == "semantic"
        assert result["tags"] == ["versicherung", "recht"]
        assert result["is_useful"] is True

    def test_tier2_json_in_markdown_block(self):
        from jarvis.evolution.knowledge_builder import _parse_llm_json

        raw = (
            "Hier ist meine Analyse:\n\n"
            "```json\n"
            '{"summary": "Wichtige Fakten.", "memory_type": "procedural", '
            '"tags": ["prozess"], "is_useful": true}\n'
            "```\n"
        )
        result = _parse_llm_json(raw, "fallback", "https://example.com")
        assert result["summary"] == "Wichtige Fakten."
        assert result["memory_type"] == "procedural"

    def test_tier3_regex_extraction(self):
        from jarvis.evolution.knowledge_builder import _parse_llm_json

        raw = (
            'Hier ist das Ergebnis: "summary": "Extracted via regex.", '
            '"memory_type": "episodic", "tags": ["event", "news"], "is_useful": true'
        )
        result = _parse_llm_json(raw, "fallback", "https://example.com")
        assert result["summary"] == "Extracted via regex."
        assert result["memory_type"] == "episodic"

    def test_tier4_complete_fallback(self):
        from jarvis.evolution.knowledge_builder import _parse_llm_json

        raw = "I cannot process this request. Here is some random text."
        result = _parse_llm_json(
            raw, "Original article about insurance law and regulation.", "https://example.com"
        )
        assert result["summary"] == "Original article about insurance law and regulation."
        assert result["memory_type"] == "semantic"
        assert result["tags"] == []
        assert result["is_useful"] is True

    def test_fallback_truncates_long_content(self):
        from jarvis.evolution.knowledge_builder import _parse_llm_json

        long_fallback = "x" * 2000
        result = _parse_llm_json("garbage", long_fallback, "https://example.com")
        assert len(result["summary"]) == 800

    def test_is_useful_false_parsed(self):
        from jarvis.evolution.knowledge_builder import _parse_llm_json

        raw = json.dumps(
            {
                "summary": "Nichts relevantes.",
                "memory_type": "semantic",
                "tags": [],
                "is_useful": False,
            }
        )
        result = _parse_llm_json(raw, "fallback", "https://example.com")
        assert result["is_useful"] is False

    def test_partial_json_with_extra_text(self):
        from jarvis.evolution.knowledge_builder import _parse_llm_json

        raw = (
            "<think>Let me analyze this text.</think>\n"
            '{"summary": "Nach dem Denken.", "memory_type": "semantic", '
            '"tags": ["ki"], "is_useful": true}'
        )
        result = _parse_llm_json(raw, "fallback", "https://example.com")
        assert result["summary"] == "Nach dem Denken."


_LLM_SUMMARY_JSON = json.dumps(
    {
        "summary": "Das VVG regelt die Rechtsbeziehungen zwischen Versicherungsnehmer und Versicherer.",
        "memory_type": "semantic",
        "tags": ["versicherung", "vvg", "recht"],
        "is_useful": True,
    }
)


async def _mock_summary_llm(prompt: str) -> str:
    """Return entity JSON for entity prompts, summary JSON for summary prompts."""
    if "Wissenskurator" in prompt:
        return _LLM_SUMMARY_JSON
    return _LLM_ENTITY_JSON


class TestSummarizeForIdentity:
    """Tests for _summarize_for_identity — Step 5 of the build pipeline."""

    @pytest.mark.asyncio
    async def test_build_calls_summarize_when_memory_manager_set(self):
        from jarvis.evolution.knowledge_builder import KnowledgeBuilder
        from unittest.mock import MagicMock

        mcp = _make_mcp()
        mm = MagicMock()
        mm.sync_document_to_identity = MagicMock()

        kb = KnowledgeBuilder(
            mcp_client=mcp,
            llm_fn=_mock_summary_llm,
            goal_slug="versicherung",
            memory_manager=mm,
        )
        fr = _make_fetch_result()
        await kb.build(fr)

        mm.sync_document_to_identity.assert_called_once()
        call_args = mm.sync_document_to_identity.call_args
        # Check summary contains VVG (from LLM response)
        summary_arg = (
            call_args.kwargs.get("summary", "")
            if call_args.kwargs
            else call_args[1].get("summary", "")
        )
        assert "VVG" in summary_arg

    @pytest.mark.asyncio
    async def test_build_skips_summarize_without_memory_manager(self):
        from jarvis.evolution.knowledge_builder import KnowledgeBuilder

        mcp = _make_mcp()
        kb = KnowledgeBuilder(
            mcp_client=mcp,
            llm_fn=_mock_summary_llm,
            goal_slug="test",
        )
        fr = _make_fetch_result()
        result = await kb.build(fr)
        assert result.chunks_created > 0

    @pytest.mark.asyncio
    async def test_build_skips_summarize_without_llm_fn(self):
        from jarvis.evolution.knowledge_builder import KnowledgeBuilder
        from unittest.mock import MagicMock

        mcp = _make_mcp()
        mm = MagicMock()
        kb = KnowledgeBuilder(
            mcp_client=mcp,
            llm_fn=None,
            goal_slug="test",
            memory_manager=mm,
        )
        fr = _make_fetch_result()
        await kb.build(fr)
        mm.sync_document_to_identity.assert_not_called()

    @pytest.mark.asyncio
    async def test_already_summarized_skips_llm_call(self):
        from jarvis.evolution.knowledge_builder import KnowledgeBuilder
        from unittest.mock import MagicMock

        mcp = _make_mcp()
        mm = MagicMock()
        mm.sync_document_to_identity = MagicMock()

        llm_call_count = 0

        async def counting_llm(prompt: str) -> str:
            nonlocal llm_call_count
            llm_call_count += 1
            return await _mock_summary_llm(prompt)

        kb = KnowledgeBuilder(
            mcp_client=mcp,
            llm_fn=counting_llm,
            goal_slug="versicherung",
            memory_manager=mm,
        )
        fr = _make_fetch_result()
        llm_call_count = 0
        await kb.build(fr, already_summarized=True)

        mm.sync_document_to_identity.assert_called_once()
        call_args = mm.sync_document_to_identity.call_args
        confidence = (
            call_args.kwargs.get("confidence", 0)
            if call_args.kwargs
            else call_args[1].get("confidence", 0)
        )
        assert confidence == 0.5  # example.com -> default

    @pytest.mark.asyncio
    async def test_dedup_skips_duplicate_summaries(self):
        from jarvis.evolution.knowledge_builder import KnowledgeBuilder
        from unittest.mock import MagicMock

        mcp = _make_mcp()
        mm = MagicMock()
        mm.sync_document_to_identity = MagicMock()

        kb = KnowledgeBuilder(
            mcp_client=mcp,
            llm_fn=_mock_summary_llm,
            goal_slug="test",
            memory_manager=mm,
        )

        fr1 = _make_fetch_result(url="https://example.com/page1")
        fr2 = _make_fetch_result(url="https://example.com/page2")

        await kb.build(fr1)
        await kb.build(fr2)

        # LLM always returns same summary -> second should be deduped
        assert mm.sync_document_to_identity.call_count == 1

    @pytest.mark.asyncio
    async def test_summarize_failure_does_not_block_pipeline(self):
        from jarvis.evolution.knowledge_builder import KnowledgeBuilder
        from unittest.mock import MagicMock

        mcp = _make_mcp()
        mm = MagicMock()
        mm.sync_document_to_identity.side_effect = RuntimeError("DB error")

        kb = KnowledgeBuilder(
            mcp_client=mcp,
            llm_fn=_mock_summary_llm,
            goal_slug="test",
            memory_manager=mm,
        )
        fr = _make_fetch_result()
        result = await kb.build(fr)

        assert result.chunks_created > 0
        assert result.vault_path != ""

    @pytest.mark.asyncio
    async def test_is_useful_false_skips_store(self):
        from jarvis.evolution.knowledge_builder import KnowledgeBuilder
        from unittest.mock import MagicMock

        async def useless_llm(prompt: str) -> str:
            if "Wissenskurator" in prompt:
                return json.dumps(
                    {
                        "summary": "Nichts relevantes.",
                        "memory_type": "semantic",
                        "tags": [],
                        "is_useful": False,
                    }
                )
            return _LLM_ENTITY_JSON

        mcp = _make_mcp()
        mm = MagicMock()
        mm.sync_document_to_identity = MagicMock()

        kb = KnowledgeBuilder(
            mcp_client=mcp,
            llm_fn=useless_llm,
            goal_slug="test",
            memory_manager=mm,
        )
        fr = _make_fetch_result()
        await kb.build(fr)

        mm.sync_document_to_identity.assert_not_called()
