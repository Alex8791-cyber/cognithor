"""Coverage-Tests fuer planner.py -- fehlende Zeilen."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from jarvis.config import JarvisConfig, ensure_directory_structure
from jarvis.core.planner import Planner
from jarvis.models import ActionPlan, WorkingMemory


@pytest.fixture()
def config(tmp_path) -> JarvisConfig:
    cfg = JarvisConfig(jarvis_home=tmp_path)
    ensure_directory_structure(cfg)
    return cfg


def _mock_ollama(response_content: str) -> AsyncMock:
    mock = AsyncMock()
    mock.chat = AsyncMock(return_value={
        "message": {"role": "assistant", "content": response_content}
    })
    mock.is_available = AsyncMock(return_value=True)
    return mock


def _mock_router() -> MagicMock:
    router = MagicMock()
    router.select_model.return_value = "qwen3:32b"
    router.get_model_config.return_value = {"temperature": 0.7, "top_p": 0.9}
    return router


# ============================================================================
# Planner.plan -- edge cases
# ============================================================================


class TestPlannerEdgeCases:
    @pytest.mark.asyncio
    async def test_plan_with_think_tags(self, config: JarvisConfig) -> None:
        """LLM returns response with <think> tags (qwen3 behavior)."""
        content = '<think>Let me think about this...</think>\nDas ist eine einfache Frage.'
        ollama = _mock_ollama(content)
        planner = Planner(config, ollama, _mock_router())
        wm = WorkingMemory(session_id="test")

        plan = await planner.plan(
            user_message="Was ist Python?",
            working_memory=wm,
            tool_schemas={},
        )
        assert isinstance(plan, ActionPlan)
        assert plan.direct_response

    @pytest.mark.asyncio
    async def test_plan_json_in_code_block(self, config: JarvisConfig) -> None:
        """LLM returns JSON in a code block."""
        content = '```json\n{"goal":"test","steps":[{"tool":"read_file","params":{"path":"/tmp/x"},"rationale":"test"}],"confidence":0.9}\n```'
        ollama = _mock_ollama(content)
        planner = Planner(config, ollama, _mock_router())
        wm = WorkingMemory(session_id="test")

        plan = await planner.plan(
            user_message="Lies /tmp/x",
            working_memory=wm,
            tool_schemas={"read_file": {"description": "reads a file"}},
        )
        assert isinstance(plan, ActionPlan)

    @pytest.mark.asyncio
    async def test_plan_invalid_json(self, config: JarvisConfig) -> None:
        """LLM returns invalid JSON -- should fallback to direct response."""
        content = 'Das ist keine JSON-Antwort sondern normaler Text.'
        ollama = _mock_ollama(content)
        planner = Planner(config, ollama, _mock_router())
        wm = WorkingMemory(session_id="test")

        plan = await planner.plan(
            user_message="test",
            working_memory=wm,
            tool_schemas={},
        )
        assert isinstance(plan, ActionPlan)
        assert plan.direct_response

    @pytest.mark.asyncio
    async def test_plan_with_empty_steps(self, config: JarvisConfig) -> None:
        """LLM returns JSON with empty steps list."""
        content = '```json\n{"goal":"test","steps":[],"confidence":0.9}\n```'
        ollama = _mock_ollama(content)
        planner = Planner(config, ollama, _mock_router())
        wm = WorkingMemory(session_id="test")

        plan = await planner.plan(
            user_message="test",
            working_memory=wm,
            tool_schemas={},
        )
        assert isinstance(plan, ActionPlan)


# ============================================================================
# Planner.replan
# ============================================================================


class TestPlannerReplan:
    @pytest.mark.asyncio
    async def test_replan_after_tool_results(self, config: JarvisConfig) -> None:
        content = 'Die Antwort basierend auf den Tool-Ergebnissen ist: 42.'
        ollama = _mock_ollama(content)
        planner = Planner(config, ollama, _mock_router())
        wm = WorkingMemory(session_id="test")

        from jarvis.models import ToolResult
        results = [ToolResult(tool_name="calc", content="42", is_error=False)]

        plan = await planner.replan(
            original_goal="Was ist 6*7?",
            results=results,
            working_memory=wm,
            tool_schemas={},
        )
        assert isinstance(plan, ActionPlan)


# ============================================================================
# Planner.formulate_response
# ============================================================================


class TestFormulateResponse:
    @pytest.mark.asyncio
    async def test_formulate_response(self, config: JarvisConfig) -> None:
        content = "Hier ist die zusammengefasste Antwort."
        ollama = _mock_ollama(content)
        planner = Planner(config, ollama, _mock_router())

        from jarvis.models import ToolResult
        results = [ToolResult(tool_name="web_search", content="Python ist toll", is_error=False)]
        wm = WorkingMemory(session_id="test")

        response = await planner.formulate_response(
            user_message="Was ist Python?",
            results=results,
            working_memory=wm,
        )
        assert isinstance(response, str)
        assert len(response) > 0


# ============================================================================
# Planner.generate_escalation
# ============================================================================


class TestGenerateEscalation:
    @pytest.mark.asyncio
    async def test_generate_escalation(self, config: JarvisConfig) -> None:
        content = "Der Befehl wurde aus Sicherheitsgruenden blockiert."
        ollama = _mock_ollama(content)
        planner = Planner(config, ollama, _mock_router())
        wm = WorkingMemory(session_id="test")

        response = await planner.generate_escalation(
            tool="exec_command",
            reason="Dangerous command blocked",
            working_memory=wm,
        )
        assert isinstance(response, str)


# ============================================================================
# TestPlannerLLMError -- plan() with OllamaError
# ============================================================================


class TestPlannerLLMError:
    @pytest.mark.asyncio
    async def test_plan_llm_error(self, config: JarvisConfig) -> None:
        """OllamaError from ollama.chat -> should return error plan with confidence=0.0."""
        from jarvis.core.model_router import OllamaError

        ollama = AsyncMock()
        ollama.chat = AsyncMock(side_effect=OllamaError("connection refused"))
        planner = Planner(config, ollama, _mock_router())
        wm = WorkingMemory(session_id="test")

        plan = await planner.plan(
            user_message="Was ist Python?",
            working_memory=wm,
            tool_schemas={},
        )
        assert isinstance(plan, ActionPlan)
        assert plan.confidence == 0.0
        assert plan.direct_response is not None
        assert "technisches Problem" in plan.direct_response

    @pytest.mark.asyncio
    async def test_plan_llm_error_with_audit_logger(self, config: JarvisConfig) -> None:
        """OllamaError with audit_logger -> audit_logger.log_tool_call called with success=False."""
        from jarvis.core.model_router import OllamaError

        ollama = AsyncMock()
        ollama.chat = AsyncMock(side_effect=OllamaError("timeout"))
        audit = MagicMock()
        planner = Planner(config, ollama, _mock_router(), audit_logger=audit)
        wm = WorkingMemory(session_id="test")

        plan = await planner.plan(
            user_message="Hallo",
            working_memory=wm,
            tool_schemas={},
        )
        assert plan.confidence == 0.0
        audit.log_tool_call.assert_called_once()
        call_kwargs = audit.log_tool_call.call_args
        assert call_kwargs[1]["success"] is False


# ============================================================================
# TestPlannerToolCalls -- plan() with native tool_calls
# ============================================================================


class TestPlannerToolCalls:
    @pytest.mark.asyncio
    async def test_plan_with_native_tool_calls(self, config: JarvisConfig) -> None:
        """Response has tool_calls (not just text) -> should parse them correctly."""
        ollama = AsyncMock()
        ollama.chat = AsyncMock(return_value={
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "read_file",
                            "arguments": {"path": "/tmp/test.txt"},
                        }
                    },
                    {
                        "function": {
                            "name": "write_file",
                            "arguments": {"path": "/tmp/out.txt", "content": "hello"},
                        }
                    },
                ],
            }
        })
        planner = Planner(config, ollama, _mock_router())
        wm = WorkingMemory(session_id="test")

        plan = await planner.plan(
            user_message="Lies und kopiere die Datei",
            working_memory=wm,
            tool_schemas={"read_file": {}, "write_file": {}},
        )
        assert isinstance(plan, ActionPlan)
        assert len(plan.steps) == 2
        assert plan.steps[0].tool == "read_file"
        assert plan.steps[0].params == {"path": "/tmp/test.txt"}
        assert plan.steps[1].tool == "write_file"
        assert plan.steps[1].params == {"path": "/tmp/out.txt", "content": "hello"}
        assert plan.confidence == 0.7


# ============================================================================
# TestReplanExtended -- more replan tests
# ============================================================================


class TestReplanExtended:
    @pytest.mark.asyncio
    async def test_replan_llm_error(self, config: JarvisConfig) -> None:
        """OllamaError during replan -> fallback plan with confidence=0.0."""
        from jarvis.core.model_router import OllamaError
        from jarvis.models import ToolResult

        ollama = AsyncMock()
        ollama.chat = AsyncMock(side_effect=OllamaError("model not found"))
        planner = Planner(config, ollama, _mock_router())
        wm = WorkingMemory(session_id="test")
        results = [ToolResult(tool_name="calc", content="42", is_error=False)]

        plan = await planner.replan(
            original_goal="Berechne etwas",
            results=results,
            working_memory=wm,
            tool_schemas={},
        )
        assert isinstance(plan, ActionPlan)
        assert plan.confidence == 0.0
        assert plan.direct_response is not None
        assert "nicht fortsetzen" in plan.direct_response

    @pytest.mark.asyncio
    async def test_replan_with_multiple_results(self, config: JarvisConfig) -> None:
        """Replan with mixed success/error results -> planner receives formatted results."""
        from jarvis.models import ToolResult

        content = "Basierend auf den Ergebnissen: Datei gelesen, aber Suche fehlgeschlagen."
        ollama = _mock_ollama(content)
        planner = Planner(config, ollama, _mock_router())
        wm = WorkingMemory(session_id="test")
        results = [
            ToolResult(tool_name="read_file", content="Dateiinhalt hier", is_error=False),
            ToolResult(tool_name="web_search", content="", is_error=True, error_message="timeout"),
        ]

        plan = await planner.replan(
            original_goal="Datei lesen und online suchen",
            results=results,
            working_memory=wm,
            tool_schemas={},
        )
        assert isinstance(plan, ActionPlan)
        # The LLM was called with both results in the prompt
        call_args = ollama.chat.call_args
        messages = call_args[1].get("messages") or call_args[0][1] if call_args[0] else call_args[1]["messages"]
        # At least one message should mention both tool names
        all_content = " ".join(m["content"] for m in messages)
        assert "read_file" in all_content
        assert "web_search" in all_content

    @pytest.mark.asyncio
    async def test_replan_returns_new_steps(self, config: JarvisConfig) -> None:
        """LLM returns JSON with new steps after replan."""
        from jarvis.models import ToolResult

        content = '```json\n{"goal":"Zweiter Versuch","steps":[{"tool":"web_search","params":{"query":"test"},"rationale":"Nochmal suchen"}],"confidence":0.8}\n```'
        ollama = _mock_ollama(content)
        planner = Planner(config, ollama, _mock_router())
        wm = WorkingMemory(session_id="test")
        results = [ToolResult(tool_name="web_search", content="", is_error=True)]

        plan = await planner.replan(
            original_goal="Online suchen",
            results=results,
            working_memory=wm,
            tool_schemas={"web_search": {}},
        )
        assert isinstance(plan, ActionPlan)
        assert len(plan.steps) >= 1
        assert plan.steps[0].tool == "web_search"
        assert plan.confidence == 0.8


# ============================================================================
# TestFormulateResponseExtended
# ============================================================================


class TestFormulateResponseExtended:
    @pytest.mark.asyncio
    async def test_formulate_with_search_results(self, config: JarvisConfig) -> None:
        """Results from web_search tool -> should trigger search-specific prompt."""
        from jarvis.models import ToolResult

        content = "Laut den Suchergebnissen ist Python eine Programmiersprache."
        ollama = _mock_ollama(content)
        planner = Planner(config, ollama, _mock_router())
        wm = WorkingMemory(session_id="test")
        results = [
            ToolResult(tool_name="web_search", content="Python ist eine Programmiersprache", is_error=False),
        ]

        response = await planner.formulate_response(
            user_message="Was ist Python?",
            results=results,
            working_memory=wm,
        )
        assert isinstance(response, str)
        assert len(response) > 0
        # Verify the search-specific prompt path was used
        call_args = ollama.chat.call_args
        messages = call_args[1].get("messages") or call_args[0][1] if call_args[0] else call_args[1]["messages"]
        system_msg = messages[0]["content"]
        assert "VERALTET" in system_msg  # search-specific system prompt mentions training data being outdated

    @pytest.mark.asyncio
    async def test_formulate_with_non_search_results(self, config: JarvisConfig) -> None:
        """Results from read_file -> normal prompt (not search-specific)."""
        from jarvis.models import ToolResult

        content = "Die Datei enthaelt Konfigurationseinstellungen."
        ollama = _mock_ollama(content)
        planner = Planner(config, ollama, _mock_router())
        wm = WorkingMemory(session_id="test")
        results = [
            ToolResult(tool_name="read_file", content="key=value", is_error=False),
        ]

        response = await planner.formulate_response(
            user_message="Was steht in der Datei?",
            results=results,
            working_memory=wm,
        )
        assert isinstance(response, str)
        # Verify the normal prompt path was used
        call_args = ollama.chat.call_args
        messages = call_args[1].get("messages") or call_args[0][1] if call_args[0] else call_args[1]["messages"]
        system_msg = messages[0]["content"]
        # Normal path says "Tool-Ergebnisse", not the aggressive search prompt
        assert "Tool-Ergebnisse" in system_msg

    @pytest.mark.asyncio
    async def test_formulate_llm_error(self, config: JarvisConfig) -> None:
        """OllamaError during formulate_response -> fallback text."""
        from jarvis.core.model_router import OllamaError
        from jarvis.models import ToolResult

        ollama = AsyncMock()
        ollama.chat = AsyncMock(side_effect=OllamaError("model unavailable"))
        planner = Planner(config, ollama, _mock_router())
        wm = WorkingMemory(session_id="test")
        results = [ToolResult(tool_name="web_search", content="data", is_error=False)]

        response = await planner.formulate_response(
            user_message="test",
            results=results,
            working_memory=wm,
        )
        assert isinstance(response, str)
        # Nach Retry-Logik: Rohergebnisse als Fallback oder Fehlermeldung
        assert (
            "Ergebnisse" in response
            or "nicht zusammenfassen" in response
            or "erneut" in response
        )

    @pytest.mark.asyncio
    async def test_formulate_with_core_memory(self, config: JarvisConfig) -> None:
        """WorkingMemory with core_memory_text set -> injected as system message."""
        from jarvis.models import ToolResult

        content = "Antwort mit Kontext aus dem Gedaechtnis."
        ollama = _mock_ollama(content)
        planner = Planner(config, ollama, _mock_router())
        wm = WorkingMemory(session_id="test", core_memory_text="Du gehoerst Alexander.")
        results = [ToolResult(tool_name="read_file", content="content", is_error=False)]

        response = await planner.formulate_response(
            user_message="Wem gehoerst du?",
            results=results,
            working_memory=wm,
        )
        assert isinstance(response, str)
        # Verify core_memory_text was injected into messages
        call_args = ollama.chat.call_args
        messages = call_args[1].get("messages") or call_args[0][1] if call_args[0] else call_args[1]["messages"]
        # There should be a system message containing the core memory text
        core_msgs = [m for m in messages if m["role"] == "system" and "Alexander" in m["content"]]
        assert len(core_msgs) >= 1


# ============================================================================
# TestGenerateEscalationExtended
# ============================================================================


class TestGenerateEscalationExtended:
    @pytest.mark.asyncio
    async def test_escalation_llm_error_fallback(self, config: JarvisConfig) -> None:
        """OllamaError during escalation -> fallback message containing tool name and reason."""
        from jarvis.core.model_router import OllamaError

        ollama = AsyncMock()
        ollama.chat = AsyncMock(side_effect=OllamaError("LLM down"))
        planner = Planner(config, ollama, _mock_router())
        wm = WorkingMemory(session_id="test")

        response = await planner.generate_escalation(
            tool="exec_command",
            reason="Gefaehrlicher Befehl",
            working_memory=wm,
        )
        assert isinstance(response, str)
        assert "exec_command" in response
        assert "Gefaehrlicher Befehl" in response


# ============================================================================
# TestRecordCost
# ============================================================================


class TestRecordCost:
    def test_record_cost_with_tracker(self, config: JarvisConfig) -> None:
        """Verify cost_tracker.record_llm_call is called with correct values."""
        cost_tracker = MagicMock()
        ollama = _mock_ollama("test")
        planner = Planner(config, ollama, _mock_router(), cost_tracker=cost_tracker)

        response = {
            "prompt_eval_count": 100,
            "eval_count": 50,
            "message": {"content": "test"},
        }
        planner._record_cost(response, "qwen3:32b", session_id="sess-1")

        cost_tracker.record_llm_call.assert_called_once_with(
            model="qwen3:32b",
            input_tokens=100,
            output_tokens=50,
            session_id="sess-1",
        )

    def test_record_cost_no_tracker(self, config: JarvisConfig) -> None:
        """No cost_tracker configured -> no error raised."""
        ollama = _mock_ollama("test")
        planner = Planner(config, ollama, _mock_router())  # no cost_tracker
        assert planner._cost_tracker is None

        response = {"prompt_eval_count": 100, "eval_count": 50}
        # Should not raise any exception
        planner._record_cost(response, "qwen3:32b")

    def test_record_cost_tracker_exception(self, config: JarvisConfig) -> None:
        """cost_tracker.record_llm_call raises -> caught silently, no propagation."""
        cost_tracker = MagicMock()
        cost_tracker.record_llm_call.side_effect = RuntimeError("DB connection lost")
        ollama = _mock_ollama("test")
        planner = Planner(config, ollama, _mock_router(), cost_tracker=cost_tracker)

        response = {"prompt_eval_count": 100, "eval_count": 50}
        # Should not raise even though the tracker explodes
        planner._record_cost(response, "qwen3:32b", session_id="s1")
        cost_tracker.record_llm_call.assert_called_once()


# ============================================================================
# TestLoadPromptFromFile
# ============================================================================


class TestLoadPromptFromFile:
    def test_load_prompt_from_md_file(self, config: JarvisConfig) -> None:
        """Write .md file in prompts dir -> loaded by _load_prompt_from_file."""
        prompts_dir = config.jarvis_home / "prompts"
        prompts_dir.mkdir(parents=True, exist_ok=True)
        md_file = prompts_dir / "TEST_PROMPT.md"
        md_file.write_text("Custom prompt content from .md", encoding="utf-8")

        ollama = _mock_ollama("test")
        planner = Planner(config, ollama, _mock_router())

        result = planner._load_prompt_from_file("TEST_PROMPT.md", "fallback default")
        assert result == "Custom prompt content from .md"

    def test_load_prompt_from_txt_fallback(self, config: JarvisConfig) -> None:
        """No .md but .txt exists -> loaded as migration fallback."""
        prompts_dir = config.jarvis_home / "prompts"
        prompts_dir.mkdir(parents=True, exist_ok=True)
        txt_file = prompts_dir / "TEST_PROMPT.txt"
        txt_file.write_text("Content from .txt fallback", encoding="utf-8")

        ollama = _mock_ollama("test")
        planner = Planner(config, ollama, _mock_router())

        result = planner._load_prompt_from_file(
            "TEST_PROMPT.md", "fallback default", fallback_txt="TEST_PROMPT.txt"
        )
        assert result == "Content from .txt fallback"

    def test_load_prompt_fallback_to_default(self, config: JarvisConfig) -> None:
        """No files on disk -> returns fallback string."""
        ollama = _mock_ollama("test")
        planner = Planner(config, ollama, _mock_router())

        result = planner._load_prompt_from_file(
            "NONEXISTENT_PROMPT.md", "the fallback value"
        )
        assert result == "the fallback value"

    def test_reload_prompts(self, config: JarvisConfig) -> None:
        """Call reload_prompts() -> no crash, prompts are refreshed."""
        ollama = _mock_ollama("test")
        planner = Planner(config, ollama, _mock_router())

        # Should not raise
        planner.reload_prompts()
        # After reload, the templates should still be valid strings
        assert isinstance(planner._system_prompt_template, str)
        assert isinstance(planner._replan_prompt_template, str)
        assert isinstance(planner._escalation_prompt_template, str)
        assert len(planner._system_prompt_template) > 0


# ============================================================================
# TestSanitizeJsonEscapes
# ============================================================================


class TestSanitizeJsonEscapes:
    def test_sanitize_valid_escapes(self) -> None:
        """Valid JSON escapes should remain unchanged."""
        valid = r'{"key": "line1\nline2\ttab\\backslash\"quote"}'
        result = Planner._sanitize_json_escapes(valid)
        assert result == valid

    def test_sanitize_invalid_escapes(self) -> None:
        r"""Invalid escapes like \q, \s, \d should be doubled to \\q, \\s, \\d."""
        # \s and \d are not valid JSON escapes
        invalid = r'{"regex": "\s+\d+"}'
        result = Planner._sanitize_json_escapes(invalid)
        assert r"\\s" in result
        assert r"\\d" in result
        # The result should now be valid JSON
        import json
        parsed = json.loads(result)
        assert "regex" in parsed


# ============================================================================
# TestTryParseJson
# ============================================================================


class TestTryParseJson:
    def test_parse_valid_json(self, config: JarvisConfig) -> None:
        """Normal valid JSON should parse successfully."""
        ollama = _mock_ollama("test")
        planner = Planner(config, ollama, _mock_router())

        result = planner._try_parse_json('{"goal": "test", "steps": [], "confidence": 0.9}')
        assert result is not None
        assert result["goal"] == "test"
        assert result["confidence"] == 0.9

    def test_parse_broken_escapes(self, config: JarvisConfig) -> None:
        r"""JSON with broken escapes like \s should still parse via sanitization."""
        ollama = _mock_ollama("test")
        planner = Planner(config, ollama, _mock_router())

        # This would fail with strict json.loads because \s is invalid
        broken = r'{"code": "re.sub(\s+, \"\", text)"}'
        result = planner._try_parse_json(broken)
        # Should parse via one of the fallback strategies
        assert result is not None
        assert "code" in result

    def test_parse_invalid_json(self, config: JarvisConfig) -> None:
        """Completely broken JSON -> returns None."""
        ollama = _mock_ollama("test")
        planner = Planner(config, ollama, _mock_router())

        result = planner._try_parse_json("this is not json at all {{{")
        assert result is None


# ============================================================================
# TestFormatResults
# ============================================================================


class TestFormatResults:
    def test_format_success_and_error(self, config: JarvisConfig) -> None:
        """Mix of successful and error results -> correct status markers."""
        from jarvis.models import ToolResult

        ollama = _mock_ollama("test")
        planner = Planner(config, ollama, _mock_router())

        results = [
            ToolResult(tool_name="read_file", content="file content here", is_error=False),
            ToolResult(tool_name="exec_command", content="", is_error=True, error_message="permission denied"),
        ]
        formatted = planner._format_results(results)

        assert isinstance(formatted, str)
        # Successful result has checkmark
        assert "\u2713" in formatted  # ✓
        # Error result has cross
        assert "\u2717" in formatted  # ✗
        assert "read_file" in formatted
        assert "exec_command" in formatted

    def test_format_search_results_full_content(self, config: JarvisConfig) -> None:
        """web_search results get 4000 chars limit (HIGH_CONTEXT_TOOLS)."""
        from jarvis.models import ToolResult

        ollama = _mock_ollama("test")
        planner = Planner(config, ollama, _mock_router())

        # Create a web_search result with content longer than 1000 but shorter than 4000
        long_content = "A" * 3000
        results = [
            ToolResult(tool_name="web_search", content=long_content, is_error=False),
        ]
        formatted = planner._format_results(results)

        # The full 3000 chars should be preserved (within 4000 limit)
        assert "A" * 3000 in formatted
        assert "[... Output" not in formatted  # Should NOT be truncated

        # Now test that a non-search tool with the same content IS truncated
        results_non_search = [
            ToolResult(tool_name="read_file", content=long_content, is_error=False),
        ]
        formatted_non_search = planner._format_results(results_non_search)

        # read_file only gets 1000 chars, so 3000 chars will be truncated
        assert "A" * 3000 not in formatted_non_search
        assert "[... Output" in formatted_non_search
