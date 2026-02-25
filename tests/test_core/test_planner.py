"""Tests für den Planner – LLM-basiertes Planen und Reflektieren.

Testet:
  - Direkte Antwort (kein Tool-Call)
  - JSON-Plan-Extraktion aus LLM-Output
  - Tool-Call Parsing (Ollama native)
  - Replan nach Tool-Ergebnissen
  - Eskalation bei wiederholter Blockierung
  - Robustheit bei kaputtem JSON
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from jarvis.config import JarvisConfig
from jarvis.core.model_router import ModelRouter, OllamaClient
from jarvis.core.planner import Planner
from jarvis.models import (
    ToolResult,
    WorkingMemory,
)


@pytest.fixture()
def config(tmp_path) -> JarvisConfig:
    return JarvisConfig(jarvis_home=tmp_path)


@pytest.fixture()
def mock_ollama() -> AsyncMock:
    return AsyncMock(spec=OllamaClient)


@pytest.fixture()
def mock_router(config: JarvisConfig) -> MagicMock:
    router = MagicMock(spec=ModelRouter)
    router.select_model.return_value = "qwen3:32b"
    router.get_model_config.return_value = {
        "temperature": 0.7,
        "top_p": 0.9,
        "context_window": 32768,
    }
    return router


@pytest.fixture()
def planner(config: JarvisConfig, mock_ollama: AsyncMock, mock_router: MagicMock) -> Planner:
    return Planner(config, mock_ollama, mock_router)


@pytest.fixture()
def working_memory() -> WorkingMemory:
    return WorkingMemory(session_id="test-session")


# ============================================================================
# Direkte Antwort
# ============================================================================


class TestDirectResponse:
    @pytest.mark.asyncio
    async def test_simple_question_direct_answer(
        self, planner: Planner, mock_ollama: AsyncMock, working_memory: WorkingMemory
    ) -> None:
        """Einfache Fragen ohne Tool-Call → direkte Antwort."""
        mock_ollama.chat.return_value = {
            "message": {
                "role": "assistant",
                "content": "Guten Morgen! Wie kann ich helfen?",
            },
        }
        plan = await planner.plan("Guten Morgen!", working_memory, {})
        assert plan.is_direct_response
        assert not plan.has_actions
        assert "Morgen" in plan.direct_response

    @pytest.mark.asyncio
    async def test_llm_error_returns_error_plan(
        self, planner: Planner, mock_ollama: AsyncMock, working_memory: WorkingMemory
    ) -> None:
        """LLM-Fehler → Plan mit Fehlermeldung."""
        from jarvis.core.model_router import OllamaError

        mock_ollama.chat.side_effect = OllamaError("Connection refused")
        plan = await planner.plan("Test", working_memory, {})
        assert plan.is_direct_response
        assert plan.confidence == 0.0
        assert "Problem" in plan.direct_response or "Fehler" in plan.direct_response


# ============================================================================
# JSON-Plan Extraktion
# ============================================================================


class TestPlanExtraction:
    @pytest.mark.asyncio
    async def test_json_plan_in_code_block(
        self, planner: Planner, mock_ollama: AsyncMock, working_memory: WorkingMemory
    ) -> None:
        """LLM gibt JSON-Plan in ```json ``` Block zurück."""
        mock_ollama.chat.return_value = {
            "message": {
                "role": "assistant",
                "content": """Ich lese die Datei für dich.

```json
{
  "goal": "Datei lesen",
  "reasoning": "User will Datei-Inhalt sehen",
  "steps": [
    {
      "tool": "read_file",
      "params": {"path": "/test/file.txt"},
      "rationale": "Datei lesen",
      "risk_estimate": "green"
    }
  ],
  "confidence": 0.9
}
```""",
            },
        }
        plan = await planner.plan(
            "Lies die Datei /test/file.txt", working_memory, {"read_file": {}}
        )
        assert plan.has_actions
        assert len(plan.steps) == 1
        assert plan.steps[0].tool == "read_file"
        assert plan.confidence == 0.9

    @pytest.mark.asyncio
    async def test_raw_json_without_code_block(
        self, planner: Planner, mock_ollama: AsyncMock, working_memory: WorkingMemory
    ) -> None:
        """LLM gibt rohes JSON ohne Code-Block zurück."""
        mock_ollama.chat.return_value = {
            "message": {
                "role": "assistant",
                "content": '{"goal": "test", "steps": [{"tool": "list_directory", "params": {"path": "/"}}], "confidence": 0.7}', # noqa: E501
            },
        }
        plan = await planner.plan(
            "Zeig mir das Verzeichnis", working_memory, {"list_directory": {}}
        )
        assert plan.has_actions
        assert plan.steps[0].tool == "list_directory"

    @pytest.mark.asyncio
    async def test_broken_json_falls_back_to_direct(
        self, planner: Planner, mock_ollama: AsyncMock, working_memory: WorkingMemory
    ) -> None:
        """Kaputtes JSON → wird als direkte Antwort interpretiert."""
        mock_ollama.chat.return_value = {
            "message": {
                "role": "assistant",
                "content": "Hier ist {kaputtes JSON das nicht parst}",
            },
        }
        plan = await planner.plan("Test", working_memory, {})
        assert plan.is_direct_response


# ============================================================================
# Ollama Native Tool-Calls
# ============================================================================


class TestNativeToolCalls:
    @pytest.mark.asyncio
    async def test_ollama_tool_calls(
        self, planner: Planner, mock_ollama: AsyncMock, working_memory: WorkingMemory
    ) -> None:
        """Ollama gibt native tool_calls zurück."""
        mock_ollama.chat.return_value = {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "write_file",
                            "arguments": {"path": "/out.txt", "content": "hello"},
                        }
                    }
                ],
            },
        }
        plan = await planner.plan("Schreib hello in out.txt", working_memory, {"write_file": {}})
        assert plan.has_actions
        assert plan.steps[0].tool == "write_file"
        assert plan.steps[0].params["content"] == "hello"


# ============================================================================
# Replan
# ============================================================================


class TestReplan:
    @pytest.mark.asyncio
    async def test_replan_after_results(
        self, planner: Planner, mock_ollama: AsyncMock, working_memory: WorkingMemory
    ) -> None:
        """Replan interpretiert Tool-Ergebnisse."""
        mock_ollama.chat.return_value = {
            "message": {
                "role": "assistant",
                "content": "Die Datei enthält 42 Zeilen mit Kundendaten.",
            },
        }
        results = [
            ToolResult(tool_name="read_file", content="line1\nline2\n...42 lines total"),
        ]
        plan = await planner.replan("Wie viele Zeilen?", results, working_memory, {})
        assert plan.is_direct_response
        assert "42" in plan.direct_response


# ============================================================================
# Eskalation
# ============================================================================


class TestEscalation:
    @pytest.mark.asyncio
    async def test_generate_escalation(
        self, planner: Planner, mock_ollama: AsyncMock, working_memory: WorkingMemory
    ) -> None:
        mock_ollama.chat.return_value = {
            "message": {
                "role": "assistant",
                "content": "Ich konnte die Datei nicht löschen, weil der Gatekeeper dies blockiert hat.",  # noqa: E501
            },
        }
        msg = await planner.generate_escalation(
            "delete_file", "Datei löschen erfordert Bestätigung", working_memory
        )
        assert "blockiert" in msg.lower() or "löschen" in msg.lower()

    @pytest.mark.asyncio
    async def test_escalation_fallback_on_error(
        self, planner: Planner, mock_ollama: AsyncMock, working_memory: WorkingMemory
    ) -> None:
        from jarvis.core.model_router import OllamaError

        mock_ollama.chat.side_effect = OllamaError("timeout")
        msg = await planner.generate_escalation("delete_file", "Blockiert", working_memory)
        assert "delete_file" in msg  # Fallback-Text enthält Tool-Namen


# ============================================================================
# Formulate Response
# ============================================================================


class TestFormulateResponse:
    @pytest.mark.asyncio
    async def test_formulate_from_results(
        self, planner: Planner, mock_ollama: AsyncMock, working_memory: WorkingMemory
    ) -> None:
        mock_ollama.chat.return_value = {
            "message": {
                "role": "assistant",
                "content": "Die Datei wurde erfolgreich geschrieben.",
            },
        }
        results = [ToolResult(tool_name="write_file", content="OK")]
        response = await planner.formulate_response("Schreib die Datei", results, working_memory)
        assert "erfolgreich" in response.lower()

    @pytest.mark.asyncio
    async def test_core_memory_in_context(
        self, planner: Planner, mock_ollama: AsyncMock, working_memory: WorkingMemory
    ) -> None:
        """Core Memory Text wird in den System-Prompt eingebaut."""
        working_memory.core_memory_text = "Ich bin Jarvis, Assistent des Benutzers."
        mock_ollama.chat.return_value = {
            "message": {"role": "assistant", "content": "Antwort"},
        }
        results = [ToolResult(tool_name="test", content="data")]
        await planner.formulate_response("Test", results, working_memory)
        # Prüfe dass core_memory im System-Prompt war
        call_args = mock_ollama.chat.call_args
        messages = call_args.kwargs.get("messages", call_args[1] if len(call_args) > 1 else [])
        if not messages:
            messages = call_args[0][1] if len(call_args[0]) > 1 else []
        # Core Memory sollte irgendwo in den Messages auftauchen
        all_content = " ".join(m.get("content", "") for m in messages)
        assert "Jarvis" in all_content
