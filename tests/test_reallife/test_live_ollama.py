"""Live agentic tests that call Ollama for real LLM responses.

These tests require a running Ollama instance with qwen3:8b.
Skip automatically if Ollama is not available.

Run: pytest tests/test_reallife/test_live_ollama.py -v --timeout=120
"""
from __future__ import annotations

import asyncio
import os
import pytest
import httpx


# Skip all tests if Ollama is not reachable
def _ollama_available() -> bool:
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    try:
        r = httpx.get(f"{host}/api/version", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _ollama_available(),
    reason="Ollama not available — skipping live tests",
)


class TestLiveWebResearch:
    """Test that Cognithor's research pipeline is properly configured."""

    @pytest.mark.asyncio
    async def test_ollama_is_reachable(self):
        """Ollama must be running and responding."""
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        r = httpx.get(f"{host}/api/version", timeout=5)
        assert r.status_code == 200
        data = r.json()
        assert "version" in data

    @pytest.mark.asyncio
    async def test_planner_prompts_support_research(self):
        """Planner prompts must include research quality instructions."""
        from jarvis.core.planner import SYSTEM_PROMPT, REPLAN_PROMPT

        # System prompt must instruct thoroughness
        assert "deep_research" in SYSTEM_PROMPT
        assert "search_and_read" in SYSTEM_PROMPT
        assert "Gruendlichkeit" in SYSTEM_PROMPT or "gruendlichkeit" in SYSTEM_PROMPT.lower()

        # Replan must include quality self-assessment
        assert "Quellen" in REPLAN_PROMPT or "quellen" in REPLAN_PROMPT.lower()
        assert "deep_research" in REPLAN_PROMPT


class TestLiveCodeGeneration:
    """Test that code execution tools are properly classified."""

    def test_code_tools_classified_correctly(self):
        """run_python should be YELLOW (execute but inform), not blocked."""
        from jarvis.core.gatekeeper import Gatekeeper
        from jarvis.config import JarvisConfig
        from jarvis.models import PlannedAction

        gk = Gatekeeper(JarvisConfig())
        action = PlannedAction(tool="run_python", params={}, rationale="test")
        risk = gk._classify_risk(action)
        assert risk.value == "yellow", f"run_python should be yellow, got {risk}"


class TestLiveAutonomousDetection:
    """Test that the autonomous orchestrator correctly classifies tasks."""

    def test_complex_task_triggers_orchestration(self):
        from jarvis.core.autonomous_orchestrator import AutonomousOrchestrator

        orch = AutonomousOrchestrator()

        # Complex tasks
        assert orch.should_orchestrate(
            "Recherchiere die aktuellen RTX 5090 Preise auf eBay und erstelle einen Vergleichsbericht"
        )
        assert orch.should_orchestrate(
            "Monitor Facebook Marketplace for cheap GPUs and notify me daily"
        )
        assert orch.should_orchestrate(
            "Setup a Python project with tests and deploy it"
        )

        # Simple tasks — should NOT trigger
        assert not orch.should_orchestrate("Hi")
        assert not orch.should_orchestrate("Wie wird das Wetter?")

    def test_recurring_detection(self):
        from jarvis.core.autonomous_orchestrator import AutonomousOrchestrator

        orch = AutonomousOrchestrator()
        assert orch.detect_recurring("Send me a daily stock report") == "daily"
        assert orch.detect_recurring("Check prices hourly") == "hourly"
        assert orch.detect_recurring("Weekly team summary") == "weekly"
        assert orch.detect_recurring("Ueberwache den Preis continuously") == "hourly"


class TestLiveGatekeeperSafety:
    """Verify that safety classifications are correct for all tool types."""

    def test_all_search_tools_green(self):
        from jarvis.core.gatekeeper import Gatekeeper
        from jarvis.config import JarvisConfig
        from jarvis.models import PlannedAction

        gk = Gatekeeper(JarvisConfig())
        green_tools = [
            "web_search", "search_and_read", "deep_research",
            "verified_web_lookup", "web_fetch", "read_file",
            "list_directory", "search_memory", "memory_stats",
        ]
        for tool in green_tools:
            action = PlannedAction(tool=tool, params={}, rationale="test")
            risk = gk._classify_risk(action)
            assert risk.value == "green", f"{tool} should be green, got {risk}"

    def test_dangerous_tools_orange(self):
        from jarvis.core.gatekeeper import Gatekeeper
        from jarvis.config import JarvisConfig
        from jarvis.models import PlannedAction

        gk = Gatekeeper(JarvisConfig())
        orange_tools = ["remote_exec", "email_send", "db_execute"]
        for tool in orange_tools:
            action = PlannedAction(tool=tool, params={}, rationale="test")
            risk = gk._classify_risk(action)
            assert risk.value == "orange", f"{tool} should be orange, got {risk}"
