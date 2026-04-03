"""CU Agent Executor — Closed-loop desktop automation agent.

Implements the Screenshot→Decide→Act cycle for Computer Use.
Uses a single vision-language model (qwen3-vl:32b) for both
planning and screenshot analysis — zero model swaps.

Architecture:
  PGE Loop detects CU plan → delegates to CUAgentExecutor
  → agent runs screenshot→decide→act cycles until DONE or abort
  → results flow back to PGE loop for response formulation
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from jarvis.models import ActionPlan, PlannedAction, ToolResult
from jarvis.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class CUAgentConfig:
    """Configuration for the CU Agent Loop."""

    max_iterations: int = 30
    max_duration_seconds: int = 480  # 8 minutes
    vision_model: str = "qwen3-vl:32b"
    screenshot_after_action: bool = True
    stuck_detection_threshold: int = 3


@dataclass
class CUSubTask:
    """A single phase of a decomposed CU goal."""

    name: str
    goal: str
    completion_hint: str
    max_iterations: int = 10
    available_tools: list[str] = field(default_factory=list)
    extract_content: bool = False
    content_key: str = ""
    output_file: str = ""
    status: str = "pending"


@dataclass
class CUTaskPlan:
    """Full decomposed plan for a complex CU goal."""

    original_goal: str
    sub_tasks: list[CUSubTask]
    output_filename: str = ""
    variables: dict[str, str] = field(default_factory=dict)


@dataclass
class CUAgentResult:
    """Result of a CU Agent execution."""

    success: bool = False
    iterations: int = 0
    duration_ms: int = 0
    tool_results: list[ToolResult] = field(default_factory=list)
    final_screenshot_description: str = ""
    abort_reason: str = ""
    extracted_content: str = ""
    action_history: list[str] = field(default_factory=list)
    output_files: list[str] = field(default_factory=list)
    task_summary: str = ""


class CUAgentExecutor:
    """Closed-loop agent for desktop automation via Computer Use tools.

    Executes a Screenshot→Decide→Act cycle until the goal is reached
    or an abort condition triggers. Uses a single vision-language model
    for both planning and screenshot analysis — zero model swaps.
    """

    _CU_DECIDE_PROMPT = (
        "Du steuerst den Desktop des Users. Ziel: {goal}\n\n"
        "Bisherige Aktionen:\n{action_history}\n\n"
        "Aktueller Screenshot:\n{screenshot_description}\n\n"
        "Erkannte UI-Elemente:\n{elements_json}\n\n"
        "Was ist der NAECHSTE einzelne Schritt? Antworte mit EINEM der folgenden:\n\n"
        "1. Ein einzelner Tool-Call als JSON:\n"
        '{{"tool": "tool_name", "params": {{...}}, "rationale": "Warum"}}\n\n'
        "2. Text-Extraktion:\n"
        '{{"tool": "extract_text", "params": {{}}, "rationale": "Text vom Bildschirm lesen"}}\n\n'
        "3. Wenn das Ziel erreicht ist:\n"
        "DONE: [Zusammenfassung was erreicht wurde]\n\n"
        "Verfuegbare Tools: exec_command, computer_screenshot, computer_click, "
        "computer_type, computer_hotkey, computer_scroll\n\n"
        "WICHTIG: Plane immer nur EINEN Schritt. Nach der Ausfuehrung "
        "bekommst du einen neuen Screenshot."
    )

    def __init__(
        self,
        planner: Any,
        mcp_client: Any,
        gatekeeper: Any,
        working_memory: Any,
        tool_schemas: dict[str, Any],
        config: CUAgentConfig | None = None,
    ) -> None:
        self._planner = planner
        self._mcp = mcp_client
        self._gatekeeper = gatekeeper
        self._wm = working_memory
        self._tool_schemas = tool_schemas
        self._config = config or CUAgentConfig()
        self._action_history: list[str] = []
        self._recent_actions: list[str] = []

    def _check_abort(
        self,
        result: CUAgentResult,
        start: float,
        cancel_check: Callable | None,
    ) -> str:
        """Check all abort conditions. Returns reason or empty string."""
        if cancel_check and cancel_check():
            return "user_cancel"
        if result.iterations >= self._config.max_iterations:
            return "max_iterations"
        if time.monotonic() - start > self._config.max_duration_seconds:
            return "timeout"
        if (
            len(self._recent_actions) >= self._config.stuck_detection_threshold
            and len(set(self._recent_actions)) == 1
        ):
            return "stuck_loop"
        return ""

    @staticmethod
    def _format_params(params: dict) -> str:
        """Compact param string for action history."""
        parts = []
        for k, v in params.items():
            sv = str(v)
            if len(sv) > 30:
                sv = sv[:27] + "..."
            parts.append(f"{k}={sv}")
        return ", ".join(parts)

    @staticmethod
    def _format_elements(elements: list[dict]) -> str:
        """Format elements list for the decide prompt."""
        if not elements:
            return "(keine Elemente erkannt)"
        compact = [
            {k: e[k] for k in ("name", "type", "x", "y", "text") if k in e} for e in elements[:15]
        ]
        return json.dumps(compact, ensure_ascii=False, indent=None)

    @staticmethod
    def _check_completion_hint(hint: str, screenshot_desc: str) -> bool:
        """Fuzzy check if the completion hint is satisfied (60% keyword overlap)."""
        if not hint:
            return False
        hint_lower = hint.lower()
        desc_lower = screenshot_desc.lower()
        keywords = [w for w in hint_lower.split() if len(w) > 4]
        if not keywords:
            return False
        matches = sum(1 for kw in keywords if kw in desc_lower)
        return matches / len(keywords) >= 0.6

    @staticmethod
    def _screenshot_similarity(prev: str, curr: str) -> float:
        """Jaccard similarity between two screenshot descriptions."""
        words_a = set(prev.lower().split())
        words_b = set(curr.lower().split())
        if not words_a or not words_b:
            return 0.0
        return len(words_a & words_b) / len(words_a | words_b)

    async def execute(
        self,
        goal: str,
        initial_plan: ActionPlan,
        status_callback: Callable | None = None,
        cancel_check: Callable | None = None,
    ) -> CUAgentResult:
        """Run the CU agent loop until done or aborted."""
        result = CUAgentResult()
        start = time.monotonic()

        async def _status(phase: str, msg: str) -> None:
            if status_callback:
                try:
                    await status_callback(phase, msg)
                except Exception:
                    pass

        # Execute initial plan steps
        await _status("computer_use", f"Starte: {goal[:60]}...")
        for step in initial_plan.steps:
            tool_result = await self._execute_tool(step.tool, step.params)
            result.tool_results.append(tool_result)
            self._action_history.append(
                f"{step.tool}({self._format_params(step.params)}) "
                f"-> {'OK' if tool_result.success else 'FAIL'}"
            )

        # Main agent loop: screenshot → decide → act
        while True:
            result.iterations += 1

            abort = self._check_abort(result, start, cancel_check)
            if abort:
                result.abort_reason = abort
                break

            await _status(
                "computer_use",
                f"Schritt {result.iterations}/{self._config.max_iterations}: "
                f"Analysiere Bildschirm...",
            )

            screenshot = await self._take_and_analyze_screenshot()
            if not screenshot:
                self._action_history.append("computer_screenshot() -> FAIL")
                continue

            result.final_screenshot_description = screenshot.get("description", "")

            decision = await self._decide_next_step(goal, screenshot)

            if decision is None:
                self._action_history.append("decide() -> no valid action")
                continue

            if decision.get("done"):
                result.success = True
                result.abort_reason = "done"
                summary = decision.get("summary", "")
                self._action_history.append(f"DONE: {summary}")
                break

            if decision.get("tool") == "extract_text":
                text = await self._extract_text_from_screen()
                if text:
                    result.extracted_content += text + "\n\n"
                    self._action_history.append(f"extract_text() -> {len(text)} chars")
                continue

            tool = decision["tool"]
            params = decision.get("params", {})
            await _status("computer_use", f"Schritt {result.iterations}: {tool}...")

            tool_result = await self._execute_tool(tool, params)
            result.tool_results.append(tool_result)
            self._action_history.append(
                f"{tool}({self._format_params(params)}) "
                f"-> {'OK' if tool_result.success else 'FAIL'}"
            )

            action_key = f"{tool}:{sorted(params.items())}"
            self._recent_actions.append(action_key)
            if len(self._recent_actions) > self._config.stuck_detection_threshold:
                self._recent_actions.pop(0)

        result.duration_ms = int((time.monotonic() - start) * 1000)
        result.action_history = list(self._action_history)
        log.info(
            "cu_agent_complete",
            success=result.success,
            iterations=result.iterations,
            duration_ms=result.duration_ms,
            abort_reason=result.abort_reason,
            actions=len(self._action_history),
        )
        return result

    def _parse_tool_decision(self, raw: str) -> dict | None:
        """Parse a single tool call from the planner response."""
        # Tier 1: direct JSON parse
        try:
            data = json.loads(raw)
            if isinstance(data, dict) and "tool" in data:
                return data
        except (json.JSONDecodeError, ValueError):
            pass

        # Tier 2: markdown code block
        md_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
        if md_match:
            try:
                data = json.loads(md_match.group(1))
                if isinstance(data, dict) and "tool" in data:
                    return data
            except (json.JSONDecodeError, ValueError):
                pass

        # Tier 3: find JSON object with "tool" key
        json_match = re.search(r'\{[^{}]*"tool"[^{}]*\}', raw)
        if json_match:
            try:
                data = json.loads(json_match.group())
                if "tool" in data:
                    return data
            except (json.JSONDecodeError, ValueError):
                pass

        return None

    async def _execute_tool(self, tool: str, params: dict) -> ToolResult:
        """Execute a single CU tool via MCP client."""
        handler = self._mcp._builtin_handlers.get(tool)
        if not handler:
            return ToolResult(
                tool_name=tool,
                content=f"Tool '{tool}' not found",
                is_error=True,
            )
        try:
            result = await handler(**params)
            content = str(result) if not isinstance(result, str) else result
            return ToolResult(
                tool_name=tool,
                content=content[:5000],
                is_error=False,
            )
        except Exception as exc:
            return ToolResult(
                tool_name=tool,
                content=f"Error: {exc}",
                is_error=True,
            )

    async def _take_and_analyze_screenshot(self) -> dict | None:
        """Take screenshot via CU tool and return result with elements."""
        handler = self._mcp._builtin_handlers.get("computer_screenshot")
        if not handler:
            return None
        try:
            return await handler()
        except Exception:
            log.debug("cu_agent_screenshot_failed", exc_info=True)
            return None

    async def _decide_next_step(self, goal: str, screenshot: dict) -> dict | None:
        """Ask the planner what to do next based on the screenshot.

        Returns:
            {"tool": "...", "params": {...}} for an action
            {"done": True, "summary": "..."} for completion
            None if parsing failed
        """
        prompt = self._CU_DECIDE_PROMPT.format(
            goal=goal,
            action_history="\n".join(self._action_history[-10:]) or "(keine)",
            screenshot_description=screenshot.get("description", "")[:1000],
            elements_json=self._format_elements(screenshot.get("elements", [])),
        )

        try:
            response = await self._planner._ollama.chat(
                model=self._config.vision_model,
                messages=[
                    {"role": "system", "content": "Du bist ein Desktop-Automations-Agent."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )
            text = response.get("message", {}).get("content", "")
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

            if text.upper().startswith("DONE"):
                summary = text.split(":", 1)[1].strip() if ":" in text else text[4:].strip()
                return {"done": True, "summary": summary}

            return self._parse_tool_decision(text)

        except Exception as exc:
            log.warning("cu_agent_decide_failed", error=str(exc)[:200])
            return None

    async def _extract_text_from_screen(self) -> str:
        """Extract all visible text from current screen via vision model."""
        try:
            from jarvis.mcp.computer_use import _take_screenshot_b64
            from jarvis.core.vision import build_vision_message, format_for_backend

            b64, _, _ = await asyncio.get_running_loop().run_in_executor(None, _take_screenshot_b64)
            msg = build_vision_message(
                "Lies ALLEN sichtbaren Text in diesem Screenshot ab. "
                "Gib den Text zeilenweise wieder. Antworte NUR mit dem Text.",
                [b64],
            )
            formatted = format_for_backend(msg, "ollama")
            response = await self._planner._ollama.chat(
                model=self._config.vision_model,
                messages=[formatted],
                temperature=0.1,
            )
            return response.get("message", {}).get("content", "")
        except Exception:
            log.debug("cu_agent_extract_text_failed", exc_info=True)
            return ""
