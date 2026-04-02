# Computer Use Phase 2B: Agent Loop — Design Spec

**Date**: 2026-04-02
**Status**: Approved
**Scope**: Multi-turn CU agent with single-model execution, no model swaps
**Depends on**: Phase 2A Vision Engine (complete)
**Roadmap**: `docs/superpowers/specs/2026-04-02-computer-use-phase2-roadmap.md`

## Problem

The current system executes a SINGLE plan with fixed steps. The PGE loop runs: plan → execute all steps → respond. For Computer Use, this means the planner creates a plan upfront without seeing the screen, and can't react to what actually happens.

Additionally, every screenshot→plan cycle requires swapping between `qwen3.5:27b` (planner, 22GB) and `qwen3-vl:32b` (vision, 22GB). With 32GB VRAM total, both can't be loaded simultaneously — each swap costs 10-30 seconds, making multi-turn CU unusably slow.

## Solution

### Single-Model CU Sessions

When Computer Use is active, switch the planner to `qwen3-vl:32b` (the vision-language model). It handles BOTH planning AND screenshot analysis — zero model swaps. The model stays loaded for the entire CU session.

Normal (non-CU) requests continue using `qwen3.5:27b-16k` as before.

### CUAgentExecutor

A new class `CUAgentExecutor` that implements the closed-loop agent cycle:

```
Screenshot → Vision Analysis → Planner Decides Next Step → Execute → Screenshot → ...
```

The PGE loop detects CU plans and delegates to the CUAgentExecutor. When the agent is done, results flow back to the PGE loop for response formulation.

## Architecture

```
User: "Oeffne calc.exe und tippe 3*9*17="

PGE Loop:
  1. Planner creates plan (detects CU tools)
  2. Gateway: _is_cu_plan(plan) == True
  3. Create CUAgentExecutor(model="qwen3-vl:32b")
  4. Delegate: cu_agent.execute(goal, initial_plan)

CUAgentExecutor (internal loop):
  Iteration 1: exec_command("start calc.exe") → OK
  Iteration 2: computer_screenshot() → Vision sees Rechner at x=200,y=300
  Iteration 3: Planner decides → computer_click(x=200,y=300) → OK
  Iteration 4: computer_screenshot() → Rechner focused
  Iteration 5: Planner decides → computer_type("3*9*17=") → OK
  Iteration 6: computer_screenshot() → Display shows 459
  Iteration 7: Planner decides → "DONE: Rechner zeigt 459"

PGE Loop (resumed):
  5. Receive CUAgentResult
  6. Formulate response: "Erledigt. Der Taschenrechner zeigt 459."
```

## Component Changes

### 1. `src/jarvis/core/cu_agent.py` — NEW FILE

The core of Phase 2B. Contains `CUAgentExecutor`, `CUAgentConfig`, `CUAgentResult`.

#### 1.1 Data Classes

```python
from __future__ import annotations

import asyncio
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
    screenshot_after_action: bool = True  # Auto-screenshot after every action
    stuck_detection_threshold: int = 3  # Same action N times = stuck


@dataclass
class CUAgentResult:
    """Result of a CU Agent execution."""

    success: bool = False
    iterations: int = 0
    duration_ms: int = 0
    tool_results: list[ToolResult] = field(default_factory=list)
    final_screenshot_description: str = ""
    abort_reason: str = ""  # "" | "done" | "max_iterations" | "timeout" | "user_cancel" | "stuck_loop"
    extracted_content: str = ""  # Collected text from screenshots
    action_history: list[str] = field(default_factory=list)  # Human-readable log
```

#### 1.2 CUAgentExecutor Class

```python
class CUAgentExecutor:
    """Closed-loop agent for desktop automation via Computer Use tools.

    Executes a Screenshot→Decide→Act cycle until the goal is reached
    or an abort condition triggers. Uses a single vision-language model
    (qwen3-vl:32b) for both planning and screenshot analysis — zero
    model swaps.

    Usage:
        agent = CUAgentExecutor(planner, mcp, gatekeeper, wm, schemas, config)
        result = await agent.execute(goal, initial_plan, status_cb, cancel_check)
    """

    _CU_DECIDE_PROMPT = (
        "Du steuerst den Desktop des Users. Ziel: {goal}\n\n"
        "Bisherige Aktionen:\n{action_history}\n\n"
        "Aktueller Screenshot:\n{screenshot_description}\n\n"
        "Erkannte UI-Elemente:\n{elements_json}\n\n"
        "Was ist der NAECHSTE einzelne Schritt? Antworte mit EINEM der folgenden:\n\n"
        "1. Ein einzelner Tool-Call als JSON:\n"
        '{{"tool": "tool_name", "params": {{...}}, "rationale": "Warum"}}\n\n'
        '2. Text-Extraktion:\n'
        '{{"tool": "extract_text", "params": {{}}, "rationale": "Text vom Bildschirm lesen"}}\n\n'
        '3. Wenn das Ziel erreicht ist:\n'
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
        self._recent_actions: list[str] = []  # For stuck detection
```

#### 1.3 The Execute Method

```python
    async def execute(
        self,
        goal: str,
        initial_plan: ActionPlan,
        status_callback: Callable | None = None,
        cancel_check: Callable | None = None,
    ) -> CUAgentResult:
        """Run the CU agent loop until done or aborted.

        Args:
            goal: User's original request text.
            initial_plan: First plan from the planner.
            status_callback: async (phase, message) for UI progress.
            cancel_check: () -> bool, True if user wants to cancel.

        Returns:
            CUAgentResult with all results, history, and abort reason.
        """
        result = CUAgentResult()
        start = time.monotonic()
        _status = status_callback or (lambda *a, **kw: asyncio.sleep(0))

        # Execute initial plan steps
        await _status("computer_use", f"Starte: {goal[:60]}...")
        for step in initial_plan.steps:
            tool_result = await self._execute_tool(step.tool, step.params)
            result.tool_results.append(tool_result)
            self._action_history.append(
                f"{step.tool}({self._format_params(step.params)}) "
                f"→ {'OK' if tool_result.success else 'FAIL'}"
            )

        # Main agent loop: screenshot → decide → act
        while True:
            result.iterations += 1

            # Abort checks
            abort = self._check_abort(result, start, cancel_check)
            if abort:
                result.abort_reason = abort
                break

            # Progress report
            await _status(
                "computer_use",
                f"Schritt {result.iterations}/{self._config.max_iterations}: "
                f"Analysiere Bildschirm..."
            )

            # Screenshot + Vision
            screenshot = await self._take_and_analyze_screenshot()
            if not screenshot:
                self._action_history.append("computer_screenshot() → FAIL")
                continue

            result.final_screenshot_description = screenshot.get("description", "")
            elements = screenshot.get("elements", [])

            # Decide next action
            decision = await self._decide_next_step(goal, screenshot)

            if decision is None:
                self._action_history.append("decide() → no valid action")
                continue

            # DONE signal
            if decision.get("done"):
                result.success = True
                result.abort_reason = "done"
                summary = decision.get("summary", "")
                self._action_history.append(f"DONE: {summary}")
                break

            # Extract text request
            if decision.get("tool") == "extract_text":
                text = await self._extract_text_from_screen()
                if text:
                    result.extracted_content += text + "\n\n"
                    self._action_history.append(
                        f"extract_text() → {len(text)} chars"
                    )
                continue

            # Execute single tool
            tool = decision["tool"]
            params = decision.get("params", {})
            await _status(
                "computer_use",
                f"Schritt {result.iterations}: {tool}..."
            )

            tool_result = await self._execute_tool(tool, params)
            result.tool_results.append(tool_result)
            self._action_history.append(
                f"{tool}({self._format_params(params)}) "
                f"→ {'OK' if tool_result.success else 'FAIL'}"
            )

            # Stuck detection
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
```

#### 1.4 Internal Methods

```python
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
        # Stuck detection: same action N times in a row
        if (
            len(self._recent_actions) >= self._config.stuck_detection_threshold
            and len(set(self._recent_actions)) == 1
        ):
            return "stuck_loop"
        return ""

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

    async def _decide_next_step(
        self, goal: str, screenshot: dict
    ) -> dict | None:
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

            # Strip think tags
            import re
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

            # Check for DONE signal
            if text.upper().startswith("DONE"):
                summary = text.split(":", 1)[1].strip() if ":" in text else text[4:].strip()
                return {"done": True, "summary": summary}

            # Parse tool call JSON
            return self._parse_tool_decision(text)

        except Exception as exc:
            log.warning("cu_agent_decide_failed", error=str(exc)[:200])
            return None

    def _parse_tool_decision(self, raw: str) -> dict | None:
        """Parse a single tool call from the planner response."""
        import json
        import re

        # Try direct JSON parse
        try:
            data = json.loads(raw)
            if isinstance(data, dict) and "tool" in data:
                return data
        except (json.JSONDecodeError, ValueError):
            pass

        # Try extracting JSON from markdown block
        md_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
        if md_match:
            try:
                data = json.loads(md_match.group(1))
                if isinstance(data, dict) and "tool" in data:
                    return data
            except (json.JSONDecodeError, ValueError):
                pass

        # Try finding JSON object with "tool" key
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

    async def _extract_text_from_screen(self) -> str:
        """Extract all visible text from current screen via vision model."""
        handler = self._mcp._builtin_handlers.get("computer_screenshot")
        if not handler:
            return ""
        try:
            ss = await handler()
            # Use vision to extract text
            if hasattr(self._planner, "_ollama"):
                from jarvis.mcp.computer_use import _take_screenshot_b64
                b64, _, _ = _take_screenshot_b64()

                from jarvis.core.vision import build_vision_message, format_for_backend
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
        import json
        # Compact: only name, type, x, y, text for each
        compact = [
            {k: e[k] for k in ("name", "type", "x", "y", "text") if k in e}
            for e in elements[:15]  # Max 15 elements to save tokens
        ]
        return json.dumps(compact, ensure_ascii=False, indent=None)
```

### 2. `src/jarvis/browser/vision.py` — New `extract_text_from_screenshot`

Add after `analyze_desktop()`, before `stats()`:

```python
    async def extract_text_from_screenshot(
        self,
        screenshot_b64: str,
    ) -> VisionAnalysisResult:
        """Extract all visible text from a screenshot (OCR-like).

        Uses the vision model to read text from the screen, without
        structured element detection. Optimized for content extraction
        (reading articles, posts, documents).

        Args:
            screenshot_b64: Base64-encoded screenshot (PNG).

        Returns:
            VisionAnalysisResult with description containing the extracted text.
        """
        if not self.is_enabled:
            return VisionAnalysisResult(error="Vision nicht aktiviert")

        if not screenshot_b64:
            return VisionAnalysisResult(error="Kein Screenshot-Daten")

        prompt = (
            "Lies ALLEN sichtbaren Text in diesem Screenshot ab. "
            "Gib den Text zeilenweise wieder, so wie er auf dem Bildschirm "
            "erscheint. Antworte NUR mit dem extrahierten Text, kein JSON, "
            "keine Erklaerungen."
        )
        return await self._send_vision_request(screenshot_b64, prompt)
```

### 3. `src/jarvis/gateway/gateway.py` — PGE Loop Integration

#### 3.1 CU Plan Detection

Add as method on Gateway class:

```python
    @staticmethod
    def _is_cu_plan(plan: ActionPlan) -> bool:
        """Check if a plan uses Computer Use tools."""
        _CU_TOOLS = frozenset({
            "computer_screenshot", "computer_click", "computer_type",
            "computer_hotkey", "computer_scroll", "computer_drag",
        })
        return plan.has_actions and any(
            step.tool in _CU_TOOLS for step in plan.steps
        )
```

#### 3.2 CU Delegation in PGE Loop

In `_run_pge_loop`, after the plan is created (after line ~2918 for first iteration, after ~2927 for replan), add before the gatekeeper check:

```python
            # Computer Use: delegate to CUAgentExecutor for multi-turn interaction
            if self._is_cu_plan(plan):
                from jarvis.core.cu_agent import CUAgentConfig, CUAgentExecutor

                _vision_model = getattr(self._config, "vision_model", "qwen3-vl:32b")
                cu_agent = CUAgentExecutor(
                    planner=self._planner,
                    mcp_client=self._mcp_client,
                    gatekeeper=self._gatekeeper,
                    working_memory=wm,
                    tool_schemas=tool_schemas,
                    config=CUAgentConfig(
                        max_iterations=30,
                        max_duration_seconds=480,
                        vision_model=_vision_model,
                    ),
                )
                cu_result = await cu_agent.execute(
                    goal=msg.text,
                    initial_plan=plan,
                    status_callback=_status_cb,
                    cancel_check=lambda: msg.session_id in self._cancelled_sessions,
                )
                all_results.extend(cu_result.tool_results)

                # Add action history to working memory for response formulation
                if cu_result.action_history:
                    wm.add_message(
                        Message(
                            role=MessageRole.SYSTEM,
                            content=(
                                "[Computer Use Ergebnis]\n"
                                + "\n".join(cu_result.action_history[-10:])
                                + f"\n\nAbschluss: {cu_result.abort_reason}"
                                + (f"\nExtrahierter Text:\n{cu_result.extracted_content[:2000]}"
                                   if cu_result.extracted_content else "")
                            ),
                            channel=msg.channel,
                        )
                    )

                await _status_cb("finishing", "Formuliere Antwort...")
                final_response = await self._formulate_response(
                    msg.text, all_results, wm, stream_callback,
                )
                break  # Exit PGE loop
```

#### 3.3 Remove Phase 1 CU Workarounds from PGE Loop

Remove the following blocks that are now handled by CUAgentExecutor:
- The `_CU_DONE` REPLAN block check (lines ~3074-3110) — CUAgent handles its own DONE detection
- The verification screenshot logic — CUAgent takes verification screenshots internally

### 4. `src/jarvis/core/planner.py` — Remove Phase 1 Workarounds

Remove these methods/blocks that are replaced by CUAgentExecutor:
- `_should_force_cu_plan()` method
- `_build_cu_plan()` method
- The `if ... self._should_force_cu_plan(...)` block after `_extract_plan` call
- The CU-specific Few-Shot example in SYSTEM_PROMPT (keep the CU instructions block — it helps when qwen3-vl:32b is used as planner)

### 5. `src/jarvis/core/executor.py` — Remove Phase 1 Workarounds

Remove these methods that are replaced by CUAgentExecutor:
- `_cu_wait_and_focus()` method
- `_cu_ensure_focus()` method
- The calls to `_cu_wait_and_focus()` and `_cu_ensure_focus()` in `_run_with_sem()`
- Keep the `max_parallel=1` logic for CU (still needed for the initial plan execution)

## Abort Conditions

| Condition | Default | Check Point | Result |
|---|---|---|---|
| User cancel | — | Before each iteration | `abort_reason="user_cancel"` |
| Planner "DONE" | — | After each decide() | `abort_reason="done"`, `success=True` |
| Timeout | 480s (8min) | Before each iteration | `abort_reason="timeout"` |
| Max iterations | 30 | Before each iteration | `abort_reason="max_iterations"` |
| Stuck loop | 3 same actions | After each execute() | `abort_reason="stuck_loop"` |

## Tests

### Test 1: CUAgentExecutor — Happy Path

```python
class TestCUAgentHappyPath:
    @pytest.mark.asyncio
    async def test_calculator_flow(self):
        """Simulate: exec → screenshot → click → type → screenshot → DONE"""
        # Mock planner that returns sequential decisions
        # Mock MCP handlers for screenshot/click/type
        # Assert: result.success == True
        # Assert: result.abort_reason == "done"
        # Assert: 6 iterations
        # Assert: action_history contains all steps
```

### Test 2: CUAgentExecutor — Abort Conditions

```python
class TestCUAgentAbort:
    @pytest.mark.asyncio
    async def test_max_iterations(self):
        """Agent stops after 30 iterations."""

    @pytest.mark.asyncio
    async def test_timeout(self):
        """Agent stops after max_duration_seconds."""

    @pytest.mark.asyncio
    async def test_user_cancel(self):
        """Agent stops when cancel_check returns True."""

    @pytest.mark.asyncio
    async def test_stuck_loop(self):
        """Agent stops after 3 identical actions."""
```

### Test 3: CUAgentExecutor — Decision Parsing

```python
class TestCUDecisionParsing:
    def test_parse_tool_json(self):
        """Parse {"tool": "computer_click", "params": {"x": 200, "y": 300}}"""

    def test_parse_done_signal(self):
        """Parse "DONE: Rechner zeigt 459" """

    def test_parse_json_in_markdown(self):
        """Parse ```json ... ``` blocks"""

    def test_parse_garbage_returns_none(self):
        """Unparseable response returns None"""

    def test_parse_with_think_tags(self):
        """<think>...</think> stripped before parsing"""
```

### Test 4: Gateway CU Detection

```python
class TestGatewayCUDetection:
    def test_is_cu_plan_with_cu_tools(self):
        """Plan with computer_click is detected as CU."""

    def test_is_cu_plan_without_cu_tools(self):
        """Plan with only exec_command is NOT CU."""

    def test_is_cu_plan_direct_response(self):
        """Plan with no actions is NOT CU."""
```

### Test 5: VisionAnalyzer extract_text

```python
class TestExtractText:
    @pytest.mark.asyncio
    async def test_extract_text_success(self):
        """Vision model returns extracted text."""

    @pytest.mark.asyncio
    async def test_extract_text_disabled(self):
        """Disabled vision returns error."""
```

## Files Changed

| File | Change |
|---|---|
| `src/jarvis/core/cu_agent.py` (NEW) | `CUAgentExecutor`, `CUAgentConfig`, `CUAgentResult`, decide loop, action history, abort logic, stuck detection, text extraction |
| `src/jarvis/browser/vision.py` (modify) | + `extract_text_from_screenshot()` method |
| `src/jarvis/gateway/gateway.py` (modify) | + `_is_cu_plan()`, CU delegation in PGE loop, remove Phase 1 CU workarounds |
| `src/jarvis/core/planner.py` (modify) | Remove `_should_force_cu_plan`, `_build_cu_plan`, CU override block |
| `src/jarvis/core/executor.py` (modify) | Remove `_cu_wait_and_focus`, `_cu_ensure_focus`, their call sites |
| `tests/test_core/test_cu_agent.py` (NEW) | Tests for CUAgentExecutor, abort conditions, decision parsing |
| `tests/test_browser/test_vision.py` (modify) | + TestExtractText |
| `tests/unit/test_computer_use_vision.py` (modify) | + TestGatewayCUDetection |

## What Does NOT Change

- The 6 CU tools (screenshot, click, type, hotkey, scroll, drag)
- VisionAnalyzer.analyze_desktop() (Phase 2A)
- Gatekeeper classification (screenshot=GREEN, actions=YELLOW)
- Clipboard paste for typing
- Multi-monitor screenshots
- Normal PGE loop for non-CU requests
- Sequential CU tool execution (max_parallel=1 in initial plan)

## What Gets REMOVED (Phase 1 Workarounds)

- `planner.py`: `_should_force_cu_plan()`, `_build_cu_plan()`, CU override block
- `executor.py`: `_cu_wait_and_focus()`, `_cu_ensure_focus()`, their call sites
- `gateway.py`: `_CU_DONE` REPLAN block, verification screenshot logic

## Expected Outcome

After implementation:
1. Calculator test: exec→screenshot→click→type→verify→DONE in ~30s (zero model swaps)
2. Multi-step tasks work: the agent reacts to screen state after every action
3. Clean abort: timeout, max iterations, stuck detection, user cancel all work
4. Text extraction: agent can read content from screenshots for Phase 2C
5. Progress visible in UI: "Schritt 3/30: Auf Rechner geklickt..."
6. No more Phase 1 workarounds — clean, professional architecture
