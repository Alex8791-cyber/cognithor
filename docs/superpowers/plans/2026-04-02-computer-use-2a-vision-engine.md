# Computer Use Phase 2A: Vision Engine — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire qwen3-vl:32b vision model into computer_screenshot so it returns structured UI elements with pixel coordinates, enabling vision-based clicking and window focusing.

**Architecture:** Add `analyze_desktop()` to the existing VisionAnalyzer with a desktop-specific prompt that asks for JSON with element coordinates. Replace the broken `analyze_image_b64()` call in computer_use.py. Replace PowerShell window focusing with vision-based screenshot→click. Update planner prompt for iterative screenshot→click→type flow.

**Tech Stack:** Python 3.13, pytest (asyncio_mode=auto), AsyncMock, json, re, pyautogui, mss, Pillow

**Spec:** `docs/superpowers/specs/2026-04-02-computer-use-2a-vision-engine-design.md`

---

## File Structure

| File | Responsibility |
|---|---|
| `src/jarvis/browser/vision.py` (modify) | Add `analyze_desktop()`, `_parse_desktop_elements()`, `_validate_elements()`, desktop prompt constants |
| `src/jarvis/mcp/computer_use.py` (modify) | Wire `analyze_desktop()` into `computer_screenshot`, return elements, add `task_context` param |
| `src/jarvis/core/executor.py` (modify) | Replace PowerShell window focus with vision-based screenshot→click |
| `src/jarvis/core/planner.py` (modify) | Updated CU instructions and few-shot example with coordinates |
| `tests/test_browser/test_vision.py` (modify) | Add desktop vision tests |
| `tests/unit/test_computer_use_vision.py` (create) | Add computer_screenshot + executor vision focus tests |

---

### Task 1: Desktop Element Parser + Validator

**Files:**
- Modify: `src/jarvis/browser/vision.py` (after line 76, before class VisionAnalyzer)
- Test: `tests/test_browser/test_vision.py`

- [ ] **Step 1: Write failing tests for `_parse_desktop_elements` and `_validate_elements`**

Add at the end of `tests/test_browser/test_vision.py`:

```python
import json

from jarvis.browser.vision import _parse_desktop_elements, _validate_elements


class TestParseDesktopElements:
    """Tests for _parse_desktop_elements — 4-tier JSON fallback parsing."""

    def test_valid_json(self):
        raw = json.dumps({"elements": [
            {"name": "Rechner", "type": "window", "x": 200, "y": 300,
             "w": 400, "h": 500, "text": "459", "clickable": True}
        ]})
        elements = _parse_desktop_elements(raw)
        assert len(elements) == 1
        assert elements[0]["name"] == "Rechner"
        assert elements[0]["x"] == 200
        assert elements[0]["clickable"] is True

    def test_json_in_markdown_block(self):
        raw = "Hier ist meine Analyse:\n```json\n" + json.dumps({
            "elements": [{"name": "Button", "type": "button", "x": 50, "y": 60}]
        }) + "\n```"
        elements = _parse_desktop_elements(raw)
        assert len(elements) == 1
        assert elements[0]["name"] == "Button"

    def test_missing_coordinates_skipped(self):
        raw = json.dumps({"elements": [
            {"name": "OK", "type": "button"},
            {"name": "Cancel", "type": "button", "x": 100, "y": 200},
        ]})
        elements = _parse_desktop_elements(raw)
        assert len(elements) == 1
        assert elements[0]["name"] == "Cancel"

    def test_garbage_returns_empty(self):
        elements = _parse_desktop_elements("This is not JSON at all.")
        assert elements == []

    def test_think_tags_with_json(self):
        raw = "<think>Let me analyze...</think>\n" + json.dumps({
            "elements": [{"name": "Start", "type": "button", "x": 24, "y": 1060}]
        })
        elements = _parse_desktop_elements(raw)
        assert len(elements) == 1
        assert elements[0]["name"] == "Start"


class TestValidateElements:
    """Tests for _validate_elements — type safety and normalization."""

    def test_int_coercion(self):
        elements = _validate_elements([
            {"name": "Test", "x": "100", "y": "200", "w": "50", "h": "30"}
        ])
        assert elements[0]["x"] == 100
        assert isinstance(elements[0]["x"], int)

    def test_non_list_returns_empty(self):
        assert _validate_elements("not a list") == []
        assert _validate_elements(None) == []

    def test_non_dict_entries_skipped(self):
        assert _validate_elements(["not a dict", 42]) == []

    def test_defaults_applied(self):
        elements = _validate_elements([{"name": "X", "x": 10, "y": 20}])
        assert elements[0]["type"] == "other"
        assert elements[0]["w"] == 0
        assert elements[0]["text"] == ""
        assert elements[0]["clickable"] is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_browser/test_vision.py::TestParseDesktopElements tests/test_browser/test_vision.py::TestValidateElements -v`
Expected: FAIL with `ImportError: cannot import name '_parse_desktop_elements'`

- [ ] **Step 3: Implement `_parse_desktop_elements`, `_validate_elements`, and prompt constants**

In `src/jarvis/browser/vision.py`, add after the `_DESCRIBE_PAGE_PROMPT` block (after line 76) and before the `class VisionAnalyzer` line (line 79):

```python
_DESKTOP_ANALYSIS_PROMPT = (
    "Analysiere diesen Desktop-Screenshot. Identifiziere ALLE sichtbaren "
    "UI-Elemente.\n\n"
    "Fuer JEDES Element liefere:\n"
    "- name: Beschreibender Name (z.B. 'Adressleiste', 'Suchfeld', 'Rechner')\n"
    "- type: window | button | textfield | menu | icon | tab | scrollbar | link | other\n"
    "- x: X-Pixel-Koordinate der Mitte des Elements\n"
    "- y: Y-Pixel-Koordinate der Mitte des Elements\n"
    "- w: Breite in Pixeln (geschaetzt)\n"
    "- h: Hoehe in Pixeln (geschaetzt)\n"
    "- text: Sichtbarer Text im Element (falls vorhanden)\n"
    "- clickable: true/false\n\n"
    "Antworte NUR mit validem JSON:\n"
    '{"elements": [{"name": "...", "type": "...", "x": 0, "y": 0, '
    '"w": 0, "h": 0, "text": "...", "clickable": true}]}'
)

_DESKTOP_CONTEXTUAL_PROMPT_SUFFIX = (
    "\n\nKontext: {context}\n"
    "Fokussiere auf Elemente die fuer diese Aufgabe relevant sind."
)


def _parse_desktop_elements(raw_response: str) -> list[dict[str, Any]]:
    """Parse structured UI elements from vision model response.

    Uses a 4-tier fallback strategy:
    1. Direct json.loads
    2. Extract ```json ... ``` markdown block
    3. Find JSON object containing "elements" in response
    4. Empty list fallback
    """
    import json
    import re

    # Tier 1: direct parse
    try:
        data = json.loads(raw_response)
        if isinstance(data, dict) and "elements" in data:
            return _validate_elements(data["elements"])
    except (json.JSONDecodeError, ValueError):
        pass

    # Tier 2: markdown code block
    md_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw_response, re.DOTALL)
    if md_match:
        try:
            data = json.loads(md_match.group(1))
            if isinstance(data, dict) and "elements" in data:
                return _validate_elements(data["elements"])
        except (json.JSONDecodeError, ValueError):
            pass

    # Tier 3: find JSON object in response
    json_match = re.search(r"\{[\s\S]*\"elements\"[\s\S]*\}", raw_response)
    if json_match:
        try:
            data = json.loads(json_match.group())
            if isinstance(data, dict) and "elements" in data:
                return _validate_elements(data["elements"])
        except (json.JSONDecodeError, ValueError):
            pass

    # Tier 4: empty list
    return []


def _validate_elements(elements: Any) -> list[dict[str, Any]]:
    """Validate and normalize element dicts from vision model."""
    if not isinstance(elements, list):
        return []

    validated = []
    for el in elements:
        if not isinstance(el, dict):
            continue
        if "name" not in el or "x" not in el or "y" not in el:
            continue
        validated.append({
            "name": str(el.get("name", "")),
            "type": str(el.get("type", "other")),
            "x": int(el.get("x", 0)),
            "y": int(el.get("y", 0)),
            "w": int(el.get("w", 0)),
            "h": int(el.get("h", 0)),
            "text": str(el.get("text", "")),
            "clickable": bool(el.get("clickable", False)),
        })

    return validated
```

Also add to the imports at top of vision.py (after `from typing import Any`):
The `Any` import is already there. No additional imports needed at module level since `json` and `re` are imported inside the function.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_browser/test_vision.py::TestParseDesktopElements tests/test_browser/test_vision.py::TestValidateElements -v`
Expected: 9 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/jarvis/browser/vision.py tests/test_browser/test_vision.py
git commit -m "feat(vision): add desktop element parser with 4-tier JSON fallback"
```

---

### Task 2: VisionAnalyzer `analyze_desktop()` Method

**Files:**
- Modify: `src/jarvis/browser/vision.py` (class VisionAnalyzer, after `describe_page`)
- Test: `tests/test_browser/test_vision.py`

- [ ] **Step 1: Write failing tests for `analyze_desktop`**

Add at the end of `tests/test_browser/test_vision.py`:

```python
class TestAnalyzeDesktop:
    """Tests for VisionAnalyzer.analyze_desktop — desktop screenshot analysis."""

    def _make_analyzer(self, llm_response: str) -> tuple[VisionAnalyzer, AsyncMock]:
        llm = AsyncMock()
        llm.chat = AsyncMock(return_value={
            "message": {"role": "assistant", "content": llm_response},
        })
        cfg = VisionConfig(enabled=True, model="qwen3-vl:32b", backend_type="ollama")
        return VisionAnalyzer(llm, cfg), llm

    @pytest.mark.asyncio
    async def test_returns_elements(self):
        response = json.dumps({"elements": [
            {"name": "Rechner", "type": "window", "x": 200, "y": 300,
             "w": 400, "h": 500, "text": "", "clickable": True},
        ]})
        v, llm = self._make_analyzer(response)
        result = await v.analyze_desktop("base64data")

        assert result.success is True
        assert len(result.elements) == 1
        assert result.elements[0]["name"] == "Rechner"
        assert result.elements[0]["x"] == 200
        llm.chat.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_disabled_returns_error(self):
        llm = AsyncMock()
        v = VisionAnalyzer(llm, VisionConfig(enabled=False))
        result = await v.analyze_desktop("base64data")

        assert result.success is False
        assert "nicht aktiviert" in result.error
        llm.chat.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_screenshot_returns_error(self):
        v, _ = self._make_analyzer("")
        result = await v.analyze_desktop("")

        assert result.success is False
        assert "Kein Screenshot" in result.error

    @pytest.mark.asyncio
    async def test_task_context_appended_to_prompt(self):
        v, llm = self._make_analyzer('{"elements": []}')
        await v.analyze_desktop("base64data", task_context="Reddit oeffnen")

        call_args = llm.chat.call_args
        messages = str(call_args)
        assert "Reddit" in messages

    @pytest.mark.asyncio
    async def test_llm_error_handled(self):
        llm = AsyncMock()
        llm.chat = AsyncMock(side_effect=RuntimeError("GPU OOM"))
        cfg = VisionConfig(enabled=True, model="qwen3-vl:32b", backend_type="ollama")
        v = VisionAnalyzer(llm, cfg)

        result = await v.analyze_desktop("base64data")
        assert result.success is False
        assert "fehlgeschlagen" in result.error

    @pytest.mark.asyncio
    async def test_non_json_response_returns_description_no_elements(self):
        v, _ = self._make_analyzer("I see a calculator and a browser window.")
        result = await v.analyze_desktop("base64data")

        assert result.success is True
        assert "calculator" in result.description
        assert result.elements == []

    @pytest.mark.asyncio
    async def test_custom_prompt_overrides_default(self):
        v, llm = self._make_analyzer('{"elements": []}')
        await v.analyze_desktop("base64data", prompt="Custom prompt here")

        messages = str(llm.chat.call_args)
        assert "Custom prompt" in messages

    @pytest.mark.asyncio
    async def test_stats_updated(self):
        v, _ = self._make_analyzer('{"elements": []}')
        assert v.stats()["calls"] == 0

        await v.analyze_desktop("base64data")
        assert v.stats()["calls"] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_browser/test_vision.py::TestAnalyzeDesktop -v`
Expected: FAIL with `AttributeError: 'VisionAnalyzer' has no attribute 'analyze_desktop'`

- [ ] **Step 3: Implement `analyze_desktop` on VisionAnalyzer**

In `src/jarvis/browser/vision.py`, add after the `describe_page` method (after line 170) and before the `stats` method (line 172):

```python
    async def analyze_desktop(
        self,
        screenshot_b64: str,
        prompt: str = "",
        task_context: str = "",
    ) -> VisionAnalysisResult:
        """Analyze a desktop screenshot and identify UI elements with coordinates.

        Unlike analyze_screenshot (browser-focused), this method is optimized
        for desktop environments: pixel coordinates instead of CSS selectors,
        window detection, taskbar elements, etc.

        Args:
            screenshot_b64: Base64-encoded screenshot (PNG).
            prompt: Optional custom prompt (default: desktop element detection).
            task_context: Optional task description to focus the analysis.

        Returns:
            VisionAnalysisResult with description and elements list.
        """
        if not self.is_enabled:
            return VisionAnalysisResult(error="Vision nicht aktiviert")

        if not screenshot_b64:
            return VisionAnalysisResult(error="Kein Screenshot-Daten")

        effective_prompt = prompt or _DESKTOP_ANALYSIS_PROMPT
        if task_context and not prompt:
            effective_prompt += _DESKTOP_CONTEXTUAL_PROMPT_SUFFIX.format(
                context=task_context
            )

        result = await self._send_vision_request(screenshot_b64, effective_prompt)

        if result.success and result.description:
            result.elements = _parse_desktop_elements(result.description)

        return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_browser/test_vision.py::TestAnalyzeDesktop -v`
Expected: 8 PASSED

Run all vision tests:
Run: `python -m pytest tests/test_browser/test_vision.py -v`
Expected: All PASSED

- [ ] **Step 5: Commit**

```bash
git add src/jarvis/browser/vision.py tests/test_browser/test_vision.py
git commit -m "feat(vision): add analyze_desktop() for desktop screenshot analysis with structured coordinates"
```

---

### Task 3: Wire Vision into `computer_screenshot`

**Files:**
- Modify: `src/jarvis/mcp/computer_use.py` (lines 78-128, computer_screenshot method)
- Test: `tests/unit/test_computer_use_vision.py` (create)

- [ ] **Step 1: Write failing tests**

Create new file `tests/unit/test_computer_use_vision.py`:

```python
"""Tests for computer_screenshot with VisionAnalyzer integration."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

from jarvis.browser.vision import VisionAnalysisResult
from jarvis.mcp.computer_use import ComputerUseTools


class TestComputerScreenshotWithVision:
    @pytest.mark.asyncio
    async def test_elements_in_result(self):
        mock_vision = AsyncMock()
        mock_vision.analyze_desktop = AsyncMock(return_value=VisionAnalysisResult(
            success=True,
            description="Desktop mit Rechner",
            elements=[{"name": "Rechner", "type": "window", "x": 200, "y": 300,
                        "w": 400, "h": 500, "text": "", "clickable": True}],
        ))

        tools = ComputerUseTools(vision_analyzer=mock_vision)

        with patch("jarvis.mcp.computer_use._take_screenshot_b64",
                    return_value=("base64", 1920, 1080)):
            result = await tools.computer_screenshot()

        assert result["success"] is True
        assert len(result["elements"]) == 1
        assert result["elements"][0]["name"] == "Rechner"
        assert "Rechner" in result["description"]
        mock_vision.analyze_desktop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_vision_returns_empty_elements(self):
        tools = ComputerUseTools(vision_analyzer=None)

        with patch("jarvis.mcp.computer_use._take_screenshot_b64",
                    return_value=("base64", 1920, 1080)):
            result = await tools.computer_screenshot()

        assert result["success"] is True
        assert result["elements"] == []
        assert "No vision" in result["description"]

    @pytest.mark.asyncio
    async def test_vision_error_returns_empty_elements(self):
        mock_vision = AsyncMock()
        mock_vision.analyze_desktop = AsyncMock(return_value=VisionAnalysisResult(
            success=False, error="GPU timeout",
        ))

        tools = ComputerUseTools(vision_analyzer=mock_vision)

        with patch("jarvis.mcp.computer_use._take_screenshot_b64",
                    return_value=("base64", 1920, 1080)):
            result = await tools.computer_screenshot()

        assert result["success"] is True
        assert result["elements"] == []
        assert "GPU timeout" in result["description"]

    @pytest.mark.asyncio
    async def test_task_context_passed_through(self):
        mock_vision = AsyncMock()
        mock_vision.analyze_desktop = AsyncMock(return_value=VisionAnalysisResult(
            success=True, description="OK", elements=[],
        ))

        tools = ComputerUseTools(vision_analyzer=mock_vision)

        with patch("jarvis.mcp.computer_use._take_screenshot_b64",
                    return_value=("base64", 1920, 1080)):
            await tools.computer_screenshot(task_context="Reddit oeffnen")

        call_args = mock_vision.analyze_desktop.call_args
        assert "Reddit" in str(call_args)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_computer_use_vision.py -v`
Expected: FAIL (no `task_context` parameter, old hasattr logic, no `elements` key)

- [ ] **Step 3: Modify `computer_screenshot` in computer_use.py**

**3a.** Update method signature (line 78):

```python
async def computer_screenshot(self, monitor: int = 0, task_context: str = "") -> dict[str, Any]:
```

**3b.** Replace the entire vision block (lines 89-122) with:

```python
            elements = []
            if self._vision:
                try:
                    result = await self._vision.analyze_desktop(
                        b64, task_context=task_context
                    )
                    description = result.description if result.success else (
                        f"Screenshot taken ({width}x{height}). "
                        f"Vision analysis failed: {result.error}"
                    )
                    elements = result.elements
                    if elements:
                        log.info(
                            "desktop_vision_elements",
                            count=len(elements),
                            names=[e["name"] for e in elements[:5]],
                        )
                except Exception as exc:
                    description = (
                        f"Screenshot taken ({width}x{height}). Vision error: {exc}"
                    )
            else:
                description = (
                    f"Screenshot taken ({width}x{height}). "
                    "No vision analyzer — use coordinates from previous analysis."
                )
```

**3c.** Update the return dict to include elements:

```python
            return {
                "success": True,
                "width": width,
                "height": height,
                "description": description,
                "elements": elements,
            }
```

**3d.** Update the registration input_schema for `computer_screenshot` (in `register_computer_use_tools`):

```python
        input_schema={
            "type": "object",
            "properties": {
                "monitor": {
                    "type": "integer",
                    "description": "0=all monitors (default), 1=primary, 2+=specific monitor",
                },
                "task_context": {
                    "type": "string",
                    "description": "What the user wants to do (helps focus element detection)",
                },
            },
        },
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_computer_use_vision.py -v`
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/jarvis/mcp/computer_use.py tests/unit/test_computer_use_vision.py
git commit -m "feat(computer_use): wire analyze_desktop into computer_screenshot, return structured elements"
```

---

### Task 4: Vision-Based Window Focus in Executor

**Files:**
- Modify: `src/jarvis/core/executor.py` (lines 327-358)

- [ ] **Step 1: Replace PowerShell focus with vision-based screenshot→click**

In `src/jarvis/core/executor.py`, replace the entire block from line 327 (`# After launching a GUI app...`) through line 358 (`pass  # Best effort`) with:

```python
                # After launching a GUI app, wait then focus via vision
                if action.tool == "exec_command" and result.success and _has_computer_use:
                    await asyncio.sleep(2.0)
                    # Vision-based focusing: screenshot → find window → click
                    try:
                        _ss_handler = self._mcp_client._builtin_handlers.get(
                            "computer_screenshot"
                        )
                        _click_handler = self._mcp_client._builtin_handlers.get(
                            "computer_click"
                        )
                        if _ss_handler and _click_handler:
                            _ss = await _ss_handler()
                            _elements = _ss.get("elements", [])
                            _windows = [
                                e
                                for e in _elements
                                if e.get("type") == "window" and e.get("clickable", True)
                            ]
                            if _windows:
                                _target = _windows[0]
                                await _click_handler(x=_target["x"], y=_target["y"])
                                await asyncio.sleep(0.3)
                                log.info(
                                    "vision_focus_window",
                                    name=_target.get("name", "?"),
                                    x=_target["x"],
                                    y=_target["y"],
                                )
                            else:
                                log.debug(
                                    "vision_focus_no_windows_found",
                                    elements=len(_elements),
                                )
                    except Exception:
                        log.debug("vision_focus_failed", exc_info=True)
```

- [ ] **Step 2: Run existing executor tests to verify no regression**

Run: `python -m pytest tests/test_core/ -x -q -k "executor" 2>&1 | tail -5`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add src/jarvis/core/executor.py
git commit -m "feat(executor): replace PowerShell window focus with vision-based screenshot→click"
```

---

### Task 5: Update Planner Prompt

**Files:**
- Modify: `src/jarvis/core/planner.py` (lines 100-150)

- [ ] **Step 1: Replace CU instructions block**

In `src/jarvis/core/planner.py`, replace the COMPUTER USE block (lines 100-111) with:

```python
**COMPUTER USE (Desktop-Automation):** Du KANNST den Desktop des Nutzers \
steuern! Wenn der Nutzer dich bittet ein Programm zu oeffnen und darin zu \
tippen/klicken, nutze die computer_* Tools. Du hast: \
computer_screenshot (zeigt dir den Bildschirm mit UI-Elementen und Koordinaten), \
computer_click (klickt auf x,y Koordinaten aus dem Screenshot), \
computer_type (tippt Text via Clipboard-Paste), \
computer_hotkey (drueckt Tastenkombinationen wie Enter, Alt+Tab). \
ABLAUF: Schritt 1: exec_command zum Programm-Start. Schritt 2: \
computer_screenshot -- du erhaeltst eine Liste von UI-Elementen mit \
Pixel-Koordinaten (x, y). Schritt 3: computer_click mit den Koordinaten \
eines Elements aus dem Screenshot. Schritt 4: computer_type zum Tippen. \
WICHTIG: Nutze IMMER die Koordinaten aus dem Screenshot-Ergebnis fuer Clicks. \
Klicke IMMER auf ein Element BEVOR du computer_type verwendest. \
Jeder Schritt ist ein EIGENER Step im JSON-Plan. SAGE NIEMALS \
"ich kann keine GUI steuern" -- du KANNST es.
```

- [ ] **Step 2: Replace Few-Shot example**

Replace the calculator example (lines 139-150) with:

```python
Beispiel -- "Oeffne den Taschenrechner und tippe 1+4=":
```json
{{"goal": "Taschenrechner oeffnen und Rechnung eintippen", \
"reasoning": "Desktop-Automation: oeffnen, Screenshot fuer Koordinaten, klicken, tippen.", \
"steps": [\
{{"tool": "exec_command", "params": {{"command": "start calc.exe"}}, \
"rationale": "Taschenrechner starten"}}, \
{{"tool": "computer_screenshot", "params": {{}}, \
"rationale": "Bildschirm ansehen, UI-Elemente mit Koordinaten erhalten"}}, \
{{"tool": "computer_click", "params": {{"x": 0, "y": 0}}, \
"rationale": "Auf Taschenrechner klicken (Koordinaten aus Screenshot)"}}, \
{{"tool": "computer_type", "params": {{"text": "1+4="}}, \
"rationale": "Rechnung eintippen"}}], "confidence": 0.85}}
```
```

- [ ] **Step 3: Commit**

```bash
git add src/jarvis/core/planner.py
git commit -m "feat(planner): update CU prompt for vision-based coordinate clicking"
```

---

### Task 6: Gatekeeper Classification Test

**Files:**
- Test: `tests/unit/test_computer_use_vision.py` (append)

- [ ] **Step 1: Add gatekeeper classification test**

Append to `tests/unit/test_computer_use_vision.py`:

```python
class TestGatekeeperCUClassification:
    """Verify security classification hasn't regressed."""

    def test_screenshot_green_actions_yellow(self):
        from jarvis.core.gatekeeper import Gatekeeper
        from jarvis.config import JarvisConfig, ToolsConfig
        from jarvis.models import PlannedAction

        config = JarvisConfig(tools=ToolsConfig(computer_use_enabled=True))
        gk = Gatekeeper(config)

        action = PlannedAction(tool="computer_screenshot", params={}, rationale="test")
        assert gk._classify_risk(action).name == "GREEN"

        for tool in ["computer_click", "computer_type", "computer_hotkey"]:
            action = PlannedAction(tool=tool, params={}, rationale="test")
            assert gk._classify_risk(action).name == "YELLOW"
```

- [ ] **Step 2: Run all new tests**

Run: `python -m pytest tests/unit/test_computer_use_vision.py tests/test_browser/test_vision.py -v`
Expected: All PASSED (9 + 8 + 4 + 1 new = 22 new tests + existing)

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_computer_use_vision.py
git commit -m "test(computer_use): add gatekeeper classification verification"
```

---

### Task 7: Integration Verification

**Files:**
- All modified files from Tasks 1-6

- [ ] **Step 1: Run targeted tests**

Run: `python -m pytest tests/test_browser/test_vision.py tests/unit/test_computer_use_vision.py tests/test_session_management/test_computer_use.py -v`
Expected: All PASS

- [ ] **Step 2: Run broader test sweep**

Run: `python -m pytest tests/ -x -q 2>&1 | tail -10`
Expected: No new failures

- [ ] **Step 3: Verify vision wiring end-to-end**

Check these connections:
1. `gateway/phases/tools.py:191` — `VisionAnalyzer` is created with `qwen3-vl:32b`
2. `gateway/phases/tools.py:490` — same `vision_analyzer` instance passed to `register_computer_use_tools`
3. `mcp/computer_use.py` — `ComputerUseTools.__init__` stores `self._vision = vision_analyzer`
4. `mcp/computer_use.py:computer_screenshot` — calls `self._vision.analyze_desktop(b64)`
5. `browser/vision.py:analyze_desktop` — calls `self._send_vision_request()` → `self._llm.chat()`
6. `core/executor.py` — after `exec_command`, calls `computer_screenshot` handler → gets `elements` → clicks on window

- [ ] **Step 4: Format and final commit**

```bash
python -m ruff format src/jarvis/browser/vision.py src/jarvis/mcp/computer_use.py src/jarvis/core/executor.py src/jarvis/core/planner.py tests/unit/test_computer_use_vision.py tests/test_browser/test_vision.py
python -m ruff check src/ tests/ --select=F821,F811 --no-fix
git add -A
git commit -m "test(vision): verify computer use phase 2A integration"
```
