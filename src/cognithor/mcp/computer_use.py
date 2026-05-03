"""Computer Use — GPT-5.4-style desktop automation via screenshots + coordinate clicking.

Takes screenshots of the desktop, sends them to a vision model for analysis,
and executes mouse/keyboard actions at specific coordinates. Enables Cognithor
to interact with any application through its visual interface.

Requires: pyautogui, mss (for fast screenshots), Pillow
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import io
import time
from typing import Any

from cognithor.utils.logging import get_logger

log = get_logger(__name__)

# Lazy imports to avoid startup cost
_pyautogui = None


def _get_pyautogui() -> Any:
    global _pyautogui
    if _pyautogui is None:
        import pyautogui

        pyautogui.FAILSAFE = True  # Move mouse to corner to abort
        pyautogui.PAUSE = 0.15  # Small pause between actions
        _pyautogui = pyautogui
    return _pyautogui


def _take_screenshot_b64(monitor_index: int = 0) -> tuple[str, int, int, float]:
    """Take a desktop screenshot, return (base64_png, width, height, scale_factor).

    Args:
        monitor_index: 0 = all monitors combined, 1 = primary, 2+ = specific monitor.
    """
    try:
        import mss
    except ImportError as err:
        raise ImportError("mss not installed. Run: pip install cognithor[desktop]") from err
    try:
        from PIL import Image
    except ImportError as err:
        raise ImportError("Pillow not installed. Run: pip install cognithor[desktop]") from err

    with mss.mss() as sct:
        # Index 0 = combined virtual screen (all monitors), 1+ = specific monitor
        if monitor_index >= len(sct.monitors):
            monitor_index = 0  # Fallback to all
        monitor = sct.monitors[monitor_index]
        img = sct.grab(monitor)

        # mss returns BGRA bytes — convert properly
        pil_img = Image.frombytes("RGB", img.size, img.bgra, "raw", "BGRX")

        # Resize for vision model (max 2560px wide to support 4K)
        max_w = 2560
        scale_factor = 1.0
        if pil_img.width > max_w:
            scale_factor = max_w / pil_img.width
            pil_img = pil_img.resize(
                (max_w, int(pil_img.height * scale_factor)), Image.Resampling.LANCZOS
            )

        buf = io.BytesIO()
        pil_img.save(buf, format="PNG", optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return b64, pil_img.width, pil_img.height, scale_factor


class ComputerUseTools:
    """MCP tools for desktop computer use via vision + coordinates."""

    def __init__(self, vision_analyzer: Any = None, uia_provider: Any = None) -> None:
        self._vision = vision_analyzer
        self._uia_provider = uia_provider
        self._last_scale_factor: float = 1.0
        self._last_screenshot_hash: str = ""
        self._last_elements: list[dict[str, Any]] = []

    async def _wait_for_stable_screen(
        self,
        min_delay_ms: int = 300,
        poll_interval_ms: int = 300,
        stability_threshold: int = 2,
        timeout_ms: int = 5000,
    ) -> None:
        """Wait until screen content stabilizes after an action."""
        await asyncio.sleep(min_delay_ms / 1000.0)

        start = time.monotonic()
        last_hash = ""
        stable_count = 0

        while (time.monotonic() - start) * 1000 < timeout_ms:
            try:
                loop = asyncio.get_running_loop()
                b64, _, _, _ = await loop.run_in_executor(None, _take_screenshot_b64)
                current_hash = hashlib.md5(b64.encode()).hexdigest()

                if current_hash == last_hash:
                    stable_count += 1
                    if stable_count >= stability_threshold:
                        return
                else:
                    stable_count = 0
                    last_hash = current_hash
            except Exception:
                return  # Screenshot failed — don't block

            await asyncio.sleep(poll_interval_ms / 1000.0)

    async def computer_screenshot(self, monitor: int = 0, task_context: str = "") -> dict[str, Any]:
        """Take a screenshot and describe what's visible.

        Args:
            monitor: 0 = all monitors combined (default), 1 = primary, 2+ = specific.
            task_context: What the user wants to do (helps focus element detection).
        """
        try:
            loop = asyncio.get_running_loop()
            b64, width, height, scale_factor = await loop.run_in_executor(
                None, lambda: _take_screenshot_b64(monitor_index=int(monitor))
            )
            self._last_scale_factor = scale_factor

            # Try UIA first for exact element coordinates
            uia_elements: list[dict[str, Any]] = []
            if self._uia_provider:
                with contextlib.suppress(Exception):
                    uia_elements = await loop.run_in_executor(
                        None, self._uia_provider.get_focused_window_elements
                    )

            if uia_elements:
                elements = uia_elements
                # Vision still provides description (scene context)
                if self._vision:
                    try:
                        result = await self._vision.analyze_desktop(b64, task_context=task_context)
                        description = (
                            result.description
                            if result.success
                            else f"Screenshot taken ({width}x{height})."
                        )
                    except Exception as exc:
                        description = f"Screenshot taken ({width}x{height}). Vision error: {exc}"
                else:
                    description = f"Screenshot taken ({width}x{height})."
                log.info(
                    "desktop_uia_elements",
                    count=len(elements),
                    names=[e["name"] for e in elements[:5]],
                )
            elif self._vision:
                # Fallback: vision provides both description AND elements
                try:
                    result = await self._vision.analyze_desktop(b64, task_context=task_context)
                    description = (
                        result.description
                        if result.success
                        else (
                            f"Screenshot taken ({width}x{height}). "
                            f"Vision analysis failed: {result.error}"
                        )
                    )
                    elements = result.elements
                    if elements:
                        log.info(
                            "desktop_vision_elements",
                            count=len(elements),
                            names=[e["name"] for e in elements[:5]],
                        )
                except Exception as exc:
                    description = f"Screenshot taken ({width}x{height}). Vision error: {exc}"
                    elements = []
            else:
                description = (
                    f"Screenshot taken ({width}x{height}). No vision analyzer or UIA available."
                )
                elements = []

            # Cache elements + screenshot hash for click_element and change detection
            self._last_elements = elements
            self._last_screenshot_hash = hashlib.md5(b64.encode()).hexdigest()

            return {
                "success": True,
                "width": width,
                "height": height,
                "description": description,
                "elements": elements,
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    async def computer_click_element(
        self,
        description: str,
        button: str = "left",
        clicks: int = 1,
    ) -> dict[str, Any]:
        """Click a UI element by description instead of coordinates.

        Takes a fresh screenshot, uses vision to find the element matching
        the description, and clicks its center coordinates.

        Args:
            description: Natural language description of the element to click
                (e.g., "Login button", "the red X in the top right").
            button: Mouse button (left, right, middle).
            clicks: Number of clicks (1=single, 2=double).
        """
        try:
            # Take fresh screenshot
            screenshot_result = await self.computer_screenshot(task_context=description)
            if not screenshot_result.get("success"):
                return screenshot_result

            elements = screenshot_result.get("elements", [])
            if not elements:
                return {
                    "success": False,
                    "error": f"No UI elements detected. Cannot find: '{description}'",
                }

            # Find best matching element
            desc_lower = description.lower()
            best_match = None
            best_score = 0

            for elem in elements:
                score = 0
                elem_name = (elem.get("name") or "").lower()
                elem_text = (elem.get("text") or "").lower()
                elem_type = (elem.get("type") or "").lower()

                # Exact name match
                if desc_lower == elem_name:
                    score = 100
                # Name contains description
                elif desc_lower in elem_name:
                    score = 80
                # Description contains name
                elif elem_name and elem_name in desc_lower:
                    score = 70
                # Text match
                elif desc_lower in elem_text:
                    score = 60
                elif elem_text and elem_text in desc_lower:
                    score = 50
                # Type match
                elif desc_lower in elem_type:
                    score = 30

                # Boost clickable elements
                if elem.get("clickable"):
                    score += 10

                if score > best_score:
                    best_score = score
                    best_match = elem

            if not best_match or best_score < 30:
                # List available elements for the agent
                available = [
                    f"- {e.get('name', '?')} ({e.get('type', '?')}) at ({e.get('x')},{e.get('y')})"
                    for e in elements[:15]
                ]
                return {
                    "success": False,
                    "error": f"No element matching '{description}' found.\n"
                    f"Available elements:\n" + "\n".join(available),
                }

            x = best_match.get("x", 0)
            y = best_match.get("y", 0)

            # Click the found element
            click_result = await self.computer_click(x, y, button=button, clicks=clicks)
            click_result["matched_element"] = best_match.get("name", "")
            click_result["match_score"] = best_score
            return click_result

        except Exception as exc:
            return {"success": False, "error": str(exc)}

    async def computer_wait_for_change(
        self,
        timeout_ms: int = 5000,
        description: str = "",
    ) -> dict[str, Any]:
        """Wait until the screen changes after an action.

        Compares the current screen against the last screenshot hash.
        Useful after clicking a button to confirm the UI responded.

        Args:
            timeout_ms: Max time to wait for a change (default: 5000ms).
            description: What change to expect (for better analysis).
        """
        if not self._last_screenshot_hash:
            return {"changed": True, "detail": "No previous screenshot to compare"}

        start = time.monotonic()
        prev_hash = self._last_screenshot_hash

        while (time.monotonic() - start) * 1000 < timeout_ms:
            try:
                loop = asyncio.get_running_loop()
                b64, _, _, _ = await loop.run_in_executor(None, _take_screenshot_b64)
                current_hash = hashlib.md5(b64.encode()).hexdigest()

                if current_hash != prev_hash:
                    self._last_screenshot_hash = current_hash
                    detail = "Screen changed"
                    # If vision available, describe what changed
                    if self._vision and description:
                        try:
                            result = await self._vision.analyze_desktop(
                                b64, task_context=f"Detect changes: {description}"
                            )
                            if result.success:
                                detail = result.description
                        except Exception:
                            log.debug("computer_use_vision_change_detect_failed", exc_info=True)

                    return {
                        "changed": True,
                        "elapsed_ms": int((time.monotonic() - start) * 1000),
                        "detail": detail,
                    }
            except Exception:
                log.debug("computer_use_screenshot_compare_failed", exc_info=True)

            await asyncio.sleep(0.3)

        return {
            "changed": False,
            "elapsed_ms": timeout_ms,
            "detail": f"No screen change detected within {timeout_ms}ms",
        }

    async def computer_click(
        self,
        x: int,
        y: int,
        button: str = "left",
        clicks: int = 1,
    ) -> dict[str, Any]:
        """Click at specific coordinates on the desktop."""
        try:
            # Scale coordinates back to actual screen pixels
            if self._last_scale_factor != 1.0 and self._last_scale_factor > 0:
                x = int(int(x) / self._last_scale_factor)
                y = int(int(y) / self._last_scale_factor)
            gui = _get_pyautogui()
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None, lambda: gui.click(x=int(x), y=int(y), button=button, clicks=int(clicks))
            )
            log.info("computer_click", x=x, y=y, button=button, clicks=clicks)
            return {
                "success": True,
                "action": "click",
                "x": x,
                "y": y,
                "button": button,
                "clicks": clicks,
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    async def computer_type(
        self,
        text: str,
        interval: float = 0.03,
    ) -> dict[str, Any]:
        """Type text via clipboard paste (Ctrl+V).

        Always uses clipboard — typewrite() sends raw key codes which fail
        for special chars (*, +, =) on non-US keyboards and can trigger
        shortcuts in other applications (e.g. YouTube rewind).
        """
        try:
            gui = _get_pyautogui()
            loop = asyncio.get_running_loop()
            import pyperclip

            # Brief wait to ensure target window has focus
            await asyncio.sleep(0.3)

            # Clipboard paste — works with ALL characters and keyboard layouts
            await loop.run_in_executor(None, lambda: pyperclip.copy(text))
            await asyncio.sleep(0.1)
            await loop.run_in_executor(None, lambda: gui.hotkey("ctrl", "v"))
            await asyncio.sleep(0.2)

            log.info("computer_type", text_len=len(text), method="clipboard")
            return {"success": True, "action": "type", "text_length": len(text)}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    async def computer_hotkey(self, keys: str = "", **kwargs: Any) -> dict[str, Any]:
        """Press a keyboard shortcut (e.g., ctrl+c, alt+tab, enter)."""
        try:
            gui = _get_pyautogui()
            loop = asyncio.get_running_loop()

            # Split string like "ctrl+c" into ["ctrl", "c"]
            key_list = [k.strip() for k in keys.split("+") if k.strip()]
            if not key_list:
                return {"success": False, "error": "No keys provided"}

            await loop.run_in_executor(None, lambda: gui.hotkey(*key_list))
            log.info("computer_hotkey", keys=key_list)
            return {"success": True, "action": "hotkey", "keys": key_list}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    async def computer_scroll(
        self,
        x: int = 0,
        y: int = 0,
        direction: str = "down",
        amount: int = 3,
    ) -> dict[str, Any]:
        """Scroll at specific coordinates."""
        try:
            if self._last_scale_factor != 1.0 and self._last_scale_factor > 0:
                x = int(int(x) / self._last_scale_factor)
                y = int(int(y) / self._last_scale_factor)
            gui = _get_pyautogui()
            loop = asyncio.get_running_loop()
            scroll_clicks = -int(amount) if direction == "down" else int(amount)
            await loop.run_in_executor(None, lambda: gui.scroll(scroll_clicks, x=int(x), y=int(y)))
            log.info("computer_scroll", x=x, y=y, direction=direction, amount=amount)
            return {
                "success": True,
                "action": "scroll",
                "x": x,
                "y": y,
                "direction": direction,
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    async def computer_drag(
        self,
        start_x: int = 0,
        start_y: int = 0,
        end_x: int = 0,
        end_y: int = 0,
    ) -> dict[str, Any]:
        """Drag from start coordinates to end coordinates."""
        try:
            if self._last_scale_factor != 1.0 and self._last_scale_factor > 0:
                start_x = int(int(start_x) / self._last_scale_factor)
                start_y = int(int(start_y) / self._last_scale_factor)
                end_x = int(int(end_x) / self._last_scale_factor)
                end_y = int(int(end_y) / self._last_scale_factor)
            gui = _get_pyautogui()
            loop = asyncio.get_running_loop()

            def _do_drag() -> None:
                gui.moveTo(int(start_x), int(start_y), duration=0.2)
                gui.drag(
                    int(end_x) - int(start_x),
                    int(end_y) - int(start_y),
                    duration=0.5,
                    button="left",
                )

            await loop.run_in_executor(None, _do_drag)
            log.info(
                "computer_drag",
                start_x=start_x,
                start_y=start_y,
                end_x=end_x,
                end_y=end_y,
            )
            return {
                "success": True,
                "action": "drag",
                "start": [start_x, start_y],
                "end": [end_x, end_y],
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}


def register_computer_use_tools(
    client: Any,
    vision_analyzer: Any = None,
    uia_provider: Any = None,
) -> ComputerUseTools | None:
    """Register computer use MCP tools."""
    try:
        _get_pyautogui()  # Test if pyautogui works
        import mss  # noqa: F401
        from PIL import Image  # noqa: F401
    except ImportError as exc:
        log.info(
            "computer_use_skip",
            reason=f"Missing dependency: {exc}. Install with: pip install cognithor[desktop]",
        )
        return None
    except Exception as exc:
        log.debug("computer_use_skip", reason=str(exc))
        return None

    tools = ComputerUseTools(vision_analyzer=vision_analyzer, uia_provider=uia_provider)

    client.register_builtin_handler(
        "computer_screenshot",
        tools.computer_screenshot,
        description=(
            "Take a screenshot of the desktop and describe visible UI elements "
            "with their approximate pixel coordinates. "
            "Use this to see what's on screen before clicking."
        ),
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
    )

    client.register_builtin_handler(
        "computer_click",
        tools.computer_click,
        description=(
            "Click at specific pixel coordinates on the desktop. "
            "Use computer_screenshot first to identify element positions."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "x": {"type": "integer", "description": "X pixel coordinate"},
                "y": {"type": "integer", "description": "Y pixel coordinate"},
                "button": {
                    "type": "string",
                    "enum": ["left", "right", "middle"],
                    "description": "Mouse button (default: left)",
                },
                "clicks": {
                    "type": "integer",
                    "description": "Number of clicks (1=single, 2=double)",
                },
            },
            "required": ["x", "y"],
        },
    )

    client.register_builtin_handler(
        "computer_type",
        tools.computer_type,
        description=(
            "Type text using the keyboard. Supports Unicode (umlauts, etc.). "
            "Click a text field first with computer_click."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to type"},
                "interval": {
                    "type": "number",
                    "description": "Delay between keystrokes in seconds (default: 0.02)",
                },
            },
            "required": ["text"],
        },
    )

    client.register_builtin_handler(
        "computer_hotkey",
        tools.computer_hotkey,
        description=(
            "Press a keyboard shortcut. Pass keys separated by +. "
            "Examples: 'ctrl+c', 'ctrl+v', 'alt+tab', 'enter', 'escape', 'ctrl+shift+s'."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "keys": {
                    "type": "string",
                    "description": "Keys separated by + (e.g., 'ctrl+c', 'alt+tab', 'enter')",
                },
            },
            "required": ["keys"],
        },
    )

    client.register_builtin_handler(
        "computer_scroll",
        tools.computer_scroll,
        description="Scroll at specific coordinates on the desktop.",
        input_schema={
            "type": "object",
            "properties": {
                "x": {"type": "integer", "description": "X coordinate"},
                "y": {"type": "integer", "description": "Y coordinate"},
                "direction": {
                    "type": "string",
                    "enum": ["up", "down"],
                    "description": "Scroll direction",
                },
                "amount": {
                    "type": "integer",
                    "description": "Scroll amount (default: 3)",
                },
            },
            "required": ["x", "y"],
        },
    )

    client.register_builtin_handler(
        "computer_drag",
        tools.computer_drag,
        description="Drag from one position to another (e.g., for drag-and-drop).",
        input_schema={
            "type": "object",
            "properties": {
                "start_x": {"type": "integer"},
                "start_y": {"type": "integer"},
                "end_x": {"type": "integer"},
                "end_y": {"type": "integer"},
            },
            "required": ["start_x", "start_y", "end_x", "end_y"],
        },
    )

    client.register_builtin_handler(
        "computer_click_element",
        tools.computer_click_element,
        description=(
            "Click a UI element by description instead of coordinates. "
            "Takes a screenshot, finds the matching element via vision, and clicks it. "
            "Example: 'Login button', 'the search field', 'Close icon'."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Natural language description of the element to click",
                },
                "button": {
                    "type": "string",
                    "enum": ["left", "right", "middle"],
                    "description": "Mouse button (default: left)",
                },
                "clicks": {
                    "type": "integer",
                    "description": "Number of clicks (1=single, 2=double)",
                },
            },
            "required": ["description"],
        },
        risk_level="yellow",
    )

    client.register_builtin_handler(
        "computer_wait_for_change",
        tools.computer_wait_for_change,
        description=(
            "Wait until the screen changes after an action. "
            "Confirms that a click/type/hotkey had a visible effect. "
            "Returns whether the screen changed and optionally what changed."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "timeout_ms": {
                    "type": "integer",
                    "description": "Max wait time in milliseconds (default: 5000)",
                },
                "description": {
                    "type": "string",
                    "description": "What change to expect (e.g., 'dialog should appear')",
                },
            },
        },
        risk_level="green",
    )

    log.info(
        "computer_use_tools_registered",
        tools=[
            "computer_screenshot",
            "computer_click",
            "computer_click_element",
            "computer_type",
            "computer_hotkey",
            "computer_scroll",
            "computer_drag",
            "computer_wait_for_change",
        ],
    )
    return tools
