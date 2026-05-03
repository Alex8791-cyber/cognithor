"""Windows UI Automation provider — reads elements from the Accessibility Tree.

Uses pywinauto with UIA backend to enumerate interactive UI elements
of the foreground window. Provides exact coordinates, names, types,
and states directly from the OS.

Graceful degradation: if pywinauto is not installed or UIA access fails,
all methods return empty lists.
"""

from __future__ import annotations

import sys
from typing import Any

from cognithor.utils.logging import get_logger

log = get_logger(__name__)

_pywinauto_available = False
if sys.platform == "win32":
    try:
        from pywinauto import Desktop  # noqa: F401

        _pywinauto_available = True
    except ImportError:
        pass

_INTERACTIVE_TYPES = frozenset(
    {
        "Button",
        "Edit",
        "MenuItem",
        "ListItem",
        "TabItem",
        "Hyperlink",
        "CheckBox",
        "ComboBox",
        "RadioButton",
        "TreeItem",
        "Slider",
        "ToggleButton",
    }
)

_MAX_ELEMENTS = 30
_MAX_DEPTH = 8


class UIAutomationProvider:
    """Reads UI elements from the Windows Accessibility Tree via pywinauto UIA."""

    def __init__(self) -> None:
        self._pywinauto_available = _pywinauto_available

    @staticmethod
    def _is_interactive_type(control_type: str) -> bool:
        """Check if a control type is interactive (clickable/typeable)."""
        return control_type in _INTERACTIVE_TYPES

    def _element_to_dict(self, elem: Any) -> dict[str, Any] | None:
        """Convert a pywinauto element to a dict with standard format.

        Returns None if element should be filtered out.
        """
        try:
            control_type = elem.element_info.control_type
            if not self._is_interactive_type(control_type):
                return None

            rect = elem.rectangle()
            w = rect.right - rect.left
            h = rect.bottom - rect.top
            if w <= 0 or h <= 0:
                return None

            name = elem.window_text() or ""
            if len(name) > 80:
                name = name[:77] + "..."

            # Try to get value (for Edit fields, etc.)
            text = ""
            try:
                iface = elem.iface_value
                if iface:
                    text = str(iface.CurrentValue or "")
                    if len(text) > 100:
                        text = text[:97] + "..."
            except Exception:
                log.debug("ui_automation_element_value_read_failed", exc_info=True)

            return {
                "name": name,
                "type": control_type,
                "x": rect.left + w // 2,
                "y": rect.top + h // 2,
                "w": w,
                "h": h,
                "clickable": bool(elem.is_enabled()),
                "text": text,
                "source": "uia",
            }
        except Exception:
            return None

    def _cap_and_sort(self, elements: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Sort by screen position (top-to-bottom, left-to-right) and cap."""
        elements.sort(key=lambda e: (e.get("y", 0) // 50, e.get("x", 0)))
        return elements[:_MAX_ELEMENTS]

    def _get_foreground_window(self) -> Any:
        """Get the foreground window wrapper via pywinauto."""
        from pywinauto import Desktop

        desktop = Desktop(backend="uia")
        windows = desktop.windows()
        if not windows:
            return None
        try:
            import ctypes

            hwnd = ctypes.windll.user32.GetForegroundWindow()
            for w in windows:
                try:
                    if w.handle == hwnd:
                        return w
                except Exception:
                    continue
        except Exception:
            log.debug("ui_automation_foreground_window_detect_failed", exc_info=True)
        return windows[0] if windows else None

    def _walk_children(self, element: Any, depth: int, results: list[dict[str, Any]]) -> None:
        """Recursively walk child elements up to max depth."""
        if depth > _MAX_DEPTH or len(results) >= _MAX_ELEMENTS * 2:
            return

        try:
            children = element.children()
        except Exception:
            return

        for child in children:
            elem_dict = self._element_to_dict(child)
            if elem_dict is not None:
                results.append(elem_dict)
            self._walk_children(child, depth + 1, results)

    def get_focused_window_elements(self) -> list[dict[str, Any]]:
        """Return interactive elements of the foreground window.

        Returns list of dicts with: name, type, x, y, w, h, clickable, text, source.
        Returns empty list on any failure (graceful degradation).
        """
        if not self._pywinauto_available:
            return []

        try:
            window = self._get_foreground_window()
            if window is None:
                return []

            results: list[dict[str, Any]] = []
            self._walk_children(window, depth=0, results=results)
            return self._cap_and_sort(results)

        except Exception as exc:
            log.debug("uia_enumeration_failed", error=str(exc)[:200])
            return []
