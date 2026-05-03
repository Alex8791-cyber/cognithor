"""Desktop Tools: Clipboard & Screenshot for Cognithor.

MCP-Tools for interacting with the desktop environment.

Tools:
  - get_clipboard: Read clipboard content (text or image)
  - set_clipboard: Copy text to clipboard
  - screenshot_desktop: Full desktop screenshot
  - screenshot_region: Screenshot of a specific screen region

Note: Screenshots may contain sensitive information.
All operations use run_in_executor for sync I/O.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from cognithor.utils.logging import get_logger

log = get_logger(__name__)

# Maximum screenshot resolution before downscaling
_MAX_SCREENSHOT_WIDTH = 3840
_MAX_SCREENSHOT_HEIGHT = 2160

__all__ = [
    "DesktopTools",
    "register_desktop_tools",
]


# ---------------------------------------------------------------------------
# Clipboard helpers
# ---------------------------------------------------------------------------


def _ps_get_clipboard_text() -> str:
    """Read text from clipboard via PowerShell."""
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", "Get-Clipboard"],
        capture_output=True,
        text=True,
        timeout=10,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if result.returncode != 0:
        raise RuntimeError(f"PowerShell Get-Clipboard failed: {result.stderr.strip()}")
    return result.stdout.rstrip("\r\n")


def _ps_get_clipboard_image(save_path: str) -> bool:
    """Try to save clipboard image via PowerShell. Returns True on success."""
    escaped = save_path.replace("'", "''")
    cmd = (
        "$img = Get-Clipboard -Format Image; "
        f"if ($img) {{ $img.Save('{escaped}'); 'OK' }} else {{ 'NO_IMAGE' }}"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", cmd],
        capture_output=True,
        text=True,
        timeout=15,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    return result.returncode == 0 and "OK" in result.stdout


def _ps_set_clipboard(text: str) -> None:
    """Write text to clipboard via PowerShell."""
    # Use stdin to avoid escaping issues
    subprocess.run(
        ["powershell", "-NoProfile", "-Command", "Set-Clipboard -Value $input"],
        input=text,
        capture_output=True,
        text=True,
        timeout=10,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        check=True,
    )


# ---------------------------------------------------------------------------
# Screenshot helpers
# ---------------------------------------------------------------------------


def _try_mss_screenshot(
    monitor_index: int = 0,
    region: dict[str, int] | None = None,
) -> tuple[bytes, int, int] | None:
    """Capture screenshot via mss. Returns (png_bytes, width, height) or None."""
    try:
        import mss
        import mss.tools

        with mss.mss() as sct:
            if region:
                mon = {
                    "left": region["x"],
                    "top": region["y"],
                    "width": region["width"],
                    "height": region["height"],
                }
            else:
                monitors = sct.monitors
                # monitors[0] is "all monitors combined", [1] is primary
                idx = min(monitor_index + 1, len(monitors) - 1)
                idx = max(idx, 0)
                mon = monitors[idx] if idx < len(monitors) else monitors[0]

            img = sct.grab(mon)
            png = mss.tools.to_png(img.rgb, img.size)
            return png, img.width, img.height  # type: ignore[return-value]
    except Exception:
        return None


def _try_pil_screenshot(
    region: tuple[int, int, int, int] | None = None,
) -> tuple[bytes, int, int] | None:
    """Capture screenshot via PIL.ImageGrab. Returns (png_bytes, w, h) or None."""
    try:
        from io import BytesIO

        from PIL import ImageGrab

        if region:
            x, y, w, h = region
            img = ImageGrab.grab(bbox=(x, y, x + w, y + h))
        else:
            img = ImageGrab.grab()

        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue(), img.width, img.height
    except Exception:
        return None


def _try_ps_screenshot(save_path: str) -> tuple[int, int] | None:
    """Capture screenshot via PowerShell. Returns (width, height) or None."""
    escaped = save_path.replace("'", "''")
    cmd = (
        "Add-Type -AssemblyName System.Windows.Forms; "
        "Add-Type -AssemblyName System.Drawing; "
        "$bounds = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds; "
        "$bmp = New-Object System.Drawing.Bitmap($bounds.Width, $bounds.Height); "
        "$g = [System.Drawing.Graphics]::FromImage($bmp); "
        "$g.CopyFromScreen($bounds.Location, [System.Drawing.Point]::Empty, $bounds.Size); "
        f"$bmp.Save('{escaped}'); "
        "$g.Dispose(); $bmp.Dispose(); "
        'Write-Output "$($bounds.Width)x$($bounds.Height)"'
    )
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", cmd],
            capture_output=True,
            text=True,
            timeout=15,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split("x")
            if len(parts) == 2:
                return int(parts[0]), int(parts[1])
    except Exception:
        log.debug("desktop_screen_resolution_detection_failed", exc_info=True)
    return None


def _downscale_if_needed(png_bytes: bytes, width: int, height: int) -> tuple[bytes, int, int]:
    """Downscale image if it exceeds 4K. Returns (png_bytes, w, h)."""
    if width <= _MAX_SCREENSHOT_WIDTH and height <= _MAX_SCREENSHOT_HEIGHT:
        return png_bytes, width, height

    try:
        from io import BytesIO

        from PIL import Image

        img = Image.open(BytesIO(png_bytes))
        ratio = min(_MAX_SCREENSHOT_WIDTH / width, _MAX_SCREENSHOT_HEIGHT / height)
        new_w = int(width * ratio)
        new_h = int(height * ratio)
        img = img.resize((new_w, new_h), Image.LANCZOS)  # type: ignore[assignment,attr-defined]
        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue(), new_w, new_h
    except Exception:
        return png_bytes, width, height


# ---------------------------------------------------------------------------
# DesktopTools class
# ---------------------------------------------------------------------------


class DesktopTools:
    """Clipboard and screenshot operations."""

    def __init__(self, workspace_dir: Path, config: Any = None) -> None:
        self._screenshots_dir = workspace_dir / "screenshots"
        self._screenshots_dir.mkdir(parents=True, exist_ok=True)
        self._config = config
        self._vision_analyzer: Any = None

    def _set_vision(self, vision_analyzer: Any) -> None:
        """Inject optional VisionAnalyzer for auto-describing screenshots."""
        self._vision_analyzer = vision_analyzer

    def _timestamp(self) -> str:
        return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

    # -- Clipboard ----------------------------------------------------------

    async def get_clipboard(self) -> dict[str, Any]:
        """Read clipboard content (text or image)."""
        loop = asyncio.get_running_loop()

        # Try text first
        try:
            text = await loop.run_in_executor(None, _ps_get_clipboard_text)
            if text:
                return {"type": "text", "content": text, "length": len(text)}
        except Exception:
            log.debug("desktop_clipboard_text_read_failed", exc_info=True)

        # Try image
        ts = self._timestamp()
        img_path = self._screenshots_dir / f"clipboard_{ts}.png"
        try:
            ok = await loop.run_in_executor(None, _ps_get_clipboard_image, str(img_path))
            if ok and img_path.exists():
                result: dict[str, Any] = {
                    "type": "image",
                    "content": "Clipboard contains image",
                    "path": str(img_path),
                }
                # Optional vision analysis
                if self._vision_analyzer is not None:
                    try:
                        desc = await self._vision_analyzer.analyze(str(img_path))
                        result["description"] = desc
                    except Exception:
                        log.debug("desktop_clipboard_image_vision_failed", exc_info=True)
                return result
        except Exception:
            log.debug("desktop_clipboard_image_read_failed", exc_info=True)

        return {"type": "empty", "content": "Clipboard is empty or unreadable."}

    async def set_clipboard(self, text: str) -> dict[str, Any]:
        """Copy text to clipboard."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _ps_set_clipboard, text)
        return {"status": "ok", "chars": len(text)}

    # -- Screenshots --------------------------------------------------------

    async def screenshot_desktop(
        self,
        monitor: int = 0,
        save_path: str | None = None,
    ) -> dict[str, Any]:
        """Take a full desktop screenshot."""
        loop = asyncio.get_running_loop()
        ts = self._timestamp()
        out_path = Path(save_path) if save_path else self._screenshots_dir / f"desktop_{ts}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # 1) mss
        result_data = await loop.run_in_executor(None, _try_mss_screenshot, monitor, None)
        if result_data is not None:
            png_bytes, w, h = result_data
            png_bytes, w, h = _downscale_if_needed(png_bytes, w, h)
            out_path.write_bytes(png_bytes)
            return await self._screenshot_result(out_path, w, h, "mss")

        # 2) PIL.ImageGrab
        result_data = await loop.run_in_executor(None, _try_pil_screenshot, None)
        if result_data is not None:
            png_bytes, w, h = result_data
            png_bytes, w, h = _downscale_if_needed(png_bytes, w, h)
            out_path.write_bytes(png_bytes)
            return await self._screenshot_result(out_path, w, h, "PIL")

        # 3) PowerShell
        if sys.platform == "win32":
            dims = await loop.run_in_executor(None, _try_ps_screenshot, str(out_path))
            if dims is not None:
                w, h = dims
                return await self._screenshot_result(out_path, w, h, "PowerShell")

        return {"error": "No screenshot backend available. Install 'mss' or 'Pillow'."}

    async def screenshot_region(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        save_path: str | None = None,
    ) -> dict[str, Any]:
        """Take a screenshot of a specific screen region."""
        loop = asyncio.get_running_loop()
        ts = self._timestamp()
        fname = f"region_{x}_{y}_{width}_{height}_{ts}.png"
        out_path = Path(save_path) if save_path else self._screenshots_dir / fname
        out_path.parent.mkdir(parents=True, exist_ok=True)

        region_dict = {"x": x, "y": y, "width": width, "height": height}
        region_tuple = (x, y, width, height)

        # 1) mss
        result_data = await loop.run_in_executor(None, _try_mss_screenshot, 0, region_dict)
        if result_data is not None:
            png_bytes, w, h = result_data
            png_bytes, w, h = _downscale_if_needed(png_bytes, w, h)
            out_path.write_bytes(png_bytes)
            return await self._screenshot_result(out_path, w, h, "mss")

        # 2) PIL.ImageGrab
        result_data = await loop.run_in_executor(None, _try_pil_screenshot, region_tuple)
        if result_data is not None:
            png_bytes, w, h = result_data
            png_bytes, w, h = _downscale_if_needed(png_bytes, w, h)
            out_path.write_bytes(png_bytes)
            return await self._screenshot_result(out_path, w, h, "PIL")

        # No PowerShell region support -- too complex
        return {"error": "No region screenshot backend available. Install 'mss' or 'Pillow'."}

    async def _screenshot_result(
        self, path: Path, width: int, height: int, backend: str
    ) -> dict[str, Any]:
        """Build result dict, optionally adding vision description."""
        result: dict[str, Any] = {
            "path": str(path),
            "width": width,
            "height": height,
            "backend": backend,
        }
        if self._vision_analyzer is not None:
            try:
                desc = await self._vision_analyzer.analyze(str(path))
                result["description"] = desc
            except Exception:
                log.debug("desktop_screenshot_vision_failed", exc_info=True)
        return result


# ---------------------------------------------------------------------------
# MCP registration
# ---------------------------------------------------------------------------


def register_desktop_tools(
    mcp_client: Any,
    config: Any,
) -> DesktopTools:
    """Register desktop (clipboard/screenshot) MCP tools.

    Args:
        mcp_client: JarvisMCPClient instance.
        config: CognithorConfig instance.

    Returns:
        DesktopTools instance.
    """
    workspace = getattr(config, "workspace_dir", Path.home() / ".cognithor" / "workspace")
    tools = DesktopTools(Path(workspace), config)

    # -- get_clipboard ------------------------------------------------------
    async def _get_clipboard(**_kwargs: Any) -> str:
        result = await tools.get_clipboard()
        if result["type"] == "text":
            content = result["content"]
            # Truncate very long text for LLM context
            if len(content) > 5000:
                content = content[:5000] + f"\n... (truncated, total {result['length']} chars)"
            return f"Clipboard text ({result['length']} chars):\n{content}"
        elif result["type"] == "image":
            desc = result.get("description", "")
            extra = f"\nDescription: {desc}" if desc else ""
            return f"Clipboard contains an image, saved to: {result['path']}{extra}"
        return cast("str", result["content"])

    mcp_client.register_builtin_handler(
        "get_clipboard",
        _get_clipboard,
        description=(
            "Read clipboard content (text or image). "
            "Note: Clipboard may contain sensitive information."
        ),
        input_schema={"type": "object", "properties": {}},
    )

    # -- set_clipboard ------------------------------------------------------
    async def _set_clipboard(**kwargs: Any) -> str:
        text = kwargs.get("text", "")
        if not text:
            return "Error: 'text' is required."
        result = await tools.set_clipboard(text)
        return f"Copied {result['chars']} characters to clipboard."

    mcp_client.register_builtin_handler(
        "set_clipboard",
        _set_clipboard,
        description="Copy text to the system clipboard.",
        input_schema={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text to copy to clipboard",
                },
            },
            "required": ["text"],
        },
    )

    # -- screenshot_desktop -------------------------------------------------
    async def _screenshot_desktop(**kwargs: Any) -> str:
        monitor = int(kwargs.get("monitor", 0))
        save_path = kwargs.get("save_path")
        result = await tools.screenshot_desktop(monitor=monitor, save_path=save_path)
        if "error" in result:
            return f"Error: {result['error']}"
        desc = result.get("description", "")
        extra = f"\nDescription: {desc}" if desc else ""
        return (
            f"Screenshot saved: {result['path']} "
            f"({result['width']}x{result['height']}, backend: {result['backend']}){extra}"
        )

    mcp_client.register_builtin_handler(
        "screenshot_desktop",
        _screenshot_desktop,
        description=(
            "Take a full desktop screenshot. Note: Screenshots may contain sensitive information."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "monitor": {
                    "type": "integer",
                    "description": "Monitor index (0 = primary, default: 0)",
                    "default": 0,
                },
                "save_path": {
                    "type": "string",
                    "description": "Custom save path (optional, auto-generated if omitted)",
                },
            },
        },
    )

    # -- screenshot_region --------------------------------------------------
    async def _screenshot_region(**kwargs: Any) -> str:
        x = int(kwargs.get("x", 0))
        y = int(kwargs.get("y", 0))
        width = int(kwargs.get("width", 800))
        height = int(kwargs.get("height", 600))
        save_path = kwargs.get("save_path")
        result = await tools.screenshot_region(
            x=x,
            y=y,
            width=width,
            height=height,
            save_path=save_path,
        )
        if "error" in result:
            return f"Error: {result['error']}"
        desc = result.get("description", "")
        extra = f"\nDescription: {desc}" if desc else ""
        return (
            f"Region screenshot saved: {result['path']} "
            f"({result['width']}x{result['height']}, backend: {result['backend']}){extra}"
        )

    mcp_client.register_builtin_handler(
        "screenshot_region",
        _screenshot_region,
        description=(
            "Take a screenshot of a specific screen region. "
            "Note: Screenshots may contain sensitive information."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "x": {
                    "type": "integer",
                    "description": "Left edge X coordinate",
                },
                "y": {
                    "type": "integer",
                    "description": "Top edge Y coordinate",
                },
                "width": {
                    "type": "integer",
                    "description": "Region width in pixels",
                },
                "height": {
                    "type": "integer",
                    "description": "Region height in pixels",
                },
                "save_path": {
                    "type": "string",
                    "description": "Custom save path (optional)",
                },
            },
            "required": ["x", "y", "width", "height"],
        },
    )

    log.info(
        "desktop_tools_registered",
        tools=["get_clipboard", "set_clipboard", "screenshot_desktop", "screenshot_region"],
    )
    return tools
