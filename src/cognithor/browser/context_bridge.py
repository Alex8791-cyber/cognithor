"""Tab-as-Context-Window bridge for browser automation.

Maps browser tabs to agent context windows. Agents can switch tabs
to change what they "see" — tab content is injected into the agent's
ContextWindow on switch.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, cast

from cognithor.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class TabHandle:
    """Handle to an open browser tab."""

    tab_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    url: str = ""
    label: str = ""


@dataclass
class TabInfo:
    """Summary information about a tab."""

    tab_id: str = ""
    url: str = ""
    label: str = ""
    title: str = ""
    active: bool = False


@dataclass
class ContextSnapshot:
    """Snapshot of tab content for context injection."""

    tab_id: str = ""
    title: str = ""
    url: str = ""
    text_content: str = ""
    tokens_estimate: int = 0


class TabContextBridge:
    """Maps browser tabs to agent context windows.

    Uses the existing browser automation (Playwright/Selenium) wrapper
    to manage tabs. Tab state is persisted in ContextWindow snapshots
    for checkpoint/resume.
    """

    def __init__(self, browser_agent: Any = None) -> None:
        self._browser = browser_agent
        self._tabs: dict[str, TabHandle] = {}
        self._active_tab: str | None = None

    async def open_tab(self, url: str, label: str = "") -> TabHandle:
        """Open a new browser tab and navigate to the URL."""
        handle = TabHandle(url=url, label=label or url[:50])
        self._tabs[handle.tab_id] = handle

        if self._browser is not None:
            try:
                await self._browser.navigate(url)
            except Exception:
                log.debug("tab_navigate_failed", url=url, exc_info=True)

        self._active_tab = handle.tab_id
        log.info("tab_opened", tab_id=handle.tab_id[:8], url=url[:60])
        return handle

    async def switch_tab(self, handle: TabHandle) -> ContextSnapshot:
        """Switch to an existing tab and return its content snapshot."""
        if handle.tab_id not in self._tabs:
            raise ValueError(f"Tab {handle.tab_id} not found")

        self._active_tab = handle.tab_id
        content = await self.get_tab_content(handle)
        log.info("tab_switched", tab_id=handle.tab_id[:8])
        return ContextSnapshot(
            tab_id=handle.tab_id,
            title=handle.label,
            url=handle.url,
            text_content=content,
            tokens_estimate=len(content.split()) * 2,  # rough estimate
        )

    async def get_tab_content(self, handle: TabHandle) -> str:
        """Extract text content from a tab."""
        if self._browser is not None:
            try:
                return cast("str", await self._browser.extract_text())

            except Exception:
                log.debug("tab_extract_failed", exc_info=True)
        return f"[Tab: {handle.label} — {handle.url}]"

    async def close_tab(self, handle: TabHandle) -> None:
        """Close a browser tab."""
        self._tabs.pop(handle.tab_id, None)
        if self._active_tab == handle.tab_id:
            self._active_tab = None
        log.info("tab_closed", tab_id=handle.tab_id[:8])

    async def list_tabs(self) -> list[TabInfo]:
        """List all open tabs."""
        return [
            TabInfo(
                tab_id=h.tab_id,
                url=h.url,
                label=h.label,
                active=(h.tab_id == self._active_tab),
            )
            for h in self._tabs.values()
        ]

    def snapshot(self) -> dict[str, Any]:
        """Serialize tab state for checkpointing."""
        return {
            "tabs": [
                {"tab_id": h.tab_id, "url": h.url, "label": h.label} for h in self._tabs.values()
            ],
            "active_tab": self._active_tab,
        }

    def restore(self, data: dict[str, Any]) -> None:
        """Restore tab state from checkpoint."""
        self._tabs.clear()
        for t in data.get("tabs", []):
            handle = TabHandle(tab_id=t["tab_id"], url=t["url"], label=t["label"])
            self._tabs[handle.tab_id] = handle
        self._active_tab = data.get("active_tab")

    @property
    def tab_count(self) -> int:
        return len(self._tabs)
