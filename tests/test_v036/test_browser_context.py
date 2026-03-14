"""Tests for Feature 7: Tab-as-Context-Window (Browser Automation)."""

from __future__ import annotations

import pytest

from jarvis.browser.context_bridge import (
    ContextSnapshot,
    TabContextBridge,
    TabHandle,
    TabInfo,
)


class TestTabOperations:
    @pytest.mark.asyncio
    async def test_tab_open_and_switch(self):
        bridge = TabContextBridge()
        handle = await bridge.open_tab("https://example.com", "Example")
        assert handle.url == "https://example.com"
        assert handle.label == "Example"
        assert bridge.tab_count == 1

        snap = await bridge.switch_tab(handle)
        assert isinstance(snap, ContextSnapshot)
        assert snap.tab_id == handle.tab_id

    @pytest.mark.asyncio
    async def test_tab_content_injected_into_context(self):
        bridge = TabContextBridge()
        handle = await bridge.open_tab("https://example.com", "Test")
        snap = await bridge.switch_tab(handle)
        assert snap.text_content  # Should have some content (at least fallback)

    @pytest.mark.asyncio
    async def test_list_tabs(self):
        bridge = TabContextBridge()
        await bridge.open_tab("https://a.com", "A")
        await bridge.open_tab("https://b.com", "B")

        tabs = await bridge.list_tabs()
        assert len(tabs) == 2
        labels = {t.label for t in tabs}
        assert labels == {"A", "B"}

    @pytest.mark.asyncio
    async def test_close_tab(self):
        bridge = TabContextBridge()
        handle = await bridge.open_tab("https://example.com", "X")
        assert bridge.tab_count == 1

        await bridge.close_tab(handle)
        assert bridge.tab_count == 0

    @pytest.mark.asyncio
    async def test_switch_nonexistent_tab_raises(self):
        bridge = TabContextBridge()
        with pytest.raises(ValueError, match="not found"):
            await bridge.switch_tab(TabHandle(tab_id="nonexistent"))


class TestCheckpointRestore:
    @pytest.mark.asyncio
    async def test_tab_state_survives_checkpoint_restore(self):
        bridge = TabContextBridge()
        h1 = await bridge.open_tab("https://a.com", "A")
        h2 = await bridge.open_tab("https://b.com", "B")

        snap = bridge.snapshot()

        bridge2 = TabContextBridge()
        bridge2.restore(snap)

        assert bridge2.tab_count == 2
        tabs = await bridge2.list_tabs()
        urls = {t.url for t in tabs}
        assert urls == {"https://a.com", "https://b.com"}


class TestHeadless:
    @pytest.mark.asyncio
    async def test_headless_mode_works_without_browser(self):
        """Bridge works without a real browser (headless/no-browser mode)."""
        bridge = TabContextBridge(browser_agent=None)
        handle = await bridge.open_tab("https://example.com", "Test")
        content = await bridge.get_tab_content(handle)
        assert content  # Fallback text returned for headless mode
