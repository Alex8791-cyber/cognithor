"""Tests für Canvas Manager."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from jarvis.channels.canvas import CanvasManager, CanvasState


class TestCanvasManager:
    """Tests für CanvasManager."""

    @pytest.mark.asyncio
    async def test_push(self) -> None:
        cm = CanvasManager()
        await cm.push("s1", "<h1>Hello</h1>", "Test")
        html = await cm.snapshot("s1")
        assert html == "<h1>Hello</h1>"
        assert cm.get_title("s1") == "Test"

    @pytest.mark.asyncio
    async def test_push_with_broadcaster(self) -> None:
        broadcaster = AsyncMock()
        cm = CanvasManager(broadcaster=broadcaster)
        await cm.push("s1", "<p>Content</p>", "Title")
        broadcaster.assert_called_once_with("s1", {
            "type": "canvas_push",
            "html": "<p>Content</p>",
            "title": "Title",
        })

    @pytest.mark.asyncio
    async def test_reset(self) -> None:
        cm = CanvasManager()
        await cm.push("s1", "<h1>Hello</h1>")
        await cm.reset("s1")
        html = await cm.snapshot("s1")
        assert html == ""

    @pytest.mark.asyncio
    async def test_reset_with_broadcaster(self) -> None:
        broadcaster = AsyncMock()
        cm = CanvasManager(broadcaster=broadcaster)
        await cm.push("s1", "<p>Content</p>")
        broadcaster.reset_mock()
        await cm.reset("s1")
        broadcaster.assert_called_once_with("s1", {"type": "canvas_reset"})

    @pytest.mark.asyncio
    async def test_snapshot_empty(self) -> None:
        cm = CanvasManager()
        html = await cm.snapshot("nonexistent")
        assert html == ""

    @pytest.mark.asyncio
    async def test_eval_js(self) -> None:
        broadcaster = AsyncMock()
        cm = CanvasManager(broadcaster=broadcaster)
        await cm.eval_js("s1", "alert('test')")
        broadcaster.assert_called_once_with("s1", {
            "type": "canvas_eval",
            "js": "alert('test')",
        })

    @pytest.mark.asyncio
    async def test_undo(self) -> None:
        cm = CanvasManager()
        await cm.push("s1", "<h1>First</h1>", "V1")
        await cm.push("s1", "<h2>Second</h2>", "V2")

        result = await cm.undo("s1")
        assert result == "<h1>First</h1>"
        assert cm.get_title("s1") == "V1"

    @pytest.mark.asyncio
    async def test_undo_empty_history(self) -> None:
        cm = CanvasManager()
        result = await cm.undo("s1")
        assert result is None

    @pytest.mark.asyncio
    async def test_redo(self) -> None:
        cm = CanvasManager()
        await cm.push("s1", "<h1>First</h1>", "V1")
        await cm.push("s1", "<h2>Second</h2>", "V2")
        await cm.undo("s1")

        result = await cm.redo("s1")
        assert result == "<h2>Second</h2>"
        assert cm.get_title("s1") == "V2"

    @pytest.mark.asyncio
    async def test_redo_empty_stack(self) -> None:
        cm = CanvasManager()
        result = await cm.redo("s1")
        assert result is None

    @pytest.mark.asyncio
    async def test_redo_cleared_on_push(self) -> None:
        cm = CanvasManager()
        await cm.push("s1", "<h1>First</h1>")
        await cm.push("s1", "<h2>Second</h2>")
        await cm.undo("s1")
        await cm.push("s1", "<h3>Third</h3>")

        result = await cm.redo("s1")
        assert result is None  # Redo stack was cleared

    @pytest.mark.asyncio
    async def test_has_content(self) -> None:
        cm = CanvasManager()
        assert not cm.has_content("s1")
        await cm.push("s1", "<p>Content</p>")
        assert cm.has_content("s1")

    @pytest.mark.asyncio
    async def test_history_count(self) -> None:
        cm = CanvasManager()
        assert cm.history_count("s1") == 0
        await cm.push("s1", "<h1>V1</h1>")
        await cm.push("s1", "<h2>V2</h2>")
        assert cm.history_count("s1") == 1  # V1 in history
        await cm.push("s1", "<h3>V3</h3>")
        assert cm.history_count("s1") == 2  # V1, V2 in history

    @pytest.mark.asyncio
    async def test_cleanup_session(self) -> None:
        cm = CanvasManager()
        await cm.push("s1", "<p>Content</p>")
        cm.cleanup_session("s1")
        assert not cm.has_content("s1")

    @pytest.mark.asyncio
    async def test_cleanup_nonexistent_session(self) -> None:
        cm = CanvasManager()
        cm.cleanup_session("nonexistent")  # Should not raise

    @pytest.mark.asyncio
    async def test_multiple_sessions(self) -> None:
        cm = CanvasManager()
        await cm.push("s1", "<h1>Session 1</h1>")
        await cm.push("s2", "<h2>Session 2</h2>")

        assert await cm.snapshot("s1") == "<h1>Session 1</h1>"
        assert await cm.snapshot("s2") == "<h2>Session 2</h2>"

    @pytest.mark.asyncio
    async def test_history_limit(self) -> None:
        cm = CanvasManager()
        state = cm._get_state("s1")
        state.max_history = 3

        for i in range(10):
            await cm.push("s1", f"<p>Version {i}</p>")

        assert cm.history_count("s1") <= 3


class TestCanvasState:
    """Tests für CanvasState Datenklasse."""

    def test_defaults(self) -> None:
        state = CanvasState()
        assert state.current_html == ""
        assert state.current_title == ""
        assert state.history == []
        assert state.redo_stack == []
        assert state.max_history == 50
