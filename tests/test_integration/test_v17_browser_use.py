"""Tests: Browser-Use v17.

Tests für alle v17-Module: Types, PageAnalyzer, SessionManager, BrowserAgent.
Playwright wird gemockt — alle Tests laufen ohne echten Browser.
"""

import asyncio
import json
import os
import tempfile
import pytest
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from jarvis.browser.types import (
    ActionResult,
    ActionType,
    BrowserAction,
    BrowserConfig,
    BrowserWorkflow,
    ElementInfo,
    ElementType,
    ExtractionMode,
    FormField,
    FormInfo,
    PageState,
    WorkflowStatus,
)
from jarvis.browser.page_analyzer import PageAnalyzer
from jarvis.browser.session_manager import SessionManager, SessionSnapshot


# ============================================================================
# Types Tests
# ============================================================================


class TestActionType:
    def test_all_actions(self):
        assert len(ActionType) >= 20
        assert ActionType.NAVIGATE.value == "navigate"
        assert ActionType.CLICK.value == "click"
        assert ActionType.FILL.value == "fill"

    def test_element_types(self):
        assert ElementType.BUTTON.value == "button"
        assert ElementType.LINK.value == "link"
        assert ElementType.INPUT.value == "input"


class TestElementInfo:
    def test_basic(self):
        e = ElementInfo(selector="#btn", element_type=ElementType.BUTTON, text="Submit")
        assert e.label == "[button] Submit"

    def test_label_fallback(self):
        e = ElementInfo(selector="#x", element_type=ElementType.INPUT, aria_label="Email")
        assert "Email" in e.label

    def test_label_placeholder(self):
        e = ElementInfo(selector="#x", element_type=ElementType.INPUT, placeholder="Enter email")
        assert "Enter email" in e.label

    def test_to_dict(self):
        e = ElementInfo(selector="a.link", element_type=ElementType.LINK, href="https://x.com")
        d = e.to_dict()
        assert d["type"] == "link"
        assert d["href"] == "https://x.com"

    def test_visibility(self):
        e = ElementInfo(selector="#x", element_type=ElementType.BUTTON,
                        is_visible=False, is_enabled=False)
        assert not e.is_visible
        assert not e.is_enabled


class TestFormField:
    def test_basic(self):
        f = FormField(name="email", field_type="email", label="E-Mail", required=True)
        d = f.to_dict()
        assert d["name"] == "email"
        assert d["required"]

    def test_with_options(self):
        f = FormField(name="country", field_type="select", options=["DE", "AT", "CH"])
        assert len(f.options) == 3


class TestFormInfo:
    def test_basic(self):
        form = FormInfo(
            action="/submit", method="POST",
            fields=[FormField(name="email", field_type="email")],
            submit_selector="button[type=submit]",
        )
        d = form.to_dict()
        assert d["method"] == "POST"
        assert len(d["fields"]) == 1


class TestPageState:
    def test_basic(self):
        state = PageState(url="https://example.com", title="Example")
        assert state.timestamp

    def test_to_dict(self):
        state = PageState(url="https://x.com", title="X", text_content="Hello World")
        d = state.to_dict()
        assert d["url"] == "https://x.com"
        assert d["text_length"] == 11

    def test_to_summary(self):
        state = PageState(
            url="https://x.com", title="Test Page",
            text_content="Hello World Content",
            links=[ElementInfo(selector="a", element_type=ElementType.LINK, text="Link1")],
            buttons=[ElementInfo(selector="button", element_type=ElementType.BUTTON, text="Click")],
        )
        summary = state.to_summary()
        assert "URL: https://x.com" in summary
        assert "Links: 1" in summary
        assert "Buttons: 1" in summary
        assert "Hello World Content" in summary

    def test_summary_truncation(self):
        state = PageState(text_content="x" * 10000)
        summary = state.to_summary(max_text=100)
        assert "truncated" in summary


class TestBrowserAction:
    def test_basic(self):
        a = BrowserAction(
            action_type=ActionType.NAVIGATE,
            params={"url": "https://example.com"},
            description="Open example",
        )
        assert a.action_id
        d = a.to_dict()
        assert d["action"] == "navigate"
        assert d["params"]["url"] == "https://example.com"


class TestActionResult:
    def test_success(self):
        r = ActionResult(action_id="a1", success=True, duration_ms=50)
        d = r.to_dict()
        assert d["success"]
        assert d["duration_ms"] == 50

    def test_failure(self):
        r = ActionResult(action_id="a2", success=False, error="Element not found")
        d = r.to_dict()
        assert not d["success"]
        assert "Element not found" in d["error"]

    def test_page_changed(self):
        r = ActionResult(action_id="a3", success=True, page_changed=True,
                         new_url="https://new.com")
        d = r.to_dict()
        assert d["page_changed"]
        assert d["new_url"] == "https://new.com"


class TestBrowserWorkflow:
    def test_basic(self):
        wf = BrowserWorkflow(
            name="Login Test",
            steps=[
                BrowserAction(action_type=ActionType.NAVIGATE, params={"url": "https://x.com"}),
                BrowserAction(action_type=ActionType.FILL, params={"selector": "#email", "value": "a@b.com"}),
                BrowserAction(action_type=ActionType.CLICK, params={"selector": "#submit"}),
            ],
        )
        assert wf.workflow_id
        assert len(wf.steps) == 3
        assert wf.status == WorkflowStatus.PENDING
        assert not wf.is_complete

    def test_success_rate(self):
        wf = BrowserWorkflow()
        wf.results = [
            ActionResult(action_id="1", success=True),
            ActionResult(action_id="2", success=True),
            ActionResult(action_id="3", success=False),
        ]
        assert wf.success_rate == pytest.approx(2/3)

    def test_is_complete(self):
        wf = BrowserWorkflow(status=WorkflowStatus.COMPLETED)
        assert wf.is_complete
        wf2 = BrowserWorkflow(status=WorkflowStatus.FAILED)
        assert wf2.is_complete
        wf3 = BrowserWorkflow(status=WorkflowStatus.RUNNING)
        assert not wf3.is_complete

    def test_to_dict(self):
        wf = BrowserWorkflow(name="Test", steps=[
            BrowserAction(action_type=ActionType.CLICK, params={}),
        ])
        d = wf.to_dict()
        assert d["name"] == "Test"
        assert d["steps"] == 1


class TestBrowserConfig:
    def test_defaults(self):
        c = BrowserConfig()
        assert c.headless
        assert c.viewport_width == 1280
        assert c.locale == "de-DE"
        assert c.timezone == "Europe/Berlin"
        assert c.persist_cookies

    def test_custom(self):
        c = BrowserConfig(headless=False, viewport_width=1920, block_ads=True)
        d = c.to_dict()
        assert d["viewport"] == "1920x720"


# ============================================================================
# PageAnalyzer Tests (mit Mock-Page)
# ============================================================================


def _mock_page(evaluate_results: dict[str, Any] | None = None):
    """Erstellt eine Mock-Playwright-Page."""
    page = AsyncMock()
    page.url = "https://example.com"
    page.title = AsyncMock(return_value="Example Page")

    default_results = {
        "links": [],
        "buttons": [{"selector": "#submit", "text": "Submit", "visible": True,
                      "enabled": True, "ariaLabel": "", "type": "submit"}],
        "inputs": [{"selector": "#email", "name": "email", "type": "email",
                     "value": "", "placeholder": "Enter email", "label": "E-Mail",
                     "required": True, "visible": True, "enabled": True,
                     "ariaLabel": "", "options": []}],
        "forms": [{"action": "/login", "method": "POST", "name": "loginForm",
                    "fields": [
                        {"name": "email", "type": "email", "label": "E-Mail",
                         "value": "", "placeholder": "Enter email",
                         "required": True, "options": [], "selector": "#email"}
                    ],
                    "submitSelector": "#submit", "selector": "form"}],
        "tables": [{"headers": ["Name", "Value"], "rows": [["A", "1"]], "rowCount": 1, "colCount": 2}],
        "text": "Example Page Content",
        "html_length": 5000,
        "cookie_banner": {"found": True, "selector": ".cookie-banner",
                           "acceptSelector": ".cookie-banner button", "text": "We use cookies"},
    }
    if evaluate_results:
        default_results.update(evaluate_results)

    async def mock_evaluate(script, *args):
        s = str(script)
        # Order matters: most specific patterns first
        if "const btns" in s:
            return default_results["buttons"]
        if "querySelectorAll('form')" in s:
            return default_results["forms"]
        if "querySelectorAll('a[href]')" in s:
            return default_results["links"]
        if "querySelectorAll('table')" in s:
            return default_results["tables"]
        if "input:not" in s:
            return default_results["inputs"]
        if "cookie" in s.lower() or "consent" in s.lower():
            return default_results["cookie_banner"]
        if "outerHTML.length" in s:
            return default_results["html_length"]
        if "body?.innerText" in s:
            return default_results["text"]
        if "document.querySelector" in s and "innerText" in s:
            return default_results["text"]
        if "localStorage" in s:
            return {"key1": "value1"}
        return None

    page.evaluate = mock_evaluate
    return page


class TestPageAnalyzer:
    @pytest.mark.asyncio
    async def test_analyze_basic(self):
        analyzer = PageAnalyzer()
        page = _mock_page()
        state = await analyzer.analyze(page)

        assert state.url == "https://example.com"
        assert state.title == "Example Page"
        assert state.is_loaded
        assert state.text_content == "Example Page Content"
        assert len(state.buttons) >= 1
        assert len(state.inputs) >= 1
        assert len(state.forms) >= 1

    @pytest.mark.asyncio
    async def test_analyze_forms(self):
        analyzer = PageAnalyzer()
        page = _mock_page()
        state = await analyzer.analyze(page)

        assert len(state.forms) == 1
        form = state.forms[0]
        assert form.method == "POST"
        assert form.action == "/login"
        assert len(form.fields) == 1
        assert form.fields[0].name == "email"
        assert form.fields[0].required

    @pytest.mark.asyncio
    async def test_analyze_tables(self):
        analyzer = PageAnalyzer()
        page = _mock_page()
        state = await analyzer.analyze(page)

        assert len(state.tables) == 1
        assert state.tables[0]["headers"] == ["Name", "Value"]

    @pytest.mark.asyncio
    async def test_detect_cookie_banner(self):
        analyzer = PageAnalyzer()
        page = _mock_page()
        result = await analyzer.detect_cookie_banner(page)
        assert result["found"]
        assert result["acceptSelector"]

    @pytest.mark.asyncio
    async def test_find_element_by_text(self):
        analyzer = PageAnalyzer()
        page = _mock_page()
        element = await analyzer.find_element(page, "submit")
        assert element is not None
        assert element.element_type == ElementType.BUTTON

    @pytest.mark.asyncio
    async def test_find_element_not_found(self):
        analyzer = PageAnalyzer()
        page = _mock_page()
        element = await analyzer.find_element(page, "nonexistent_xyz_123")
        assert element is None

    @pytest.mark.asyncio
    async def test_analyze_error_handling(self):
        analyzer = PageAnalyzer()
        page = AsyncMock()
        page.url = "about:blank"
        page.title = AsyncMock(side_effect=Exception("Title error"))
        page.evaluate = AsyncMock(side_effect=Exception("JS error"))

        state = await analyzer.analyze(page)
        assert state.url == "about:blank"
        assert len(state.errors) > 0

    def test_stats(self):
        analyzer = PageAnalyzer()
        assert analyzer.stats()["analysis_count"] == 0


# ============================================================================
# SessionManager Tests
# ============================================================================


class TestSessionSnapshot:
    def test_basic(self):
        snap = SessionSnapshot(session_id="test-1", domain="example.com")
        assert snap.created_at
        assert snap.updated_at

    def test_with_cookies(self):
        snap = SessionSnapshot(
            session_id="s1", domain="x.com",
            cookies=[{"name": "sid", "value": "abc123", "domain": "x.com"}],
        )
        assert len(snap.cookies) == 1


class TestSessionManager:
    @pytest.fixture
    def tmpdir(self):
        with tempfile.TemporaryDirectory() as d:
            yield d

    def test_save_and_load(self, tmpdir):
        mgr = SessionManager(storage_dir=tmpdir)
        snap = SessionSnapshot(
            session_id="test-1", domain="example.com",
            cookies=[{"name": "c", "value": "v"}],
            local_storage={"key": "val"},
            last_url="https://example.com",
        )
        assert mgr.save_session(snap)

        # Load from fresh instance
        mgr2 = SessionManager(storage_dir=tmpdir)
        loaded = mgr2.get_session("test-1")
        assert loaded is not None
        assert loaded.domain == "example.com"
        assert len(loaded.cookies) == 1
        assert loaded.local_storage["key"] == "val"

    def test_delete_session(self, tmpdir):
        mgr = SessionManager(storage_dir=tmpdir)
        snap = SessionSnapshot(session_id="del-1", domain="x.com")
        mgr.save_session(snap)
        assert mgr.delete_session("del-1")
        assert mgr.get_session("del-1") is None

    def test_list_sessions(self, tmpdir):
        mgr = SessionManager(storage_dir=tmpdir)
        mgr.save_session(SessionSnapshot(session_id="s1", domain="a.com"))
        mgr.save_session(SessionSnapshot(session_id="s2", domain="b.com"))
        sessions = mgr.list_sessions()
        assert len(sessions) == 2

    def test_nonexistent_session(self, tmpdir):
        mgr = SessionManager(storage_dir=tmpdir)
        assert mgr.get_session("nope") is None

    @pytest.mark.asyncio
    async def test_save_from_page(self, tmpdir):
        mgr = SessionManager(storage_dir=tmpdir)
        page = _mock_page()
        page.context = AsyncMock()
        page.context.cookies = AsyncMock(return_value=[
            {"name": "sid", "value": "xyz", "domain": "example.com",
             "path": "/", "expires": -1, "httpOnly": True, "secure": False, "sameSite": "Lax"},
        ])

        snap = await mgr.save_from_page(page, "page-session")
        assert snap.domain == "example.com"
        assert len(snap.cookies) == 1
        assert snap.visit_count == 1

    @pytest.mark.asyncio
    async def test_restore_to_context(self, tmpdir):
        mgr = SessionManager(storage_dir=tmpdir)
        snap = SessionSnapshot(
            session_id="restore-1", domain="x.com",
            cookies=[{"name": "c", "value": "v"}],
        )
        mgr.save_session(snap)

        context = AsyncMock()
        result = await mgr.restore_to_context(context, "restore-1")
        assert result
        context.add_cookies.assert_called_once()

    @pytest.mark.asyncio
    async def test_restore_empty(self, tmpdir):
        mgr = SessionManager(storage_dir=tmpdir)
        context = AsyncMock()
        result = await mgr.restore_to_context(context, "nonexistent")
        assert not result

    def test_stats(self, tmpdir):
        mgr = SessionManager(storage_dir=tmpdir)
        stats = mgr.stats()
        assert stats["total_sessions"] == 0

    def test_cleanup(self, tmpdir):
        mgr = SessionManager(storage_dir=tmpdir)
        snap = SessionSnapshot(session_id="old-1", domain="x.com")
        mgr.save_session(snap)
        # Overwrite file with old timestamp
        import json
        path = Path(tmpdir) / "old-1.json"
        data = json.loads(path.read_text())
        data["updated_at"] = "2020-01-01T00:00:00Z"
        path.write_text(json.dumps(data))
        # Force reload
        mgr._sessions.clear()
        mgr._loaded = False

        removed = mgr.cleanup(max_age_days=1)
        assert removed == 1


# ============================================================================
# BrowserAgent Tests (Playwright gemockt)
# ============================================================================


class TestBrowserAgentNoPlaywright:
    """Tests die ohne Playwright laufen (Availability-Check)."""

    def test_config_defaults(self):
        config = BrowserConfig()
        assert config.headless
        assert config.locale == "de-DE"

    def test_stats_not_running(self):
        from jarvis.browser.agent import BrowserAgent
        agent = BrowserAgent()
        stats = agent.stats()
        assert not stats["running"]
        assert stats["total_actions"] == 0

    def test_workflow_types(self):
        wf = BrowserWorkflow(
            name="Test",
            steps=[
                BrowserAction(action_type=ActionType.NAVIGATE, params={"url": "https://x.com"}),
                BrowserAction(action_type=ActionType.CLICK, params={"selector": "#btn"}),
            ],
        )
        assert len(wf.steps) == 2
        assert wf.status == WorkflowStatus.PENDING


class TestBrowserAgentMocked:
    """Tests mit gemocktem Playwright."""

    @pytest.fixture
    def mock_playwright(self):
        """Mock für Playwright async_playwright."""
        pw = AsyncMock()
        browser = AsyncMock()
        context = AsyncMock()
        page = _mock_page()

        # Chain: playwright → browser → context → page
        pw.chromium.launch = AsyncMock(return_value=browser)
        browser.new_context = AsyncMock(return_value=context)
        context.new_page = AsyncMock(return_value=page)
        context.cookies = AsyncMock(return_value=[])
        context.add_cookies = AsyncMock()
        context.on = MagicMock()

        # Page navigation
        response = AsyncMock()
        response.status = 200
        page.goto = AsyncMock(return_value=response)
        page.click = AsyncMock()
        page.fill = AsyncMock()
        page.select_option = AsyncMock()
        page.hover = AsyncMock()
        page.keyboard = AsyncMock()
        page.keyboard.press = AsyncMock()
        page.mouse = AsyncMock()
        page.mouse.wheel = AsyncMock()
        page.wait_for_selector = AsyncMock()
        page.wait_for_load_state = AsyncMock()
        page.screenshot = AsyncMock(return_value=b"\x89PNG fake image data")
        page.go_back = AsyncMock()
        page.go_forward = AsyncMock()
        page.reload = AsyncMock()
        page.close = AsyncMock()
        page.context = context

        return pw, browser, context, page

    @pytest.fixture
    def agent_with_mocks(self, mock_playwright):
        """BrowserAgent mit gemocktem Playwright (bereits gestartet)."""
        pw, browser, context, page = mock_playwright
        from jarvis.browser.agent import BrowserAgent

        agent = BrowserAgent()
        agent._playwright = pw
        agent._browser = browser
        agent._context = context
        agent._pages = [page]
        agent._active_page_idx = 0
        agent._running = True
        agent._start_time = __import__("time").time()
        return agent, page

    @pytest.mark.asyncio
    async def test_navigate(self, agent_with_mocks):
        agent, page = agent_with_mocks
        state = await agent.navigate("https://example.com")
        assert state.url == "https://example.com"
        assert state.title == "Example Page"
        assert state.is_loaded

    @pytest.mark.asyncio
    async def test_click(self, agent_with_mocks):
        agent, page = agent_with_mocks
        result = await agent.click("#submit")
        assert result.success
        page.click.assert_called_once()

    @pytest.mark.asyncio
    async def test_fill(self, agent_with_mocks):
        agent, page = agent_with_mocks
        result = await agent.fill("#email", "test@example.com")
        assert result.success

    @pytest.mark.asyncio
    async def test_select(self, agent_with_mocks):
        agent, page = agent_with_mocks
        result = await agent.select("#country", "DE")
        assert result.success

    @pytest.mark.asyncio
    async def test_hover(self, agent_with_mocks):
        agent, page = agent_with_mocks
        result = await agent.hover("#menu")
        assert result.success

    @pytest.mark.asyncio
    async def test_press_key(self, agent_with_mocks):
        agent, page = agent_with_mocks
        result = await agent.press_key("Enter")
        assert result.success

    @pytest.mark.asyncio
    async def test_scroll(self, agent_with_mocks):
        agent, page = agent_with_mocks
        result = await agent.scroll("down", 500)
        assert result.success

    @pytest.mark.asyncio
    async def test_wait_for(self, agent_with_mocks):
        agent, page = agent_with_mocks
        result = await agent.wait_for("#loaded")
        assert result.success

    @pytest.mark.asyncio
    async def test_execute_js(self, agent_with_mocks):
        agent, page = agent_with_mocks
        result = await agent.execute_js("return 42")
        assert result.success

    @pytest.mark.asyncio
    async def test_screenshot(self, agent_with_mocks):
        agent, page = agent_with_mocks
        result = await agent.screenshot()
        assert result.success
        assert result.screenshot_b64  # base64 encoded

    @pytest.mark.asyncio
    async def test_analyze_page(self, agent_with_mocks):
        agent, page = agent_with_mocks
        state = await agent.analyze_page()
        assert state.url == "https://example.com"
        assert len(state.buttons) >= 1

    @pytest.mark.asyncio
    async def test_extract_text(self, agent_with_mocks):
        agent, page = agent_with_mocks
        text = await agent.extract_text()
        assert text == "Example Page Content"

    @pytest.mark.asyncio
    async def test_extract_tables(self, agent_with_mocks):
        agent, page = agent_with_mocks
        tables = await agent.extract_tables()
        assert len(tables) >= 1

    @pytest.mark.asyncio
    async def test_extract_links(self, agent_with_mocks):
        agent, page = agent_with_mocks
        links = await agent.extract_links()
        assert isinstance(links, list)

    @pytest.mark.asyncio
    async def test_find_and_click(self, agent_with_mocks):
        agent, page = agent_with_mocks
        result = await agent.find_and_click("submit")
        assert result.success

    @pytest.mark.asyncio
    async def test_find_and_click_not_found(self, agent_with_mocks):
        agent, page = agent_with_mocks
        result = await agent.find_and_click("nonexistent_xyz_123")
        assert not result.success

    @pytest.mark.asyncio
    async def test_fill_form(self, agent_with_mocks):
        agent, page = agent_with_mocks
        results = await agent.fill_form({"email": "test@test.com"}, submit=True)
        assert len(results) >= 1

    # ── Tab Management ───────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_new_tab(self, agent_with_mocks):
        agent, page = agent_with_mocks
        agent._context.new_page = AsyncMock(return_value=_mock_page())
        result = await agent.new_tab("https://new.com")
        assert result.success
        assert agent.page_count == 2

    @pytest.mark.asyncio
    async def test_close_tab(self, agent_with_mocks):
        agent, page = agent_with_mocks
        page2 = _mock_page()
        page2.close = AsyncMock()
        agent._pages.append(page2)

        result = await agent.close_tab(1)
        assert result.success
        assert agent.page_count == 1

    @pytest.mark.asyncio
    async def test_close_last_tab(self, agent_with_mocks):
        agent, page = agent_with_mocks
        result = await agent.close_tab()
        assert not result.success  # cannot close last tab

    def test_switch_tab(self, agent_with_mocks):
        agent, page = agent_with_mocks
        agent._pages.append(_mock_page())
        result = agent.switch_tab(1)
        assert result.success
        assert agent._active_page_idx == 1

    def test_switch_tab_invalid(self, agent_with_mocks):
        agent, page = agent_with_mocks
        result = agent.switch_tab(99)
        assert not result.success

    @pytest.mark.asyncio
    async def test_max_tabs(self, agent_with_mocks):
        agent, page = agent_with_mocks
        agent._config.max_pages = 1
        result = await agent.new_tab()
        assert not result.success

    # ── Workflow Execution ───────────────────────────────────────

    @pytest.mark.asyncio
    async def test_execute_workflow(self, agent_with_mocks):
        agent, page = agent_with_mocks
        wf = BrowserWorkflow(
            name="Simple Test",
            steps=[
                BrowserAction(action_type=ActionType.NAVIGATE, params={"url": "https://x.com"}),
                BrowserAction(action_type=ActionType.WAIT, params={"seconds": 0.01}),
                BrowserAction(action_type=ActionType.EXTRACT_TEXT, params={"selector": "body"}),
            ],
        )
        result = await agent.execute_workflow(wf)
        assert result.status == WorkflowStatus.COMPLETED
        assert len(result.results) == 3
        assert result.success_rate == 1.0

    @pytest.mark.asyncio
    async def test_execute_workflow_with_failure(self, agent_with_mocks):
        agent, page = agent_with_mocks
        page.click = AsyncMock(side_effect=Exception("Element not found"))
        agent._config.screenshot_on_error = False  # avoid screenshot in test

        wf = BrowserWorkflow(
            name="Failing Test",
            max_retries=0,
            steps=[
                BrowserAction(action_type=ActionType.CLICK, params={"selector": "#nonexistent"}),
            ],
        )
        result = await agent.execute_workflow(wf)
        assert result.status == WorkflowStatus.FAILED

    # ── Error Handling ───────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_click_error(self, agent_with_mocks):
        agent, page = agent_with_mocks
        page.click = AsyncMock(side_effect=Exception("Timeout"))
        result = await agent.click("#x")
        assert not result.success
        assert "Timeout" in result.error

    @pytest.mark.asyncio
    async def test_fill_error(self, agent_with_mocks):
        agent, page = agent_with_mocks
        page.fill = AsyncMock(side_effect=Exception("Not found"))
        result = await agent.fill("#x", "value")
        assert not result.success

    def test_not_started_error(self):
        from jarvis.browser.agent import BrowserAgent
        agent = BrowserAgent()
        with pytest.raises(RuntimeError, match="not started"):
            asyncio.get_event_loop().run_until_complete(agent.click("#x"))

    # ── Stats ────────────────────────────────────────────────────

    def test_stats(self, agent_with_mocks):
        agent, page = agent_with_mocks
        stats = agent.stats()
        assert stats["running"]
        assert stats["tab_count"] == 1
        assert stats["headless"]

    # ── Stop ─────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_stop(self, agent_with_mocks):
        agent, page = agent_with_mocks
        agent._context.close = AsyncMock()
        agent._browser.close = AsyncMock()
        agent._playwright.stop = AsyncMock()
        await agent.stop()
        assert not agent.is_running


# ============================================================================
# MCP Tools Registration Tests
# ============================================================================


class TestBrowserUseTools:
    def test_registration(self):
        """Prüft dass alle MCP-Tools registriert werden."""
        from jarvis.browser.tools import register_browser_use_tools

        mcp_mock = MagicMock()
        registered_tools: dict[str, Any] = {}

        def mock_register(name: str, **kwargs: Any) -> None:
            registered_tools[name] = kwargs

        mcp_mock.register_tool = mock_register

        agent = register_browser_use_tools(mcp_mock)
        assert agent is not None
        assert len(registered_tools) >= 9

        expected = [
            "browser_navigate", "browser_click", "browser_fill",
            "browser_fill_form", "browser_screenshot", "browser_extract",
            "browser_analyze", "browser_execute_js", "browser_tab",
        ]
        for tool_name in expected:
            assert tool_name in registered_tools, f"Missing tool: {tool_name}"


# ============================================================================
# BrowserAgent — Vision-Integration
# ============================================================================


class TestBrowserAgentVision:
    """Tests für die Vision-Integration im BrowserAgent."""

    def test_init_without_vision(self) -> None:
        """Bestehender Code funktioniert ohne vision_analyzer."""
        from jarvis.browser.agent import BrowserAgent
        agent = BrowserAgent()
        assert agent._vision is None

    def test_init_with_vision(self) -> None:
        """vision_analyzer wird korrekt gespeichert."""
        from jarvis.browser.agent import BrowserAgent
        mock_vision = MagicMock()
        agent = BrowserAgent(vision_analyzer=mock_vision)
        assert agent._vision is mock_vision

    def test_stats_without_vision(self) -> None:
        """Stats ohne Vision enthalten kein 'vision' Feld."""
        from jarvis.browser.agent import BrowserAgent
        agent = BrowserAgent()
        s = agent.stats()
        assert "vision" not in s

    def test_stats_with_vision(self) -> None:
        """Stats mit Vision enthalten Vision-Stats."""
        from jarvis.browser.agent import BrowserAgent
        mock_vision = MagicMock()
        mock_vision.stats.return_value = {"enabled": True, "calls": 5}
        agent = BrowserAgent(vision_analyzer=mock_vision)
        s = agent.stats()
        assert "vision" in s
        assert s["vision"]["enabled"] is True
        assert s["vision"]["calls"] == 5

    @pytest.mark.asyncio
    async def test_analyze_page_without_vision(self) -> None:
        """analyze_page_with_vision funktioniert auch ohne Vision."""
        from jarvis.browser.agent import BrowserAgent

        agent = BrowserAgent()
        agent._running = True

        # Mock page + analyzer
        mock_page = AsyncMock()
        mock_page.url = "https://example.com"
        mock_page.title = AsyncMock(return_value="Example")
        agent._pages = [mock_page]
        agent._active_page_idx = 0

        mock_state = MagicMock()
        mock_state.to_summary.return_value = "DOM: Example page"
        agent._analyzer.analyze = AsyncMock(return_value=mock_state)

        result = await agent.analyze_page_with_vision()
        assert result["dom"] == "DOM: Example page"
        assert result["vision"] == ""
        assert result["combined"] == "DOM: Example page"

    @pytest.mark.asyncio
    async def test_analyze_page_with_vision(self) -> None:
        """analyze_page_with_vision kombiniert DOM + Vision + page_content."""
        from jarvis.browser.agent import BrowserAgent
        from jarvis.browser.vision import VisionAnalysisResult

        mock_vision = AsyncMock()
        mock_vision.is_enabled = True
        mock_vision.analyze_screenshot = AsyncMock(return_value=VisionAnalysisResult(
            success=True, description="Login-Formular sichtbar"
        ))

        agent = BrowserAgent(vision_analyzer=mock_vision)
        agent._running = True

        # Mock page + analyzer + screenshot
        mock_page = AsyncMock()
        mock_page.url = "https://example.com"
        mock_page.title = AsyncMock(return_value="Example")
        mock_page.screenshot = AsyncMock(return_value=b"\x89PNG")
        mock_page.evaluate = AsyncMock(return_value="<div>Login Form</div>")
        agent._pages = [mock_page]
        agent._active_page_idx = 0

        mock_state = MagicMock()
        mock_state.to_summary.return_value = "DOM Summary"
        agent._analyzer.analyze = AsyncMock(return_value=mock_state)

        result = await agent.analyze_page_with_vision()
        assert result["dom"] == "DOM Summary"
        assert "Login-Formular" in result["vision"]
        assert "## DOM-Analyse" in result["combined"]
        assert "## Vision-Analyse" in result["combined"]

        # Prüfe dass page_content an analyze_screenshot übergeben wurde
        call_kwargs = mock_vision.analyze_screenshot.call_args
        assert call_kwargs.kwargs.get("page_content") == "<div>Login Form</div>"

    @pytest.mark.asyncio
    async def test_find_and_click_text_match_first(self) -> None:
        """Text-Match gewinnt — Vision wird nicht aufgerufen."""
        from jarvis.browser.agent import BrowserAgent
        from jarvis.browser.types import ActionResult

        mock_vision = AsyncMock()
        mock_vision.is_enabled = True

        agent = BrowserAgent(vision_analyzer=mock_vision)
        agent._running = True

        # Mock successful text-based find_and_click
        mock_page = AsyncMock()
        mock_page.url = "https://example.com"
        agent._pages = [mock_page]
        agent._active_page_idx = 0

        success_result = ActionResult(action_id="test", success=True, data={"clicked": True})
        agent.find_and_click = AsyncMock(return_value=success_result)

        result = await agent.find_and_click_with_vision("Login")
        assert result.success is True
        mock_vision.find_element_by_vision.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_find_and_click_vision_fallback(self) -> None:
        """Vision-Fallback wenn Text-Match fehlschlägt, mit page_content."""
        from jarvis.browser.agent import BrowserAgent
        from jarvis.browser.types import ActionResult
        from jarvis.browser.vision import VisionAnalysisResult

        mock_vision = AsyncMock()
        mock_vision.is_enabled = True
        mock_vision.find_element_by_vision = AsyncMock(return_value=VisionAnalysisResult(
            success=True, description="Button oben rechts, blau"
        ))

        agent = BrowserAgent(vision_analyzer=mock_vision)
        agent._running = True

        mock_page = AsyncMock()
        mock_page.url = "https://example.com"
        mock_page.screenshot = AsyncMock(return_value=b"\x89PNG")
        mock_page.evaluate = AsyncMock(return_value="<button>Absenden</button>")
        agent._pages = [mock_page]
        agent._active_page_idx = 0

        fail_result = ActionResult(action_id="test", success=False, data={}, error="Not found")
        agent.find_and_click = AsyncMock(return_value=fail_result)

        result = await agent.find_and_click_with_vision("Absenden")
        assert "vision_hint" in result.data
        assert "oben rechts" in result.data["vision_hint"]

        # Prüfe dass page_content übergeben wurde
        call_kwargs = mock_vision.find_element_by_vision.call_args
        assert call_kwargs.kwargs.get("page_content") == "<button>Absenden</button>"
