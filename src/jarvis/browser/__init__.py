"""Jarvis Browser-Use v17 — Autonome Browser-Automatisierung.

Ermöglicht dem Agent das selbstständige Navigieren, Lesen und
Interagieren mit Webseiten. Headless Chromium über Playwright.

OPTIONAL: pip install playwright && playwright install chromium

Kern-Komponenten:
  - BrowserAgent:     Autonomer Browser-Controller
  - PageAnalyzer:     Seiten-Analyse (Formulare, Links, Tabellen)
  - SessionManager:   Cookie/State-Persistierung
  - register_browser_use_tools(): MCP-Tool-Integration
"""

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
from jarvis.browser.agent import BrowserAgent
from jarvis.browser.tools import register_browser_use_tools

__all__ = [
    "ActionResult", "ActionType", "BrowserAction", "BrowserConfig",
    "BrowserWorkflow", "ElementInfo", "ElementType", "ExtractionMode",
    "FormField", "FormInfo", "PageState", "WorkflowStatus",
    "PageAnalyzer", "SessionManager", "SessionSnapshot",
    "BrowserAgent", "register_browser_use_tools",
]
