"""Browser-Use Types -- v17.

Datenmodelle für autonome Browser-Automatisierung.
Trennung von Low-Level (Playwright) und High-Level (Agent-Steuerung).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ── Action Types ─────────────────────────────────────────────────

class ActionType(str, Enum):
    """Browser-Aktionen die der Agent ausführen kann."""
    NAVIGATE = "navigate"
    CLICK = "click"
    FILL = "fill"
    SELECT = "select"
    SCROLL = "scroll"
    SCREENSHOT = "screenshot"
    EXTRACT_TEXT = "extract_text"
    EXTRACT_TABLE = "extract_table"
    EXTRACT_LINKS = "extract_links"
    EXECUTE_JS = "execute_js"
    WAIT = "wait"
    WAIT_FOR = "wait_for"
    GO_BACK = "go_back"
    GO_FORWARD = "go_forward"
    REFRESH = "refresh"
    NEW_TAB = "new_tab"
    CLOSE_TAB = "close_tab"
    SWITCH_TAB = "switch_tab"
    HOVER = "hover"
    PRESS_KEY = "press_key"
    UPLOAD_FILE = "upload_file"
    DOWNLOAD = "download"
    ACCEPT_DIALOG = "accept_dialog"
    DISMISS_DIALOG = "dismiss_dialog"


class ElementType(str, Enum):
    """Interaktive Element-Typen auf einer Seite."""
    LINK = "link"
    BUTTON = "button"
    INPUT = "input"
    TEXTAREA = "textarea"
    SELECT = "select"
    CHECKBOX = "checkbox"
    RADIO = "radio"
    FILE_INPUT = "file_input"
    IMAGE = "image"
    FORM = "form"
    TABLE = "table"
    HEADING = "heading"
    IFRAME = "iframe"
    OTHER = "other"


class ExtractionMode(str, Enum):
    """Wie Inhalte extrahiert werden."""
    TEXT = "text"            # Nur sichtbarer Text
    HTML = "html"            # Rohes HTML
    MARKDOWN = "markdown"    # Strukturiertes Markdown
    TABLES = "tables"        # Tabellen als JSON
    LINKS = "links"          # Alle Links
    FORMS = "forms"          # Formular-Felder
    STRUCTURED = "structured"  # Intelligente Struktur-Erkennung


# ── Element Info ─────────────────────────────────────────────────

@dataclass
class ElementInfo:
    """Beschreibt ein interaktives Element auf der Seite."""
    selector: str
    element_type: ElementType
    text: str = ""
    value: str = ""
    name: str = ""
    placeholder: str = ""
    href: str = ""
    aria_label: str = ""
    is_visible: bool = True
    is_enabled: bool = True
    is_required: bool = False
    position: dict[str, int] = field(default_factory=dict)  # x, y, width, height
    attributes: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "selector": self.selector, "type": self.element_type.value,
            "text": self.text, "value": self.value, "name": self.name,
            "placeholder": self.placeholder, "href": self.href,
            "aria_label": self.aria_label, "visible": self.is_visible,
            "enabled": self.is_enabled, "required": self.is_required,
        }

    @property
    def label(self) -> str:
        """Menschenlesbare Beschreibung des Elements."""
        parts = []
        if self.text:
            parts.append(self.text[:80])
        elif self.aria_label:
            parts.append(self.aria_label[:80])
        elif self.placeholder:
            parts.append(self.placeholder[:80])
        elif self.name:
            parts.append(self.name)
        if not parts:
            parts.append(self.selector[:60])
        return f"[{self.element_type.value}] {' '.join(parts)}"


# ── Form Info ────────────────────────────────────────────────────

@dataclass
class FormField:
    """Ein Feld innerhalb eines Formulars."""
    name: str
    field_type: str  # text, email, password, number, date, select, etc.
    label: str = ""
    value: str = ""
    placeholder: str = ""
    required: bool = False
    options: list[str] = field(default_factory=list)  # für select
    selector: str = ""

    def to_dict(self) -> dict[str, Any]:
        r: dict[str, Any] = {
            "name": self.name, "type": self.field_type,
            "label": self.label, "required": self.required,
        }
        if self.value:
            r["value"] = self.value
        if self.placeholder:
            r["placeholder"] = self.placeholder
        if self.options:
            r["options"] = self.options
        return r


@dataclass
class FormInfo:
    """Beschreibt ein erkanntes Formular."""
    action: str = ""
    method: str = "GET"
    fields: list[FormField] = field(default_factory=list)
    submit_selector: str = ""
    selector: str = ""
    name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action, "method": self.method, "name": self.name,
            "fields": [f.to_dict() for f in self.fields],
            "submit_selector": self.submit_selector,
        }


# ── Page State ───────────────────────────────────────────────────

@dataclass
class PageState:
    """Snapshot des aktuellen Seiten-Zustands."""
    url: str = ""
    title: str = ""
    text_content: str = ""
    html_length: int = 0
    load_time_ms: int = 0
    status_code: int = 0
    is_loaded: bool = False
    has_dialog: bool = False
    dialog_message: str = ""
    # Interaktive Elemente
    links: list[ElementInfo] = field(default_factory=list)
    buttons: list[ElementInfo] = field(default_factory=list)
    inputs: list[ElementInfo] = field(default_factory=list)
    forms: list[FormInfo] = field(default_factory=list)
    tables: list[dict[str, Any]] = field(default_factory=list)
    # Tab-Info
    tab_index: int = 0
    tab_count: int = 1
    # Timing
    timestamp: str = ""
    # Errors
    errors: list[str] = field(default_factory=list)
    console_messages: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url, "title": self.title,
            "text_length": len(self.text_content),
            "html_length": self.html_length,
            "load_time_ms": self.load_time_ms,
            "status_code": self.status_code,
            "is_loaded": self.is_loaded,
            "links_count": len(self.links),
            "buttons_count": len(self.buttons),
            "inputs_count": len(self.inputs),
            "forms_count": len(self.forms),
            "tables_count": len(self.tables),
            "tab_index": self.tab_index,
            "tab_count": self.tab_count,
            "errors": self.errors,
        }

    def to_summary(self, max_text: int = 2000) -> str:
        """Kompakte Zusammenfassung für LLM-Kontext."""
        parts = [
            f"URL: {self.url}",
            f"Title: {self.title}",
        ]
        if self.errors:
            parts.append(f"Errors: {', '.join(self.errors[:3])}")
        if self.forms:
            parts.append(f"Forms: {len(self.forms)}")
            for form in self.forms[:2]:
                fields_desc = ", ".join(f.label or f.name for f in form.fields[:5])
                parts.append(f"  Form '{form.name}': {fields_desc}")
        if self.links:
            parts.append(f"Links: {len(self.links)}")
        if self.buttons:
            parts.append(f"Buttons: {len(self.buttons)}")
            btn_labels = [b.text[:30] for b in self.buttons[:5] if b.text]
            if btn_labels:
                parts.append(f"  Labels: {', '.join(btn_labels)}")
        if self.inputs:
            parts.append(f"Inputs: {len(self.inputs)}")
        if self.text_content:
            truncated = self.text_content[:max_text]
            if len(self.text_content) > max_text:
                truncated += f"\n... ({len(self.text_content) - max_text} chars truncated)"
            parts.append(f"\nContent:\n{truncated}")
        return "\n".join(parts)


# ── Browser Action ───────────────────────────────────────────────

@dataclass
class BrowserAction:
    """Eine einzelne Browser-Aktion."""
    action_type: ActionType
    params: dict[str, Any] = field(default_factory=dict)
    description: str = ""  # Menschenlesbare Beschreibung
    action_id: str = ""

    def __post_init__(self) -> None:
        if not self.action_id:
            self.action_id = uuid.uuid4().hex[:10]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.action_id, "action": self.action_type.value,
            "params": self.params, "description": self.description,
        }


@dataclass
class ActionResult:
    """Ergebnis einer ausgeführten Browser-Aktion."""
    action_id: str
    success: bool
    data: Any = None
    error: str = ""
    duration_ms: int = 0
    page_changed: bool = False
    new_url: str = ""
    screenshot_b64: str = ""

    def to_dict(self) -> dict[str, Any]:
        r: dict[str, Any] = {
            "action_id": self.action_id, "success": self.success,
            "duration_ms": self.duration_ms,
        }
        if self.error:
            r["error"] = self.error
        if self.data is not None:
            r["data"] = self.data
        if self.page_changed:
            r["page_changed"] = True
            r["new_url"] = self.new_url
        return r


# ── Workflow ─────────────────────────────────────────────────────

class WorkflowStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


@dataclass
class BrowserWorkflow:
    """Multi-Step Browser-Automatisierung."""
    workflow_id: str = ""
    name: str = ""
    description: str = ""
    steps: list[BrowserAction] = field(default_factory=list)
    results: list[ActionResult] = field(default_factory=list)
    status: WorkflowStatus = WorkflowStatus.PENDING
    current_step: int = 0
    max_retries: int = 2
    created_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.workflow_id:
            self.workflow_id = uuid.uuid4().hex[:12]
        if not self.created_at:
            self.created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.workflow_id, "name": self.name, "status": self.status.value,
            "steps": len(self.steps), "current_step": self.current_step,
            "completed": len(self.results),
            "success_rate": self.success_rate,
        }

    @property
    def success_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.success) / len(self.results)

    @property
    def is_complete(self) -> bool:
        return self.status in (WorkflowStatus.COMPLETED, WorkflowStatus.FAILED,
                               WorkflowStatus.CANCELED)


# ── Session Config ───────────────────────────────────────────────

@dataclass
class BrowserConfig:
    """Konfiguration für Browser-Sessions."""
    headless: bool = True
    viewport_width: int = 1280
    viewport_height: int = 720
    timeout_ms: int = 30000
    user_agent: str = ""
    locale: str = "de-DE"
    timezone: str = "Europe/Berlin"
    persist_cookies: bool = True
    cookie_dir: str = ""
    screenshot_on_error: bool = True
    max_pages: int = 5
    block_images: bool = False
    block_ads: bool = True
    proxy: str = ""
    # Vision-Integration
    vision_enabled: bool = False
    vision_model: str = ""       # z.B. "gpt-4o", "claude-sonnet-4-20250514", "llava:13b"
    vision_backend: str = "ollama"

    def to_dict(self) -> dict[str, Any]:
        return {
            "headless": self.headless,
            "viewport": f"{self.viewport_width}x{self.viewport_height}",
            "timeout_ms": self.timeout_ms, "locale": self.locale,
            "timezone": self.timezone, "persist_cookies": self.persist_cookies,
            "max_pages": self.max_pages,
        }
