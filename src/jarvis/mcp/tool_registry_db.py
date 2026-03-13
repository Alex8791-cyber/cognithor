"""Datenbank-gestütztes Tool-Registry für dynamische Prompt-Generierung.

Speichert Tool-Metadaten (lokalisierte Beschreibungen, Beispiele,
Rollen-Zuordnungen) in SQLite und generiert kontextspezifische
Prompt-Abschnitte pro Agentenrolle und Sprache.

Verwendet von ``gateway._sync_core_inventory()`` um den statischen
Markdown-Dump durch dynamische, rollenbasierte Tool-Abschnitte zu ersetzen.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from jarvis.utils.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

    from jarvis.mcp.client import JarvisMCPClient

log = get_logger(__name__)

__all__ = [
    "DEFAULT_EXAMPLES",
    "TOOL_CATEGORIES",
    "TOOL_ROLE_DEFAULTS",
    "ToolInfo",
    "ToolRegistryDB",
    "deduplicate_procedures",
]

# ============================================================================
# Datenklassen
# ============================================================================


@dataclass
class ToolInfo:
    """Metadaten eines registrierten Tools."""

    name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)
    example_input: str = ""
    example_output: str = ""
    category: str = "other"
    agent_roles: list[str] = field(default_factory=lambda: ["all"])


# ============================================================================
# Konstanten: Rollen-Zuordnungen
# ============================================================================

TOOL_ROLE_DEFAULTS: dict[str, set[str]] = {
    "planner": {
        "search_memory", "get_core_memory", "get_recent_episodes",
        "search_procedures", "knowledge_synthesize", "knowledge_gaps",
        "knowledge_contradictions", "knowledge_timeline", "vault_search",
        "vault_read", "vault_list", "web_search", "search_and_read",
        "web_news_search", "memory_stats", "list_skills",
    },
    "executor": {
        "read_file", "write_file", "edit_file", "list_directory",
        "exec_command", "run_python", "analyze_code",
        "web_fetch", "web_search", "search_and_read",
        "save_to_memory", "add_entity", "add_relation", "get_entity",
        "vault_save", "vault_update", "vault_link",
        "document_export", "analyze_document",
        "media_extract_text", "media_analyze_image", "media_tts",
        "media_resize_image", "media_convert_audio", "media_transcribe_audio",
        "read_pdf", "read_docx", "read_ppt",
        "http_request", "create_skill", "record_procedure_usage",
        "git_status", "git_diff", "git_log", "git_commit", "git_branch",
        "search_files", "find_in_files", "find_and_replace",
        "db_query", "db_schema", "db_execute", "db_connect",
        "create_chart", "create_table_image", "chart_from_csv",
        "email_read_inbox", "email_search", "email_send", "email_summarize",
        "calendar_today", "calendar_upcoming", "calendar_create_event",
        "calendar_check_availability",
        "set_reminder", "list_reminders", "send_notification",
        "get_clipboard", "set_clipboard", "screenshot_desktop", "screenshot_region",
        "docker_ps", "docker_logs", "docker_inspect", "docker_run", "docker_stop",
        "api_list", "api_connect", "api_call", "api_disconnect",
    },
    "browser": {
        "browser_navigate", "browser_click", "browser_fill", "browser_fill_form",
        "browser_extract", "browser_analyze", "browser_screenshot",
        "browser_execute_js", "browser_key", "browser_tab",
        "browser_vision_analyze", "browser_vision_find", "browser_vision_screenshot",
    },
    "researcher": {
        "web_search", "web_news_search", "web_fetch", "search_and_read",
        "search_memory", "vault_search", "vault_read",
        "knowledge_synthesize", "knowledge_gaps", "knowledge_contradictions",
        "knowledge_timeline", "get_entity", "search_procedures",
    },
}

# ============================================================================
# Konstanten: Tool-Kategorien (name-prefix → category)
# ============================================================================

TOOL_CATEGORIES: dict[str, str] = {
    "read_file": "filesystem",
    "write_file": "filesystem",
    "edit_file": "filesystem",
    "list_directory": "filesystem",
    "search_files": "filesystem",
    "find_in_files": "filesystem",
    "find_and_replace": "filesystem",
    "exec_command": "shell",
    "run_python": "shell",
    "web_search": "web",
    "web_news_search": "web",
    "web_fetch": "web",
    "search_and_read": "web",
    "http_request": "web",
    "search_memory": "memory",
    "save_to_memory": "memory",
    "get_core_memory": "memory",
    "get_recent_episodes": "memory",
    "search_procedures": "memory",
    "memory_stats": "memory",
    "record_procedure_usage": "memory",
    "add_entity": "memory",
    "add_relation": "memory",
    "get_entity": "memory",
    "vault_save": "vault",
    "vault_update": "vault",
    "vault_link": "vault",
    "vault_search": "vault",
    "vault_read": "vault",
    "vault_list": "vault",
    "knowledge_synthesize": "knowledge",
    "knowledge_gaps": "knowledge",
    "knowledge_contradictions": "knowledge",
    "knowledge_timeline": "knowledge",
    "document_export": "documents",
    "analyze_document": "documents",
    "read_pdf": "documents",
    "read_docx": "documents",
    "read_ppt": "documents",
    "media_extract_text": "media",
    "media_analyze_image": "media",
    "media_tts": "media",
    "media_resize_image": "media",
    "media_convert_audio": "media",
    "media_transcribe_audio": "media",
    "analyze_code": "code",
    "git_status": "git",
    "git_diff": "git",
    "git_log": "git",
    "git_commit": "git",
    "git_branch": "git",
    "db_query": "database",
    "db_schema": "database",
    "db_execute": "database",
    "db_connect": "database",
    "create_chart": "visualization",
    "create_table_image": "visualization",
    "chart_from_csv": "visualization",
    "email_read_inbox": "email",
    "email_search": "email",
    "email_send": "email",
    "email_summarize": "email",
    "calendar_today": "calendar",
    "calendar_upcoming": "calendar",
    "calendar_create_event": "calendar",
    "calendar_check_availability": "calendar",
    "set_reminder": "notifications",
    "list_reminders": "notifications",
    "send_notification": "notifications",
    "get_clipboard": "desktop",
    "set_clipboard": "desktop",
    "screenshot_desktop": "desktop",
    "screenshot_region": "desktop",
    "docker_ps": "docker",
    "docker_logs": "docker",
    "docker_inspect": "docker",
    "docker_run": "docker",
    "docker_stop": "docker",
    "api_list": "api",
    "api_connect": "api",
    "api_call": "api",
    "api_disconnect": "api",
    "create_skill": "skills",
    "list_skills": "skills",
    "install_community_skill": "skills",
    "search_community_skills": "skills",
    "report_skill": "skills",
}

# Kategorie-Labels nach Sprache
_CATEGORY_LABELS: dict[str, dict[str, str]] = {
    "de": {
        "filesystem": "Dateisystem",
        "shell": "Shell & Ausfuehrung",
        "web": "Web & Recherche",
        "memory": "Gedaechtnis",
        "vault": "Vault (Dokumente)",
        "knowledge": "Wissenssynthese",
        "documents": "Dokumente",
        "media": "Medien",
        "code": "Code-Analyse",
        "git": "Git",
        "database": "Datenbank",
        "visualization": "Visualisierung",
        "email": "E-Mail",
        "calendar": "Kalender",
        "notifications": "Benachrichtigungen",
        "desktop": "Desktop",
        "docker": "Docker",
        "api": "API-Integration",
        "skills": "Skills",
        "browser": "Browser",
        "other": "Sonstige",
    },
    "en": {
        "filesystem": "Filesystem",
        "shell": "Shell & Execution",
        "web": "Web & Research",
        "memory": "Memory",
        "vault": "Vault (Documents)",
        "knowledge": "Knowledge Synthesis",
        "documents": "Documents",
        "media": "Media",
        "code": "Code Analysis",
        "git": "Git",
        "database": "Database",
        "visualization": "Visualization",
        "email": "Email",
        "calendar": "Calendar",
        "notifications": "Notifications",
        "desktop": "Desktop",
        "docker": "Docker",
        "api": "API Integration",
        "skills": "Skills",
        "browser": "Browser",
        "other": "Other",
    },
    "zh": {
        "filesystem": "文件系统",
        "shell": "Shell 与执行",
        "web": "网络与研究",
        "memory": "记忆",
        "vault": "保险库 (文档)",
        "knowledge": "知识综合",
        "documents": "文档",
        "media": "媒体",
        "code": "代码分析",
        "git": "Git",
        "database": "数据库",
        "visualization": "可视化",
        "email": "电子邮件",
        "calendar": "日历",
        "notifications": "通知",
        "desktop": "桌面",
        "docker": "Docker",
        "api": "API 集成",
        "skills": "技能",
        "browser": "浏览器",
        "other": "其他",
    },
}

# ============================================================================
# Konstanten: Standard-Beispiele
# ============================================================================

DEFAULT_EXAMPLES: dict[str, tuple[str, str]] = {
    # ---- Web & Research ----
    "web_search": (
        '{"query": "OpenAI GPT-5 release", "num_results": 5}',
        '{"results": [{"title": "GPT-5 Launch Announced",'
        ' "url": "https://openai.com/blog/gpt5",'
        ' "snippet": "OpenAI announced GPT-5 with..."}]}',
    ),
    "web_news_search": (
        '{"query": "AI regulation EU",'
        ' "num_results": 3, "timelimit": "7d"}',
        '{"results": [{"title": "EU AI Act Takes Effect",'
        ' "url": "https://reuters.com/...",'
        ' "summary": "The European Union AI Act...",'
        ' "source": "Reuters", "date": "2026-03-10"}]}',
    ),
    "search_and_read": (
        '{"query": "Python asyncio tutorial beginners"}',
        '{"url": "https://realpython.com/async-io-python/",'
        ' "title": "Async IO in Python",'
        ' "text": "asyncio is a library to write'
        ' concurrent code using async/await..."}',
    ),
    "web_fetch": (
        '{"url": "https://api.github.com/repos/python/cpython"}',
        '{"status": 200, "content_type": "application/json",'
        ' "text": "{\\"full_name\\": \\"python/cpython\\",'
        ' \\"stargazers_count\\": 65000}"}',
    ),
    "http_request": (
        '{"url": "https://api.example.com/data",'
        ' "method": "POST",'
        ' "headers": {"Content-Type": "application/json"},'
        ' "body": "{\\"key\\": \\"value\\"}"}',
        '{"status": 200,'
        ' "headers": {"content-type": "application/json"},'
        ' "body": "{\\"id\\": 42, \\"status\\": \\"created\\"}"}',
    ),
    # ---- Filesystem ----
    "read_file": (
        '{"path": "/home/user/config.yaml"}',
        '{"content": "server:\\n  host: localhost\\n'
        '  port: 8080\\n", "size": 42, "encoding": "utf-8"}',
    ),
    "write_file": (
        '{"path": "/tmp/output.txt", "content": "Hello World"}',
        '{"written": true, "bytes": 11, "path": "/tmp/output.txt"}',
    ),
    "edit_file": (
        '{"path": "/tmp/app.py",'
        ' "old_text": "DEBUG = True",'
        ' "new_text": "DEBUG = False"}',
        '{"edited": true, "replacements": 1}',
    ),
    "list_directory": (
        '{"path": "/home/user/project", "recursive": false}',
        '{"entries": [{"name": "src", "type": "directory"},'
        ' {"name": "README.md", "type": "file",'
        ' "size": 1024}]}',
    ),
    # ---- Shell ----
    "exec_command": (
        '{"command": "echo hello && date", "timeout": 30}',
        '{"stdout": "hello\\n2026-03-13\\n",'
        ' "stderr": "", "exit_code": 0}',
    ),
    "run_python": (
        '{"code": "import math\\nprint(math.pi)"}',
        '{"stdout": "3.141592653589793\\n",'
        ' "stderr": "", "exit_code": 0}',
    ),
    # ---- Memory ----
    "search_memory": (
        '{"query": "meeting notes project alpha"}',
        '{"results": [{"text": "Project Alpha kickoff:'
        ' deadline March 30...",'
        ' "score": 0.92, "tags": ["meeting", "project"]}]}',
    ),
    "save_to_memory": (
        '{"text": "Dentist appointment March 20 at 2pm",'
        ' "tags": ["appointment", "health"]}',
        '{"saved": true, "id": "mem_abc123"}',
    ),
    "get_core_memory": (
        "{}",
        '{"content": "# CORE.md\\n## Identity\\nName: Jarvis..."}',
    ),
    "get_recent_episodes": (
        '{"days": 3}',
        '{"episodes": [{"date": "2026-03-12",'
        ' "summary": "Helped user debug Python script..."},'
        ' {"date": "2026-03-11",'
        ' "summary": "Created weekly report..."}]}',
    ),
    # ---- Vault ----
    "vault_save": (
        '{"title": "Meeting Notes Q1",'
        ' "content": "# Q1 Review\\n- Revenue up 15%..."}',
        '{"id": "vault_xyz789",'
        ' "title": "Meeting Notes Q1",'
        ' "created": "2026-03-13T10:00:00Z"}',
    ),
    "vault_search": (
        '{"query": "rental contract"}',
        '{"results": [{"id": "vault_abc",'
        ' "title": "Rental_Contract_2025.pdf",'
        ' "score": 0.88, "snippet": "Monthly rent: 950 EUR..."}]}',
    ),
    # ---- Knowledge ----
    "knowledge_synthesize": (
        '{"topic": "machine learning basics", "language": "en"}',
        '{"synthesis": "Machine learning is a subset of AI'
        ' that enables systems to learn from data...",'
        ' "sources": 5, "confidence": 0.85}',
    ),
    "knowledge_gaps": (
        '{"topic": "quantum computing", "language": "en"}',
        '{"completeness": 0.3,'
        ' "known": ["basic concepts", "qubit definition"],'
        ' "gaps": ["error correction",'
        ' "hardware implementations"],'
        ' "research_suggestions":'
        ' ["Search for quantum error correction 2025"]}',
    ),
    "knowledge_contradictions": (
        '{"topic": "project deadline", "language": "en"}',
        '{"contradictions": [{"claim_a": "Deadline is March 30",'
        ' "source_a": "memory",'
        ' "claim_b": "Deadline extended to April 15",'
        ' "source_b": "vault"}]}',
    ),
    # ---- Documents ----
    "analyze_document": (
        '{"path": "/tmp/report.pdf"}',
        '{"pages": 12,'
        ' "summary": "Q4 quarterly report covering'
        ' revenue, expenses...",'
        ' "key_figures": ["Revenue: 2.3M EUR",'
        ' "Growth: 15%"]}',
    ),
    "document_export": (
        '{"format": "pdf",'
        ' "content": "# Report\\nContent here...",'
        ' "output_path": "/tmp/report.pdf"}',
        '{"path": "/tmp/report.pdf",'
        ' "format": "pdf", "pages": 3, "size": 45200}',
    ),
    # ---- Browser ----
    "browser_navigate": (
        '{"url": "https://example.com/login"}',
        '{"title": "Login - Example",'
        ' "status": 200, "url": "https://example.com/login"}',
    ),
    "browser_click": (
        '{"selector": "#submit-btn"}',
        '{"clicked": true, "element": "button", "text": "Submit"}',
    ),
    "browser_fill": (
        '{"selector": "#email", "value": "user@example.com"}',
        '{"filled": true, "selector": "#email"}',
    ),
    # ---- Git ----
    "git_status": (
        "{}",
        '{"branch": "feature/auth",'
        ' "modified": ["src/auth.py"],'
        ' "untracked": ["tests/test_auth.py"], "staged": []}',
    ),
    "git_diff": (
        '{"file": "src/auth.py"}',
        '{"diff": "- old_line\\n+ new_line",'
        ' "additions": 5, "deletions": 2}',
    ),
    # ---- Visualization ----
    "create_chart": (
        '{"type": "bar",'
        ' "data": {"Q1": 150, "Q2": 230, "Q3": 180, "Q4": 310},'
        ' "title": "Revenue by Quarter"}',
        '{"path": "/tmp/chart_revenue.png",'
        ' "format": "png", "size": 24500}',
    ),
    # ---- Email ----
    "email_send": (
        '{"to": "colleague@company.com",'
        ' "subject": "Meeting Tomorrow",'
        ' "body": "Hi, reminder about our 2pm meeting."}',
        '{"sent": true, "message_id": "msg_20260313_001"}',
    ),
    # ---- Docker ----
    "docker_ps": (
        "{}",
        '{"containers": [{"id": "a1b2c3",'
        ' "image": "nginx:latest",'
        ' "status": "Up 2 hours", "ports": "80/tcp"}]}',
    ),
    # ---- Skills ----
    "list_skills": (
        "{}",
        '{"skills": [{"slug": "morning-briefing",'
        ' "name": "Morning Briefing", "status": "active"},'
        ' {"slug": "code-review",'
        ' "name": "Code Review", "status": "active"}]}',
    ),
    "search_procedures": (
        '{"query": "deploy production"}',
        '{"procedures": [{"name": "deploy_to_prod",'
        ' "uses": 12,'
        ' "steps": ["git pull", "run tests",'
        ' "docker build", "deploy"]}]}',
    ),
}

# ============================================================================
# Lokalisierte Abschnitts-Header
# ============================================================================

_SECTION_HEADERS: dict[str, dict[str, str]] = {
    "de": {
        "inventory_title": "INVENTAR (automatisch aktualisiert)",
        "tools_title": "Registrierte Tools ({count}) -- Rolle: {role}",
        "skills_title": "Installierte Skills ({count})",
        "procedures_title": "Gelernte Prozeduren ({count})",
        "params_note": "Parameter mit * sind erforderlich.",
        "example_prefix": "Beispiel",
        "variants_suffix": "{count} Varianten, jeweils {uses} genutzt",
        "role_all": "Alle",
        "role_planner": "Planner",
        "role_executor": "Executor",
        "role_browser": "Browser",
        "role_researcher": "Researcher",
    },
    "en": {
        "inventory_title": "INVENTORY (auto-updated)",
        "tools_title": "Registered Tools ({count}) -- Role: {role}",
        "skills_title": "Installed Skills ({count})",
        "procedures_title": "Learned Procedures ({count})",
        "params_note": "Parameters marked with * are required.",
        "example_prefix": "Example",
        "variants_suffix": "{count} variants, each {uses} used",
        "role_all": "All",
        "role_planner": "Planner",
        "role_executor": "Executor",
        "role_browser": "Browser",
        "role_researcher": "Researcher",
    },
    "zh": {
        "inventory_title": "清单 (自动更新)",
        "tools_title": "已注册工具 ({count}) -- 角色: {role}",
        "skills_title": "已安装技能 ({count})",
        "procedures_title": "已学习流程 ({count})",
        "params_note": "标有 * 的参数为必填项。",
        "example_prefix": "示例",
        "variants_suffix": "{count} 个变体, 各 {uses} 次使用",
        "role_all": "全部",
        "role_planner": "Planner",
        "role_executor": "Executor",
        "role_browser": "Browser",
        "role_researcher": "Researcher",
    },
}

# ============================================================================
# Lokalisierte Tool-Beschreibungen
# ============================================================================

_TOOL_DESCRIPTIONS_DE: dict[str, str] = {
    "web_search": "Durchsucht das Internet nach Informationen.",
    "web_news_search": "Sucht aktuelle Nachrichten zu einem Thema.",
    "search_and_read": "Sucht im Internet und liest die besten Ergebnisse vollstaendig.",
    "web_fetch": "Ruft den Inhalt einer URL ab.",
    "http_request": "Fuehrt einen HTTP-Request aus (GET/POST/PUT/PATCH/DELETE).",
    "read_file": "Liest den Inhalt einer Datei.",
    "write_file": "Schreibt Inhalt in eine Datei.",
    "edit_file": "Ersetzt einen String in einer Datei (str_replace).",
    "list_directory": "Listet Dateien und Ordner in einem Verzeichnis.",
    "search_files": "Sucht Dateien nach Name/Pattern.",
    "find_in_files": "Sucht Text in Dateien (grep-artig).",
    "find_and_replace": "Sucht und ersetzt Text in mehreren Dateien.",
    "exec_command": "Fuehrt einen Shell-Befehl in einer Sandbox aus.",
    "run_python": "Fuehrt Python-Code in einer Sandbox aus.",
    "search_memory": "Durchsucht das Langzeitgedaechtnis.",
    "save_to_memory": "Speichert Information im Langzeitgedaechtnis.",
    "get_core_memory": "Gibt CORE.md (Identitaet, Regeln, Praeferenzen) zurueck.",
    "get_recent_episodes": "Laedt die Tageslog-Eintraege der letzten Tage.",
    "search_procedures": "Sucht gelernte Prozeduren nach Stichwort.",
    "record_procedure_usage": "Vermerkt die Nutzung einer Prozedur.",
    "memory_stats": "Zeigt Statistiken zum Gedaechtnis-System.",
    "add_entity": "Fuegt eine Entitaet zum Wissensgraphen hinzu.",
    "add_relation": "Fuegt eine Relation zwischen Entitaeten hinzu.",
    "get_entity": "Laedt eine Entitaet aus dem Wissensgraphen.",
    "vault_save": "Speichert ein Dokument im Vault.",
    "vault_update": "Aktualisiert ein Vault-Dokument.",
    "vault_link": "Verlinkt zwei Vault-Dokumente miteinander.",
    "vault_search": "Durchsucht den Vault nach Dokumenten.",
    "vault_read": "Liest ein Vault-Dokument.",
    "vault_list": "Listet alle Vault-Dokumente.",
    "knowledge_synthesize": "Synthesiert Wissen zu einem Thema aus allen Quellen.",
    "knowledge_gaps": "Analysiert Wissensluecken zu einem Thema.",
    "knowledge_contradictions": "Findet Widersprueche in gespeichertem Wissen.",
    "knowledge_timeline": "Erstellt eine Zeitleiste zu einem Thema.",
    "document_export": "Exportiert Inhalt als PDF/DOCX/Markdown.",
    "analyze_document": "Analysiert ein Dokument (PDF, DOCX, etc.).",
    "media_extract_text": "Extrahiert Text aus Bildern/Dokumenten (OCR).",
    "media_analyze_image": "Analysiert ein Bild mit Vision-Modell.",
    "media_tts": "Konvertiert Text zu Sprache.",
    "media_transcribe_audio": "Transkribiert eine Audio-Datei.",
    "browser_navigate": "Navigiert zu einer URL im Browser.",
    "browser_click": "Klickt ein Element im Browser.",
    "browser_fill": "Fuellt ein Eingabefeld im Browser.",
    "browser_extract": "Extrahiert Daten aus der aktuellen Seite.",
    "browser_screenshot": "Macht einen Screenshot der aktuellen Seite.",
    "git_status": "Zeigt den Git-Status des Repositories.",
    "git_diff": "Zeigt Aenderungen (Git-Diff).",
    "git_log": "Zeigt die Git-Historie.",
    "git_commit": "Erstellt einen Git-Commit.",
    "git_branch": "Verwaltet Git-Branches.",
    "create_chart": "Erstellt ein Diagramm (Bar, Line, Pie, etc.).",
    "email_send": "Sendet eine E-Mail.",
    "email_read_inbox": "Liest den Posteingang.",
    "docker_ps": "Listet laufende Docker-Container.",
    "list_skills": "Listet alle registrierten Skills.",
    "create_skill": "Erstellt einen neuen Skill.",
    "db_query": "Fuehrt eine Datenbank-Abfrage aus.",
    "db_schema": "Zeigt das Datenbank-Schema.",
}

_TOOL_DESCRIPTIONS_ZH: dict[str, str] = {
    "web_search": "搜索互联网信息。",
    "web_news_search": "搜索最新新闻。",
    "search_and_read": "搜索互联网并完整阅读最佳结果。",
    "web_fetch": "获取URL的内容。",
    "read_file": "读取文件内容。",
    "write_file": "将内容写入文件。",
    "edit_file": "替换文件中的字符串。",
    "list_directory": "列出目录中的文件和文件夹。",
    "exec_command": "在沙箱中执行Shell命令。",
    "run_python": "在沙箱中执行Python代码。",
    "search_memory": "搜索长期记忆。",
    "save_to_memory": "保存信息到长期记忆。",
    "get_core_memory": "返回CORE.md（身份、规则、偏好）。",
    "vault_save": "保存文档到保险库。",
    "vault_search": "搜索保险库中的文档。",
    "knowledge_synthesize": "从所有来源综合某主题的知识。",
    "knowledge_gaps": "分析某主题的知识空白。",
    "browser_navigate": "在浏览器中导航到URL。",
    "browser_click": "点击浏览器中的元素。",
    "git_status": "显示Git仓库状态。",
    "create_chart": "创建图表（柱状图、折线图等）。",
    "email_send": "发送电子邮件。",
    "docker_ps": "列出运行中的Docker容器。",
    "list_skills": "列出所有已注册的技能。",
}

# ============================================================================
# Prozedur-Deduplizierung
# ============================================================================


@dataclass
class _ProcedureEntry:
    """Interne Hilfsklasse fuer Prozedur-Gruppierung."""

    name: str
    total_uses: int
    trigger_keywords: list[str]


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard-Aehnlichkeit zwischen zwei Mengen."""
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def deduplicate_procedures(
    procedures: list[_ProcedureEntry],
    *,
    similarity_threshold: float = 0.6,
    group_min_size: int = 4,
    language: str = "de",
) -> list[str]:
    """Dedupliziert aehnliche Prozeduren fuer die CORE.md-Anzeige.

    Prozeduren mit Jaccard-Aehnlichkeit > threshold bei den Trigger-Keywords
    werden gruppiert. Gruppen mit > group_min_size Mitgliedern werden als
    zusammengefasster Eintrag dargestellt. Einzelne Prozeduren mit > 1 Nutzung
    oder einzigartige Prozeduren werden individuell angezeigt.

    Args:
        procedures: Liste von Prozedur-Eintraegen.
        similarity_threshold: Jaccard-Schwellenwert fuer Gruppierung.
        group_min_size: Minimale Gruppengroesse fuer Zusammenfassung.
        language: Sprache fuer die Ausgabe.

    Returns:
        Liste von formatierten Zeilen fuer die CORE.md.
    """
    if not procedures:
        return []

    headers = _SECTION_HEADERS.get(language, _SECTION_HEADERS["en"])

    # Schritt 1: Gruppen nach Keyword-Aehnlichkeit bilden
    used: set[int] = set()
    groups: list[list[int]] = []

    for i in range(len(procedures)):
        if i in used:
            continue
        group = [i]
        kw_i = {k.lower() for k in procedures[i].trigger_keywords}
        for j in range(i + 1, len(procedures)):
            if j in used:
                continue
            kw_j = {k.lower() for k in procedures[j].trigger_keywords}
            if _jaccard(kw_i, kw_j) > similarity_threshold:
                group.append(j)
        if len(group) >= group_min_size:
            for idx in group:
                used.add(idx)
            groups.append(group)

    # Schritt 2: Ergebniszeilen zusammenstellen
    lines: list[str] = []

    # Zusammengefasste Gruppen
    for group in groups:
        members = [procedures[idx] for idx in group]
        # Repraesentativen Namen waehlen (erster alphabetisch)
        rep_name = sorted(m.name for m in members)[0]
        total_uses_sum = sum(m.total_uses for m in members)
        avg_uses = f"{total_uses_sum // len(members)}x" if len(members) > 0 else "0x"
        # Gemeinsame Keywords sammeln
        all_kw: set[str] = set()
        for m in members:
            all_kw.update(k.lower() for k in m.trigger_keywords)
        kw_str = ", ".join(sorted(all_kw)[:5])
        variant_text = headers["variants_suffix"].format(
            count=len(members), uses=avg_uses,
        )
        lines.append(f"- `{rep_name}` ({variant_text}) [{kw_str}]")

    # Individuelle Prozeduren (nicht in Gruppen)
    for i, proc in enumerate(procedures):
        if i in used:
            continue
        uses = f"{proc.total_uses}x" if proc.total_uses else "0x"
        kw = ", ".join(proc.trigger_keywords[:3])
        suffix = f" [{kw}]" if kw else ""
        lines.append(f"- `{proc.name}` ({uses} used){suffix}")

    return lines


# ============================================================================
# ToolRegistryDB
# ============================================================================

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS tools (
    name TEXT PRIMARY KEY,
    description_de TEXT DEFAULT '',
    description_en TEXT DEFAULT '',
    description_zh TEXT DEFAULT '',
    input_schema TEXT DEFAULT '{}',
    example_input TEXT DEFAULT '',
    example_output TEXT DEFAULT '',
    category TEXT DEFAULT 'other',
    agent_roles TEXT DEFAULT 'all',
    updated_at TEXT DEFAULT ''
);
"""


class ToolRegistryDB:
    """Datenbank-gestuetztes Tool-Registry fuer dynamische Prompt-Generierung.

    Speichert Tool-Metadaten in SQLite und generiert lokalisierte,
    rollenbasierte Prompt-Abschnitte.
    """

    def __init__(self, db_path: Path) -> None:
        """Initialisiert die Datenbank.

        Args:
            db_path: Pfad zur SQLite-Datenbankdatei. Wird automatisch
                     erstellt, falls nicht vorhanden.
        """
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()
        log.debug("tool_registry_db_init", path=str(db_path))

    # ---- CRUD ---------------------------------------------------------------

    def upsert_tool(
        self,
        name: str,
        description_de: str = "",
        description_en: str = "",
        description_zh: str = "",
        input_schema: dict[str, Any] | None = None,
        example_input: str = "",
        example_output: str = "",
        category: str = "other",
        agent_roles: list[str] | None = None,
    ) -> None:
        """Fuegt ein Tool ein oder aktualisiert es.

        Args:
            name: Eindeutiger Tool-Name.
            description_de: Deutsche Beschreibung.
            description_en: Englische Beschreibung.
            description_zh: Chinesische Beschreibung.
            input_schema: JSON-Schema der Eingabeparameter.
            example_input: Beispiel-Aufruf als String.
            example_output: Beispiel-Ausgabe als String.
            category: Kategorie (z.B. 'filesystem', 'web').
            agent_roles: Liste der Rollen ('planner', 'executor', etc.).
        """
        roles_str = ",".join(agent_roles) if agent_roles else "all"
        schema_str = json.dumps(input_schema or {}, ensure_ascii=False)
        now = datetime.now(UTC).isoformat()

        self._conn.execute(
            """INSERT INTO tools (name, description_de, description_en, description_zh,
                                  input_schema, example_input, example_output,
                                  category, agent_roles, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(name) DO UPDATE SET
                   description_de=excluded.description_de,
                   description_en=excluded.description_en,
                   description_zh=excluded.description_zh,
                   input_schema=excluded.input_schema,
                   example_input=CASE WHEN excluded.example_input != '' THEN excluded.example_input
                                      ELSE tools.example_input END,
                   example_output=CASE WHEN excluded.example_output != ''
                                      THEN excluded.example_output
                                      ELSE tools.example_output END,
                   category=excluded.category,
                   agent_roles=excluded.agent_roles,
                   updated_at=excluded.updated_at
            """,
            (name, description_de, description_en, description_zh,
             schema_str, example_input, example_output, category, roles_str, now),
        )
        self._conn.commit()

    def get_tool(self, name: str) -> ToolInfo | None:
        """Gibt ToolInfo fuer einen bestimmten Tool-Namen zurueck."""
        row = self._conn.execute(
            "SELECT * FROM tools WHERE name = ?", (name,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_info(row, "en")

    def get_tools_for_role(
        self, role: str, language: str = "de",
    ) -> list[ToolInfo]:
        """Gibt Tools zurueck, die einer bestimmten Agentenrolle zugeordnet sind.

        Args:
            role: Rollenname ('planner', 'executor', 'browser', 'researcher', 'all').
            language: Sprachcode fuer die Beschreibung ('de', 'en', 'zh').

        Returns:
            Liste von ToolInfo-Objekten, sortiert nach Kategorie und Name.
        """
        rows = self._conn.execute("SELECT * FROM tools ORDER BY category, name").fetchall()
        results: list[ToolInfo] = []
        for row in rows:
            roles = row["agent_roles"].split(",") if row["agent_roles"] else ["all"]
            if role == "all" or "all" in roles or role in roles:
                results.append(self._row_to_info(row, language))
        return results

    def get_tool_prompt_section(
        self, role: str, language: str = "de",
    ) -> str:
        """Generiert einen formatierten Tool-Abschnitt fuer Agent-Prompts.

        Args:
            role: Agentenrolle fuer Filterung.
            language: Sprache fuer Header und Beschreibungen.

        Returns:
            Markdown-formatierter String mit Tools nach Kategorie gruppiert.
        """
        tools = self.get_tools_for_role(role, language)
        headers = _SECTION_HEADERS.get(language, _SECTION_HEADERS["en"])
        cat_labels = _CATEGORY_LABELS.get(language, _CATEGORY_LABELS["en"])

        # Rolle lokalisieren
        role_key = f"role_{role}"
        role_label = headers.get(role_key, role.capitalize())

        title = headers["tools_title"].format(count=len(tools), role=role_label)
        lines = [f"### {title}\n"]
        lines.append(f"{headers['params_note']}\n")

        # Nach Kategorie gruppieren
        by_cat: dict[str, list[ToolInfo]] = {}
        for tool in tools:
            by_cat.setdefault(tool.category, []).append(tool)

        for cat in sorted(by_cat.keys()):
            cat_label = cat_labels.get(cat, cat.capitalize())
            lines.append(f"**{cat_label}:**")
            for tool in by_cat[cat]:
                # Parameter-String aufbauen
                param_str = self._format_params(tool.input_schema)
                sig = f"`{tool.name}({param_str})`" if param_str else f"`{tool.name}()`"
                lines.append(f"- {sig} -- {tool.description}")
                # Structured example, if available
                if tool.example_input:
                    ex_prefix = headers["example_prefix"]
                    lines.append(f"  {ex_prefix} Input: `{tool.example_input}`")
                    if tool.example_output:
                        lines.append(f"  {ex_prefix} Output: `{tool.example_output}`")
            lines.append("")  # Leerzeile nach Kategorie

        return "\n".join(lines)

    def sync_from_mcp(
        self, mcp_client: JarvisMCPClient, *, keep_examples: bool = True,
    ) -> int:
        """Synchronisiert Tool-Metadaten aus dem MCP-Client.

        Aktualisiert Beschreibungen und Schemas aus den live-registrierten
        Tools. Manuell hinzugefuegte Beispiele bleiben erhalten.

        Args:
            mcp_client: Aktiver MCP-Client mit registrierten Tools.
            keep_examples: Wenn True, werden bestehende Beispiele nicht ueberschrieben.

        Returns:
            Anzahl der synchronisierten Tools.
        """
        schemas = mcp_client.get_tool_schemas()
        count = 0

        for name, schema in schemas.items():
            desc = schema.get("description", "")
            input_schema = schema.get("inputSchema", {})

            # Kategorie bestimmen
            category = TOOL_CATEGORIES.get(name, "other")
            if category == "other":
                # Versuch anhand von Praefix
                for prefix, cat in [
                    ("browser_", "browser"), ("media_", "media"),
                    ("vault_", "vault"), ("knowledge_", "knowledge"),
                    ("git_", "git"), ("db_", "database"),
                    ("docker_", "docker"), ("email_", "email"),
                    ("calendar_", "calendar"), ("api_", "api"),
                ]:
                    if name.startswith(prefix):
                        category = cat
                        break

            # Rollen bestimmen
            roles: list[str] = []
            for role, tool_set in TOOL_ROLE_DEFAULTS.items():
                if name in tool_set:
                    roles.append(role)
            if not roles:
                roles = ["all"]

            # Beispiele aus Defaults holen
            ex_in, ex_out = "", ""
            if name in DEFAULT_EXAMPLES:
                ex_in, ex_out = DEFAULT_EXAMPLES[name]

            # Localized descriptions: use tool-specific overrides if available,
            # otherwise the MCP description (usually English) goes to all langs
            desc_de = _TOOL_DESCRIPTIONS_DE.get(name, desc)
            desc_en = desc  # MCP descriptions are typically in English
            desc_zh = _TOOL_DESCRIPTIONS_ZH.get(name, desc)

            self.upsert_tool(
                name=name,
                description_de=desc_de,
                description_en=desc_en,
                description_zh=desc_zh,
                input_schema=input_schema,
                example_input=ex_in,
                example_output=ex_out,
                category=category,
                agent_roles=roles,
            )
            count += 1

        log.info("tool_registry_synced", tools=count)
        return count

    def add_example(
        self, tool_name: str, example_input: str, example_output: str,
    ) -> bool:
        """Fuegt ein Beispiel fuer ein Tool hinzu oder aktualisiert es.

        Args:
            tool_name: Name des Tools.
            example_input: Beispiel-Aufruf.
            example_output: Beispiel-Ausgabe.

        Returns:
            True wenn das Tool existiert und aktualisiert wurde.
        """
        cursor = self._conn.execute(
            "UPDATE tools SET example_input = ?, example_output = ?, updated_at = ? WHERE name = ?",
            (example_input, example_output,
             datetime.now(UTC).isoformat(), tool_name),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def set_agent_roles(self, tool_name: str, roles: list[str]) -> bool:
        """Setzt die Agentenrollen fuer ein Tool.

        Args:
            tool_name: Name des Tools.
            roles: Liste der Rollen.

        Returns:
            True wenn das Tool existiert und aktualisiert wurde.
        """
        roles_str = ",".join(roles)
        cursor = self._conn.execute(
            "UPDATE tools SET agent_roles = ?, updated_at = ? WHERE name = ?",
            (roles_str, datetime.now(UTC).isoformat(), tool_name),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def tool_count(self) -> int:
        """Gibt die Gesamtanzahl registrierter Tools zurueck."""
        row = self._conn.execute("SELECT COUNT(*) FROM tools").fetchone()
        return row[0] if row else 0

    def close(self) -> None:
        """Schliesst die Datenbankverbindung."""
        self._conn.close()

    # ---- Hilfsmethoden ------------------------------------------------------

    def _row_to_info(self, row: sqlite3.Row, language: str) -> ToolInfo:
        """Konvertiert eine Datenbankzeile in ein ToolInfo-Objekt."""
        desc_col = f"description_{language}"
        # Fallback: en -> de -> erster verfuegbarer
        try:
            desc = row[desc_col] or ""
        except (IndexError, KeyError):
            desc = ""
        if not desc:
            desc = row["description_en"] or row["description_de"] or ""

        schema_raw = row["input_schema"]
        try:
            input_schema = json.loads(schema_raw) if schema_raw else {}
        except (json.JSONDecodeError, TypeError):
            input_schema = {}

        roles_raw = row["agent_roles"] or "all"
        agent_roles = [r.strip() for r in roles_raw.split(",") if r.strip()]

        return ToolInfo(
            name=row["name"],
            description=desc,
            input_schema=input_schema,
            example_input=row["example_input"] or "",
            example_output=row["example_output"] or "",
            category=row["category"] or "other",
            agent_roles=agent_roles,
        )

    @staticmethod
    def _format_params(input_schema: dict[str, Any]) -> str:
        """Formatiert die Parameter eines Tools als kompakten String mit Typen."""
        props = input_schema.get("properties", {})
        if not props:
            return ""
        required = set(input_schema.get("required", []))
        parts: list[str] = []
        for k, v in props.items():
            ptype = v.get("type", "any") if isinstance(v, dict) else "any"
            req = " *" if k in required else ""
            parts.append(f"{k}: {ptype}{req}")
        return ", ".join(parts)
