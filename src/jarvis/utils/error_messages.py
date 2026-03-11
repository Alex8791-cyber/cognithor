"""User-friendly error messages for Jarvis channels.

Replaces generic error messages with contextual, empathetic error descriptions.
Used across all channels. Uses the i18n language pack system for translations.
"""

from __future__ import annotations

from jarvis.i18n import t


# ── Tool-Friendly-Names ─────────────────────────────────────────
# Maps internal MCP tool names to human-readable names via i18n keys.

_TOOL_KEY_MAP: dict[str, str] = {
    "exec_command": "tool.exec_command",
    "write_file": "tool.write_file",
    "read_file": "tool.read_file",
    "edit_file": "tool.edit_file",
    "list_directory": "tool.list_directory",
    "run_python": "tool.run_python",
    "web_search": "tool.web_search",
    "web_news_search": "tool.web_news_search",
    "web_fetch": "tool.web_fetch",
    "search_and_read": "tool.search_and_read",
    "search_memory": "tool.search_memory",
    "save_to_memory": "tool.save_to_memory",
    "document_export": "tool.document_export",
    "media_analyze_image": "tool.analyze_image",
    "media_transcribe_audio": "tool.transcribe_audio",
    "media_extract_text": "tool.extract_text",
    "media_tts": "tool.text_to_speech",
    "vault_search": "tool.vault_search",
    "vault_write": "tool.vault_entry",
    "analyze_code": "tool.analyze_code",
    "browser_navigate": "tool.browser_navigate",
    "browser_screenshot": "tool.browser_screenshot",
}


def _friendly_tool_name(tool: str) -> str:
    """Returns a user-friendly name for a tool."""
    key = _TOOL_KEY_MAP.get(tool)
    if key:
        return t(key)
    return tool


# ── Error Classification ────────────────────────────────────────


def classify_error_for_user(exc: BaseException) -> str:
    """Classifies an exception into a user-friendly error message.

    Used by all channels instead of generic error messages.
    Language is determined by the active i18n locale.
    """
    exc_type = type(exc).__name__
    exc_str = str(exc)[:200]

    # Ollama-specific errors (model not found, connection refused)
    if exc_type == "OllamaError":
        status_code = getattr(exc, "status_code", None)
        if status_code == 404:
            return t("error.model_not_installed", model="<modelname>") + f"\nDetails: {exc_str}"
        if "nicht erreichbar" in exc_str.lower() or "connect" in exc_str.lower():
            return t("error.ollama_unreachable")

    if exc_type in ("TimeoutError", "asyncio.TimeoutError") or "timeout" in exc_str.lower():
        return t("error.timeout") + f"\nDetail: {exc_str[:200]}"

    if (
        exc_type in ("ConnectionError", "ConnectError", "OSError")
        or "connection" in exc_str.lower()
    ):
        return t("error.connection_failed")

    if exc_type in ("PermissionError", "AuthenticationError") or "permission" in exc_str.lower():
        return t("error.permission_denied")

    if exc_type == "FileNotFoundError" or "not found" in exc_str.lower():
        return t("error.not_found")

    if "rate limit" in exc_str.lower() or "429" in exc_str:
        return t("error.rate_limited")

    if "memory" in exc_str.lower() or exc_type == "MemoryError":
        return t("error.memory_error")

    # Generic fallback
    return t("error.generic", exc_type=exc_type) + f"\nDetail: {exc_str[:300]}"


# ── Gatekeeper Block Messages ───────────────────────────────────


def gatekeeper_block_message(tool: str, reason: str) -> str:
    """Creates a user-friendly message when the Gatekeeper blocks an action."""
    friendly = _friendly_tool_name(tool)
    return t("error.gatekeeper_blocked", friendly=friendly, reason=reason)


# ── Retry Exhausted Messages ────────────────────────────────────


def retry_exhausted_message(tool: str, attempts: int, error: str) -> str:
    """Creates a user-friendly message when all retries are exhausted."""
    friendly = _friendly_tool_name(tool)

    # Classify the error for a friendlier sub-message
    error_lower = error.lower()
    if "timeout" in error_lower:
        cause = t("error.retry_timeout")
    elif "connection" in error_lower:
        cause = t("error.retry_connection")
    elif "rate" in error_lower or "429" in error:
        cause = t("error.retry_overloaded")
    else:
        cause = t("error.retry_technical", detail=error[:100])

    return t("error.retry_exhausted", friendly=friendly, attempts=attempts, detail=cause)


# ── All-Blocked Message ─────────────────────────────────────────


def all_actions_blocked_message(
    steps: list,
    decisions: list,
) -> str:
    """Creates a specific message when all planned actions are blocked."""
    reasons: list[str] = []
    for step, decision in zip(steps, decisions, strict=False):
        friendly = _friendly_tool_name(step.tool)
        reasons.append(f"- {friendly}: {decision.reason}")

    reasons_text = "\n".join(reasons) if reasons else t("error.no_details")
    return t("error.all_actions_blocked") + f"\n{reasons_text}"
