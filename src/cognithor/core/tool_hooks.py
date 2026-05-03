"""Pre/Post Tool-Use Hook-System.

Ermoeglicht automatische Aktionen VOR und NACH jeder Tool-Ausfuehrung:
  - Secret-Redacting (PreToolUse)
  - Audit-Logging (PostToolUse)
  - Cost-Tracking (PostToolUse)
  - Permission-Override (PreToolUse)

Integriert sich in den Executor-Loop.

Bibel-Referenz: Phase 2, Verbesserung 3.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

from cognithor.utils.logging import get_logger

log = get_logger(__name__)


# ============================================================================
# Datenmodell
# ============================================================================


class HookEvent(StrEnum):
    PRE_TOOL_USE = "PreToolUse"
    POST_TOOL_USE = "PostToolUse"
    POST_TOOL_USE_FAILURE = "PostToolUseFailure"


@dataclass
class HookResult:
    denied: bool = False
    deny_reason: str = ""
    updated_input: dict[str, Any] | None = None
    messages: list[str] = field(default_factory=list)


# ============================================================================
# HookRunner
# ============================================================================


class ToolHookRunner:
    """Fuehrt registrierte Hooks vor/nach Tool-Ausfuehrung aus."""

    def __init__(self) -> None:
        self._hooks: dict[HookEvent, list[tuple[str, Callable[..., Any]]]] = {
            HookEvent.PRE_TOOL_USE: [],
            HookEvent.POST_TOOL_USE: [],
            HookEvent.POST_TOOL_USE_FAILURE: [],
        }

    def register(self, event: HookEvent, name: str, hook: Callable[..., Any]) -> None:
        """Registriert einen Hook fuer ein Event."""
        self._hooks[event].append((name, hook))
        log.debug("tool_hook_registered", hook_event=event.value, hook_name=name)

    def run_pre_tool_use(self, tool_name: str, tool_input: dict[str, Any]) -> HookResult:
        """Fuehrt alle PreToolUse-Hooks aus.

        Returns:
            HookResult — denied=True wenn ein Hook blockiert.
        """
        result = HookResult()
        for hook_name, hook_fn in self._hooks[HookEvent.PRE_TOOL_USE]:
            try:
                output = hook_fn(tool_name, tool_input)
                if output and isinstance(output, dict):
                    if output.get("deny"):
                        result.denied = True
                        result.deny_reason = output.get("reason", f"Denied by hook '{hook_name}'")
                        result.messages.append(result.deny_reason)
                        return result
                    if output.get("updated_input"):
                        result.updated_input = output["updated_input"]
            except Exception as exc:
                result.messages.append(f"Hook '{hook_name}' failed: {exc}")
                log.warning("tool_hook_pre_failed", hook=hook_name, error=str(exc))
        return result

    def run_post_tool_use(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_output: str,
        duration_ms: int = 0,
    ) -> None:
        """Fuehrt alle PostToolUse-Hooks aus (fire-and-forget)."""
        for hook_name, hook_fn in self._hooks[HookEvent.POST_TOOL_USE]:
            try:
                hook_fn(tool_name, tool_input, tool_output, duration_ms)
            except Exception as exc:
                log.warning("tool_hook_post_failed", hook=hook_name, error=str(exc))

    def run_post_failure(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        error: str,
    ) -> None:
        """Fuehrt alle PostToolUseFailure-Hooks aus."""
        for hook_name, hook_fn in self._hooks[HookEvent.POST_TOOL_USE_FAILURE]:
            try:
                hook_fn(tool_name, tool_input, error)
            except Exception as exc:
                log.warning("tool_hook_failure_failed", hook=hook_name, error=str(exc))

    @property
    def hook_count(self) -> int:
        return sum(len(hooks) for hooks in self._hooks.values())


# ============================================================================
# Standard-Hooks
# ============================================================================

# Patterns fuer Secret-Redacting
_SECRET_PATTERNS = [
    re.compile(r"(sk-[a-zA-Z0-9]{20,})"),  # OpenAI keys
    re.compile(r"(ghp_[a-zA-Z0-9]{36})"),  # GitHub PAT
    re.compile(r"(gho_[a-zA-Z0-9]{36})"),  # GitHub OAuth
    re.compile(r"(xoxb-[a-zA-Z0-9-]+)"),  # Slack bot token
    re.compile(r"(AKIA[A-Z0-9]{16})"),  # AWS access key
    re.compile(r"(eyJ[a-zA-Z0-9_-]{20,}\.[a-zA-Z0-9_-]{20,})"),  # JWT
]


def secret_redacting_hook(tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any] | None:
    """PreToolUse: Entfernt API-Keys und Tokens aus Shell-Commands."""
    if tool_name not in ("exec_command", "shell_exec", "shell"):
        return None

    command = tool_input.get("command", "")
    if not isinstance(command, str):
        return None

    redacted = command
    changed = False
    for pattern in _SECRET_PATTERNS:
        new_val = pattern.sub("[REDACTED]", redacted)
        if new_val != redacted:
            changed = True
            redacted = new_val

    if changed:
        log.info("secret_redacted_in_command", tool=tool_name)
        return {"updated_input": {**tool_input, "command": redacted}}
    return None


def audit_logging_hook(
    tool_name: str,
    tool_input: dict[str, Any],
    tool_output: str,
    duration_ms: int = 0,
) -> None:
    """PostToolUse: Loggt jede Tool-Ausfuehrung strukturiert."""
    log.info(
        "tool_audit",
        tool=tool_name,
        duration_ms=duration_ms,
        output_len=len(tool_output) if tool_output else 0,
        success=True,
    )


# ============================================================================
# Security Extension Hook
# ============================================================================

# Patterns for dangerous file content (written via write_file / file_write)
DANGEROUS_FILE_CONTENT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\brm\s+-rf\s+/(\s|$)", re.I), "Destructive root delete in file content"),
    (re.compile(r":\(\)\s*\{.*\};\s*:", re.I), "Fork-bomb in file content"),
    (
        re.compile(r"\bcurl\b[^\n|]*\|\s*(sh|bash|zsh)\b", re.I),
        "Remote shell execution in file content",
    ),
]

# Patterns for dangerous bash/shell commands (exfiltration)
DANGEROUS_BASH_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"\b(cat|sed|awk)\b[^|]*\.(env|pem|key|p12)\b[^|]*(\|\s*(curl|wget)|>\s*/dev/tcp)",
            re.I,
        ),
        "Exfiltration of sensitive files",
    ),
    (
        re.compile(
            r"\b(printenv|env)\b[^|]*(\|\s*(curl|wget)|>\s*/dev/tcp)",
            re.I,
        ),
        "Exfiltration of environment variables",
    ),
]

# Binary office formats that must not be written as plain text
BINARY_OFFICE_RE = re.compile(r"\.(docx|xlsx|pptx|pdf)$", re.I)

# Tool names that execute shell commands
_SHELL_TOOLS = frozenset({"exec_command", "shell_exec", "shell"})

# Tool names that write file content
_FILE_WRITE_TOOLS = frozenset({"write_file", "file_write", "create_file"})


def security_extension_hook(tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any] | None:
    """PreToolUse: Blocks exfiltration, dangerous file content, and binary office writes.

    Returns ``{"deny": True, "reason": "..."}`` when the call must be blocked,
    or ``None`` when the call is safe to proceed.
    """

    # ── 1. Shell exfiltration detection ─────────────────────────────
    if tool_name in _SHELL_TOOLS:
        command = tool_input.get("command", "")
        if isinstance(command, str):
            for pattern, reason in DANGEROUS_BASH_PATTERNS:
                if pattern.search(command):
                    log.warning(
                        "security_hook_blocked",
                        tool=tool_name,
                        reason=reason,
                    )
                    return {"deny": True, "reason": reason}

    # ── 2. File-write checks ────────────────────────────────────────
    if tool_name in _FILE_WRITE_TOOLS:
        file_path = tool_input.get("path", "") or tool_input.get("file_path", "")
        content = tool_input.get("content", "")

        # 2a. Binary office file blocking
        if isinstance(file_path, str) and BINARY_OFFICE_RE.search(file_path):
            log.warning(
                "security_hook_blocked",
                tool=tool_name,
                reason="Binary office format — use document_export tool instead",
                path=file_path,
            )
            return {
                "deny": True,
                "reason": (
                    f"Cannot write plain text to binary office file '{file_path}'. "
                    "Use the document_export tool for .docx/.xlsx/.pptx/.pdf files."
                ),
            }

        # 2b. Dangerous content patterns
        if isinstance(content, str):
            for pattern, reason in DANGEROUS_FILE_CONTENT_PATTERNS:
                if pattern.search(content):
                    log.warning(
                        "security_hook_blocked",
                        tool=tool_name,
                        reason=reason,
                    )
                    return {"deny": True, "reason": reason}

    return None
