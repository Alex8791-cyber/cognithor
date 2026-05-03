"""Shell command safety analyser using bashlex AST parsing.

Replaces the regex-based file-command check. bashlex builds a proper
parse tree for bash, detecting command substitution, chained operators,
pipelines, and heredocs that trivially bypass regex approaches.

Falls back to a conservative regex-based check when bashlex is not installed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["ShellViolation", "analyse_shell", "is_safe_shell"]

try:
    import bashlex

    _BASHLEX_AVAILABLE = True
except ImportError:
    _BASHLEX_AVAILABLE = False


@dataclass
class ShellViolation:
    pos: int
    rule: str
    detail: str


BLOCKED_COMMANDS: frozenset[str] = frozenset(
    {
        "rm",
        "rmdir",
        "dd",
        "mkfs",
        "fdisk",
        "parted",
        "shred",
        "wget",
        "curl",
        "nc",
        "ncat",
        "netcat",
        "ssh",
        "scp",
        "rsync",
        "chmod",
        "chown",
        "sudo",
        "su",
        "doas",
        "pkexec",
        "env",
        "xargs",
        "tee",
        "install",
        "mv",
        "cp",
        "ln",
        "mount",
        "umount",
        "insmod",
        "modprobe",
        "systemctl",
        "service",
        "init",
        "kill",
        "killall",
        "pkill",
        "reboot",
        "halt",
        "shutdown",
        "poweroff",
        "crontab",
        "at",
        "batch",
        "python",
        "python3",
        "perl",
        "ruby",
        "node",
        "php",
        "bash",
        "sh",
        "dash",
        "zsh",
        "fish",
        "csh",
        "tcsh",
        "ksh",
        "exec",
        "eval",
        "source",
        ".",
    }
)

# Patterns that indicate chaining/substitution (regex fallback)
_DANGEROUS_SHELL_PATTERNS = [
    re.compile(r"\$\("),  # Command substitution $(...)
    re.compile(r"`"),  # Backtick substitution
    re.compile(r"[;&|]{1,2}"),  # Chained operators ; && || |
    re.compile(r">\s*/"),  # Redirect to absolute path
    re.compile(r"<\("),  # Process substitution <(...)
]


class _ShellASTVisitor:
    """Walk a bashlex part tree and collect ShellViolations."""

    def __init__(self) -> None:
        self.violations: list[ShellViolation] = []

    def visit(self, part: object) -> None:
        kind = getattr(part, "kind", "unknown")
        method = f"visit_{kind}"
        getattr(self, method, self.generic_visit)(part)

    def generic_visit(self, part: object) -> None:
        for child in getattr(part, "parts", []):
            self.visit(child)

    def visit_command(self, part: object) -> None:
        parts = getattr(part, "parts", [])
        if parts:
            first = parts[0]
            if getattr(first, "kind", "") == "word":
                cmd = getattr(first, "word", "")
                # Strip path prefix: /usr/bin/rm → rm
                cmd_base = cmd.rsplit("/", 1)[-1] if "/" in cmd else cmd
                if cmd_base in BLOCKED_COMMANDS:
                    pos = getattr(part, "pos", (0,))
                    self.violations.append(
                        ShellViolation(
                            pos=pos[0] if isinstance(pos, tuple) else pos,
                            rule="blocked-command",
                            detail=f"Blocked shell command: {cmd!r}",
                        )
                    )
        self.generic_visit(part)

    def visit_commandsubstitution(self, part: object) -> None:
        pos = getattr(part, "pos", (0,))
        self.violations.append(
            ShellViolation(
                pos=pos[0] if isinstance(pos, tuple) else pos,
                rule="command-substitution",
                detail=(
                    "Command substitution detected ($(...) or backticks)."
                    " Bypasses command-name checks."
                ),
            )
        )
        self.generic_visit(part)

    def visit_processsubstitution(self, part: object) -> None:
        pos = getattr(part, "pos", (0,))
        self.violations.append(
            ShellViolation(
                pos=pos[0] if isinstance(pos, tuple) else pos,
                rule="process-substitution",
                detail="Process substitution detected (<(...) or >(...)).",
            )
        )
        self.generic_visit(part)

    def visit_operator(self, part: object) -> None:
        op = getattr(part, "op", "")
        if op in {"&&", "||", ";", "|", "|&"}:
            pos = getattr(part, "pos", (0,))
            self.violations.append(
                ShellViolation(
                    pos=pos[0] if isinstance(pos, tuple) else pos,
                    rule="chained-operator",
                    detail=f"Chained operator {op!r} detected. Can inject additional commands.",
                )
            )
        self.generic_visit(part)

    def visit_compound(self, part: object) -> None:
        self.generic_visit(part)

    def visit_list(self, part: object) -> None:
        self.generic_visit(part)

    def visit_pipeline(self, part: object) -> None:
        pos = getattr(part, "pos", (0,))
        self.violations.append(
            ShellViolation(
                pos=pos[0] if isinstance(pos, tuple) else pos,
                rule="pipeline",
                detail="Pipeline detected. Can chain dangerous commands after safe ones.",
            )
        )
        self.generic_visit(part)

    def visit_word(self, part: object) -> None:
        pass  # Leaf node

    def visit_redirect(self, part: object) -> None:
        self.generic_visit(part)

    def visit_assignment(self, part: object) -> None:
        self.generic_visit(part)


def _analyse_shell_regex(command: str) -> list[ShellViolation]:
    """Fallback regex-based analysis when bashlex is not available."""
    violations: list[ShellViolation] = []

    # Check blocked commands
    tokens = re.split(r"\s+", command.strip())
    if tokens:
        cmd = tokens[0].rsplit("/", 1)[-1]
        if cmd in BLOCKED_COMMANDS:
            violations.append(
                ShellViolation(
                    pos=0, rule="blocked-command", detail=f"Blocked shell command: {cmd!r}"
                )
            )

    # Check dangerous patterns
    for pattern in _DANGEROUS_SHELL_PATTERNS:
        m = pattern.search(command)
        if m:
            violations.append(
                ShellViolation(
                    pos=m.start(),
                    rule="dangerous-pattern",
                    detail=f"Dangerous shell pattern detected: {m.group()!r}",
                )
            )

    return violations


def analyse_shell(command: str) -> list[ShellViolation]:
    """Parse *command* as bash and return a list of ShellViolations.

    Uses bashlex if available, falls back to regex-based analysis.
    """
    if not _BASHLEX_AVAILABLE:
        return _analyse_shell_regex(command)

    try:
        parts = bashlex.parse(command)
    except Exception:
        # Parse error — treat as suspicious, use regex fallback
        return _analyse_shell_regex(command)

    visitor = _ShellASTVisitor()
    for part in parts:
        visitor.visit(part)
    return visitor.violations


def is_safe_shell(command: str) -> bool:
    """Return True only if ``analyse_shell`` finds zero violations."""
    try:
        return len(analyse_shell(command)) == 0
    except Exception:
        return False
