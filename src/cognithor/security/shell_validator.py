"""Shell-Command Validation Pipeline — 6-stufige semantische Pruefung.

Laeuft VOR der Sandbox-Ausfuehrung und ergaenzt die bestehende
Layer-0-Validierung (Null-Bytes, Path-Traversal) in mcp/shell.py
sowie die Gatekeeper-Regex-Patterns.

Stufen:
  1. read_only   — Schreiboperationen im read-only Modus blockieren
  2. mode        — Workspace-write: System-Pfade warnen
  3. sed         — sed -i im read-only Modus blockieren
  4. destructive — Destruktive Patterns erkennen
  5. path        — Workspace-Escape erkennen
  6. semantics   — Informational: CommandIntent Klassifikation

Bibel-Referenz: Phase 2, Verbesserung 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, cast

# ============================================================================
# Datenmodell
# ============================================================================


class ValidationVerdict(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"
    WARN = "warn"


class CommandIntent(str, Enum):
    READ_ONLY = "read_only"
    WRITE = "write"
    DESTRUCTIVE = "destructive"
    NETWORK = "network"
    PROCESS_MANAGEMENT = "process_management"
    PACKAGE_MANAGEMENT = "package_management"
    SYSTEM_ADMIN = "system_admin"
    UNKNOWN = "unknown"


@dataclass
class ValidationResult:
    verdict: ValidationVerdict
    reason: str = ""
    stage: str = ""
    intent: CommandIntent = CommandIntent.UNKNOWN


# ============================================================================
# Command-Klassifikations-Listen
# ============================================================================

WRITE_COMMANDS = frozenset(
    {
        "cp",
        "mv",
        "rm",
        "mkdir",
        "rmdir",
        "touch",
        "chmod",
        "chown",
        "chgrp",
        "ln",
        "install",
        "tee",
        "truncate",
        "shred",
        "mkfifo",
        "mknod",
        "dd",
    }
)

STATE_MODIFYING_COMMANDS = frozenset(
    {
        "apt",
        "apt-get",
        "yum",
        "dnf",
        "pacman",
        "brew",
        "pip",
        "pip3",
        "npm",
        "yarn",
        "pnpm",
        "bun",
        "cargo",
        "gem",
        "go",
        "rustup",
        "docker",
        "systemctl",
        "service",
        "mount",
        "umount",
        "kill",
        "pkill",
        "killall",
        "reboot",
        "shutdown",
        "halt",
        "poweroff",
        "useradd",
        "userdel",
        "usermod",
        "groupadd",
        "groupdel",
        "crontab",
        "at",
    }
)

READ_ONLY_COMMANDS = frozenset(
    {
        "ls",
        "cat",
        "head",
        "tail",
        "less",
        "more",
        "wc",
        "sort",
        "uniq",
        "grep",
        "egrep",
        "fgrep",
        "find",
        "which",
        "whereis",
        "whatis",
        "man",
        "info",
        "file",
        "stat",
        "du",
        "df",
        "free",
        "uptime",
        "uname",
        "hostname",
        "whoami",
        "id",
        "groups",
        "env",
        "printenv",
        "echo",
        "printf",
        "date",
        "cal",
        "bc",
        "expr",
        "test",
        "true",
        "false",
        "pwd",
        "tree",
        "diff",
        "cmp",
        "md5sum",
        "sha256sum",
        "sha1sum",
        "xxd",
        "od",
        "hexdump",
        "strings",
        "readlink",
        "realpath",
        "basename",
        "dirname",
        "seq",
        "yes",
        "tput",
        "column",
        "jq",
        "yq",
        "xargs",
        "tr",
        "cut",
        "paste",
        "awk",
        "sed",
        # Windows-gaengige Commands
        "dir",
        "type",
        "where",
        "ver",
        "systeminfo",
        "tasklist",
        "set",
    }
)

NETWORK_COMMANDS = frozenset(
    {
        "curl",
        "wget",
        "ssh",
        "scp",
        "rsync",
        "ftp",
        "sftp",
        "nc",
        "ncat",
        "telnet",
        "ping",
        "traceroute",
        "dig",
        "nslookup",
        "host",
        "whois",
        "ifconfig",
        "ip",
        "netstat",
        "ss",
        "nmap",
    }
)

PROCESS_COMMANDS = frozenset(
    {
        "kill",
        "pkill",
        "killall",
        "ps",
        "top",
        "htop",
        "bg",
        "fg",
        "jobs",
        "nohup",
        "disown",
        "wait",
        "nice",
        "renice",
    }
)

PACKAGE_COMMANDS = frozenset(
    {
        "apt",
        "apt-get",
        "yum",
        "dnf",
        "pacman",
        "brew",
        "pip",
        "pip3",
        "npm",
        "yarn",
        "pnpm",
        "bun",
        "cargo",
        "gem",
        "go",
        "rustup",
        "snap",
        "flatpak",
    }
)

SYSTEM_ADMIN_COMMANDS = frozenset(
    {
        "sudo",
        "su",
        "chroot",
        "mount",
        "umount",
        "fdisk",
        "parted",
        "lsblk",
        "blkid",
        "systemctl",
        "service",
        "journalctl",
        "dmesg",
        "modprobe",
        "insmod",
        "rmmod",
        "iptables",
        "ufw",
        "firewall-cmd",
        "sysctl",
        "crontab",
        "at",
        "useradd",
        "userdel",
        "usermod",
        "groupadd",
        "groupdel",
        "passwd",
        "visudo",
    }
)

GIT_READ_ONLY_SUBCOMMANDS = frozenset(
    {
        "status",
        "log",
        "diff",
        "show",
        "branch",
        "tag",
        "stash",
        "remote",
        "fetch",
        "ls-files",
        "ls-tree",
        "cat-file",
        "rev-parse",
        "describe",
        "shortlog",
        "blame",
        "bisect",
        "reflog",
        "config",
    }
)

DESTRUCTIVE_PATTERNS: list[tuple[str, str]] = [
    ("rm -rf /", "Recursive forced deletion at root"),
    ("rm -rf ~", "Recursive forced deletion of home directory"),
    ("rm -rf *", "Recursive forced deletion of all files"),
    ("rm -rf .", "Recursive forced deletion of current directory"),
    ("mkfs", "Filesystem creation destroys existing data"),
    ("dd if=", "Direct disk write can overwrite partitions"),
    ("> /dev/sd", "Writing to raw disk device"),
    ("chmod -R 777", "Recursively setting world-writable permissions"),
    ("chmod -R 000", "Recursively removing all permissions"),
    (":(){ :|:& };:", "Fork bomb"),
]

ALWAYS_DESTRUCTIVE_COMMANDS = frozenset({"shred", "wipefs"})
WRITE_REDIRECTIONS = (">", ">>")
SYSTEM_PATHS = (
    "/etc/",
    "/usr/",
    "/var/",
    "/boot/",
    "/sys/",
    "/proc/",
    "/dev/",
    "/sbin/",
    "/lib/",
    "/opt/",
    "C:\\Windows\\",
    "C:\\Program Files",
)


# ============================================================================
# Hilfsfunktionen
# ============================================================================


def extract_first_command(command: str) -> str:
    """Extrahiert den ersten Befehl, ueberspringt ENV=val Prefixes."""
    for part in command.strip().split():
        if "=" in part and part.split("=", 1)[0].replace("_", "").isalnum():
            continue
        return part
    return ""


def classify_command(command: str) -> CommandIntent:
    """Klassifiziert ein Shell-Command nach Absicht."""
    first = extract_first_command(command)
    if not first:
        return CommandIntent.UNKNOWN

    if first in ALWAYS_DESTRUCTIVE_COMMANDS or first == "rm":
        return CommandIntent.DESTRUCTIVE
    if first in READ_ONLY_COMMANDS:
        if first == "sed" and " -i" in command:
            return CommandIntent.WRITE
        return CommandIntent.READ_ONLY
    if first in WRITE_COMMANDS:
        return CommandIntent.WRITE
    if first in NETWORK_COMMANDS:
        return CommandIntent.NETWORK
    if first in PROCESS_COMMANDS:
        return CommandIntent.PROCESS_MANAGEMENT
    if first in PACKAGE_COMMANDS:
        return CommandIntent.PACKAGE_MANAGEMENT
    if first in SYSTEM_ADMIN_COMMANDS:
        return CommandIntent.SYSTEM_ADMIN
    if first == "git":
        parts = command.split()
        sub = next((p for p in parts[1:] if not p.startswith("-")), None)
        if sub and sub in GIT_READ_ONLY_SUBCOMMANDS:
            return CommandIntent.READ_ONLY
        return CommandIntent.WRITE
    return CommandIntent.UNKNOWN


# ============================================================================
# 6-Stufen-Pipeline
# ============================================================================


def validate_command(
    command: str,
    permission_mode: str = "full_access",
    workspace: str = "",
) -> ValidationResult:
    """Validiert ein Shell-Command in 6 Stufen.

    Args:
        command: Der zu pruefende Shell-Befehl.
        permission_mode: "read_only" | "workspace_write" | "full_access"
        workspace: Workspace-Root-Pfad fuer Path-Validation.

    Returns:
        ValidationResult mit verdict, reason, stage und intent.
    """
    intent = classify_command(command)

    stages: list[tuple[str, Any]] = [
        ("read_only", lambda: _validate_read_only(command, permission_mode)),
        ("mode", lambda: _validate_mode(command, permission_mode)),
        ("sed", lambda: _validate_sed(command, permission_mode)),
        ("destructive", lambda: _check_destructive(command)),
        ("path", lambda: _validate_paths(command, workspace)),
    ]

    for stage_name, stage_fn in stages:
        result = stage_fn()
        if result.verdict != ValidationVerdict.ALLOW:
            result.stage = stage_name
            result.intent = intent
            return cast("ValidationResult", result)

    return ValidationResult(
        verdict=ValidationVerdict.ALLOW,
        stage="semantics",
        intent=intent,
    )


def _validate_read_only(command: str, mode: str) -> ValidationResult:
    if mode != "read_only":
        return ValidationResult(ValidationVerdict.ALLOW)

    first = extract_first_command(command)
    if first in WRITE_COMMANDS:
        return ValidationResult(
            ValidationVerdict.BLOCK,
            f"'{first}' modifies filesystem -- blocked in read-only mode",
        )
    if first in STATE_MODIFYING_COMMANDS:
        return ValidationResult(
            ValidationVerdict.BLOCK,
            f"'{first}' modifies system state -- blocked in read-only mode",
        )
    if first == "sudo":
        inner = command.split("sudo", 1)[1].strip()
        return _validate_read_only(inner, mode)

    for redir in WRITE_REDIRECTIONS:
        if redir in command:
            return ValidationResult(
                ValidationVerdict.BLOCK,
                f"Write redirection '{redir}' blocked in read-only mode",
            )

    if first == "git":
        parts = command.split()
        sub = next((p for p in parts[1:] if not p.startswith("-")), None)
        if sub and sub not in GIT_READ_ONLY_SUBCOMMANDS:
            return ValidationResult(
                ValidationVerdict.BLOCK,
                f"git {sub} modifies repo -- blocked in read-only mode",
            )

    return ValidationResult(ValidationVerdict.ALLOW)


def _validate_mode(command: str, mode: str) -> ValidationResult:
    if mode != "workspace_write":
        return ValidationResult(ValidationVerdict.ALLOW)

    first = extract_first_command(command)
    if first not in WRITE_COMMANDS and first not in STATE_MODIFYING_COMMANDS:
        return ValidationResult(ValidationVerdict.ALLOW)

    for sys_path in SYSTEM_PATHS:
        if sys_path in command:
            return ValidationResult(
                ValidationVerdict.WARN,
                f"Command targets '{sys_path}' -- requires elevated permission",
            )

    return ValidationResult(ValidationVerdict.ALLOW)


def _validate_sed(command: str, mode: str) -> ValidationResult:
    if extract_first_command(command) != "sed":
        return ValidationResult(ValidationVerdict.ALLOW)
    if mode == "read_only" and " -i" in command:
        return ValidationResult(
            ValidationVerdict.BLOCK,
            "sed -i blocked in read-only mode",
        )
    return ValidationResult(ValidationVerdict.ALLOW)


def _check_destructive(command: str) -> ValidationResult:
    for pattern, warning in DESTRUCTIVE_PATTERNS:
        if pattern in command:
            return ValidationResult(
                ValidationVerdict.WARN,
                f"Destructive: {warning}",
            )
    first = extract_first_command(command)
    if first in ALWAYS_DESTRUCTIVE_COMMANDS:
        return ValidationResult(
            ValidationVerdict.WARN,
            f"'{first}' is inherently destructive",
        )
    if "rm " in command and "-r" in command and "-f" in command:
        return ValidationResult(
            ValidationVerdict.WARN,
            "Recursive forced deletion -- verify target",
        )
    return ValidationResult(ValidationVerdict.ALLOW)


def _validate_paths(command: str, workspace: str) -> ValidationResult:
    if not workspace:
        return ValidationResult(ValidationVerdict.ALLOW)

    if "../" in command and workspace not in command:
        return ValidationResult(
            ValidationVerdict.WARN,
            "Directory traversal '../' -- verify target is within workspace",
        )
    if "~/" in command or "$HOME" in command:
        return ValidationResult(
            ValidationVerdict.WARN,
            "Home directory reference -- verify workspace scope",
        )
    return ValidationResult(ValidationVerdict.ALLOW)
