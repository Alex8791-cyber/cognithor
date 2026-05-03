"""Tool-Loop-Detection — SHA-256-basierte Endlosschleifen-Erkennung.

Erkennt zwei Muster:
  - generic_repeat: Gleiches Tool + Args + Result N-mal hintereinander
  - ping_pong: Alternation zwischen 2 Tools ohne neue Information

Sliding-Window von 24 Eintraegen. Nur ueberwachte Tools werden gehasht.

Bibel-Referenz: Phase 3, Verbesserung 1 (HybridClaw).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Literal

TOOL_CALL_HISTORY_SIZE = 24
NO_PROGRESS_REPEAT_THRESHOLD = 4
PING_PONG_THRESHOLD = 6

GUARDED_TOOL_NAMES = frozenset(
    {
        "fs_read",
        "read_file",
        "vault_read",
        "vault_search",
        "memory_search",
        "search_memory",
        "web_search",
        "web_fetch",
        "search_and_read",
        "shell_exec",
        "exec_command",
        "list_directory",
    }
)


# ============================================================================
# Datenmodell
# ============================================================================


@dataclass
class ToolCallHistoryEntry:
    tool_name: str
    args_hash: str
    result_hash: str
    timestamp: float


@dataclass
class ToolLoopDetected:
    stuck: bool
    detector: Literal["generic_repeat", "ping_pong"] | None = None
    count: int = 0
    message: str = ""


# ============================================================================
# Hashing
# ============================================================================


def _stable_stringify(value: object) -> str:
    """Deterministisches JSON — Schluessel sortiert."""
    if value is None or isinstance(value, str | int | float | bool):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "[" + ",".join(_stable_stringify(v) for v in value) + "]"
    if isinstance(value, dict):
        keys = sorted(value.keys())
        return "{" + ",".join(f"{json.dumps(k)}:{_stable_stringify(value[k])}" for k in keys) + "}"
    return str(value)


def _sha256(value: object) -> str:
    return hashlib.sha256(_stable_stringify(value).encode()).hexdigest()[:16]


def _hash_call(tool_name: str, args: dict[str, Any]) -> str:
    return f"{tool_name}:{_sha256(args)}"


def _hash_outcome(output: str, is_error: bool) -> str:
    return _sha256({"is_error": is_error, "output": output[:2000]})


# ============================================================================
# Detector
# ============================================================================


class ToolLoopDetector:
    """Erkennt Tool-Endlosschleifen per SHA-256 Sliding-Window."""

    def __init__(self) -> None:
        self._history: list[ToolCallHistoryEntry] = []

    def record(self, tool_name: str, args: dict[str, Any], output: str, is_error: bool) -> None:
        """Zeichnet einen Tool-Call auf."""
        self._history.append(
            ToolCallHistoryEntry(
                tool_name=tool_name,
                args_hash=_hash_call(tool_name, args),
                result_hash=_hash_outcome(output, is_error),
                timestamp=time.time(),
            )
        )
        if len(self._history) > TOOL_CALL_HISTORY_SIZE:
            self._history = self._history[-TOOL_CALL_HISTORY_SIZE:]

    def detect(self, tool_name: str, args: dict[str, Any]) -> ToolLoopDetected:
        """Prueft ob der naechste Call eine Schleife waere."""
        if tool_name not in GUARDED_TOOL_NAMES:
            return ToolLoopDetected(stuck=False)

        signature = _hash_call(tool_name, args)

        # Check 1: Generic Repeat
        streak = self._no_progress_streak(tool_name, signature)
        if streak + 1 >= NO_PROGRESS_REPEAT_THRESHOLD:
            return ToolLoopDetected(
                stuck=True,
                detector="generic_repeat",
                count=streak + 1,
                message=(
                    f"Tool-Loop: {tool_name} wurde {streak + 1}x mit "
                    f"identischen Argumenten und identischem Ergebnis "
                    f"aufgerufen. Verwende die vorhandenen Daten."
                ),
            )

        # Check 2: Ping-Pong
        pp_count, no_progress = self._ping_pong_streak(signature)
        if no_progress and pp_count >= PING_PONG_THRESHOLD:
            return ToolLoopDetected(
                stuck=True,
                detector="ping_pong",
                count=pp_count,
                message=(
                    f"Ping-Pong: {pp_count} abwechselnde Aufrufe ohne "
                    f"neue Information. Verwende vorhandene Daten."
                ),
            )

        return ToolLoopDetected(stuck=False)

    def reset(self) -> None:
        """Setzt die History zurueck (neuer Request)."""
        self._history.clear()

    @property
    def history_size(self) -> int:
        return len(self._history)

    # ------------------------------------------------------------------

    def _no_progress_streak(self, tool_name: str, signature: str) -> int:
        count = 0
        latest_result: str | None = None
        for entry in reversed(self._history):
            if entry.tool_name != tool_name or entry.args_hash != signature:
                continue
            if latest_result is None:
                latest_result = entry.result_hash
                count = 1
                continue
            if entry.result_hash != latest_result:
                break
            count += 1
        return count

    def _ping_pong_streak(self, current_signature: str) -> tuple[int, bool]:
        if not self._history:
            return (0, False)

        last = self._history[-1]
        other_sig: str | None = None
        for entry in reversed(self._history[:-1]):
            if entry.args_hash != last.args_hash and entry.tool_name in GUARDED_TOOL_NAMES:
                other_sig = entry.args_hash
                break

        if not other_sig or current_signature != other_sig:
            return (0, False)

        alt_count = 0
        for i in range(len(self._history) - 1, -1, -1):
            expected = last.args_hash if alt_count % 2 == 0 else other_sig
            if self._history[i].args_hash != expected:
                break
            alt_count += 1

        if alt_count < 2:
            return (0, False)

        # No-Progress-Evidence
        tail_start = max(0, len(self._history) - alt_count)
        hash_a: str | None = None
        hash_b: str | None = None
        no_progress = True
        for entry in self._history[tail_start:]:
            if entry.args_hash == last.args_hash:
                if hash_a is None:
                    hash_a = entry.result_hash
                elif hash_a != entry.result_hash:
                    no_progress = False
                    break
            elif entry.args_hash == other_sig:
                if hash_b is None:
                    hash_b = entry.result_hash
                elif hash_b != entry.result_hash:
                    no_progress = False
                    break

        return (alt_count + 1, no_progress)
