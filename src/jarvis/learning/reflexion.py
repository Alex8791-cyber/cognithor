"""Reflexion-based error learning — stores failures with prevention rules.

Inspired by SuperClaude's reflexion pattern. Enables Cognithor to learn
from tool execution errors and prevent recurring failures.

Storage: JSONL file for fast append + in-memory index for fast lookup.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jarvis.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class ReflexionEntry:
    """A recorded error with root cause analysis and prevention rule."""

    entry_id: str  # UUID
    timestamp: float  # time.time()
    tool_name: str  # Which tool failed
    error_category: str  # timeout, bad_parameters, hallucination, etc.
    error_message: str  # Original error message
    error_signature: str  # Normalized hash for grouping
    root_cause: str  # Why it failed
    prevention_rule: str  # How to prevent it
    task_context: str  # What the user was trying to do
    recurrence_count: int = 1  # How many times this pattern occurred
    status: str = "pending"  # pending, adopted, rejected
    solution: str = ""  # How it was fixed (if known)
    channel: str = ""  # Which channel it occurred on


class ReflexionMemory:
    """Stores and retrieves error patterns for learning.

    Uses JSONL for durable append-only storage and an in-memory index
    for fast signature-based lookup.
    """

    MAX_ENTRIES = 5000  # Prune beyond this

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._file = self._data_dir / "reflexion.jsonl"
        self._entries: dict[str, ReflexionEntry] = {}  # signature -> entry
        self._all_entries: list[ReflexionEntry] = []
        self._load()

    def _load(self) -> None:
        """Load existing entries from JSONL file."""
        if not self._file.exists():
            return
        try:
            for line in self._file.read_text(encoding="utf-8").strip().split("\n"):
                if not line.strip():
                    continue
                data = json.loads(line)
                entry = ReflexionEntry(
                    **{k: v for k, v in data.items() if k in ReflexionEntry.__dataclass_fields__}
                )
                self._entries[entry.error_signature] = entry
                self._all_entries.append(entry)
            log.info("reflexion_memory_loaded", entries=len(self._entries))
        except Exception:
            log.debug("reflexion_load_failed", exc_info=True)

    def _save_entry(self, entry: ReflexionEntry) -> None:
        """Append a single entry to the JSONL file."""
        try:
            with self._file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry.__dict__, ensure_ascii=False) + "\n")
        except Exception:
            log.debug("reflexion_save_failed", exc_info=True)

    def _rewrite_all(self) -> None:
        """Rewrite the entire JSONL file (after pruning or status changes)."""
        try:
            with self._file.open("w", encoding="utf-8") as f:
                for entry in self._all_entries:
                    f.write(json.dumps(entry.__dict__, ensure_ascii=False) + "\n")
        except Exception:
            log.debug("reflexion_rewrite_failed", exc_info=True)

    @staticmethod
    def compute_signature(tool_name: str, error_category: str, error_message: str) -> str:
        """Create a normalized hash for error grouping."""
        normalized = error_message.lower()
        # Order matters: timestamps and UUIDs before paths to avoid partial matches
        normalized = re.sub(r"\d{4}-\d{2}-\d{2}[tT ]\d{2}:\d{2}:\d{2}", "<ts>", normalized)
        normalized = re.sub(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            "<uuid>",
            normalized,
        )
        normalized = re.sub(r"0x[0-9a-f]+", "<hex>", normalized)
        normalized = re.sub(r"/\S+\.\w+", "<path>", normalized)
        normalized = re.sub(r"line \d+", "line N", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()

        raw = f"{tool_name}:{error_category}:{normalized}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def record_error(
        self,
        tool_name: str,
        error_category: str,
        error_message: str,
        root_cause: str,
        prevention_rule: str,
        task_context: str = "",
        solution: str = "",
        channel: str = "",
    ) -> ReflexionEntry:
        """Record a new error or increment recurrence of existing one."""
        signature = self.compute_signature(tool_name, error_category, error_message)

        existing = self._entries.get(signature)
        if existing:
            existing.recurrence_count += 1
            existing.timestamp = time.time()
            if solution and not existing.solution:
                existing.solution = solution
            self._rewrite_all()
            log.info(
                "reflexion_recurrence",
                signature=signature,
                count=existing.recurrence_count,
                tool=tool_name,
            )
            return existing

        entry = ReflexionEntry(
            entry_id=uuid.uuid4().hex[:12],
            timestamp=time.time(),
            tool_name=tool_name,
            error_category=error_category,
            error_message=error_message[:500],
            error_signature=signature,
            root_cause=root_cause,
            prevention_rule=prevention_rule,
            task_context=task_context[:300],
            channel=channel,
            solution=solution,
        )

        self._entries[signature] = entry
        self._all_entries.append(entry)
        self._save_entry(entry)
        log.info(
            "reflexion_recorded",
            signature=signature,
            tool=tool_name,
            category=error_category,
        )
        return entry

    def get_solution(
        self, tool_name: str, error_category: str, error_message: str
    ) -> ReflexionEntry | None:
        """Look up a known solution for this error pattern."""
        signature = self.compute_signature(tool_name, error_category, error_message)
        return self._entries.get(signature)

    def get_prevention_rules(self, tool_name: str | None = None) -> list[str]:
        """Get all prevention rules, optionally filtered by tool."""
        rules = []
        for entry in self._entries.values():
            if entry.status == "rejected":
                continue
            if tool_name and entry.tool_name != tool_name:
                continue
            if entry.prevention_rule:
                rules.append(entry.prevention_rule)
        return rules

    def get_recent_errors(self, limit: int = 20) -> list[ReflexionEntry]:
        """Get most recent error entries."""
        return sorted(self._all_entries, key=lambda e: e.timestamp, reverse=True)[:limit]

    def get_recurring_errors(self, min_count: int = 3) -> list[ReflexionEntry]:
        """Get errors that recur frequently."""
        return [e for e in self._entries.values() if e.recurrence_count >= min_count]

    def adopt_rule(self, signature: str) -> bool:
        """Mark a prevention rule as adopted (verified effective)."""
        entry = self._entries.get(signature)
        if entry:
            entry.status = "adopted"
            self._rewrite_all()
            return True
        return False

    def reject_rule(self, signature: str) -> bool:
        """Mark a prevention rule as rejected (not effective)."""
        entry = self._entries.get(signature)
        if entry:
            entry.status = "rejected"
            self._rewrite_all()
            return True
        return False

    def prune_old(self, older_than_days: int = 90) -> int:
        """Remove old entries beyond the limit."""
        cutoff = time.time() - older_than_days * 86400
        before = len(self._all_entries)
        self._all_entries = [
            e for e in self._all_entries if e.timestamp > cutoff or e.recurrence_count >= 3
        ]
        self._entries = {e.error_signature: e for e in self._all_entries}
        after = len(self._all_entries)
        if before != after:
            self._rewrite_all()
        return before - after

    def stats(self) -> dict[str, Any]:
        """Summary statistics."""
        return {
            "total_entries": len(self._all_entries),
            "unique_patterns": len(self._entries),
            "adopted_rules": sum(1 for e in self._entries.values() if e.status == "adopted"),
            "rejected_rules": sum(1 for e in self._entries.values() if e.status == "rejected"),
            "pending_rules": sum(1 for e in self._entries.values() if e.status == "pending"),
            "top_recurring": [
                {
                    "tool": e.tool_name,
                    "category": e.error_category,
                    "count": e.recurrence_count,
                }
                for e in sorted(
                    self._entries.values(),
                    key=lambda e: e.recurrence_count,
                    reverse=True,
                )[:5]
            ],
        }
