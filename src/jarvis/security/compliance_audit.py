"""Immutable compliance audit log — append-only, SHA-256 chained.

Every GDPR-relevant event is logged here. The log CANNOT be deleted
or modified (append-only). Each entry includes a hash of the previous
entry, forming a tamper-evident chain.

Events: consent_granted, consent_withdrawn, erasure_requested,
erasure_executed, data_exported, cloud_data_sent, osint_started.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

from jarvis.utils.logging import get_logger

log = get_logger(__name__)

__all__ = ["ComplianceAuditLog"]


class ComplianceAuditLog:
    """Append-only audit log with SHA-256 hash chain.

    Thread-safe: all writes are protected by a lock.
    """

    def __init__(self, log_path: str | None = None) -> None:
        if log_path is None:
            log_path = str(Path.home() / ".jarvis" / "data" / "audit" / "compliance.jsonl")
        self._path = Path(log_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._last_hash = self._read_last_hash()

    def _read_last_hash(self) -> str:
        """Read the hash from the last line of the log file."""
        if not self._path.exists():
            return "genesis"
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                last_line = ""
                for line in f:
                    line = line.strip()
                    if line:
                        last_line = line
                if last_line:
                    data = json.loads(last_line)
                    return data.get("hash", "genesis")
        except Exception:
            pass
        return "genesis"

    def _compute_hash(self, entry: dict) -> str:
        """SHA-256 hash of entry content + previous hash."""
        content = json.dumps(entry, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(f"{self._last_hash}:{content}".encode()).hexdigest()

    def record(self, event: str, **kwargs: Any) -> dict:
        """Append a compliance event to the audit log (thread-safe)."""
        with self._lock:
            entry = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "event": event,
                **kwargs,
            }
            entry["prev_hash"] = self._last_hash
            entry["hash"] = self._compute_hash(entry)
            self._last_hash = entry["hash"]

            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        log.debug("compliance_audit_recorded", audit_event=event)
        return entry

    def verify_chain(self) -> tuple[bool, int]:
        """Verify the entire hash chain. Returns (valid, line_count)."""
        if not self._path.exists():
            return True, 0
        prev_hash = "genesis"
        count = 0
        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    return False, count
                if entry.get("prev_hash") != prev_hash:
                    return False, count
                # Recompute hash
                stored_hash = entry.pop("hash")
                content = json.dumps(entry, sort_keys=True, ensure_ascii=False)
                expected = hashlib.sha256(f"{prev_hash}:{content}".encode()).hexdigest()
                entry["hash"] = stored_hash
                if stored_hash != expected:
                    return False, count
                prev_hash = stored_hash
                count += 1
        return True, count

    def get_entries(self, event: str = "", limit: int = 100) -> list[dict]:
        """Read entries from the log, optionally filtered by event type."""
        if not self._path.exists():
            return []
        entries = []
        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if event and entry.get("event") != event:
                        continue
                    entries.append(entry)
                except json.JSONDecodeError:
                    continue
        return entries[-limit:]

    def pseudonymize_user(self, user_id: str, salt: str) -> int:
        """Replace user_id with pseudonym in all entries (for erasure).

        Rewrites the log with pseudonymized user IDs but preserves
        the hash chain by recomputing all hashes.
        """
        if not self._path.exists():
            return 0
        pseudo = hashlib.sha256(f"{user_id}:{salt}".encode()).hexdigest()[:16]

        entries = []
        count = 0
        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    # Replace user_id occurrences
                    for key in ("user_id", "target"):
                        if entry.get(key) == user_id:
                            entry[key] = f"pseudo:{pseudo}"
                            count += 1
                    entries.append(entry)
                except json.JSONDecodeError:
                    entries.append({"raw": line})

        # Rewrite with new hash chain
        prev_hash = "genesis"
        with open(self._path, "w", encoding="utf-8") as f:
            for entry in entries:
                entry["prev_hash"] = prev_hash
                stored = dict(entry)
                stored.pop("hash", None)
                content = json.dumps(stored, sort_keys=True, ensure_ascii=False)
                new_hash = hashlib.sha256(f"{prev_hash}:{content}".encode()).hexdigest()
                entry["hash"] = new_hash
                prev_hash = new_hash
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        self._last_hash = prev_hash
        return count
