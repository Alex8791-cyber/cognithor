# Copyright 2024-2026 Cognithor Contributors
# Licensed under the Apache License, Version 2.0
"""Integration mit Cognithors SHA-256 Audit Chain."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

from cognithor.utils.logging import get_logger

if TYPE_CHECKING:
    from cognithor.hashline.models import EditIntent, EditResult

log = get_logger(__name__)


_GENESIS_PREV_HASH = "genesis"


class HashlineAuditor:
    """Append-only JSONL audit log for hashline operations.

    Writes audit entries to ``~/.cognithor/hashline_audit.jsonl``. Each entry
    is a single JSON line containing timestamp, file, operation, hashes,
    agent ID, and a ``prev_hash`` field linking back to the previous entry's
    SHA-256 — forming a tamper-evident chain. The first entry uses
    ``"genesis"`` as its ``prev_hash``. Entries are never modified or deleted.

    Use :meth:`verify_chain` to validate the entire on-disk log: it walks
    every entry in order, recomputes each line's SHA-256 (with the ``hash``
    field stripped to avoid self-reference), and confirms that every
    ``prev_hash`` matches the previous entry's recomputed hash. A break in
    the chain points to the line where tampering started.

    Args:
        data_dir: Base directory for audit files (defaults to ~/.cognithor).
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        self._data_dir = data_dir or Path.home() / ".cognithor"
        self._audit_file = self._data_dir / "hashline_audit.jsonl"
        self._lock = threading.Lock()
        # Cache of the last entry's hash so we can chain new appends
        # without scanning the whole file every time.
        self._last_hash: str | None = None

        # Ensure directory exists
        self._data_dir.mkdir(parents=True, exist_ok=True)

    def log_edit(
        self,
        result: EditResult,
        intent: EditIntent,
        agent_id: str,
    ) -> str:
        """Log an edit operation to the audit trail.

        Args:
            result: The result of the edit operation.
            intent: The original edit intent.
            agent_id: Identifier of the agent that performed the edit.

        Returns:
            SHA-256 hash of the audit entry.
        """
        entry = {
            "timestamp": time.time(),
            "type": "edit",
            "file": str(result.file_path),
            "line": result.line_number,
            "operation": result.operation,
            "old_hash": intent.target_hash,
            "new_hash": result.audit_hash,
            "agent_id": agent_id,
            "success": result.success,
            "error": result.error,
            "retry_count": result.retry_count,
        }
        return self._append(entry)

    def log_read(
        self,
        path: Path,
        line_count: int,
        agent_id: str,
    ) -> str:
        """Log a file read operation to the audit trail.

        Args:
            path: Path to the file that was read.
            line_count: Number of lines read.
            agent_id: Identifier of the agent that read the file.

        Returns:
            SHA-256 hash of the audit entry.
        """
        entry = {
            "timestamp": time.time(),
            "type": "read",
            "file": str(path),
            "line_count": line_count,
            "agent_id": agent_id,
        }
        return self._append(entry)

    def get_file_history(
        self,
        path: Path,
        limit: int = 50,
    ) -> list[dict]:
        """Retrieve audit history for a specific file.

        Args:
            path: Path to the file (matched as substring).
            limit: Maximum number of entries to return.

        Returns:
            List of audit entry dicts, newest first.
        """
        path_str = str(path)
        entries: list[dict] = []

        with self._lock:
            if not self._audit_file.exists():
                return []

            try:
                with open(self._audit_file, encoding="utf-8") as f:
                    for raw_line in f:
                        raw_line = raw_line.strip()
                        if not raw_line:
                            continue
                        try:
                            entry = json.loads(raw_line)
                            if entry.get("file", "").endswith(path_str) or path_str in entry.get(
                                "file", ""
                            ):
                                entries.append(entry)
                        except json.JSONDecodeError:
                            continue
            except OSError:
                log.debug("audit_read_failed", path=str(self._audit_file), exc_info=True)

        # Newest first, limited
        entries.reverse()
        return entries[:limit]

    def _resolve_last_hash(self) -> str:
        """Return the SHA-256 of the last on-disk entry, or genesis if empty.

        Walks the file once on first call (per process / per fresh auditor)
        and caches the result. Subsequent appends just chain off the cache.
        """
        if self._last_hash is not None:
            return self._last_hash
        if not self._audit_file.exists():
            self._last_hash = _GENESIS_PREV_HASH
            return self._last_hash
        last_line: str | None = None
        try:
            with open(self._audit_file, encoding="utf-8") as f:
                for raw in f:
                    raw = raw.strip()
                    if raw:
                        last_line = raw
        except OSError:
            log.debug("audit_last_hash_read_failed", path=str(self._audit_file), exc_info=True)
            self._last_hash = _GENESIS_PREV_HASH
            return self._last_hash
        if last_line is None:
            self._last_hash = _GENESIS_PREV_HASH
            return self._last_hash
        try:
            data = json.loads(last_line)
            recorded = data.get("hash")
            if isinstance(recorded, str) and recorded:
                self._last_hash = recorded
                return self._last_hash
            # Legacy entry without hash field — recompute over the line.
            self._last_hash = hashlib.sha256(last_line.encode("utf-8")).hexdigest()
            return self._last_hash
        except json.JSONDecodeError:
            self._last_hash = _GENESIS_PREV_HASH
            return self._last_hash

    def _append(self, entry: dict) -> str:
        """Append a chained entry to the audit file and return its SHA-256.

        Each entry carries a ``prev_hash`` that links back to the prior
        entry's SHA-256, plus its own ``hash`` field for verifiers. The first
        entry has ``prev_hash = "genesis"``.

        Args:
            entry: Dict to serialize as JSON and append. Caller-provided
                ``prev_hash``/``hash`` keys are overwritten.

        Returns:
            SHA-256 hex digest of the persisted JSON line.
        """
        with self._lock:
            prev = self._resolve_last_hash()
            entry["prev_hash"] = prev
            # Hash is computed over the entry WITHOUT the hash field so the
            # verifier can recompute the same value deterministically.
            line_for_hash = json.dumps(entry, ensure_ascii=False, sort_keys=True)
            entry_hash = hashlib.sha256(line_for_hash.encode("utf-8")).hexdigest()
            entry["hash"] = entry_hash
            persisted = json.dumps(entry, ensure_ascii=False, sort_keys=True)

            with open(self._audit_file, "a", encoding="utf-8") as f:
                f.write(persisted + "\n")
            self._last_hash = entry_hash

        log.debug("audit_logged", type=entry.get("type"), file=entry.get("file"))
        return entry_hash

    def verify_chain(self) -> dict:
        """Walk the on-disk audit log and confirm the prev_hash chain.

        Returns a dict ``{"status", "total_entries", "valid_entries",
        "broken_at_line", "log_file"}``. ``status`` is one of
        ``"intact"``, ``"empty"``, or ``"broken"``. On break, ``broken_at_line``
        points at the first 1-indexed entry whose ``prev_hash`` doesn't match
        the previous entry's recomputed hash, or whose own recomputed hash
        differs from the stored ``hash`` field.

        This is the same shape as the gatekeeper-chain verifier in
        ``/api/v1/audit/verify`` so a single endpoint can call both.
        """
        result = {
            "status": "intact",
            "total_entries": 0,
            "valid_entries": 0,
            "broken_at_line": None,
            "log_file": str(self._audit_file),
        }
        if not self._audit_file.exists():
            result["status"] = "empty"
            return result

        prev_hash = _GENESIS_PREV_HASH
        try:
            with open(self._audit_file, encoding="utf-8") as f:
                for line_no, raw in enumerate(f, 1):
                    raw = raw.strip()
                    if not raw:
                        continue
                    result["total_entries"] += 1
                    try:
                        entry = json.loads(raw)
                    except json.JSONDecodeError:
                        if result["broken_at_line"] is None:
                            result["broken_at_line"] = line_no
                            result["status"] = "broken"
                        continue
                    stored_prev = entry.get("prev_hash", "")
                    stored_hash = entry.get("hash", "")
                    if stored_prev != prev_hash and result["broken_at_line"] is None:
                        result["broken_at_line"] = line_no
                        result["status"] = "broken"
                    # Recompute hash over entry WITHOUT the hash field.
                    body = {k: v for k, v in entry.items() if k != "hash"}
                    canonical = json.dumps(body, ensure_ascii=False, sort_keys=True)
                    recomputed = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
                    if (
                        stored_hash
                        and stored_hash != recomputed
                        and result["broken_at_line"] is None
                    ):
                        result["broken_at_line"] = line_no
                        result["status"] = "broken"
                    if result["broken_at_line"] is None:
                        result["valid_entries"] += 1
                    prev_hash = stored_hash or recomputed
        except OSError as exc:
            log.debug("audit_verify_io_error", error=str(exc))
            result["status"] = "broken"
            result["broken_at_line"] = result.get("total_entries") or 1
        return result
