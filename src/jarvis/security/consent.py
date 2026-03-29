"""GDPR Consent Manager — per-channel consent tracking with versioning."""

from __future__ import annotations

import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

from jarvis.utils.logging import get_logger

log = get_logger(__name__)

# Late import to avoid circular deps
_audit_log: Any = None


def _get_audit_log() -> Any:
    """Lazy-load the compliance audit log."""
    global _audit_log
    if _audit_log is None:
        try:
            from jarvis.security.compliance_audit import ComplianceAuditLog

            _audit_log = ComplianceAuditLog()
        except Exception:
            pass
    return _audit_log


__all__ = ["ConsentManager"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS consent (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    channel TEXT NOT NULL,
    consent_type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'accepted',
    policy_version TEXT DEFAULT '1.0',
    granted_at TEXT,
    withdrawn_at TEXT,
    context TEXT DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_consent_user
    ON consent(user_id, channel, consent_type);
CREATE INDEX IF NOT EXISTS idx_consent_status
    ON consent(user_id, status);
"""


class ConsentManager:
    """Track per-user, per-channel GDPR consent with versioning."""

    def __init__(self, db_path: str | None = None) -> None:
        if db_path is None:
            db_path = str(Path.home() / ".jarvis" / "index" / "consent.db")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            from jarvis.security.encrypted_db import encrypted_connect

            self._conn = encrypted_connect(db_path, check_same_thread=False)
        except ImportError:
            self._conn = sqlite3.connect(db_path, check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def has_consent(
        self,
        user_id: str,
        channel: str,
        consent_type: str = "data_processing",
        policy_version: str | None = None,
    ) -> bool:
        """Check if user has active consent for this channel and type."""
        query = (
            "SELECT 1 FROM consent "
            "WHERE user_id = ? AND channel = ? AND consent_type = ? AND status = 'accepted'"
        )
        params: list = [user_id, channel, consent_type]
        if policy_version:
            query += " AND policy_version = ?"
            params.append(policy_version)
        query += " LIMIT 1"
        row = self._conn.execute(query, params).fetchone()
        return row is not None

    def grant_consent(
        self,
        user_id: str,
        channel: str,
        consent_type: str = "data_processing",
        context: str = "",
        policy_version: str = "1.0",
    ) -> None:
        """Record that user granted consent."""
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        # Upsert: withdraw any existing, then insert new
        self._conn.execute(
            "UPDATE consent SET status = 'superseded', withdrawn_at = ? "
            "WHERE user_id = ? AND channel = ? AND consent_type = ? AND status = 'accepted'",
            (now, user_id, channel, consent_type),
        )
        self._conn.execute(
            "INSERT INTO consent (id, user_id, channel, consent_type, status, "
            "policy_version, granted_at, context, created_at) "
            "VALUES (?, ?, ?, ?, 'accepted', ?, ?, ?, ?)",
            (
                uuid.uuid4().hex[:16],
                user_id,
                channel,
                consent_type,
                policy_version,
                now,
                context,
                now,
            ),
        )
        self._conn.commit()
        log.info("consent_granted", user_id=user_id[:8], channel=channel, type=consent_type)
        # Compliance audit log
        audit = _get_audit_log()
        if audit:
            audit.record(
                "consent_granted",
                user_id=user_id,
                channel=channel,
                consent_type=consent_type,
                policy_version=policy_version,
            )

    def withdraw_consent(
        self,
        user_id: str,
        channel: str,
        consent_type: str = "data_processing",
    ) -> None:
        """Record consent withdrawal."""
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._conn.execute(
            "UPDATE consent SET status = 'withdrawn', withdrawn_at = ? "
            "WHERE user_id = ? AND channel = ? AND consent_type = ? AND status = 'accepted'",
            (now, user_id, channel, consent_type),
        )
        self._conn.commit()
        log.info("consent_withdrawn", user_id=user_id[:8], channel=channel, type=consent_type)
        # Compliance audit log
        audit = _get_audit_log()
        if audit:
            audit.record(
                "consent_withdrawn", user_id=user_id, channel=channel, consent_type=consent_type
            )

    def requires_consent(self, user_id: str, channel: str) -> bool:
        """Check if user still needs to give consent for this channel."""
        return not self.has_consent(user_id, channel, "data_processing")

    def get_user_consents(self, user_id: str) -> list[dict]:
        """Return all consent records for a user."""
        cursor = self._conn.execute(
            "SELECT * FROM consent WHERE user_id = ? AND status = 'accepted' "
            "ORDER BY created_at DESC",
            (user_id,),
        )
        columns = [d[0] for d in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def delete_user(self, user_id: str) -> int:
        """Delete all consent records for a user (for erasure)."""
        cursor = self._conn.execute("DELETE FROM consent WHERE user_id = ?", (user_id,))
        self._conn.commit()
        return cursor.rowcount

    def restrict_purpose(self, user_id: str, channel: str, purpose: str) -> None:
        """Record that user restricts a specific processing purpose."""
        self.grant_consent(user_id, channel, f"restrict_{purpose}")

    def unrestrict_purpose(self, user_id: str, channel: str, purpose: str) -> None:
        """Remove a purpose restriction."""
        self.withdraw_consent(user_id, channel, f"restrict_{purpose}")

    def is_restricted(self, user_id: str, channel: str, purpose: str) -> bool:
        """Check if user has restricted a specific purpose."""
        return self.has_consent(user_id, channel, f"restrict_{purpose}")

    def get_restrictions(self, user_id: str) -> list[str]:
        """Get all active restrictions for a user."""
        consents = self.get_user_consents(user_id)
        return [
            c["consent_type"].replace("restrict_", "")
            for c in consents
            if c["consent_type"].startswith("restrict_")
        ]

    def close(self) -> None:
        self._conn.close()
