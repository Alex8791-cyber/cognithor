"""WORM (Write Once Read Many) Storage Backend for Audit Compliance.

Uploads daily JSONL audit files to S3 or MinIO with Object Lock
(Compliance mode). Files locked in Compliance mode cannot be deleted
or overwritten until the retention period expires — even by the root
account.

boto3 is an optional dependency. If not installed, upload_daily()
logs a warning and returns an empty list.

Bible reference: §3.5 (Audit & Compliance)
"""

from __future__ import annotations

import logging
import os
import sqlite3
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from jarvis.config import AuditConfig

log = logging.getLogger(__name__)

# Optional boto3 import
try:
    import boto3  # type: ignore[import-untyped]
    from botocore.exceptions import ClientError  # type: ignore[import-untyped]

    _HAS_BOTO3 = True
except ImportError:
    _HAS_BOTO3 = False
    boto3 = None  # type: ignore[assignment]
    ClientError = Exception  # fallback for type hints

_STATE_DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS worm_uploads (
    filename TEXT PRIMARY KEY,
    uploaded_at REAL NOT NULL,
    bucket TEXT NOT NULL,
    key TEXT NOT NULL,
    retention_until TEXT NOT NULL
)
"""


class WORMUploader:
    """Upload audit JSONL files to S3/MinIO with Object Lock retention.

    Usage::

        uploader = WORMUploader(config.audit, config.jarvis_home)
        uploaded = uploader.upload_daily(audit_dir)
    """

    def __init__(self, config: AuditConfig, jarvis_home: Path) -> None:
        self._backend: str = config.worm_backend
        self._bucket: str = config.worm_bucket
        self._retention_days: int = config.worm_retention_days
        self._jarvis_home = jarvis_home

        # State tracking DB
        self._db_path = jarvis_home / "worm_state.db"
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

        # boto3 client (lazy — None if boto3 missing or backend is "none")
        self._s3_client: Any = None
        if self._backend != "none" and _HAS_BOTO3:
            self._s3_client = self._create_client()

    # ── Public API ──────────────────────────────────────────────

    def upload_daily(self, audit_dir: Path) -> list[str]:
        """Upload all un-uploaded JSONL files with Object Lock retention.

        Args:
            audit_dir: Directory containing ``audit_YYYY-MM-DD.jsonl`` files.

        Returns:
            List of uploaded filenames.
        """
        if self._backend == "none":
            return []

        if not _HAS_BOTO3:
            log.warning(
                "worm_boto3_missing: boto3 not installed — skipping WORM upload (pip install boto3)"
            )
            return []

        if not self._s3_client:
            log.warning("worm_client_unavailable: S3 client not initialised")
            return []

        if not audit_dir.is_dir():
            log.debug("worm_no_audit_dir: %s does not exist", audit_dir)
            return []

        uploaded: list[str] = []
        already = self._uploaded_set()

        for jsonl_file in sorted(audit_dir.glob("audit_*.jsonl")):
            fname = jsonl_file.name
            if fname in already:
                continue

            try:
                self._upload_file(jsonl_file)
                uploaded.append(fname)
                log.info(
                    "worm_uploaded",
                    filename=fname,
                    bucket=self._bucket,
                    backend=self._backend,
                )
            except Exception:
                log.error("worm_upload_failed", filename=fname, exc_info=True)

        return uploaded

    def list_uploaded(self) -> list[dict[str, Any]]:
        """List all uploaded files with their retention dates."""
        conn = sqlite3.connect(str(self._db_path))
        try:
            rows = conn.execute(
                "SELECT filename, uploaded_at, bucket, key, retention_until "
                "FROM worm_uploads ORDER BY uploaded_at"
            ).fetchall()
        finally:
            conn.close()

        return [
            {
                "filename": r[0],
                "uploaded_at": r[1],
                "bucket": r[2],
                "key": r[3],
                "retention_until": r[4],
            }
            for r in rows
        ]

    def verify_lock(self, key: str) -> dict[str, Any]:
        """Check if a specific file still has its retention lock.

        Args:
            key: The S3 object key to check.

        Returns:
            Dict with lock status, mode, and retain_until_date.
        """
        if not _HAS_BOTO3 or not self._s3_client:
            return {"locked": False, "error": "boto3 not available"}

        try:
            resp = self._s3_client.get_object_retention(Bucket=self._bucket, Key=key)
            retention = resp.get("Retention", {})
            return {
                "locked": True,
                "mode": retention.get("Mode", ""),
                "retain_until_date": str(retention.get("RetainUntilDate", "")),
            }
        except ClientError as exc:
            return {"locked": False, "error": str(exc)}

    # ── Internal ────────────────────────────────────────────────

    def _create_client(self) -> Any:
        """Create a boto3 S3 client, using custom endpoint for MinIO."""
        kwargs: dict[str, Any] = {"service_name": "s3"}

        if self._backend == "minio":
            endpoint = os.environ.get("MINIO_ENDPOINT_URL", "http://localhost:9000")
            kwargs["endpoint_url"] = endpoint
            log.debug("worm_minio_endpoint: %s", endpoint)

        return boto3.client(**kwargs)

    def _upload_file(self, path: Path) -> None:
        """Upload a single JSONL file with Object Lock Compliance retention."""
        now = datetime.now(UTC)
        retain_until = now + timedelta(days=self._retention_days)
        key = f"audit/{path.name}"

        with path.open("rb") as fh:
            self._s3_client.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=fh,
                ContentType="application/x-ndjson",
                ObjectLockMode="COMPLIANCE",
                ObjectLockRetainUntilDate=retain_until,
            )

        self._record_upload(
            filename=path.name,
            bucket=self._bucket,
            key=key,
            retention_until=retain_until.isoformat(),
        )

    def _init_db(self) -> None:
        """Ensure the state DB and table exist."""
        conn = sqlite3.connect(str(self._db_path))
        try:
            conn.execute(_STATE_DB_SCHEMA)
            conn.commit()
        finally:
            conn.close()

    def _record_upload(
        self,
        *,
        filename: str,
        bucket: str,
        key: str,
        retention_until: str,
    ) -> None:
        """Record a successful upload in the state DB."""
        conn = sqlite3.connect(str(self._db_path))
        try:
            conn.execute(
                "INSERT OR REPLACE INTO worm_uploads "
                "(filename, uploaded_at, bucket, key, retention_until) "
                "VALUES (?, ?, ?, ?, ?)",
                (filename, time.time(), bucket, key, retention_until),
            )
            conn.commit()
        finally:
            conn.close()

    def _uploaded_set(self) -> set[str]:
        """Return the set of already-uploaded filenames."""
        conn = sqlite3.connect(str(self._db_path))
        try:
            rows = conn.execute("SELECT filename FROM worm_uploads").fetchall()
        finally:
            conn.close()
        return {r[0] for r in rows}
