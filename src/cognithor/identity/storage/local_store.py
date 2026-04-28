"""
storage/local_store.py

Local JSON storage — for fallback and development environments.

Stores all snapshots on the local filesystem.
Continues to work even without an internet connection.
"""

import contextlib
import json
import logging
import os
import re as _re
from datetime import UTC, datetime

_SAFE_ID_RE = _re.compile(r"[^a-zA-Z0-9_\-]")

logger = logging.getLogger(__name__)


class LocalStore:
    """
    Local filesystem-based storage.

    Stores snapshots in JSON format.
    Each snapshot is linked via a hash chain.

    Parameters:
        base_dir: Storage directory
    """

    def __init__(self, base_dir: str = "data/local_snapshots") -> None:
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        logger.info(f"LocalStore initialized: {base_dir}")

    def save_snapshot(self, snapshot: dict, identity_id: str) -> dict:
        """
        Save a snapshot to a local file.

        Parameters:
            snapshot: Data to save
            identity_id: AI identity ID

        Returns:
            dict: {'uri': str, 'filepath': str, 'hash': str}
        """
        import hashlib
        import uuid

        # Timestamp has 1-second resolution; rapid back-to-back saves with the
        # same identity_id would collide and silently overwrite each other.
        # Append a short uuid4 fragment to guarantee a unique filename.
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        unique_suffix = uuid.uuid4().hex[:8]
        safe_id = _SAFE_ID_RE.sub("_", identity_id[:16])
        filename = f"{safe_id}_{timestamp}_{unique_suffix}.json"
        filepath = os.path.join(self.base_dir, filename)

        content = json.dumps(snapshot, ensure_ascii=False, indent=2)
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        with contextlib.suppress(NotImplementedError):
            os.chmod(filepath, 0o600)

        uri = f"local://{filepath}"
        logger.info(f"Snapshot saved: {filepath}")

        return {
            "uri": uri,
            "filepath": filepath,
            "hash": content_hash,
            "timestamp": timestamp,
        }

    def load_snapshot(self, uri: str) -> dict | None:
        """
        Load a local snapshot.

        Parameters:
            uri: local:// URI or direct file path

        Returns:
            dict: Snapshot data or None
        """
        filepath = uri.replace("local://", "")

        # Prevent path traversal: resolve and confirm within base_dir
        safe_base = os.path.realpath(self.base_dir)
        safe_path = os.path.realpath(filepath)
        if not safe_path.startswith(safe_base + os.sep):
            logger.warning("Path traversal attempt blocked: %s", filepath)
            return None

        if not os.path.exists(filepath):
            logger.warning(f"Snapshot not found: {filepath}")
            return None

        try:
            with open(filepath, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Snapshot could not be loaded: {e}")
            return None

    def list_snapshots(self, identity_id: str | None = None) -> list[dict]:
        """
        List available snapshots.

        Parameters:
            identity_id: Identity ID to filter by (None = all)

        Returns:
            list[dict]: List of snapshot metadata
        """
        snapshots = []
        for filename in os.listdir(self.base_dir):
            if not filename.endswith(".json"):
                continue
            if identity_id and not filename.startswith(identity_id[:16]):
                continue

            filepath = os.path.join(self.base_dir, filename)
            stat = os.stat(filepath)
            snapshots.append(
                {
                    "filename": filename,
                    "filepath": filepath,
                    "uri": f"local://{filepath}",
                    "size_bytes": stat.st_size,
                    "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                }
            )

        return sorted(snapshots, key=lambda x: x["modified_at"], reverse=True)

    def get_latest_snapshot(self, identity_id: str) -> dict | None:
        """Get the most recent snapshot."""
        snapshots = self.list_snapshots(identity_id)
        if not snapshots:
            return None
        return self.load_snapshot(snapshots[0]["uri"])

    def cleanup_old_snapshots(self, keep_last: int = 10, identity_id: str | None = None) -> int:
        """
        Clean up old snapshots, keeping the last N.

        Parameters:
            keep_last: How many snapshots to retain
            identity_id: Identity ID to filter by

        Returns:
            int: Number of deleted files
        """
        snapshots = self.list_snapshots(identity_id)
        to_delete = snapshots[keep_last:]

        deleted = 0
        for snapshot in to_delete:
            try:
                os.remove(snapshot["filepath"])
                deleted += 1
            except Exception as e:
                logger.warning(f"Snapshot could not be deleted: {e}")

        logger.info(f"LocalStore cleanup: {deleted} snapshots deleted")
        return deleted
