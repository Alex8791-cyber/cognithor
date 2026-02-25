from __future__ import annotations

import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from jarvis.models import PolicyChange
from jarvis.utils.logging import get_logger

logger = get_logger(__name__)


class PolicyPatcher:
    """Applies and rolls back policy changes to YAML policy files."""

    BACKUP_SUFFIX = ".bak"

    def __init__(self, policies_dir: str) -> None:
        self.policies_dir = Path(policies_dir)
        self._backup_stack: list[str] = []
        logger.info("PolicyPatcher initialized with policies_dir=%s", policies_dir)

    def apply_change(self, change: PolicyChange) -> bool:
        """Read the relevant YAML policy file, create a backup, apply the
        patch described in *change*, and write the result back.

        Returns True on success, False on failure.
        """
        try:
            category = change.category
            policy_file = self.policies_dir / f"{category}.yaml"

            # If the policy file does not exist yet, start from an empty dict
            if policy_file.exists():
                with open(policy_file, "r", encoding="utf-8") as fh:
                    policy_data: dict[str, Any] = yaml.safe_load(fh) or {}
            else:
                policy_data = {}

            # Create a timestamped backup before modifying
            backup_path = self._create_backup(policy_file)

            # Apply the change -- merge suggested change dict into the policy
            change_dict = change.change if isinstance(change.change, dict) else {}
            policy_data = self._merge_change(policy_data, change_dict)

            # Write the updated policy
            self.policies_dir.mkdir(parents=True, exist_ok=True)
            with open(policy_file, "w", encoding="utf-8") as fh:
                yaml.safe_dump(policy_data, fh, default_flow_style=False)

            logger.info(
                "Applied change for proposal #%d to %s (backup: %s)",
                change.proposal_id,
                policy_file,
                backup_path,
            )
            return True

        except Exception:
            logger.exception(
                "Failed to apply change for proposal #%d", change.proposal_id
            )
            return False

    def rollback_last(self) -> bool:
        """Restore the most recent backup, undoing the last applied change.

        Returns True on success, False if there is nothing to roll back or
        an error occurs.
        """
        if not self._backup_stack:
            logger.warning("No backups available to rollback")
            return False

        backup_path_str = self._backup_stack.pop()
        backup_path = Path(backup_path_str)

        if not backup_path.exists():
            logger.error("Backup file not found: %s", backup_path)
            return False

        try:
            # Derive the original file path from the backup name
            # Backup format: <original_stem>_<timestamp>.yaml.bak
            # We need to restore to <original_stem>.yaml
            original_name = backup_path.name
            # Remove the .bak suffix
            without_bak = original_name.rsplit(self.BACKUP_SUFFIX, 1)[0]
            # Remove the _<timestamp> portion to get the original filename
            # Format: category_20260224T120000Z.yaml -> category.yaml
            parts = without_bak.rsplit("_", 1)
            if len(parts) == 2 and "." in parts[1]:
                # timestamp part contains the extension, e.g. "20260224T120000Z.yaml"
                ext_parts = parts[1].split(".", 1)
                original_filename = parts[0] + "." + ext_parts[1]
            else:
                original_filename = without_bak

            original_path = self.policies_dir / original_filename

            # If backup is empty (original didn't exist), delete instead of restoring
            if backup_path.stat().st_size == 0:
                if original_path.exists():
                    original_path.unlink()
                logger.info("Rolled back: deleted %s (was non-existent before)", original_path)
            else:
                shutil.copy2(str(backup_path), str(original_path))
                logger.info("Rolled back to backup: %s -> %s", backup_path, original_path)
            return True

        except Exception:
            logger.exception("Failed to rollback from backup: %s", backup_path)
            return False

    def list_backups(self) -> list[str]:
        """Return a list of all backup file paths in the policies directory."""
        backups: list[str] = []
        if not self.policies_dir.exists():
            return backups

        for entry in sorted(self.policies_dir.iterdir()):
            if entry.name.endswith(self.BACKUP_SUFFIX):
                backups.append(str(entry))

        return backups

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_backup(self, policy_file: Path) -> str:
        """Create a timestamped backup of the given policy file.

        If the file does not exist yet an empty backup is created so that
        rollback can restore the 'no file' state.
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        stem = policy_file.stem
        suffix = policy_file.suffix  # e.g. ".yaml"
        backup_name = f"{stem}_{timestamp}{suffix}{self.BACKUP_SUFFIX}"
        backup_path = self.policies_dir / backup_name

        self.policies_dir.mkdir(parents=True, exist_ok=True)

        if policy_file.exists():
            shutil.copy2(str(policy_file), str(backup_path))
        else:
            # Create an empty backup so rollback knows the file didn't exist
            backup_path.touch()

        self._backup_stack.append(str(backup_path))
        logger.debug("Created backup: %s", backup_path)
        return str(backup_path)

    @staticmethod
    def _merge_change(
        policy_data: dict[str, Any], change: dict[str, Any]
    ) -> dict[str, Any]:
        """Merge *change* into *policy_data*.

        Keys from *change* are written into a top-level ``changes`` list
        within the policy so the full change record is preserved.  The
        ``action`` key, if present, is also reflected at the top level for
        easy querying.
        """
        if "changes" not in policy_data:
            policy_data["changes"] = []

        policy_data["changes"].append(change)

        # Promote action to top-level for convenience
        action = change.get("action")
        if action:
            policy_data["last_action"] = action

        return policy_data
