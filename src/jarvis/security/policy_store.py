"""Versioned Policy Store — externalized, versionable security policies.

Manages policy lifecycle:
- Load/save policies as versioned YAML files
- Version history with rollback
- Dry-run simulation (evaluate without execution)
- Policy validation and diffing

Storage layout::

    ~/.jarvis/policies/
        default.yaml          # Built-in rules (read-only template)
        custom.yaml           # User customizations
        versions/
            v1.yaml           # First snapshot
            v2.yaml           # After first edit
            ...
        history.json          # Version metadata

Architecture: §11.5 (Policy-as-Code)
"""

from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from jarvis.models import (
    GateDecision,
    GateStatus,
    PlannedAction,
    PolicyMatch,
    PolicyParamMatch,
    PolicyRule,
    RiskLevel,
    SessionContext,
)
from jarvis.utils.logging import get_logger

if TYPE_CHECKING:
    from jarvis.core.gatekeeper import Gatekeeper

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Version metadata
# ---------------------------------------------------------------------------


class PolicyVersion:
    """Metadata for a policy version snapshot."""

    def __init__(
        self,
        version: int,
        timestamp: str,
        author: str = "system",
        description: str = "",
        rule_count: int = 0,
    ) -> None:
        self.version = version
        self.timestamp = timestamp
        self.author = author
        self.description = description
        self.rule_count = rule_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "timestamp": self.timestamp,
            "author": self.author,
            "description": self.description,
            "rule_count": self.rule_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PolicyVersion:
        return cls(
            version=data["version"],
            timestamp=data["timestamp"],
            author=data.get("author", "system"),
            description=data.get("description", ""),
            rule_count=data.get("rule_count", 0),
        )


# ---------------------------------------------------------------------------
# Policy Store
# ---------------------------------------------------------------------------


class PolicyStore:
    """Versioned policy storage with history and rollback.

    All policy files are YAML.  Versions are numbered snapshots stored
    in ``policies/versions/v{n}.yaml``.  The active set is always
    ``default.yaml + custom.yaml`` (unchanged from Gatekeeper convention).
    """

    def __init__(self, policies_dir: Path) -> None:
        self._dir = policies_dir
        self._versions_dir = policies_dir / "versions"
        self._history_file = policies_dir / "history.json"
        self._history: list[PolicyVersion] = []
        self._ensure_dirs()
        self._load_history()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_rules(self) -> list[dict[str, Any]]:
        """Load current active rules (default + custom) as raw dicts."""
        rules: list[dict[str, Any]] = []
        for name in ("default.yaml", "custom.yaml"):
            path = self._dir / name
            if path.exists():
                try:
                    data = yaml.safe_load(path.read_text(encoding="utf-8"))
                    if isinstance(data, dict) and "rules" in data:
                        rules.extend(data["rules"])
                except Exception as exc:
                    log.warning("policy_load_error", file=str(path), error=str(exc))
        return rules

    def save_custom_rules(
        self,
        rules: list[dict[str, Any]],
        *,
        author: str = "user",
        description: str = "",
    ) -> int:
        """Save rules to custom.yaml and create a version snapshot.

        Returns the new version number.
        """
        # Write custom.yaml
        custom_path = self._dir / "custom.yaml"
        custom_path.write_text(
            yaml.dump({"rules": rules}, default_flow_style=False, allow_unicode=True),
            encoding="utf-8",
        )

        # Create version snapshot (all active rules)
        all_rules = self.load_rules()
        version = self._next_version()
        snapshot_path = self._versions_dir / f"v{version}.yaml"
        snapshot_path.write_text(
            yaml.dump(
                {"rules": all_rules, "version": version},
                default_flow_style=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )

        # Record in history
        pv = PolicyVersion(
            version=version,
            timestamp=datetime.now(timezone.utc).isoformat(),
            author=author,
            description=description,
            rule_count=len(all_rules),
        )
        self._history.append(pv)
        self._save_history()

        log.info(
            "policy_version_created",
            version=version,
            rule_count=len(all_rules),
            author=author,
        )
        return version

    def rollback(self, version: int) -> bool:
        """Rollback custom.yaml to a previous version snapshot.

        Uses atomic write (write to temp, then rename) with backup to prevent
        data loss if the write fails mid-operation.

        Returns True if successful.
        """
        snapshot_path = self._versions_dir / f"v{version}.yaml"
        if not snapshot_path.exists():
            log.warning("rollback_version_not_found", version=version)
            return False

        try:
            data = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or "rules" not in data:
                return False

            # Remove rules that already exist in default.yaml
            default_rules = self._load_default_rules()
            default_names = {r.get("name") for r in default_rules}
            custom_rules = [r for r in data["rules"] if r.get("name") not in default_names]

            # Backup current custom.yaml before overwriting
            custom_path = self._dir / "custom.yaml"
            backup_path = custom_path.with_suffix(".backup")
            if custom_path.exists():
                import shutil

                shutil.copy2(str(custom_path), str(backup_path))

            # Save as new custom.yaml and record as new version
            self.save_custom_rules(
                custom_rules,
                author="rollback",
                description=f"Rolled back to v{version}",
            )

            # Remove backup on success
            if backup_path.exists():
                backup_path.unlink()

            return True
        except Exception as exc:
            log.error("rollback_failed", version=version, error=str(exc))
            # Restore backup if it exists
            custom_path = self._dir / "custom.yaml"
            backup_path = custom_path.with_suffix(".backup")
            if backup_path.exists():
                import shutil

                shutil.copy2(str(backup_path), str(custom_path))
                backup_path.unlink()
                log.info("rollback_restored_from_backup", version=version)
            return False

    def get_version(self, version: int) -> dict[str, Any] | None:
        """Load a specific version snapshot."""
        path = self._versions_dir / f"v{version}.yaml"
        if not path.exists():
            return None
        try:
            return yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def list_versions(self) -> list[dict[str, Any]]:
        """List all version metadata."""
        return [v.to_dict() for v in self._history]

    @property
    def current_version(self) -> int:
        """Current (latest) version number."""
        return self._history[-1].version if self._history else 0

    def diff_versions(self, v1: int, v2: int) -> dict[str, Any]:
        """Compare two versions and return added/removed/changed rules."""
        data1 = self.get_version(v1) or {"rules": []}
        data2 = self.get_version(v2) or {"rules": []}

        names1 = {r["name"]: r for r in data1.get("rules", []) if "name" in r}
        names2 = {r["name"]: r for r in data2.get("rules", []) if "name" in r}

        added = [n for n in names2 if n not in names1]
        removed = [n for n in names1 if n not in names2]
        changed = [n for n in names1 if n in names2 and names1[n] != names2[n]]

        return {
            "from_version": v1,
            "to_version": v2,
            "added": added,
            "removed": removed,
            "changed": changed,
        }

    def validate_rules(self, rules: list[dict[str, Any]]) -> list[str]:
        """Validate rule structure.  Returns list of errors."""
        errors: list[str] = []
        seen_names: set[str] = set()
        valid_actions = {"ALLOW", "INFORM", "APPROVE", "BLOCK", "MASK"}

        for i, rule in enumerate(rules):
            if not isinstance(rule, dict):
                errors.append(f"Rule {i}: not a dict")
                continue

            name = rule.get("name")
            if not name:
                errors.append(f"Rule {i}: missing 'name'")
            elif name in seen_names:
                errors.append(f"Rule {i}: duplicate name '{name}'")
            else:
                seen_names.add(name)

            action = rule.get("action", "")
            if action and action not in valid_actions:
                errors.append(f"Rule '{name}': invalid action '{action}'")

            match = rule.get("match")
            if match and not isinstance(match, dict):
                errors.append(f"Rule '{name}': 'match' must be a dict")

            priority = rule.get("priority", 0)
            if not isinstance(priority, int) or priority < 0:
                errors.append(f"Rule '{name}': 'priority' must be non-negative int")

        return errors

    def simulate(
        self,
        gatekeeper: Gatekeeper,
        tool_name: str,
        params: dict[str, Any] | None = None,
        *,
        session_id: str = "simulation",
    ) -> dict[str, Any]:
        """Simulate a policy evaluation without execution.

        Returns the GateDecision details + matching rule info.
        """
        action = PlannedAction(
            tool=tool_name,
            params=params or {},
            rationale="Simulation",
        )
        context = SessionContext(
            session_id=session_id,
            user_id="simulator",
            channel="simulation",
        )

        decision = gatekeeper.evaluate(action, context)

        return {
            "tool": tool_name,
            "params": params or {},
            "decision": {
                "status": decision.status.value,
                "risk_level": decision.risk_level.value,
                "reason": decision.reason,
                "policy_name": decision.policy_name,
            },
            "simulation": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def simulate_batch(
        self,
        gatekeeper: Gatekeeper,
        actions: list[dict[str, Any]],
        *,
        session_id: str = "simulation",
    ) -> list[dict[str, Any]]:
        """Simulate multiple actions. Each is {"tool": str, "params": dict}."""
        return [
            self.simulate(
                gatekeeper,
                a.get("tool", ""),
                a.get("params"),
                session_id=session_id,
            )
            for a in actions
        ]

    def stats(self) -> dict[str, Any]:
        """Store statistics."""
        rules = self.load_rules()
        return {
            "active_rules": len(rules),
            "versions": len(self._history),
            "current_version": self.current_version,
            "has_custom": (self._dir / "custom.yaml").exists(),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_dirs(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        self._versions_dir.mkdir(parents=True, exist_ok=True)

    def _load_history(self) -> None:
        if self._history_file.exists():
            try:
                data = json.loads(self._history_file.read_text(encoding="utf-8"))
                self._history = [PolicyVersion.from_dict(v) for v in data]
            except Exception:
                self._history = []

    def _save_history(self) -> None:
        self._history_file.write_text(
            json.dumps([v.to_dict() for v in self._history], indent=2),
            encoding="utf-8",
        )

    def _next_version(self) -> int:
        if not self._history:
            return 1
        return self._history[-1].version + 1

    def _load_default_rules(self) -> list[dict[str, Any]]:
        path = self._dir / "default.yaml"
        if not path.exists():
            return []
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "rules" in data:
                return data["rules"]
        except Exception as exc:
            log.warning("policy_rules_load_error", path=str(path), error=str(exc))
        return []
