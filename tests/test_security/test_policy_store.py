"""Tests for the versioned PolicyStore.

Covers YAML loading, version snapshots, rollback, validation,
simulation, diffing, and edge cases.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import yaml

from jarvis.security.policy_store import PolicyStore, PolicyVersion


# ============================================================================
# Helpers
# ============================================================================


def _write_policy(path: Path, rules: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.dump({"rules": rules}, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
    )


def _simple_rule(name: str, action: str = "ALLOW", tool: str = "*", priority: int = 0) -> dict[str, Any]:
    return {
        "name": name,
        "action": action,
        "match": {"tool": tool},
        "reason": f"Test rule {name}",
        "priority": priority,
    }


# ============================================================================
# PolicyVersion
# ============================================================================


class TestPolicyVersion:
    def test_round_trip(self) -> None:
        pv = PolicyVersion(version=1, timestamp="2026-03-04T10:00:00Z", author="user", description="initial", rule_count=5)
        d = pv.to_dict()
        pv2 = PolicyVersion.from_dict(d)
        assert pv2.version == 1
        assert pv2.author == "user"
        assert pv2.rule_count == 5

    def test_defaults(self) -> None:
        pv = PolicyVersion(version=1, timestamp="now")
        assert pv.author == "system"
        assert pv.description == ""


# ============================================================================
# Load Rules
# ============================================================================


class TestLoadRules:
    def test_load_empty_dir(self, tmp_path: Path) -> None:
        store = PolicyStore(tmp_path / "policies")
        assert store.load_rules() == []

    def test_load_default_only(self, tmp_path: Path) -> None:
        pol_dir = tmp_path / "policies"
        _write_policy(pol_dir / "default.yaml", [_simple_rule("r1")])
        store = PolicyStore(pol_dir)
        rules = store.load_rules()
        assert len(rules) == 1
        assert rules[0]["name"] == "r1"

    def test_load_default_and_custom(self, tmp_path: Path) -> None:
        pol_dir = tmp_path / "policies"
        _write_policy(pol_dir / "default.yaml", [_simple_rule("r1")])
        _write_policy(pol_dir / "custom.yaml", [_simple_rule("r2")])
        store = PolicyStore(pol_dir)
        rules = store.load_rules()
        assert len(rules) == 2
        names = {r["name"] for r in rules}
        assert names == {"r1", "r2"}

    def test_load_ignores_invalid_yaml(self, tmp_path: Path) -> None:
        pol_dir = tmp_path / "policies"
        pol_dir.mkdir(parents=True)
        (pol_dir / "default.yaml").write_text("{{invalid", encoding="utf-8")
        store = PolicyStore(pol_dir)
        assert store.load_rules() == []

    def test_load_ignores_no_rules_key(self, tmp_path: Path) -> None:
        pol_dir = tmp_path / "policies"
        pol_dir.mkdir(parents=True)
        (pol_dir / "default.yaml").write_text(
            yaml.dump({"not_rules": []}), encoding="utf-8",
        )
        store = PolicyStore(pol_dir)
        assert store.load_rules() == []


# ============================================================================
# Save & Version
# ============================================================================


class TestSaveAndVersion:
    def test_save_creates_custom_yaml(self, tmp_path: Path) -> None:
        store = PolicyStore(tmp_path / "policies")
        store.save_custom_rules([_simple_rule("custom1")])
        custom = tmp_path / "policies" / "custom.yaml"
        assert custom.exists()
        data = yaml.safe_load(custom.read_text(encoding="utf-8"))
        assert data["rules"][0]["name"] == "custom1"

    def test_save_creates_version_snapshot(self, tmp_path: Path) -> None:
        store = PolicyStore(tmp_path / "policies")
        v = store.save_custom_rules([_simple_rule("r1")])
        assert v == 1
        snapshot = tmp_path / "policies" / "versions" / "v1.yaml"
        assert snapshot.exists()

    def test_save_increments_version(self, tmp_path: Path) -> None:
        store = PolicyStore(tmp_path / "policies")
        v1 = store.save_custom_rules([_simple_rule("r1")])
        v2 = store.save_custom_rules([_simple_rule("r1"), _simple_rule("r2")])
        assert v1 == 1
        assert v2 == 2

    def test_save_records_history(self, tmp_path: Path) -> None:
        store = PolicyStore(tmp_path / "policies")
        store.save_custom_rules([_simple_rule("r1")], author="alex", description="first")
        versions = store.list_versions()
        assert len(versions) == 1
        assert versions[0]["author"] == "alex"
        assert versions[0]["description"] == "first"

    def test_current_version_after_save(self, tmp_path: Path) -> None:
        store = PolicyStore(tmp_path / "policies")
        assert store.current_version == 0
        store.save_custom_rules([_simple_rule("r1")])
        assert store.current_version == 1
        store.save_custom_rules([_simple_rule("r2")])
        assert store.current_version == 2

    def test_history_persists_across_instances(self, tmp_path: Path) -> None:
        pol_dir = tmp_path / "policies"
        store1 = PolicyStore(pol_dir)
        store1.save_custom_rules([_simple_rule("r1")])
        store2 = PolicyStore(pol_dir)
        assert store2.current_version == 1
        assert len(store2.list_versions()) == 1


# ============================================================================
# Rollback
# ============================================================================


class TestRollback:
    def test_rollback_to_previous_version(self, tmp_path: Path) -> None:
        pol_dir = tmp_path / "policies"
        _write_policy(pol_dir / "default.yaml", [_simple_rule("default_rule")])
        store = PolicyStore(pol_dir)
        store.save_custom_rules([_simple_rule("v1_rule")])
        store.save_custom_rules([_simple_rule("v2_rule")])
        assert store.rollback(1) is True
        # After rollback, custom.yaml should have v1 rules (minus defaults)
        rules = store.load_rules()
        names = {r["name"] for r in rules}
        assert "v1_rule" in names

    def test_rollback_nonexistent_version(self, tmp_path: Path) -> None:
        store = PolicyStore(tmp_path / "policies")
        assert store.rollback(999) is False

    def test_rollback_creates_new_version(self, tmp_path: Path) -> None:
        pol_dir = tmp_path / "policies"
        store = PolicyStore(pol_dir)
        store.save_custom_rules([_simple_rule("r1")])
        store.save_custom_rules([_simple_rule("r2")])
        store.rollback(1)
        # Should now be at version 3 (rollback creates a new version)
        assert store.current_version == 3


# ============================================================================
# Version Retrieval
# ============================================================================


class TestGetVersion:
    def test_get_existing_version(self, tmp_path: Path) -> None:
        store = PolicyStore(tmp_path / "policies")
        store.save_custom_rules([_simple_rule("r1")])
        data = store.get_version(1)
        assert data is not None
        assert data["version"] == 1
        assert len(data["rules"]) >= 1

    def test_get_nonexistent_version(self, tmp_path: Path) -> None:
        store = PolicyStore(tmp_path / "policies")
        assert store.get_version(999) is None


# ============================================================================
# Diff
# ============================================================================


class TestDiffVersions:
    def test_diff_added_rules(self, tmp_path: Path) -> None:
        store = PolicyStore(tmp_path / "policies")
        store.save_custom_rules([_simple_rule("r1")])
        store.save_custom_rules([_simple_rule("r1"), _simple_rule("r2")])
        diff = store.diff_versions(1, 2)
        assert "r2" in diff["added"]
        assert diff["removed"] == []

    def test_diff_removed_rules(self, tmp_path: Path) -> None:
        store = PolicyStore(tmp_path / "policies")
        store.save_custom_rules([_simple_rule("r1"), _simple_rule("r2")])
        store.save_custom_rules([_simple_rule("r1")])
        diff = store.diff_versions(1, 2)
        assert "r2" in diff["removed"]

    def test_diff_changed_rules(self, tmp_path: Path) -> None:
        store = PolicyStore(tmp_path / "policies")
        store.save_custom_rules([_simple_rule("r1", action="ALLOW")])
        store.save_custom_rules([_simple_rule("r1", action="BLOCK")])
        diff = store.diff_versions(1, 2)
        assert "r1" in diff["changed"]

    def test_diff_nonexistent_versions(self, tmp_path: Path) -> None:
        store = PolicyStore(tmp_path / "policies")
        diff = store.diff_versions(1, 2)
        assert diff["added"] == []
        assert diff["removed"] == []


# ============================================================================
# Validation
# ============================================================================


class TestValidateRules:
    def test_valid_rules(self, tmp_path: Path) -> None:
        store = PolicyStore(tmp_path / "policies")
        errors = store.validate_rules([
            _simple_rule("r1", action="ALLOW"),
            _simple_rule("r2", action="BLOCK"),
        ])
        assert errors == []

    def test_missing_name(self, tmp_path: Path) -> None:
        store = PolicyStore(tmp_path / "policies")
        errors = store.validate_rules([{"action": "ALLOW"}])
        assert any("missing 'name'" in e for e in errors)

    def test_duplicate_names(self, tmp_path: Path) -> None:
        store = PolicyStore(tmp_path / "policies")
        errors = store.validate_rules([
            _simple_rule("same"),
            _simple_rule("same"),
        ])
        assert any("duplicate" in e for e in errors)

    def test_invalid_action(self, tmp_path: Path) -> None:
        store = PolicyStore(tmp_path / "policies")
        errors = store.validate_rules([_simple_rule("r1", action="DESTROY")])
        assert any("invalid action" in e for e in errors)

    def test_invalid_priority(self, tmp_path: Path) -> None:
        store = PolicyStore(tmp_path / "policies")
        errors = store.validate_rules([{
            "name": "r1", "action": "ALLOW", "priority": -1,
        }])
        assert any("priority" in e for e in errors)

    def test_match_must_be_dict(self, tmp_path: Path) -> None:
        store = PolicyStore(tmp_path / "policies")
        errors = store.validate_rules([{
            "name": "r1", "action": "ALLOW", "match": "bad",
        }])
        assert any("must be a dict" in e for e in errors)

    def test_not_a_dict(self, tmp_path: Path) -> None:
        store = PolicyStore(tmp_path / "policies")
        errors = store.validate_rules(["not a dict"])
        assert any("not a dict" in e for e in errors)


# ============================================================================
# Simulation
# ============================================================================


class TestSimulation:
    def test_simulate_returns_decision(self, tmp_path: Path) -> None:
        store = PolicyStore(tmp_path / "policies")
        gatekeeper = MagicMock()
        decision = MagicMock()
        decision.status.value = "ALLOW"
        decision.risk_level.value = "green"
        decision.reason = "Safe tool"
        decision.policy_name = "test_policy"
        gatekeeper.evaluate.return_value = decision

        result = store.simulate(gatekeeper, "read_file", {"path": "/test"})
        assert result["simulation"] is True
        assert result["tool"] == "read_file"
        assert result["decision"]["status"] == "ALLOW"
        assert result["decision"]["policy_name"] == "test_policy"

    def test_simulate_calls_evaluate(self, tmp_path: Path) -> None:
        store = PolicyStore(tmp_path / "policies")
        gatekeeper = MagicMock()
        decision = MagicMock()
        decision.status.value = "BLOCK"
        decision.risk_level.value = "red"
        decision.reason = "Blocked"
        decision.policy_name = ""
        gatekeeper.evaluate.return_value = decision

        store.simulate(gatekeeper, "exec_command", {"command": "rm -rf /"})
        gatekeeper.evaluate.assert_called_once()

    def test_simulate_batch(self, tmp_path: Path) -> None:
        store = PolicyStore(tmp_path / "policies")
        gatekeeper = MagicMock()
        decision = MagicMock()
        decision.status.value = "ALLOW"
        decision.risk_level.value = "green"
        decision.reason = ""
        decision.policy_name = ""
        gatekeeper.evaluate.return_value = decision

        results = store.simulate_batch(gatekeeper, [
            {"tool": "read_file", "params": {"path": "/a"}},
            {"tool": "write_file", "params": {"path": "/b"}},
        ])
        assert len(results) == 2
        assert all(r["simulation"] for r in results)


# ============================================================================
# Stats
# ============================================================================


class TestStats:
    def test_stats_empty(self, tmp_path: Path) -> None:
        store = PolicyStore(tmp_path / "policies")
        s = store.stats()
        assert s["active_rules"] == 0
        assert s["versions"] == 0
        assert s["has_custom"] is False

    def test_stats_with_rules(self, tmp_path: Path) -> None:
        pol_dir = tmp_path / "policies"
        _write_policy(pol_dir / "default.yaml", [_simple_rule("r1"), _simple_rule("r2")])
        store = PolicyStore(pol_dir)
        store.save_custom_rules([_simple_rule("r3")])
        s = store.stats()
        assert s["active_rules"] == 3
        assert s["versions"] == 1
        assert s["current_version"] == 1
        assert s["has_custom"] is True
