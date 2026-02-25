"""Tests fuer PolicyPatcher (Feature 5)."""

from __future__ import annotations

import pytest
import yaml

from jarvis.governance.policy_patcher import PolicyPatcher
from jarvis.models import PolicyChange


@pytest.fixture()
def policies_dir(tmp_path):
    """Policies-Verzeichnis mit einer Default-Policy."""
    pol_dir = tmp_path / "policies"
    pol_dir.mkdir()
    default_yaml = pol_dir / "default.yaml"
    default_yaml.write_text(yaml.dump({
        "rules": [
            {"name": "rule1", "match": {"tool": "read_file"}, "action": "ALLOW", "reason": "OK"},
        ]
    }), encoding="utf-8")
    return str(pol_dir)


@pytest.fixture()
def patcher(policies_dir):
    return PolicyPatcher(policies_dir)


class TestPolicyPatcher:
    def test_apply_change_modifies_yaml(self, patcher, policies_dir):
        change = PolicyChange(
            proposal_id=1,
            category="default",
            title="Add rule",
            change={
                "action": "add_rule",
                "tool": "exec_command",
                "reason": "Blocked",
            },
        )
        result = patcher.apply_change(change)
        assert result is True

        import pathlib
        content = yaml.safe_load(pathlib.Path(policies_dir, "default.yaml").read_text(encoding="utf-8"))
        # Die Aenderung wird in einer "changes" Liste gespeichert
        assert "changes" in content
        assert len(content["changes"]) >= 1
        assert content["changes"][0]["action"] == "add_rule"

    def test_backup_created_before_change(self, patcher, policies_dir):
        change = PolicyChange(
            proposal_id=1,
            category="default",
            title="Modify rule",
            change={"action": "modify_rule", "timeout": 30},
        )
        patcher.apply_change(change)

        backups = patcher.list_backups()
        assert len(backups) >= 1

    def test_rollback_restores_original(self, patcher, policies_dir):
        import pathlib
        original = pathlib.Path(policies_dir, "default.yaml").read_text(encoding="utf-8")

        change = PolicyChange(
            proposal_id=1,
            category="default",
            title="Change all",
            change={"action": "block_all", "reason": "All blocked"},
        )
        patcher.apply_change(change)

        # Verify it changed
        modified = pathlib.Path(policies_dir, "default.yaml").read_text(encoding="utf-8")
        assert modified != original

        # Rollback
        result = patcher.rollback_last()
        assert result is True

        restored = pathlib.Path(policies_dir, "default.yaml").read_text(encoding="utf-8")
        assert restored == original

    def test_new_category_creates_file(self, patcher, policies_dir):
        """Neue Kategorie erzeugt neue YAML-Datei."""
        change = PolicyChange(
            proposal_id=1,
            category="newcat",
            title="New category",
            change={"action": "add_rule", "tool": "new_tool"},
        )
        result = patcher.apply_change(change)
        assert result is True

        import pathlib
        new_file = pathlib.Path(policies_dir, "newcat.yaml")
        assert new_file.exists()
        content = yaml.safe_load(new_file.read_text(encoding="utf-8"))
        assert "changes" in content

    def test_list_backups(self, patcher, policies_dir):
        # Initially no backups
        assert len(patcher.list_backups()) == 0

        change = PolicyChange(
            proposal_id=1,
            category="default",
            title="Test backup",
            change={"action": "test"},
        )
        patcher.apply_change(change)
        assert len(patcher.list_backups()) >= 1
