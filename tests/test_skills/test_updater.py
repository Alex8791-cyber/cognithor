"""Tests für Skill-Update-Mechanismus."""

from __future__ import annotations

import pytest

from jarvis.skills.updater import (
    SecurityRecall,
    SkillUpdater,
    UpdateCheck,
    UpdatePolicy,
    UpdateSeverity,
    UpdateStrategy,
)


class TestUpdateCheck:
    def test_has_update(self) -> None:
        check = UpdateCheck(package_id="p1", current_version="1.0.0", available_version="1.1.0")
        assert check.has_update

    def test_no_update(self) -> None:
        check = UpdateCheck(package_id="p1", current_version="1.0.0", available_version="1.0.0")
        assert not check.has_update

    def test_is_major(self) -> None:
        check = UpdateCheck(package_id="p1", current_version="1.0.0", available_version="2.0.0")
        assert check.is_major

    def test_is_not_major(self) -> None:
        check = UpdateCheck(package_id="p1", current_version="1.0.0", available_version="1.2.0")
        assert not check.is_major

    def test_to_dict(self) -> None:
        check = UpdateCheck(package_id="p1", current_version="1.0.0", available_version="1.1.0")
        d = check.to_dict()
        assert d["has_update"] is True
        assert d["package_id"] == "p1"


class TestUpdatePolicy:
    def test_defaults(self) -> None:
        policy = UpdatePolicy()
        assert policy.strategy == UpdateStrategy.NOTIFY
        assert policy.auto_security_updates is True

    def test_to_dict(self) -> None:
        policy = UpdatePolicy(strategy=UpdateStrategy.AUTO_MINOR)
        d = policy.to_dict()
        assert d["strategy"] == "auto_minor"


class TestSkillUpdater:
    def test_register_installed(self) -> None:
        updater = SkillUpdater()
        updater.register_installed("skill_a", "1.0.0")
        assert updater.installed_count() == 1
        assert updater.installed_packages()["skill_a"] == "1.0.0"

    def test_check_update(self) -> None:
        updater = SkillUpdater()
        updater.register_installed("skill_a", "1.0.0")
        check = updater.check_update("skill_a", "1.1.0")
        assert check.has_update
        assert check.current_version == "1.0.0"

    def test_check_no_update(self) -> None:
        updater = SkillUpdater()
        updater.register_installed("skill_a", "1.0.0")
        check = updater.check_update("skill_a", "1.0.0")
        assert not check.has_update

    def test_pending_updates(self) -> None:
        updater = SkillUpdater()
        updater.register_installed("a", "1.0.0")
        updater.register_installed("b", "2.0.0")
        updater.check_update("a", "1.1.0")
        updater.check_update("b", "2.0.0")
        assert len(updater.pending_updates()) == 1

    def test_auto_installable_notify_policy(self) -> None:
        updater = SkillUpdater(UpdatePolicy(strategy=UpdateStrategy.NOTIFY))
        updater.register_installed("a", "1.0.0")
        updater.check_update("a", "1.1.0")
        assert len(updater.auto_installable_updates()) == 0

    def test_auto_installable_auto_minor(self) -> None:
        updater = SkillUpdater(UpdatePolicy(strategy=UpdateStrategy.AUTO_MINOR))
        updater.register_installed("a", "1.0.0")
        updater.check_update("a", "1.1.0", signature_valid=True)
        assert len(updater.auto_installable_updates()) == 1

    def test_auto_minor_blocks_major(self) -> None:
        updater = SkillUpdater(UpdatePolicy(strategy=UpdateStrategy.AUTO_MINOR))
        updater.register_installed("a", "1.0.0")
        updater.check_update("a", "2.0.0", signature_valid=True)
        assert len(updater.auto_installable_updates()) == 0

    def test_auto_all(self) -> None:
        updater = SkillUpdater(UpdatePolicy(strategy=UpdateStrategy.AUTO_ALL))
        updater.register_installed("a", "1.0.0")
        updater.check_update("a", "2.0.0", signature_valid=True)
        assert len(updater.auto_installable_updates()) == 1

    def test_security_update_auto(self) -> None:
        updater = SkillUpdater(UpdatePolicy(
            strategy=UpdateStrategy.NOTIFY, auto_security_updates=True,
        ))
        updater.register_installed("a", "1.0.0")
        updater.check_update("a", "1.0.1", severity=UpdateSeverity.CRITICAL, signature_valid=True)
        assert len(updater.auto_installable_updates()) == 1

    def test_blocked_package_not_auto(self) -> None:
        updater = SkillUpdater(UpdatePolicy(
            strategy=UpdateStrategy.AUTO_ALL, blocked_packages=["a"],
        ))
        updater.register_installed("a", "1.0.0")
        updater.check_update("a", "1.1.0", signature_valid=True)
        assert len(updater.auto_installable_updates()) == 0

    def test_unsigned_not_auto(self) -> None:
        updater = SkillUpdater(UpdatePolicy(
            strategy=UpdateStrategy.AUTO_ALL, require_signature=True,
        ))
        updater.register_installed("a", "1.0.0")
        updater.check_update("a", "1.1.0", signature_valid=False)
        assert len(updater.auto_installable_updates()) == 0

    def test_install_update(self) -> None:
        updater = SkillUpdater()
        updater.register_installed("a", "1.0.0")
        updater.check_update("a", "1.1.0")
        result = updater.install_update("a")
        assert result.success
        assert result.from_version == "1.0.0"
        assert result.to_version == "1.1.0"
        assert updater.installed_packages()["a"] == "1.1.0"

    def test_install_no_update(self) -> None:
        updater = SkillUpdater()
        result = updater.install_update("nonexistent")
        assert not result.success
        assert result.error == "no_update_available"

    def test_recall_package(self) -> None:
        updater = SkillUpdater()
        updater.register_installed("bad_skill", "1.0.0")
        recall = updater.recall_package("bad_skill", "Malware detected")
        assert updater.is_recalled("bad_skill")
        assert len(updater.active_recalls()) == 1

    def test_recall_force_uninstall(self) -> None:
        updater = SkillUpdater()
        updater.register_installed("bad_skill", "1.0.0")
        updater.recall_package("bad_skill", "Critical", force_uninstall=True)
        assert "bad_skill" not in updater.installed_packages()

    def test_recalled_package_not_auto(self) -> None:
        updater = SkillUpdater(UpdatePolicy(strategy=UpdateStrategy.AUTO_ALL))
        updater.register_installed("a", "1.0.0")
        updater.recall_package("a", "Danger")
        updater.check_update("a", "1.1.0", signature_valid=True)
        assert len(updater.auto_installable_updates()) == 0

    def test_stats(self) -> None:
        updater = SkillUpdater()
        updater.register_installed("a", "1.0.0")
        updater.check_update("a", "1.1.0")
        s = updater.stats()
        assert s["installed_count"] == 1
        assert s["pending_updates"] == 1

    def test_update_history(self) -> None:
        updater = SkillUpdater()
        updater.register_installed("a", "1.0.0")
        updater.check_update("a", "1.1.0")
        updater.install_update("a")
        history = updater.update_history()
        assert len(history) == 1
        assert history[0]["to"] == "1.1.0"

    def test_policy_setter(self) -> None:
        updater = SkillUpdater()
        new_policy = UpdatePolicy(strategy=UpdateStrategy.AUTO_ALL)
        updater.policy = new_policy
        assert updater.policy.strategy == UpdateStrategy.AUTO_ALL
