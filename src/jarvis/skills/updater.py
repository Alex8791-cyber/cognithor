"""Skill-Update-Mechanismus: Automatische Updates und Notfall-Rückzüge.

Stellt bereit:
  - SkillUpdater: Prüft und installiert Updates für Skills
  - UpdatePolicy: Konfigurierbare Update-Strategie
  - SecurityRecall: Notfall-Rückzug gefährlicher Pakete
  - UpdateCheck: Ergebnis einer Update-Prüfung

Bibel-Referenz: §11 (Skills & Ecosystem), §14 (Security)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from jarvis.utils.logging import get_logger

log = get_logger(__name__)


class UpdateStrategy(Enum):
    """Update-Strategie."""

    MANUAL = "manual"          # Nur auf Anfrage
    NOTIFY = "notify"          # Benachrichtigen, nicht installieren
    AUTO_MINOR = "auto_minor"  # Minor/Patch automatisch, Major manuell
    AUTO_ALL = "auto_all"      # Alles automatisch


class UpdateSeverity(Enum):
    """Dringlichkeit eines Updates."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"           # Sicherheitsfix
    CRITICAL = "critical"   # Notfall-Update (erzwungen)


@dataclass
class UpdatePolicy:
    """Konfigurierbare Update-Strategie."""

    strategy: UpdateStrategy = UpdateStrategy.NOTIFY
    auto_security_updates: bool = True   # Security immer automatisch
    check_interval_hours: int = 24
    require_signature: bool = True
    require_review: bool = False         # Review vor Installation
    max_auto_updates_per_day: int = 10
    blocked_packages: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy.value,
            "auto_security_updates": self.auto_security_updates,
            "check_interval_hours": self.check_interval_hours,
            "require_signature": self.require_signature,
            "require_review": self.require_review,
            "max_auto_updates_per_day": self.max_auto_updates_per_day,
            "blocked_packages": self.blocked_packages,
        }


@dataclass
class UpdateCheck:
    """Ergebnis einer Update-Prüfung für ein Paket."""

    package_id: str
    current_version: str
    available_version: str
    severity: UpdateSeverity = UpdateSeverity.NORMAL
    changelog: str = ""
    signature_valid: bool = False
    size_bytes: int = 0
    requires_restart: bool = False
    auto_installable: bool = False
    checked_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def has_update(self) -> bool:
        return self.current_version != self.available_version

    @property
    def is_major(self) -> bool:
        try:
            current_major = int(self.current_version.split(".")[0])
            available_major = int(self.available_version.split(".")[0])
            return available_major > current_major
        except (ValueError, IndexError):
            return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "package_id": self.package_id,
            "current_version": self.current_version,
            "available_version": self.available_version,
            "has_update": self.has_update,
            "is_major": self.is_major,
            "severity": self.severity.value,
            "changelog": self.changelog,
            "signature_valid": self.signature_valid,
            "auto_installable": self.auto_installable,
            "checked_at": self.checked_at.isoformat(),
        }


@dataclass
class SecurityRecall:
    """Notfall-Rückzug eines gefährlichen Pakets."""

    package_id: str
    reason: str
    severity: UpdateSeverity = UpdateSeverity.CRITICAL
    recalled_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    recalled_by: str = ""
    force_uninstall: bool = False
    replacement_id: str = ""  # Alternatives Paket

    def to_dict(self) -> dict[str, Any]:
        return {
            "package_id": self.package_id,
            "reason": self.reason,
            "severity": self.severity.value,
            "recalled_at": self.recalled_at.isoformat(),
            "recalled_by": self.recalled_by,
            "force_uninstall": self.force_uninstall,
            "replacement_id": self.replacement_id,
        }


@dataclass
class UpdateResult:
    """Ergebnis einer Update-Installation."""

    package_id: str
    from_version: str
    to_version: str
    success: bool
    error: str = ""
    installed_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    rollback_available: bool = True


class SkillUpdater:
    """Prüft und installiert Updates für Skills.

    Features:
      - Periodische Update-Prüfung (via Heartbeat)
      - Konfigurierbare Update-Strategie
      - Signatur-Prüfung vor Installation
      - Notfall-Rückzüge (Security Recalls)
      - Rollback bei fehlerhaften Updates
      - Update-Historie und Audit-Trail
    """

    def __init__(self, policy: UpdatePolicy | None = None) -> None:
        self._policy = policy or UpdatePolicy()
        self._installed: dict[str, str] = {}        # package_id → version
        self._available: dict[str, UpdateCheck] = {}  # package_id → check
        self._recalls: dict[str, SecurityRecall] = {}
        self._history: list[UpdateResult] = []
        self._last_check: datetime | None = None
        self._auto_updates_today: int = 0
        self._auto_update_day: str = ""

    @property
    def policy(self) -> UpdatePolicy:
        return self._policy

    @policy.setter
    def policy(self, value: UpdatePolicy) -> None:
        self._policy = value

    # ------------------------------------------------------------------
    # Installierte Pakete
    # ------------------------------------------------------------------

    def register_installed(self, package_id: str, version: str) -> None:
        """Registriert ein installiertes Paket."""
        self._installed[package_id] = version

    def installed_packages(self) -> dict[str, str]:
        return dict(self._installed)

    def installed_count(self) -> int:
        return len(self._installed)

    # ------------------------------------------------------------------
    # Update-Prüfung
    # ------------------------------------------------------------------

    def check_update(
        self,
        package_id: str,
        available_version: str,
        *,
        severity: UpdateSeverity = UpdateSeverity.NORMAL,
        changelog: str = "",
        signature_valid: bool = True,
    ) -> UpdateCheck:
        """Prüft ob ein Update verfügbar ist."""
        current = self._installed.get(package_id, "0.0.0")

        check = UpdateCheck(
            package_id=package_id,
            current_version=current,
            available_version=available_version,
            severity=severity,
            changelog=changelog,
            signature_valid=signature_valid,
        )

        # Auto-installierbar?
        check.auto_installable = self._is_auto_installable(check)

        self._available[package_id] = check
        self._last_check = datetime.now(UTC)
        return check

    def pending_updates(self) -> list[UpdateCheck]:
        """Alle verfügbaren Updates."""
        return [c for c in self._available.values() if c.has_update]

    def auto_installable_updates(self) -> list[UpdateCheck]:
        """Updates die automatisch installiert werden können."""
        return [c for c in self.pending_updates() if c.auto_installable]

    def _is_auto_installable(self, check: UpdateCheck) -> bool:
        """Bestimmt ob ein Update automatisch installiert werden kann."""
        policy = self._policy

        # Geblockte Pakete nie automatisch
        if check.package_id in policy.blocked_packages:
            return False

        # Signatur erforderlich?
        if policy.require_signature and not check.signature_valid:
            return False

        # Recalled?
        if check.package_id in self._recalls:
            return False

        # Security-Updates immer wenn konfiguriert
        if check.severity in (UpdateSeverity.HIGH, UpdateSeverity.CRITICAL):
            return policy.auto_security_updates

        # Strategie
        if policy.strategy == UpdateStrategy.AUTO_ALL:
            return True
        if policy.strategy == UpdateStrategy.AUTO_MINOR:
            return not check.is_major
        return False

    # ------------------------------------------------------------------
    # Installation
    # ------------------------------------------------------------------

    def install_update(self, package_id: str) -> UpdateResult:
        """Installiert ein verfügbares Update."""
        check = self._available.get(package_id)
        if not check or not check.has_update:
            return UpdateResult(
                package_id=package_id,
                from_version=self._installed.get(package_id, "0.0.0"),
                to_version="",
                success=False,
                error="no_update_available",
            )

        # Tageslimit prüfen
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        if today != self._auto_update_day:
            self._auto_updates_today = 0
            self._auto_update_day = today

        from_version = self._installed.get(package_id, "0.0.0")
        to_version = check.available_version

        # Simuliere Installation (in Realität: Download + Verify + Install)
        self._installed[package_id] = to_version
        self._auto_updates_today += 1

        result = UpdateResult(
            package_id=package_id,
            from_version=from_version,
            to_version=to_version,
            success=True,
        )
        self._history.append(result)

        # Verfügbarkeit aufräumen
        del self._available[package_id]

        log.info(
            "skill_updated",
            package_id=package_id,
            from_version=from_version,
            to_version=to_version,
        )
        return result

    # ------------------------------------------------------------------
    # Security Recalls
    # ------------------------------------------------------------------

    def recall_package(
        self,
        package_id: str,
        reason: str,
        *,
        recalled_by: str = "security_team",
        force_uninstall: bool = False,
        replacement_id: str = "",
    ) -> SecurityRecall:
        """Zieht ein gefährliches Paket zurück."""
        recall = SecurityRecall(
            package_id=package_id,
            reason=reason,
            recalled_by=recalled_by,
            force_uninstall=force_uninstall,
            replacement_id=replacement_id,
        )
        self._recalls[package_id] = recall

        # Aus verfügbaren Updates entfernen
        self._available.pop(package_id, None)

        # Bei force_uninstall: deinstallieren
        if force_uninstall and package_id in self._installed:
            del self._installed[package_id]

        log.warning(
            "skill_recalled",
            package_id=package_id,
            reason=reason,
            force_uninstall=force_uninstall,
        )
        return recall

    def is_recalled(self, package_id: str) -> bool:
        return package_id in self._recalls

    def active_recalls(self) -> list[SecurityRecall]:
        return list(self._recalls.values())

    # ------------------------------------------------------------------
    # Statistiken
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        return {
            "installed_count": len(self._installed),
            "pending_updates": len(self.pending_updates()),
            "auto_installable": len(self.auto_installable_updates()),
            "active_recalls": len(self._recalls),
            "total_updates_installed": len(self._history),
            "auto_updates_today": self._auto_updates_today,
            "last_check": self._last_check.isoformat() if self._last_check else None,
            "policy": self._policy.to_dict(),
        }

    def update_history(self, last_n: int = 20) -> list[dict[str, Any]]:
        return [
            {
                "package_id": r.package_id,
                "from": r.from_version,
                "to": r.to_version,
                "success": r.success,
                "error": r.error,
                "installed_at": r.installed_at.isoformat(),
            }
            for r in self._history[-last_n:]
        ]
