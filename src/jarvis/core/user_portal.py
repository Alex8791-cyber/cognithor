"""Jarvis · Endnutzer-Portal.

Nicht-technisches Dashboard für Endanwender:

  - UserConsent:           Einwilligungsmanagement (DSGVO Art. 7)
  - ConsentManager:        Verwaltet Einwilligungen pro Nutzer
  - DecisionView:          Verständliche Entscheidungsdarstellung
  - UserNotification:      Benachrichtigungen für Endnutzer
  - UserActivityLog:       Nachvollziehbare Aktivitätshistorie
  - UserPortal:            Hauptklasse

Architektur-Bibel: §16.4 (Endnutzer-Transparenz), §17.1 (DSGVO-Compliance)
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


# ============================================================================
# User Consent (DSGVO Art. 7)
# ============================================================================


class ConsentPurpose(Enum):
    AI_PROCESSING = "ai_processing"  # KI-Verarbeitung
    DATA_ANALYSIS = "data_analysis"  # Datenanalyse
    PROFILING = "profiling"  # Profilbildung
    THIRD_PARTY = "third_party_sharing"  # Weitergabe an Dritte
    MARKETING = "marketing"  # Marketing
    INSURANCE_ADVICE = "insurance_advice"  # Versicherungsberatung
    HEALTH_DATA = "health_data"  # Gesundheitsdaten (BU)
    AUTOMATED_DECISIONS = "automated_decisions"  # Automatisierte Entscheidungen


class ConsentStatus(Enum):
    GRANTED = "granted"
    DENIED = "denied"
    WITHDRAWN = "withdrawn"
    PENDING = "pending"
    EXPIRED = "expired"


@dataclass
class UserConsent:
    """Eine einzelne Einwilligung."""

    consent_id: str
    user_id: str
    purpose: ConsentPurpose
    status: ConsentStatus = ConsentStatus.PENDING
    granted_at: str = ""
    expires_at: str = ""
    withdrawn_at: str = ""
    legal_basis: str = "Art. 6(1)(a) DSGVO"
    description: str = ""

    @property
    def is_valid(self) -> bool:
        if self.status != ConsentStatus.GRANTED:
            return False
        if self.expires_at and self.expires_at < time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()):
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "consent_id": self.consent_id,
            "purpose": self.purpose.value,
            "status": self.status.value,
            "valid": self.is_valid,
            "legal_basis": self.legal_basis,
        }


class ConsentManager:
    """Verwaltet Einwilligungen pro Nutzer (DSGVO-konform).

    Args:
        db_path: Pfad zur SQLite-Datenbank für persistente Speicherung.
                 None = In-Memory (für Tests / Abwärtskompatibilität).
    """

    # Pflicht-Einwilligungen für Versicherungsberatung
    REQUIRED_FOR_INSURANCE = [
        ConsentPurpose.AI_PROCESSING,
        ConsentPurpose.INSURANCE_ADVICE,
    ]

    _CREATE_TABLE = """
        CREATE TABLE IF NOT EXISTS consents (
            consent_id  TEXT PRIMARY KEY,
            user_id     TEXT NOT NULL,
            purpose     TEXT NOT NULL,
            status      TEXT NOT NULL DEFAULT 'pending',
            granted_at  TEXT DEFAULT '',
            expires_at  TEXT DEFAULT '',
            withdrawn_at TEXT DEFAULT '',
            legal_basis TEXT DEFAULT 'Art. 6(1)(a) DSGVO',
            description TEXT DEFAULT '',
            created_at  TEXT DEFAULT ''
        )
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is not None:
            db_path = Path(db_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(str(db_path))
        else:
            self._db = sqlite3.connect(":memory:")
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.execute(self._CREATE_TABLE)
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_consents_user ON consents(user_id)"
        )
        self._db.commit()

        # In-memory cache for live object references (tests check obj.status directly)
        self._cache: dict[str, UserConsent] = {}

        # Counter: max existing ID
        row = self._db.execute(
            "SELECT consent_id FROM consents ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        if row and row[0].startswith("CON-"):
            try:
                self._counter = int(row[0][4:])
            except ValueError:
                self._counter = 0
        else:
            self._counter = 0

    def _row_to_consent(self, row: tuple) -> UserConsent:
        return UserConsent(
            consent_id=row[0],
            user_id=row[1],
            purpose=ConsentPurpose(row[2]),
            status=ConsentStatus(row[3]),
            granted_at=row[4] or "",
            expires_at=row[5] or "",
            withdrawn_at=row[6] or "",
            legal_basis=row[7] or "Art. 6(1)(a) DSGVO",
            description=row[8] or "",
        )

    def request_consent(
        self,
        user_id: str,
        purpose: ConsentPurpose,
        *,
        description: str = "",
        ttl_days: int = 365,
    ) -> UserConsent:
        self._counter += 1
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        expires_ts = time.time() + ttl_days * 86400
        expires = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(expires_ts))

        consent = UserConsent(
            consent_id=f"CON-{self._counter:04d}",
            user_id=user_id,
            purpose=purpose,
            description=description or f"Einwilligung für {purpose.value}",
            expires_at=expires,
        )
        self._db.execute(
            "INSERT INTO consents (consent_id, user_id, purpose, status, "
            "granted_at, expires_at, withdrawn_at, legal_basis, description, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (consent.consent_id, user_id, purpose.value, consent.status.value,
             consent.granted_at, consent.expires_at, consent.withdrawn_at,
             consent.legal_basis, consent.description, now),
        )
        self._db.commit()
        self._cache[consent.consent_id] = consent
        return consent

    def grant(self, consent_id: str) -> bool:
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        cur = self._db.execute(
            "UPDATE consents SET status=?, granted_at=? WHERE consent_id=?",
            (ConsentStatus.GRANTED.value, now, consent_id),
        )
        self._db.commit()
        if cur.rowcount > 0:
            cached = self._cache.get(consent_id)
            if cached:
                cached.status = ConsentStatus.GRANTED
                cached.granted_at = now
            return True
        return False

    def deny(self, consent_id: str) -> bool:
        cur = self._db.execute(
            "UPDATE consents SET status=? WHERE consent_id=?",
            (ConsentStatus.DENIED.value, consent_id),
        )
        self._db.commit()
        if cur.rowcount > 0:
            cached = self._cache.get(consent_id)
            if cached:
                cached.status = ConsentStatus.DENIED
            return True
        return False

    def withdraw(self, consent_id: str) -> bool:
        """Widerruf -- DSGVO Art. 7(3)."""
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        cur = self._db.execute(
            "UPDATE consents SET status=?, withdrawn_at=? WHERE consent_id=?",
            (ConsentStatus.WITHDRAWN.value, now, consent_id),
        )
        self._db.commit()
        if cur.rowcount > 0:
            cached = self._cache.get(consent_id)
            if cached:
                cached.status = ConsentStatus.WITHDRAWN
                cached.withdrawn_at = now
            return True
        return False

    def has_consent(self, user_id: str, purpose: ConsentPurpose) -> bool:
        for c in self.user_consents(user_id):
            if c.purpose == purpose and c.is_valid:
                return True
        return False

    def can_advise(self, user_id: str) -> bool:
        """Prüft ob alle Pflicht-Einwilligungen für Beratung vorliegen."""
        return all(self.has_consent(user_id, p) for p in self.REQUIRED_FOR_INSURANCE)

    def user_consents(self, user_id: str) -> list[UserConsent]:
        rows = self._db.execute(
            "SELECT consent_id, user_id, purpose, status, granted_at, "
            "expires_at, withdrawn_at, legal_basis, description "
            "FROM consents WHERE user_id=? ORDER BY rowid",
            (user_id,),
        ).fetchall()
        return [self._row_to_consent(r) for r in rows]

    def withdraw_all(self, user_id: str) -> int:
        """Alle Einwilligungen eines Nutzers widerrufen."""
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        cur = self._db.execute(
            "UPDATE consents SET status=?, withdrawn_at=? "
            "WHERE user_id=? AND status=?",
            (ConsentStatus.WITHDRAWN.value, now, user_id, ConsentStatus.GRANTED.value),
        )
        self._db.commit()
        return cur.rowcount

    @property
    def user_count(self) -> int:
        row = self._db.execute(
            "SELECT COUNT(DISTINCT user_id) FROM consents"
        ).fetchone()
        return row[0] if row else 0

    def stats(self) -> dict[str, Any]:
        row = self._db.execute(
            "SELECT "
            "  COUNT(DISTINCT user_id), "
            "  COUNT(*), "
            "  SUM(CASE WHEN status='granted' THEN 1 ELSE 0 END), "
            "  SUM(CASE WHEN status='denied' THEN 1 ELSE 0 END), "
            "  SUM(CASE WHEN status='withdrawn' THEN 1 ELSE 0 END), "
            "  SUM(CASE WHEN status='pending' THEN 1 ELSE 0 END) "
            "FROM consents"
        ).fetchone()
        return {
            "total_users": row[0] or 0,
            "total_consents": row[1] or 0,
            "granted": row[2] or 0,
            "denied": row[3] or 0,
            "withdrawn": row[4] or 0,
            "pending": row[5] or 0,
        }


# ============================================================================
# Decision View (Verständliche Darstellung)
# ============================================================================


@dataclass
class SimpleDecisionView:
    """Verständliche Entscheidungsdarstellung für Endnutzer.

    Keine technischen Details -- nur was der Nutzer wissen muss.
    """

    view_id: str
    question: str  # "Welche BU-Versicherung passt zu mir?"
    recommendation: str  # "WWK BU Premium"
    confidence_label: str  # "Hohe Sicherheit" / "Mittlere Sicherheit"
    why_this: list[str] = field(default_factory=list)  # ["Hohe Leistungsquote", ...]
    what_else: list[str] = field(default_factory=list)  # ["R&V Classic als günstigere Alternative"]
    what_to_watch: list[str] = field(
        default_factory=list
    )  # ["Gesundheitsfragen genau beantworten"]
    data_used: list[str] = field(default_factory=list)  # ["Alter", "Beruf", "Gesundheitsstatus"]
    ai_disclosure: str = "Diese Empfehlung wurde mit KI-Unterstützung erstellt (Art. 52 EU-AI-Act)."
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "view_id": self.view_id,
            "question": self.question,
            "recommendation": self.recommendation,
            "confidence": self.confidence_label,
            "why": self.why_this,
            "alternatives": self.what_else,
            "cautions": self.what_to_watch,
            "data_used": self.data_used,
            "ai_disclosure": self.ai_disclosure,
        }


class DecisionViewBuilder:
    """Erstellt verständliche Entscheidungsansichten."""

    CONFIDENCE_LABELS = {
        (0.9, 1.0): "Sehr hohe Sicherheit",
        (0.7, 0.9): "Hohe Sicherheit",
        (0.5, 0.7): "Mittlere Sicherheit",
        (0.0, 0.5): "Vorläufige Einschätzung",
    }

    def __init__(self) -> None:
        self._views: list[SimpleDecisionView] = []
        self._counter = 0

    def build(
        self,
        question: str,
        recommendation: str,
        confidence: float,
        *,
        why: list[str] | None = None,
        alternatives: list[str] | None = None,
        cautions: list[str] | None = None,
        data_used: list[str] | None = None,
    ) -> SimpleDecisionView:
        self._counter += 1
        label = "Einschätzung"
        for (low, high), lbl in self.CONFIDENCE_LABELS.items():
            if low <= confidence <= high:
                label = lbl
                break

        view = SimpleDecisionView(
            view_id=f"DV-{self._counter:04d}",
            question=question,
            recommendation=recommendation,
            confidence_label=label,
            why_this=why or [],
            what_else=alternatives or [],
            what_to_watch=cautions or [],
            data_used=data_used or [],
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        self._views.append(view)
        return view

    def all_views(self) -> list[SimpleDecisionView]:
        return list(self._views)

    @property
    def view_count(self) -> int:
        return len(self._views)

    def stats(self) -> dict[str, Any]:
        return {
            "total_views": len(self._views),
        }


# ============================================================================
# User Notifications
# ============================================================================


class NotificationType(Enum):
    CONSENT_REQUEST = "consent_request"
    DECISION_READY = "decision_ready"
    DATA_USAGE = "data_usage"
    SECURITY_ALERT = "security_alert"
    CONSENT_EXPIRING = "consent_expiring"
    RIGHT_TO_EXPLAIN = "right_to_explain"


@dataclass
class UserNotification:
    """Eine Benachrichtigung für den Endnutzer."""

    notification_id: str
    user_id: str
    ntype: NotificationType
    title: str
    message: str
    read: bool = False
    created_at: str = ""
    action_url: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.notification_id,
            "type": self.ntype.value,
            "title": self.title,
            "read": self.read,
            "created_at": self.created_at,
        }


class NotificationCenter:
    """Benachrichtigungszentrale für Endnutzer."""

    def __init__(self) -> None:
        self._notifications: dict[str, list[UserNotification]] = {}
        self._counter = 0

    def notify(
        self,
        user_id: str,
        ntype: NotificationType,
        title: str,
        message: str,
        *,
        action_url: str = "",
    ) -> UserNotification:
        self._counter += 1
        n = UserNotification(
            notification_id=f"NOT-{self._counter:04d}",
            user_id=user_id,
            ntype=ntype,
            title=title,
            message=message,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            action_url=action_url,
        )
        self._notifications.setdefault(user_id, []).append(n)
        return n

    def mark_read(self, notification_id: str) -> bool:
        for notifs in self._notifications.values():
            for n in notifs:
                if n.notification_id == notification_id:
                    n.read = True
                    return True
        return False

    def unread(self, user_id: str) -> list[UserNotification]:
        return [n for n in self._notifications.get(user_id, []) if not n.read]

    def user_notifications(self, user_id: str, limit: int = 20) -> list[UserNotification]:
        return list(reversed(self._notifications.get(user_id, [])[-limit:]))

    @property
    def total_count(self) -> int:
        return sum(len(n) for n in self._notifications.values())

    def stats(self) -> dict[str, Any]:
        all_n = [n for notifs in self._notifications.values() for n in notifs]
        return {
            "total_notifications": len(all_n),
            "users": len(self._notifications),
            "unread": sum(1 for n in all_n if not n.read),
        }


# ============================================================================
# User Activity Log
# ============================================================================


@dataclass
class UserActivity:
    """Eine nachvollziehbare Nutzeraktivität."""

    activity_id: str
    user_id: str
    action: str
    details: str = ""
    data_accessed: list[str] = field(default_factory=list)
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.activity_id,
            "action": self.action,
            "details": self.details,
            "data": self.data_accessed,
            "time": self.timestamp,
        }


class UserActivityLog:
    """Nachvollziehbare Aktivitätshistorie (Art. 15 DSGVO Auskunftsrecht)."""

    def __init__(self) -> None:
        self._activities: dict[str, list[UserActivity]] = {}
        self._counter = 0

    def log(
        self,
        user_id: str,
        action: str,
        *,
        details: str = "",
        data_accessed: list[str] | None = None,
    ) -> UserActivity:
        self._counter += 1
        activity = UserActivity(
            activity_id=f"ACT-{self._counter:04d}",
            user_id=user_id,
            action=action,
            details=details,
            data_accessed=data_accessed or [],
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        self._activities.setdefault(user_id, []).append(activity)
        return activity

    def user_history(self, user_id: str, limit: int = 50) -> list[UserActivity]:
        return list(reversed(self._activities.get(user_id, [])[-limit:]))

    def data_access_report(self, user_id: str) -> dict[str, int]:
        """DSGVO Art. 15: Welche Daten wurden wie oft verarbeitet?"""
        data_counts: dict[str, int] = {}
        for act in self._activities.get(user_id, []):
            for d in act.data_accessed:
                data_counts[d] = data_counts.get(d, 0) + 1
        return data_counts

    def export_user_data(self, user_id: str) -> dict[str, Any]:
        """DSGVO Art. 20: Datenportabilität."""
        return {
            "user_id": user_id,
            "activities": [a.to_dict() for a in self._activities.get(user_id, [])],
            "data_access_summary": self.data_access_report(user_id),
            "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    def delete_user_data(self, user_id: str) -> int:
        """DSGVO Art. 17: Recht auf Löschung."""
        activities = self._activities.pop(user_id, [])
        return len(activities)

    @property
    def user_count(self) -> int:
        return len(self._activities)

    def stats(self) -> dict[str, Any]:
        all_a = [a for acts in self._activities.values() for a in acts]
        return {
            "total_users": len(self._activities),
            "total_activities": len(all_a),
        }


# ============================================================================
# User Portal (Hauptklasse)
# ============================================================================


class UserPortal:
    """Hauptklasse: Endnutzer-Portal mit Einwilligung, Transparenz, DSGVO."""

    def __init__(self, consent_db_path: str | Path | None = None) -> None:
        self._consents = ConsentManager(db_path=consent_db_path)
        self._decisions = DecisionViewBuilder()
        self._notifications = NotificationCenter()
        self._activities = UserActivityLog()

    @property
    def consents(self) -> ConsentManager:
        return self._consents

    @property
    def decisions(self) -> DecisionViewBuilder:
        return self._decisions

    @property
    def notifications(self) -> NotificationCenter:
        return self._notifications

    @property
    def activities(self) -> UserActivityLog:
        return self._activities

    def onboard_user(self, user_id: str) -> list[UserConsent]:
        """Erstellt alle Pflicht-Einwilligungsanfragen für einen neuen Nutzer."""
        consents = []
        for purpose in ConsentPurpose:
            c = self._consents.request_consent(
                user_id,
                purpose,
                description=f"Einwilligung zur {purpose.value}",
            )
            consents.append(c)

        self._notifications.notify(
            user_id,
            NotificationType.CONSENT_REQUEST,
            "Einwilligungen erforderlich",
            "Bitte prüfen und bestätigen Sie Ihre Datenschutz-Einwilligungen.",
        )
        self._activities.log(user_id, "onboarding", details="Nutzer-Portal eingerichtet")
        return consents

    def user_dashboard(self, user_id: str) -> dict[str, Any]:
        """Aggregierte Dashboard-Daten für einen Endnutzer."""
        return {
            "consents": [c.to_dict() for c in self._consents.user_consents(user_id)],
            "can_advise": self._consents.can_advise(user_id),
            "decisions": [d.to_dict() for d in self._decisions.all_views()],
            "unread_notifications": len(self._notifications.unread(user_id)),
            "recent_activity": [a.to_dict() for a in self._activities.user_history(user_id, 10)],
        }

    def exercise_right_to_erasure(self, user_id: str) -> dict[str, int]:
        """DSGVO Art. 17: Komplettlöschung aller Nutzerdaten."""
        withdrawn = self._consents.withdraw_all(user_id)
        deleted = self._activities.delete_user_data(user_id)
        return {"consents_withdrawn": withdrawn, "activities_deleted": deleted}

    def stats(self) -> dict[str, Any]:
        return {
            "consents": self._consents.stats(),
            "decisions": self._decisions.stats(),
            "notifications": self._notifications.stats(),
            "activities": self._activities.stats(),
        }
