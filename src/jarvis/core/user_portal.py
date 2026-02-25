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

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ============================================================================
# User Consent (DSGVO Art. 7)
# ============================================================================


class ConsentPurpose(Enum):
    AI_PROCESSING = "ai_processing"             # KI-Verarbeitung
    DATA_ANALYSIS = "data_analysis"             # Datenanalyse
    PROFILING = "profiling"                      # Profilbildung
    THIRD_PARTY = "third_party_sharing"         # Weitergabe an Dritte
    MARKETING = "marketing"                      # Marketing
    INSURANCE_ADVICE = "insurance_advice"        # Versicherungsberatung
    HEALTH_DATA = "health_data"                  # Gesundheitsdaten (BU)
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
    """Verwaltet Einwilligungen pro Nutzer (DSGVO-konform)."""

    # Pflicht-Einwilligungen für Versicherungsberatung
    REQUIRED_FOR_INSURANCE = [
        ConsentPurpose.AI_PROCESSING,
        ConsentPurpose.INSURANCE_ADVICE,
    ]

    def __init__(self) -> None:
        self._consents: dict[str, list[UserConsent]] = {}
        self._counter = 0

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
        self._consents.setdefault(user_id, []).append(consent)
        return consent

    def grant(self, consent_id: str) -> bool:
        for consents in self._consents.values():
            for c in consents:
                if c.consent_id == consent_id:
                    c.status = ConsentStatus.GRANTED
                    c.granted_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    return True
        return False

    def deny(self, consent_id: str) -> bool:
        for consents in self._consents.values():
            for c in consents:
                if c.consent_id == consent_id:
                    c.status = ConsentStatus.DENIED
                    return True
        return False

    def withdraw(self, consent_id: str) -> bool:
        """Widerruf -- DSGVO Art. 7(3)."""
        for consents in self._consents.values():
            for c in consents:
                if c.consent_id == consent_id:
                    c.status = ConsentStatus.WITHDRAWN
                    c.withdrawn_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    return True
        return False

    def has_consent(self, user_id: str, purpose: ConsentPurpose) -> bool:
        for c in self._consents.get(user_id, []):
            if c.purpose == purpose and c.is_valid:
                return True
        return False

    def can_advise(self, user_id: str) -> bool:
        """Prüft ob alle Pflicht-Einwilligungen für Beratung vorliegen."""
        return all(self.has_consent(user_id, p) for p in self.REQUIRED_FOR_INSURANCE)

    def user_consents(self, user_id: str) -> list[UserConsent]:
        return self._consents.get(user_id, [])

    def withdraw_all(self, user_id: str) -> int:
        """Alle Einwilligungen eines Nutzers widerrufen."""
        count = 0
        for c in self._consents.get(user_id, []):
            if c.status == ConsentStatus.GRANTED:
                c.status = ConsentStatus.WITHDRAWN
                c.withdrawn_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                count += 1
        return count

    @property
    def user_count(self) -> int:
        return len(self._consents)

    def stats(self) -> dict[str, Any]:
        all_c = [c for consents in self._consents.values() for c in consents]
        return {
            "total_users": len(self._consents),
            "total_consents": len(all_c),
            "granted": sum(1 for c in all_c if c.status == ConsentStatus.GRANTED),
            "denied": sum(1 for c in all_c if c.status == ConsentStatus.DENIED),
            "withdrawn": sum(1 for c in all_c if c.status == ConsentStatus.WITHDRAWN),
            "pending": sum(1 for c in all_c if c.status == ConsentStatus.PENDING),
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
    question: str                        # "Welche BU-Versicherung passt zu mir?"
    recommendation: str                  # "WWK BU Premium"
    confidence_label: str               # "Hohe Sicherheit" / "Mittlere Sicherheit"
    why_this: list[str] = field(default_factory=list)         # ["Hohe Leistungsquote", ...]
    what_else: list[str] = field(default_factory=list)        # ["R&V Classic als günstigere Alternative"]
    what_to_watch: list[str] = field(default_factory=list)    # ["Gesundheitsfragen genau beantworten"]
    data_used: list[str] = field(default_factory=list)        # ["Alter", "Beruf", "Gesundheitsstatus"]
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

    def __init__(self) -> None:
        self._consents = ConsentManager()
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
                user_id, purpose,
                description=f"Einwilligung zur {purpose.value}",
            )
            consents.append(c)

        self._notifications.notify(
            user_id, NotificationType.CONSENT_REQUEST,
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
