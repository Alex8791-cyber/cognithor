"""Jarvis · Multi-Tenant, Trust-Negotiation & Emergency-Updates.

Enterprise-Ready Features:

  - Tenant:                Mandant mit eigener Konfiguration
  - TenantManager:         Multi-Tenant-Verwaltung (isolierte Daten)
  - TrustPolicy:           Vertrauensregeln für Federation-Links
  - TrustNegotiator:       Automatische Trust-Aushandlung zwischen Agenten
  - EmergencyController:   Notfall-Updates, Kill-Switches, Quarantäne
  - MultiTenantGovernor:   Hauptklasse

Architektur-Bibel: §17.1 (Multi-Tenancy), §12.4 (Trust), §14.8 (Emergency)
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ============================================================================
# Multi-Tenant
# ============================================================================


class TenantStatus(Enum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    TRIAL = "trial"
    ARCHIVED = "archived"


class TenantPlan(Enum):
    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


# Plan-Limits
PLAN_LIMITS: dict[TenantPlan, dict[str, Any]] = {
    TenantPlan.FREE: {
        "max_agents": 1,
        "max_skills": 10,
        "max_users": 1,
        "daily_budget_eur": 5.0,
        "monthly_budget_eur": 50.0,
        "federation_enabled": False,
        "audit_retention_days": 30,
        "priority_support": False,
    },
    TenantPlan.STARTER: {
        "max_agents": 3,
        "max_skills": 50,
        "max_users": 5,
        "daily_budget_eur": 25.0,
        "monthly_budget_eur": 500.0,
        "federation_enabled": False,
        "audit_retention_days": 90,
        "priority_support": False,
    },
    TenantPlan.PROFESSIONAL: {
        "max_agents": 10,
        "max_skills": 200,
        "max_users": 25,
        "daily_budget_eur": 100.0,
        "monthly_budget_eur": 2000.0,
        "federation_enabled": True,
        "audit_retention_days": 365,
        "priority_support": True,
    },
    TenantPlan.ENTERPRISE: {
        "max_agents": -1,  # Unbegrenzt
        "max_skills": -1,
        "max_users": -1,
        "daily_budget_eur": -1,
        "monthly_budget_eur": -1,
        "federation_enabled": True,
        "audit_retention_days": 730,
        "priority_support": True,
    },
}


@dataclass
class TenantUser:
    """Ein Benutzer innerhalb eines Mandanten."""

    user_id: str
    name: str
    email: str
    role: str = "user"  # admin, user, viewer
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "name": self.name,
            "email": self.email,
            "role": self.role,
        }


@dataclass
class Tenant:
    """Ein Mandant mit eigener Konfiguration und Isolation."""

    tenant_id: str
    name: str
    plan: TenantPlan = TenantPlan.FREE
    status: TenantStatus = TenantStatus.ACTIVE
    owner_email: str = ""
    created_at: str = ""
    users: list[TenantUser] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)

    # Isolierte Pfade
    @property
    def data_path(self) -> str:
        return f"/data/tenants/{self.tenant_id}"

    @property
    def secrets_path(self) -> str:
        return f"/run/secrets/tenants/{self.tenant_id}"

    @property
    def db_name(self) -> str:
        return f"jarvis_{self.tenant_id}"

    @property
    def limits(self) -> dict[str, Any]:
        return PLAN_LIMITS.get(self.plan, PLAN_LIMITS[TenantPlan.FREE])

    def can_add_agent(self, current_agents: int) -> bool:
        max_a = self.limits["max_agents"]
        return max_a == -1 or current_agents < max_a

    def can_add_user(self) -> bool:
        max_u = self.limits["max_users"]
        return max_u == -1 or len(self.users) < max_u

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "name": self.name,
            "plan": self.plan.value,
            "status": self.status.value,
            "users": len(self.users),
            "limits": self.limits,
            "data_path": self.data_path,
        }


class TenantManager:
    """Multi-Tenant-Verwaltung mit vollständiger Isolation."""

    def __init__(self) -> None:
        self._tenants: dict[str, Tenant] = {}
        self._counter = 0

    def create(
        self,
        name: str,
        owner_email: str,
        plan: TenantPlan = TenantPlan.FREE,
    ) -> Tenant:
        self._counter += 1
        tenant_id = hashlib.sha256(f"tenant:{name}:{time.time()}".encode()).hexdigest()[:12]
        tenant = Tenant(
            tenant_id=tenant_id,
            name=name,
            plan=plan,
            owner_email=owner_email,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            users=[TenantUser(
                user_id=f"usr-{tenant_id[:6]}-001",
                name="Admin",
                email=owner_email,
                role="admin",
                created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )],
        )
        self._tenants[tenant_id] = tenant
        return tenant

    def get(self, tenant_id: str) -> Tenant | None:
        return self._tenants.get(tenant_id)

    def find_by_email(self, email: str) -> list[Tenant]:
        return [
            t for t in self._tenants.values()
            if t.owner_email == email or any(u.email == email for u in t.users)
        ]

    def suspend(self, tenant_id: str) -> bool:
        tenant = self._tenants.get(tenant_id)
        if tenant:
            tenant.status = TenantStatus.SUSPENDED
            return True
        return False

    def activate(self, tenant_id: str) -> bool:
        tenant = self._tenants.get(tenant_id)
        if tenant:
            tenant.status = TenantStatus.ACTIVE
            return True
        return False

    def upgrade(self, tenant_id: str, plan: TenantPlan) -> bool:
        tenant = self._tenants.get(tenant_id)
        if tenant:
            tenant.plan = plan
            return True
        return False

    def add_user(self, tenant_id: str, user: TenantUser) -> bool:
        tenant = self._tenants.get(tenant_id)
        if not tenant or not tenant.can_add_user():
            return False
        tenant.users.append(user)
        return True

    def remove_user(self, tenant_id: str, user_id: str) -> bool:
        tenant = self._tenants.get(tenant_id)
        if not tenant:
            return False
        before = len(tenant.users)
        tenant.users = [u for u in tenant.users if u.user_id != user_id]
        return len(tenant.users) < before

    def active_tenants(self) -> list[Tenant]:
        return [t for t in self._tenants.values() if t.status == TenantStatus.ACTIVE]

    @property
    def tenant_count(self) -> int:
        return len(self._tenants)

    def stats(self) -> dict[str, Any]:
        tenants = list(self._tenants.values())
        return {
            "total_tenants": len(tenants),
            "active": sum(1 for t in tenants if t.status == TenantStatus.ACTIVE),
            "by_plan": {
                p.value: sum(1 for t in tenants if t.plan == p)
                for p in TenantPlan
                if any(t.plan == p for t in tenants)
            },
            "total_users": sum(len(t.users) for t in tenants),
        }


# ============================================================================
# Trust Negotiation (Cross-Agent)
# ============================================================================


class TrustLevel(Enum):
    UNTRUSTED = "untrusted"
    BASIC = "basic"
    VERIFIED = "verified"
    TRUSTED = "trusted"
    PRIVILEGED = "privileged"


@dataclass
class TrustPolicy:
    """Vertrauensregeln für Federation-Links."""

    policy_id: str
    min_trust_for_delegation: TrustLevel = TrustLevel.VERIFIED
    min_trust_for_data_share: TrustLevel = TrustLevel.TRUSTED
    min_trust_for_admin: TrustLevel = TrustLevel.PRIVILEGED
    require_mutual_auth: bool = True
    require_encrypted_transport: bool = True
    max_delegation_depth: int = 2  # A→B→C, nicht tiefer
    allow_transitive_trust: bool = False
    timeout_hours: int = 24

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "delegation_min": self.min_trust_for_delegation.value,
            "data_share_min": self.min_trust_for_data_share.value,
            "mutual_auth": self.require_mutual_auth,
            "encrypted": self.require_encrypted_transport,
            "max_depth": self.max_delegation_depth,
            "transitive": self.allow_transitive_trust,
        }


@dataclass
class TrustRelation:
    """Vertrauensbeziehung zwischen zwei Agenten."""

    local_agent_id: str
    remote_agent_id: str
    trust_level: TrustLevel = TrustLevel.UNTRUSTED
    established_at: str = ""
    last_verified: str = ""
    verification_method: str = ""  # "public_key", "challenge_response", "manual"
    violations: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "local": self.local_agent_id,
            "remote": self.remote_agent_id,
            "trust": self.trust_level.value,
            "violations": self.violations,
            "verified": self.last_verified,
        }


class TrustNegotiator:
    """Automatische Trust-Aushandlung zwischen Agenten."""

    def __init__(self, policy: TrustPolicy | None = None) -> None:
        self._policy = policy or TrustPolicy(policy_id="default")
        self._relations: dict[str, TrustRelation] = {}

    @property
    def policy(self) -> TrustPolicy:
        return self._policy

    def _key(self, local: str, remote: str) -> str:
        return f"{local}↔{remote}"

    def initiate(self, local_id: str, remote_id: str, method: str = "public_key") -> TrustRelation:
        """Initiiert eine Vertrauensbeziehung."""
        key = self._key(local_id, remote_id)
        relation = TrustRelation(
            local_agent_id=local_id,
            remote_agent_id=remote_id,
            trust_level=TrustLevel.BASIC,
            established_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            verification_method=method,
        )
        self._relations[key] = relation
        return relation

    def verify(self, local_id: str, remote_id: str) -> TrustRelation | None:
        """Verifiziert und erhöht das Trust-Level."""
        key = self._key(local_id, remote_id)
        relation = self._relations.get(key)
        if not relation:
            return None

        # Stufe hochsetzen
        levels = [TrustLevel.UNTRUSTED, TrustLevel.BASIC, TrustLevel.VERIFIED, TrustLevel.TRUSTED, TrustLevel.PRIVILEGED]
        idx = levels.index(relation.trust_level)
        if idx < len(levels) - 1:
            relation.trust_level = levels[idx + 1]
        relation.last_verified = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return relation

    def report_violation(self, local_id: str, remote_id: str) -> TrustRelation | None:
        """Meldet einen Vertrauensbruch → Trust-Level sinkt."""
        key = self._key(local_id, remote_id)
        relation = self._relations.get(key)
        if not relation:
            return None
        relation.violations += 1
        # Bei 3+ Violations: auf Untrusted zurücksetzen
        if relation.violations >= 3:
            relation.trust_level = TrustLevel.UNTRUSTED
        elif relation.trust_level != TrustLevel.UNTRUSTED:
            levels = [TrustLevel.UNTRUSTED, TrustLevel.BASIC, TrustLevel.VERIFIED, TrustLevel.TRUSTED, TrustLevel.PRIVILEGED]
            idx = levels.index(relation.trust_level)
            if idx > 0:
                relation.trust_level = levels[idx - 1]
        return relation

    def can_delegate(self, local_id: str, remote_id: str) -> bool:
        """Prüft ob Delegation erlaubt ist."""
        key = self._key(local_id, remote_id)
        relation = self._relations.get(key)
        if not relation:
            return False
        levels = [TrustLevel.UNTRUSTED, TrustLevel.BASIC, TrustLevel.VERIFIED, TrustLevel.TRUSTED, TrustLevel.PRIVILEGED]
        return levels.index(relation.trust_level) >= levels.index(self._policy.min_trust_for_delegation)

    def can_share_data(self, local_id: str, remote_id: str) -> bool:
        key = self._key(local_id, remote_id)
        relation = self._relations.get(key)
        if not relation:
            return False
        levels = [TrustLevel.UNTRUSTED, TrustLevel.BASIC, TrustLevel.VERIFIED, TrustLevel.TRUSTED, TrustLevel.PRIVILEGED]
        return levels.index(relation.trust_level) >= levels.index(self._policy.min_trust_for_data_share)

    def get_relation(self, local_id: str, remote_id: str) -> TrustRelation | None:
        return self._relations.get(self._key(local_id, remote_id))

    def all_relations(self) -> list[TrustRelation]:
        return list(self._relations.values())

    @property
    def relation_count(self) -> int:
        return len(self._relations)

    def stats(self) -> dict[str, Any]:
        relations = list(self._relations.values())
        return {
            "total_relations": len(relations),
            "by_trust_level": {
                level.value: sum(1 for r in relations if r.trust_level == level)
                for level in TrustLevel
                if any(r.trust_level == level for r in relations)
            },
            "total_violations": sum(r.violations for r in relations),
        }


# ============================================================================
# Emergency Controller
# ============================================================================


class EmergencyAction(Enum):
    KILL_SWITCH = "kill_switch"               # Alles stoppen
    QUARANTINE_AGENT = "quarantine_agent"     # Einzelnen Agent isolieren
    QUARANTINE_SKILL = "quarantine_skill"     # Skill deaktivieren
    REVOKE_FEDERATION = "revoke_federation"   # Federation-Link kappen
    EMERGENCY_UPDATE = "emergency_update"     # Sofort-Update pushen
    LOCKDOWN = "lockdown"                     # Nur Admin-Zugriff
    ROLLBACK = "rollback"                     # Auf letzte sichere Version


@dataclass
class EmergencyEvent:
    """Ein Notfall-Event."""

    event_id: str
    action: EmergencyAction
    reason: str
    target: str = ""  # agent_id, skill_id, etc.
    executed_by: str = ""
    timestamp: str = ""
    reverted: bool = False
    reverted_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "action": self.action.value,
            "reason": self.reason,
            "target": self.target,
            "executed_by": self.executed_by,
            "timestamp": self.timestamp,
            "reverted": self.reverted,
        }


class EmergencyController:
    """Notfall-Steuerung: Kill-Switches, Quarantäne, Emergency-Updates."""

    def __init__(self) -> None:
        self._events: list[EmergencyEvent] = []
        self._counter = 0
        self._lockdown_active = False
        self._quarantined_agents: set[str] = set()
        self._quarantined_skills: set[str] = set()

    def execute(
        self,
        action: EmergencyAction,
        reason: str,
        *,
        target: str = "",
        executed_by: str = "system",
    ) -> EmergencyEvent:
        """Führt eine Notfall-Aktion aus."""
        self._counter += 1
        event = EmergencyEvent(
            event_id=f"EMG-{self._counter:04d}",
            action=action,
            reason=reason,
            target=target,
            executed_by=executed_by,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        self._events.append(event)

        # Aktion durchführen
        if action == EmergencyAction.KILL_SWITCH:
            self._lockdown_active = True
            # KILL_SWITCH: lockdown blocks ALL operations via is_lockdown check
        elif action == EmergencyAction.LOCKDOWN:
            self._lockdown_active = True
        elif action == EmergencyAction.QUARANTINE_AGENT:
            self._quarantined_agents.add(target)
        elif action == EmergencyAction.QUARANTINE_SKILL:
            self._quarantined_skills.add(target)
        elif action == EmergencyAction.ROLLBACK:
            self._lockdown_active = False
            self._quarantined_agents.clear()
            self._quarantined_skills.clear()

        return event

    def revert(self, event_id: str) -> bool:
        """Macht eine Notfall-Aktion rückgängig."""
        for event in self._events:
            if event.event_id == event_id and not event.reverted:
                event.reverted = True
                event.reverted_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                if event.action == EmergencyAction.LOCKDOWN:
                    self._lockdown_active = False
                elif event.action == EmergencyAction.QUARANTINE_AGENT:
                    self._quarantined_agents.discard(event.target)
                elif event.action == EmergencyAction.QUARANTINE_SKILL:
                    self._quarantined_skills.discard(event.target)
                return True
        return False

    @property
    def is_lockdown(self) -> bool:
        return self._lockdown_active

    def is_quarantined(self, agent_or_skill_id: str) -> bool:
        return agent_or_skill_id in self._quarantined_agents or agent_or_skill_id in self._quarantined_skills

    @property
    def quarantined_agents(self) -> set[str]:
        return set(self._quarantined_agents)

    @property
    def quarantined_skills(self) -> set[str]:
        return set(self._quarantined_skills)

    def history(self, limit: int = 20) -> list[EmergencyEvent]:
        return list(reversed(self._events[-limit:]))

    @property
    def event_count(self) -> int:
        return len(self._events)

    def stats(self) -> dict[str, Any]:
        events = self._events
        return {
            "total_events": len(events),
            "lockdown_active": self._lockdown_active,
            "quarantined_agents": len(self._quarantined_agents),
            "quarantined_skills": len(self._quarantined_skills),
            "by_action": {
                a.value: sum(1 for e in events if e.action == a)
                for a in EmergencyAction
                if any(e.action == a for e in events)
            },
        }


# ============================================================================
# Multi-Tenant Governor (Hauptklasse)
# ============================================================================


class MultiTenantGovernor:
    """Hauptklasse: Multi-Tenant + Trust + Emergency."""

    def __init__(self) -> None:
        self._tenants = TenantManager()
        self._trust = TrustNegotiator()
        self._emergency = EmergencyController()

    @property
    def tenants(self) -> TenantManager:
        return self._tenants

    @property
    def trust(self) -> TrustNegotiator:
        return self._trust

    @property
    def emergency(self) -> EmergencyController:
        return self._emergency

    def pre_action_check(self, tenant_id: str, agent_id: str) -> dict[str, Any]:
        """Pre-Flight-Check vor jeder Agent-Aktion."""
        if self._emergency.is_lockdown:
            return {"allowed": False, "reason": "System im Lockdown-Modus"}
        if self._emergency.is_quarantined(agent_id):
            return {"allowed": False, "reason": f"Agent '{agent_id}' ist quarantiniert"}
        tenant = self._tenants.get(tenant_id)
        if not tenant:
            return {"allowed": False, "reason": f"Tenant '{tenant_id}' nicht gefunden"}
        if tenant.status != TenantStatus.ACTIVE:
            return {"allowed": False, "reason": f"Tenant '{tenant.name}' ist {tenant.status.value}"}
        return {"allowed": True, "reason": "OK"}

    def stats(self) -> dict[str, Any]:
        return {
            "tenants": self._tenants.stats(),
            "trust": self._trust.stats(),
            "emergency": self._emergency.stats(),
        }
