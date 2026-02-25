"""Jarvis · Strikte Sandbox-Isolierung & Multi-Tenant.

Behandelt jeden Agenten wie untrusted Code:

  - AgentSandbox:         Isolierte Ausführungsumgebung pro Agent
  - SecretVault:          Per-Agent Secret-Store (kein zentraler Zugriff)
  - NamespaceIsolation:   Getrennte Namensräume für Memory, Files, Config
  - TenantManager:        Multi-Tenant: getrennte Daten pro Organisation
  - DelegatedAdmin:       Delegierte Admin-Rollen pro Tenant
  - IsolationEnforcer:    Hauptklasse, erzwingt Isolation

Architektur-Bibel: §9.3 (Sandbox), §16.1 (Multi-Tenant)
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ============================================================================
# Agent Sandbox
# ============================================================================


class SandboxState(Enum):
    CREATED = "created"
    RUNNING = "running"
    SUSPENDED = "suspended"
    TERMINATED = "terminated"


class ResourceType(Enum):
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    API_CALLS = "api_calls"


@dataclass
class ResourceLimit:
    """Ressourcen-Limit für eine Sandbox."""

    resource: ResourceType
    max_value: float
    current_value: float = 0.0
    unit: str = ""

    @property
    def utilization(self) -> float:
        return (self.current_value / self.max_value * 100) if self.max_value > 0 else 0

    @property
    def exceeded(self) -> bool:
        return self.current_value > self.max_value

    def to_dict(self) -> dict[str, Any]:
        return {
            "resource": self.resource.value,
            "max": self.max_value,
            "current": round(self.current_value, 2),
            "utilization": round(self.utilization, 1),
            "exceeded": self.exceeded,
        }


@dataclass
class AgentSandbox:
    """Isolierte Ausführungsumgebung für einen Agenten."""

    sandbox_id: str
    agent_id: str
    tenant_id: str = "default"
    state: SandboxState = SandboxState.CREATED
    created_at: str = ""
    limits: dict[ResourceType, ResourceLimit] = field(default_factory=dict)
    allowed_tools: set[str] = field(default_factory=set)
    denied_tools: set[str] = field(default_factory=set)
    allowed_endpoints: set[str] = field(default_factory=set)
    environment: dict[str, str] = field(default_factory=dict)
    filesystem_root: str = ""
    network_isolated: bool = True
    can_spawn_children: bool = False

    def check_tool_access(self, tool_name: str) -> bool:
        if tool_name in self.denied_tools:
            return False
        if self.allowed_tools and tool_name not in self.allowed_tools:
            return False
        return True

    def check_endpoint_access(self, endpoint: str) -> bool:
        if not self.allowed_endpoints:
            return not self.network_isolated
        return any(endpoint.startswith(e) for e in self.allowed_endpoints)

    def consume_resource(self, resource: ResourceType, amount: float) -> bool:
        limit = self.limits.get(resource)
        if not limit:
            return True
        if limit.current_value + amount > limit.max_value:
            return False
        limit.current_value += amount
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "sandbox_id": self.sandbox_id,
            "agent_id": self.agent_id,
            "tenant_id": self.tenant_id,
            "state": self.state.value,
            "limits": {k.value: v.to_dict() for k, v in self.limits.items()},
            "allowed_tools": sorted(self.allowed_tools),
            "network_isolated": self.network_isolated,
        }


class SandboxManager:
    """Verwaltet isolierte Sandbox-Umgebungen."""

    DEFAULT_LIMITS = {
        ResourceType.CPU: ResourceLimit(ResourceType.CPU, 100.0, unit="percent"),
        ResourceType.MEMORY: ResourceLimit(ResourceType.MEMORY, 512.0, unit="MB"),
        ResourceType.DISK: ResourceLimit(ResourceType.DISK, 1024.0, unit="MB"),
        ResourceType.API_CALLS: ResourceLimit(ResourceType.API_CALLS, 1000.0, unit="calls/hour"),
    }

    def __init__(self) -> None:
        self._sandboxes: dict[str, AgentSandbox] = {}

    def create(
        self,
        agent_id: str,
        tenant_id: str = "default",
        *,
        allowed_tools: set[str] | None = None,
        denied_tools: set[str] | None = None,
        network_isolated: bool = True,
        custom_limits: dict[ResourceType, ResourceLimit] | None = None,
    ) -> AgentSandbox:
        sandbox_id = hashlib.sha256(f"sb:{agent_id}:{time.time()}".encode()).hexdigest()[:12]
        limits = custom_limits or {k: ResourceLimit(k, v.max_value, unit=v.unit)
                                   for k, v in self.DEFAULT_LIMITS.items()}

        sandbox = AgentSandbox(
            sandbox_id=sandbox_id,
            agent_id=agent_id,
            tenant_id=tenant_id,
            state=SandboxState.RUNNING,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            limits=limits,
            allowed_tools=allowed_tools or set(),
            denied_tools=denied_tools or set(),
            network_isolated=network_isolated,
            filesystem_root=f"/sandboxes/{tenant_id}/{agent_id}",
        )
        self._sandboxes[sandbox_id] = sandbox
        return sandbox

    def get(self, sandbox_id: str) -> AgentSandbox | None:
        return self._sandboxes.get(sandbox_id)

    def get_by_agent(self, agent_id: str) -> AgentSandbox | None:
        for sb in self._sandboxes.values():
            if sb.agent_id == agent_id and sb.state == SandboxState.RUNNING:
                return sb
        return None

    def terminate(self, sandbox_id: str) -> bool:
        sb = self._sandboxes.get(sandbox_id)
        if sb:
            sb.state = SandboxState.TERMINATED
            return True
        return False

    def suspend(self, sandbox_id: str) -> bool:
        sb = self._sandboxes.get(sandbox_id)
        if sb and sb.state == SandboxState.RUNNING:
            sb.state = SandboxState.SUSPENDED
            return True
        return False

    def running(self) -> list[AgentSandbox]:
        return [sb for sb in self._sandboxes.values() if sb.state == SandboxState.RUNNING]

    @property
    def sandbox_count(self) -> int:
        return len(self._sandboxes)

    def stats(self) -> dict[str, Any]:
        sbs = list(self._sandboxes.values())
        return {
            "total": len(sbs),
            "running": sum(1 for s in sbs if s.state == SandboxState.RUNNING),
            "suspended": sum(1 for s in sbs if s.state == SandboxState.SUSPENDED),
            "terminated": sum(1 for s in sbs if s.state == SandboxState.TERMINATED),
        }


# ============================================================================
# Per-Agent Secret Vault
# ============================================================================


@dataclass
class AgentSecret:
    """Ein Secret, das zu genau einem Agenten gehört."""

    key: str
    encrypted_value: str  # Simuliert verschlüsselt
    agent_id: str
    tenant_id: str = "default"
    created_at: str = ""
    rotated_at: str = ""
    expires_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "agent_id": self.agent_id,
            "tenant_id": self.tenant_id,
            "has_expiry": bool(self.expires_at),
        }


class PerAgentSecretVault:
    """Jeder Agent hat seinen eigenen Secret-Store.

    Kein Agent kann auf Secrets anderer Agenten zugreifen.
    """

    def __init__(self) -> None:
        self._secrets: dict[str, dict[str, AgentSecret]] = {}  # agent_id → {key → secret}
        self._access_log: list[dict[str, Any]] = []

    def store(self, agent_id: str, key: str, value: str, tenant_id: str = "default") -> AgentSecret:
        if agent_id not in self._secrets:
            self._secrets[agent_id] = {}
        encrypted = hashlib.sha256(value.encode()).hexdigest()
        secret = AgentSecret(
            key=key,
            encrypted_value=encrypted,
            agent_id=agent_id,
            tenant_id=tenant_id,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        self._secrets[agent_id][key] = secret
        return secret

    def retrieve(self, agent_id: str, key: str, requesting_agent: str = "") -> str | None:
        """Holt ein Secret -- nur der eigene Agent darf zugreifen."""
        effective_requester = requesting_agent or agent_id
        self._access_log.append({
            "agent_id": agent_id,
            "key": key,
            "requester": effective_requester,
            "allowed": effective_requester == agent_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })

        if effective_requester != agent_id:
            return None  # BLOCKIERT: Cross-Agent-Zugriff

        agent_secrets = self._secrets.get(agent_id, {})
        secret = agent_secrets.get(key)
        return secret.encrypted_value if secret else None

    def revoke(self, agent_id: str, key: str) -> bool:
        agent_secrets = self._secrets.get(agent_id, {})
        if key in agent_secrets:
            del agent_secrets[key]
            return True
        return False

    def revoke_all(self, agent_id: str) -> int:
        count = len(self._secrets.get(agent_id, {}))
        self._secrets.pop(agent_id, None)
        return count

    def list_keys(self, agent_id: str) -> list[str]:
        return list(self._secrets.get(agent_id, {}).keys())

    def blocked_attempts(self) -> list[dict[str, Any]]:
        return [a for a in self._access_log if not a["allowed"]]

    @property
    def total_secrets(self) -> int:
        return sum(len(s) for s in self._secrets.values())

    def stats(self) -> dict[str, Any]:
        return {
            "agents_with_secrets": len(self._secrets),
            "total_secrets": self.total_secrets,
            "total_access_attempts": len(self._access_log),
            "blocked_attempts": len(self.blocked_attempts()),
        }


# ============================================================================
# Namespace Isolation
# ============================================================================


@dataclass
class Namespace:
    """Isolierter Namensraum für einen Agenten/Tenant."""

    namespace_id: str
    agent_id: str
    tenant_id: str = "default"
    memory_prefix: str = ""
    file_root: str = ""
    config_scope: str = ""
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "namespace_id": self.namespace_id,
            "agent_id": self.agent_id,
            "tenant_id": self.tenant_id,
            "memory_prefix": self.memory_prefix,
            "file_root": self.file_root,
        }


class NamespaceIsolation:
    """Erzwingt getrennte Namensräume für Memory, Files, Config."""

    def __init__(self) -> None:
        self._namespaces: dict[str, Namespace] = {}

    def create(self, agent_id: str, tenant_id: str = "default") -> Namespace:
        ns_id = f"{tenant_id}:{agent_id}"
        ns = Namespace(
            namespace_id=ns_id,
            agent_id=agent_id,
            tenant_id=tenant_id,
            memory_prefix=f"/{tenant_id}/{agent_id}/memory/",
            file_root=f"/data/{tenant_id}/{agent_id}/",
            config_scope=f"config:{tenant_id}:{agent_id}",
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        self._namespaces[ns_id] = ns
        return ns

    def get(self, agent_id: str, tenant_id: str = "default") -> Namespace | None:
        return self._namespaces.get(f"{tenant_id}:{agent_id}")

    def validate_path(self, agent_id: str, path: str, tenant_id: str = "default") -> bool:
        """Prüft ob ein Pfad im erlaubten Namensraum liegt."""
        ns = self.get(agent_id, tenant_id)
        if not ns:
            return False
        return path.startswith(ns.file_root) or path.startswith(ns.memory_prefix)

    def list_namespaces(self, tenant_id: str = "") -> list[Namespace]:
        if tenant_id:
            return [ns for ns in self._namespaces.values() if ns.tenant_id == tenant_id]
        return list(self._namespaces.values())

    @property
    def namespace_count(self) -> int:
        return len(self._namespaces)

    def stats(self) -> dict[str, Any]:
        nss = list(self._namespaces.values())
        tenants = {ns.tenant_id for ns in nss}
        return {
            "total_namespaces": len(nss),
            "total_tenants": len(tenants),
            "tenants": sorted(tenants),
        }


# ============================================================================
# Multi-Tenant Manager
# ============================================================================


class TenantTier(Enum):
    FREE = "free"
    STANDARD = "standard"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


@dataclass
class Tenant:
    """Ein Mandant (Organisation) im System."""

    tenant_id: str
    name: str
    tier: TenantTier = TenantTier.STANDARD
    max_agents: int = 10
    max_users: int = 50
    active_agents: int = 0
    active_users: int = 0
    created_at: str = ""
    admin_emails: list[str] = field(default_factory=list)
    settings: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "name": self.name,
            "tier": self.tier.value,
            "max_agents": self.max_agents,
            "active_agents": self.active_agents,
            "max_users": self.max_users,
            "active_users": self.active_users,
        }


class TenantManager:
    """Multi-Tenant: Getrennte Daten und Konfiguration pro Organisation."""

    def __init__(self) -> None:
        self._tenants: dict[str, Tenant] = {}

    def create(
        self,
        tenant_id: str,
        name: str,
        tier: TenantTier = TenantTier.STANDARD,
        *,
        max_agents: int = 10,
        max_users: int = 50,
        admin_emails: list[str] | None = None,
    ) -> Tenant:
        tenant = Tenant(
            tenant_id=tenant_id,
            name=name,
            tier=tier,
            max_agents=max_agents,
            max_users=max_users,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            admin_emails=admin_emails or [],
        )
        self._tenants[tenant_id] = tenant
        return tenant

    def get(self, tenant_id: str) -> Tenant | None:
        return self._tenants.get(tenant_id)

    def delete(self, tenant_id: str) -> bool:
        if tenant_id in self._tenants:
            del self._tenants[tenant_id]
            return True
        return False

    def can_add_agent(self, tenant_id: str) -> bool:
        tenant = self._tenants.get(tenant_id)
        if not tenant:
            return False
        return tenant.active_agents < tenant.max_agents

    def add_agent(self, tenant_id: str) -> bool:
        tenant = self._tenants.get(tenant_id)
        if tenant and tenant.active_agents < tenant.max_agents:
            tenant.active_agents += 1
            return True
        return False

    def remove_agent(self, tenant_id: str) -> bool:
        tenant = self._tenants.get(tenant_id)
        if tenant and tenant.active_agents > 0:
            tenant.active_agents -= 1
            return True
        return False

    @property
    def tenant_count(self) -> int:
        return len(self._tenants)

    def stats(self) -> dict[str, Any]:
        tenants = list(self._tenants.values())
        return {
            "total_tenants": len(tenants),
            "total_agents": sum(t.active_agents for t in tenants),
            "total_users": sum(t.active_users for t in tenants),
            "by_tier": {
                tier.value: sum(1 for t in tenants if t.tier == tier)
                for tier in TenantTier
                if any(t.tier == tier for t in tenants)
            },
        }


# ============================================================================
# Delegated Admin Roles
# ============================================================================


class AdminRole(Enum):
    SUPER_ADMIN = "super_admin"
    TENANT_ADMIN = "tenant_admin"
    SECURITY_ADMIN = "security_admin"
    READONLY = "readonly"


@dataclass
class DelegatedAdmin:
    """Ein delegierter Admin innerhalb eines Tenants."""

    admin_id: str
    email: str
    tenant_id: str
    role: AdminRole
    permissions: set[str] = field(default_factory=set)

    def can(self, permission: str) -> bool:
        if self.role == AdminRole.SUPER_ADMIN:
            return True
        return permission in self.permissions

    def to_dict(self) -> dict[str, Any]:
        return {
            "admin_id": self.admin_id,
            "email": self.email,
            "tenant_id": self.tenant_id,
            "role": self.role.value,
            "permissions": sorted(self.permissions),
        }


# Standard-Permissions pro Rolle
ROLE_PERMISSIONS: dict[AdminRole, set[str]] = {
    AdminRole.SUPER_ADMIN: {"*"},
    AdminRole.TENANT_ADMIN: {
        "manage_agents", "manage_users", "view_audit", "manage_skills",
        "manage_budgets", "view_security", "manage_config",
    },
    AdminRole.SECURITY_ADMIN: {
        "view_audit", "manage_security", "view_incidents", "manage_gates",
        "run_scans", "view_compliance",
    },
    AdminRole.READONLY: {"view_dashboard", "view_audit", "view_security"},
}


class AdminManager:
    """Verwaltet delegierte Admin-Rollen."""

    def __init__(self) -> None:
        self._admins: dict[str, DelegatedAdmin] = {}

    def create(
        self,
        email: str,
        tenant_id: str,
        role: AdminRole,
    ) -> DelegatedAdmin:
        admin_id = hashlib.sha256(f"admin:{email}:{tenant_id}".encode()).hexdigest()[:10]
        admin = DelegatedAdmin(
            admin_id=admin_id,
            email=email,
            tenant_id=tenant_id,
            role=role,
            permissions=ROLE_PERMISSIONS.get(role, set()),
        )
        self._admins[admin_id] = admin
        return admin

    def get(self, admin_id: str) -> DelegatedAdmin | None:
        return self._admins.get(admin_id)

    def by_tenant(self, tenant_id: str) -> list[DelegatedAdmin]:
        return [a for a in self._admins.values() if a.tenant_id == tenant_id]

    def check_permission(self, admin_id: str, permission: str) -> bool:
        admin = self._admins.get(admin_id)
        if not admin:
            return False
        return admin.can(permission)

    def revoke(self, admin_id: str) -> bool:
        if admin_id in self._admins:
            del self._admins[admin_id]
            return True
        return False

    @property
    def admin_count(self) -> int:
        return len(self._admins)

    def stats(self) -> dict[str, Any]:
        admins = list(self._admins.values())
        return {
            "total_admins": len(admins),
            "by_role": {
                role.value: sum(1 for a in admins if a.role == role)
                for role in AdminRole
                if any(a.role == role for a in admins)
            },
        }


# ============================================================================
# Isolation Enforcer (Hauptklasse)
# ============================================================================


class IsolationEnforcer:
    """Orchestriert alle Isolierungsmechanismen."""

    def __init__(self) -> None:
        self._sandboxes = SandboxManager()
        self._secrets = PerAgentSecretVault()
        self._namespaces = NamespaceIsolation()
        self._tenants = TenantManager()
        self._admins = AdminManager()

    @property
    def sandboxes(self) -> SandboxManager:
        return self._sandboxes

    @property
    def secrets(self) -> PerAgentSecretVault:
        return self._secrets

    @property
    def namespaces(self) -> NamespaceIsolation:
        return self._namespaces

    @property
    def tenants(self) -> TenantManager:
        return self._tenants

    @property
    def admins(self) -> AdminManager:
        return self._admins

    def provision_agent(
        self,
        agent_id: str,
        tenant_id: str = "default",
        *,
        allowed_tools: set[str] | None = None,
    ) -> dict[str, Any]:
        """Provisioniert einen komplett isolierten Agenten."""
        sandbox = self._sandboxes.create(agent_id, tenant_id, allowed_tools=allowed_tools)
        ns = self._namespaces.create(agent_id, tenant_id)
        self._tenants.add_agent(tenant_id)

        return {
            "sandbox_id": sandbox.sandbox_id,
            "namespace": ns.namespace_id,
            "file_root": ns.file_root,
            "memory_prefix": ns.memory_prefix,
        }

    def decommission_agent(self, agent_id: str, tenant_id: str = "default") -> dict[str, Any]:
        """Deprovisioniert einen Agenten sauber."""
        sandbox = self._sandboxes.get_by_agent(agent_id)
        if sandbox:
            self._sandboxes.terminate(sandbox.sandbox_id)
        secrets_revoked = self._secrets.revoke_all(agent_id)
        self._tenants.remove_agent(tenant_id)

        return {
            "sandbox_terminated": sandbox is not None,
            "secrets_revoked": secrets_revoked,
        }

    def stats(self) -> dict[str, Any]:
        return {
            "sandboxes": self._sandboxes.stats(),
            "secrets": self._secrets.stats(),
            "namespaces": self._namespaces.stats(),
            "tenants": self._tenants.stats(),
            "admins": self._admins.stats(),
        }
