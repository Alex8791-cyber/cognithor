"""Agent-Isolation: Vollständige Trennung von Workspaces, Sessions und Credentials.

Schließt die Lücke zur OpenClaw-Architektur:
  - WorkspaceGuard: Filesystem-Isolation pro Agent (kein Zugriff auf fremde Verzeichnisse)
  - AgentResourceQuota: Token-Budgets, Memory-Limits, Rate-Limits pro Agent
  - UserAgentScope: Multi-User-Isolation (User → Agent → eigene Session/Creds/Workspace)

Bibel-Referenz: §11 (Agent-Separation), §7.4 (Resource-Quotas)
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jarvis.utils.logging import get_logger

log = get_logger(__name__)


# ============================================================================
# WorkspaceGuard: Filesystem-Isolation pro Agent
# ============================================================================


@dataclass
class WorkspacePolicy:
    """Policy für einen Agent-Workspace."""

    agent_id: str
    base_dir: Path  # ~/jarvis/workspace/<agent_id>/
    allow_read_shared: bool = False  # Darf Shared-Workspace lesen
    allow_write_shared: bool = False  # Darf Shared-Workspace schreiben
    allowed_external_paths: list[str] = field(default_factory=list)  # Whitelist
    max_total_size_mb: int = 500
    max_files: int = 1000


class WorkspaceGuard:
    """Erzwingt Filesystem-Isolation pro Agent.

    Jeder Agent hat ein eigenes Workspace-Verzeichnis.
    Zugriffe außerhalb werden blockiert, es sei denn
    explizit in der Policy erlaubt.
    """

    MAX_VIOLATIONS = 1000

    def __init__(self, workspace_root: Path) -> None:
        self._root = workspace_root
        self._policies: dict[str, WorkspacePolicy] = {}
        self._shared_dir = workspace_root / "_shared"
        self._violations: list[dict[str, Any]] = []

    def register_agent(
        self,
        agent_id: str,
        *,
        allow_read_shared: bool = False,
        allow_write_shared: bool = False,
        allowed_external_paths: list[str] | None = None,
        max_total_size_mb: int = 500,
        max_files: int = 1000,
    ) -> WorkspacePolicy:
        """Registriert einen Agent mit Workspace-Policy."""
        agent_dir = self._root / agent_id
        agent_dir.mkdir(parents=True, exist_ok=True)

        policy = WorkspacePolicy(
            agent_id=agent_id,
            base_dir=agent_dir,
            allow_read_shared=allow_read_shared,
            allow_write_shared=allow_write_shared,
            allowed_external_paths=allowed_external_paths or [],
            max_total_size_mb=max_total_size_mb,
            max_files=max_files,
        )
        self._policies[agent_id] = policy
        log.info("workspace_registered", agent_id=agent_id, path=str(agent_dir))
        return policy

    def get_workspace(self, agent_id: str) -> Path | None:
        """Gibt das Workspace-Verzeichnis eines Agents zurück."""
        policy = self._policies.get(agent_id)
        return policy.base_dir if policy else None

    def check_access(
        self,
        agent_id: str,
        target_path: Path,
        mode: str = "read",
    ) -> bool:
        """Prüft ob ein Agent auf einen Pfad zugreifen darf.

        Args:
            agent_id: Agent der zugreifen möchte.
            target_path: Zielpfad.
            mode: "read" oder "write".

        Returns:
            True wenn erlaubt, False wenn blockiert.
        """
        policy = self._policies.get(agent_id)
        if not policy:
            self._record_violation(agent_id, target_path, mode, "agent_not_registered")
            return False

        resolved = target_path.resolve()
        agent_dir = policy.base_dir.resolve()

        # 1. Eigenes Workspace → immer erlaubt
        if self._is_subpath(resolved, agent_dir):
            return True

        # 2. Shared-Workspace
        shared_dir = self._shared_dir.resolve()
        if self._is_subpath(resolved, shared_dir):
            if mode == "read" and policy.allow_read_shared:
                return True
            if mode == "write" and policy.allow_write_shared:
                return True
            self._record_violation(agent_id, target_path, mode, "shared_access_denied")
            return False

        # 3. Whitelist
        for allowed in policy.allowed_external_paths:
            if self._is_subpath(resolved, Path(allowed).resolve()):
                return True

        # 4. Fremdes Agent-Workspace → IMMER blockiert
        for other_id, other_policy in self._policies.items():
            if other_id != agent_id and self._is_subpath(resolved, other_policy.base_dir.resolve()):
                self._record_violation(
                    agent_id, target_path, mode,
                    f"cross_agent_access_blocked:{other_id}",
                )
                return False

        # 5. Sonstiger Pfad außerhalb Root
        self._record_violation(agent_id, target_path, mode, "outside_workspace")
        return False

    def resolve_path(self, agent_id: str, relative_path: str) -> Path | None:
        """Löst einen relativen Pfad im Agent-Workspace auf.

        Verhindert Path-Traversal (../).
        """
        policy = self._policies.get(agent_id)
        if not policy:
            return None

        # Normalize und prüfe auf Traversal
        candidate = (policy.base_dir / relative_path).resolve()
        if not self._is_subpath(candidate, policy.base_dir.resolve()):
            self._record_violation(agent_id, candidate, "resolve", "path_traversal_attempt")
            return None

        return candidate

    def violations(self, agent_id: str = "") -> list[dict[str, Any]]:
        """Gibt Violations zurück, optional gefiltert."""
        if agent_id:
            return [v for v in self._violations if v["agent_id"] == agent_id]
        return list(self._violations)

    @property
    def registered_agents(self) -> list[str]:
        return list(self._policies.keys())

    def _is_subpath(self, child: Path, parent: Path) -> bool:
        """Prüft ob child ein Unterpfad von parent ist."""
        try:
            child.relative_to(parent)
            return True
        except ValueError:
            return False

    def _record_violation(
        self,
        agent_id: str,
        path: Path,
        mode: str,
        reason: str,
    ) -> None:
        self._violations.append({
            "agent_id": agent_id,
            "path": str(path),
            "mode": mode,
            "reason": reason,
            "timestamp": time.time(),
        })
        # Evict oldest if over limit
        if len(self._violations) > self.MAX_VIOLATIONS:
            self._violations = self._violations[-self.MAX_VIOLATIONS:]
        log.warning(
            "workspace_violation",
            agent_id=agent_id,
            path=str(path),
            mode=mode,
            reason=reason,
        )


# ============================================================================
# AgentResourceQuota: Per-Agent Token-Budgets und Limits
# ============================================================================


@dataclass
class AgentResourceQuota:
    """Konfigurierbare Resource-Limits pro Agent.

    Ergänzt die bestehende session-basierte ResourceQuota
    um agent-weite Budgets die über Sessions hinweg gelten.
    """

    agent_id: str

    # Token-Limits (pro Tag)
    daily_token_budget: int = 100_000  # Max Tokens pro Tag
    tokens_used_today: int = 0

    # API-Rate-Limits
    max_requests_per_minute: int = 30
    max_concurrent_requests: int = 5

    # Session-Limits
    max_active_sessions: int = 10
    max_session_duration_seconds: int = 3600

    # Tool-Limits
    max_tool_calls_per_session: int = 100
    blocked_tools: list[str] = field(default_factory=list)

    # Memory
    max_memory_entries: int = 500
    max_workspace_mb: int = 500

    def check_token_budget(self, tokens_requested: int) -> bool:
        """Prüft ob Token-Budget ausreicht."""
        return self.tokens_used_today + tokens_requested <= self.daily_token_budget

    def consume_tokens(self, tokens: int) -> bool:
        """Verbraucht Tokens vom Budget. Gibt False zurück wenn erschöpft."""
        if not self.check_token_budget(tokens):
            return False
        self.tokens_used_today += tokens
        return True

    def reset_daily(self) -> None:
        """Setzt tägliche Zähler zurück."""
        self.tokens_used_today = 0

    @property
    def tokens_remaining(self) -> int:
        return max(0, self.daily_token_budget - self.tokens_used_today)

    @property
    def budget_utilization_percent(self) -> float:
        if self.daily_token_budget == 0:
            return 100.0
        return round(self.tokens_used_today / self.daily_token_budget * 100, 1)

    def is_tool_allowed(self, tool_name: str) -> bool:
        """Prüft ob ein Tool für diesen Agent erlaubt ist."""
        return tool_name not in self.blocked_tools

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "daily_token_budget": self.daily_token_budget,
            "tokens_used_today": self.tokens_used_today,
            "tokens_remaining": self.tokens_remaining,
            "budget_utilization_percent": self.budget_utilization_percent,
            "max_requests_per_minute": self.max_requests_per_minute,
            "max_concurrent_requests": self.max_concurrent_requests,
            "max_active_sessions": self.max_active_sessions,
            "blocked_tools": self.blocked_tools,
        }


# ============================================================================
# Rate-Limiter
# ============================================================================


class RateLimiter:
    """Token-Bucket Rate-Limiter pro Agent."""

    def __init__(self) -> None:
        self._windows: dict[str, list[float]] = defaultdict(list)

    def check_and_consume(self, agent_id: str, max_per_minute: int) -> bool:
        """Prüft ob ein Request erlaubt ist (Sliding Window)."""
        now = time.time()
        window = self._windows[agent_id]

        # Alte Einträge entfernen (älter als 60s)
        cutoff = now - 60.0
        self._windows[agent_id] = [t for t in window if t > cutoff]

        if len(self._windows[agent_id]) >= max_per_minute:
            return False

        self._windows[agent_id].append(now)
        return True

    def current_rate(self, agent_id: str) -> int:
        """Aktuelle Request-Rate (letzte Minute)."""
        now = time.time()
        cutoff = now - 60.0
        return len([t for t in self._windows.get(agent_id, []) if t > cutoff])


# ============================================================================
# UserAgentScope: Multi-User-Isolation
# ============================================================================


@dataclass
class UserAgentScope:
    """Isolierter Scope pro User+Agent Kombination.

    Stellt sicher dass:
      - User A's Sessions von User B's Sessions getrennt sind
      - Agent X's Credentials nur für Agent X sichtbar sind
      - Jeder User+Agent eigene Working Memory hat
      - Workspace-Isolation enforced wird
    """

    user_id: str
    agent_id: str

    # Scope-Key für eindeutige Zuordnung
    @property
    def scope_key(self) -> str:
        return f"{self.user_id}:{self.agent_id}"

    # Session-Isolation
    session_ids: list[str] = field(default_factory=list)

    # Credential-Isolation
    credential_namespace: str = ""

    @property
    def effective_credential_namespace(self) -> str:
        """Namespace für Credentials: user:agent."""
        return self.credential_namespace or self.scope_key


class MultiUserIsolation:
    """Erzwingt vollständige Isolation zwischen Usern und Agents.

    Implementiert die OpenClaw-äquivalente Trennung:
    - Jeder User hat eigene Scopes pro Agent
    - Kein Cross-User Zugriff auf Sessions/Credentials/Memory
    - Agent-übergreifende Operationen nur mit expliziter Delegation
    """

    def __init__(self) -> None:
        self._scopes: dict[str, UserAgentScope] = {}
        self._quotas: dict[str, AgentResourceQuota] = {}
        self._rate_limiter = RateLimiter()

    def get_or_create_scope(
        self,
        user_id: str,
        agent_id: str,
    ) -> UserAgentScope:
        """Holt oder erstellt einen User+Agent Scope."""
        key = f"{user_id}:{agent_id}"
        if key not in self._scopes:
            self._scopes[key] = UserAgentScope(
                user_id=user_id,
                agent_id=agent_id,
            )
        return self._scopes[key]

    def validate_session_access(
        self,
        user_id: str,
        agent_id: str,
        session_id: str,
    ) -> bool:
        """Prüft ob ein User auf eine Session zugreifen darf."""
        scope = self._scopes.get(f"{user_id}:{agent_id}")
        if not scope:
            return False
        return session_id in scope.session_ids

    def register_session(
        self,
        user_id: str,
        agent_id: str,
        session_id: str,
    ) -> UserAgentScope:
        """Registriert eine neue Session für einen User+Agent Scope."""
        scope = self.get_or_create_scope(user_id, agent_id)
        if session_id not in scope.session_ids:
            scope.session_ids.append(session_id)
        return scope

    def revoke_session(
        self,
        user_id: str,
        agent_id: str,
        session_id: str,
    ) -> bool:
        """Entfernt eine Session aus einem Scope."""
        scope = self._scopes.get(f"{user_id}:{agent_id}")
        if scope and session_id in scope.session_ids:
            scope.session_ids.remove(session_id)
            return True
        return False

    def get_credential_namespace(
        self,
        user_id: str,
        agent_id: str,
    ) -> str:
        """Gibt den isolierten Credential-Namespace zurück."""
        scope = self.get_or_create_scope(user_id, agent_id)
        return scope.effective_credential_namespace

    def can_delegate(
        self,
        from_user: str,
        from_agent: str,
        to_agent: str,
    ) -> bool:
        """Prüft ob eine Agent-zu-Agent Delegation erlaubt ist.

        Delegation innerhalb eines Users ist erlaubt.
        Cross-User Delegation ist blockiert.
        """
        source = self._scopes.get(f"{from_user}:{from_agent}")
        return source is not None

    def user_scopes(self, user_id: str) -> list[UserAgentScope]:
        """Alle Scopes eines Users."""
        return [s for s in self._scopes.values() if s.user_id == user_id]

    def agent_scopes(self, agent_id: str) -> list[UserAgentScope]:
        """Alle Scopes eines Agents (über alle User)."""
        return [s for s in self._scopes.values() if s.agent_id == agent_id]

    # --- Resource-Quota Management ---

    def set_quota(self, agent_id: str, quota: AgentResourceQuota) -> None:
        """Setzt ein Resource-Quota für einen Agent."""
        self._quotas[agent_id] = quota

    def get_quota(self, agent_id: str) -> AgentResourceQuota | None:
        return self._quotas.get(agent_id)

    def check_rate_limit(self, agent_id: str) -> bool:
        """Prüft Rate-Limit für einen Agent."""
        quota = self._quotas.get(agent_id)
        if not quota:
            return True  # Kein Quota → erlaubt
        return self._rate_limiter.check_and_consume(
            agent_id, quota.max_requests_per_minute,
        )

    def consume_tokens(self, agent_id: str, tokens: int) -> bool:
        """Verbraucht Tokens vom Agent-Budget."""
        quota = self._quotas.get(agent_id)
        if not quota:
            return True  # Kein Quota → erlaubt
        return quota.consume_tokens(tokens)

    # --- Statistiken ---

    def stats(self) -> dict[str, Any]:
        return {
            "total_scopes": len(self._scopes),
            "unique_users": len(set(s.user_id for s in self._scopes.values())),
            "unique_agents": len(set(s.agent_id for s in self._scopes.values())),
            "active_sessions": sum(len(s.session_ids) for s in self._scopes.values()),
            "quotas_configured": len(self._quotas),
        }
