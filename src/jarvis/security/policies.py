"""Erweiterte Policy-Engine: Regeln für Sub-Agents und Parallel-Execution.

Erweitert das bestehende Policy-System des Gatekeepers um:
  - Sub-Agent Permission-Restrictions
  - Parallel-Execution-Limits
  - Resource-Quotas pro Session
  - Tool-Allowlists für Agent-Typen

Bibel-Referenz: §11.1 (Sicherheitsarchitektur), §7.4 (Agent-Limits)
"""

from __future__ import annotations

from dataclasses import dataclass

from jarvis.models import (
    AgentType,
    PlannedAction,
    SandboxLevel,
)
from jarvis.utils.logging import get_logger

log = get_logger(__name__)


# ============================================================================
# Agent-Type Permissions [B§7.4]
# ============================================================================


@dataclass(frozen=True)
class AgentPermissions:
    """Berechtigungen für einen Agent-Typ. [B§7.4]"""

    allowed_tools: frozenset[str]
    sandbox_level: SandboxLevel
    max_iterations: int
    can_spawn_sub_agents: bool
    network_access: bool
    max_memory_mb: int
    max_timeout_seconds: int
    sandbox_profile: str = "STANDARD"  # RESTRICTIVE | STANDARD | PERMISSIVE


# Default-Permissions pro Agent-Typ
_DEFAULT_PERMISSIONS: dict[AgentType, AgentPermissions] = {
    AgentType.PLANNER: AgentPermissions(
        allowed_tools=frozenset(
            {
                "read_file",
                "list_directory",
                "search_files",
                "memory_search",
                "memory_save",
                "web_search",
                "web_fetch",
            }
        ),
        sandbox_level=SandboxLevel.PROCESS,
        max_iterations=10,
        can_spawn_sub_agents=True,
        network_access=True,
        max_memory_mb=512,
        max_timeout_seconds=300,
    ),
    AgentType.WORKER: AgentPermissions(
        allowed_tools=frozenset(
            {
                "read_file",
                "write_file",
                "list_directory",
                "run_command",
            }
        ),
        sandbox_level=SandboxLevel.PROCESS,
        max_iterations=5,
        can_spawn_sub_agents=False,
        network_access=False,
        max_memory_mb=256,
        max_timeout_seconds=120,
    ),
    AgentType.CODER: AgentPermissions(
        allowed_tools=frozenset(
            {
                "read_file",
                "write_file",
                "edit_file",
                "list_directory",
                "run_command",
                "search_files",
            }
        ),
        sandbox_level=SandboxLevel.NAMESPACE,
        max_iterations=10,
        can_spawn_sub_agents=False,
        network_access=False,
        max_memory_mb=512,
        max_timeout_seconds=300,
    ),
    AgentType.RESEARCHER: AgentPermissions(
        allowed_tools=frozenset(
            {
                "read_file",
                "list_directory",
                "web_search",
                "web_fetch",
                "search_and_read",
                "memory_search",
                "memory_save",
            }
        ),
        sandbox_level=SandboxLevel.PROCESS,
        max_iterations=8,
        can_spawn_sub_agents=True,  # Depth < max_depth erlaubt Sub-Spawns
        network_access=True,
        max_memory_mb=256,
        max_timeout_seconds=180,
    ),
}


@dataclass
class ResourceQuota:
    """Resource-Quota für eine Session. [B§7.4]"""

    max_sub_agents: int = 4
    max_parallel: int = 3
    max_total_iterations: int = 50
    max_total_tool_calls: int = 100
    max_session_duration_seconds: int = 3600
    max_depth: int = 3  # Maximale Spawn-Tiefe (0=top, 3=deepest allowed)

    # Aktuelle Nutzung
    current_sub_agents: int = 0
    current_parallel: int = 0
    total_iterations: int = 0
    total_tool_calls: int = 0


@dataclass
class PolicyViolation:
    """Eine Policy-Verletzung."""

    rule: str
    details: str
    severity: str = "error"  # error | warning


class PolicyEngine:
    """Erweiterte Policy-Engine mit Agent-Permissions. [B§11.1]

    Kombiniert die bestehenden Gatekeeper-Policies mit neuen
    Regeln für Sub-Agent-Management und Resource-Quotas.
    """

    def __init__(
        self,
        *,
        custom_permissions: dict[AgentType, AgentPermissions] | None = None,
        default_quota: ResourceQuota | None = None,
    ) -> None:
        self._permissions = dict(_DEFAULT_PERMISSIONS)
        if custom_permissions:
            self._permissions.update(custom_permissions)
        self._default_quota = default_quota or ResourceQuota()
        self._session_quotas: dict[str, ResourceQuota] = {}

    def get_permissions(self, agent_type: AgentType) -> AgentPermissions:
        """Gibt die Permissions für einen Agent-Typ zurück."""
        return self._permissions.get(
            agent_type,
            self._permissions[AgentType.WORKER],
        )

    def check_tool_allowed(self, agent_type: AgentType, tool_name: str) -> PolicyViolation | None:
        """Prüft ob ein Tool für einen Agent-Typ erlaubt ist.

        Args:
            agent_type: Typ des Agents.
            tool_name: Name des Tools.

        Returns:
            PolicyViolation wenn nicht erlaubt, sonst None.
        """
        perms = self.get_permissions(agent_type)
        if tool_name not in perms.allowed_tools:
            return PolicyViolation(
                rule="tool_not_allowed",
                details=(
                    f"Agent-Typ '{agent_type.value}' darf Tool "
                    f"'{tool_name}' nicht verwenden. "
                    f"Erlaubt: {sorted(perms.allowed_tools)}"
                ),
            )
        return None

    def check_spawn_allowed(
        self,
        session_id: str,
        parent_type: AgentType,
        current_depth: int = 0,
    ) -> PolicyViolation | None:
        """Prüft ob ein Agent Sub-Agents spawnen darf.

        Args:
            session_id: Session-ID.
            parent_type: Typ des Parent-Agents.
            current_depth: Aktuelle Tiefe des Parent-Agents (0 = top-level).

        Returns:
            PolicyViolation wenn nicht erlaubt.
        """
        perms = self.get_permissions(parent_type)
        if not perms.can_spawn_sub_agents:
            return PolicyViolation(
                rule="spawn_not_allowed",
                details=(f"Agent-Typ '{parent_type.value}' darf keine Sub-Agents spawnen."),
            )

        quota = self._get_quota(session_id)

        # Depth-Check: Spawn nur erlaubt wenn current_depth < max_depth
        if current_depth >= quota.max_depth:
            return PolicyViolation(
                rule="max_depth_exceeded",
                details=(
                    f"Max. Spawn-Tiefe erreicht: {current_depth}/{quota.max_depth}"
                ),
            )

        if quota.current_sub_agents >= quota.max_sub_agents:
            return PolicyViolation(
                rule="max_sub_agents_exceeded",
                details=(
                    f"Max. Sub-Agents erreicht: {quota.current_sub_agents}/{quota.max_sub_agents}"
                ),
            )

        if quota.current_parallel >= quota.max_parallel:
            return PolicyViolation(
                rule="max_parallel_exceeded",
                details=(
                    f"Max. parallele Agents erreicht: {quota.current_parallel}/{quota.max_parallel}"
                ),
            )

        return None

    def check_depth_allowed(self, session_id: str, current_depth: int) -> PolicyViolation | None:
        """Prüft ob die aktuelle Tiefe noch Spawns erlaubt.

        Args:
            session_id: Session-ID.
            current_depth: Aktuelle Tiefe.

        Returns:
            PolicyViolation wenn max_depth erreicht.
        """
        quota = self._get_quota(session_id)
        if current_depth >= quota.max_depth:
            return PolicyViolation(
                rule="max_depth_exceeded",
                details=(
                    f"Max. Spawn-Tiefe erreicht: {current_depth}/{quota.max_depth}"
                ),
            )
        return None

    def check_iteration_limit(self, session_id: str) -> PolicyViolation | None:
        """Prüft ob die Gesamt-Iterationen das Limit erreicht haben."""
        quota = self._get_quota(session_id)
        if quota.total_iterations >= quota.max_total_iterations:
            return PolicyViolation(
                rule="max_iterations_exceeded",
                details=(
                    f"Max. Gesamt-Iterationen: "
                    f"{quota.total_iterations}/{quota.max_total_iterations}"
                ),
            )
        return None

    def check_tool_call_limit(self, session_id: str) -> PolicyViolation | None:
        """Prüft ob die Tool-Calls das Limit erreicht haben."""
        quota = self._get_quota(session_id)
        if quota.total_tool_calls >= quota.max_total_tool_calls:
            return PolicyViolation(
                rule="max_tool_calls_exceeded",
                details=(f"Max. Tool-Calls: {quota.total_tool_calls}/{quota.max_total_tool_calls}"),
            )
        return None

    def record_spawn(self, session_id: str) -> None:
        """Registriert einen Sub-Agent-Spawn."""
        quota = self._get_quota(session_id)
        quota.current_sub_agents += 1
        quota.current_parallel += 1

    def record_agent_done(self, session_id: str) -> None:
        """Registriert Beendigung eines Sub-Agents."""
        quota = self._get_quota(session_id)
        quota.current_parallel = max(0, quota.current_parallel - 1)

    def record_iteration(self, session_id: str) -> None:
        """Registriert eine Iteration."""
        quota = self._get_quota(session_id)
        quota.total_iterations += 1

    def record_tool_call(self, session_id: str) -> None:
        """Registriert einen Tool-Call."""
        quota = self._get_quota(session_id)
        quota.total_tool_calls += 1

    def get_quota(self, session_id: str) -> ResourceQuota:
        """Gibt die aktuelle Quota für eine Session zurück (public)."""
        return self._get_quota(session_id)

    def reset_session(self, session_id: str) -> None:
        """Setzt die Quota einer Session zurück."""
        if session_id in self._session_quotas:
            del self._session_quotas[session_id]

    def validate_action_for_agent(
        self,
        agent_type: AgentType,
        action: PlannedAction,
        session_id: str,
    ) -> list[PolicyViolation]:
        """Vollständige Policy-Prüfung für eine Agent-Aktion.

        Args:
            agent_type: Typ des ausführenden Agents.
            action: Die geplante Aktion.
            session_id: Session-ID.

        Returns:
            Liste von PolicyViolations (leer = OK).
        """
        violations: list[PolicyViolation] = []

        # 1. Tool erlaubt?
        tool_check = self.check_tool_allowed(agent_type, action.tool)
        if tool_check:
            violations.append(tool_check)

        # 2. Iterations-Limit?
        iter_check = self.check_iteration_limit(session_id)
        if iter_check:
            violations.append(iter_check)

        # 3. Tool-Call-Limit?
        call_check = self.check_tool_call_limit(session_id)
        if call_check:
            violations.append(call_check)

        if violations:
            log.warning(
                "policy_violations",
                agent_type=agent_type.value,
                tool=action.tool,
                violations=[v.rule for v in violations],
            )

        return violations

    def _get_quota(self, session_id: str) -> ResourceQuota:
        """Gibt oder erstellt eine Session-Quota."""
        if session_id not in self._session_quotas:
            self._session_quotas[session_id] = ResourceQuota(
                max_sub_agents=self._default_quota.max_sub_agents,
                max_parallel=self._default_quota.max_parallel,
                max_total_iterations=self._default_quota.max_total_iterations,
                max_total_tool_calls=self._default_quota.max_total_tool_calls,
                max_session_duration_seconds=self._default_quota.max_session_duration_seconds,
                max_depth=self._default_quota.max_depth,
            )
        return self._session_quotas[session_id]
