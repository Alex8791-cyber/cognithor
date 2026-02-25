"""Orchestrator: Multi-Agent-Management.

Ermöglicht dem Planner, spezialisierte Sub-Agents zu spawnen,
die parallel Aufgaben abarbeiten. Der Orchestrator verwaltet
den Lebenszyklus, überwacht Limits und sammelt Ergebnisse.

Architektur-Prinzipien [B§7]:
  - Max Depth: 3 (konfigurierbar via ResourceQuota.max_depth)
  - Supervisor-Pattern: Planner steuert, Orchestrator überwacht
  - Jeder Sub-Agent hat eigene Tool-Permissions
  - Parallele Ausführung mit asyncio.gather
  - Timeout pro Agent + Global-Timeout
  - Depth-Tracking: Jeder Sub-Agent kennt seine Tiefe

Bibel-Referenz: §7.1 (Orchestrator), §7.2 (Sub-Agent Lifecycle)
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable, Coroutine
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from jarvis.models import (
    AgentHandle,
    AgentResult,
    AgentType,
    SubAgentConfig,
)
from jarvis.security.policies import PolicyEngine, PolicyViolation
from jarvis.utils.logging import get_logger

if TYPE_CHECKING:
    from jarvis.security.audit import AuditTrail

log = get_logger(__name__)

# Type alias für Agent-Runner-Funktion
AgentRunner = Callable[
    [SubAgentConfig, str],
    Coroutine[Any, Any, AgentResult],
]


class Orchestrator:
    """Multi-Agent Orchestrator. [B§7.1]

    Verwaltet Sub-Agent-Spawning, parallele Ausführung und
    Ergebnis-Aggregation. Erzwingt Policy-Limits.

    Der Orchestrator kennt NICHT die LLM-Logik — er delegiert
    die eigentliche Ausführung an eine AgentRunner-Funktion
    (typischerweise ein vereinfachter Gateway-Loop).
    """

    def __init__(
        self,
        *,
        policy_engine: PolicyEngine | None = None,
        audit_trail: AuditTrail | None = None,
        runner: AgentRunner | None = None,
    ) -> None:
        """Initialisiert den Orchestrator.

        Args:
            policy_engine: Policy-Engine für Permission-Checks.
            audit_trail: Audit-Trail für Logging.
            runner: Funktion die einen Sub-Agent ausführt.
        """
        self._policy = policy_engine or PolicyEngine()
        self._audit = audit_trail
        self._runner = runner
        self._agents: dict[str, AgentHandle] = {}
        self._results: dict[str, AgentResult] = {}

    def set_runner(self, runner: AgentRunner) -> None:
        """Setzt die Agent-Runner-Funktion.

        Wird vom Gateway nach Initialisierung aufgerufen.
        """
        self._runner = runner

    async def spawn_agent(
        self,
        config: SubAgentConfig,
        session_id: str,
        parent_type: AgentType = AgentType.PLANNER,
        depth: int = 0,
    ) -> AgentHandle | PolicyViolation:
        """Spawnt einen neuen Sub-Agent.

        Prüft Policy-Limits (inkl. Depth), erstellt Handle und startet
        die Ausführung (noch nicht parallel).

        Args:
            config: Konfiguration des Sub-Agents.
            session_id: Eltern-Session-ID.
            parent_type: Typ des aufrufenden Agents.
            depth: Aktuelle Tiefe des Parents (0 = top-level).

        Returns:
            AgentHandle bei Erfolg, PolicyViolation bei Ablehnung.
        """
        # 1. Policy-Check: Darf spawnen? (inkl. Depth-Check)
        violation = self._policy.check_spawn_allowed(session_id, parent_type, depth)
        if violation:
            log.warning(
                "spawn_rejected",
                session=session_id,
                reason=violation.rule,
                depth=depth,
            )
            if self._audit:
                self._audit.record_event(
                    session_id=session_id,
                    event_type="spawn_rejected",
                    details={
                        "task": config.task,
                        "agent_type": config.agent_type.value,
                        "reason": violation.rule,
                        "depth": depth,
                    },
                )
            return violation

        # Child depth = parent depth + 1
        child_depth = depth + 1

        # 2. Handle erstellen (mit Depth-Info)
        agent_id = f"agent_{uuid.uuid4().hex[:12]}"
        handle = AgentHandle(
            agent_id=agent_id,
            config=config,
            session_id=session_id,
            depth=child_depth,
            status="pending",
        )
        self._agents[agent_id] = handle

        # 3. Quota aktualisieren
        self._policy.record_spawn(session_id)

        # 4. Audit
        if self._audit:
            self._audit.record_event(
                session_id=session_id,
                event_type="agent_spawned",
                details={
                    "agent_id": agent_id,
                    "task": config.task,
                    "agent_type": config.agent_type.value,
                    "model": config.model,
                    "timeout": config.timeout_seconds,
                    "depth": child_depth,
                },
            )

        log.info(
            "agent_spawned",
            agent_id=agent_id,
            agent_type=config.agent_type.value,
            task=config.task[:100],
            depth=child_depth,
        )
        return handle

    async def run_agent(
        self,
        agent_id: str,
        session_id: str,
    ) -> AgentResult:
        """Führt einen gespawnten Agent aus.

        Args:
            agent_id: ID des Agents.
            session_id: Session-ID.

        Returns:
            AgentResult.
        """
        handle = self._agents.get(agent_id)
        if not handle:
            return AgentResult(
                response="",
                success=False,
                error=f"Agent {agent_id} nicht gefunden",
            )

        if not self._runner:
            return AgentResult(
                response="",
                success=False,
                error="Kein Agent-Runner konfiguriert",
            )

        handle.status = "running"
        handle.started_at = datetime.now(UTC)

        try:
            result = await asyncio.wait_for(
                self._runner(handle.config, session_id),
                timeout=handle.config.timeout_seconds,
            )
            handle.status = "completed"
            handle.result = result
        except TimeoutError:
            result = AgentResult(
                response="",
                success=False,
                error=f"Agent-Timeout nach {handle.config.timeout_seconds}s",
            )
            handle.status = "timeout"
            handle.result = result
        except Exception as exc:
            result = AgentResult(
                response="",
                success=False,
                error=str(exc),
            )
            handle.status = "failed"
            handle.result = result

        handle.completed_at = datetime.now(UTC)
        self._results[agent_id] = result

        # Quota: Parallele Agents runter
        self._policy.record_agent_done(session_id)

        # Audit
        if self._audit:
            duration = 0
            if handle.started_at and handle.completed_at:
                duration = int((handle.completed_at - handle.started_at).total_seconds() * 1000)
            self._audit.record_event(
                session_id=session_id,
                event_type="agent_completed",
                details={
                    "agent_id": agent_id,
                    "status": handle.status,
                    "success": result.success,
                    "duration_ms": duration,
                    "iterations": result.total_iterations,
                },
            )

        log.info(
            "agent_completed",
            agent_id=agent_id,
            status=handle.status,
            success=result.success,
        )
        return result

    async def run_parallel(
        self,
        agent_ids: list[str],
        session_id: str,
    ) -> dict[str, AgentResult]:
        """Führt mehrere Agents parallel aus.

        Args:
            agent_ids: Liste der Agent-IDs.
            session_id: Session-ID.

        Returns:
            Dict von Agent-ID → AgentResult.
        """
        if not agent_ids:
            return {}

        log.info(
            "parallel_execution_start",
            count=len(agent_ids),
            agents=agent_ids,
        )

        tasks = [self.run_agent(aid, session_id) for aid in agent_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        output: dict[str, AgentResult] = {}
        for aid, result in zip(agent_ids, results, strict=False):
            if isinstance(result, BaseException):
                output[aid] = AgentResult(
                    response="",
                    success=False,
                    error=str(result),
                )
            else:
                output[aid] = result

        log.info(
            "parallel_execution_complete",
            count=len(output),
            successes=sum(1 for r in output.values() if r.success),
        )
        return output

    async def spawn_and_run(
        self,
        config: SubAgentConfig,
        session_id: str,
        parent_type: AgentType = AgentType.PLANNER,
        depth: int = 0,
    ) -> AgentResult:
        """Convenience: Spawnt und führt einen Agent in einem Schritt aus.

        Args:
            config: Sub-Agent-Konfiguration.
            session_id: Session-ID.
            parent_type: Typ des Parents.
            depth: Aktuelle Tiefe des Parents (0 = top-level).

        Returns:
            AgentResult.
        """
        handle_or_violation = await self.spawn_agent(
            config, session_id, parent_type, depth=depth,
        )
        if isinstance(handle_or_violation, PolicyViolation):
            return AgentResult(
                response="",
                success=False,
                error=f"Policy-Verletzung: {handle_or_violation.details}",
            )

        return await self.run_agent(handle_or_violation.agent_id, session_id)

    async def spawn_and_run_parallel(
        self,
        configs: list[SubAgentConfig],
        session_id: str,
        parent_type: AgentType = AgentType.PLANNER,
        depth: int = 0,
    ) -> list[AgentResult]:
        """Spawnt und führt mehrere Agents parallel aus.

        Args:
            configs: Liste von Sub-Agent-Konfigurationen.
            session_id: Session-ID.
            parent_type: Typ des Parents.
            depth: Aktuelle Tiefe des Parents (0 = top-level).

        Returns:
            Liste von AgentResults (gleiche Reihenfolge wie configs).
        """
        # Alle spawnen
        handles: list[AgentHandle] = []
        results: list[AgentResult] = []

        for config in configs:
            handle_or_violation = await self.spawn_agent(
                config, session_id, parent_type, depth=depth,
            )
            if isinstance(handle_or_violation, PolicyViolation):
                results.append(
                    AgentResult(
                        response="",
                        success=False,
                        error=f"Policy-Verletzung: {handle_or_violation.details}",
                    )
                )
            else:
                handles.append(handle_or_violation)
                results.append(None)  # type: ignore[arg-type]

        # Erfolgreiche parallel ausführen
        if handles:
            agent_ids = [h.agent_id for h in handles]
            parallel_results = await self.run_parallel(agent_ids, session_id)

            # Ergebnisse zusammenführen
            handle_idx = 0
            for i, r in enumerate(results):
                if r is None:
                    aid = handles[handle_idx].agent_id
                    results[i] = parallel_results.get(
                        aid,
                        AgentResult(response="", success=False, error="Unbekannt"),
                    )
                    handle_idx += 1

        return results

    def get_handle(self, agent_id: str) -> AgentHandle | None:
        """Gibt den Handle eines Agents zurück."""
        return self._agents.get(agent_id)

    def get_result(self, agent_id: str) -> AgentResult | None:
        """Gibt das Ergebnis eines Agents zurück."""
        return self._results.get(agent_id)

    def get_all_handles(self) -> list[AgentHandle]:
        """Gibt alle Agent-Handles zurück."""
        return list(self._agents.values())

    def get_active_agents(self) -> list[AgentHandle]:
        """Gibt alle laufenden Agents zurück."""
        return [h for h in self._agents.values() if h.status in ("pending", "running")]

    def cleanup_session(self, session_id: str) -> int:
        """Entfernt alle Agents einer Session.

        Returns:
            Anzahl entfernter Agents.
        """
        to_remove = [
            aid
            for aid, h in self._agents.items()
            if h.status in ("completed", "failed", "timeout")
            and getattr(h, "session_id", None) == session_id
        ]
        for aid in to_remove:
            del self._agents[aid]
            self._results.pop(aid, None)

        self._policy.reset_session(session_id)
        return len(to_remove)

    @property
    def agent_count(self) -> int:
        """Gesamtanzahl verwalteter Agents."""
        return len(self._agents)

    @property
    def active_count(self) -> int:
        """Anzahl aktiver (laufender) Agents."""
        return len(self.get_active_agents())
