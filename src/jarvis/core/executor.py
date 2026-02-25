"""Executor: Executes approved actions in a sandbox.

The Executor:
  - Executes ONLY actions approved by the Gatekeeper
  - Dispatches tool calls to the correct MCP server
  - Enforces timeouts and resource limits
  - Respects dependencies between steps
  - Catches errors and reports them in a structured way

Bible reference: §3.3 (Executor)
"""

from __future__ import annotations

import asyncio
import contextvars
import time
from typing import TYPE_CHECKING, Any

from jarvis.models import (
    GateDecision,
    GateStatus,
    PlannedAction,
    ToolResult,
)
from jarvis.utils.logging import get_logger

if TYPE_CHECKING:
    from jarvis.audit import AuditLogger
    from jarvis.config import JarvisConfig
    from jarvis.mcp.client import JarvisMCPClient
    from jarvis.security.monitor import RuntimeMonitor
    from jarvis.skills.generator import GapDetector

# Thread-/Task-safe agent context via contextvars
_agent_workspace_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("agent_workspace", default=None)
_agent_sandbox_var: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar("agent_sandbox", default=None)
_agent_name_var: contextvars.ContextVar[str] = contextvars.ContextVar("agent_name", default="")
_session_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("session_id", default="")

log = get_logger(__name__)


class ExecutionError(Exception):
    """Error during tool execution."""


class Executor:
    """Sandboxed Executor -- only executes approved actions. [B§3.3]

    The Executor has NO decision logic of its own.
    It executes exactly what the Gatekeeper has approved.

    Agent context:
      The Executor optionally receives an AgentContext containing
      workspace_dir and sandbox_overrides. These are automatically
      injected into tool params (exec_command -> working_dir,
      Sandbox -> network/limits).
    """

    # Retryable Error-Typen (transiente Fehler, die sich lohnen zu wiederholen)
    RETRYABLE_ERRORS = frozenset(
        {
            "TimeoutError",
            "ConnectionError",
            "ConnectError",
            "ReadTimeout",
            "OllamaError",
            "HTTPStatusError",
        }
    )

    # Tools die working_dir akzeptieren und Agent-Workspace nutzen sollen
    WORKSPACE_TOOLS = frozenset(
        {
            "exec_command",
            "write_file",
            "read_file",
            "edit_file",
            "list_directory",
        }
    )

    def __init__(
        self,
        config: JarvisConfig,
        mcp_client: JarvisMCPClient | None = None,
        gap_detector: GapDetector | None = None,
        runtime_monitor: RuntimeMonitor | None = None,
        audit_logger: AuditLogger | None = None,
        task_profiler: Any = None,
        task_telemetry: Any = None,
        error_clusterer: Any = None,
    ) -> None:
        """Initialisiert den Executor mit Konfiguration und MCP-Client."""
        self._config = config
        self._mcp_client = mcp_client
        self._gap_detector = gap_detector
        self._runtime_monitor = runtime_monitor
        self._audit_logger = audit_logger
        self._task_profiler = task_profiler
        self._task_telemetry = task_telemetry
        self._error_clusterer = error_clusterer
        self._default_timeout = 30  # Sekunden
        self._max_retries = 3
        self._base_delay = 1.0  # Sekunden (exponentiell: 1s, 2s, 4s)
        # Agent context tokens (for contextvar reset)
        self._ctx_tokens: list[contextvars.Token] = []

    def set_mcp_client(self, client: JarvisMCPClient) -> None:
        """Setzt den MCP-Client (kann nach Initialisierung gesetzt werden)."""
        self._mcp_client = client

    def set_agent_context(
        self,
        workspace_dir: str | None = None,
        sandbox_overrides: dict[str, Any] | None = None,
        agent_name: str = "",
        session_id: str = "",
    ) -> None:
        """Setzt den Agent-Kontext für die nächste Ausführung.

        Wird vom Gateway pro Request gesetzt, basierend auf dem
        gerouteten Agenten.

        Args:
            workspace_dir: Agent-spezifisches Workspace-Verzeichnis.
            sandbox_overrides: Sandbox-Config des Agenten
                (network, max_memory_mb, timeout, etc.)
            agent_name: Name des aktiven Agenten (für Audit/Monitor).
            session_id: Session-ID für Profiling/Telemetry.
        """
        # Alte Tokens zurücksetzen, bevor neue gesetzt werden
        self.clear_agent_context()
        self._ctx_tokens = [
            _agent_workspace_var.set(workspace_dir),
            _agent_sandbox_var.set(sandbox_overrides),
            _agent_name_var.set(agent_name),
            _session_id_var.set(session_id),
        ]

    def clear_agent_context(self) -> None:
        """Löscht den Agent-Kontext nach der Ausführung."""
        for token in self._ctx_tokens:
            try:
                token.var.reset(token)
            except ValueError:
                pass
        self._ctx_tokens = []

    async def execute(
        self,
        actions: list[PlannedAction],
        decisions: list[GateDecision],
    ) -> list[ToolResult]:
        """Führt alle genehmigten Aktionen sequentiell aus.

        Respektiert Dependencies (depends_on) und Gatekeeper-Entscheidungen.
        Nur ALLOW und INFORM Aktionen werden ausgeführt.
        APPROVE Aktionen müssen vorher vom User bestätigt worden sein.
        BLOCK und ungenehmigte Aktionen werden übersprungen.

        Args:
            actions: Liste der geplanten Aktionen
            decisions: Korrespondierende Gatekeeper-Entscheidungen

        Returns:
            Liste von ToolResults (ein Ergebnis pro Aktion).
        """
        if len(actions) != len(decisions):
            raise ExecutionError(
                f"Anzahl Aktionen ({len(actions)}) ≠ Entscheidungen ({len(decisions)})"
            )

        results: list[ToolResult] = []
        completed_indices: set[int] = set()

        for i, (action, decision) in enumerate(zip(actions, decisions, strict=False)):
            # --- Nur erlaubte Aktionen ausführen ---
            if decision.status not in (GateStatus.ALLOW, GateStatus.INFORM, GateStatus.MASK):
                # Audit: Gatekeeper-Blockierung protokollieren
                if self._audit_logger:
                    self._audit_logger.log_gatekeeper(
                        decision.status.value,
                        decision.reason,
                        tool_name=action.tool,
                        agent_name=_agent_name_var.get(),
                    )
                results.append(
                    ToolResult(
                        tool_name=action.tool,
                        content=f"Aktion übersprungen: {decision.status.value} -- {decision.reason}",
                        is_error=True,
                        error_type="GatekeeperBlock",
                    )
                )
                continue

            # --- Dependencies prüfen ---
            unmet_deps = [dep for dep in action.depends_on if dep not in completed_indices]
            if unmet_deps:
                results.append(
                    ToolResult(
                        tool_name=action.tool,
                        content=(
                            f"Aktion übersprungen: Abhängigkeiten nicht erfüllt ({unmet_deps})"
                        ),
                        is_error=True,
                        error_type="DependencyError",
                    )
                )
                continue

            # --- Aktion ausführen ---
            # Bei MASK: maskierte Params verwenden (Kopie um frozen Model nicht zu mutieren)
            if decision.status == GateStatus.MASK and decision.masked_params:
                params = dict(decision.masked_params)
            else:
                params = dict(action.params)

            result = await self._execute_single(action.tool, params)
            results.append(result)

            if result.success:
                completed_indices.add(i)

            # Logging
            log.info(
                "executor_tool_result",
                tool=action.tool,
                success=result.success,
                duration_ms=result.duration_ms,
                content_length=len(result.content),
            )

        return results

    async def _execute_single(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> ToolResult:
        """Führt ein einzelnes Tool aus mit Retry, Backoff und Timeout.

        Agent-Kontext-Injection:
          - Für WORKSPACE_TOOLS: Injiziert working_dir aus Agent-Workspace
          - Für exec_command: Injiziert Sandbox-Overrides (_sandbox_network, _sandbox_timeout)

        Retry-Strategie:
          - Maximal 3 Versuche (konfigurierbar)
          - Exponentieller Backoff: 1s → 2s → 4s
          - Nur bei transienten Fehlern (Timeout, Connection, Ollama)
          - Kein Retry bei logischen Fehlern (Permission, NotFound)

        Args:
            tool_name: Name des MCP-Tools
            params: Tool-Parameter

        Returns:
            ToolResult mit Erfolg oder Fehler.
        """
        if self._mcp_client is None:
            return ToolResult(
                tool_name=tool_name,
                content="Kein MCP-Client verfügbar",
                is_error=True,
                error_type="NoMCPClient",
            )

        timeout = params.pop("_timeout", self._default_timeout)

        # --- Agent-Kontext in Tool-Params injizieren ---
        if _agent_workspace_var.get() and tool_name in self.WORKSPACE_TOOLS:
            # Nur setzen wenn nicht explizit vom Planner vorgegeben
            if "working_dir" not in params:
                params["working_dir"] = _agent_workspace_var.get()

        if _agent_sandbox_var.get() and tool_name == "exec_command":
            # Sandbox-Overrides als interne Params durchreichen
            overrides = _agent_sandbox_var.get()
            if "_sandbox_network" not in params and "network" in overrides:
                params["_sandbox_network"] = overrides["network"]
            if "_sandbox_max_memory_mb" not in params and "max_memory_mb" in overrides:
                params["_sandbox_max_memory_mb"] = overrides["max_memory_mb"]
            if "_sandbox_max_processes" not in params and "max_processes" in overrides:
                params["_sandbox_max_processes"] = overrides["max_processes"]
            if "_sandbox_timeout" not in params and "timeout" in overrides:
                timeout = overrides["timeout"]

        last_error: str = ""
        last_error_type: str = ""
        total_start = time.monotonic()

        # --- Runtime Monitor: Sicherheitscheck VOR Ausführung ---
        if self._runtime_monitor:
            security_event = self._runtime_monitor.check_tool_call(
                tool_name, params, agent_name=_agent_name_var.get(),
            )
            if security_event.is_blocked:
                if self._audit_logger:
                    self._audit_logger.log_security(
                        security_event.description,
                        tool_name=tool_name,
                        agent_name=_agent_name_var.get(),
                        blocked=True,
                    )
                return ToolResult(
                    tool_name=tool_name,
                    content=f"Sicherheitscheck blockiert: {security_event.description}",
                    is_error=True,
                    error_type="SecurityBlock",
                )

        for attempt in range(1, self._max_retries + 1):
            start = time.monotonic()

            try:
                # MCP-Tool-Call mit Timeout
                result = await asyncio.wait_for(
                    self._mcp_client.call_tool(tool_name, params),
                    timeout=float(timeout),
                )

                duration_ms = int((time.monotonic() - start) * 1000)

                # Output kürzen wenn zu lang
                content = result.content if hasattr(result, "content") else str(result)
                truncated = False
                max_output = 10000  # 10k Zeichen maximal
                if len(content) > max_output:
                    content = content[:max_output]
                    truncated = True

                is_error = result.is_error if hasattr(result, "is_error") else False

                if attempt > 1:
                    log.info(
                        "executor_retry_success",
                        tool=tool_name,
                        attempt=attempt,
                        duration_ms=duration_ms,
                    )

                tool_result = ToolResult(
                    tool_name=tool_name,
                    content=content,
                    is_error=is_error,
                    duration_ms=duration_ms,
                    truncated=truncated,
                )
                # Profiler: Tool-Call aufzeichnen
                if self._task_profiler:
                    try:
                        self._task_profiler.record_tool_call(
                            tool_name=tool_name,
                            latency_ms=float(duration_ms),
                            success=not is_error,
                            session_id=_session_id_var.get(),
                        )
                    except Exception:
                        pass
                # Audit: Erfolgreiche Tool-Ausführung protokollieren
                if self._audit_logger:
                    self._audit_logger.log_tool_call(
                        tool_name,
                        params,
                        agent_name=_agent_name_var.get(),
                        result=content[:200] if not is_error else f"ERROR: {content[:200]}",
                        success=not is_error,
                        duration_ms=float(duration_ms),
                    )
                return tool_result

            except TimeoutError:
                last_error = f"Timeout nach {timeout} Sekunden"
                last_error_type = "TimeoutError"

            except Exception as exc:
                last_error_type = type(exc).__name__
                last_error = str(exc)[:500]

            duration_ms = int((time.monotonic() - start) * 1000)

            # Profiler: fehlgeschlagenen Tool-Call aufzeichnen
            if self._task_profiler:
                try:
                    self._task_profiler.record_tool_call(
                        tool_name=tool_name,
                        latency_ms=float(duration_ms),
                        success=False,
                        error_type=last_error_type,
                        session_id=_session_id_var.get(),
                    )
                except Exception:
                    pass

            # Error-Clustering
            if self._error_clusterer:
                try:
                    self._error_clusterer.add_error(
                        last_error_type, last_error, f"tool={tool_name}",
                    )
                except Exception:
                    pass

            # Retry-Entscheidung
            if last_error_type not in self.RETRYABLE_ERRORS:
                log.error(
                    "executor_error_no_retry",
                    tool=tool_name,
                    error_type=last_error_type,
                    error=last_error,
                    duration_ms=duration_ms,
                )
                # Gap melden für Auto-Skill-Generator
                if self._gap_detector:
                    self._gap_detector.report_unknown_tool(
                        tool_name, context=f"{last_error_type}: {last_error[:200]}",
                    )
                # Audit: Fehler protokollieren
                if self._audit_logger:
                    self._audit_logger.log_tool_call(
                        tool_name, params,
                        agent_name=_agent_name_var.get(),
                        result=f"{last_error_type}: {last_error[:200]}",
                        success=False,
                        duration_ms=float(duration_ms),
                    )
                return ToolResult(
                    tool_name=tool_name,
                    content=f"Fehler: {last_error}",
                    is_error=True,
                    error_type=last_error_type,
                    duration_ms=duration_ms,
                )

            if attempt < self._max_retries:
                delay = self._base_delay * (2 ** (attempt - 1))
                log.warning(
                    "executor_retry",
                    tool=tool_name,
                    attempt=attempt,
                    max_retries=self._max_retries,
                    error_type=last_error_type,
                    error=last_error,
                    delay_s=delay,
                )
                await asyncio.sleep(delay)

        # Alle Retries erschöpft
        total_duration_ms = int((time.monotonic() - total_start) * 1000)
        log.error(
            "executor_retries_exhausted",
            tool=tool_name,
            attempts=self._max_retries,
            error_type=last_error_type,
            error=last_error,
            total_duration_ms=total_duration_ms,
        )
        # Gap melden für Auto-Skill-Generator
        if self._gap_detector:
            self._gap_detector.report_repeated_failure(
                tool_name, f"Retries exhausted: {last_error_type}: {last_error[:200]}",
            )
        # Audit: Retries erschöpft
        if self._audit_logger:
            self._audit_logger.log_tool_call(
                tool_name, params,
                agent_name=_agent_name_var.get(),
                result=f"Retries exhausted: {last_error_type}: {last_error[:200]}",
                success=False,
                duration_ms=float(total_duration_ms),
            )
        return ToolResult(
            tool_name=tool_name,
            content=(f"Fehler nach {self._max_retries} Versuchen: {last_error}"),
            is_error=True,
            error_type=last_error_type,
            duration_ms=total_duration_ms,
        )
