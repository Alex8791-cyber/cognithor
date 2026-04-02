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
import contextlib
import contextvars
import time
from typing import TYPE_CHECKING, Any

from jarvis.core.plan_graph import PlanGraph
from jarvis.models import (
    ActionPlan,
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
_agent_workspace_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "agent_workspace", default=None
)
_agent_sandbox_var: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "agent_sandbox", default=None
)
_agent_name_var: contextvars.ContextVar[str] = contextvars.ContextVar("agent_name", default="")
_session_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("session_id", default="")
_fact_question_var: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "fact_question", default=False
)

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

    # Retryable error types (transient errors worth retrying)
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

    # Tools that accept working_dir and should use agent workspace
    WORKSPACE_TOOLS = frozenset(
        {
            "exec_command",
            "write_file",
            "read_file",
            "edit_file",
            "list_directory",
            "run_python",
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
        """Initialize the executor with configuration and MCP client."""
        self._config = config
        self._mcp_client = mcp_client
        self._gap_detector = gap_detector
        self._runtime_monitor = runtime_monitor
        self._audit_logger = audit_logger
        self._task_profiler = task_profiler
        self._task_telemetry = task_telemetry
        self._error_clusterer = error_clusterer
        # Read executor limits from config (with safe defaults)
        _exec = getattr(config, "executor", None)
        self._default_timeout: int = getattr(_exec, "default_timeout_seconds", 30)
        self._max_retries: int = getattr(_exec, "max_retries", 3)
        self._base_delay: float = getattr(_exec, "backoff_base_delay_seconds", 1.0)
        self._max_output: int = getattr(_exec, "max_output_chars", 10000)
        self._max_parallel: int = getattr(_exec, "max_parallel_tools", 4)
        # Longer timeouts for tools that load large models (e.g. Vision 20 GB+)
        self._tool_timeouts: dict[str, int] = {
            "media_analyze_image": getattr(_exec, "media_analyze_image_timeout", 180),
            "media_transcribe_audio": getattr(_exec, "media_transcribe_audio_timeout", 120),
            "media_extract_text": getattr(_exec, "media_extract_text_timeout", 120),
            "media_tts": getattr(_exec, "media_tts_timeout", 120),
            "run_python": getattr(_exec, "run_python_timeout", 120),
            # Synthesis tools need longer — they process large entity sets via LLM
            "knowledge_contradictions": 120,
            "knowledge_synthesize": 120,
            "knowledge_gaps": 120,
            "knowledge_timeline": 90,
            # Deep research can take multiple rounds
            "deep_research": 180,
            "search_and_read": 60,
            # OSINT investigations run multiple collectors
            "investigate_person": 120,
            "investigate_project": 120,
            "investigate_org": 120,
        }
        # Agent context tokens (for contextvar reset)
        self._ctx_tokens: list[contextvars.Token] = []
        # Status callback (set by Gateway for progress feedback)
        self._status_callback: Any = None
        # Tactical Memory (wired by gateway after init)
        self._tactical_memory: Any = None

    def reload_config(self, config: JarvisConfig) -> None:
        """Update executor limits from new config (live reload).

        Called by the gateway when the user changes settings in the UI.
        """
        self._config = config
        _exec = getattr(config, "executor", None)
        self._default_timeout = getattr(_exec, "default_timeout_seconds", 30)
        self._max_retries = getattr(_exec, "max_retries", 3)
        self._base_delay = getattr(_exec, "backoff_base_delay_seconds", 1.0)
        self._max_output = getattr(_exec, "max_output_chars", 10000)
        self._max_parallel = getattr(_exec, "max_parallel_tools", 4)
        self._tool_timeouts = {
            "media_analyze_image": getattr(_exec, "media_analyze_image_timeout", 180),
            "media_transcribe_audio": getattr(_exec, "media_transcribe_audio_timeout", 120),
            "media_extract_text": getattr(_exec, "media_extract_text_timeout", 120),
            "media_tts": getattr(_exec, "media_tts_timeout", 120),
            "knowledge_contradictions": 120,
            "knowledge_synthesize": 120,
            "knowledge_gaps": 120,
            "knowledge_timeline": 90,
            "deep_research": 180,
            "search_and_read": 60,
            "investigate_person": 120,
            "investigate_project": 120,
            "investigate_org": 120,
            "run_python": getattr(_exec, "run_python_timeout", 120),
        }
        log.info("executor_config_reloaded")

    def set_mcp_client(self, client: JarvisMCPClient) -> None:
        """Set the MCP client (can be set after initialization)."""
        self._mcp_client = client

    def set_status_callback(self, callback: Any) -> None:
        """Set the status callback for progress messages."""
        self._status_callback = callback

    def set_agent_context(
        self,
        workspace_dir: str | None = None,
        sandbox_overrides: dict[str, Any] | None = None,
        agent_name: str = "",
        session_id: str = "",
    ) -> None:
        """Set the agent context for the next execution.

        Set by the gateway per request, based on the
        routed agent.

        Args:
            workspace_dir: Agent-specific workspace directory.
            sandbox_overrides: Agent sandbox config
                (network, max_memory_mb, timeout, etc.)
            agent_name: Name of the active agent (for audit/monitor).
            session_id: Session ID for profiling/telemetry.
        """
        # Reset old tokens before setting new ones
        self.clear_agent_context()
        self._ctx_tokens = [
            _agent_workspace_var.set(workspace_dir),
            _agent_sandbox_var.set(sandbox_overrides),
            _agent_name_var.set(agent_name),
            _session_id_var.set(session_id),
        ]

    def set_fact_question_context(self, is_fact: bool) -> None:
        """Mark the current request as a factual question.

        When True, ``cross_check=True`` is automatically injected into
        ``search_and_read`` calls so multiple sources are compared.
        """
        self._ctx_tokens.append(_fact_question_var.set(is_fact))

    def clear_agent_context(self) -> None:
        """Clear the agent context after execution."""
        for token in self._ctx_tokens:
            with contextlib.suppress(ValueError):
                token.var.reset(token)
        self._ctx_tokens = []

    async def execute(
        self,
        actions: list[PlannedAction],
        decisions: list[GateDecision],
        *,
        max_parallel: int | None = None,
    ) -> list[ToolResult]:
        """Execute approved actions in parallel using DAG-based scheduling.

        Builds a PlanGraph from the actions, respects dependencies
        and executes independent actions in parallel waves.
        Only ALLOW, INFORM and MASK actions are executed.
        BLOCK and unapproved actions are skipped.

        Args:
            actions: List of planned actions.
            decisions: Corresponding gatekeeper decisions.
            max_parallel: Maximum number of parallel running tools.

        Returns:
            List of ToolResults (one result per action).
        """
        if len(actions) != len(decisions):
            raise ExecutionError(
                f"Anzahl Aktionen ({len(actions)}) ≠ Entscheidungen ({len(decisions)})"
            )

        if not actions:
            return []

        if max_parallel is None:
            max_parallel = self._max_parallel

        # --- Computer Use: force sequential execution ---
        # When computer_* tools are in the plan, they MUST run one-by-one
        # because each step depends on the screen state from the previous.
        _CU_TOOLS = frozenset(
            {
                "computer_screenshot",
                "computer_click",
                "computer_type",
                "computer_hotkey",
                "computer_scroll",
                "computer_drag",
            }
        )
        _has_computer_use = any(a.tool in _CU_TOOLS for a in actions)
        if _has_computer_use:
            max_parallel = 1

        # --- Build DAG from actions ---
        plan = ActionPlan(goal="execution", steps=actions)
        graph = PlanGraph.from_action_plan(plan)

        # Map node_id → (original_index, action, decision)
        node_ids = list(graph._nodes.keys())
        node_map: dict[str, tuple[int, PlannedAction, GateDecision]] = {}
        for i, node_id in enumerate(node_ids):
            node_map[node_id] = (i, actions[i], decisions[i])

        results: list[ToolResult | None] = [None] * len(actions)
        completed_ids: set[str] = set()
        semaphore = asyncio.Semaphore(max_parallel)

        # --- Pre-pass: Mark blocked actions and add to completed_ids ---
        # Blocked actions count as "completed" for dependency resolution,
        # so their dependents CAN proceed. The blocked action itself is
        # recorded as GatekeeperBlock error, but doesn't block the DAG.
        for node_id, (idx, action, decision) in node_map.items():
            if decision.status not in (GateStatus.ALLOW, GateStatus.INFORM, GateStatus.MASK):
                if self._audit_logger:
                    self._audit_logger.log_gatekeeper(
                        decision.status.value,
                        decision.reason,
                        tool_name=action.tool,
                        agent_name=_agent_name_var.get(),
                    )
                results[idx] = ToolResult(
                    tool_name=action.tool,
                    content=f"Aktion übersprungen: {decision.status.value} -- {decision.reason}",
                    is_error=True,
                    error_type="GatekeeperBlock",
                )
                completed_ids.add(node_id)

        # --- Wave loop ---
        async def _run_with_sem(nid: str) -> None:
            idx, action, decision = node_map[nid]
            # Use masked params when MASK
            if decision.status == GateStatus.MASK and decision.masked_params:
                params = dict(decision.masked_params)
            else:
                params = dict(action.params)

            async with semaphore:
                result = await self._execute_single(action.tool, params)

            results[idx] = result
            if result.success:
                completed_ids.add(nid)

            log.info(
                "executor_tool_result",
                tool=action.tool,
                success=result.success,
                duration_ms=result.duration_ms,
                content_length=len(result.content),
            )

            if self._tactical_memory is not None:
                with contextlib.suppress(Exception):
                    self._tactical_memory.record_outcome(
                        tool=action.tool,
                        params=action.params or {},
                        success=result.success,
                        duration_ms=int(getattr(result, "duration_ms", 0)),
                        context="",
                        error=result.content[:100]
                        if not result.success and result.content
                        else None,
                    )

        while True:
            ready = [
                nid
                for nid in graph.get_ready_nodes(completed_ids)
                if results[node_map[nid][0]] is None
            ]

            pending = [
                nid
                for nid in node_ids
                if nid not in completed_ids and results[node_map[nid][0]] is None
            ]

            if not ready:
                if pending:
                    # Deadlock: remaining nodes have unresolvable deps
                    for nid in pending:
                        idx, action, _dec = node_map[nid]
                        results[idx] = ToolResult(
                            tool_name=action.tool,
                            content="Aktion übersprungen: Abhängigkeiten nicht erfüllt (Deadlock)",
                            is_error=True,
                            error_type="DependencyError",
                        )
                break

            await asyncio.gather(*[_run_with_sem(nid) for nid in ready])

        # --- Fill any remaining None slots (safety net) ---
        for i, r in enumerate(results):
            if r is None:
                results[i] = ToolResult(
                    tool_name=actions[i].tool,
                    content="Aktion übersprungen: Abhängigkeiten nicht erfüllt",
                    is_error=True,
                    error_type="DependencyError",
                )

        return results  # type: ignore[return-value]

    async def _execute_single(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> ToolResult:
        """Execute a single tool with retry, backoff and timeout.

        Agent context injection:
          - For WORKSPACE_TOOLS: injects working_dir from agent workspace
          - For exec_command: injects sandbox overrides (_sandbox_network, _sandbox_timeout)

        Retry strategy:
          - Maximum 3 attempts (configurable)
          - Exponential backoff: 1s -> 2s -> 4s
          - Only for transient errors (Timeout, Connection, Ollama)
          - No retry for logical errors (Permission, NotFound)

        Args:
            tool_name: Name of the MCP tool.
            params: Tool parameters.

        Returns:
            ToolResult with success or error.
        """
        if self._mcp_client is None:
            return ToolResult(
                tool_name=tool_name,
                content="Kein MCP-Client verfügbar",
                is_error=True,
                error_type="NoMCPClient",
            )

        _MAX_TIMEOUT = 300  # Hard ceiling: 5 minutes, regardless of LLM request
        raw_timeout = params.pop(
            "_timeout", self._tool_timeouts.get(tool_name, self._default_timeout)
        )
        try:
            timeout = min(int(raw_timeout), _MAX_TIMEOUT)
        except (TypeError, ValueError):
            timeout = self._default_timeout

        # --- Inject agent context into tool params ---
        if (
            _agent_workspace_var.get()
            and tool_name in self.WORKSPACE_TOOLS
            and "working_dir" not in params
        ):
            params["working_dir"] = _agent_workspace_var.get()

        # --- Fact question: inject cross_check for search_and_read ---
        if (
            _fact_question_var.get()
            and tool_name == "search_and_read"
            and "cross_check" not in params
        ):
            params["cross_check"] = True
            log.debug("fact_question_cross_check_injected", tool=tool_name)

        if _agent_sandbox_var.get() and tool_name == "exec_command":
            # Pass sandbox overrides as internal params
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

        # --- Runtime Monitor: security check BEFORE execution ---
        if self._runtime_monitor:
            security_event = self._runtime_monitor.check_tool_call(
                tool_name,
                params,
                agent_name=_agent_name_var.get(),
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

                # Truncate output if too long
                content = result.content if hasattr(result, "content") else str(result)
                truncated = False
                if len(content) > self._max_output:
                    content = content[: self._max_output]
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
                # Profiler: record tool call
                if self._task_profiler:
                    try:
                        self._task_profiler.record_tool_call(
                            tool_name=tool_name,
                            latency_ms=float(duration_ms),
                            success=not is_error,
                            session_id=_session_id_var.get(),
                        )
                    except Exception as exc:
                        log.debug("profiler_record_error", error=str(exc))
                # Audit: log successful tool execution
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
                last_error = f"Timeout after {timeout} seconds"
                last_error_type = "TimeoutError"

            except Exception as exc:
                last_error_type = type(exc).__name__
                last_error = str(exc)[:500]

            duration_ms = int((time.monotonic() - start) * 1000)

            # Profiler: record failed tool call
            if self._task_profiler:
                try:
                    self._task_profiler.record_tool_call(
                        tool_name=tool_name,
                        latency_ms=float(duration_ms),
                        success=False,
                        error_type=last_error_type,
                        session_id=_session_id_var.get(),
                    )
                except Exception as exc:
                    log.debug("profiler_record_error", error=str(exc))

            # Error-Clustering
            if self._error_clusterer:
                try:
                    self._error_clusterer.add_error(
                        last_error_type,
                        last_error,
                        f"tool={tool_name}",
                    )
                except Exception as exc:
                    log.debug("error_clusterer_error", error=str(exc))

            # Retry decision
            if last_error_type not in self.RETRYABLE_ERRORS:
                log.error(
                    "executor_error_no_retry",
                    tool=tool_name,
                    error_type=last_error_type,
                    error=last_error,
                    duration_ms=duration_ms,
                )
                # Report gap for auto skill generator
                if self._gap_detector:
                    self._gap_detector.report_unknown_tool(
                        tool_name,
                        context=f"{last_error_type}: {last_error[:200]}",
                    )
                # Audit: log error
                if self._audit_logger:
                    self._audit_logger.log_tool_call(
                        tool_name,
                        params,
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
                import random

                _exp_delay = self._base_delay * (2 ** (attempt - 1))
                delay = min(_exp_delay * (0.5 + random.random()), 30.0)
                log.warning(
                    "executor_retry",
                    tool=tool_name,
                    attempt=attempt,
                    max_retries=self._max_retries,
                    error_type=last_error_type,
                    error=last_error,
                    delay_s=round(delay, 2),
                )
                # Status callback: retry visibility
                if self._status_callback is not None:
                    try:
                        await self._status_callback(
                            "retrying",
                            f"Versuch {attempt + 1} von {self._max_retries}...",
                        )
                    except Exception as exc:
                        log.debug("status_callback_error", error=str(exc))
                await asyncio.sleep(delay)

        # All retries exhausted
        total_duration_ms = int((time.monotonic() - total_start) * 1000)
        log.error(
            "executor_retries_exhausted",
            tool=tool_name,
            attempts=self._max_retries,
            error_type=last_error_type,
            error=last_error,
            total_duration_ms=total_duration_ms,
        )
        # Report gap for auto skill generator
        if self._gap_detector:
            self._gap_detector.report_repeated_failure(
                tool_name,
                f"Retries exhausted: {last_error_type}: {last_error[:200]}",
            )
        # Audit: retries exhausted
        if self._audit_logger:
            self._audit_logger.log_tool_call(
                tool_name,
                params,
                agent_name=_agent_name_var.get(),
                result=f"Retries exhausted: {last_error_type}: {last_error[:200]}",
                success=False,
                duration_ms=float(total_duration_ms),
            )
        # User-friendly error message
        try:
            from jarvis.utils.error_messages import retry_exhausted_message

            friendly_msg = retry_exhausted_message(tool_name, self._max_retries, last_error)
        except Exception:
            friendly_msg = f"Fehler nach {self._max_retries} Versuchen: {last_error}"
        return ToolResult(
            tool_name=tool_name,
            content=friendly_msg,
            is_error=True,
            error_type=last_error_type,
            duration_ms=total_duration_ms,
        )
