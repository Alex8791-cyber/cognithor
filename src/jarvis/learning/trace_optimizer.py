"""Targeted optimization engine for the GEPA self-improvement system.

Takes CausalFindings from the CausalAttributor and generates specific,
actionable optimization proposals. Supports optional LLM-based patch
generation with template-based fallbacks.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from jarvis.db import SQLITE_BUSY_TIMEOUT_MS
from jarvis.utils.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

    from .causal_attributor import CausalFinding
    from .execution_trace import ExecutionTrace, TraceStore

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OPTIMIZATION_TYPES: dict[str, str] = {
    "prompt_patch": "Modify planner or executor system prompt",
    "tool_param": "Adjust tool default parameters or validation",
    "strategy_change": "Change tool selection or execution strategy",
    "new_procedure": "Add procedural memory entry for better handling",
    "guardrail": "Add gatekeeper rule to prevent specific failure",
    "context_enrichment": "Add context pipeline step for better information",
}

_CATEGORY_HANDLER_MAP: dict[str, str] = {
    "timeout": "_propose_for_timeout",
    "wrong_tool": "_propose_for_wrong_tool",
    "bad_params": "_propose_for_bad_params",
    "hallucination": "_propose_for_hallucination",
    "missing_context": "_propose_for_missing_context",
    "cascade_failure": "_propose_for_cascade",
    "permission_denied": "_propose_for_permission",
    "rate_limit": "_propose_for_rate_limit",
    "parse_error": "_propose_for_parse_error",
}

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class OptimizationProposal:
    """A specific, actionable improvement proposal."""

    proposal_id: str
    finding_id: str
    optimization_type: str
    target: str
    description: str
    patch_before: str
    patch_after: str
    estimated_impact: float
    confidence: float
    failure_category: str
    tool_name: str
    evidence_trace_ids: list[str]
    status: str = "proposed"
    applied_at: float = 0.0
    metrics_before: dict[str, float] = field(default_factory=dict)
    metrics_after: dict[str, float] = field(default_factory=dict)
    created_at: float = 0.0


# ---------------------------------------------------------------------------
# SQLite persistence
# ---------------------------------------------------------------------------


class ProposalStore:
    """SQLite persistence for optimization proposals."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = str(db_path)
        self._conn: sqlite3.Connection | None = None
        self._ensure_schema()

    # -- connection ----------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def _ensure_schema(self) -> None:
        conn = self._get_conn()
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS optimization_proposals (
                proposal_id   TEXT PRIMARY KEY,
                finding_id    TEXT NOT NULL,
                optimization_type TEXT NOT NULL,
                target        TEXT NOT NULL,
                description   TEXT NOT NULL,
                patch_before  TEXT NOT NULL DEFAULT '',
                patch_after   TEXT NOT NULL DEFAULT '',
                estimated_impact REAL NOT NULL DEFAULT 0.0,
                confidence    REAL NOT NULL DEFAULT 0.0,
                failure_category TEXT NOT NULL DEFAULT '',
                tool_name     TEXT NOT NULL DEFAULT '',
                evidence_trace_ids_json TEXT NOT NULL DEFAULT '[]',
                status        TEXT NOT NULL DEFAULT 'proposed',
                applied_at    REAL NOT NULL DEFAULT 0.0,
                metrics_before_json TEXT NOT NULL DEFAULT '{}',
                metrics_after_json  TEXT NOT NULL DEFAULT '{}',
                created_at    REAL NOT NULL DEFAULT 0.0
            );
            CREATE INDEX IF NOT EXISTS idx_opt_status
                ON optimization_proposals(status);
            CREATE INDEX IF NOT EXISTS idx_opt_finding
                ON optimization_proposals(finding_id);
            CREATE INDEX IF NOT EXISTS idx_opt_created
                ON optimization_proposals(created_at);
            """
        )
        conn.commit()

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _row_to_proposal(row: sqlite3.Row) -> OptimizationProposal:
        return OptimizationProposal(
            proposal_id=row["proposal_id"],
            finding_id=row["finding_id"],
            optimization_type=row["optimization_type"],
            target=row["target"],
            description=row["description"],
            patch_before=row["patch_before"],
            patch_after=row["patch_after"],
            estimated_impact=row["estimated_impact"],
            confidence=row["confidence"],
            failure_category=row["failure_category"],
            tool_name=row["tool_name"],
            evidence_trace_ids=json.loads(row["evidence_trace_ids_json"]),
            status=row["status"],
            applied_at=row["applied_at"],
            metrics_before=json.loads(row["metrics_before_json"]),
            metrics_after=json.loads(row["metrics_after_json"]),
            created_at=row["created_at"],
        )

    # -- CRUD ----------------------------------------------------------------

    def save_proposal(self, proposal: OptimizationProposal) -> None:
        conn = self._get_conn()
        conn.execute(
            """
            INSERT OR REPLACE INTO optimization_proposals (
                proposal_id, finding_id, optimization_type, target, description,
                patch_before, patch_after, estimated_impact, confidence,
                failure_category, tool_name, evidence_trace_ids_json,
                status, applied_at, metrics_before_json, metrics_after_json,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                proposal.proposal_id,
                proposal.finding_id,
                proposal.optimization_type,
                proposal.target,
                proposal.description,
                proposal.patch_before,
                proposal.patch_after,
                proposal.estimated_impact,
                proposal.confidence,
                proposal.failure_category,
                proposal.tool_name,
                json.dumps(proposal.evidence_trace_ids),
                proposal.status,
                proposal.applied_at,
                json.dumps(proposal.metrics_before),
                json.dumps(proposal.metrics_after),
                proposal.created_at,
            ),
        )
        conn.commit()

    def update_status(self, proposal_id: str, status: str, **kwargs: Any) -> None:
        conn = self._get_conn()
        sets = ["status = ?"]
        params: list[Any] = [status]

        if "applied_at" in kwargs:
            sets.append("applied_at = ?")
            params.append(kwargs["applied_at"])
        if "metrics_before" in kwargs:
            sets.append("metrics_before_json = ?")
            params.append(json.dumps(kwargs["metrics_before"]))
        if "metrics_after" in kwargs:
            sets.append("metrics_after_json = ?")
            params.append(json.dumps(kwargs["metrics_after"]))

        params.append(proposal_id)
        conn.execute(
            f"UPDATE optimization_proposals SET {', '.join(sets)} WHERE proposal_id = ?",
            params,
        )
        conn.commit()

    def get_proposal(self, proposal_id: str) -> OptimizationProposal | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM optimization_proposals WHERE proposal_id = ?",
            (proposal_id,),
        ).fetchone()
        return self._row_to_proposal(row) if row else None

    def get_pending(self) -> list[OptimizationProposal]:
        return self.get_by_status("proposed")

    def get_applied(self) -> list[OptimizationProposal]:
        return self.get_by_status("applied")

    def get_by_status(self, status: str) -> list[OptimizationProposal]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM optimization_proposals WHERE status = ? ORDER BY created_at DESC",
            (status,),
        ).fetchall()
        return [self._row_to_proposal(r) for r in rows]

    def get_history(self, limit: int = 50) -> list[OptimizationProposal]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM optimization_proposals ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_proposal(r) for r in rows]

    def delete_old(self, older_than_days: int = 90) -> int:
        cutoff = time.time() - older_than_days * 86400
        conn = self._get_conn()
        cur = conn.execute(
            "DELETE FROM optimization_proposals WHERE created_at < ? AND status IN ('rejected', 'rolled_back')",
            (cutoff,),
        )
        conn.commit()
        return cur.rowcount


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------


_PREVENTION_RULES: dict[str, str] = {
    "timeout": "Rate-limit {tool}: max 1 call per 2 seconds",
    "bad_parameters": "Validate {parameter} exists before calling {tool}",
    "bad_params": "Validate {parameter} exists before calling {tool}",
    "hallucination": "Cross-reference {tool} results with web_search before presenting",
    "wrong_tool_choice": "For {context}, prefer {alternative_tool} over {tool}",
    "wrong_tool": "For {context}, prefer {alternative_tool} over {tool}",
    "missing_context": "Load memory + vault before executing {tool}",
    "cascade_failure": "Check {upstream_tool} success before running {tool}",
    "permission_denied": "Request approval before {tool} on {resource}",
    "rate_limited": "Add exponential backoff: 1s, 2s, 4s for {tool}",
    "rate_limit": "Add exponential backoff: 1s, 2s, 4s for {tool}",
    "parse_error": "Validate {tool} output format before processing",
}


class TraceOptimizer:
    """Generates targeted optimization proposals from causal findings."""

    def __init__(
        self,
        proposal_store: ProposalStore,
        llm_client: Any = None,
        reflexion_memory: Any = None,
    ) -> None:
        self._store = proposal_store
        self._llm = llm_client
        self._reflexion_memory = reflexion_memory

    # -- public API ----------------------------------------------------------

    def propose_optimizations(
        self,
        targets: list[dict[str, Any]],
        trace_store: TraceStore,
    ) -> list[OptimizationProposal]:
        """Generate proposals for each improvement target.

        For each target, call the appropriate generator based on
        ``failure_category``.  Also generates a prevention rule and
        stores it via ReflexionMemory if available.
        """
        proposals: list[OptimizationProposal] = []

        for target in targets:
            category = target.get("failure_category", "")
            tool_name = target.get("tool_name", "unknown")
            trace_ids: list[str] = target.get("trace_ids", [])

            # Collect example traces for context
            examples: list[ExecutionTrace] = []
            for tid in trace_ids[:10]:  # cap to avoid excessive lookups
                trace = trace_store.get_trace(tid)
                if trace is not None:
                    examples.append(trace)

            handler_name = _CATEGORY_HANDLER_MAP.get(category, "_propose_generic")
            handler = getattr(self, handler_name, self._propose_generic)

            try:
                proposal = handler(target, examples)
                if proposal is not None:
                    proposal.created_at = proposal.created_at or time.time()
                    proposal.estimated_impact = self.score_proposal(proposal, trace_store)
                    self._store.save_proposal(proposal)
                    proposals.append(proposal)
                    log.info(
                        "Generated %s proposal for %s (impact=%.2f)",
                        proposal.optimization_type,
                        tool_name,
                        proposal.estimated_impact,
                    )

                    # Generate and store prevention rule
                    rule = self._generate_prevention_rule(target, category)
                    if rule and self._reflexion_memory is not None:
                        try:
                            self._reflexion_memory.record_error(
                                tool_name=tool_name,
                                error_category=category,
                                error_message=proposal.description[:200],
                                root_cause=proposal.description,
                                prevention_rule=rule,
                                task_context=f"proposal:{proposal.proposal_id}",
                            )
                            log.info(
                                "prevention_rule_stored",
                                tool=tool_name,
                                category=category,
                                rule=rule[:80],
                            )
                        except Exception:
                            log.debug("prevention_rule_store_failed", exc_info=True)
            except Exception:
                log.exception("Failed to generate proposal for target %s", target)

        return proposals

    # -- prevention rule generation ------------------------------------------

    @staticmethod
    def _generate_prevention_rule(
        target: dict[str, Any],
        category: str,
    ) -> str:
        """Generate a concrete prevention rule string for a failure category.

        Substitutes context-specific values (tool name, parameters, etc.)
        into the template for the given category.
        """
        template = _PREVENTION_RULES.get(category, "")
        if not template:
            return ""

        tool = target.get("tool_name", "unknown")
        parameter = target.get("error_param", "input")
        context = target.get("context", "this task")
        alternative_tool = target.get("suggested_tool", "alternative_tool")
        upstream_tool = target.get("upstream_tool", "upstream_tool")
        resource = target.get("resource", "resource")

        return template.format(
            tool=tool,
            parameter=parameter,
            context=context,
            alternative_tool=alternative_tool,
            upstream_tool=upstream_tool,
            resource=resource,
        )

    # -- category-specific generators ----------------------------------------

    def _propose_for_timeout(
        self, target: dict[str, Any], examples: list[ExecutionTrace]
    ) -> OptimizationProposal:
        """Propose timeout adjustment or retry strategy."""
        tool = target.get("tool_name", "unknown")
        finding_id = target.get("finding_id", "")
        trace_ids = target.get("trace_ids", [])
        avg_duration = _avg_field(examples, "duration", default=30.0)

        suggested_timeout = max(60, int(avg_duration * 2.5))

        llm_patch = self._generate_with_llm(target, examples, "tool_param")
        patch_after = llm_patch or (
            f"Set {tool}.timeout = {suggested_timeout}s (was likely default). "
            f"Add retry with exponential backoff: max_retries=2, "
            f"base_delay=2s, max_delay=10s."
        )

        return OptimizationProposal(
            proposal_id=str(uuid.uuid4()),
            finding_id=finding_id,
            optimization_type="tool_param",
            target=f"{tool}.timeout_config",
            description=(
                f"Tool '{tool}' is timing out frequently "
                f"(avg duration {avg_duration:.1f}s). "
                f"Increase timeout to {suggested_timeout}s and add retry logic."
            ),
            patch_before=f"{tool}.timeout = default",
            patch_after=patch_after,
            estimated_impact=0.0,  # scored later
            confidence=0.75,
            failure_category="timeout",
            tool_name=tool,
            evidence_trace_ids=trace_ids,
            created_at=time.time(),
        )

    def _propose_for_wrong_tool(
        self, target: dict[str, Any], examples: list[ExecutionTrace]
    ) -> OptimizationProposal:
        """Propose tool selection guidance in planner prompt."""
        tool = target.get("tool_name", "unknown")
        finding_id = target.get("finding_id", "")
        trace_ids = target.get("trace_ids", [])

        # Try to identify the better alternative from examples
        better_tool = _extract_field(examples, "suggested_tool") or "the correct tool"

        llm_patch = self._generate_with_llm(target, examples, "strategy_change")
        patch_after = llm_patch or (
            f"When the task involves '{tool}' capabilities, evaluate whether "
            f"'{better_tool}' would be more appropriate. "
            f"Prefer '{better_tool}' for this category of requests."
        )

        return OptimizationProposal(
            proposal_id=str(uuid.uuid4()),
            finding_id=finding_id,
            optimization_type="strategy_change",
            target="planner.system_prompt",
            description=(
                f"Planner frequently selects '{tool}' when '{better_tool}' "
                f"would be more effective. Add tool selection guidance."
            ),
            patch_before="",
            patch_after=patch_after,
            estimated_impact=0.0,
            confidence=0.65,
            failure_category="wrong_tool",
            tool_name=tool,
            evidence_trace_ids=trace_ids,
            created_at=time.time(),
        )

    def _propose_for_bad_params(
        self, target: dict[str, Any], examples: list[ExecutionTrace]
    ) -> OptimizationProposal:
        """Propose parameter validation or defaults."""
        tool = target.get("tool_name", "unknown")
        finding_id = target.get("finding_id", "")
        trace_ids = target.get("trace_ids", [])

        bad_params = _extract_field(examples, "error_param") or "parameters"

        llm_patch = self._generate_with_llm(target, examples, "tool_param")
        patch_after = llm_patch or (
            f"Add input validation for {tool}: "
            f"check '{bad_params}' before execution. "
            f"Apply sensible defaults when values are missing or out of range."
        )

        return OptimizationProposal(
            proposal_id=str(uuid.uuid4()),
            finding_id=finding_id,
            optimization_type="tool_param",
            target=f"{tool}.params",
            description=(
                f"Tool '{tool}' fails due to bad parameter values "
                f"('{bad_params}'). Add validation and defaults."
            ),
            patch_before="",
            patch_after=patch_after,
            estimated_impact=0.0,
            confidence=0.70,
            failure_category="bad_params",
            tool_name=tool,
            evidence_trace_ids=trace_ids,
            created_at=time.time(),
        )

    def _propose_for_hallucination(
        self, target: dict[str, Any], examples: list[ExecutionTrace]
    ) -> OptimizationProposal:
        """Propose anti-hallucination guardrail."""
        tool = target.get("tool_name", "unknown")
        finding_id = target.get("finding_id", "")
        trace_ids = target.get("trace_ids", [])

        llm_patch = self._generate_with_llm(target, examples, "guardrail")
        patch_after = llm_patch or (
            "Always verify factual claims with web_search before presenting "
            "them to the user. If a claim cannot be verified, explicitly "
            "state the uncertainty level."
        )

        return OptimizationProposal(
            proposal_id=str(uuid.uuid4()),
            finding_id=finding_id,
            optimization_type="guardrail",
            target="gatekeeper.rules",
            description=(
                f"Hallucinated output detected in '{tool}' responses. "
                f"Add verification guardrail before presenting results."
            ),
            patch_before="",
            patch_after=patch_after,
            estimated_impact=0.0,
            confidence=0.60,
            failure_category="hallucination",
            tool_name=tool,
            evidence_trace_ids=trace_ids,
            created_at=time.time(),
        )

    def _propose_for_missing_context(
        self, target: dict[str, Any], examples: list[ExecutionTrace]
    ) -> OptimizationProposal:
        """Propose context enrichment procedure."""
        tool = target.get("tool_name", "unknown")
        finding_id = target.get("finding_id", "")
        trace_ids = target.get("trace_ids", [])

        llm_patch = self._generate_with_llm(target, examples, "context_enrichment")
        patch_after = llm_patch or (
            f"Before invoking '{tool}', retrieve relevant context from "
            f"memory and vault. Check episodic memory for similar past "
            f"interactions and include key details in the tool call."
        )

        return OptimizationProposal(
            proposal_id=str(uuid.uuid4()),
            finding_id=finding_id,
            optimization_type="context_enrichment",
            target="procedure.context_enrichment",
            description=(
                f"Tool '{tool}' fails due to missing context. "
                f"Add pre-execution context lookup step."
            ),
            patch_before="",
            patch_after=patch_after,
            estimated_impact=0.0,
            confidence=0.65,
            failure_category="missing_context",
            tool_name=tool,
            evidence_trace_ids=trace_ids,
            created_at=time.time(),
        )

    def _propose_for_cascade(
        self, target: dict[str, Any], examples: list[ExecutionTrace]
    ) -> OptimizationProposal:
        """Propose error recovery strategy."""
        tool = target.get("tool_name", "unknown")
        finding_id = target.get("finding_id", "")
        trace_ids = target.get("trace_ids", [])

        fallback_tool = _extract_field(examples, "fallback_tool") or "an alternative approach"

        llm_patch = self._generate_with_llm(target, examples, "strategy_change")
        patch_after = llm_patch or (
            f"If '{tool}' fails, attempt {fallback_tool} before reporting "
            f"failure. Limit cascade depth to 2 retries with different "
            f"strategies."
        )

        return OptimizationProposal(
            proposal_id=str(uuid.uuid4()),
            finding_id=finding_id,
            optimization_type="strategy_change",
            target="planner.system_prompt",
            description=(
                f"Cascade failures originating from '{tool}'. "
                f"Add fallback strategy and cascade depth limit."
            ),
            patch_before="",
            patch_after=patch_after,
            estimated_impact=0.0,
            confidence=0.70,
            failure_category="cascade_failure",
            tool_name=tool,
            evidence_trace_ids=trace_ids,
            created_at=time.time(),
        )

    def _propose_for_permission(
        self, target: dict[str, Any], examples: list[ExecutionTrace]
    ) -> OptimizationProposal:
        """Propose gatekeeper allowlist update or pre-check."""
        tool = target.get("tool_name", "unknown")
        finding_id = target.get("finding_id", "")
        trace_ids = target.get("trace_ids", [])

        llm_patch = self._generate_with_llm(target, examples, "guardrail")
        patch_after = llm_patch or (
            f"Add '{tool}' to gatekeeper green_list if operation is safe, "
            f"or add a pre-flight permission check that validates access "
            f"before attempting the operation."
        )

        return OptimizationProposal(
            proposal_id=str(uuid.uuid4()),
            finding_id=finding_id,
            optimization_type="guardrail",
            target="gatekeeper.green_list",
            description=(
                f"Tool '{tool}' frequently blocked by permissions. "
                f"Evaluate allowlist update or add pre-check."
            ),
            patch_before="",
            patch_after=patch_after,
            estimated_impact=0.0,
            confidence=0.55,
            failure_category="permission_denied",
            tool_name=tool,
            evidence_trace_ids=trace_ids,
            created_at=time.time(),
        )

    def _propose_for_rate_limit(
        self, target: dict[str, Any], examples: list[ExecutionTrace]
    ) -> OptimizationProposal:
        """Propose rate limiting strategy (backoff, caching)."""
        tool = target.get("tool_name", "unknown")
        finding_id = target.get("finding_id", "")
        trace_ids = target.get("trace_ids", [])

        llm_patch = self._generate_with_llm(target, examples, "tool_param")
        patch_after = llm_patch or (
            f"Add rate-limit handling for '{tool}': cache results for 60s, "
            f"use exponential backoff (base=2s, max=30s) on 429 responses, "
            f"and batch requests where possible."
        )

        return OptimizationProposal(
            proposal_id=str(uuid.uuid4()),
            finding_id=finding_id,
            optimization_type="tool_param",
            target=f"{tool}.rate_limit",
            description=(f"Tool '{tool}' hitting rate limits. Add caching and backoff strategy."),
            patch_before="",
            patch_after=patch_after,
            estimated_impact=0.0,
            confidence=0.80,
            failure_category="rate_limit",
            tool_name=tool,
            evidence_trace_ids=trace_ids,
            created_at=time.time(),
        )

    def _propose_for_parse_error(
        self, target: dict[str, Any], examples: list[ExecutionTrace]
    ) -> OptimizationProposal:
        """Propose output parsing improvement."""
        tool = target.get("tool_name", "unknown")
        finding_id = target.get("finding_id", "")
        trace_ids = target.get("trace_ids", [])

        llm_patch = self._generate_with_llm(target, examples, "tool_param")
        patch_after = llm_patch or (
            f"Add robust output parsing for '{tool}': strip markdown "
            f"fences before JSON parse, handle partial/truncated output "
            f"gracefully, and add format instructions to the tool prompt."
        )

        return OptimizationProposal(
            proposal_id=str(uuid.uuid4()),
            finding_id=finding_id,
            optimization_type="tool_param",
            target=f"{tool}.output_parsing",
            description=(
                f"Tool '{tool}' produces unparseable output. "
                f"Add format instructions and robust parsing."
            ),
            patch_before="",
            patch_after=patch_after,
            estimated_impact=0.0,
            confidence=0.75,
            failure_category="parse_error",
            tool_name=tool,
            evidence_trace_ids=trace_ids,
            created_at=time.time(),
        )

    def _propose_generic(
        self, target: dict[str, Any], examples: list[ExecutionTrace]
    ) -> OptimizationProposal:
        """Fallback for uncategorized failures."""
        tool = target.get("tool_name", "unknown")
        category = target.get("failure_category", "unknown")
        finding_id = target.get("finding_id", "")
        trace_ids = target.get("trace_ids", [])

        error_summary = _summarize_errors(examples)

        llm_patch = self._generate_with_llm(target, examples, "new_procedure")
        patch_after = llm_patch or (
            f"When using '{tool}', watch for '{category}' failures. "
            f"Common symptoms: {error_summary}. "
            f"Add pre-checks and validate inputs before execution."
        )

        return OptimizationProposal(
            proposal_id=str(uuid.uuid4()),
            finding_id=finding_id,
            optimization_type="new_procedure",
            target=f"procedure.{tool}_error_handling",
            description=(
                f"Recurring '{category}' failures in '{tool}'. "
                f"Add procedural guidance for error avoidance."
            ),
            patch_before="",
            patch_after=patch_after,
            estimated_impact=0.0,
            confidence=0.50,
            failure_category=category,
            tool_name=tool,
            evidence_trace_ids=trace_ids,
            created_at=time.time(),
        )

    # -- LLM integration (optional) -----------------------------------------

    def _generate_with_llm(
        self,
        target: dict[str, Any],
        examples: list[ExecutionTrace],
        proposal_type: str,
    ) -> str | None:
        """Use LLM to generate a higher-quality patch.

        Returns ``None`` if the LLM client is unavailable or the call fails,
        so the caller falls back to the template.
        """
        if self._llm is None:
            return None

        tool = target.get("tool_name", "unknown")
        category = target.get("failure_category", "unknown")

        # Build a concise context from examples
        example_summaries: list[str] = []
        for ex in examples[:5]:
            error = getattr(ex, "error", "") or ""
            result = getattr(ex, "result_summary", "") or ""
            example_summaries.append(
                f"- tool={getattr(ex, 'tool_name', tool)}, "
                f"error={error[:120]}, result={result[:80]}"
            )
        examples_text = "\n".join(example_summaries) if example_summaries else "None"

        prompt = (
            f"You are an optimization engine for an AI agent. "
            f"Generate a concise patch (1-5 sentences) to fix recurring "
            f"'{category}' failures in tool '{tool}'.\n\n"
            f"Optimization type: {proposal_type}\n"
            f"Type description: {OPTIMIZATION_TYPES.get(proposal_type, 'general fix')}\n\n"
            f"Recent failure examples:\n{examples_text}\n\n"
            f"Write ONLY the patch text — no explanation, no markdown."
        )

        system_msg = (
            "Du bist ein Optimierungs-Experte fuer KI-Agenten. "
            "Generiere konkrete, spezifische Verbesserungsvorschlaege."
        )

        try:
            # Try chat-style API first (preferred — supports system message)
            if hasattr(self._llm, "chat"):
                response = self._llm.chat(
                    model="",  # Use default model from router
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.3,
                )
                text = (
                    getattr(response, "content", None)
                    or getattr(response, "text", None)
                    or (response if isinstance(response, str) else str(response))
                )
            else:
                # Fallback to generate-style API
                response = self._llm.generate(prompt)
                text = (
                    response
                    if isinstance(response, str)
                    else getattr(response, "text", str(response))
                )
            text = text.strip()
            if text and len(text) < 1000:
                return text
            log.warning("LLM patch too long or empty, falling back to template")
            return None
        except Exception as exc:
            log.debug("llm_patch_generation_failed", error=str(exc))
            return None

    # -- scoring -------------------------------------------------------------

    def score_proposal(
        self,
        proposal: OptimizationProposal,
        trace_store: TraceStore,
    ) -> float:
        """Estimate impact based on how many recent traces share this failure.

        ``estimated_impact = (affected_count / total_recent) * confidence``
        """
        try:
            recent = trace_store.get_recent_traces(limit=200)
        except Exception:
            log.debug("Could not fetch recent traces for scoring", exc_info=True)
            return proposal.confidence * 0.1  # minimal fallback

        if not recent:
            return 0.0

        total = len(recent)
        affected = 0
        for trace in recent:
            trace_tool = getattr(trace, "tool_name", None)
            trace_category = getattr(trace, "failure_category", None)
            trace_success = getattr(trace, "success", True)

            if (
                not trace_success
                and trace_tool == proposal.tool_name
                and trace_category == proposal.failure_category
            ):
                affected += 1

        ratio = affected / total if total > 0 else 0.0
        return round(ratio * proposal.confidence, 4)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _avg_field(
    traces: list[Any],
    field_name: str,
    default: float = 0.0,
) -> float:
    """Average a numeric attribute across traces."""
    values = [
        getattr(t, field_name, None) for t in traces if getattr(t, field_name, None) is not None
    ]
    if not values:
        return default
    return sum(values) / len(values)


def _extract_field(traces: list[Any], field_name: str) -> str:
    """Extract the most common non-empty string value for a field."""
    counts: dict[str, int] = {}
    for t in traces:
        val = getattr(t, field_name, None)
        if val and isinstance(val, str):
            counts[val] = counts.get(val, 0) + 1
    if not counts:
        return ""
    return max(counts, key=counts.get)  # type: ignore[arg-type]


def _summarize_errors(traces: list[Any], max_items: int = 3) -> str:
    """Build a short summary of error messages from traces."""
    errors: list[str] = []
    seen: set[str] = set()
    for t in traces:
        err = getattr(t, "error", None) or ""
        short = err[:80].strip()
        if short and short not in seen:
            seen.add(short)
            errors.append(short)
            if len(errors) >= max_items:
                break
    return "; ".join(errors) if errors else "unspecified errors"
