"""Causal root-cause analysis engine for the GEPA self-improvement system.

Analyzes execution traces to determine WHY failures happened, not just
THAT they happened.  Pure heuristic/graph-based -- no LLM calls, no
database, no I/O.
"""

from __future__ import annotations

import re
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jarvis.utils.logging import get_logger

if TYPE_CHECKING:
    pass

from .execution_trace import ExecutionTrace, TraceStep

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Failure categories
# ---------------------------------------------------------------------------

FAILURE_CATEGORIES: dict[str, str] = {
    "timeout": "Tool execution exceeded time limit",
    "wrong_tool_choice": "Planner selected inappropriate tool for the task",
    "bad_parameters": "Tool called with incorrect or missing parameters",
    "hallucination": "Agent generated factually incorrect content",
    "missing_context": "Insufficient context for accurate execution",
    "tool_unavailable": "Required tool not available or not registered",
    "cascade_failure": "Failure in upstream step caused downstream failures",
    "permission_denied": "Gatekeeper blocked the operation",
    "rate_limited": "External API rate limit hit",
    "parse_error": "Failed to parse tool output or LLM response",
}

# ---------------------------------------------------------------------------
# Regex patterns for error normalization
# ---------------------------------------------------------------------------

_NORM_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Timestamps  (2024-01-15T12:30:45 or 2024-01-15 12:30:45)
    (re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}"), "<TIMESTAMP>"),
    # UUIDs
    (re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I), "<UUID>"),
    # File paths  (/foo/bar.py, C:\foo\bar.py, etc.)
    (re.compile(r"/\S+\.\w+"), "<PATH>"),
    # Hex addresses
    (re.compile(r"0x[0-9a-fA-F]+"), "<HEX>"),
    # Line numbers
    (re.compile(r"line \d+", re.I), "line <N>"),
]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CausalFinding:
    """A root cause finding from trace analysis."""

    finding_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str = ""
    root_step_id: str = ""
    causal_chain: list[str] = field(default_factory=list)
    failure_category: str = ""
    confidence: float = 0.0
    explanation: str = ""
    affected_downstream: int = 0
    tool_name: str = ""
    error_signature: str = ""


# ---------------------------------------------------------------------------
# Attributor
# ---------------------------------------------------------------------------

class CausalAttributor:
    """Analyzes execution traces to find root causes of failures."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_trace(self, trace: ExecutionTrace) -> list[CausalFinding]:
        """Analyze a single trace, return all causal findings.

        Algorithm:
        1. Find all failed steps (status in error/timeout).
        2. For each failed step, walk up the causal chain (parent_id links).
        3. Identify the FIRST failure in the chain -- that's the root cause.
        4. Classify the failure category based on error_detail + tool_name.
        5. Count downstream affected steps.
        6. Generate confidence score based on chain clarity.
        """
        step_index: dict[str, TraceStep] = {s.step_id: s for s in trace.steps}
        children_map: dict[str, list[str]] = defaultdict(list)
        for s in trace.steps:
            if s.parent_id:
                children_map[s.parent_id].append(s.step_id)

        failed_steps = [
            s for s in trace.steps if s.status in ("error", "timeout")
        ]
        if not failed_steps:
            return []

        # Track which root-cause step_ids we already reported so we don't
        # duplicate findings for the same root.
        seen_roots: set[str] = set()
        findings: list[CausalFinding] = []

        for step in failed_steps:
            # Walk up parent chain, collecting failed ancestors
            chain: list[str] = [step.step_id]
            current = step
            while current.parent_id and current.parent_id in step_index:
                parent = step_index[current.parent_id]
                chain.append(parent.step_id)
                current = parent

            # Reverse so chain goes root -> ... -> leaf
            chain.reverse()

            # Find first failed step in chain (the root cause)
            root_cause: TraceStep | None = None
            root_idx = 0
            for idx, sid in enumerate(chain):
                s = step_index[sid]
                if s.status in ("error", "timeout"):
                    root_cause = s
                    root_idx = idx
                    break

            if root_cause is None:
                # Shouldn't happen, but guard anyway
                continue

            if root_cause.step_id in seen_roots:
                continue
            seen_roots.add(root_cause.step_id)

            # Causal chain from root cause to final failure
            causal_chain = chain[root_idx:]

            # Count all downstream steps affected by the root cause
            affected = self._count_downstream(root_cause.step_id, children_map)

            # Classify
            category = self.classify_failure(
                root_cause, step_index=step_index,
            )

            # Confidence
            confidence = self._compute_confidence(
                causal_chain, step_index, category,
            )

            error_text = getattr(root_cause, "error_detail", "") or ""
            explanation = self._build_explanation(
                root_cause, category, len(causal_chain), affected,
            )

            findings.append(
                CausalFinding(
                    trace_id=trace.trace_id,
                    root_step_id=root_cause.step_id,
                    causal_chain=causal_chain,
                    failure_category=category,
                    confidence=confidence,
                    explanation=explanation,
                    affected_downstream=affected,
                    tool_name=getattr(root_cause, "tool_name", "") or "",
                    error_signature=self.normalize_error(error_text),
                ),
            )

        return findings

    def analyze_traces(self, traces: list[ExecutionTrace]) -> list[CausalFinding]:
        """Analyze multiple traces, return all findings."""
        findings: list[CausalFinding] = []
        for trace in traces:
            findings.extend(self.analyze_trace(trace))
        return findings

    def classify_failure(
        self,
        step: TraceStep,
        *,
        step_index: dict[str, TraceStep] | None = None,
    ) -> str:
        """Classify a failed step into a failure category.

        Uses keyword heuristics on the error_detail string plus structural
        information (parent step, tool name).
        """
        error = (getattr(step, "error_detail", "") or "").lower()
        tool = (getattr(step, "tool_name", "") or "").lower()

        # Ordered heuristics -- first match wins (except cascade which
        # needs parent inspection).

        if any(kw in error for kw in ("timeout", "timed out", "deadline")):
            return "timeout"

        if any(kw in error for kw in ("blocked", "denied", "gatekeeper")):
            return "permission_denied"

        if any(kw in error for kw in ("rate limit", "429", "too many")):
            return "rate_limited"

        if any(kw in error for kw in ("json", "parse", "decode", "syntax")):
            return "parse_error"

        if any(kw in error for kw in ("not found", "not registered", "unknown tool")):
            return "tool_unavailable"

        if any(kw in error for kw in ("parameter", "argument", "missing", "invalid", "required")):
            return "bad_parameters"

        if any(kw in error for kw in ("incorrect", "wrong", "inaccurate")) or tool in (
            "planner",
            "formulate",
            "formulate_response",
        ):
            return "hallucination"

        # Structural checks requiring step_index
        if step_index:
            parent_id = getattr(step, "parent_id", None)
            if parent_id and parent_id in step_index:
                parent = step_index[parent_id]
                parent_tool = (getattr(parent, "tool_name", "") or "").lower()

                # If parent also failed => cascade
                if parent.status in ("error", "timeout"):
                    return "cascade_failure"

                # If parent was a planner decision => wrong tool choice
                if parent_tool in ("planner", "plan", "formulate", "formulate_response"):
                    return "wrong_tool_choice"

        return "missing_context"

    def normalize_error(self, error: str) -> str:
        """Normalize error string for grouping.

        Strips file paths, timestamps, UUIDs, hex addresses, and line numbers
        so that structurally identical errors hash to the same signature.
        """
        result = error
        for pattern, replacement in _NORM_PATTERNS:
            result = pattern.sub(replacement, result)
        # Collapse whitespace
        result = re.sub(r"\s+", " ", result).strip()
        return result

    def aggregate_findings(self, findings: list[CausalFinding]) -> list[dict]:
        """Group findings by (failure_category, tool_name, error_signature).

        Returns list of dicts sorted descending by priority
        (count * avg_confidence).
        """
        groups: dict[
            tuple[str, str, str],
            list[CausalFinding],
        ] = defaultdict(list)

        for f in findings:
            key = (f.failure_category, f.tool_name, f.error_signature)
            groups[key].append(f)

        results: list[dict] = []
        for (category, tool, sig), group in groups.items():
            count = len(group)
            avg_conf = sum(f.confidence for f in group) / count
            # Pick explanation from highest-confidence finding
            best = max(group, key=lambda f: f.confidence)
            results.append(
                {
                    "failure_category": category,
                    "tool_name": tool,
                    "error_signature": sig,
                    "count": count,
                    "avg_confidence": round(avg_conf, 4),
                    "priority": round(count * avg_conf, 4),
                    "trace_ids": list({f.trace_id for f in group}),
                    "explanation": best.explanation,
                },
            )

        results.sort(key=lambda r: r["priority"], reverse=True)
        return results

    def get_improvement_targets(
        self,
        findings: list[CausalFinding],
        min_count: int = 2,
        min_confidence: float = 0.5,
    ) -> list[dict]:
        """Top improvement targets from aggregated findings, filtered by thresholds."""
        aggregated = self.aggregate_findings(findings)
        return [
            entry
            for entry in aggregated
            if entry["count"] >= min_count and entry["avg_confidence"] >= min_confidence
        ]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _count_downstream(
        step_id: str,
        children_map: dict[str, list[str]],
    ) -> int:
        """BFS count of all transitive children of *step_id*."""
        count = 0
        queue = list(children_map.get(step_id, []))
        while queue:
            child = queue.pop(0)
            count += 1
            queue.extend(children_map.get(child, []))
        return count

    @staticmethod
    def _compute_confidence(
        causal_chain: list[str],
        step_index: dict[str, TraceStep],
        category: str,
    ) -> float:
        """Heuristic confidence score for a finding.

        - Clear single-step failure            -> 0.9
        - Chain with one identifiable root      -> 0.7
        - Ambiguous chain (multiple failures)   -> 0.5
        - Cascade failures                      -> 0.4
        """
        if category == "cascade_failure":
            return 0.4

        chain_len = len(causal_chain)
        if chain_len == 1:
            return 0.9

        # Count how many steps in the chain are themselves failed
        failed_in_chain = sum(
            1
            for sid in causal_chain
            if sid in step_index and step_index[sid].status in ("error", "timeout")
        )

        if failed_in_chain == 1:
            return 0.7

        # Multiple failures in chain -> ambiguous
        return 0.5

    @staticmethod
    def _build_explanation(
        root: TraceStep,
        category: str,
        chain_length: int,
        affected: int,
    ) -> str:
        """Build a human-readable explanation for a finding."""
        tool = getattr(root, "tool_name", "unknown") or "unknown"
        error = getattr(root, "error_detail", "") or "unknown error"
        cat_desc = FAILURE_CATEGORIES.get(category, category)

        parts = [
            f"Root cause: {cat_desc}.",
            f"Tool '{tool}' failed with: {error}.",
        ]
        if chain_length > 1:
            parts.append(
                f"Failure propagated through {chain_length} steps in the causal chain.",
            )
        if affected > 0:
            parts.append(f"{affected} downstream step(s) were affected.")

        return " ".join(parts)
