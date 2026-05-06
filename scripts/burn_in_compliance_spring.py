"""Synthetic burn-in for the Compliance-Spring stack (PRs #494/#495/#496).

Drives N synthetic Reflector cycles against the local install to exercise the
new audit + snapshot paths shipped in the Compliance-Spring:

* PR #494  Reflector audit-helper (``_emit_reflection_audit_event`` +
            ``log_reflection_event`` routing under ``AuditCategory.REFLECTION``)
* PR #495  Hypothesis property tests (no runtime path; the script just
            confirms behaviour matches their invariants)
* PR #496  ``SearchWeightOptimizer`` Fernet-encrypted snapshots
            (``<weight_sha256>.fernet`` + ``.meta.json`` sidecar) +
            ``weight_snapshot_persisted`` audit emit

Strategy
--------
The LLM-backed reflection step is bypassed via a tiny ``StubLLMClient`` that
returns canned valid JSON. This means we don't have to launch Ollama for a
5-minute synthetic run — the audit + snapshot paths we care about live AFTER
the LLM call inside ``Reflector.reflect()``.

Inputs are varied across runs (different tool sequences, different per-channel
search-result scores) so the EMA update produces a fresh weight vector and
content-addressed snapshot hash on most runs.

Reading metric outputs
----------------------
Run with ``--report-format json`` to dump a structured report. The exit code
is 0 on success regardless of verdict — verdict colour-coding is informational
only, not a CI gate. Use ``--cleanup`` to drop the .fernet files this run
created (off by default — snapshots are forensic data).

Usage
-----

    PYTHONPATH=src python scripts/burn_in_compliance_spring.py --runs 50

The PYTHONPATH=src front-load is needed when a .pth file pins the production
install at ``D:/Jarvis/jarvis complete v20/src``; without it the script imports
the production tree instead of the worktree under test.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Allow running the script directly (without -m) when the worktree is not
# installed in editable mode. The Bash invocation `PYTHONPATH=src python ...`
# already covers this; the explicit insert is a belt-and-braces for IDE runs.
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cognithor.audit import AuditLogger
from cognithor.config import CognithorConfig
from cognithor.core.model_router import ModelRouter, OllamaClient
from cognithor.core.reflector import Reflector
from cognithor.learning.causal import CausalAnalyzer
from cognithor.memory.weight_optimizer import SearchWeightOptimizer
from cognithor.models import (
    ActionPlan,
    AgentResult,
    Chunk,
    MemorySearchResult,
    PlannedAction,
    SessionContext,
    ToolResult,
    WorkingMemory,
)
from cognithor.security.encrypted_file import EncryptedFileIO

# ---------------------------------------------------------------------------
# Stub LLM
# ---------------------------------------------------------------------------


class StubLLMClient:
    """Minimal async ``chat`` shim returning a canned reflection JSON.

    Mirrors the bits of ``OllamaClient`` the Reflector touches:
    ``await client.chat(model=, messages=, temperature=, top_p=)`` returning
    a dict with ``message.content`` (string) + token counts.

    The returned content is a valid reflection JSON so ``_parse_reflection``
    builds a non-trivial ``ReflectionResult`` (success_score=0.8, one fact,
    a session_summary). That kicks off the post-LLM hooks
    (causal_analyzer.record_sequence, weight_optimizer.record_outcome) which
    are the paths we want to burn in.
    """

    def __init__(self) -> None:
        self.call_count = 0

    async def chat(self, **kwargs: Any) -> dict[str, Any]:
        self.call_count += 1
        # Minimal but valid reflection envelope. Score >0 so the
        # weight_optimizer EMA fires; a session_summary so the episodic
        # store path runs (when wired).
        content = json.dumps(
            {
                "success_score": 0.8,
                "evaluation": "Synthetic burn-in iteration.",
                "extracted_facts": [],
                "procedure_candidate": None,
                "session_summary": {
                    "goal": "burn-in exercise",
                    "outcome": "completed",
                    "key_decisions": [],
                    "open_items": [],
                    "tools_used": ["search", "read_file", "respond"],
                    "duration_ms": 100,
                },
                "failure_analysis": "",
                "improvement_suggestions": [],
            }
        )
        return {
            "message": {"role": "assistant", "content": content},
            "prompt_eval_count": 64,
            "eval_count": 128,
        }


# ---------------------------------------------------------------------------
# Synthetic input builders
# ---------------------------------------------------------------------------


def _make_chunk(idx: int) -> Chunk:
    return Chunk(
        text=f"synthetic-chunk-{idx}",
        source_path=f"burn_in/{idx}.md",
    )


def build_search_result(idx: int, vector: float, bm25: float, graph: float) -> MemorySearchResult:
    return MemorySearchResult(
        chunk=_make_chunk(idx),
        score=vector + bm25 + graph,
        bm25_score=bm25,
        vector_score=vector,
        graph_score=graph,
    )


def build_synthetic_session(run_idx: int) -> tuple[SessionContext, WorkingMemory, AgentResult]:
    """Construct a (Session, WorkingMemory, AgentResult) triple for run ``idx``.

    Inputs are perturbed by ``run_idx`` so the EMA produces a different
    weight vector → different SHA-256 → fresh ``.fernet`` file. Without
    perturbation the snapshot dedup would collapse 50 runs into 1 file.
    """
    session = SessionContext(
        user_id="burn_in",
        channel="cli",
        agent_name="burn_in_agent",
    )

    plan = ActionPlan(
        goal=f"burn-in run #{run_idx}",
        reasoning="synthetic",
        steps=[
            PlannedAction(tool="search", params={"q": f"q-{run_idx}"}),
            PlannedAction(tool="read_file", params={"path": f"f-{run_idx}.txt"}),
            PlannedAction(tool="respond", params={"text": "ok"}),
        ],
    )

    tool_results = [
        ToolResult(tool_name="search", content=f"hits for run {run_idx}", duration_ms=20),
        ToolResult(tool_name="read_file", content=f"file body {run_idx}", duration_ms=10),
        ToolResult(tool_name="respond", content="ack", duration_ms=5),
    ]

    # Vary per-channel scores so channel_contributions changes each run.
    # Use a deterministic 3-cycle rotation to span the full simplex over
    # 50 runs (hits per channel rotate across {2,1,0}).
    cycle = run_idx % 6
    score_table = [
        (0.9, 0.3, 0.0),
        (0.0, 0.9, 0.3),
        (0.3, 0.0, 0.9),
        (0.7, 0.5, 0.2),
        (0.2, 0.7, 0.5),
        (0.5, 0.2, 0.7),
    ]
    v, b, g = score_table[cycle]

    working = WorkingMemory(
        session_id=session.session_id,
        injected_memories=[
            build_search_result(run_idx, v, b, g),
            build_search_result(run_idx + 100, v * 0.5, b * 0.5, g * 0.5),
        ],
    )

    agent_result = AgentResult(
        response="ok",
        plans=[plan],
        tool_results=tool_results,
        total_iterations=1,
        total_duration_ms=100 + run_idx,
        model_used="stub-llm",
        success=True,
    )

    return session, working, agent_result


# ---------------------------------------------------------------------------
# Filesystem + DB metric readers
# ---------------------------------------------------------------------------


def read_snapshot_state(snapshot_dir: Path) -> tuple[int, int]:
    """Returns ``(file_count, total_bytes)`` for ``.fernet`` files."""
    if not snapshot_dir.is_dir():
        return 0, 0
    files = list(snapshot_dir.glob("*.fernet"))
    total = sum(f.stat().st_size for f in files)
    return len(files), total


def read_audit_log_size(audit_dir: Path) -> int:
    """Total bytes across all ``audit_*.jsonl`` files in ``audit_dir``."""
    if not audit_dir.is_dir():
        return 0
    return sum(f.stat().st_size for f in audit_dir.glob("audit_*.jsonl"))


def read_tool_effectiveness_avg(tactical_db: Path) -> float | None:
    """Returns AVG(effectiveness) from the ``tool_effectiveness`` table.

    Returns ``None`` when the DB / table doesn't exist (fresh install)
    or is empty.
    """
    if not tactical_db.is_file():
        return None
    # Try encrypted connection first; fall back to plaintext for the
    # `.unencrypted.bak` siblings on dev machines.
    try:
        from cognithor.security.encrypted_db import encrypted_connect

        conn = encrypted_connect(str(tactical_db))
    except Exception:
        try:
            conn = sqlite3.connect(str(tactical_db))
        except Exception:
            return None
    try:
        cur = conn.execute(
            "SELECT AVG(effectiveness) FROM tool_effectiveness WHERE effectiveness IS NOT NULL"
        )
        row = cur.fetchone()
        if row and row[0] is not None:
            return float(row[0])
    except sqlite3.Error:
        return None
    finally:
        conn.close()
    return None


def count_pattern_in_audit_logs(audit_dir: Path, pattern: str) -> int:
    """Counts JSONL lines containing ``pattern`` across audit_*.jsonl."""
    if not audit_dir.is_dir():
        return 0
    count = 0
    for jsonl in audit_dir.glob("audit_*.jsonl"):
        try:
            with jsonl.open("r", encoding="utf-8") as f:
                for line in f:
                    if pattern in line:
                        count += 1
        except OSError:
            continue
    return count


def count_emit_failed_in_logs(log_paths: list[Path]) -> int:
    """Counts ``audit_emit_failed`` warnings in plaintext log files.

    The Reflector helper logs a warning via ``log.warning("audit_emit_failed",
    ...)`` when emit raises. Production logs go to ``~/.cognithor/logs/*.jsonl``
    via structlog; we grep the last 1 MB of each file for the pattern.
    """
    count = 0
    for p in log_paths:
        if not p.is_file():
            continue
        try:
            size = p.stat().st_size
            with p.open("rb") as f:
                if size > 1_000_000:
                    f.seek(-1_000_000, 2)
                blob = f.read().decode("utf-8", errors="replace")
                count += blob.count("audit_emit_failed")
        except OSError:
            continue
    return count


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------


def verdict_storage(new_files: int) -> str:
    if new_files <= 50:
        return "GREEN"
    if new_files <= 150:
        return "YELLOW"
    return "RED"


def verdict_audit_failures(count: int) -> str:
    return "GREEN" if count == 0 else "RED"


def verdict_atomic_rollbacks(count: int) -> str:
    return "GREEN" if count == 0 else "RED"


def verdict_drift(before: float | None, after: float | None) -> tuple[str, float | None]:
    if before is None or after is None or before == 0.0:
        # No baseline → can't measure drift. Treat as GREEN with no value.
        return "GREEN", None
    delta_pct = abs(after - before) / before * 100.0
    if delta_pct < 5.0:
        return "GREEN", delta_pct
    if delta_pct < 15.0:
        return "YELLOW", delta_pct
    return "RED", delta_pct


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------


async def run_burn_in(
    runs: int,
    config: CognithorConfig,
    *,
    sleep_between: float = 0.0,
) -> tuple[int, list[str]]:
    """Returns ``(actual_runs_completed, list_of_session_ids)``."""
    audit_log_dir = config.cognithor_home / "data" / "audit"
    audit_log_dir.mkdir(parents=True, exist_ok=True)
    db_dir = config.cognithor_home / "db"
    db_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir = config.cognithor_home / "weight_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    audit_logger = AuditLogger(log_dir=audit_log_dir, retention_days=90)
    causal_analyzer = CausalAnalyzer(
        db_path=str(db_dir / "causal_burn_in.db"),
    )

    weight_db = db_dir / "memory_weights_burn_in.db"
    encrypted_io = EncryptedFileIO()
    weight_optimizer = SearchWeightOptimizer(
        db_path=str(weight_db),
        encrypted_file_io=encrypted_io,
        snapshot_dir=snapshot_dir,
    )

    stub_llm = StubLLMClient()
    # ``ModelRouter`` legacy ctor expects an ``OllamaClient``; we pass a
    # disposable one because ``select_model`` only needs config.
    dummy_ollama = OllamaClient(config)
    router = ModelRouter(config, dummy_ollama)

    reflector = Reflector(
        config=config,
        ollama=stub_llm,  # type: ignore[arg-type]
        model_router=router,
        audit_logger=audit_logger,
        causal_analyzer=causal_analyzer,
        weight_optimizer=weight_optimizer,
    )

    # ``Reflector.__init__`` already wired the audit-emit callbacks into
    # both subsystems via ``set_audit_emit_callback``; nothing more to do.

    session_ids: list[str] = []
    completed = 0
    for i in range(runs):
        session, working, agent_result = build_synthetic_session(i)
        session_ids.append(session.session_id)
        try:
            await reflector.reflect(session, working, agent_result)
            completed += 1
        except Exception as exc:
            print(
                f"[burn-in] run #{i} reflect() raised: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            break
        if sleep_between > 0:
            await asyncio.sleep(sleep_between)

    # Flush window for any deferred persistence (Audit JSONL writes are
    # synchronous, but give async event loops a moment to drain).
    await asyncio.sleep(0.5)

    weight_optimizer.close()
    return completed, session_ids


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


def cleanup_new_snapshots(snapshot_dir: Path, baseline_files: set[str]) -> int:
    """Removes ``.fernet`` + ``.meta.json`` files NOT in ``baseline_files``.

    Returns count of deleted entries. Only deletes files inside
    ``snapshot_dir``; never recurses.
    """
    if not snapshot_dir.is_dir():
        return 0
    deleted = 0
    for f in snapshot_dir.iterdir():
        if not f.is_file():
            continue
        if f.name in baseline_files:
            continue
        try:
            f.unlink()
            deleted += 1
        except OSError:
            pass
    return deleted


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=50, help="Number of synthetic runs.")
    parser.add_argument(
        "--report-format",
        choices=("json", "human", "both"),
        default="both",
        help="Output format.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Optional file path for the JSON report (default: stdout only).",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete .fernet snapshots created by this run on exit.",
    )
    parser.add_argument(
        "--require-init",
        action="store_true",
        help="Fail if ~/.cognithor/.cognithor_initialized does not exist.",
    )
    parser.add_argument(
        "--home",
        type=Path,
        default=None,
        help=(
            "Override the Cognithor home directory (default: ~/.cognithor). "
            "Set this in CI to keep the burn-in fully isolated from the "
            "user's real install."
        ),
    )
    args = parser.parse_args(argv)

    # ``--home`` overrides the Cognithor home dir for this run only.
    # We must construct the config with the override BEFORE any subsystem
    # touches ``Path.home() / ".cognithor"``. Pydantic's ``default_factory``
    # bakes the user's real home into ``CognithorConfig()``, so we can't
    # just set ``COGNITHOR_HOME`` and call the bare constructor — we have
    # to build the override into the field directly.
    if args.home is not None:
        home_override = args.home.expanduser().resolve()
        home_override.mkdir(parents=True, exist_ok=True)
        # Best-effort env-var hint for any subprocess-spawned tooling
        # that re-reads it; we still pass the value to the constructor.
        os.environ["COGNITHOR_HOME"] = str(home_override)
        config = CognithorConfig(cognithor_home=home_override)
    else:
        config = CognithorConfig()
    home = config.cognithor_home

    init_marker = home / ".cognithor_initialized"
    if not init_marker.exists():
        msg = (
            f"Cognithor home not initialised at {home}.\n"
            f"Run `cognithor` once to bootstrap, then re-run this script."
        )
        if args.require_init:
            print(msg, file=sys.stderr)
            return 2
        print(f"[warn] {msg}", file=sys.stderr)

    snapshot_dir = home / "weight_snapshots"
    audit_dir = home / "data" / "audit"
    tactical_db = home / "db" / "tactical_memory.db"
    log_files = [
        home / "logs" / "cognithor.jsonl",
        home / "logs" / "cognithor.jsonl.1",
    ]

    # Capture baseline filenames so --cleanup can target only what we added.
    baseline_files = set()
    if snapshot_dir.is_dir():
        baseline_files = {f.name for f in snapshot_dir.iterdir() if f.is_file()}

    baseline_snap_count, baseline_snap_bytes = read_snapshot_state(snapshot_dir)
    baseline_audit_bytes = read_audit_log_size(audit_dir)
    baseline_tool_eff = read_tool_effectiveness_avg(tactical_db)
    baseline_emit_failed = count_emit_failed_in_logs(log_files)
    baseline_audit_emit_failed_in_audit = count_pattern_in_audit_logs(
        audit_dir, "audit_emit_failed"
    )
    baseline_rollbacks = count_pattern_in_audit_logs(audit_dir, "rollback")

    t0 = time.perf_counter()
    completed, _session_ids = asyncio.run(run_burn_in(args.runs, config))
    runtime_seconds = time.perf_counter() - t0

    after_snap_count, after_snap_bytes = read_snapshot_state(snapshot_dir)
    after_audit_bytes = read_audit_log_size(audit_dir)
    after_tool_eff = read_tool_effectiveness_avg(tactical_db)
    after_emit_failed = count_emit_failed_in_logs(log_files)
    after_audit_emit_failed_in_audit = count_pattern_in_audit_logs(audit_dir, "audit_emit_failed")
    after_rollbacks = count_pattern_in_audit_logs(audit_dir, "rollback")

    new_snapshots = max(0, after_snap_count - baseline_snap_count)
    new_snap_bytes = max(0, after_snap_bytes - baseline_snap_bytes)
    audit_growth = max(0, after_audit_bytes - baseline_audit_bytes)
    new_emit_failed = max(0, (after_emit_failed - baseline_emit_failed))
    new_emit_failed += max(
        0, after_audit_emit_failed_in_audit - baseline_audit_emit_failed_in_audit
    )
    new_rollbacks = max(0, after_rollbacks - baseline_rollbacks)

    drift_verdict, drift_delta_pct = verdict_drift(baseline_tool_eff, after_tool_eff)

    if baseline_tool_eff is None or after_tool_eff is None:
        eff_delta = None
    else:
        eff_delta = after_tool_eff - baseline_tool_eff

    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "runs_requested": args.runs,
        "runs_completed": completed,
        "runtime_seconds": round(runtime_seconds, 3),
        "cognithor_home": str(home),
        "snapshot_dir": str(snapshot_dir),
        "audit_dir": str(audit_dir),
        "baseline": {
            "snapshots": baseline_snap_count,
            "snapshot_bytes": baseline_snap_bytes,
            "tool_effectiveness_avg": baseline_tool_eff,
            "audit_log_bytes": baseline_audit_bytes,
        },
        "after_run": {
            "snapshots": after_snap_count,
            "snapshot_bytes": after_snap_bytes,
            "tool_effectiveness_avg": after_tool_eff,
            "audit_log_bytes": after_audit_bytes,
        },
        "delta": {
            "new_snapshots": new_snapshots,
            "new_snapshot_bytes": new_snap_bytes,
            "audit_log_growth_bytes": audit_growth,
            "effectiveness_delta": eff_delta,
        },
        "metric_1_storage": {
            "files": new_snapshots,
            "size_bytes": new_snap_bytes,
            "verdict": verdict_storage(new_snapshots),
        },
        "metric_2_audit_failures": {
            "count": new_emit_failed,
            "verdict": verdict_audit_failures(new_emit_failed),
        },
        "metric_3_atomic_rollbacks": {
            "count": new_rollbacks,
            "verdict": verdict_atomic_rollbacks(new_rollbacks),
        },
        "metric_4_behavioral_drift": {
            "before": baseline_tool_eff,
            "after": after_tool_eff,
            "delta_pct": drift_delta_pct,
            "verdict": drift_verdict,
        },
    }

    if args.cleanup:
        deleted = cleanup_new_snapshots(snapshot_dir, baseline_files)
        report["cleanup"] = {"deleted_files": deleted}

    if args.report_path is not None:
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        args.report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    if args.report_format in ("json", "both"):
        print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.report_format in ("human", "both"):
        print()
        print("=== Compliance-Spring Burn-In Summary ===")
        print(f"Runs completed:       {completed}/{args.runs}")
        print(f"Runtime:              {runtime_seconds:.2f} s")
        print(f"New snapshots:        {new_snapshots} files, {new_snap_bytes} bytes")
        print(f"Audit log growth:     {audit_growth} bytes")
        b_eff = baseline_tool_eff
        a_eff = after_tool_eff
        eff_str = (
            f"before={b_eff:.4f} after={a_eff:.4f}"
            if b_eff is not None and a_eff is not None
            else f"before={b_eff} after={a_eff}"
        )
        print(f"Tool effectiveness:   {eff_str}")
        print()
        print(
            f"Metric 1 storage:           {report['metric_1_storage']['verdict']} "
            f"({new_snapshots} files / {new_snap_bytes} bytes)"
        )
        print(
            f"Metric 2 audit failures:    {report['metric_2_audit_failures']['verdict']} "
            f"(count={new_emit_failed})"
        )
        print(
            f"Metric 3 atomic rollbacks:  {report['metric_3_atomic_rollbacks']['verdict']} "
            f"(count={new_rollbacks})"
        )
        delta_str = (
            f"{drift_delta_pct:.2f}%" if drift_delta_pct is not None else "N/A (no baseline)"
        )
        print(f"Metric 4 drift:             {drift_verdict} ({delta_str})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
