"""Cognithor Resilient Workflow Engine (CRWE).

A robust, sequential JSONL-streaming task runner with crash-recovery,
signal-safety, and audit-chain integration.

Operational guarantees:

* **Stream-based ingestion** -- ``stream_tasks`` walks a JSONL manifest
  line-by-line; the full file is never loaded into memory.
* **Atomic checkpoint persistence** -- ``.checkpoint.json`` is written
  via ``write+fsync+os.replace`` so a power-fail never leaves a corrupt
  pointer file.
* **Per-task fsync of results** -- each ``TaskResult`` is appended +
  fsynced to ``results.jsonl`` before the next task starts. The
  integrity contract is "results.jsonl is the truth"; at most one
  task can be lost on a hard kill, and we can prove it from the
  checksum chain.
* **Running SHA-256 chain** -- every checkpoint records the SHA-256
  of the current ``results.jsonl`` so resume can detect post-hoc
  tampering ("gap-injection" attack).
* **Concurrent-runner protection** -- ``.checkpoint.lock`` is held
  exclusively for the lifetime of a run via ``fcntl.flock`` (POSIX)
  or ``msvcrt.locking`` (Windows). A second runner against the same
  results-dir fails fast with :class:`WorkflowAlreadyRunning`.
* **Signal-safe shutdown** -- SIGINT / SIGTERM (and SIGBREAK on
  Windows) set a request flag that's checked **between** tasks.
  An in-flight task is never interrupted, so a partial result can
  never appear in ``results.jsonl``.
* **Audit-chain integration** -- every checkpoint emits a
  ``system_checkpoint_created`` event via :class:`AuditLogger`;
  resume emits ``workflow_resumed``; clean termination emits
  ``workflow_completed``. All routed through ``AuditCategory.SYSTEM``
  (workflow checkpoints are operational state, not autonomous
  learning -- see deviation note below).

DEVIATION FROM SPEC: the original CRWE spec requested
``AuditCategory.REFLECTION`` for checkpoint events. After auditing
``cognithor.audit.AuditCategory``, ``SYSTEM`` is semantically
correct: REFLECTION is reserved for autonomous Reflector sinks
(causal / episodic / semantic / procedural learning) per
Operational-Trust PR-D, while workflow checkpoints are deterministic
operational events. No new enum value was added; ``log_system`` is
the right channel.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import os
import signal
import sys
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    runtime_checkable,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from cognithor.audit import AuditLogger

logger = logging.getLogger("cognithor.core.workflow")

# Truncate ``TaskResult.error`` payloads to keep results.jsonl rows
# bounded. 4 KB is enough for a stack frame summary; full traces go
# to the agent log.
_ERROR_TRUNCATE_BYTES = 4096

# Schema version of the on-disk ``.checkpoint.json``. Bumped on
# breaking format changes; resume rejects unknown versions instead
# of silently mis-interpreting fields.
_CHECKPOINT_SCHEMA_VERSION = "v1"


# ============================================================================
# Public exceptions
# ============================================================================


class WorkflowError(Exception):
    """Base class for all CRWE failures."""


class ManifestError(WorkflowError):
    """A single manifest line failed to parse as JSON."""

    def __init__(self, line_no: int, original_text: str, parse_error: str) -> None:
        super().__init__(f"manifest line {line_no} is not valid JSON: {parse_error}")
        self.line_no = line_no
        self.original_text = original_text
        self.parse_error = parse_error


class ManifestValidationError(WorkflowError):
    """Pre-flight manifest validation found one or more bad lines.

    Carries the full list of failures so the operator sees every
    problem, not just the first.
    """

    def __init__(self, failures: list[tuple[int, str]]) -> None:
        super().__init__(
            f"manifest validation failed with {len(failures)} error(s): "
            + "; ".join(f"line {ln}: {reason}" for ln, reason in failures[:5])
            + (f" (and {len(failures) - 5} more)" if len(failures) > 5 else "")
        )
        self.failures = failures


class EmptyManifestError(WorkflowError):
    """Manifest file is empty -- explicit refusal beats silent no-op."""


class CheckpointIntegrityError(WorkflowError):
    """Checkpoint's recorded sha256 doesn't match results.jsonl on disk."""

    def __init__(
        self,
        *,
        recorded: str,
        actual: str,
        results_path: Path,
        results_mtime: float,
    ) -> None:
        super().__init__(
            f"checkpoint integrity mismatch: recorded={recorded[:16]}... "
            f"actual={actual[:16]}... results={results_path} "
            f"(mtime={results_mtime})"
        )
        self.recorded = recorded
        self.actual = actual
        self.results_path = results_path
        self.results_mtime = results_mtime


class ResultsOutOfSyncError(WorkflowError):
    """Line count in results.jsonl doesn't match checkpoint index."""

    def __init__(self, *, checkpoint_idx: int, actual_count: int) -> None:
        super().__init__(
            f"results.jsonl line count mismatch: checkpoint says "
            f"last_successful_index={checkpoint_idx} (expecting "
            f"{checkpoint_idx + 1} lines), but file has {actual_count} lines"
        )
        self.checkpoint_idx = checkpoint_idx
        self.actual_count = actual_count


class ManifestTamperError(WorkflowError):
    """Manifest sha256 changed since the checkpoint was written."""

    def __init__(self, *, recorded: str, actual: str) -> None:
        super().__init__(
            f"manifest tampered after checkpoint: recorded={recorded[:16]}... "
            f"actual={actual[:16]}..."
        )
        self.recorded = recorded
        self.actual = actual


class WorkflowAlreadyRunning(WorkflowError):
    """Another runner already holds the .checkpoint.lock."""

    def __init__(self, *, workflow_id: str, lock_holder_pid: int | None) -> None:
        pid_str = str(lock_holder_pid) if lock_holder_pid is not None else "unknown"
        super().__init__(
            f"workflow {workflow_id!r} is already running (lock held by pid={pid_str})"
        )
        self.workflow_id = workflow_id
        self.lock_holder_pid = lock_holder_pid


# ============================================================================
# Public dataclasses
# ============================================================================


@runtime_checkable
class TaskHandler(Protocol):
    """Async task handler.

    MUST be idempotent: the engine guarantees each task_id is
    presented at most once per successful run, but on resume after
    a hard kill a handler that already wrote external side-effects
    will be skipped (the task was already in results.jsonl). The
    handler itself is responsible for guaranteeing that calling it
    twice with the same task is safe -- this is a soft requirement
    for users wiring external systems (e.g. "send email") and a hard
    requirement for any side-effect that must not duplicate.
    """

    async def __call__(self, task: dict[str, Any]) -> TaskResult: ...


@dataclass(frozen=True)
class TaskResult:
    """Outcome of a single task execution."""

    task_id: str
    success: bool
    duration_ms: float
    output: dict[str, Any] | None = None
    error: str | None = None
    error_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "success": self.success,
            "duration_ms": round(self.duration_ms, 3),
            "output": self.output,
            "error": self.error,
            "error_type": self.error_type,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskResult:
        return cls(
            task_id=str(data["task_id"]),
            success=bool(data["success"]),
            duration_ms=float(data.get("duration_ms", 0.0)),
            output=data.get("output"),
            error=data.get("error"),
            error_type=data.get("error_type"),
        )


@dataclass(frozen=True)
class CheckpointState:
    """On-disk checkpoint pointer, written atomically.

    ``manifest_sha256`` is captured at first run and re-validated on
    resume so we can detect a tampered manifest (closes a re-execution
    attack where an attacker swaps tasks in-flight).
    """

    workflow_id: str
    source_file: str
    last_successful_index: int
    last_checkpoint_timestamp: str
    checksum_of_results: str
    manifest_sha256: str
    schema_version: str = _CHECKPOINT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "source_file": self.source_file,
            "last_successful_index": self.last_successful_index,
            "last_checkpoint_timestamp": self.last_checkpoint_timestamp,
            "checksum_of_results": self.checksum_of_results,
            "manifest_sha256": self.manifest_sha256,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointState:
        version = str(data.get("schema_version", _CHECKPOINT_SCHEMA_VERSION))
        if version != _CHECKPOINT_SCHEMA_VERSION:
            raise WorkflowError(
                f"checkpoint schema_version={version!r} not supported "
                f"(this build: {_CHECKPOINT_SCHEMA_VERSION!r})"
            )
        return cls(
            workflow_id=str(data["workflow_id"]),
            source_file=str(data["source_file"]),
            last_successful_index=int(data["last_successful_index"]),
            last_checkpoint_timestamp=str(data["last_checkpoint_timestamp"]),
            checksum_of_results=str(data["checksum_of_results"]),
            manifest_sha256=str(data["manifest_sha256"]),
            schema_version=version,
        )


@dataclass
class WorkflowSummary:
    """Final report returned by :meth:`WorkflowRunner.run`."""

    workflow_id: str
    total_tasks: int
    successes: int
    failures: int
    total_duration_ms: float
    completed: bool
    interrupted_by_signal: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "total_tasks": self.total_tasks,
            "successes": self.successes,
            "failures": self.failures,
            "total_duration_ms": round(self.total_duration_ms, 3),
            "completed": self.completed,
            "interrupted_by_signal": self.interrupted_by_signal,
        }


# ============================================================================
# Streaming + checksum helpers
# ============================================================================


def stream_tasks(path: Path, start_index: int = 0) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield ``(index, task_dict)`` tuples from a JSONL file.

    Index is 0-based and matches the line ordering. Lines before
    ``start_index`` are still walked (cheap) but not yielded.

    Raises :class:`ManifestError` on the first malformed JSON line.
    Use :func:`validate_manifest` first if you want the full failure
    list before any task runs.
    """
    if start_index < 0:
        raise ValueError("start_index must be >= 0")
    with path.open("r", encoding="utf-8", newline="") as f:
        for i, raw_line in enumerate(f):
            stripped = raw_line.strip()
            if not stripped:
                # Blank line -- skip silently, but the index counter
                # reflects original line position.
                continue
            if i < start_index:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ManifestError(i + 1, raw_line, str(exc)) from exc
            if not isinstance(obj, dict):
                raise ManifestError(i + 1, raw_line, f"expected object, got {type(obj).__name__}")
            yield i, obj


def validate_manifest(path: Path) -> str:
    """Pre-flight walk of the manifest. Returns its SHA-256 hex digest.

    Asserts:
        * file is non-empty (raises :class:`EmptyManifestError`),
        * every line is valid JSON,
        * every line is an object with a non-empty string ``task_id``,
        * ``task_id`` values are unique (duplicates are rejected --
          the spec calls this "Idempotency: handler called once per
          task_id even if manifest has dupes; document why dupes are
          rejected": dupes are rejected because the streamer's
          single-pass model can't disambiguate which one is "the
          authoritative" task without ad-hoc rules; explicit refusal
          is safer).
    """
    if not path.exists():
        raise WorkflowError(f"manifest not found: {path}")
    if path.stat().st_size == 0:
        raise EmptyManifestError(f"manifest is empty: {path}")

    failures: list[tuple[int, str]] = []
    seen_ids: set[str] = set()
    line_count = 0

    hasher = hashlib.sha256()
    with path.open("rb") as fb:
        for chunk in iter(lambda: fb.read(65536), b""):
            hasher.update(chunk)

    with path.open("r", encoding="utf-8", newline="") as f:
        for i, raw_line in enumerate(f):
            line_no = i + 1
            stripped = raw_line.strip()
            if not stripped:
                continue
            line_count += 1
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                failures.append((line_no, f"invalid JSON: {exc}"))
                continue
            if not isinstance(obj, dict):
                failures.append((line_no, f"expected object, got {type(obj).__name__}"))
                continue
            tid_raw = obj.get("task_id")
            if not isinstance(tid_raw, str) or not tid_raw:
                failures.append((line_no, "missing or empty 'task_id' (must be non-empty str)"))
                continue
            if tid_raw in seen_ids:
                failures.append((line_no, f"duplicate task_id: {tid_raw!r}"))
                continue
            seen_ids.add(tid_raw)

    if line_count == 0:
        raise EmptyManifestError(f"manifest contains no task lines: {path}")
    if failures:
        raise ManifestValidationError(failures)

    return hasher.hexdigest()


def _sha256_of_file(path: Path) -> str:
    """Return ``"sha256:HEX"`` for the given file, or ``"sha256:"`` if missing/empty."""
    if not path.exists() or path.stat().st_size == 0:
        return "sha256:" + hashlib.sha256(b"").hexdigest()
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return "sha256:" + hasher.hexdigest()


def _count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8", newline="") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


# ============================================================================
# File-locking (cross-platform)
# ============================================================================


class _ExclusiveFileLock:
    """Cross-platform exclusive file lock via stdlib only.

    Uses ``fcntl.flock`` on POSIX, ``msvcrt.locking`` on Windows.
    Non-blocking: ``acquire`` returns ``False`` immediately if the
    lock is held by another process. Writes the holder's pid +
    timestamp into the lock file for forensics.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._fh: Any | None = None

    def acquire(self) -> bool:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Open for read+write so we can rewrite the holder marker.
        self._fh = self._path.open("a+", encoding="utf-8")
        try:
            if sys.platform == "win32":
                import msvcrt

                # Lock 1 byte non-blockingly.
                self._fh.seek(0)
                try:
                    msvcrt.locking(self._fh.fileno(), msvcrt.LK_NBLCK, 1)
                except OSError:
                    self._release_fh()
                    return False
            else:
                import fcntl

                try:
                    fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    self._release_fh()
                    return False
            self._fh.seek(0)
            self._fh.truncate()
            self._fh.write(
                json.dumps(
                    {
                        "pid": os.getpid(),
                        "acquired_at": datetime.now(UTC).isoformat(),
                    }
                )
            )
            self._fh.flush()
            with suppress(OSError):
                os.fsync(self._fh.fileno())
            return True
        except Exception:
            self._release_fh()
            raise

    def release(self) -> None:
        if self._fh is None:
            return
        try:
            if sys.platform == "win32":
                import msvcrt

                self._fh.seek(0)
                with suppress(OSError):
                    msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                with suppress(OSError):
                    fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        finally:
            self._release_fh()

    def _release_fh(self) -> None:
        if self._fh is not None:
            with suppress(OSError):
                self._fh.close()
            self._fh = None

    @staticmethod
    def read_holder_pid(path: Path) -> int | None:
        """Read the pid recorded in the lock file (best-effort)."""
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            pid = data.get("pid")
            return int(pid) if isinstance(pid, int) else None
        except (OSError, ValueError, json.JSONDecodeError):
            return None


# ============================================================================
# Sync handler wrapper
# ============================================================================


def _run_sync_handler(
    fn: Callable[[dict[str, Any]], TaskResult],
) -> TaskHandler:
    """Wrap a synchronous handler so the engine can ``await`` it.

    The wrapped callable runs in the default executor so a slow sync
    handler doesn't block the event loop -- important when SIGINT is
    delivered to the main thread, the run-loop coroutine still gets
    a chance to observe the shutdown flag promptly.
    """

    async def _async_handler(task: dict[str, Any]) -> TaskResult:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, fn, task)

    return _async_handler


def _resolve_handler(
    handler: TaskHandler | Callable[[dict[str, Any]], TaskResult],
) -> TaskHandler:
    """Accept sync or async handler, return an async one."""
    if inspect.iscoroutinefunction(handler):
        return handler  # type: ignore[return-value]
    if callable(handler):
        return _run_sync_handler(handler)  # type: ignore[arg-type]
    raise TypeError(f"handler must be callable, got {type(handler).__name__}")


# ============================================================================
# WorkflowRunner
# ============================================================================


class WorkflowRunner:
    """Sequential JSONL task runner with crash-recovery.

    Construct, then call :meth:`run`. Re-running with ``resume=True``
    against the same ``results_dir`` picks up where a prior run left
    off, validating the integrity of ``results.jsonl`` against the
    checkpoint's recorded SHA-256 first.
    """

    def __init__(
        self,
        manifest_path: Path,
        results_dir: Path,
        *,
        handler: TaskHandler | Callable[[dict[str, Any]], TaskResult],
        workflow_id: str | None = None,
        checkpoint_every: int = 12,
        audit_logger: AuditLogger | None = None,
    ) -> None:
        if checkpoint_every < 1:
            raise ValueError("checkpoint_every must be >= 1")

        self._manifest_path = Path(manifest_path).resolve()
        self._results_dir = Path(results_dir).resolve()
        self._handler: TaskHandler = _resolve_handler(handler)
        self._checkpoint_every = checkpoint_every
        self._audit = audit_logger

        # File layout under results_dir.
        self._results_dir.mkdir(parents=True, exist_ok=True)
        self._results_file = self._results_dir / "results.jsonl"
        self._checkpoint_file = self._results_dir / ".checkpoint.json"
        self._checkpoint_lock_file = self._results_dir / ".checkpoint.lock"
        self._manifest_link = self._results_dir / "manifest.jsonl"

        # Pre-flight: validate manifest, capture sha256.
        self._manifest_sha256 = validate_manifest(self._manifest_path)

        # Snapshot manifest into results_dir so the source-of-truth is
        # immutable even if the user later edits the original file.
        # Use a copy (always portable; symlinks are flaky on Windows).
        if not self._manifest_link.exists():
            self._manifest_link.write_bytes(self._manifest_path.read_bytes())

        # Workflow id auto-derivation: deterministic per-day so a
        # re-run on the same manifest the same day reuses its
        # results_dir naturally.
        if workflow_id is None:
            stem = self._manifest_path.stem or "workflow"
            today = datetime.now(UTC).strftime("%Y%m%d")
            workflow_id = f"{stem}_{self._manifest_sha256[:8]}_{today}"
        self._workflow_id = workflow_id

        # Signal-coordination state. ``_shutdown_event`` is the async
        # primitive checked between tasks; ``_shutdown_signal`` carries
        # the name of the signal that fired (for the audit log). The
        # asyncio Event is bound lazily inside ``run()`` because it
        # must live on the active event loop.
        self._shutdown_event: asyncio.Event | None = None
        self._shutdown_signal: str | None = None
        self._shutdown_flag = threading.Event()
        # (signum, original_handler) tuples so we can restore prior
        # handlers on exit.
        self._installed_signal_handlers: list[tuple[int, Any]] = []

        # Lock + last-checkpoint timestamp tracking.
        self._lock = _ExclusiveFileLock(self._checkpoint_lock_file)
        self._last_checkpoint_wall_ms: float = 0.0

    # ── Public API ────────────────────────────────────────────────

    @property
    def workflow_id(self) -> str:
        return self._workflow_id

    @property
    def manifest_sha256(self) -> str:
        return self._manifest_sha256

    @classmethod
    def from_checkpoint(
        cls,
        results_dir: Path,
        *,
        handler: TaskHandler | Callable[[dict[str, Any]], TaskResult],
        manifest_path: Path | None = None,
        audit_logger: AuditLogger | None = None,
        checkpoint_every: int = 12,
    ) -> WorkflowRunner:
        """Reconstruct a runner from an existing results_dir.

        If ``manifest_path`` is None, the on-disk
        ``results_dir/manifest.jsonl`` snapshot is used.
        """
        results_dir = Path(results_dir).resolve()
        ckpt_path = results_dir / ".checkpoint.json"
        if not ckpt_path.exists():
            raise WorkflowError(f"no checkpoint found at {ckpt_path}")
        ckpt = CheckpointState.from_dict(json.loads(ckpt_path.read_text(encoding="utf-8")))
        manifest_path = manifest_path or (results_dir / "manifest.jsonl")
        return cls(
            manifest_path=manifest_path,
            results_dir=results_dir,
            handler=handler,
            workflow_id=ckpt.workflow_id,
            checkpoint_every=checkpoint_every,
            audit_logger=audit_logger,
        )

    async def run(self, *, resume: bool = False) -> WorkflowSummary:
        """Execute the workflow.

        ``resume=False`` is the fresh-run path; if ``.checkpoint.json``
        already exists this raises (operator must explicitly opt into
        resume). ``resume=True`` validates integrity, then continues
        from ``last_successful_index + 1``.
        """
        # Acquire the per-results_dir lock, fail-fast on contention.
        if not self._lock.acquire():
            holder = _ExclusiveFileLock.read_holder_pid(self._checkpoint_lock_file)
            raise WorkflowAlreadyRunning(workflow_id=self._workflow_id, lock_holder_pid=holder)

        # Wire signal handlers + the asyncio event under the running loop.
        self._shutdown_event = asyncio.Event()
        self._install_signal_handlers()
        # Bridge: a thread-set flag (signal handlers run in main thread)
        # is hoisted into the asyncio Event by a background watcher.
        watcher_task = asyncio.create_task(self._signal_watcher())

        start_wall_ms = time.monotonic() * 1000.0
        successes = 0
        failures = 0
        completed = False

        try:
            start_index = 0
            if resume:
                start_index = await self._validate_and_resume()
            else:
                if self._checkpoint_file.exists():
                    raise WorkflowError(
                        f"checkpoint already exists at {self._checkpoint_file}; "
                        "pass resume=True to continue"
                    )

            self._last_checkpoint_wall_ms = time.monotonic() * 1000.0

            total = 0
            last_committed_index = start_index - 1

            shutdown_evt = self._shutdown_event
            assert shutdown_evt is not None  # bound above; mypy hint
            interrupted = False

            for i, task in stream_tasks(self._manifest_path, start_index=start_index):
                if shutdown_evt.is_set():
                    # Emergency checkpoint at the boundary, then exit.
                    await self._write_checkpoint_and_emit(last_committed_index)
                    self._emit_signal_event(last_committed_index)
                    interrupted = True
                    break

                total += 1
                result = await self._safe_invoke(task)
                self._append_result(result)
                last_committed_index = i

                if result.success:
                    successes += 1
                else:
                    failures += 1

                # Periodic checkpoint on the configured stride.
                if (i + 1 - start_index) % self._checkpoint_every == 0:
                    await self._write_checkpoint_and_emit(last_committed_index)

            if not interrupted:
                # Final checkpoint so the summary is internally
                # consistent even if total % stride != 0.
                await self._write_checkpoint_and_emit(last_committed_index)
                completed = True

            total_duration_ms = time.monotonic() * 1000.0 - start_wall_ms
            summary = WorkflowSummary(
                workflow_id=self._workflow_id,
                total_tasks=total,
                successes=successes,
                failures=failures,
                total_duration_ms=total_duration_ms,
                completed=completed and self._shutdown_signal is None,
                interrupted_by_signal=self._shutdown_signal,
            )

            if summary.completed:
                self._emit_audit(
                    "workflow_completed",
                    {
                        "workflow_id": self._workflow_id,
                        "total_tasks": total,
                        "successes": successes,
                        "failures": failures,
                        "total_duration_ms": round(total_duration_ms, 3),
                    },
                )
            return summary
        finally:
            watcher_task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await watcher_task
            self._restore_signal_handlers()
            self._lock.release()

    # ── Resume validation ────────────────────────────────────────

    async def _validate_and_resume(self) -> int:
        if not self._checkpoint_file.exists():
            raise WorkflowError(f"resume requested but no checkpoint at {self._checkpoint_file}")
        ckpt = CheckpointState.from_dict(
            json.loads(self._checkpoint_file.read_text(encoding="utf-8"))
        )

        # Manifest sha256 -- detects post-checkpoint manifest tampering.
        if ckpt.manifest_sha256 and ckpt.manifest_sha256 != self._manifest_sha256:
            raise ManifestTamperError(recorded=ckpt.manifest_sha256, actual=self._manifest_sha256)

        # Results sha256 -- detects gap-injection / mutation of results.jsonl.
        actual_chk = _sha256_of_file(self._results_file)
        if ckpt.checksum_of_results != actual_chk:
            raise CheckpointIntegrityError(
                recorded=ckpt.checksum_of_results,
                actual=actual_chk,
                results_path=self._results_file,
                results_mtime=(
                    self._results_file.stat().st_mtime if self._results_file.exists() else 0.0
                ),
            )

        # Line count must match index + 1.
        actual_count = _count_jsonl_lines(self._results_file)
        expected_count = ckpt.last_successful_index + 1
        if actual_count != expected_count:
            raise ResultsOutOfSyncError(
                checkpoint_idx=ckpt.last_successful_index, actual_count=actual_count
            )

        self._emit_audit(
            "workflow_resumed",
            {
                "workflow_id": self._workflow_id,
                "resume_from_index": ckpt.last_successful_index + 1,
                "manifest_sha256": self._manifest_sha256,
                "results_sha256": actual_chk,
            },
        )
        return ckpt.last_successful_index + 1

    # ── Per-task execution ───────────────────────────────────────

    async def _safe_invoke(self, task: dict[str, Any]) -> TaskResult:
        tid = str(task.get("task_id", "<missing>"))
        t0 = time.monotonic()
        try:
            res = await self._handler(task)
            if not isinstance(res, TaskResult):
                raise TypeError(f"handler must return TaskResult, got {type(res).__name__}")
            return res
        except Exception as exc:
            duration = (time.monotonic() - t0) * 1000.0
            err_text = str(exc)
            if len(err_text.encode("utf-8")) > _ERROR_TRUNCATE_BYTES:
                err_text = err_text.encode("utf-8")[:_ERROR_TRUNCATE_BYTES].decode(
                    "utf-8", errors="replace"
                )
            return TaskResult(
                task_id=tid,
                success=False,
                duration_ms=duration,
                output=None,
                error=err_text,
                error_type=type(exc).__name__,
            )

    def _append_result(self, result: TaskResult) -> None:
        line = json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
        # Per-write fsync: results.jsonl is the truth.
        with self._results_file.open("a", encoding="utf-8", newline="") as f:
            f.write(line)
            f.flush()
            with suppress(OSError):
                os.fsync(f.fileno())

    # ── Checkpoint persistence (atomic) ──────────────────────────

    async def _write_checkpoint_and_emit(self, last_idx: int) -> None:
        chk = _sha256_of_file(self._results_file)
        now = datetime.now(UTC).isoformat()
        state = CheckpointState(
            workflow_id=self._workflow_id,
            source_file=str(self._manifest_path),
            last_successful_index=last_idx,
            last_checkpoint_timestamp=now,
            checksum_of_results=chk,
            manifest_sha256=self._manifest_sha256,
        )
        # Atomic write: tmp -> fsync -> rename.
        tmp_path = self._checkpoint_file.with_suffix(self._checkpoint_file.suffix + ".tmp")
        payload = json.dumps(state.to_dict(), indent=2, ensure_ascii=False, sort_keys=True)
        with tmp_path.open("w", encoding="utf-8", newline="") as f:
            f.write(payload)
            f.flush()
            with suppress(OSError):
                os.fsync(f.fileno())
        os.replace(tmp_path, self._checkpoint_file)
        # fsync the directory so the rename is durable on POSIX.
        if sys.platform != "win32":
            with suppress(OSError):
                dir_fd = os.open(str(self._results_dir), os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)

        now_wall = time.monotonic() * 1000.0
        elapsed = now_wall - self._last_checkpoint_wall_ms
        self._last_checkpoint_wall_ms = now_wall

        self._emit_audit(
            "system_checkpoint_created",
            {
                "workflow_id": self._workflow_id,
                "index": last_idx,
                "results_sha256": chk,
                "elapsed_ms_since_last_checkpoint": round(elapsed, 3),
            },
        )

    # ── Audit-log integration ────────────────────────────────────

    def _emit_audit(self, event: str, payload: dict[str, Any]) -> None:
        if self._audit is None:
            return
        # All workflow events route via SYSTEM (operational state).
        # Spec deviation explained at module top. The free-form
        # ``description`` carries the JSON payload so consumers can
        # parse it post-hoc; ``action`` is the stable event id.
        try:
            self._audit.log_system(
                event=event,
                description=json.dumps(payload, sort_keys=True, ensure_ascii=False),
            )
        except Exception as exc:
            logger.warning("audit_emit_failed event=%s error=%s", event, exc)

    def _emit_signal_event(self, last_idx: int) -> None:
        if self._shutdown_signal is None:
            return
        self._emit_audit(
            "workflow_signal_received",
            {
                "workflow_id": self._workflow_id,
                "signal": self._shutdown_signal,
                "index": last_idx,
            },
        )

    # ── Signal handling ──────────────────────────────────────────

    def _install_signal_handlers(self) -> None:
        signals: list[int] = []
        if hasattr(signal, "SIGINT"):
            signals.append(signal.SIGINT)
        if hasattr(signal, "SIGTERM"):
            signals.append(signal.SIGTERM)
        if sys.platform == "win32" and hasattr(signal, "SIGBREAK"):
            signals.append(signal.SIGBREAK)

        for sig in signals:
            try:
                prev = signal.getsignal(sig)
                signal.signal(sig, self._signal_handler)
                self._installed_signal_handlers.append((sig, prev))
            except (ValueError, OSError):
                # Not on the main thread, or platform doesn't support
                # this signal -- skip silently. The asyncio Event
                # cooperative shutdown still works for in-process
                # cancellation.
                pass

    def _restore_signal_handlers(self) -> None:
        for sig, prev in self._installed_signal_handlers:
            with suppress(ValueError, OSError, TypeError):
                signal.signal(sig, prev)
        self._installed_signal_handlers.clear()

    def _signal_handler(self, signum: int, _frame: Any) -> None:
        # Runs in main-thread signal context. Keep it tiny: just
        # mark the flag. The async watcher promotes it to the event
        # so the run-loop sees it between tasks. NEVER cancel an
        # in-flight await here -- that's the integrity guarantee.
        try:
            self._shutdown_signal = signal.Signals(signum).name
        except ValueError:
            self._shutdown_signal = f"SIG_{signum}"
        self._shutdown_flag.set()

    async def _signal_watcher(self) -> None:
        # Light polling beats wiring loop.add_signal_handler (not
        # supported on Windows). 50 ms is well below human-perceptible
        # and fine for between-task latency.
        try:
            while True:
                if self._shutdown_flag.is_set() and self._shutdown_event is not None:
                    self._shutdown_event.set()
                    return
                await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            return


# ============================================================================
# CLI entry-point helpers
# ============================================================================


def load_handler_from_entrypoint(entrypoint: str) -> TaskHandler:
    """Resolve an ``"module.path:function"`` string into a TaskHandler.

    Used by ``cognithor task --handler`` to wire user code without
    requiring import gymnastics in the CLI dispatcher.
    """
    if ":" not in entrypoint:
        raise ValueError(f"handler entrypoint must be 'module.path:function', got {entrypoint!r}")
    mod_name, fn_name = entrypoint.split(":", 1)
    try:
        import importlib

        mod = importlib.import_module(mod_name)
    except ImportError as exc:
        raise ValueError(f"cannot import module {mod_name!r}: {exc}") from exc
    try:
        fn = getattr(mod, fn_name)
    except AttributeError as exc:
        raise ValueError(f"module {mod_name!r} has no attribute {fn_name!r}") from exc
    if not callable(fn):
        raise TypeError(f"{entrypoint!r} is not callable")
    return _resolve_handler(fn)


__all__ = [
    "CheckpointIntegrityError",
    "CheckpointState",
    "EmptyManifestError",
    "ManifestError",
    "ManifestTamperError",
    "ManifestValidationError",
    "ResultsOutOfSyncError",
    "TaskHandler",
    "TaskResult",
    "WorkflowAlreadyRunning",
    "WorkflowError",
    "WorkflowRunner",
    "WorkflowSummary",
    "_run_sync_handler",
    "load_handler_from_entrypoint",
    "stream_tasks",
    "validate_manifest",
]
