"""CLI command for ``cognithor task <manifest>``.

Operator-facing front door to :class:`cognithor.core.workflow.WorkflowRunner`.
Wires manifest-path + handler entry-point + flags into a single
``cmd_run`` function returning a process exit code -- mirrors the
``receipt_cmd`` / ``agent_cmd`` style.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from cognithor.core.workflow import (
    CheckpointIntegrityError,
    EmptyManifestError,
    ManifestError,
    ManifestTamperError,
    ManifestValidationError,
    ResultsOutOfSyncError,
    WorkflowAlreadyRunning,
    WorkflowError,
    WorkflowRunner,
    load_handler_from_entrypoint,
)

if TYPE_CHECKING:
    from cognithor.audit import AuditLogger

log = logging.getLogger("cognithor.cli.task")


def _default_results_dir(manifest: Path) -> Path:
    """Default results dir under ``~/.cognithor/workflows/<stem>/``."""
    return Path.home() / ".cognithor" / "workflows" / manifest.stem


def cmd_run(
    *,
    manifest: Path,
    results_dir: Path | None = None,
    resume: bool = False,
    checkpoint_every: int = 12,
    workflow_id: str | None = None,
    handler_entrypoint: str,
    audit_log_dir: Path | None = None,
) -> int:
    """Run a JSONL workflow manifest.

    Returns
    -------
    0 on clean completion, 1 on workflow failure (e.g. integrity
    error), 2 on bad CLI arguments, 3 on partial success after
    SIGINT/SIGTERM.
    """
    if not manifest.exists():
        print(f"error: manifest not found: {manifest}", file=sys.stderr)
        return 2
    if checkpoint_every < 1:
        print("error: --checkpoint-every must be >= 1", file=sys.stderr)
        return 2

    try:
        handler = load_handler_from_entrypoint(handler_entrypoint)
    except (ValueError, TypeError) as exc:
        print(f"error: bad --handler {handler_entrypoint!r}: {exc}", file=sys.stderr)
        return 2

    rdir = results_dir or _default_results_dir(manifest)

    audit_logger: AuditLogger | None = None
    if audit_log_dir is not None:
        try:
            from cognithor.audit import AuditLogger as _AL

            audit_logger = _AL(log_dir=audit_log_dir)
        except Exception as exc:
            print(
                f"warning: audit logger init failed ({exc}); proceeding without audit",
                file=sys.stderr,
            )
            audit_logger = None

    try:
        runner = WorkflowRunner(
            manifest_path=manifest,
            results_dir=rdir,
            handler=handler,
            workflow_id=workflow_id,
            checkpoint_every=checkpoint_every,
            audit_logger=audit_logger,
        )
    except (
        ManifestError,
        ManifestValidationError,
        EmptyManifestError,
        WorkflowError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        summary = asyncio.run(runner.run(resume=resume))
    except WorkflowAlreadyRunning as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (
        ManifestError,
        ManifestValidationError,
        ManifestTamperError,
        CheckpointIntegrityError,
        ResultsOutOfSyncError,
        WorkflowError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("interrupted (KeyboardInterrupt)", file=sys.stderr)
        return 3

    print(json.dumps(summary.to_dict(), indent=2, ensure_ascii=False))
    if summary.interrupted_by_signal:
        return 3
    return 0
