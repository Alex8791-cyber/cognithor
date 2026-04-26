"""Trace-UI REST endpoints — read crew audit events from JSONL.

Endpoints (all owner-gated):
  GET /api/crew/traces?status=&since=&limit=
  GET /api/crew/trace/{trace_id}
  GET /api/crew/trace/{trace_id}/stats

Source: ~/.cognithor/logs/audit.jsonl (Hashline-Guard chain). Corrupt
lines are skipped with a counter surfaced in response meta.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

log = logging.getLogger(__name__)


def read_audit_lines(path: Path) -> tuple[list[dict[str, Any]], int]:
    """Read JSONL audit events. Returns (events, skipped_corrupt_count).

    Missing file → ([], 0). Corrupt JSON lines are logged and skipped;
    valid lines are returned in file order.
    """
    if not path.exists():
        return [], 0

    events: list[dict[str, Any]] = []
    skipped = 0
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        log.warning("audit_jsonl_read_failed path=%s", path, exc_info=exc)
        return [], 0

    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            log.error("audit_jsonl_corruption path=%s line_no=%d", path, line_no)
            skipped += 1
            continue
        if isinstance(obj, dict):
            events.append(obj)
        else:
            skipped += 1
    return events, skipped
