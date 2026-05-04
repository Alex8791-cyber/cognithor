"""JsonlSink — newline-delimited JSON to stdout / file (Sprint-27 PR-B).

First concrete consumer of :class:`Sink`. Writes one
JSON-encoded event per line to a text stream. Defaults to
:data:`sys.stdout` so an extension shelling out to
``cognithor agent run --plan FILE.json --stream`` reads events
straight from the subprocess pipe.

The wire format is exactly the per-event ``to_dict()`` payload
defined by the v1 JSON Schema — no extra envelope, no decoration,
no batching. One event = one line. That makes the stream trivial
to parse with ``readline()`` from any consumer language.

Critical events still bypass the bounded buffer (H4); ``stop()``
flushes anything queued before returning.
"""

from __future__ import annotations

import contextlib
import json
import sys
from typing import IO, TYPE_CHECKING

from cognithor.streaming.sinks.base import Sink, SinkBufferConfig

if TYPE_CHECKING:
    from pathlib import Path

    from cognithor.streaming.events import StreamEvent


class JsonlSink(Sink):
    """Writes events as ``json.dumps(event.to_dict()) + "\\n"`` to a text stream.

    Three construction patterns:

    1. ``JsonlSink()`` — defaults to ``sys.stdout``.
    2. ``JsonlSink(stream=open_text_handle)`` — wrap any text handle.
       Caller owns the handle's lifecycle.
    3. ``JsonlSink.to_path(Path("run.jsonl"))`` — convenience that
       opens the path in append-binary mode and lets the sink own
       its lifecycle (closed in ``_close``).
    """

    name = "jsonl"

    def __init__(
        self,
        *,
        stream: IO[str] | None = None,
        buffer: SinkBufferConfig | None = None,
        owns_stream: bool = False,
    ) -> None:
        super().__init__(buffer=buffer)
        self._stream: IO[str] = stream if stream is not None else sys.stdout
        self._owns_stream = owns_stream

    @classmethod
    def to_path(
        cls,
        path: Path,
        *,
        buffer: SinkBufferConfig | None = None,
    ) -> JsonlSink:
        """Open ``path`` in line-buffered append-text mode and own the handle."""

        path.parent.mkdir(parents=True, exist_ok=True)
        stream = path.open("a", encoding="utf-8", buffering=1)
        return cls(stream=stream, buffer=buffer, owns_stream=True)

    async def _write(self, event: StreamEvent) -> None:
        line = json.dumps(event.to_dict(), ensure_ascii=False, sort_keys=True)
        # Single write call so partially-buffered output stays
        # line-aligned even if the consumer is reading concurrently.
        self._stream.write(line + "\n")
        # Stdout is line-buffered by default in CPython; explicit
        # flush keeps the extension-side reader from blocking on
        # idle producers (e.g. waiting on a slow LLM mid-step).
        try:
            self._stream.flush()
        except (OSError, ValueError):
            # Stream closed or detached mid-write. Quarantine
            # ourselves rather than spinning on the same error.
            self._faulted = True

    async def _close(self) -> None:
        if self._owns_stream:
            with contextlib.suppress(OSError, ValueError):
                self._stream.close()
