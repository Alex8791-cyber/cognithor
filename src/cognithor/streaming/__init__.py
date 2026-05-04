"""Sprint-27 streaming package — `cognithor agent run --stream` / `cognithor agent ws`.

Public surface for the IDE-Integration runtime: a single
:class:`EventEmitter` Producer fans out schema-versioned
:class:`StreamEvent` instances to one-or-more :class:`Sink`
consumers. The two ship-with-the-CLI sinks are
:class:`JsonlSink` (PR-B) and :class:`WebSocketSink` (PR-C);
PR-A (this PR) only ships the Producer, the Sink ABC, the
event taxonomy, and the JSON Schema.

Owner-decision D5 (`docs/superpowers/plans/2026-05-04-sprint27-ide-integration-decisions.md`)
locks "single Producer + multiple Sinks" as the architecture so
neither the JSONL nor WebSocket transport drifts in shape.

Hardening points H1-H5 (same companion doc):

H1  Per-event ``schema_version`` so single events upgrade independently.
H2  Seven event types from day 1 (run_started / plan_step /
    gate_decision / tool_result / run_complete / run_error /
    run_cancelled), plus the producer-emitted ``sink_dropped``.
H3  WebSocket security defaults — see ``cognithor.streaming.sinks.ws_sink``
    (PR-C).
H4  Backpressure: each sink owns a bounded buffer; producer fans out
    async; critical events bypass the drop mechanism via a reserved
    16-slot pool per sink.
H5  Machine-readable JSON Schema at
    :data:`SCHEMA_PATH` — ``cognithor/streaming/schemas/v1/events.json``
    (Draft 2020-12). Phase-2 codegens TypeScript types from it.
"""

from __future__ import annotations

from cognithor.streaming.emitter import EventEmitter
from cognithor.streaming.events import (
    SCHEMA_PATH,
    SCHEMA_VERSION,
    DecisionExplanation,
    GateDecision,
    PlanStep,
    RunCancelled,
    RunComplete,
    RunError,
    RunStarted,
    SinkDropped,
    StreamEvent,
    ToolResult,
)
from cognithor.streaming.sinks.base import Sink, SinkBufferConfig
from cognithor.streaming.sinks.jsonl_sink import JsonlSink

__all__ = [
    "SCHEMA_PATH",
    "SCHEMA_VERSION",
    "DecisionExplanation",
    "EventEmitter",
    "GateDecision",
    "JsonlSink",
    "PlanStep",
    "RunCancelled",
    "RunComplete",
    "RunError",
    "RunStarted",
    "Sink",
    "SinkBufferConfig",
    "SinkDropped",
    "StreamEvent",
    "ToolResult",
]
