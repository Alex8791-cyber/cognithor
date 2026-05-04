"""Sprint-27 streaming-event taxonomy + envelope.

Frozen, JSON-serialisable dataclasses for the eight event types
defined by the v1 JSON Schema at
``cognithor/streaming/schemas/v1/events.json``.

**H1 (per-event schema_version):** every concrete event class
exposes a class-level ``SCHEMA_VERSION`` constant and embeds it
into the wire payload via :meth:`StreamEvent.to_dict`. The
envelope-level schema version is intentionally absent — bumping
``RunComplete``'s schema in v2 must NOT force every other event to
revision.

**H2 (run_started / run_error / run_cancelled defined day 1):**
all seven happy-and-sad-path events are first-class types; the
extension's discriminator switch can rely on a closed set.

**H4 critical bypass:** :data:`CRITICAL_EVENT_TYPES` is the
authoritative set of events that bypass the bounded-buffer drop
mechanism (terminal events, plus :class:`SinkDropped` itself so
operators never lose the "I dropped events" signal).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Final, Literal

# Absolute path to the v1 JSON Schema. Tests round-trip every
# emitted event through ``jsonschema.validate`` against this file.
SCHEMA_PATH: Final[Path] = Path(__file__).parent / "schemas" / "v1" / "events.json"

# Top-level schema-version label exposed for telemetry (e.g. the
# "Cognithor streaming v1" footer in the extension status bar).
# NOT mixed into the wire payload — per H1, individual event
# classes carry their own ``schema_version`` field.
SCHEMA_VERSION: Final[str] = "v1"

# Producer-side truncation cap for ``DecisionExplanation.matched_pattern``.
# Mirrors the cap baked into the JSON Schema and into
# ``cognithor.security.permission_scope`` — keep them in sync.
_MATCHED_PATTERN_MAX_LEN: Final[int] = 200


def _utcnow_iso() -> str:
    """ISO 8601 UTC timestamp with explicit ``Z`` suffix."""

    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


@dataclass(frozen=True, slots=True)
class DecisionExplanation:
    """TRUST-2 structured 'why' attached to Gatekeeper block paths.

    Mirrors the runtime ``DecisionExplanation`` in
    :mod:`cognithor.core.gatekeeper`. Producer-side truncates
    ``matched_pattern`` to 200 chars before emission.
    """

    rule_id: str
    rule_source: str
    matched_pattern: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "rule_id": self.rule_id,
            "rule_source": self.rule_source,
        }
        if self.matched_pattern is not None:
            out["matched_pattern"] = self.matched_pattern[:_MATCHED_PATTERN_MAX_LEN]
        return out


@dataclass(frozen=True, slots=True, kw_only=True)
class StreamEvent:
    """Abstract envelope shared by every emitted event.

    Concrete subclasses MUST set the class-var ``EVENT_TYPE`` to
    the wire-format discriminator string and ``SCHEMA_VERSION`` to
    the current per-event schema number (H1).
    """

    EVENT_TYPE: ClassVar[str] = ""
    SCHEMA_VERSION: ClassVar[int] = 1

    run_id: str
    ts: str = field(default_factory=_utcnow_iso)

    def __post_init__(self) -> None:
        if not self.EVENT_TYPE:
            msg = (
                f"{type(self).__name__}.EVENT_TYPE is unset — concrete "
                "StreamEvent subclasses must define EVENT_TYPE"
            )
            raise TypeError(msg)
        if not self.run_id:
            msg = "StreamEvent.run_id must be a non-empty string"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, Any]:
        """Wire-format payload. Subclasses override + chain via super()."""

        return {
            "event": self.EVENT_TYPE,
            "schema_version": self.SCHEMA_VERSION,
            "run_id": self.run_id,
            "ts": self.ts,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class RunStarted(StreamEvent):
    """First event for any run — always precedes plan_step / gate_decision."""

    EVENT_TYPE: ClassVar[str] = "run_started"
    SCHEMA_VERSION: ClassVar[int] = 1

    plan_path: str
    step_count: int
    agent_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out = StreamEvent.to_dict(self)
        out["plan_path"] = self.plan_path
        out["step_count"] = self.step_count
        if self.agent_id is not None:
            out["agent_id"] = self.agent_id
        return out


@dataclass(frozen=True, slots=True, kw_only=True)
class PlanStep(StreamEvent):
    """Per-step Planner-output event."""

    EVENT_TYPE: ClassVar[str] = "plan_step"
    SCHEMA_VERSION: ClassVar[int] = 1

    step: int
    action: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        out = StreamEvent.to_dict(self)
        out["step"] = self.step
        out["action"] = dict(self.action)
        return out


GateStatus = Literal[
    "green",
    "yellow",
    "orange_approved",
    "orange_blocked",
    "red",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class GateDecision(StreamEvent):
    """TRUST-2 Gatekeeper outcome event with optional structured explanation."""

    EVENT_TYPE: ClassVar[str] = "gate_decision"
    SCHEMA_VERSION: ClassVar[int] = 1

    step: int
    status: GateStatus
    explanation: DecisionExplanation | None = None

    def to_dict(self) -> dict[str, Any]:
        out = StreamEvent.to_dict(self)
        out["step"] = self.step
        out["status"] = self.status
        if self.explanation is not None:
            out["explanation"] = self.explanation.to_dict()
        return out


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolResult(StreamEvent):
    """Per-tool invocation result event."""

    EVENT_TYPE: ClassVar[str] = "tool_result"
    SCHEMA_VERSION: ClassVar[int] = 1

    step: int
    ok: bool
    duration_ms: int | None = None
    chunks: int | None = None
    preview: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out = StreamEvent.to_dict(self)
        out["step"] = self.step
        out["ok"] = self.ok
        if self.duration_ms is not None:
            out["duration_ms"] = self.duration_ms
        if self.chunks is not None:
            out["chunks"] = self.chunks
        if self.preview is not None:
            out["preview"] = self.preview[:500]
        if self.error is not None:
            out["error"] = self.error
        return out


@dataclass(frozen=True, slots=True, kw_only=True)
class RunComplete(StreamEvent):
    """Terminal happy-path event — always carries the TRUST-1 receipt bundle."""

    EVENT_TYPE: ClassVar[str] = "run_complete"
    SCHEMA_VERSION: ClassVar[int] = 1

    receipt: dict[str, Any]
    duration_ms: int | None = None
    status: Literal["success"] = "success"

    def to_dict(self) -> dict[str, Any]:
        out = StreamEvent.to_dict(self)
        out["status"] = self.status
        if self.duration_ms is not None:
            out["duration_ms"] = self.duration_ms
        out["receipt"] = dict(self.receipt)
        return out


@dataclass(frozen=True, slots=True, kw_only=True)
class RunError(StreamEvent):
    """Terminal failure event — always carries a FailureMode (TRUST-3)."""

    EVENT_TYPE: ClassVar[str] = "run_error"
    SCHEMA_VERSION: ClassVar[int] = 1

    failure_mode: str
    error: str | None = None
    step: int | None = None
    receipt: dict[str, Any] | None = None
    status: Literal["error"] = "error"

    def to_dict(self) -> dict[str, Any]:
        out = StreamEvent.to_dict(self)
        out["status"] = self.status
        out["failure_mode"] = self.failure_mode
        if self.error is not None:
            out["error"] = self.error
        if self.step is not None:
            out["step"] = self.step
        if self.receipt is not None:
            out["receipt"] = dict(self.receipt)
        return out


@dataclass(frozen=True, slots=True, kw_only=True)
class RunCancelled(StreamEvent):
    """Terminal user-cancellation event."""

    EVENT_TYPE: ClassVar[str] = "run_cancelled"
    SCHEMA_VERSION: ClassVar[int] = 1

    reason: str | None = None
    step: int | None = None
    receipt: dict[str, Any] | None = None
    status: Literal["cancelled"] = "cancelled"

    def to_dict(self) -> dict[str, Any]:
        out = StreamEvent.to_dict(self)
        out["status"] = self.status
        if self.reason is not None:
            out["reason"] = self.reason
        if self.step is not None:
            out["step"] = self.step
        if self.receipt is not None:
            out["receipt"] = dict(self.receipt)
        return out


@dataclass(frozen=True, slots=True, kw_only=True)
class SinkDropped(StreamEvent):
    """Producer-emitted notice that a Sink dropped buffered events.

    Sinks emit this *into their own stream* only — operators must
    learn that their consumer fell behind, even if no other channel
    survives. Counted as a critical event so it never gets dropped
    by the same backpressure mechanism it warns about (H4).
    """

    EVENT_TYPE: ClassVar[str] = "sink_dropped"
    SCHEMA_VERSION: ClassVar[int] = 1

    sink: str
    dropped_count: int

    def to_dict(self) -> dict[str, Any]:
        out = StreamEvent.to_dict(self)
        out["sink"] = self.sink
        out["dropped_count"] = self.dropped_count
        return out


# H4 — critical events bypass the bounded-buffer drop mechanism.
# Terminal lifecycle events MUST always be delivered; the
# ``sink_dropped`` notification MUST always be delivered for the
# operator to know they have an incomplete picture.
CRITICAL_EVENT_TYPES: Final[frozenset[str]] = frozenset(
    {
        RunComplete.EVENT_TYPE,
        RunError.EVENT_TYPE,
        RunCancelled.EVENT_TYPE,
        SinkDropped.EVENT_TYPE,
    }
)
