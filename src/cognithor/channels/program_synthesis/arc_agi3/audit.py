# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-12 — ARC-AGI-3 Audit Trail (lifted from cognithor.arc.audit).

Append-only chain of audit events (game_start, step, level_complete,
game_end, error) with a SHA-256 hash chain for tamper detection. Every
event includes its predecessor's hash, so any post-hoc edit to a logged
event is detectable via :meth:`verify_integrity`.

Drop-in usable from :class:`Sprint10DSLAgent` to record episode
provenance — useful for replay debugging and for cross-episode
analytics in :class:`GameProfileStore`.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any

__all__ = ["ArcAuditEvent", "ArcAuditTrail"]


@dataclass
class ArcAuditEvent:
    """A single auditable event in an ARC game session.

    Sprint-15 added the optional ``llm_*`` fields so Phase-3 telemetry
    rides on the existing hash chain — no separate schema, no join
    needed at bench-comparison time. All defaults are ``None`` so old
    JSONL exports stay readable and the verifier doesn't break on
    legacy events.
    """

    timestamp: float
    event_type: str  # "game_start", "step", "level_complete", "game_end", "error"
    game_id: str
    level: int
    step: int
    action: str | None = None
    game_state: str | None = None
    pixels_changed: int | None = None
    score: float | None = None
    error: str | None = None
    metadata: dict[str, Any] | None = None
    # Sprint-15 telemetry — populated when an LLM call drove the step.
    llm_input_tokens: int | None = None
    llm_output_tokens: int | None = None
    llm_think_tokens: int | None = None
    llm_finish_reason: str | None = None  # "stop"/"length"/"tool_calls"/"abort"
    llm_wall_clock_s: float | None = None
    # Sprint-15 follow-up: time-to-first-token from vLLM's
    # ``RequestStateStats.first_token_latency``. Captured on every
    # call so the analyzer can split prefill from decode and run
    # the multiplicative MTP-speedup formula correctly.
    llm_ttft_s: float | None = None
    # Sprint-15 MTP — present when speculative decoding fired this step.
    mtp_drafts_proposed: int | None = None
    mtp_drafts_accepted: int | None = None
    mtp_acceptance_rate: float | None = None
    # Sprint-19 Hebel P — raw LLM reasoning persisted with the step.
    # Lets post-mortem analysis read what the model was thinking when
    # it picked a destructive plan-step. Truncated to 4000 chars by
    # the producer to keep audit JSONL parse-friendly. ``None`` means
    # "no LLM call drove this step" (DSL fallback) or "the choice-fn
    # didn't surface a reasoning string".
    llm_reasoning: str | None = None


def _event_to_json(event: ArcAuditEvent) -> str:
    """Serialize an event to a canonical JSON string (sorted keys)."""
    return json.dumps(asdict(event), sort_keys=True, ensure_ascii=False)


class ArcAuditTrail:
    """Append-only audit trail with a SHA-256 hash chain for tamper detection."""

    def __init__(self, game_id: str, agent_version: str = "cognithor-arc-v1") -> None:
        self.game_id = game_id
        self.agent_version = agent_version
        self.events: list[ArcAuditEvent] = []
        self._hashes: list[str] = []
        self._previous_hash: str | None = None

        # run_id: first 16 chars of SHA-256(game_id:timestamp:version:uuid)
        seed = f"{game_id}:{time.time()}:{agent_version}:{uuid.uuid4().hex}"
        self.run_id = hashlib.sha256(seed.encode()).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Core append method
    # ------------------------------------------------------------------

    def log_event(self, event: ArcAuditEvent) -> str:
        """Append *event* to the trail, compute its chain hash, and return it."""
        event_json = _event_to_json(event)
        prev = self._previous_hash if self._previous_hash is not None else "GENESIS"
        chain_input = f"{prev}:{event_json}"
        new_hash = hashlib.sha256(chain_input.encode()).hexdigest()

        self.events.append(event)
        self._hashes.append(new_hash)
        self._previous_hash = new_hash
        return new_hash

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------

    def log_game_start(self) -> str:
        """Log a game_start event and return its chain hash."""
        event = ArcAuditEvent(
            timestamp=time.time(),
            event_type="game_start",
            game_id=self.game_id,
            level=0,
            step=0,
            metadata={"agent_version": self.agent_version, "run_id": self.run_id},
        )
        return self.log_event(event)

    def log_game_end(self, final_score: float) -> str:
        """Log a game_end event with the final score and return its chain hash."""
        event = ArcAuditEvent(
            timestamp=time.time(),
            event_type="game_end",
            game_id=self.game_id,
            level=0,
            step=0,
            score=final_score,
        )
        return self.log_event(event)

    def log_step(
        self,
        level: int,
        step: int,
        action: str,
        game_state: str,
        pixels_changed: int,
        *,
        llm_input_tokens: int | None = None,
        llm_output_tokens: int | None = None,
        llm_think_tokens: int | None = None,
        llm_finish_reason: str | None = None,
        llm_wall_clock_s: float | None = None,
        llm_ttft_s: float | None = None,
        mtp_drafts_proposed: int | None = None,
        mtp_drafts_accepted: int | None = None,
        mtp_acceptance_rate: float | None = None,
        llm_reasoning: str | None = None,
    ) -> str:
        """Log a single agent step and return its chain hash.

        Sprint-15: optional ``llm_*`` and ``mtp_*`` kwargs ride on the
        same hash-chained event so per-step token counts + speculative-
        decoding stats are tamper-evident alongside the action history.
        All defaults ``None`` preserve backwards compatibility.

        Sprint-19 Hebel P: ``llm_reasoning`` carries the model's
        top-level reasoning string (truncated to 4000 chars by the
        producer) so post-mortem analysis can read what the LLM was
        thinking when it picked the action — without re-running the
        episode.
        """
        # Hebel P safety: if the producer forgot to truncate, do it
        # here. Audit JSONL parsers choke on arbitrarily-long fields,
        # and 4000 chars is plenty for one reasoning sentence + plan
        # step rationales.
        if llm_reasoning is not None and len(llm_reasoning) > 4000:
            llm_reasoning = llm_reasoning[:4000]
        event = ArcAuditEvent(
            timestamp=time.time(),
            event_type="step",
            game_id=self.game_id,
            level=level,
            step=step,
            action=action,
            game_state=game_state,
            pixels_changed=pixels_changed,
            llm_input_tokens=llm_input_tokens,
            llm_output_tokens=llm_output_tokens,
            llm_think_tokens=llm_think_tokens,
            llm_finish_reason=llm_finish_reason,
            llm_wall_clock_s=llm_wall_clock_s,
            llm_ttft_s=llm_ttft_s,
            mtp_drafts_proposed=mtp_drafts_proposed,
            mtp_drafts_accepted=mtp_drafts_accepted,
            mtp_acceptance_rate=mtp_acceptance_rate,
            llm_reasoning=llm_reasoning,
        )
        return self.log_event(event)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_jsonl(
        self,
        filepath: str,
        *,
        seal_into_hashline: bool = False,
        hashline_data_dir: Any = None,
    ) -> str | None:
        """Write all events as JSONL (one JSON object per line).

        Sprint-15: when ``seal_into_hashline=True`` the export hashes
        the JSONL payload (SHA-256 over its bytes) and appends a
        single chained entry to Cognithor's :class:`HashlineAuditor`
        so the per-episode artefact has cross-system tamper-evidence.
        Returns the audit entry's SHA-256 hash on seal, ``None``
        otherwise.
        """
        with open(filepath, "w", encoding="utf-8") as fh:
            for event in self.events:
                fh.write(_event_to_json(event) + "\n")

        if not seal_into_hashline:
            return None

        # Lazy import to keep audit.py importable without the
        # hashline subsystem (used by the new program_synthesis
        # stack standalone in unit tests).
        from cognithor.hashline.audit import HashlineAuditor

        with open(filepath, "rb") as rb:
            payload = rb.read()
        digest = hashlib.sha256(payload).hexdigest()
        auditor = HashlineAuditor(data_dir=hashline_data_dir)
        return auditor._append(
            {
                "timestamp": time.time(),
                "type": "arc_episode_export",
                "game_id": self.game_id,
                "run_id": self.run_id,
                "agent_version": self.agent_version,
                "events_count": len(self.events),
                "jsonl_path": str(filepath),
                "jsonl_sha256": digest,
                "agent_id": "cognithor.channels.program_synthesis.arc_agi3",
            }
        )

    # ------------------------------------------------------------------
    # Integrity verification
    # ------------------------------------------------------------------

    def verify_integrity(self) -> bool:
        """Replay the hash chain from scratch and confirm it matches stored hashes."""
        if not self.events:
            return True

        prev = "GENESIS"
        for event, stored_hash in zip(self.events, self._hashes, strict=False):
            event_json = _event_to_json(event)
            chain_input = f"{prev}:{event_json}"
            expected = hashlib.sha256(chain_input.encode()).hexdigest()
            if expected != stored_hash:
                return False
            prev = stored_hash

        return True
