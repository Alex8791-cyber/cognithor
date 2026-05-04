# Cognithor Streaming Events — v1

This is the human-readable companion to
[`events.json`](./events.json). The JSON Schema is the source of
truth; this Markdown documents intent, reasoning, and consumer-side
expectations.

**Source-of-truth path:** `src/cognithor/streaming/schemas/v1/events.json`
(JSON Schema Draft 2020-12).

**Producer:** `cognithor.streaming.EventEmitter` (`src/cognithor/streaming/emitter.py`)
fans events out to one-or-more `Sink` consumers (`base.py` ABC,
concrete `JsonlSink` in PR-B, `WebSocketSink` in PR-C).

**Consumers:**
- `cognithor agent run --plan FILE.json --stream` (PR-B) writes JSONL to stdout.
- `cognithor agent ws --port 8742` (PR-C) serves frames over WebSocket.
- The VS Code extension (Phase-2) parses these events to drive UI.
- External orchestrators wrapping the headless CLI rely on the same shapes.

## Envelope

Every event carries:

| field | type | notes |
|---|---|---|
| `event` | string | Discriminator; one of the eight types below |
| `schema_version` | integer ≥ 1 | **Per-event, NOT envelope-level** (H1) |
| `run_id` | non-empty string | Matches the `trace_id` used by `GET /api/crew/trace/{trace_id}/receipt` |
| `ts` | RFC 3339 / ISO 8601 UTC | Producer-side timestamp |

### H1 — per-event `schema_version`

Per-event versioning means a Sprint-29 schema bump on, say,
`run_complete` does NOT force every other event type to revision.
Consumers MUST switch on the pair `(event, schema_version)` rather
than a top-level stream version.

## Event types

### `run_started`

Emitted exactly once per run, before any `plan_step`.
Lets a late-connecting consumer know what's currently running and
how long the run is expected to be.

| field | type | required |
|---|---|---|
| `plan_path` | string | yes — path to the ActionPlan JSON file (or `"<inline>"`) |
| `step_count` | int ≥ 0 | yes — Planner-reported step count |
| `agent_id` | string | optional |

### `plan_step`

Emitted per Planner-output step, before the corresponding
`gate_decision`.

| field | type | required |
|---|---|---|
| `step` | int ≥ 0 | yes |
| `action` | object with `tool` (string), optional `arguments` and `rationale` | yes |

### `gate_decision`

TRUST-2 — the Gatekeeper outcome for this step. Emitted after
`plan_step`, before `tool_result`.

| field | type | required |
|---|---|---|
| `step` | int ≥ 0 | yes |
| `status` | one of `green`, `yellow`, `orange_approved`, `orange_blocked`, `red` | yes |
| `explanation` | `DecisionExplanation` (`rule_id` + `rule_source` + optional truncated `matched_pattern`) | optional, but always present for block paths (status ∈ `{red, orange_blocked}`) |

`matched_pattern` is producer-side truncated to 200 characters.

### `tool_result`

Per-tool-invocation result.

| field | type | required |
|---|---|---|
| `step` | int ≥ 0 | yes |
| `ok` | bool | yes |
| `duration_ms` | int ≥ 0 | optional |
| `chunks` | int ≥ 0 | optional — count of chunks for chunked results |
| `preview` | string ≤ 500 chars | optional — UI-friendly preview |
| `error` | string | optional — set when `ok=false` |

### `run_complete` *(critical, terminal — H2 + H4)*

Emitted exactly once at the end of a successful run.

| field | type | required |
|---|---|---|
| `status` | const `"success"` | yes |
| `duration_ms` | int ≥ 0 | optional |
| `receipt` | object | yes — TRUST-1 bundle, same shape as `GET /api/crew/trace/{trace_id}/receipt?include_trust=true` |

### `run_error` *(critical, terminal — H2 + H4)*

Emitted exactly once when a run fails.

| field | type | required |
|---|---|---|
| `status` | const `"error"` | yes |
| `failure_mode` | one of the 15 `FailureMode` values | yes |
| `error` | string | optional — short summary |
| `step` | int ≥ 0 | optional — step where the error occurred |
| `receipt` | object | optional — partial TRUST-1 bundle |

### `run_cancelled` *(critical, terminal — H2 + H4)*

Emitted exactly once on user-initiated abort.

| field | type | required |
|---|---|---|
| `status` | const `"cancelled"` | yes |
| `reason` | string | optional |
| `step` | int ≥ 0 | optional |
| `receipt` | object | optional |

### `sink_dropped` *(critical — H4)*

Producer-emitted notice that a sink dropped buffered non-critical
events. Always delivered (cannot itself be dropped).

| field | type | required |
|---|---|---|
| `sink` | string | yes — sink name (`jsonl`, `websocket`, ...) |
| `dropped_count` | int ≥ 1 | yes |

## H4 — Backpressure semantics

Each sink owns:

1. A bounded `asyncio.Queue` of normal events (default capacity **1000**).
2. A reserved-slot pool for critical events (default capacity **16**).

Producer fans an event out to all sinks via `Sink.offer`, which
returns `True` on success and `False` on full-queue. **Producer
does NOT block** on slow sinks for non-critical events — slow
sinks drop, then surface the loss as a `sink_dropped` event the
next time they catch up.

Critical events (`run_complete`, `run_error`, `run_cancelled`,
`sink_dropped`) always bypass the normal-capacity gate and draw
from the reserve pool. Reserve exhaustion logs at ERROR level —
that's a real failure that operators should investigate (sink is
chronically slow or hung).

## H3 — WebSocket security defaults (preview, lands in PR-C)

The `WebSocketSink` in PR-C will:

- Bind `127.0.0.1` by default. `--bind 0.0.0.0` requires explicit warning + still requires the token.
- Require a `Bearer` token from `~/.cognithor/auth.token` (auto-generated 32-byte hex, mode 0600).
- Close with code `1008` (policy violation) on auth mismatch.

This is documented here in v1 so consumers (the extension client,
external orchestrators) build against it from day one.

## H5 — TypeScript codegen

Phase-2 will generate TS types from `events.json` via:

```bash
npx json-schema-to-typescript src/cognithor/streaming/schemas/v1/events.json \
  -o extensions/vscode/src/types/events.ts
```

The Markdown above is for humans; the JSON Schema is for both
humans and machines. Hand-maintained TS types in two languages
were explicitly rejected as a maintenance pattern (see decisions
memo D5 and H5).

## Versioning policy

A schema version bump on event `X` means at least one of:

- A required field was added.
- A field's type narrowed in a non-backward-compatible way.
- A field's semantics changed.

Adding an *optional* field does NOT bump the schema version.

A new event type is added by adding it to the top-level `oneOf`
in `events.json` and to the `EVENT_TYPE` constants in `events.py`,
with `SCHEMA_VERSION: ClassVar[int] = 1` for the new type. Other
event types are unaffected.
