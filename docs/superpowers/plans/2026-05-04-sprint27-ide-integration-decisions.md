# Sprint-27 — IDE-Integration: Owner Decisions (Companion Doc)

**Date:** 2026-05-04
**Status:** Decisions locked. Phase-1 unblocked. Companion to
[`2026-05-04-sprint27-ide-integration.md`](./2026-05-04-sprint27-ide-integration.md).
**Owner:** Alexander Söllner

This document records the five decisions made before Sprint-27 kicks
off, and the architectural / process refinements derived from owner
review of the original plan-doc. The plan-doc itself stays the
canonical scope reference; this companion holds the *binding*
choices and the *why* behind them.

---

## D1 — Repo Layout: **Monorepo, `extensions/vscode/` subdirectory**

The original plan-doc proposed a separate `cognithor-vscode/` repo.
Decision: **NO**. Use a `extensions/vscode/` subdirectory inside
the main `cognithor` repo.

**Why:**
- Separate repo is *convention*, not a Marketplace requirement.
  Marketplace cares about the `.vsix` artifact, not the source repo
  layout.
- Phase-1 will iterate on the streaming-protocol; backend changes
  and extension-side parser changes ship best in atomic PRs (single
  PR touches both `src/cognithor/streaming/` and
  `extensions/vscode/src/`). A two-repo split fragments that.
- One CI, one issue tracker, one place where Apache-2.0 contributors
  see the work — better discoverability for a solo-dev project with
  one core contributor.
- JetBrains plugin (when it lands in Sprint-28+) goes under
  `extensions/jetbrains/` for the same reason.

**Consequences:**
- The Microsoft Marketplace publisher identity is still its own thing
  (Azure-DevOps PAT) — that's orthogonal to repo layout.
- The `.gitignore` and CI must be careful that `extensions/vscode/`
  doesn't break Python-side test collection.
- Versioning: extension's `package.json` version stays in lockstep
  with `pyproject.toml` and the four other version-canonical files.

---

## D2 — Marketplace Publishing: **Azure-DevOps PAT, NO code-signing cert**

The original plan-doc claimed publishing a VS-Code extension required
a code-signing certificate with ~1 week lead time. **That claim was
wrong** and would have created a phantom blocker for Phase-2.

The actual VS-Code Marketplace requirements:
- Microsoft account (already exists)
- Azure DevOps organization (free, ~5 minutes to create)
- Personal Access Token with scope **Marketplace (Manage)**
- `vsce create-publisher <name>` to claim a publisher identity
- `vsce publish` to upload the `.vsix`

Total realistic setup: **30 minutes**, not one week. Microsoft signs
extensions on upload; publisher-side signing is an in-progress
initiative, currently mandatory only for Microsoft-owned extensions.

**Owner-side action**: 30-minute Azure-DevOps + PAT + `vsce
create-publisher cognithor` whenever convenient — does not block
Phase-1.

**Note for future**: Authenticode (Windows binary signing,
DigiCert ~$500/year) is *unrelated* to VS-Code Marketplace
publishing. Cognithor already has Authenticode for the
`CognithorSetup-*.exe` Inno-Setup installer; it has nothing to do
with the `.vsix` flow.

### Anti-pattern lesson (META — applies beyond this sprint)

**When a future Claude session claims a technical requirement that
introduces lead-time risk** — "you need a cert", "this requires an
external review", "hardware procurement needed", "approval cycle from
upstream", etc. — **verify the claim against current vendor docs
before accepting it as a sprint constraint.** The wrong assumption
here would have padded Sprint-27 by an entire week of fake
parallelism. The cost of asking "is this still true today?" is
seconds; the cost of letting it through is days.

This is the more valuable lesson than the specific cert correction.
The same shape of error returns in different costumes:
"compliance review takes 6 weeks", "legal sign-off required for X",
"vendor needs an NDA before they answer". Verify each one.

---

## D3 — JetBrains Plugin: **Defer to Sprint-28+**

The original plan-doc had a JetBrains plugin scaffold (Track C, task
#10) inside Sprint-27. Decision: **DEFER**.

**Why:**
- IntelliJ Platform is a different stack (Kotlin/Java + Gradle, own
  Marketplace pipeline). Building both plugins in parallel means
  debugging both at the same time, with each fix possibly reflecting
  back into the other.
- Smaller DACH dev userbase for IntelliJ vs VS-Code.
- Sprint-27 will surface lessons about what the streaming-protocol
  actually needs (vs what the plan-doc *thinks* it needs). Those
  lessons go straight into the JetBrains spec, instead of being
  paid for twice.
- Cursor and Windsurf are VS-Code forks — the same `.vsix` is very
  likely to install and work there without separate distribution.
  A *Compat smoke test* (load the extension in Cursor, click
  "Run Plan", verify receipt sidebar populates) replaces the
  Cursor/Windsurf section of the plan-doc cheaper.

**Consequences:**
- Plan-doc tasks #10 (JetBrains scaffold) → Sprint-28+
- Plan-doc task #11 (Cursor + Claude-Desktop docs) → still in
  Sprint-27, but downgraded from "build a plugin" to "smoke-test
  the .vsix in each host + write a 1-page integration note"
- Sprint-27 effective scope: Track A + Track B only.

---

## D4 — Pricing: **Extension fully free, including Receipt-Sidebar + Cost-Gutter**

The original plan-doc left this as an open decision: extension
free, but require a paid pack to unlock advanced features
(Receipt-Sidebar, Cost-Gutter)?

Decision: **NO paywall on the extension itself or on the
trust-visualization features.** Apache-2.0, the whole thing.

**Why:**
- Receipt-Sidebar and Cost-Gutter are exactly the features that
  *prove* Cognithor's differentiation against CrewAI / AutoGen /
  LangGraph. Putting them behind a paywall kills the showcase —
  reviewers writing comparison posts (Reddit, HN) need to see them
  in their default Marketplace install.
- The Anti-Enshittification Promise (`/promise` page on
  cognithor.ai) commits to: tooling free, packs add but never
  subtract. A paywall on Receipt-Sidebar would violate the spirit
  of that promise even if not the letter.
- The correct paywall surface is **domain-specific knowledge**:
  insurance pre-advisory templates (Versicherungs-Pack), bAV
  workflows, content-marketing prompt libraries, etc. That's where
  Cognithor's commercial wedge actually lives.
- Paid packs unlock new *capabilities*, not visibility into existing
  capabilities.

**Consequences:**
- `extensions/vscode/` ships under Apache-2.0
- All six features from plan-doc §2 Track A are in the free tier
- Marketplace listing copy emphasizes the trust-receipt + decision-
  explanation features as core differentiators, not as upsell hooks

---

## D5 — Architecture: **Single EventEmitter + multiple Sinks, NOT separate impls**

The original plan-doc treated JSONL-stdout streaming (task #4) and
WebSocket streaming (task #5) as separate implementations.
Decision: **single producer, two encoders/sinks.**

**Layout:**
- `cognithor/streaming/event_emitter.py` — the Producer. Emits
  schema-versioned `StreamEvent` instances.
- `cognithor/streaming/sinks/jsonl_sink.py` — `JsonlSink` writes
  one event per line to stdout (or a file).
- `cognithor/streaming/sinks/ws_sink.py` — `WebSocketSink` pushes
  events as JSON frames to all connected clients.
- `cognithor/streaming/schemas/v1/events.json` — JSON Schema for
  every event type, machine-readable.
- `cognithor/streaming/schemas/v1/events.md` — Markdown companion,
  human-readable.

**Why:**
- One producer means one source of truth for event shapes. No
  drift between transports.
- A schema bug found in Sprint-28 is fixed in *one place*, not two.
- Adding a third sink later (e.g. a SQLite-backed sink for
  post-mortem replay, or a Prometheus pushgateway sink) becomes a
  ~50-LOC addition instead of a fork-the-CLI affair.
- The JSON Schema at `schemas/v1/events.json` lets us run
  `npx json-schema-to-typescript` in the extension build to generate
  TS types, instead of hand-maintaining them in two languages.

**Refactored task split for Sprint-27 Phase-1:**
- **PR-A**: `EventEmitter` + event types + JSON Schema + Markdown
  companion + the five hardening points below (~250-300 LOC)
- **PR-B**: `JsonlSink` + `cognithor agent run --plan FILE.json
  --stream` (~250 LOC)
- **PR-C**: `WebSocketSink` + `cognithor agent ws --port 8742` +
  security defaults (~250 LOC)

Total ~750-800 LOC, single coherent architecture.

---

## PR-A non-retrofittable hardening (must ship in PR-A, not later)

Five things that become breaking-change migrations if we punt them
to a later PR. All five locked in before PR-A merge:

### H1 — Per-event `schema_version`, NOT envelope-level

Bad: `{"schema_version": 1, "events": [...]}` at the top.
Good: `{"event":"plan_step", "schema_version":1, "step":N, ...}`
on every event.

Reason: in Sprint-29 we'll want to upgrade *one* event type
without rev'ing the entire stream. Per-event versioning costs
nothing and preserves that flexibility.

### H2 — `run_started`, `run_error`, `run_cancelled` defined from day 1

The original plan-doc listed only happy-path events:
`plan_step`, `gate_decision`, `tool_result`, `run_complete`.

That's not enough:
- **`run_started`**: lets the extension show "this run is now
  active" before any plan step lands. Also lets the extension
  reconnect mid-run and learn what's currently running (handshake
  reply).
- **`run_error`**: distinct from `run_complete`. The receipt
  bundle for a failed run is shaped differently (carries the
  `FailureMode`); the UI needs to know to render the red banner
  instead of the green checkmark.
- **`run_cancelled`**: user-initiated abort. Different telemetry
  bucket, different UI affordance, different audit shape.

If these arrive in Sprint-28, every existing client breaks because
their event-type discriminator is now incomplete. Ship all seven
event types in v1 of the schema.

### H3 — WebSocket security defaults: localhost-only + token auth

`cognithor agent ws` defaults:
- **Bind to `127.0.0.1` only.** Never `0.0.0.0` by default.
- **Require token auth.** Simple pre-shared key from
  `~/.cognithor/auth.token` (auto-generated on first start, 32-byte
  random hex, mode 0600). Client supplies it via `Authorization:
  Bearer <token>` header on the WebSocket upgrade request.
- **`--bind 0.0.0.0`** flag exists but emits a security warning,
  and *still* requires the token (no auth-bypass-with-bind).
- Token-mismatch closes the WS with code 1008 (policy violation)
  before the upgrade completes.

Reason: the SEC-CRIT-1 bootstrap-endpoint-CVSS-9.8 incident
(audit-fixed in PR #173) was the same class of error — exposed
network surface with no auth. Don't repeat it. The VS-Code
extension only needs localhost; making `0.0.0.0` the default would
gain nothing and lose the security default.

### H4 — Backpressure semantics defined in PR-A, async fan-out

The Producer-Sink relationship needs explicit backpressure rules
*before* multiple sinks ship, not after.

Spec:
- Each Sink has its own bounded buffer (default 1000 events).
- Producer fans out async to all sinks in parallel.
- Producer **only blocks** when *all* sinks are full.
- A Sink falling behind drops events (oldest-first) and emits a
  one-time `sink_dropped` warning into its own stream so the
  consumer knows it's seen partial data.
- **Critical events bypass the drop mechanism:** `run_complete`,
  `run_error`, `run_cancelled` are always delivered, even if the
  buffer is at limit (these get a small reserved-slot pool of 16
  events per sink).

Cost: ~30-50 LOC more in PR-A. Saves a full architecture migration
later. Without this, PR-C ships and somebody's WebSocket client
disconnects mid-run; current behavior is undefined; that's an
ugly bug to discover in production.

### H5 — JSON Schema in machine-readable path, not just prose

`cognithor/streaming/schemas/v1/events.json` ships as a real JSON
Schema (Draft 2020-12). The Markdown companion at
`cognithor/streaming/schemas/v1/events.md` documents the human
side, but the schema file is the source of truth.

Phase-2 then runs `npx json-schema-to-typescript
cognithor/streaming/schemas/v1/events.json -o
extensions/vscode/src/types/events.ts` as part of the extension
build. TS types are auto-generated; no hand-maintenance, no drift.

Schema validation in PR-A's tests: every emitted event round-trips
through `jsonschema.validate()` against the schema file. A test
failure means either the schema needs updating *or* the producer
is emitting an off-spec event — either way, blocking error.

---

## Sprint-26 → Sprint-27 transition

Sprint-26 closed 2026-05-04 (PRs #389-#392, all four cut-offs
delivered ahead of the 4-week calendar plan). Real-Score-Validation
for Spider/HumanEval-Plus is hardware-gated and runs separately
from the codebase work — not a Sprint-27 dependency.

**Cadence override (2026-05-04):** Owner-direktive *"wir machen
keinerlei pausen! maximaler fokus … vollautonom go"*. The
Wednesday-pause rule from the Sprint-26 memo is **suspended for
Sprint-27**. PR-A starts immediately after this memo lands. PR-B
and PR-C follow without a calendar gap as soon as PR-A is merged.

---

## What this Memo locks in

- **D1**: Monorepo, `extensions/vscode/`
- **D2**: Azure-DevOps PAT, no signing cert needed (30-min owner task)
- **D3**: JetBrains deferred to Sprint-28+
- **D4**: Extension + Receipt-Sidebar + Cost-Gutter all free, Apache-2.0
- **D5**: Single EventEmitter + multiple Sinks (PR-A / PR-B / PR-C structure)
- **H1-H5**: Five PR-A hardening points, all non-retrofittable

## What this Memo does NOT cover

- Concrete wire-format examples for each event type — those land
  in PR-A itself, generated from the JSON Schema.
- Extension-side architecture (webview vs tree-view, etc.) —
  Phase-2 / Phase-3 concern.
- JetBrains plugin design — Sprint-28+.

## Cross-references

- Plan-doc: [`2026-05-04-sprint27-ide-integration.md`](./2026-05-04-sprint27-ide-integration.md)
- TRUST-1 receipt API consumed by Receipt-Sidebar:
  `src/cognithor/api/crew_traces.py:305` — `GET /api/crew/trace/{trace_id}/receipt?include_trust=true`
- Receipt CLI consumed by extension shell-out:
  `src/cognithor/cli/receipt_cmd.py` — `cognithor receipt show / verify / list / export-all / diff`
- MCP-stdio entry consumed by extension MCP-bridge:
  `src/cognithor/__main__.py:389` — `_run_mcp_server_mode`
- SEC-CRIT-1 bootstrap-endpoint CVSS-9.8 incident (referenced in H3):
  closed via PR #173.
- Anti-Enshittification Promise (referenced in D4):
  https://cognithor.ai/promise
