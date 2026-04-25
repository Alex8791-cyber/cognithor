# ADR 0001: PGE Trinity as Multi-Agent Control Model

## Status
Accepted — 2026-04-25

## Context

Multi-agent systems need a way to coordinate multiple LLM-backed agents
working on a shared task. The dominant patterns in 2025-2026 are:

1. **AutoGen GroupChat patterns** (`autogen-agentchat`):
   - `RoundRobinGroupChat` — agents take turns in a fixed order.
   - `SelectorGroupChat` — an LLM decides who speaks next.
   - `Swarm` — agents pass control via `HandoffMessage`.
   - `MagenticOneGroupChat` — central orchestrator agent
     ([Magentic-One paper](https://arxiv.org/abs/2411.04468)).
2. **Graph orchestration** (LangGraph, MAF) — explicit DAG with nodes and
   edges; control-flow declared in code, not emergent from chat.
3. **Pure handoff/swarm** — no central coordinator; agents exchange
   ownership tokens.

Cognithor needs a Multi-Agent control model that supports:
- Auditability — every action attributable to a verifiable chain
  (Hashline Guard).
- DSGVO-grade safety — PII filtering, allow-list enforcement, before any
  external call.
- Predictability — no agent should "drift" into a role it wasn't
  authorized for, including via prompt injection.
- Local inference compatibility — must work without an external selector
  LLM in the critical path.

The question this ADR answers: **Why does Cognithor not just adopt one of
the AutoGen GroupChat patterns?**

## Decision

Cognithor uses **PGE Trinity** — Planner / Gatekeeper / Executor — as
enforced role separation:

- **Planner** decides what should happen next (intent, plan steps).
- **Gatekeeper** decides whether each proposed action is permissible
  (DSGVO PII check, tool allow-list, risk classification GREEN /
  YELLOW / ORANGE / RED).
- **Executor** runs the action and emits the audit record.

These roles are **separate concerns implemented as separate components**
(see `src/cognithor/core/planner.py`, `src/cognithor/core/gatekeeper.py`,
`src/cognithor/core/gateway.py`). They are not three prompts to the same
agent; they are three pipeline stages with explicit hand-offs and audit
points.

`cognithor.crew` (added in v0.93.0) wraps this trio for declarative
multi-agent scenarios, but the trio itself is not optional or
configurable away. Every Crew kickoff routes through Planner →
Gatekeeper → Executor.

## Consequences

### Positive

- **Auditability**: Every action passes through Gatekeeper, which writes
  a Hashline-Guard chain entry. The chain is verifiable end-to-end.
- **DSGVO**: PII filtering and allow-list enforcement live in one place,
  not duplicated per agent. Disabling them requires touching one
  component, which is reviewable.
- **No agent drift**: Roles cannot emerge from prompt-engineering
  accident. A Planner cannot execute; an Executor cannot decide policy.
- **Local-first**: No selector-LLM hop in the critical path. Gatekeeper
  is rule-based with optional LLM augmentation, not LLM-only.
- **Composability**: `cognithor.crew` can adopt new orchestration
  patterns above PGE; the trio is a substrate, not a replacement.

### Negative / Trade-offs

- **Higher latency than direct GroupChat**: Every action incurs a
  Gatekeeper hop (typically 5-50ms for rule-based classification, more
  if an LLM-backed risk check is enabled).
- **Less "creative" emergent behaviour**: SelectorGroupChat-style
  setups where an LLM picks the next speaker can produce surprising
  task-decompositions. PGE forecloses some of that surface area
  intentionally.
- **Higher entry barrier for AutoGen migrants**: Users coming from
  `RoundRobinGroupChat` or `SelectorGroupChat` see more boilerplate
  in PGE-Trinity for simple cases. The `cognithor.compat.autogen`
  shim mitigates this for the 1-shot and round-robin paths but
  cannot reproduce SelectorGroupChat or Swarm semantics.
- **Operational complexity**: Three components to monitor, three log
  streams to correlate. The Hashline-Guard chain ties them but
  understanding the chain is a learning curve.

## Alternatives Considered

1. **`RoundRobinGroupChat` equivalent without Gatekeeper** — rejected:
   no audit point. The chain would be a record of "who said what" but
   not of "which actions were authorized to run". DSGVO compliance
   would have to be reimplemented per agent.
2. **`SelectorGroupChat` equivalent (LLM picks the next speaker)** —
   rejected: an LLM as a security boundary is not load-bearing. Prompt
   injection trivially redirects a selector LLM. Gatekeeper rules are
   inspectable and testable.
3. **Pure Handoff / Swarm** — rejected: no central policy enforcement.
   Each agent would carry its own DSGVO logic; impossible to audit
   centrally.
4. **Graph orchestration (LangGraph / MAF style)** — rejected as
   substrate (would conflict with the conversation-style API surface
   `cognithor.crew` exposes), but kept as a future-feature option:
   spec §6 "Flows" (deferred to v1.x) would let users compose Crews
   into larger DAGs without replacing PGE-Trinity inside each Crew.

## References

- AutoGen GroupChat docs: https://microsoft.github.io/autogen/stable//user-guide/agentchat-user-guide/tutorial/teams.html
- Magentic-One paper: https://arxiv.org/abs/2411.04468
- Cognithor Gatekeeper code: `src/cognithor/core/gatekeeper.py`
- Cognithor PGE pipeline: `src/cognithor/gateway/gateway.py`
- Hashline Guard: `docs/hashline-guard.md`
- v0.94.0 AutoGen-compat shim (PR 3): `src/cognithor/compat/autogen/`
