# Cognithor Architecture

> Internal architecture reference for developers and contributors.
> For user-facing setup, see [QUICKSTART.md](QUICKSTART.md).

## Table of Contents

- [Overview](#overview)
- [PGE-Trinity](#pge-trinity)
- [Message Flow](#message-flow)
- [Initialization Phases](#initialization-phases)
- [Memory System](#memory-system)
- [Security Model](#security-model)
- [Channel Architecture](#channel-architecture)
- [Model Router](#model-router)
- [Context Pipeline](#context-pipeline)
- [Role System (v0.36)](#role-system)
- [Evolution Engine](#evolution-engine)
- [OSINT / HIM Module](#osint--him-module)
- [GDPR Compliance Layer](#gdpr-compliance-layer)
- [Bible Reference Index](#bible-reference-index)

---

## Overview

Cognithor is an agent OS built around the **PGE-Trinity**: three cooperating
subsystems that process every user message.

```
User Message
     │
     ▼
┌─────────┐     ┌────────────┐     ┌──────────┐
│ Planner │────▶│ Gatekeeper │────▶│ Executor │
│ (Think) │     │  (Guard)   │     │  (Act)   │
└─────────┘     └────────────┘     └──────────┘
     │                                   │
     ◀───────────── Replan ──────────────┘
     │
     ▼
  Response
```

- **Planner** — LLM-based reasoning. Creates structured `ActionPlan`s. Has NO
  direct tool access; can only read memory and think.
- **Gatekeeper** — Deterministic policy engine. No LLM. Checks every planned
  action against security rules, path policies, and risk classification.
- **Executor** — Runs approved actions in a sandboxed environment. Returns
  `ToolResult`s that feed back into the Planner for replanning.

Key design principles:
- The Planner never touches the filesystem or network directly
- The Gatekeeper never uses an LLM — all decisions are rule-based
- The Executor only runs actions the Gatekeeper approved
- Every decision is immutably logged for audit

---

## PGE-Trinity

### Planner (`core/planner.py` — Bible §3.1)

The Planner receives the user message plus enriched context (memory, vault,
episodes) and produces an `ActionPlan` — a structured JSON with steps.

```
Input:  System Prompt + Working Memory + User Message
Output: ActionPlan { steps: [{ tool, params, reasoning }], confidence }
```

On subsequent iterations it calls `replan()` instead of `plan()`, incorporating
tool results from the previous cycle. The Planner detects stuck loops (repeated
REPLAN text masquerading as answers) and forces termination.

### Gatekeeper (`core/gatekeeper.py` — Bible §3.2)

Every step in the ActionPlan passes through a 6-step evaluation pipeline:

1. **ToolEnforcer** — Community skills can only use their declared tools
2. **Credential Scan** — Regex detection of API keys, passwords → MASK
3. **Policy Rules** — YAML-defined rules matched by tool name + params
4. **Path Validation** — File operations must stay within `allowed_paths`
5. **Command Safety** — Blocks `rm -rf /`, `sudo`, `dd`, etc.
6. **Risk Classification** — Default categorization by tool type

Each step produces a `GateDecision` with one of four risk levels:

| Risk Level | Gate Status | Behavior |
|------------|-------------|----------|
| **GREEN**  | ALLOW       | Execute immediately |
| **YELLOW** | INFORM      | Execute + notify user |
| **ORANGE** | APPROVE     | User must confirm first |
| **RED**    | BLOCK       | Rejected, logged |

Tool classification examples:
- GREEN: `read_file`, `list_directory`, `web_search`, `get_entity`
- YELLOW: `write_file`, `edit_file`, `save_to_memory`, `run_python`
- ORANGE: `email_send`, `delete_file`, `docker_run`
- RED: Destructive shell patterns, path violations

Audit writes are buffered (threshold: 10 entries) and flushed to
`gatekeeper.jsonl`. An `atexit` handler ensures no data loss.

### Executor (`core/executor.py` — Bible §3.3)

Runs only Gatekeeper-approved actions. Supports:
- Agent-specific workspace directories
- Sandbox level overrides per agent profile
- Automatic retry for transient errors (timeout, connection) with exponential backoff
- Output capped at 50 KB per tool call

Sandbox levels (selected automatically by platform):

| Level | Platform | Isolation |
|-------|----------|-----------|
| `bwrap` | Linux | Namespaces (PID, network, filesystem) |
| `firejail` | Linux (fallback) | Application sandboxing |
| `jobobject` | Windows | Job Objects with resource limits |
| `bare` | Any (fallback) | Timeout + output limit only |

---

## Message Flow

Complete flow through `Gateway.handle_message()`:

```
1. ROUTING & SESSION
   ├── Agent Router selects agent (explicit target or LLM-based)
   ├── Session created/retrieved per (channel, user_id, agent)
   ├── Skill Registry matches message to active skills
   └── Working Memory cleared for new request

2. PARALLEL ENRICHMENT (asyncio.gather)
   ├── Context Pipeline: memory + vault + episodes → WM
   ├── Coding Classification: detect code tasks → model override
   └── Pre-search: factual queries bypass PGE if answered

3. SENTIMENT & PREFERENCES
   ├── Sentiment detection adds system hints to WM
   └── User preferences adjust verbosity

4. PGE LOOP (max N iterations)
   ├── Planner.plan() / replan()
   ├── Gatekeeper.evaluate_plan()
   ├── Executor.execute(approved_actions)
   └── Break conditions:
       ├── Single-step success → formulate response
       ├── Success threshold (30% of max iterations)
       ├── Iteration ceiling (80% of max iterations)
       ├── Failure threshold (50% of max iterations)
       └── No tool execution for 2+ iterations

5. REFLECTION & POST-PROCESSING
   ├── Reflector extracts knowledge
   ├── Memory tiers updated (episodic, semantic, procedural)
   ├── Skill usage recorded
   └── Telemetry + profiler metrics

6. SESSION PERSISTENCE
   └── Chat history persisted to SQLite SessionStore
```

---

## Initialization Phases

Gateway initialization is modular — each phase is a separate module under
`gateway/phases/`. Phases declare their attributes and dependencies:

| Phase | Module | Key Components | Depends On |
|-------|--------|----------------|------------|
| **A: Core** | `phases/core.py` | LLM client, model router, session store | — |
| **B: Security** | `phases/security.py` | Gatekeeper, audit logger, vault, red team | A |
| **C: Memory** | `phases/memory.py` | MemoryManager, hygiene, integrity | B |
| **D: Tools** | `phases/tools.py` | MCP client, browser, graph engine, A2A | A, C |
| **E: PGE** | `phases/pge.py` | Planner, Executor, Reflector, Personality | A, B, D |
| **F: Agents** | `phases/agents.py` | Skill registry, agent router, cron engine | C, D |
| **G: Compliance** | `phases/compliance.py` | Compliance framework, decision log, explainability | — |
| **H: Advanced** | `phases/advanced.py` | Monitoring, workflows, governance, prompt evolution | Multiple |

Each phase follows the pattern:
```python
def declare_*_attrs(config) -> PhaseResult:
    """Returns dict of attribute names → default values."""

async def init_*(config, **dependencies) -> PhaseResult:
    """Async initialization. Returns populated instances."""
```

Independent phases run in parallel via `asyncio.gather` where possible.

---

## Memory System

Five-tier cognitive memory architecture (Bible §4.1):

```
┌─────────────────────────────────────────────┐
│            Tier 5: Working Memory           │  ← Current session
│  Chat history, injected context, temp vars  │
├─────────────────────────────────────────────┤
│         Tier 4: Procedural Memory           │  ← How to do things
│  Learned skills, workflows, failure patterns│
├─────────────────────────────────────────────┤
│          Tier 3: Semantic Memory            │  ← Knowledge graph
│  Entities, relations, concepts (SQLite+Graph)│
├─────────────────────────────────────────────┤
│          Tier 2: Episodic Memory            │  ← What happened when
│  Daily logs, time-sensitive, recency decay  │
├─────────────────────────────────────────────┤
│           Tier 1: Core Memory              │  ← Identity
│  CORE.md, persistent, never fades           │
└─────────────────────────────────────────────┘
```

### Hybrid Search Algorithm

All memory tiers are searched simultaneously using three channels:

```
final_score = (0.50 × vector_score +
               0.30 × bm25_score   +
               0.20 × graph_score  ) × recency_decay(age, half_life=30d)
```

| Channel | Engine | Speed | Strength |
|---------|--------|-------|----------|
| **BM25** | SQLite FTS5 | ~5-20ms | Exact phrases, keywords |
| **Vector** | FAISS HNSW | ~10-50ms | Semantic similarity |
| **Graph** | PageRank + staleness | ~5-15ms | Relationship traversal |

Supporting components:
- `QueryDecomposer` — breaks complex queries into sub-queries
- `FrequencyTracker` — weights frequently-queried terms
- `EpisodicCompressor` — summarizes old episodic entries
- `SearchWeightOptimizer` — EMA-based auto-tuning of search weights

---

## Security Model

### Defense in Depth

```
User Input
  │
  ▼
┌──────────────┐
│   Sanitizer  │  ← Injection patterns, prompt injection detection
├──────────────┤
│  Gatekeeper  │  ← Risk classification, policy rules, path validation
├──────────────┤
│   Sandbox    │  ← Process isolation (bwrap/jobobject/firejail)
├──────────────┤
│ Audit Logger │  ← Immutable decision log, buffered writes
└──────────────┘
```

### Key Security Features

- **Path validation**: `.resolve()` + `.relative_to(root)` for all user-supplied paths
- **Credential masking**: Regex patterns detect API keys, passwords in tool params
- **ToolEnforcer**: Community skills can only call their declared `tools_required`
- **Sandbox resource limits**: 512 MB memory, 64 processes, 10s CPU, 50 KB output
- **Audit trail**: Every Gatekeeper decision logged with params hash
- **Red Team engine**: Automated adversarial testing (Bible §11.9)

---

## Channel Architecture

Channels connect users to the Gateway. Each channel implements:

```python
class Channel(ABC):
    name: str                              # Unique identifier
    async start(handler) -> None           # Register Gateway callback
    async stop() -> None                   # Clean shutdown
    async send(OutgoingMessage) -> None     # Send response
    async request_approval(...) -> bool    # ORANGE action confirmation
    async send_streaming_token(...) -> None # Token-by-token streaming
    async send_status(...) -> None         # Progress updates
```

Status types: `THINKING`, `SEARCHING`, `EXECUTING`, `RETRYING`, `PROCESSING`, `FINISHING`

```
User ──▶ Channel.receive() ──▶ IncomingMessage
                                     │
                              Gateway.handle_message()
                                     │
User ◀── Channel.send() ◀──── OutgoingMessage
```

Built-in channels: CLI, WebUI, Telegram, Discord, Slack, WhatsApp, Signal,
Matrix, IRC, Mattermost, Teams, Google Chat, Feishu, iMessage, Twitch, Voice, API

---

## Model Router

The Model Router (`core/model_router.py` — Bible §8.2) selects the right
LLM for each task:

```python
model = router.select_model(task_type="planning", complexity="high")
```

### Selection Priority

1. **Coding Override** (ContextVar, concurrency-safe) — if a coding task is
   detected, all non-embedding calls use the coder model
2. **Per-task overrides** — `config.model_overrides.skill_models`
3. **Default mapping**:
   - `planning, reflection` → planner model (e.g., gpt-5.2)
   - `code (high)` → coder model (e.g., qwen3-coder:30b)
   - `code (low)` → coder_fast model
   - `simple_tool_call, summarization` → executor model (e.g., gpt-5-mini)
   - `embedding` → embedding model
4. **Fallback** — planner → executor → any non-embedding model

### Tool Timeout Overrides

| Tool | Timeout |
|------|---------|
| `media_analyze_image` | 180s |
| `media_transcribe_audio`, `media_extract_text`, `media_tts` | 120s |
| `run_python` | 120s |
| All others | 30s |

---

## Context Pipeline

The Context Pipeline (`core/context_pipeline.py`) enriches Working Memory
before the Planner runs. Three searches execute in parallel:

| Search | Engine | Latency | Target |
|--------|--------|---------|--------|
| Memory | BM25 (sync) | ~5-20ms | `wm.injected_memories` |
| Vault | Full-text (async) | ~10-50ms | `wm.injected_procedures` |
| Episodes | Date-filtered (sync) | ~1-5ms | `wm.injected_procedures` |

The pipeline skips enrichment for smalltalk (short messages, greeting patterns)
and when disabled in config.

---

## Role System

Added in v0.36.0 (`core/roles.py`). Three roles with distinct behaviors:

| Aspect | Orchestrator | Worker | Monitor |
|--------|-------------|--------|---------|
| Extended thinking | Yes | No | No |
| Log output | No | Yes | Yes |
| Can spawn agents | Yes | No | No |
| Tool access | All | All | Read-only (~50 tools) |

Direction-based delegation (`a2a/delegation.py`):

| Direction | Meaning | Who can send |
|-----------|---------|-------------|
| `remember` | Write to memory | Orchestrator |
| `act` | Execute as task | Orchestrator |
| `notes` | Append to log (fire-and-forget) | All roles |

---

## Evolution Engine

The Evolution Engine enables Cognithor to autonomously learn, research, and build
new skills during idle time — with hardware-aware resource management, per-agent
budget tracking, and checkpoint/resume support.

### Architecture (4 Phases)

```
Phase 1: SystemDetector          Phase 2: Idle Learning Loop
┌──────────────────────┐         ┌─────────────────────────────────┐
│ detect_cpu/ram/gpu   │         │ IdleDetector (5min threshold)   │
│ detect_ollama/net    │         │        │                        │
│ SystemProfile        │         │  ┌─────▼──────┐                │
│ tier/mode recommend  │         │  │   Scout     │ (find gaps)    │
└──────────────────────┘         │  │   Research  │ (deep search)  │
                                 │  │   Build     │ (create skill) │
Phase 3: Budget + Resources      │  │   Reflect   │ (evaluate)     │
┌──────────────────────┐         │  └─────────────┘                │
│ ResourceMonitor      │         └─────────────────────────────────┘
│ CPU/RAM/GPU sampling │
│ should_yield()       │         Phase 4: Checkpoint/Resume
│ Per-agent CostTracker│         ┌─────────────────────────────────┐
│ Cooperative scheduling│        │ EvolutionCheckpoint (per step)  │
└──────────────────────┘         │ EvolutionResumer (load + skip)  │
                                 │ Delta snapshots                 │
                                 │ POST /evolution/resume          │
                                 └─────────────────────────────────┘
```

### Key Files

| Component | File | Responsibility |
|-----------|------|----------------|
| SystemDetector | `system/detector.py` | 8 hardware/software detection targets |
| ResourceMonitor | `system/resource_monitor.py` | Async CPU/RAM/GPU sampling, busy detection |
| IdleDetector | `evolution/idle_detector.py` | User activity tracking, idle threshold |
| EvolutionLoop | `evolution/loop.py` | Scout→Research→Build→Reflect orchestration |
| EvolutionCheckpoint | `evolution/checkpoint.py` | Step-level state persistence |
| EvolutionResumer | `evolution/resume.py` | Checkpoint-based resume logic |
| CostTracker | `telemetry/cost_tracker.py` | Per-agent LLM cost tracking + budgets |
| CheckpointStore | `core/checkpointing.py` | Generic JSON checkpoint persistence |

### Design Decisions

- **Cooperative scheduling** — The EvolutionLoop yields to user activity AND high
  system load. `ResourceMonitor.should_yield()` checks CPU > 80%, RAM > 90%,
  GPU > 80% before each step.
- **Per-agent budgets** — Each agent (scout, skill_builder) has a configurable
  daily USD limit. Budget exhaustion gracefully pauses evolution, not crashes.
- **Step-level checkpointing** — Every completed step is persisted. Interrupted
  cycles resume from the exact next step, not from scratch.
- **Delta snapshots** — Only changed data since last checkpoint is stored,
  reducing disk usage for long-running knowledge bases.

---

## OSINT / HIM Module

The Human Investigation Module provides structured OSINT capabilities:

```
HIMAgent.run(HIMRequest)
    |
    v
GDPRGatekeeper.check()
    |
    v
Collectors (parallel): GitHub, Web, arXiv, [Scholar, LinkedIn, Crunchbase, Social]
    |
    v
EvidenceAggregator: cross-verify, classify claims, detect contradictions
    |
    v
TrustScorer: 5-dimension weighted score (0-100)
    |
    v
HIMReporter: Markdown/JSON/Quick + SHA-256 signature
    |
    v
vault_save(report)
```

Located at `src/jarvis/osint/`. Exposed as 3 MCP tools: `investigate_person`, `investigate_project`, `investigate_org`.

---

## GDPR Compliance Layer

```
Request -> ComplianceEngine -> Gatekeeper -> Executor
              |
              v
         ConsentManager (SQLite)
              |
              v
         ComplianceAuditLog (JSONL, SHA-256 chain)
```

Key components:
- `security/consent.py` — Per-channel consent tracking
- `security/compliance_engine.py` — Runtime policy enforcement
- `security/compliance_audit.py` — Immutable audit log
- `security/encrypted_db.py` — SQLCipher wrapper
- `security/gdpr.py` — DataPurpose, DPIARiskLevel, ErasureManager

---

## Bible Reference Index

The codebase uses "Bible references" (§) to cross-reference architectural
decisions. Here is the complete mapping:

| Section | Topic | Key Files |
|---------|-------|-----------|
| §2.1-2.2 | Installation, First Run | `core/installer.py` |
| §3.1 | Planner | `core/planner.py` |
| §3.2 | Gatekeeper, Risk Levels | `core/gatekeeper.py` |
| §3.3 | Executor, Sandbox | `core/executor.py`, `core/sandbox.py` |
| §3.4 | PGE Cycle | `gateway/gateway.py` |
| §3.5 | Audit & Compliance | `audit/__init__.py` |
| §4.1 | Memory Tiers | `memory/manager.py` |
| §4.4 | Knowledge Graph | `memory/graph_ranking.py` |
| §4.6 | Working Memory Injection | `skills/registry.py` |
| §4.7 | Hybrid Search | `memory/search.py`, `memory/vector_index.py` |
| §5.2-5.5 | MCP Protocol | `mcp/client.py`, `mcp/server.py`, `mcp/bridge.py` |
| §6.2 | Procedural Skills | `skills/registry.py`, `skills/community/` |
| §6.4 | Self-Improvement | `skills/generator.py` |
| §7.1-7.4 | Sub-Agents, Delegation | `core/orchestrator.py`, `core/delegation.py` |
| §8 | Model Router | `core/model_router.py` |
| §9.1 | Gateway | `gateway/gateway.py` |
| §9.2 | Channels, Routing | `channels/base.py`, `core/agent_router.py` |
| §9.3 | Channel Implementations | `channels/cli.py`, `channels/telegram.py`, etc. |
| §10 | Cron & Proactive | `cron/engine.py` |
| §11 | Security | `core/gatekeeper.py`, `security/` |
| §12 | Configuration | `config.py`, `gateway/wizards.py` |
| §13 | P2P Ecosystem | `skills/circles.py`, `audit/ethics.py` |
| §14 | Marketplace Security | `skills/governance.py`, `security/cicd_gate.py` |
| §15 | Monitoring | `gateway/monitoring.py`, `healthcheck.py` |
| §16 | Explainability | `core/explainability.py`, `audit/eu_ai_act.py` |
| §17 | GDPR, Multi-Tenancy | `core/multitenant.py`, `telemetry/` |
| §18 | Performance | `core/performance.py`, `benchmark/suite.py` |
| §19 | Evolution Engine | `evolution/loop.py`, `evolution/checkpoint.py`, `evolution/resume.py`, `system/resource_monitor.py` |
