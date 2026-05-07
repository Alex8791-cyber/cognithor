# Cognithor — Complete System Architecture

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            USER INTERFACES                                   │
│                                                                              │
│  Flutter Command Center (Web/Desktop/Mobile)                                 │
│  ┌──────┬───────────┬────────┬───────┬──────────┬────────┬────────┐         │
│  │ Chat │ Dashboard │ Skills │ Admin │ Identity │ Kanban │ Leads  │         │
│  └──┬───┴─────┬─────┴───┬────┴───┬───┴────┬─────┴───┬────┴───┬────┘         │
│     │         │         │        │        │         │        │               │
│  CLI    Telegram   Discord   Slack   WhatsApp   Signal   +11 more           │
│     │         │         │        │        │         │        │               │
└─────┴─────────┴─────────┴────────┴────────┴─────────┴────────┴───────────────┘
                                    │
                              WebSocket / REST API
                              (FastAPI, Port 8741)
                                    │
┌───────────────────────────────────┴──────────────────────────────────────────┐
│                              GATEWAY                                         │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │                     6-Phase Initialization                          │     │
│  │  A: Declare attrs → B: Core (LLM, Planner, Gatekeeper, Executor)   │     │
│  │  C: Memory → D: MCP Tools (141) → E: Agents → F: Advanced         │     │
│  └─────────────────────────────────────────────────────────────────────┘     │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │                    Message Processing Pipeline                      │     │
│  │                                                                     │     │
│  │  IncomingMessage                                                    │     │
│  │       │                                                             │     │
│  │       ▼                                                             │     │
│  │  ┌──────────────────┐                                               │     │
│  │  │ Compliance Gate  │  GDPR check, rate limiting                    │     │
│  │  └────────┬─────────┘                                               │     │
│  │           ▼                                                         │     │
│  │  ┌──────────────────┐                                               │     │
│  │  │  Session Load    │  SQLite sessions, user preferences            │     │
│  │  └────────┬─────────┘                                               │     │
│  │           ▼                                                         │     │
│  │  ┌──────────────────┐                                               │     │
│  │  │ Context Pipeline │  Memory search + Vault + Episodes             │     │
│  │  │                  │  → injected into WorkingMemory                │     │
│  │  └────────┬─────────┘                                               │     │
│  │           ▼                                                         │     │
│  │  ┌──────────────────┐                                               │     │
│  │  │  Skill Matching  │  SkillRegistry.match(message)                 │     │
│  │  │                  │  → skill.body injected as procedure           │     │
│  │  └────────┬─────────┘                                               │     │
│  │           ▼                                                         │     │
│  │  ┌══════════════════════════════════════════════════════════┐       │     │
│  │  ║              PGE LOOP (iterates until done)              ║       │     │
│  │  ║                                                          ║       │     │
│  │  ║  ┌──────────┐   ┌────────────┐   ┌──────────┐          ║       │     │
│  │  ║  │ PLANNER  │──▶│ GATEKEEPER │──▶│ EXECUTOR │          ║       │     │
│  │  ║  │          │   │            │   │          │          ║       │     │
│  │  ║  │ LLM call │   │ Risk eval  │   │ MCP tool │          ║       │     │
│  │  ║  │ → JSON   │   │ GREEN/     │   │ dispatch │          ║       │     │
│  │  ║  │   plan   │   │ YELLOW/    │   │ + DAG    │          ║       │     │
│  │  ║  │          │   │ ORANGE/RED │   │ parallel │          ║       │     │
│  │  ║  └──────────┘   └────────────┘   └────┬─────┘          ║       │     │
│  │  ║                                        │                ║       │     │
│  │  ║                    Results → WorkingMemory → Replan     ║       │     │
│  │  ╚════════════════════════════════════════╧════════════════╝       │     │
│  │           │                                                         │     │
│  │           ▼                                                         │     │
│  │  ┌──────────────────┐                                               │     │
│  │  │  Post-Processing │  Reflection, skill tracking, telemetry        │     │
│  │  └────────┬─────────┘                                               │     │
│  │           ▼                                                         │     │
│  │    OutgoingMessage → Channel.send()                                 │     │
│  └─────────────────────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                  MCP TOOL LAYER (141 tools, see catalog.json)                │
│                                                                              │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐               │
│  │ Filesystem │ │   Shell    │ │    Web     │ │   Media    │               │
│  │ read/write │ │ exec/run   │ │ search/    │ │ image/     │               │
│  │ edit/list  │ │ python     │ │ fetch/read │ │ audio/tts  │               │
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘               │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐               │
│  │   Memory   │ │   Vault    │ │  Browser   │ │   Code     │               │
│  │ search/    │ │ save/read/ │ │ navigate/  │ │ analyze/   │               │
│  │ save/graph │ │ search/link│ │ click/fill │ │ run/git    │               │
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘               │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐               │
│  │   Skills   │ │  Reddit    │ │  Kanban    │ │ Computer   │               │
│  │ create/    │ │ scan/leads │ │ create/    │ │ screenshot │               │
│  │ list/install│ │ reply/     │ │ update/    │ │ click/type │               │
│  │            │ │ refine/    │ │ list       │ │ scroll/drag│               │
│  │            │ │ discover/  │ │            │ │            │               │
│  │            │ │ templates  │ │            │ │            │               │
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘               │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                          6-TIER COGNITIVE MEMORY                             │
│                                                                              │
│  ┌─────────┐ ┌──────────┐ ┌──────────┐ ┌───────────┐ ┌────────┐ ┌────────┐│
│  │  Tier 1 │ │  Tier 2  │ │  Tier 3  │ │  Tier 4   │ │ Tier 5 │ │ Tier 6 ││
│  │  CORE   │ │ EPISODIC │ │ SEMANTIC │ │PROCEDURAL │ │WORKING │ │TACTICAL││
│  │         │ │          │ │          │ │           │ │        │ │        ││
│  │ CORE.md │ │ Daily    │ │Knowledge │ │ Skills/   │ │  RAM   │ │ SQLite ││
│  │Identity │ │ logs     │ │ graph +  │ │ Procedures│ │Session │ │ Goals/ ││
│  │ Rules   │ │ Episodes │ │ SQLite   │ │ .md files │ │Context │ │Actions ││
│  └─────────┘ └──────────┘ └──────────┘ └───────────┘ └────────┘ └────────┘│
│                                                                              │
│  Search: BM25 (FTS5) + Vector (Embeddings) + Graph Traversal               │
│  Score Fusion with configurable weights + recency decay                      │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                         BACKGROUND SYSTEMS                                   │
│                                                                              │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐  │
│  │   Evolution Engine  │  │    Cron Scheduler    │  │  Reddit Lead Hunter │  │
│  │                     │  │                      │  │                     │  │
│  │ Scout → Research    │  │ reddit_scan   */30   │  │ Scan → Score →      │  │
│  │ → Build → Reflect   │  │ reply_tracker */6h   │  │ Draft → Refine →    │  │
│  │                     │  │ style_learner weekly  │  │ Queue → Reply →     │  │
│  │ Autonomous learning │  │ governance    daily   │  │ Track → Learn       │  │
│  │ during idle time    │  │ retention     daily   │  │                     │  │
│  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘  │
│                                                                              │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐  │
│  │  Knowledge Ingest   │  │   Ralph Agent Loop   │  │  Skill Lifecycle    │  │
│  │                     │  │                      │  │                     │  │
│  │ Upload → Chunk →    │  │ CONTINUE/STOP        │  │ Performance track   │  │
│  │ Deep Learn (bg) →   │  │ Multi-step autonomy  │  │ Auto-disable        │  │
│  │ KnowledgeBuilder    │  │ Budget limits        │  │ Cooldown recovery   │  │
│  │ + PDF Vision + OCR  │  │                      │  │                     │  │
│  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                            SECURITY LAYER                                    │
│                                                                              │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐               │
│  │ Gatekeeper │ │  Sandbox   │ │ Audit Trail│ │ Encryption │               │
│  │            │ │            │ │            │ │            │               │
│  │ 4 risk     │ │ Process/   │ │ SHA-256    │ │ SQLCipher  │               │
│  │ levels     │ │ Namespace/ │ │ hash chain │ │ AES-256    │               │
│  │ GREEN →    │ │ Container/ │ │ Ed25519    │ │ Fernet     │               │
│  │ YELLOW →   │ │ JobObject  │ │ RFC 3161   │ │ OS Keyring │               │
│  │ ORANGE →   │ │            │ │ GDPR Art.  │ │            │               │
│  │ RED        │ │ Path guard │ │ 15/17/33   │ │ TLS support│               │
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘               │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                          LLM PROVIDERS (17)                                  │
│                                                                              │
│  LOCAL: Ollama │ LM Studio │ vLLM │ llama.cpp                               │
│  CLOUD: OpenAI │ Anthropic │ Gemini │ Groq │ DeepSeek │ Mistral │           │
│         Together │ OpenRouter │ xAI │ Cerebras │ GitHub │ Bedrock │          │
│         HuggingFace │ Moonshot                                               │
│                                                                              │
│  UnifiedLLMClient → auto-detect backend from API keys                       │
│  Model Router → select best model per task (planning/execution/coding)       │
│  Circuit Breaker → fail-fast with recovery timeout                           │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                        REDDIT LEAD HUNTER FLOW                               │
│                                                                              │
│  ┌──────┐   ┌───────┐   ┌───────┐   ┌────────┐   ┌───────┐   ┌─────────┐  │
│  │ Scan │──▶│ Score │──▶│ Draft │──▶│ Refine │──▶│ Queue │──▶│  Reply  │  │
│  │Reddit│   │ LLM   │   │ LLM   │   │ LLM +  │   │Wizard │   │Clipboard│  │
│  │ JSON │   │ 0-100 │   │+ Style│   │Variants│   │ A/S/R │   │/Browser │  │
│  │      │   │       │   │Profile│   │+Hint   │   │       │   │/Playwrt │  │
│  └──────┘   └───────┘   └──┬────┘   └────────┘   └───────┘   └────┬────┘  │
│                             │                                       │       │
│                    Few-Shot Examples ◀──┐                           │       │
│                    Style Profile    ◀──┤                           │       │
│                                        │                           ▼       │
│  ┌──────────┐   ┌──────────┐   ┌──────┴───┐   ┌──────────┐               │
│  │ Template │   │ Feedback │   │  Learner │   │ Tracker  │               │
│  │ Manager  │   │ Dialog   │   │ (weekly) │   │  (6h)    │               │
│  │          │   │          │   │          │   │          │               │
│  │Auto-save │   │converted │   │Top vs    │   │Upvotes   │               │
│  │if score  │   │convers.  │   │Bottom    │   │Replies   │               │
│  │  > 85    │   │ignored   │   │analysis  │   │Author    │               │
│  │          │   │negative  │   │→ Profile │   │responded │               │
│  │          │   │deleted   │   │→ Tone    │   │          │               │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘               │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                      FLUTTER COMMAND CENTER (7 Tabs)                         │
│                                                                              │
│  Tab 1: CHAT           Tab 2: DASHBOARD         Tab 3: SKILLS               │
│  ┌─────────────────┐   ┌──────────────────┐     ┌──────────────────┐       │
│  │ WebSocket stream │   │ Robot Office     │     │ Skill registry   │       │
│  │ Markdown render  │   │ (live agents +   │     │ Create/edit/     │       │
│  │ Tool indicators  │   │  metrics + kanban│     │ install/publish  │       │
│  │ Token/model info │   │  dots + tooltips)│     │                  │       │
│  │ Incognito mode   │   │ Radial gauges    │     │ Community market │       │
│  │ Edit/retry/branch│   │ Status overlay   │     │                  │       │
│  └─────────────────┘   └──────────────────┘     └──────────────────┘       │
│                                                                              │
│  Tab 4: ADMIN HUB      Tab 5: IDENTITY          Tab 6: KANBAN              │
│  ┌─────────────────┐   ┌──────────────────┐     ┌──────────────────┐       │
│  │ Agents           │   │ Genesis anchors  │     │ 6-column board   │       │
│  │ Models            │   │ Personality      │     │ Drag & drop      │       │
│  │ Credentials       │   │ Cognitive state  │     │ Sub-tasks        │       │
│  │ Devices           │   │ Dream cycle      │     │ Dynamic agents   │       │
│  │ Network           │   │ Emotional state  │     │ Config dialog    │       │
│  │ Evolution         │   │                  │     │                  │       │
│  │ Documents         │   │                  │     │                  │       │
│  └─────────────────┘   └──────────────────┘     └──────────────────┘       │
│                                                                              │
│  Tab 7: LEADS                                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │ Stats bar (New/Reviewed/Replied) │ Filter (All/New/Reviewed/Replied)│     │
│  │ Lead cards (score badge, status chip, action buttons)               │     │
│  │ Detail sheet (reply editor, refine panel, performance, feedback)    │     │
│  │ Wizard mode (sequential processing, keyboard shortcuts A/S/R/I)     │     │
│  │ Scan Now FAB │ Process Queue button │ Template picker               │     │
│  └─────────────────────────────────────────────────────────────────────┘     │
│                                                                              │
│  CONFIG (20+ pages under 5 categories)                                       │
│  AI Engine: Providers │ Planner │ Executor │ Prompts                         │
│  Knowledge: Memory │ Bindings │ Web │ Vault                                  │
│  Security: Security │ Tools │ Audit │ Database                               │
│  Channels: Channel config                                                    │
│  System: General │ Language │ Logging │ Cron │ MCP │ Profile │ Budget │      │
│          Social Listening │ System                                            │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                           DATA PERSISTENCE                                   │
│                                                                              │
│  ~/.cognithor/                                                                  │
│  ├── config.yaml              Config (3-layer cascade)                       │
│  ├── .env                     API keys + secrets                             │
│  ├── .cognithor_initialized   Version marker (JSON)                          │
│  ├── logs/                    Structured JSON logs                            │
│  │   └── gatekeeper.jsonl     SHA-256 audit chain                            │
│  ├── memory/                  6-tier cognitive memory                         │
│  │   ├── core/CORE.md         Identity + rules                               │
│  │   ├── episodic/            Daily episode logs                             │
│  │   ├── semantic.db          Knowledge graph (SQLCipher)                    │
│  │   └── tactical.db          Goals + actions (SQLCipher)                    │
│  ├── vault/                   Obsidian-compatible knowledge vault            │
│  ├── skills/                  Learned + generated procedures                  │
│  ├── sessions.db              Chat sessions (SQLCipher)                      │
│  ├── kanban.db                Task board (SQLCipher)                          │
│  ├── leads.db                 Reddit leads + performance + templates         │
│  ├── cache/                   Web search cache                               │
│  └── browser_data/            Playwright cookies (Reddit session)             │
│                                                                              │
│  All SQLite databases encrypted with SQLCipher (AES-256)                     │
│  Keys stored in OS Keyring (not on disk)                                     │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Statistics

> **Counts grow over time.** The values below are a snapshot taken at the v0.99.0 cut. The authoritative live counts are: `pytest --collect-only -q tests/ \| tail -5` for tests, [`docs/integrations/catalog.json`](integrations/catalog.json) for MCP tools, `find src/cognithor -name "*.py" \| xargs wc -l` for source LOC.

| Metric | Value (v0.99.0 cut) |
|--------|---------------------|
| Python Source | ~257,000 LOC |
| Python Tests | ~224,000 LOC |
| Flutter UI | ~66,000 LOC |
| MCP Tools | 141 (auto-generated → `docs/integrations/catalog.json`) |
| REST Endpoints | 60+ |
| Channels | 17 (chat, voice, REST, CLI) |
| LLM Providers | 19 (Ollama, LM Studio, vLLM, llama-cpp + 15 cloud + Claude Code) |
| Tests | 17,000+ test functions |
| Coverage Gate | 89% (`--cov-fail-under=89` in CI) |
| Lint Errors | 0 (`ruff check src/cognithor/ tests/`) |
| Flutter Issues | 0 (`flutter analyze`) |
| i18n Languages | 4 (EN/DE/ZH/AR) |
| i18n Keys | 900+ |
| Operational-Trust Ledgers | 6 (Provenance, Permission-Scopes, Tool-Fingerprints, Cloud-Escalation, Cost, Migration) |
| Audit Categories | 3 (`SYSTEM`, `REFLECTION`, `SKILL`) — see [`docs/operational_trust.md`](operational_trust.md) |
| Video Renderers | 1 default (HyperFrames, Apache-2.0) via pluggable `cognithor.video.RendererABC` |

See also: [`docs/hashline-guard.md`](hashline-guard.md) for audit-chain integrity, [`docs/operational_trust.md`](operational_trust.md) for the TRUST-1..10 reference, [`../ARCHITECTURE.md#video-composition--rendering-hyperframes`](../ARCHITECTURE.md) for the video pipeline.
