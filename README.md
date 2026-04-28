<p align="center">
  <h1 align="center">Cognithor &middot; Agent OS</h1>
  <p align="center">
    <strong>A local-first, autonomous agent operating system for AI experimentation and personal automation.</strong>
  </p>
  <p align="center">
    <em>Cognition + Thor — Intelligence with Power</em>
  </p>
  <p align="center">
    <a href="https://cognithor.ai"><strong>cognithor.ai</strong></a> &middot;
    <a href="https://cognithor.ai/packs">Agent Packs</a> &middot;
    <a href="https://cognithor.ai/docs">Docs</a> &middot;
    <a href="https://cognithor.ai/blog">Blog</a>
  </p>
  <p align="center">
    <a href="#llm-providers">19 LLM Providers</a> &middot; <a href="#channels">18 Channels</a> &middot; <a href="#6-tier-cognitive-memory">6-Tier Memory</a> &middot; <a href="#4-channel-hybrid-search">4-Channel Search</a> &middot; <a href="#knowledge-vault">Knowledge Vault</a> &middot; <a 
    href="#security">Security</a> &middot; <a href="LICENSE">Apache 2.0</a>
  </p>
  <p align="center">
    <a href="https://github.com/Alex8791-cyber/cognithor/stargazers"><img src="https://img.shields.io/github/stars/Alex8791-cyber/cognithor?style=flat-square&color=yellow" alt="GitHub Stars"></a>
    <img src="https://img.shields.io/badge/status-Beta%20%2F%20Experimental-orange?style=flat-square" alt="Status: Beta">
    <a href="#quick-start"><img src="https://img.shields.io/badge/python-%3E%3D3.12-blue?style=flat-square" alt="Python"></a>
    <a href="https://github.com/Alex8791-cyber/cognithor/actions/workflows/ci.yml"><img src="https://github.com/Alex8791-cyber/cognithor/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
    <a href="https://pypi.org/project/cognithor/"><img src="https://img.shields.io/pypi/v/cognithor.svg?style=flat-square" alt="PyPI"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue?style=flat-square" alt="License"></a>
    <a href="https://github.com/Alex8791-cyber/cognithor/releases"><img src="https://img.shields.io/github/v/release/Alex8791-cyber/cognithor?style=flat-square&color=blue" alt="Release"></a>
  </p>
</p>

> **Pre-v1.0 Beta** — Cognithor is under active development. APIs may change between releases. Not recommended for production customer-facing deployments. Bug reports and feedback welcome via [Issues](../../issues).
>
> While the test suite is extensive (13,000+ tests, 89% coverage), the project has not been battle-tested in production environments. Expect rough edges, breaking changes between versions, and some German-language strings in system prompts and error messages. See [Status & Maturity](#status--maturity) for details. For non-technical users, wait until version 1.0.0 for stable long-term support.

  [![clawdboard](https://clawdboard.ai/api/badge/Alex8791-cyber)](https://clawdboard.ai/user/Alex8791-cyber)

<p align="center">
  <a href="https://clawdboard.ai/recap/6fd37b26-7e41-4b0f-958a-3f2580427ccf"><strong>Weekly Recap: Rank #1 | $1,644 spent vibe-engineering</strong></a>
</p>

> **Vibe-Engineered, not vibe-coded.** Cognithor is not a weekend hack held together by AI-generated spaghetti. Every module follows a deliberate architecture (PGE-Trinity, 6-phase gateway init, 3-layer security), backed by 13,000+ tests, structured plans, spec compliance reviews, and code quality gates. The AI writes the code — but a human engineers the system. There's a difference.

---

## Why Cognithor?

Most AI assistants send your data to the cloud. Cognithor runs entirely on your machine — with Ollama or LM Studio, no API keys required. Cloud providers are optional, not mandatory.

It replaces a patchwork of tools with one integrated system: 18 channels, 145+ MCP tools, 6-tier memory with 4-channel hybrid search, knowledge vault, voice, browser automation, Computer Use, cross-platform social listening, and more — all wired together from day one. 13,000+ tests at 89% coverage keep it honest. See [Status & Maturity](#status--maturity) for what that does and does not guarantee.

**In plain terms:** Cognithor is an AI assistant that lives entirely on your computer. You talk to it through your terminal, a web UI, Telegram, Discord, or any of 18 supported channels — and it talks back, remembers what you said last week, and acts on your behalf. It can search the web, write and edit files, run shell commands, control your browser, automate your desktop (clicking, typing, reading windows), manage your calendar, and learn new skills over time. Think of it as a local, private, self-improving Jarvis.

Unlike cloud-based assistants, Cognithor keeps all your data on your machine. Your conversations, memories, documents, and credentials never leave your hardware unless you explicitly configure a cloud LLM provider. It works fully offline with Ollama or LM Studio, and it encrypts everything at rest with SQLCipher (AES-256). If privacy matters to you — and it should — this is the architecture you want.

What makes it different from other local AI tools is that Cognithor is not just a chatbot. It is an agent operating system: it plans multi-step tasks, evaluates its own results, learns from mistakes, and improves autonomously. It can control your desktop through Computer Use (screenshots, clicks, keystrokes, window automation), compete in ARC-AGI-3 reasoning benchmarks, and manage a marketplace of community-contributed skills. It is built to grow with you.

---

## Status & Maturity

**Cognithor is Beta / Experimental software.** It is under rapid, active development.

| Aspect | Status |
|--------|--------|
| **Core agent loop (PGE)** | Stable — well-tested and functional |
| **Memory system** | Stable — 6-tier architecture works reliably |
| **CLI channel** | Stable — primary development interface |
| **Flutter Command Center** | Beta — Sci-Fi aesthetic, cross-platform, GEPA pipeline visualization, Robot Office pathfinding, 20+ config pages, chat, voice, learning dashboard |
| **Messaging channels** (Telegram, Discord, etc.) | Beta — basic flows work, edge cases may break |
| **Voice mode / TTS** | Alpha — experimental, hardware-dependent |
| **Browser automation** | Stable — Playwright-based, CAPTCHA solving, stealth mode |
| **Computer Use** | Stable — 6 phases (Vision, Agent Loop, Planner Intelligence, Security, Robustness, UI Automation) |
| **ARC-AGI-3 Benchmark** | Beta — 13/25 games solved (24 levels), 4 solver families incl. SmartExplorer |
| **Skill Marketplace** | Stable — GitHub registry, 5-check validation, publisher verification |
| **Windows UI Automation** | Beta — pywinauto UIA for exact element coordinates |
| **Deployment (Docker, bare-metal)** | Beta — tested on limited configurations |
| **SSH Remote Execution** | Beta — tested against Docker containers, key-based auth |
| **Evolution Engine** | Stable — all 6 phases complete, autonomous deep learning with quality self-examination, GDPR-compliant |
| **Autonomous Task Framework** | Beta — task decomposition, self-evaluation, recurring scheduling |
| **Background Process Manager** | Beta — 6 MCP tools, 5-method ProcessMonitor, SQLite persistence |
| **Multi-Agent System** | Beta — 5 specialized agents with model/temperature/top_p overrides |
| **Audit & Compliance** | Beta — HMAC + Ed25519 signatures, RFC 3161 TSA, GDPR Art. 15/33, WORM-ready |
| **Enterprise features** (GDPR, A2A, Governance) | Stable — GDPR 100% user rights, consent management, SQLCipher encryption, audit trail |
| **Encryption at Rest** | Stable — SQLCipher (AES-256) for all databases, Fernet for files, OS Keyring key management |
| **Cross-Platform Social Listening** | Beta — Reddit + Hacker News + Discord scanning, LLM-scored leads, unified MCP tools |
| **Hierarchical Document Reasoning** | Beta — Tree-based retrieval for PDF/DOCX/HTML/Markdown, LLM-navigated section selection |
| **CAG Layer (Cache-Augmented Generation)** | Beta — Deterministic prefix generation for LLM KV-cache reuse, prefix + native builders |
| **CLI Config TUI** | Stable — Interactive terminal config editor (rich + prompt_toolkit), model discovery |
| **AST-Based Security** | Stable — Python AST + bashlex shell analysis replacing regex-based guards |
| **OSINT / HIM Module** | Beta — person/project/org investigation with trust scoring |
| **Observer Audit Layer** | Stable — post-response LLM quality check across 4 dimensions (hallucinations, sycophancy, laziness, tool-ignorance); triggers regeneration or full PGE re-loop; fails open |

**What the test suite covers:** Unit tests, integration tests, real-life scenario tests, and live Ollama tests for all modules. The 13,000+ tests verify code correctness in controlled environments.

**What the test suite does NOT cover:** Real-world deployment scenarios, network edge cases, long-running stability, multi-user load, hardware-specific voice/GPU issues, or actual LLM response quality.

**Important notes for users:**
- This project is developed by a solo developer with AI assistance. Code is human-reviewed, but the pace is fast.
- Breaking changes may occur between minor versions. Pin your version if stability matters.
- The default language is **German**, switchable to **English** via the Flutter Command Center or `config.yaml`. See [Language & Internationalization](#language--internationalization).
- For production use, thorough testing in your specific environment is strongly recommended.
- Bug reports and contributions are welcome — see [Issues](https://github.com/Alex8791-cyber/cognithor/issues).

---

> **Cognithor** is a fully local, Ollama/LM Studio-powered, autonomous agent operating system that acts as your personal AI assistant. All data stays on your machine — no cloud, no mandatory API keys, full GDPR compliance. It supports tasks ranging from research, project management, and knowledge organization to file management and automated workflows. Optional cloud LLM providers (OpenAI, Anthropic, Gemini, and 12 more) can be enabled with a single API key. Users can add custom skills and rules to tailor the agent to their needs.


## Table of Contents

- [Why Cognithor?](#why-cognithor)
- [Status & Maturity](#status--maturity)
- [Highlights](#highlights)
- [Architecture](#architecture)
- [LLM Providers](#llm-providers)
- [Channels](#channels)
- [Quick Start](#quick-start) (under 5 minutes)
- [Configuration](#configuration)
- [Security](#security)
- [MCP Tools](#mcp-tools)
- [Tests](#tests)
- [Deployment](#deployment)
- [Language & Internationalization](#language--internationalization)
- [License](#license)
- [What's New](#whats-new)

## Highlights

- **19 LLM Providers** — Ollama (local), LM Studio (local), vLLM (local), llama-cpp-python (local), OpenAI, Anthropic, Google Gemini, Groq, DeepSeek, Mistral, Together AI, OpenRouter, xAI (Grok), Cerebras, GitHub Models, AWS Bedrock, Hugging Face, Moonshot/Kimi, Claude Code — plus any custom OpenAI-compatible endpoint
- **18 Communication Channels** — CLI, Web UI, REST API, Telegram, Discord, Slack, WhatsApp, Signal, iMessage, Microsoft Teams, Matrix, Google Chat, Mattermost, Feishu/Lark, IRC, Twitch, Voice (STT/TTS)
- **6-Tier Cognitive Memory** — Core identity, episodic logs, semantic knowledge graph, procedural skills, working memory, tactical memory
- **4-Channel Hybrid Search** — BM25 full-text + vector embeddings + knowledge graph traversal + hierarchical document reasoning with score fusion
- **PGE Architecture** — Planner (LLM) -> Gatekeeper (deterministic policy engine) -> Executor (sandboxed)
- **Crew-Layer (v0.93.0)** — high-level Multi-Agent API (`from cognithor import Crew, CrewAgent, CrewTask`) on top of PGE-Trinity. Async kickoff with parallel fan-out, idempotent replay, guardrails, 5 templates (research, customer-support, data-analyst, content, versicherungs-vergleich).
- **8-page Quickstart** — [`docs/quickstart/`](docs/quickstart/) — from `pip install` to your first Crew in under 10 minutes, bilingual (DE primary + EN). Includes runnable examples for first-crew, first-tool, first-skill, guardrails, and PKV report.
- **Security** — Platform-adaptive sandbox (bubblewrap on Linux, subprocess+timeout fallback), AST-based Python/Shell code analysis (Python `ast.NodeVisitor` + `bashlex` parser), SHA-256 audit chain, credential vault, runtime token encryption (Fernet AES-256), Gatekeeper policy engine with GREEN/YELLOW/ORANGE/RED risk classification (not independently audited — see [Status & Maturity](#status--maturity))
- **Knowledge Vault** — Obsidian-compatible Markdown vault with YAML frontmatter, tags, `[[backlinks]]`, full-text search
- **Document Analysis** — LLM-powered structured analysis of PDF/DOCX/HTML (summary, risks, action items, decisions)
- **Video Input** — Attach local videos (`.mp4` / `.webm` / `.mov` / `.mkv` / `.avi`) or paste direct video URLs; Qwen3.6-27B (or any video-capable VLM) analyzes them end-to-end via vLLM's native `video_url` content type. Adaptive frame sampling (fps=3 for short clips, `num_frames=32` for long) via `ffprobe`. Single video per turn, served from a 127.0.0.1-only HTTP file server, 24h auto-cleanup. Requires vLLM backend — Windows installer bundles LGPL ffmpeg. See [`docs/vllm-user-guide.md`](docs/vllm-user-guide.md).
- **Model Context Protocol (MCP)** — 145+ tools across 14 modules (filesystem, shell, memory, web, browser, media, vault, synthesis, code, skills, documents, reddit, social, kanban, identity) + A2A delegation
- **Computer Use** — Complete desktop automation: screenshots, clicking, typing, scrolling, dragging, Windows UI Automation via pywinauto for exact element coordinates, 3-layer security, adaptive wait
- **ARC-AGI-3 Benchmark Agent** — Compete in ARC Prize 2026: 13/25 games solved (24 levels), 4 solver families (ClusterClick, SequenceClick+SimA*, KeyboardDFS, SmartExplorer), persistent game profiles, multimodal vision (qwen3-vl)
- **Distributed Locking** — Redis-backed (with file-based fallback) locks for multi-instance deployments
- **Durable Message Queue** — SQLite-backed persistent queue with priorities, DLQ, and automatic retry
- **Prometheus Metrics** — /metrics endpoint with Grafana dashboard for production observability
- **Skill Marketplace** — SQLite-persisted skill marketplace with ratings, search, and REST API
- **Community Skill Marketplace** — GitHub-hosted registry with publisher verification (4 trust levels), 5-check validation pipeline, ToolEnforcer runtime sandboxing, async install/search/report
- **Telegram Webhook** — Polling + webhook mode with sub-100ms latency
- **Auto-Dependency Loading** — Missing optional packages detected and installed at startup
- **Agent-to-Agent Protocol (A2A)** — Linux Foundation RC v1.0 with full JSON-RPC 2.0 server/client, Planner-level delegation via MCP tools, auto-discovery, SSE streaming
- **Integrated Chat** — Full chat page in the Flutter Command Center with WebSocket streaming, tool indicators, canvas panel, approval banners, and voice mode
- **Flutter Command Center** — Cross-platform UI (Flutter 3.41, Web/Desktop/Mobile) with Sci-Fi aesthetic, GEPA pipeline visualization, Robot Office pathfinding, 20+ editable config pages, Observe panel, Knowledge Graph, Voice Mode, Learning Dashboard, Light/Dark theme, 4-language i18n
- **Active Learning & Curiosity** — CuriosityEngine detects knowledge gaps, KnowledgeConfidenceManager with time decay and feedback, ActiveLearner processes files in background during idle time
- **Human Feel** — Personality Engine (warmth, humor, greetings), sentiment detection (frustrated/urgent/confused/positive), user preference learning, real-time status callbacks, user-friendly German error messages
- **Auto-Detect Channels** — Channels activate automatically when tokens are present in `.env` — no manual config flags needed
- **Knowledge Synthesis** — Meta-analysis across Memory + Vault + Web with LLM fusion: `knowledge_synthesize` (full synthesis with confidence ratings), `knowledge_contradictions` (fact-checking), `knowledge_timeline` (causal chains), `knowledge_gaps` (completeness score + research suggestions)
- **Adaptive Context Pipeline** — Automatic context enrichment before every Planner call: BM25 memory search + vault full-text search + recent episodes, injected into WorkingMemory in <50ms
- **Security Hardening** — Runtime token encryption (Fernet AES-256) across all channels, TLS support for webhook servers, file-size limits on all upload/processing paths, persistent session mappings in SQLite
- **One-Click Start** — Double-click `start_cognithor.bat` -> browser opens -> click **Power On** -> done
- **Enhanced Web Research** — 4-provider search fallback (SearXNG -> Brave -> Google CSE -> DuckDuckGo), Jina AI Reader for JS-heavy sites, domain filtering, source cross-checking
- **Procedural Learning** — Reflector auto-synthesizes reusable skills from successful sessions
- **DAG Workflow Engine** — Directed acyclic graph execution with parallel branches, conditional edges, cycle detection, automatic retry. Now wired into the Executor for parallel tool execution
- **Distributed Workers** — Capability-based job routing, health monitoring, failover, dead-letter queue
- **Multi-Agent Collaboration** — Debate, voting, and pipeline patterns for agent teams
- **i18n Language Packs** — JSON-based internationalization with SHA-256 integrity verification, German and English included, extensible to any language
- **Tool Sandbox Hardening** — Per-tool resource limits, network guards, escape detection (8 attack categories)
- **GDPR Compliance Toolkit** — Data processing logs (Art. 30), retention enforcement, right-to-erasure (Art. 17), audit export
- **ARC-AGI-3 Benchmark** — Compete in the [ARC Prize 2026](https://arcprize.org/competitions/2026) ($2M+ prize pool) via the `arc/` module: hybrid agent (algorithmic + LLM + CNN), 3 MCP tools (`arc_play`, `arc_status`, `arc_replay`), CLI mode, swarm mode. `pip install cognithor[arc]`
- **Multi-Agent Benchmark Scaffold** — Reproducible benchmark harness in the top-level `cognithor_bench/` package: own `pyproject.toml`, `cognithor-bench` CLI, Cognithor + AutoGen adapters
- **Deterministic Replay** — Record and replay agent executions with what-if analysis and diff comparison
- **Agent SDK** — Decorator-based agent registration (`@agent`, `@tool`, `@hook`), project scaffolding
- **Plugin Remote Registry** — Remote manifests with SHA-256 checksums, dependency resolution, install/update/rollback
- **Cross-Platform Social Listening** — Reddit + Hacker News (Firebase/Algolia) + Discord (REST API v10) scanning with LLM lead scoring. 2 unified MCP tools (`social_scan`, `social_leads`), configurable per-platform intervals, auto-post opt-in
- **Hierarchical Document Reasoning** — Tree-based retrieval (4th search channel): 5 parsers (Markdown, PDF, DOCX, HTML, plaintext) build heading trees, LLM-navigated section selection, no embedding required
- **CAG Layer (Cache-Augmented Generation)** — Deterministic prefix generation for LLM KV-cache reuse. Prefix builder (Ollama-compatible) + native llama.cpp builder, content normalization, hit-rate metrics
- **CLI Config TUI** — Interactive terminal config editor (rich + prompt_toolkit): model discovery from live providers, section navigation, validation, save/discard
- **AST-Based Security Guards** — Python `ast.NodeVisitor` + `bashlex` shell parser replacing regex-based code analysis. Detects imports, subprocess calls, eval, network access, file operations at the syntax tree level
- **`_safe_call()` Pattern** — Unified error handling replacing silent `except Exception: pass`. Failure registry with per-function tracking, async variant, circuit-breaker integration
- **uv Installer Support** — Automatic uv detection for 10x faster installs, transparent pip fallback
- **Observer Audit Layer** — Every response audited against 4 quality dimensions (hallucination, sycophancy, laziness, tool-ignorance) with differentiated retry strategies. Runs locally with qwen3:32b.
- **Integrations Catalog** — Auto-generated from `src/cognithor/mcp/` — see [`docs/integrations/catalog.json`](docs/integrations/catalog.json). DACH-specific: sevDesk REST connector (accounting).
- **Competitive Analysis** — [`docs/competitive-analysis/`](docs/competitive-analysis/README.md) — Cognithor vs AutoGen / MAF / LangGraph / CrewAI.
- **13,000+ tests** · **89% coverage** · **0 lint errors** · **0 CodeQL alerts**

## Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│      Flutter Command Center (Dart/Flutter 3.41, cross-platform)   │
│   Sci-Fi UI · GEPA Pipeline · Robot Office · 18 Config Pages     │
│   Chat · Voice · Observe · Knowledge Graph · Learning Dashboard   │
├───────────────────────────────────────────────────────────────────┤
├───────────────────────────────────────────────────────────────────┤
│         Prometheus /metrics · Grafana Dashboard                    │
├───────────────────────────────────────────────────────────────────┤
│           REST API (FastAPI, 48+ endpoints, port 8741)            │
├───────────────────────────────────────────────────────────────────┤
│                       Channels (18)                                │
│   CLI · Web · Telegram (poll+webhook) · Discord · Slack           │
│   WhatsApp · Signal · iMessage · Teams · Matrix · Voice · ...     │
├───────────────────────────────────────────────────────────────────┤
│                     Gateway Layer                                  │
│   Session · Agent Loop · Distributed Lock · Status Callbacks       │
│   Personality · Sentiment · User Preferences                       │
├───────────────────────────────────────────────────────────────────┤
│        Durable Message Queue (SQLite, priorities, DLQ)             │
├───────────────────────────────────────────────────────────────────┤
│           Context Pipeline (Memory · Vault · Episodes)             │
├─────────────┬──────────────┬──────────────────────────────────────┤
│  Planner    │  Gatekeeper  │  Executor                            │
│  (LLM)      │  (Policy)    │  (Sandbox)                           │
├─────────────┴──────────────┴──────────────────────────────────────┤
│  DAG Workflow Engine · Workflow Adapter                               │
├───────────────────────────────────────────────────────────────────┤
│                   MCP Tool Layer (145+ tools)                        │
│   Filesystem · Shell · Memory · Web · Browser · Media · Vault      │
│   Synthesis · Skills · Social Listening · Kanban · Documents        │
├───────────────────────────────────────────────────────────────────┤
│               Multi-LLM Backend Layer (18)                         │
│   Ollama · OpenAI · Anthropic · Gemini · Groq · DeepSeek           │
│   Mistral · Together · OpenRouter · xAI · Cerebras · ...           │
├───────────────────────────────────────────────────────────────────┤
│               6-Tier Cognitive Memory                               │
│   Core · Episodic · Semantic · Procedural · Working · Tactical      │
├───────────────────────────────────────────────────────────────────┤
│         Infrastructure: Redis/File Distributed Lock                 │
│         SQLite Durable Queue · Prometheus Telemetry                 │
│         Worker Pool · GDPR Compliance · Deterministic Replay        │
└───────────────────────────────────────────────────────────────────┘
```

### PGE Trinity (Planner -> Gatekeeper -> Executor)

Every user request passes through three stages:

1. **Planner** — LLM-based understanding and planning. Analyzes the request, searches memory for relevant context, creates structured action plans with tool calls. Supports re-planning on failures.
2. **Gatekeeper** — Deterministic policy engine. Validates every planned tool call against security rules (risk levels GREEN/YELLOW/ORANGE/RED, sandbox policy, parameter validation). No LLM, no hallucinations, no exceptions.
3. **Executor** — Executes approved actions via DAG-based parallel scheduling (independent actions run concurrently in waves). Shell commands run isolated (Process -> Namespace -> Container), file access restricted to allowed paths.

### 6-Tier Cognitive Memory

| Tier | Name | Persistence | Purpose |
|------|------|------------|---------|
| 1 | **Core** | `CORE.md` | Identity, rules, personality |
| 2 | **Episodic** | Daily log files | What happened today/yesterday |
| 3 | **Semantic** | Knowledge graph + SQLite | Customers, products, facts, relations |
| 4 | **Procedural** | Markdown + frontmatter | Learned skills and workflows |
| 5 | **Working** | RAM (volatile) | Active session context |
| 6 | **Tactical** | SQLite | Active goals, pending actions, rollback |

Memory search uses a 4-channel hybrid approach: **BM25** (full-text search with FTS5, optimized for German compound words) + **Vector Search** (Ollama embeddings, cosine similarity) + **Graph Traversal** (entity relations) + **Hierarchical Document Reasoning** (tree-based section navigation for long documents). Score fusion with configurable weights and recency decay.

### Knowledge Vault

In addition to the 6-tier memory, Cognithor includes an **Obsidian-compatible Knowledge Vault** (`~/.cognithor/vault/`) for persistent, human-readable notes:

- **Folder structure**: `recherchen/`, `meetings/`, `wissen/`, `projekte/`, `daily/`
- **Obsidian format**: YAML frontmatter (title, tags, sources, dates), `[[backlinks]]`
- **6 tools**: `vault_save`, `vault_search`, `vault_list`, `vault_read`, `vault_update`, `vault_link`
- Open the vault folder directly in [Obsidian](https://obsidian.md) for graph visualization

### Reflection & Procedural Learning

After completed sessions, the Reflector evaluates results, extracts facts for semantic memory, and identifies repeatable patterns as procedure candidates. Learned procedures are automatically suggested for future similar requests.

### Computer Use Pipeline

```
User Goal → CUTaskDecomposer → Sub-Tasks
  → UIA (exact coords) / Vision (fallback)
  → 3-Layer Tool Enforcement
  → Adaptive Wait → Next Iteration
```

Cognithor can control your desktop like a human: take screenshots, analyze them with a vision model, click at pixel coordinates, type text, scroll, and drag. Phase 3 adds Windows UI Automation via pywinauto, which reads the OS accessibility tree to get exact element coordinates without relying on vision alone. A 3-layer security model (allowlist + gatekeeper + tool enforcer) ensures destructive operations require explicit approval.

### Architecture Decision Records

- [`docs/adr/0001-pge-trinity-vs-group-chat.md`](docs/adr/0001-pge-trinity-vs-group-chat.md)
  — Why Cognithor uses Planner/Gatekeeper/Executor instead of conversational
  GroupChat patterns.

### Comparison with Other Frameworks

- [`docs/competitive-analysis/`](docs/competitive-analysis/README.md) —
  Cognithor vs AutoGen, Microsoft Agent Framework, LangGraph, CrewAI.

## LLM Providers

Cognithor auto-detects your backend from API keys. Set one key and models are configured automatically:

| Provider | Backend Type | Config Key | Models (Planner / Executor) |
|----------|-------------|------------|----------------------------|
| **Ollama** (local) | `ollama` | *(none needed)* | qwen3:32b / qwen3:8b |
| **LM Studio** (local) | `lmstudio` | *(none needed)* | *(your loaded models)* |
| **OpenAI** | `openai` | `openai_api_key` | gpt-5.2 / gpt-5-mini |
| **Anthropic** | `anthropic` | `anthropic_api_key` | claude-opus-4-6 / claude-haiku-4-5 |
| **Google Gemini** | `gemini` | `gemini_api_key` | gemini-2.5-pro / gemini-2.5-flash |
| **Groq** | `groq` | `groq_api_key` | llama-4-maverick / llama-3.1-8b-instant |
| **DeepSeek** | `deepseek` | `deepseek_api_key` | deepseek-chat (V3.2) |
| **Mistral** | `mistral` | `mistral_api_key` | mistral-large-latest / mistral-small-latest |
| **Together AI** | `together` | `together_api_key` | Llama-4-Maverick / Llama-4-Scout |
| **OpenRouter** | `openrouter` | `openrouter_api_key` | claude-opus-4.6 / gemini-2.5-flash |
| **xAI (Grok)** | `xai` | `xai_api_key` | grok-4-1-fast-reasoning / grok-4-1-fast |
| **Cerebras** | `cerebras` | `cerebras_api_key` | gpt-oss-120b / llama3.1-8b |
| **GitHub Models** | `github` | `github_api_key` | gpt-4.1 / gpt-4.1-mini |
| **AWS Bedrock** | `bedrock` | `bedrock_api_key` | claude-opus-4-6 / claude-haiku-4-5 |
| **Hugging Face** | `huggingface` | `huggingface_api_key` | Llama-3.3-70B / Llama-3.1-8B |
| **Moonshot/Kimi** | `moonshot` | `moonshot_api_key` | kimi-k2.5 / kimi-k2-turbo |
| **Claude Code** | `claude-code` | *(uses Claude Code CLI)* | *(Claude Code model)* |

```yaml
# ~/.cognithor/config.yaml — just set one key, everything else is auto-configured
gemini_api_key: "AIza..."
# That's it. Backend, models, and operation mode are auto-detected.

# Or use LM Studio (local, no API key needed):
llm_backend_type: "lmstudio"
# lmstudio_base_url: "http://localhost:1234/v1"  # default
```

## Channels

| Channel | Protocol | Features |
|---------|----------|----------|
| **CLI** | Terminal REPL | Rich formatting, streaming, `/commands`, status feedback |
| **Web UI** | WebSocket | Real-time streaming, voice recording, file upload, dark theme, status events |
| **REST API** | FastAPI + SSE | Programmatic access, server-sent events |
| **Telegram** | Bot API (poll + webhook) | Text, voice messages (Whisper STT), photos, documents, webhook mode (<100ms), typing indicator |
| **Discord** | Gateway + REST | Embeds, reactions, thread support, typing indicator |
| **Slack** | Socket Mode | Block Kit, interactive buttons, thread support |
| **WhatsApp** | Meta Cloud API | Text, media, location, contacts |
| **Signal** | signal-cli bridge | Encrypted messaging, attachments |
| **iMessage** | PyObjC (macOS) | Native macOS integration |
| **Microsoft Teams** | Bot Framework v4 | Adaptive cards, approvals |
| **Matrix** | matrix-nio | Federated, encrypted rooms |
| **Google Chat** | Chat API | Workspace integration |
| **Mattermost** | REST API | Self-hosted team chat |
| **Feishu/Lark** | Bot API | ByteDance enterprise messaging |
| **IRC** | IRC protocol | Classic internet relay chat |
| **Twitch** | TwitchIO | Live stream chat integration |
| **Voice** | Whisper + Piper + ElevenLabs | STT, TTS, wake word (Levenshtein), Konversationsmodus, Piper TTS (Thorsten Emotional) |

## Demo

```bash
python demo.py           # Full experience (~3 minutes)
python demo.py --fast    # Speed run (~15 seconds)
```

## Quick Start

**Under 5 minutes from download to running agent.**

### Option A: Windows Installer (recommended for Windows users)

1. **Download** `CognithorSetup-{version}.exe` from the [latest release](https://github.com/Alex8791-cyber/cognithor/releases/latest)
2. **Run the installer** — it bundles Python, Ollama, Flutter UI, and all dependencies
3. **Launch** via Desktop shortcut or Start Menu
4. **Done** — browser opens at `http://localhost:8741`

> The installer auto-detects your GPU, downloads the right models, and runs a health check. No prerequisites needed.

### Option B: Linux / macOS / Source Install

```bash
# Clone
git clone https://github.com/Alex8791-cyber/cognithor.git
cd cognithor

# Linux/macOS: Interactive installer (venv, Ollama, models, systemd)
chmod +x install.sh && ./install.sh

# Windows (from source): Double-click install.bat

# Or manual: pip install
pip install -e ".[all,dev]"
```

### Option C: PyPI (any platform)

```bash
pip install cognithor               # Core only: PGE, CLI, Ollama
pip install cognithor[telegram]     # + Telegram channel
pip install cognithor[web]          # + Web UI (Flutter Command Center)
pip install cognithor[all]          # Everything (large install)
ollama pull qwen3:8b                # Pull a model
cognithor                           # Start

# Upgrade to a specific version:
pip install cognithor[all]==0.92.2

# Upgrade to latest:
pip install --upgrade cognithor[all]
```

### Prerequisites (Option B/C only)

- Python >= 3.12
- [Ollama](https://ollama.ai) (local, free) **or** any cloud LLM API key (OpenAI, Anthropic, Gemini, etc.)

### After Installation: Pull Models

```bash
ollama pull qwen3:32b              # Planner (20 GB VRAM)
ollama pull qwen3:8b               # Executor (6 GB VRAM)
ollama pull qwen3-embedding:0.6b   # Embeddings (500 MB VRAM)
```

No GPU? Use `qwen3:8b` for both, or set a cloud API key — models are auto-configured.

> **Uninstall:** Windows: run `uninstall.bat`. Linux: `./install.sh --uninstall`. PyPI: `pip uninstall cognithor`.

### Step 2: Pull an LLM (~2 min)

```bash
ollama pull qwen3:32b           # Planner (20 GB VRAM)
ollama pull qwen3:8b            # Executor (6 GB VRAM)
ollama pull qwen3-embedding:0.6b    # Embeddings (500 MB VRAM)
# Optional:
ollama pull qwen3-coder:30b     # Code tasks (20 GB VRAM)
```

No GPU? Use smaller models (`qwen3:8b` for both) or a cloud provider — just set one API key.

### Step 3: Start (~10 sec)

**Option A: One-Click (Windows)** — includes a pre-built Web UI, no Node.js needed

```
Double-click  start_cognithor.bat  ->  Browser opens  ->  Click "Power On"  ->  Done.
```

> The launcher auto-installs Python and Ollama via winget if missing. Node.js is only needed for UI development (`npm run dev`).

**Option B: CLI**

```bash
cognithor                          # Interactive CLI
python -m cognithor                   # Same thing (always works, no PATH needed)

python -m cognithor --lite            # Lite mode: qwen3:8b only (6 GB VRAM)
python -m cognithor --no-cli          # Headless mode (API only)
COGNITHOR_HOME=~/my-cognithor cognithor  # Custom home directory
```

> **Windows:** If `cognithor` is not recognized after `pip install`, use `python -m cognithor` instead — this always works regardless of PATH configuration. Alternatively, add Python's `Scripts` directory to your PATH (typically `%APPDATA%\Python\PythonXY\Scripts` or the `Scripts` folder inside your venv).

**Option C: Flutter Command Center (Development)**

```bash
cd flutter_app
flutter pub get
flutter run       # Desktop, or:
flutter run -d chrome  # Web
```

The Flutter Command Center connects to the Python backend on port 8741. Start the backend first (`python -m cognithor --no-cli`), then launch the Flutter app. The **Chat page** opens as the default start page — start talking to Cognithor immediately, or activate **Voice Mode** for hands-free conversation. The Sci-Fi aesthetic features dark translucent panels, neon accents, and GEPA pipeline visualization.

All configuration — agents, prompts, cron jobs, MCP servers, A2A settings — can be edited and saved through the dashboard. Changes persist to YAML files under `~/.cognithor/`.

> **Legacy React UI (deprecated):** The old React + Vite UI in `ui/` is deprecated and will be removed in a future release. Use the Flutter Command Center instead.

> **Windows users:** A desktop shortcut named **Cognithor** is included for convenience.

### Channel Auto-Detection

Channels start automatically when their tokens are found in `~/.cognithor/.env`:

```bash
# ~/.cognithor/.env — just add your tokens, channels activate automatically
COGNITHOR_TELEGRAM_TOKEN=your-bot-token
COGNITHOR_TELEGRAM_ALLOWED_USERS=123456789
COGNITHOR_DISCORD_TOKEN=your-discord-token
COGNITHOR_SLACK_TOKEN=xoxb-your-slack-token
```

No need to set `telegram_enabled: true` in the config — the presence of the token is sufficient.

### Directory Structure (Auto-Created on First Start)

```
~/.cognithor/
├── config.yaml          # User configuration
├── CORE.md              # Identity and rules
├── memory/
│   ├── episodes/        # Daily log files
│   ├── knowledge/       # Knowledge graph files
│   ├── procedures/      # Learned skills
│   └── sessions/        # Session snapshots
├── vault/               # Knowledge Vault (Obsidian-compatible)
│   ├── recherchen/      # Web research results
│   ├── meetings/        # Meeting notes
│   ├── wissen/          # Knowledge articles
│   ├── projekte/        # Project notes
│   ├── daily/           # Daily notes
│   └── _index.json      # Quick lookup index
├── index/
│   └── cognithor.db     # SQLite index (FTS5 + vectors + entities)
├── hierarchical/
│   └── trees.db         # Hierarchical document trees (SQLite)
├── cag/
│   └── cache.db         # CAG prefix cache (SQLite)
├── mcp/
│   └── config.yaml      # MCP server configuration
├── queue/
│   └── messages.db      # Durable message queue (SQLite)
└── logs/
    └── cognithor.log    # Structured logs (JSON)
```

## CLI Commands

Once Cognithor is running, you can use these slash commands in the chat:

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/status` | Show system status |
| `/version` | Show version |
| `/config` | Open interactive config editor |
| `/ui` | Open web UI in browser |
| `/clear` | Clear screen |
| `/quit` | Exit Cognithor |

### Startup Flags

| Flag | Description |
|------|-------------|
| `cognithor` | Start with CLI chat |
| `cognithor --ui` | Start headless + auto-open browser |
| `cognithor --no-cli` | Headless backend only |
| `cognithor config` | Open config editor (one-shot) |
| `cognithor config set KEY VAL` | Set a config value |
| `cognithor --lite` | Lite mode (6GB VRAM) |
| `cognithor --log-level DEBUG` | Verbose logging |

## Configuration

Cognithor is configured via `~/.cognithor/config.yaml`. All values can be overridden with environment variables using the `COGNITHOR_` prefix (`JARVIS_*` still accepted as legacy alias).

```yaml
# Example: ~/.cognithor/config.yaml
owner_name: "Alex"
language: "de"  # "de" (German, default) or "en" (English)

# LLM Backend — set a key, backend is auto-detected
# openai_api_key: "sk-..."
# anthropic_api_key: "sk-ant-..."
# gemini_api_key: "AIza..."
# groq_api_key: "gsk_..."
# xai_api_key: "xai-..."
# Or: llm_backend_type: "lmstudio"  # Local, no key needed

ollama:
  base_url: "http://localhost:11434"
  timeout_seconds: 120

web:
  # Search providers (all optional, fallback chain: SearXNG -> Brave -> Google CSE -> DDG)
  # searxng_url: "http://localhost:8888"
  # brave_api_key: "BSA..."
  # google_cse_api_key: "AIza..."
  # google_cse_cx: "a1b2c3..."
  # jina_api_key: ""              # Optional, free tier works without key
  # domain_blocklist: []          # Blocked domains
  # domain_allowlist: []          # If set, ONLY these domains allowed

vault:
  enabled: true
  path: "~/.cognithor/vault"
  # auto_save_research: false     # Auto-save web research results

channels:
  cli_enabled: true
  # Channels auto-detect from tokens in ~/.cognithor/.env
  # Set to false only to explicitly disable a channel:
  # telegram_enabled: false

security:
  allowed_paths:
    - "~/.cognithor"
    - "~/Documents"

# Personality
personality:
  warmth: 0.7                    # 0.0 = sachlich, 1.0 = sehr warm
  humor: 0.3                     # 0.0 = kein Humor, 1.0 = viel Humor
  greeting_enabled: true          # Tageszeit-Grüße
  follow_up_questions: true       # Rückfragen anbieten
  success_celebration: true       # Erfolge feiern

# Scaling
distributed_lock:
  backend: "file"                 # "redis" or "file"
  # redis_url: "redis://localhost:6379/0"

message_queue:
  enabled: true
  # max_retries: 3
  # dlq_enabled: true

telemetry:
  prometheus_enabled: true
  # metrics_port: 9090
```

## Security

Cognithor implements multi-layered security (not independently audited):

| Feature | Description |
|---------|-------------|
| **Gatekeeper** | Deterministic policy engine (no LLM). 4 risk levels: GREEN (auto) -> YELLOW (inform) -> ORANGE (approve) -> RED (block) |
| **Sandbox** | 4 isolation levels: Process -> Linux Namespaces (nsjail) -> Docker -> Windows Job Objects |
| **Audit Trail** | Append-only JSONL with SHA-256 chain. Tamper-evident. Credentials masked before logging |
| **Credential Vault** | Fernet-encrypted, per-agent secret storage |
| **Runtime Token Encryption** | All channel tokens (Telegram, Discord, Slack, ...) encrypted in memory with ephemeral Fernet keys (AES-256). Never stored as plaintext in RAM |
| **TLS Support** | Optional SSL/TLS for all webhook servers (Teams, WhatsApp, API, WebUI). Minimum TLS 1.2 enforced. Warning logged for non-localhost without TLS |
| **File-Size Limits** | Upload/processing limits on all paths: 50 MB documents, 100 MB audio, 1 MB code execution, 50 MB WebUI uploads |
| **Session Persistence** | Channel-to-session mappings stored in SQLite (WAL mode). Survives restarts — no lost Telegram/Discord sessions |
| **Input Sanitization** | Injection protection for shell commands and file paths |
| **Sub-Agent Depth Guard** | Configurable `max_sub_agent_depth` (default 3) prevents infinite handle_message() recursion |
| **SSRF Protection** | Private IP blocking for http_request and web_fetch tools |
| **EU AI Act** | Compliance module, impact assessments, transparency reports |
| **Red-Teaming** | Automated offensive security tests (1,425 LOC) |

### GDPR Compliance (Privacy by Design) — 100% User Rights

Cognithor implements GDPR compliance at the architecture level with full coverage of all user rights:

- **Consent Management** — Per-channel consent with versioning. No data processing without explicit consent.
- **ComplianceEngine** — Runtime enforcement gate. Fail-closed design: blocks processing if consent store unavailable.
- **Art. 15 (Access)** — Complete export across 11 data tiers (sessions, vault, entities, relations, episodes, procedures, core memory, preferences, processing logs, model usage, consents). JSON + CSV formats.
- **Art. 16 (Rectification)** — `PATCH /api/v1/user/data` for entities, preferences, vault notes.
- **Art. 17 (Erasure)** — `DELETE /api/v1/user/data` with 7 erasure handlers covering all data tiers including vault notes.
- **Art. 18/21 (Restriction)** — Per-purpose restriction (evolution, cloud_llm, memory, osint) via REST API + ComplianceEngine enforcement.
- **Art. 20 (Portability)** — Export format v2.0 `cognithor_portable` + `POST /api/v1/user/data/import`.
- **Encryption at Rest** — SQLCipher (AES-256) for all 33 SQLite databases, Fernet for memory files, OS Keyring key management.
- **Audit Trail** — Append-only, SHA-256-chained compliance log with tamper detection.
- **TTL Enforcement** — Automated daily retention enforcement via cron.
- **Privacy Mode** — Runtime toggle disabling all persistent storage.
- **Processing Register** — Art. 30 compliant register of all 13 processing activities.

## MCP Tools

| Tool Server | Tools | Description |
|-------------|-------|-------------|
| **Filesystem** | read, write, edit, list, delete | Path-sandboxed file operations |
| **Shell** | exec_command | Sandboxed command execution with timeout |
| **Memory** | search, save, get_entity, add_entity, ... | 10 memory tools across all 6 tiers |
| **Web** | web_search, web_fetch, search_and_read, web_news_search, http_request | 4-provider search (SearXNG -> Brave -> Google CSE -> DDG), Jina Reader fallback, domain filtering, cross-check, full HTTP method support (POST/PUT/PATCH/DELETE) |
| **Browser** | navigate, screenshot, click, fill_form, execute_js, get_page_content | Playwright-based browser automation |
| **Media** | transcribe_audio, analyze_image, extract_text, analyze_document, convert_audio, resize_image, tts, document_export | Multimodal pipeline + LLM-powered document analysis (all local) |
| **Vault** | vault_save, vault_search, vault_list, vault_read, vault_update, vault_link | Obsidian-compatible Knowledge Vault with frontmatter, tags, backlinks |
| **Synthesis** | knowledge_synthesize, knowledge_contradictions, knowledge_timeline, knowledge_gaps | Meta-analysis across Memory + Vault + Web with LLM fusion, confidence scoring, fact-checking |

## Tests

```bash
# All tests
make test

# With coverage report
make test-cov

# Specific areas
python -m pytest tests/test_core/ -v
python -m pytest tests/test_memory/ -v
python -m pytest tests/test_channels/ -v
```

Current status: **13,000+ tests** · **100% pass rate** · **89% coverage** · **~205,000 LOC source** · **~168,000 LOC tests**

Notable test suites: 183 Computer Use tests, 176 ARC tests, 136 Hierarchical Retrieval tests, 71 CAG tests.

| Area | Tests | Description |
|------|-------|-------------|
| Core | 1,893 | Planner, Gatekeeper, Executor, Config, Models, Reflector, Distributed Lock, Model Router, DAG Engine, Delegation, Collaboration, Agent SDK, Workers, Personality, Sentiment |
| Integration | 1,314 | End-to-end tests, phase wiring, entrypoint, A2A protocol |
| Channels | 1,360 | CLI, Telegram (incl. Webhook), Discord, Slack, WhatsApp, API, WebUI, Voice, iMessage, Signal, Teams |
| MCP | 825 | Client, filesystem, shell, memory server, web, media, synthesis, vault, browser, bridge, resources |
| Memory | 658 | All 6 tiers, indexer, hybrid search, chunker, watcher, token estimation, integrity, hygiene |
| Skills | 534 | Skill registry, generator, marketplace, persistence, API, CLI tools, scaffolder, linter, BaseSkill, remote registry |
| Security | 469 | Audit, credentials, token store, TLS, policies, sandbox, sanitizer, agent vault, resource limits, GDPR |
| Gateway | 252 | Session management, agent loop, context pipeline, phase init, approval flow |
| A2A | 158 | Agent-to-Agent protocol, client, HTTP handler, streaming |
| Telemetry | 175 | Cost tracking, metrics, tracing, Prometheus export, instrumentation, recorder, replay |
| Other | 247 | HITL, governance, learning, proactive, config manager |
| Tools | 103 | Refactoring agent, code analyzer, skill CLI developer tools |
| Utils | 126 | Logging, helper functions, error messages, installer |
| Cron | 63 | Engine, job store, scheduling |
| UI API | 55 | Command Center endpoints (config, agents, prompts, cron, MCP, A2A) |

## Deployment

> **Caution:** Cognithor is Beta software. Test thoroughly in your environment before relying on it for important workflows. Back up your data regularly. See [Status & Maturity](#status--maturity).

### One-Click (Windows)

Double-click `start_cognithor.bat` -> browser opens -> click **Power On** -> done.

### Docker (Development)

```bash
docker compose up -d                         # Core backend
docker compose --profile webui up -d         # + Web UI
```

### Docker (Production)

```bash
cp .env.example .env   # Edit: set COGNITHOR_API_TOKEN, etc.
docker compose -f docker-compose.prod.yml up -d

# With optional services
docker compose -f docker-compose.prod.yml --profile postgres up -d   # + PostgreSQL
docker compose -f docker-compose.prod.yml --profile nginx up -d      # + Nginx TLS
docker compose -f docker-compose.prod.yml --profile monitoring up -d # + Prometheus + Grafana
```

Services: Cognithor (headless) + WebUI + Ollama + optional PostgreSQL (pgvector) + optional Nginx reverse proxy + optional Prometheus/Grafana monitoring. GPU support via nvidia-container-toolkit (uncomment in compose file).

### Bare-Metal Server (Ubuntu/Debian)

```bash
sudo bash deploy/install-server.sh --domain cognithor.example.com --email admin@example.com
# Or with self-signed cert:
sudo bash deploy/install-server.sh --domain test.local --self-signed
```

Installs to `/opt/cognithor/`, data in `/var/lib/cognithor/`, systemd services `cognithor` + `cognithor-webui`.

### Systemd (User-Level)

```bash
./install.sh --systemd
systemctl --user enable --now cognithor
journalctl --user -u cognithor -f    # Logs
```

### Health Checks

```bash
curl http://localhost:8741/api/v1/health     # Command Center API
curl http://localhost:8080/api/v1/health     # WebUI (standalone)
curl http://localhost:9090/metrics           # Prometheus metrics
```

### Backup

```bash
./scripts/backup.sh                    # Create backup
./scripts/backup.sh --list             # List backups
./scripts/backup.sh --restore latest   # Restore
```

See [`deploy/README.md`](deploy/README.md) for full deployment documentation (Docker profiles, TLS, Nginx/Caddy, bare-metal install, monitoring, troubleshooting).

## Language & Internationalization

Cognithor ships with a **JSON-based i18n language pack system** (since v0.33.0). The default language is English, switchable via the Flutter Command Center or `config.yaml`.

### How It Works

```python
from cognithor.i18n import t, set_locale

set_locale("en")  # or "de"
print(t("error.timeout"))  # "The operation timed out..."
```

- **Language packs**: JSON files in `src/cognithor/i18n/locales/` (e.g., `en.json`, `de.json`)
- **Dot-notation keys**: `{"error": {"timeout": "..."}}` → `t("error.timeout")`
- **Fallback chain**: Current locale → English → raw key
- **SHA-256 integrity**: Optional `.sha256` sidecar files for community pack verification
- **Thread-safe**: Locale switching via `set_locale()` is thread-safe

### Switching Language

1. **Flutter Command Center**: General → "Sprache / Language" dropdown, or click the language button in the header
2. **Config file**: Set `language: en` in `~/.cognithor/config.yaml`
3. **Environment variable**: `COGNITHOR_LANGUAGE=en`

| Area | i18n Status | Notes |
|------|-------------|-------|
| **Error messages** | Fully i18n | `utils/error_messages.py` uses `t()` |
| **Tool names** | Fully i18n | All 20 MCP tools have translated names |
| **Personality / Greetings** | Fully i18n | Greetings, empathy, success messages |
| **System prompts** (Planner) | i18n keys | Prompt templates in language packs |
| **Flutter Command Center** | Partial | Config labels still hardcoded (planned) |
| **Log messages** | English only | structlog keys are not translated |

### Contributing Translations

1. Copy `src/cognithor/i18n/locales/en.json` to `<locale>.json` (e.g., `zh.json`, `fr.json`)
2. Translate all ~250 string values
3. Run `python -c "from cognithor.i18n import generate_pack_hash; generate_pack_hash('<locale>')"`
4. Submit a PR

**Metrics:** ~205,000 LOC source · ~168,000 LOC tests · 13,000+ tests · 89% coverage · 0 lint errors · **Status: Beta**

## Contributors

| Contributor | Role | Focus |
|-------------|------|-------|
| [@Alex8791-cyber](https://github.com/Alex8791-cyber) | Creator & Maintainer | Architecture, Core Development |
| [@TomiWebPro](https://github.com/TomiWebPro) | Core Contributor & QA Lead | Real world Deployment & Marketing & Testing |

### Special Thanks

[@TomiWebPro](https://github.com/TomiWebPro) — Now a core member of the development team and Head of Marketing. Helped with early testing and debugging, contributed security suggestions, and supports community Q&A.

[u/_D4rk4_](https://www.reddit.com/user/_D4rk4_/) — Provided valuable feedback about non-solidified features between releases. Highlighted the need for more stable, well tested releases. 

## License

Apache 2.0 — see [LICENSE](LICENSE)

Copyright 2026 Alexander Soellner

## Star History

<a href="https://www.star-history.com/?repos=Alex8791-cyber%2Fcognithor&type=date&legend=bottom-right">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=Alex8791-cyber/cognithor&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=Alex8791-cyber/cognithor&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=Alex8791-cyber/cognithor&type=date&legend=top-left" />
 </picture>
</a>

---

## What's New

### v0.95.0 — Trace-UI (2026-04-27)

- **Live-Visibility for running Crews** — new Flutter Trace-UI screen with master-detail
  list + timeline-log + per-agent stats sidebar. Owner-gated.
- **`cognithor.crew.trace_bus.TraceBus`** — in-process pub/sub hooked into
  `compiler.append_audit()`; broadcasts crew lifecycle + per-trace events to
  WebSocket subscribers without changing the Hashline-Guard JSONL persistence.
- **REST endpoints `/api/crew/traces`, `/trace/{id}`, `/trace/{id}/stats`** — read
  the existing audit chain for replay + history. Corruption-tolerant; surfaces
  skipped-line counts via `meta`.
- **WebSocket message types `crew_lifecycle_subscribe` / `crew_subscribe` /
  `crew_unsubscribe`** for live event streaming. Disconnect auto-cleans.
- **AutoGen-Shim coverage verified** — `cognithor.compat.autogen.AssistantAgent.run()`
  routes through `cognithor.crew.Crew` so AutoGen-shim runs surface in Trace-UI.
- **Owner-Token gating** — `cognithor.security.owner.require_owner` reads
  `COGNITHOR_OWNER_USER_ID` env (fallback: `pyproject.toml` author name).

### v0.94.0 — AutoGen Strategy Adoption (2026-04-25)

- **`cognithor.compat.autogen`** — source-compat shim for `autogen-agentchat==0.7.5`.
  Search-and-replace migration path for AutoGen-AgentChat code onto Cognithor's
  PGE-Trinity. See [migration guide](src/cognithor/compat/autogen/README.md).
- **`cognithor_bench/`** — reproducible Multi-Agent benchmark scaffold with
  `cognithor-bench run|tabulate` console tool and Cognithor / AutoGen adapters.
- **`examples/insurance-agent-pack/`** — standalone-installable DACH insurance
  pre-advisory reference pack with visible ComplianceGatekeeper PGE-demo.
- **Architecture Decision Records** — first ADR
  ([0001 — PGE Trinity vs Group Chat](docs/adr/0001-pge-trinity-vs-group-chat.md))
  documents why Cognithor doesn't adopt SelectorGroupChat / Swarm patterns.
- **Competitive analysis docs** — comparison with AutoGen, MAF, LangGraph,
  CrewAI under [`docs/competitive-analysis/`](docs/competitive-analysis/README.md).

### v0.92.7 (2026-04-23)
- **Video input via vLLM** — end-to-end video analysis in chat using Qwen3.6-27B or any vLLM-served VLM. Paperclip → "Video hochladen" or paste a direct `.mp4`/`.webm`/`.mov`/`.mkv`/`.avi` URL. Adaptive frame sampling via `ffprobe` (short clips: `fps=3`, long clips: `num_frames=32`), 500 MB per-file cap + 5 GB quota with LRU eviction, session-lifetime + 24 h TTL cleanup, no silent fallback to Ollama (Ollama has no vision). Windows installer now bundles LGPL ffmpeg + CI verifies the GPL-free build. See [`docs/vllm-user-guide.md`](docs/vllm-user-guide.md).
- **Default vLLM image flipped to `cu130-nightly`** — the tagged `v0.19.1` crashes Qwen3.6-27B-NVFP4 at warmup on SM120 (Blackwell / RTX 50xx). The nightly ships the `FlashInferCutlassNvFp4LinearKernel` fix. Day-1 spike findings at [`docs/superpowers/spikes/2026-04-23-video-input-vllm-spike-findings.md`](docs/superpowers/spikes/2026-04-23-video-input-vllm-spike-findings.md).
- **Orchestrator unified** — `backends_api` and `Gateway` now share a single `VLLMOrchestrator` instance via `app.state`, so `media_url` wired after the media server starts actually reaches the `docker run` command.
- **14,118 Python tests + 31 Flutter tests passing** (+180 new for this release), 0 failures, 0 ruff errors.

### v0.92.2 (2026-04-19)
- **Windows Launcher Hardening** — `Cognithor.exe` (the .NET tray app) no longer vanishes silently on startup. All exceptions in the AppShell constructor are now caught, logged to `%LOCALAPPDATA%\Cognithor\launcher-crash.log`, and shown via MessageBox. Previously a missing `python\python.exe` (e.g. AV quarantine) would throw `FileNotFoundException` before `Application.Run()` started the message loop, making the process disappear from Task Manager with no trace.
- **Python Resolution 3-Tier Fallback** — Both `Cognithor.exe` and the installer-generated `cognithor.bat` now resolve Python in this order: (1) bundled `<install>\python\python.exe`, (2) `python`/`py` on PATH, (3) well-known install paths (`%LOCALAPPDATA%\Programs\Python\Python313|312`, `%ProgramFiles%\Python313|312`, `C:\Python313|312`). Users who installed Python without ticking "Add to PATH" (the default since Python 3.9) are no longer broken.
- **Single-Instance Guard Actually Works** — The process-wide `Mutex` in `Program.cs` is no longer scoped with `using`, which previously disposed it at first return and defeated the guard. Now released explicitly in `finally`.
- **Graceful Python-Missing UX** — `ProcessManager.StartPython` no longer throws when Python is missing. Instead it sets `PythonMissing = true`, and the tray icon shows "Backend not installed" with a balloon tip explaining what's wrong.
- **Square Logos** — `flutter_app/assets/logo.png` and `assets/cognithor_logo.png` were 1536×1024 non-square and got cropped in every UI that used `BoxFit.cover`. Regenerated as 1024×1024 square from `Icon-512.png`. `cognithor_banner.png` rebuilt as 1920×640 dark-theme banner with the logo centered. `cognithor-logo.png` bumped from 52×52 to 192×192 (PWA standard).
- **Flutter web `index.html` branding** — replaced leftover `jarvis_ui` title + apple-touch-title and the default "A new Flutter project." description with real Cognithor branding.
- **Release-workflow idempotency** — explicit release-notes generation replaces `generate_release_notes: true`. The old setup prepended a fresh `**Full Changelog**: ...` line on every workflow re-run; `append_body: false` now guarantees idempotent re-runs.
- **Installer Build Fixes** — `subprocess.run(["flutter",...])` on Windows now resolves via `shutil.which` (flutter is a `.bat`); failed `flutter build windows` (e.g. Developer Mode off) is caught as soft-fail so the rest of the installer still builds; `step_inno_setup` reports the correct version instead of alphabetically-first.
- **Legacy `jarvis_ui.exe` fallback removed** from `ProcessManager.FindDesktopUi` — the only accepted name is `cognithor_ui.exe` (v0.42.0 rename).

### v0.92.1 (2026-04-18)
- **E2E Audit Fixes** — `zh.json` JSON syntax error (unescaped ASCII quotes inside Chinese text, line 458-460) — Chinese locale silently fell back to English until now. Fixed with `「」` corner brackets.
- **Test isolation** — 3× fixture bug in `test_enhanced_integration.py` passed `jarvis_dir=` (unknown field, silently ignored by Pydantic), causing `MemoryManager` to hit the real `~/.cognithor/db/` and fail with "database disk image is malformed". Now uses `jarvis_home=` + all 52 Config classes have `model_config = ConfigDict(extra="forbid")` so unknown kwargs fail loudly.
- **Tool-Risk Decorator** — `@cognithor_tool(risk_level="green"|"yellow"|"orange"|"red")` + `PackManifest.tool_risks` field. PackLoader pushes declared risks into `mcp_client._tool_registry` on load. Previously 15/18 pack tools defaulted to ORANGE (user approval required for read-only scans); now every pack tool gets its declared level.
- **CI Hardening** — new `scripts` job runs `verify_all.py` and `smoke_test.py` so these stop rotting outside CI. New `flutter` job runs `dart analyze` and `flutter test`. Lint job gains full `ruff check` + `ruff format --check`.
- **UTF-8 on Windows** — `PYTHONIOENCODING=utf-8` + `PYTHONUTF8=1` added to `start_cognithor.bat` and `install.bat`. `_DEFAULT_CORE_MEMORY` template now ships real umlauts ("Identität") instead of ASCII fallback ("Identitaet").
- **Scripts repaired** — `smoke_test.py` cp1252-crash fixed (box-drawing + checkmark chars → ASCII `[OK]`/`[WARN]`/`[FAIL]`); `live_smoke_test.py` stdout UTF-8 + legacy `jarvis.config` import + `list_models()` API signature; `verify_all.py` completely rewritten for v0.92.x architecture (was checking deleted `cognithor.social` module and stale `src/jarvis/` paths).
- **Flutter ARB parity** — 11 missing keys added to `app_zh.arb` + `app_ar.arb` (connectionLost, actionApproved, errorWithDetail, …).
- **Lint + format green** — 90 ruff errors resolved (74 auto-fix + 16 manual), 17 dart-analyze issues resolved; `ruff format --check` now clean.
- **13,791 tests passing** (+19 new tool-risk tests), 0 failures.

### v0.92.0 (2026-04-16)
- **Agent Pack Architecture** — Full plugin system for paid & free agent packs. `cognithor.packs` module: `PackManifest` (Pydantic v2), `PackLoader` (importlib isolation), `PackInstaller` (zip/URL + EULA click-through), CLI (`cognithor pack install|list|remove|update|accept-eula`). Packs live in `~/.cognithor/packs/` and register at startup.
- **Leads SDK** — `cognithor.leads` replaces the old `cognithor.social` module. Source-agnostic `LeadSource` ABC, `SourceRegistry`, `LeadService` orchestrator, `LeadStore` with `source_id` column. Any pack can register a lead source — Reddit, HN, Discord, RSS, or future custom sources.
- **Reddit Lead Hunter Pro** — First paid Agent Pack (€75 indie / €179 commercial via Lemon Squeezy). Extracted from Core into private `cognithor-packs` repo. Includes: Reddit scanner, 50 reply templates, Playwright auto-poster, style learner, subreddit discovery. Core retains zero Reddit-specific code.
- **3 Free Bundled Packs** — Hacker News Lead Hunter, Discord Lead Hunter, RSS Lead Hunter. Apache 2.0, auto-installed on first Cognithor start. Same pack interface as paid packs.
- **cognithor.ai** — Marketing site deployed on Vercel. Build-time pack catalog fetch from private GitHub repo via Octokit. Auto-rebuild on pack repo push via deploy hook. Pack detail pages with strikethrough pricing, checkout flow, post-purchase install command page.
- **Flutter Pack-Aware UI** — `PackPreviewOverlay` shows real pack UI greyed out with unlock CTA. `LockedPackCard` for free packs. `SourcesProvider` polls `/api/v1/leads/sources`. Leads tab gated on registered sources. Settings → Social shows Reddit config disabled with "Get this Pack" overlay when pack not installed.
- **Anti-Enshittification Promise** — `/promise` page: Core stays free forever, packs add but never subtract, zero telemetry. "This isn't marketing copy. It's a contract. If we ever break it, fork the repo."
- **EULA System** — Every pack ships `eula.md` with SHA-256 pinned in manifest. Installer prompts click-through, persists acceptance. Loader rejects packs with hash mismatch or missing acceptance.
- **`src/cognithor/social/` DELETED** — 15 Python files (~4000 LOC) extracted to agent packs. Gateway rewritten for source-agnostic `LeadService`. No backward compatibility shim remains.
- **13,600+ tests passing** across Core + packs + site (657 vitest)

### v0.91.0 (2026-04-14)
- **Unified Leads Engine** — Generalized the Reddit-only lead system into a multi-source engine (#113). RSS/Atom feed scanner (stdlib `xml.etree`, no new deps), configurable per-source in Flutter Settings → Social. Sidebar Leads tab gated on `leads_engine_enabled` — hidden when all sources disabled.
- **Installer Language Fix** — Inno Setup language choice now persists to Cognithor config (#114). `install_language.txt` marker from installer → `bootstrap_windows.py` reads before OS-locale fallback. `first_run.py` skips the language question when marker present.
- **OLLAMA_HOST Normalization** — `OLLAMA_HOST=0.0.0.0` no longer crashes httpx (#115). `_normalize_ollama_url()` helper accepts bare hosts, host:port, full URLs. Wired as `default_factory` + `before` validator on `OllamaConfig.base_url`.
- **13,500+ tests passing**

### v0.90.0 (2026-04-11)
- **Cross-Platform Social Listening** — Hacker News (Firebase + Algolia API, zero auth) + Discord (REST API v10, bot token) scanners join Reddit. Unified `social_scan` and `social_leads` MCP tools. Per-platform config: intervals, min scores, categories/channels. Flutter config UI with HN + Discord sections.
- **Hierarchical Document Reasoning** — 4th retrieval channel: tree-based section navigation for PDF, DOCX, HTML, Markdown, plaintext. `TreeBuilder` + `NodeSelector` (LLM-navigated) + `TreeStore` (SQLite). No embeddings required — structural understanding through heading hierarchy.
- **CAG Layer (Cache-Augmented Generation)** — Deterministic prefix generation for LLM KV-cache reuse. `PrefixBuilder` (Ollama-compatible) + `NativeLlamaCppBuilder`, `ContentNormalizer`, `CacheStore`, hit-rate metrics. Hooks into Planner for automatic prefix injection.
- **CLI Config TUI** — Interactive terminal config editor via `cognithor --config-tui`. Dynamic model discovery from live LLM providers, section navigation, validation, rich formatting.
- **AST-Based Security** — Python `ast.NodeVisitor` guard (40 tests) + `bashlex` shell parser (48 tests) replace regex-based code analysis. Detects dangerous imports, subprocess calls, eval, network access at syntax tree level.
- **`_safe_call()` Pattern** — Unified error handling replacing 79 silent `except Exception: pass` across codebase. Per-function failure registry, async variant, structured logging.
- **Package Rename** — `jarvis` → `cognithor` across 1,265 files (5,770 replacements). Both `cognithor` and `jarvis` entry points preserved for compatibility.
- **Installer Overhaul** — `install.bat` + `install.sh` rewritten with GPU detection, model auto-pull, health checks, auto-upgrade system syncing source tree to installed version on every launch.
- **13,000+ tests passing** across Python 3.12/3.13 x Ubuntu/Windows

### v0.84.0 (2026-04-09)
- **Reddit Lead Hunter** — Full social listening system: scans Reddit for high-intent posts, scores them 0-100 via LLM, drafts context-aware replies. 3 trigger paths (Chat, Cron, REST API). No Reddit API key needed.
- **Reddit Leads Flutter Tab** — 7th navigation tab with filterable lead pipeline, score badges, reply editor, Scan Now FAB. LeadDetailSheet with edit/post/copy/archive actions.
- **Robot Office Live Wiring** — Dynamic robots from real agents, real-time PGE state sync via WebSocket, system metrics driving server rack LEDs, kanban board with real task dots + hover tooltips.
- **Deep Learning Upload Pipeline** — Hybrid: immediate chunk-indexing + background KnowledgeBuilder pipeline (Vault, entities, identity memory). Priority queue, PDF vision, OCR fallback, YouTube frames.
- **20+ Bug Fixes** — Issues #62-#89 resolved: duplicate UI pages removed, i18n gaps filled, audit verify crash, incognito flag, translate prompts, QR pairing, operation mode description, logo fallback, and more.
- **Flutter Auto-Rebuild** — Bootstrap detects outdated builds, auto-rebuilds or downloads pre-built UI from release.
- **13,000+ tests passing** across Python 3.12/3.13 x Ubuntu/Windows

### v0.80.1 (2026-04-08)
- **Full Lint Cleanup** — 299 Ruff lint errors resolved (E501, F841, B904, SIM102, RUF006, etc.), zero errors across all rules
- **Kanban Board** — Interactive task management with 6 columns, drag-and-drop, sub-tasks, auto-tasks from Cron/Evolution/Executor
- **Evolution Engine Phase 5** — Autonomous exam-based deep learning with CycleController, stagnation detection, mastery progression
- **Community Skill Marketplace** — GitHub registry, 5-check validation, publisher verification, ToolEnforcer
- **Ralph Agent-Loop** — CONTINUE/STOP multi-step autonomous task execution with budget limits
- **HybridClaw Phase 3** — Tool-Loop-Detection, Context Guard, Security Hooks, Config-Versioning, LLM Retry, Concierge Routing
- **Mobile Pairing & Networking** — HMAC device tokens, QR pairing, multi-interface VPN detection
- **i18n Phase 2** — 648 locale keys across EN/DE/ZH, gateway/planner/gatekeeper migrated
- **Computer Use Phase 2** — Vision-guided clicking and screen change detection
- **Windows Installer** — Inno Setup with embedded Python, Ollama, Flutter UI, health-check polling
- **Android APK + iOS IPA** — Automated mobile builds via GitHub Actions
- **13,000+ tests passing** across Python 3.12/3.13 × Ubuntu/Windows

### What's New in v0.74.0

**ARC-AGI-3: 13/25 Games Solved (24 Levels) — SmartExplorer Breakthrough**

Inspired by the [3rd-place ARC-AGI-3 Preview solution](https://arxiv.org/abs/2512.24156), the new SmartExplorer uses systematic state-action graph exploration to solve 7 previously unsolvable games in a single session.

- **SmartExplorer** (`smart_explorer.py`): Tracks tested/untested actions per state, navigates to nearest frontier via BFS on known transitions, prunes no-effect actions, detects click targets via connected components. No ML — pure graph search with smart prioritization
- **7 new games solved**: TR87 (block-matching, 90 steps), BP35 (platformer, 32 steps), CD82 (5 steps), TU93 (18 steps), KA59 (184s), SU15 (122s), TN36 (131s)
- **Key insight**: systematic action testing at every state + frontier navigation >> blind DFS. The SmartExplorer tests every action at every reachable state and navigates back to states with untested actions via shortest known path
- **VisionAgent** (`vision_agent.py`): Prototype for qwen3-vl guided step-by-step gameplay
- **Action 7 fix**: Was completely dropped from keyboard solver filter — now included
- **Clicks as DFS actions**: Click positions are regular DFS branches, enabling multi-click sequences
- **Incremental click-DFS**: For deep click puzzles (LP85 L2), steps forward with env.step instead of replaying
- **Click path shortening**: Removes redundant clicks from solutions
- **Full benchmark**: All 25 games tested: FT09(10), VC33(2), LP85(2), SP80(1), CN04(1), M0R0(1), TR87(1), BP35(1), CD82(1), TU93(1), KA59(1), SU15(1), TN36(1)

### What's New in v0.73.0

**ARC-AGI-3: 7/25 Games Solved — ClickSequenceSolver + KeyboardSolver** — Three solver families covering click, keyboard, and mixed game types.

- **ClickSequenceSolver**: BFS + Simulation A* for water-routing puzzles (VC33: 3/7 levels)
- **KeyboardSolver**: Incremental DFS ~50x faster than replay-based BFS
- **Path shortening**, false positive detection, pump-then-trigger architecture

### What's New in v0.72.0

**ARC-AGI-3: GameAnalyzer + Smart Solver** — Fully automated game analysis and solving pipeline for click-based ARC-AGI-3 games.

- **GameAnalyzer** (`game_analyzer.py`): Sacrifices one level to learn game mechanics, 2 vision calls (qwen3-vl:32b) for strategy guidance, persistent GameProfile cache
- **PerGameSolver** (`per_game_solver.py`): Budget-based strategy mix, smart elimination search with poison-cluster removal
- **760x faster combo testing**: `env.reset()` (0.5ms) replaces `arcade.make()` (380ms)
- **FT09: 10/10 levels solved** (reproducible, ~1s per level after analysis)
- **Toggle-pair detection**: Automatically identifies clickable colors from sacrifice level data

### What's New in v0.71.0

**Computer Use: Complete Desktop Automation Pipeline**
- Phase 2C: Sub-task decomposition with content bags and file creation
- Phase 2D: 3-layer security (allowlist + gatekeeper), coordinate scaling, adaptive wait, prompt injection hardening
- Phase 2E: Oscillation detection, content limits, dialog handling
- Phase 3: Windows UI Automation via pywinauto — exact element coordinates from OS

**ARC-AGI-3: Complete Redesign**
- Dual-mode agent: RL for interactive games + DSL solver for classic puzzles
- 25 grid transformation primitives with combinatorial search
- Multimodal vision agent using qwen3-vl:32b
- ClusterSolver: first level wins (ft09 Level 1+2 solved)
- Frame analyzer, telemetry tracker, epsilon-greedy explorer

**Skill Lifecycle Fix**
- Context pipeline wired to skill registry — skills now proactively suggested
- Daily lifecycle audit cron

**Bug Fixes**
- SQLCipher _DictRow compatibility (fixes all KeyError: 0 across codebase)
- 23 test failures + 48 errors resolved

### What's New in v0.68.0

**Document Powerhouse** — Cognithor can now create, read, and manage all major document formats:

- **7 document tools**: `read_pdf`, `read_docx`, `read_ppt`, `read_xlsx` (new), `document_export`, `document_create` (new), `typst_render` (new)
- **Template system**: `template_list` + `template_render` — fill Typst templates (Brief, Rechnung, Bericht) and compile to PDF
- **Structured creation**: JSON input with sections, tables, lists → DOCX, PDF, PPTX, XLSX
- **Typst pipeline**: modern LaTeX alternative for high-quality PDFs (<1s compilation)

**Skill Lifecycle** — Skills created by Cognithor are now immediately usable:

- Hot-loading: skills available instantly after creation (no restart)
- Startup scan of generated skills directory
- SkillLifecycleManager: audit, auto-repair, suggest, prune

**Tactical Memory (Tier 6)** — Tool outcome tracking across sessions:

- Learns which tools work best in which context
- Auto-creates avoidance rules after 3 consecutive failures (24h TTL)
- Injects tactical insights into Planner context

### What's New in v0.69.0

**Autonomous Thinking Loop (ATL)** — Cognithor now thinks proactively without user input:

- **GoalManager** — Structured YAML-persisted goals with progress tracking, priority, sub-goals, and success criteria. Migrates existing `learning_goals` automatically.
- **Thinking Cycles** — Every 5 minutes (configurable), the agent evaluates its goals, proposes and executes research actions, and writes a daily Markdown journal.
- **ActionQueue** — Priority-based action dispatch with blocked-type filtering. Actions routed through `search_and_read` with automatic parameter normalization.
- **Risk Ceiling** — Gatekeeper enforces per-context risk limits. ATL is capped at YELLOW (no destructive operations).
- **3 new MCP tools**: `atl_status`, `atl_goals`, `atl_journal`
- **Quiet Hours** — No autonomous thinking between 23:00-07:00 (configurable)

**CAPTCHA Solver** — Browser automation can now detect and solve CAPTCHAs:

- **7 types supported**: Text, reCAPTCHA v2 (checkbox + image grid), reCAPTCHA v3, hCaptcha, Cloudflare Turnstile, FunCaptcha
- **Vision-LLM solving** — Local models only (minicpm-v4.5 for simple, qwen3-vl:32b for complex). No external services.
- **Browser Stealth** — Anti-bot-detection: `navigator.webdriver=false`, realistic user-agent, plugin spoofing
- **Gatekeeper ORANGE** — Requires explicit user approval before solving
- **1 new MCP tool**: `browser_solve_captcha`

**AACS Phase 1 — Capability Tokens** — Cryptographic access control foundation:

- **Ed25519-signed tokens** — Unforgeable, short-lived (10s-1h), attenuation-only (child tokens can never exceed parent rights)
- **Token Issuer + Validator** — Full PGE delegation chain: Planner issues root → Gatekeeper delegates → Executor validates
- **Replay protection** — Nonce cache prevents token reuse
- **Revocation** — Instant token invalidation
- **Feature-flagged** — `log_only` mode for gradual rollout (Phases 2-6 coming)

**Dead Path Fixes** — Critical wiring issues found and fixed via deep audit:

- **SessionAnalyzer**: Improvements now applied automatically (was: proposed but discarded)
- **Hybrid Search**: Context Pipeline now uses BM25 + Vector + Graph search (was: BM25-only, ignoring 7000+ Knowledge Graph entities)
- **EpisodicCompressor**: Daily background task compresses episodes older than 30 days (was: never called)
- **Tactical Memory**: Now injected into Planner system prompt as "Taktische Einsichten" (was: populated but silently dropped)
- **PersonalityEngine.enhance_response()**: Method now exists (was: called but missing, failing silently)

**Production Fixes**:

- **VirtualLock elimination**: `SetProcessWorkingSetSize` expands Windows memory quota; `encrypted_connect()` respects `encryption_enabled` config
- **LLM Timeout**: 600s dynamic (was: 120s with artificial caps on embeddings)
- **PDF garbage entities**: Extended filter blocks XRef, Object/Root/Info IDs, font names, PDF structure markers
- **Corrupt sessions.db**: Graceful fallback when SQLCipher DB was encrypted with a different key

**126 new tests** (45 ATL + 45 CAPTCHA + 36 AACS), **125+ MCP tools** (was 122).

### What's New in v0.67.0

**ARC-AGI-3 Benchmark Integration** — Cognithor can now compete in the [ARC Prize 2026](https://arcprize.org/competitions/2026) ($2M+ prize pool). New `src/cognithor/arc/` module with 14 files implements a hybrid agent (algorithmic exploration + optional LLM planning + optional CNN prediction) for interactive reasoning benchmarks.

- **3 new MCP tools**: `arc_play`, `arc_status`, `arc_replay` — playable from any Cognithor channel
- **CLI**: `python -m cognithor.arc --game ls20 [--mode benchmark|swarm]`
- **105 new tests** covering all ARC subsystems
- **Dependency groups**: `pip install cognithor[arc]` or `cognithor[arc-gpu]`

### v0.77.0 Highlights

#### Interactive Kanban Board
- 6th tab in the Flutter Command Center with drag-and-drop task management
- Tasks from 6 sources: manual, chat, cron, evolution, agents, system
- Sub-tasks with cascade cancel and auto-verification
- 10 REST endpoints, 3 MCP tools, SQLCipher encrypted storage

#### Computer Use Phase 2
- `computer_click_element`: Click UI elements by description (e.g., "Login button")
- `computer_wait_for_change`: Detect screen changes after actions
- 8 tools total for full desktop automation

#### Evolution Engine — Autonomous Learning
- Self-directed learning cycles with automatic quality exams
- Score >= 80%: goal mastered. Stagnating: frequency reduced, Kanban task created
- ATL (Autonomous Thinking Loop) with goal management and file management actions
- REST API for goals, plans, journal, statistics

#### i18n — Full Internationalization
- 314+ user-facing strings migrated from hardcoded German to EN/DE/ZH
- Flutter screens localized via AppLocalizations
- SHA-256 integrity verification for locale packs

### v0.66.0 — Encryption at Rest, Vault Dual-Backend, GDPR 100%

**Encryption at Rest — Full Disk Clone Protection**
- **SQLCipher** — All 33 SQLite databases encrypted with AES-256. Key stored in OS Keyring (never on disk).
- **EncryptedFileIO** — Transparent Fernet encryption for memory files (CORE.md, episodes, procedures).
- **Auto-migration** — Existing unencrypted databases migrated to SQLCipher on first startup.

**Vault Dual-Backend**
- **VaultBackend ABC** — Pluggable storage: FileBackend (.md, Obsidian-compatible) or DBBackend (SQLCipher + FTS5).
- **Bidirectional migration** — Switch between file and DB mode without data loss.

**GDPR User Rights — 100% Coverage**
- Art. 15 (Access), Art. 16 (Rectification), Art. 17 (Erasure), Art. 18/21 (Restriction), Art. 20 (Portability) — all fully implemented across 11 data tiers.

**122 MCP tools**, **11,769+ total tests** (was 11,779+). 12 bug fixes including SQLCipher compatibility, cron consent, and tool timeouts.

### v0.65.0 — GDPR Compliance, OSINT Module, Evolution Engine Stable

**GDPR Compliance Layer**
- **ComplianceEngine** — Runtime enforcement gate with fail-closed design. Blocks processing if consent store unavailable.
- **ConsentManager** — Per-channel consent tracking with versioning. No data processing without explicit consent.
- **Right to Erasure (Art. 17)** — `DELETE /api/v1/user/data` deletes across all data tiers (memory, vault, sessions, episodes).
- **Right of Access (Art. 15)** — `GET /api/v1/user/data` exports all personal data as JSON.
- **ComplianceAuditLog** — Append-only JSONL with SHA-256 chain, tamper detection.
- **TTL Enforcement** — Automated daily retention enforcement via cron.
- **Processing Register** — Art. 30 compliant register of all 13 processing activities.
- **Privacy Mode** — Runtime toggle disabling all persistent storage.
- **Encryption at Rest** — SQLCipher wrapper for all SQLite databases.

**OSINT / HIM Module**
- **HIMAgent** — Structured OSINT investigation for persons, projects, and organizations.
- **3 new MCP tools** — `investigate_person`, `investigate_project`, `investigate_org`.
- **TrustScorer** — 5-dimension weighted trust score (0-100) with evidence cross-verification.
- **GDPRGatekeeper** — Pre-investigation consent and purpose check.
- **EvidenceAggregator** — Multi-source claim classification and contradiction detection.
- **HIMReporter** — Markdown/JSON/Quick report formats with SHA-256 signature.

**Evolution Engine — Stable**
- Promoted from Beta to Stable after extensive testing and quality self-examination integration.
- All 4 phases production-ready: SystemDetector, IdleLoop, Budget/Resources, Checkpoint/Resume.

**122 MCP tools** (was 106), **11,769+ total tests**.

### v0.60.0 — Autonomous Evolution Engine (Premium)

**Per-Agent Budget + Resource Monitor (Phase 3)**
- **ResourceMonitor** — Real-time CPU/RAM/GPU sampling. Cooperative scheduling pauses background tasks when system is busy.
- **Per-Agent Cost Tracking** — Every LLM call tagged by agent. Daily budgets per agent with 80% warning threshold.
- **Flutter Budget Dashboard** — Per-agent cost table (today/week/month), live resource bars, budget status.

**Checkpoint/Resume Engine (Phase 4)**
- **Step-Level Checkpointing** — Evolution cycles save state after each step (Scout→Research→Build→Reflect). Resume interrupted cycles exactly where they stopped.
- **Delta Snapshots** — Only changed data persisted between checkpoints.
- **Flutter Evolution Dashboard** — Visual stepper, one-click resume, recent activity feed.

**Full Evolution Engine** (Phases 1-4 complete)
- Phase 1: Hardware-Aware System Profile (8 detectors, tier classification)
- Phase 2: Idle Learning Loop (autonomous skill building during idle time)
- Phase 3: Per-Agent Budget + Resource Monitor (cooperative scheduling)
- Phase 4: Checkpoint/Resume Engine (resilient cycle execution)
- **REST API** — 7 new endpoints for budget, resources, evolution stats, and resume.
- **63 new tests**, **11,769+ total**.

### v0.54.0 — Computer Use, Deep Research v2, VS Code Extension

**Computer Use (GPT-5.4-style)**
- **6 new MCP tools** — `computer_screenshot`, `computer_click`, `computer_type`, `computer_hotkey`, `computer_scroll`, `computer_drag`
- Takes desktop screenshots, analyzes with vision model, clicks at pixel coordinates
- Auto-installed via `start_cognithor.bat`

**Deep Research v2 (Perplexity-style)**
- **Iterative search engine** — up to 25 rounds with query decomposition, source evaluation, cross-verification, confidence scoring
- Source priority: Official Docs → GitHub → Community → Lateral approaches
- Automatically invoked for complex queries ("recherchiere", "analysiere", "untersuche")

**VS Code Extension**
- **cognithor-vscode/** — Full extension with Chat sidebar, Code Lens, 11 commands
- WebSocket streaming, context-aware code assistance, editor integration
- `POST /api/v1/chat/completions` backend endpoint

**Autonomous Coding**
- 50 iterations for coding tasks, auto-debug, auto-fix
- Ollama/Qwen3:32b as default local planner
- GREEN gatekeeper for core tools (write_file, run_python, exec_command)

### v0.52.0 — Autonomous Agent Framework, SSH Backend, Session Management Overhaul

**Autonomous Task Execution**
- **Autonomous Orchestrator** — Complex tasks are automatically decomposed into subtasks, self-evaluated after execution, and learned from for future tasks. Recurring tasks get automatic cron scheduling.
- **Research Auto-Escalation** — Planner self-assesses source quality. When results are thin or contradictory, automatically escalates to `deep_research` or `search_and_read` for deeper analysis.
- **Marketplace Monitor Skill** — Built-in skill for price tracking, fake detection, and recurring marketplace alerts (Tomi's 5090 example).
- **GEPA Robustness** — Longer evaluation windows (20 traces, 15 sessions), user approval for high-impact proposals, LLM-powered patch generation, cascade failure auto-detection.

**SSH Remote Shell Backend**
- **3 new MCP tools** — `remote_exec`, `remote_list_hosts`, `remote_test_connection` for executing commands on remote servers via SSH.
- **Security** — Dangerous command blocking, ORANGE gatekeeper approval for remote execution.
- **Tested** against Docker containers with key-based SSH auth.

**Session Management**
- **Auto-New-Session** — Fresh session after 30 min inactivity (configurable). No more resuming stale chats.
- **Project Folders** — Group sessions into projects with sidebar grouping.
- **Incognito Mode** — Sessions without memory enrichment or chat persistence.
- **Session Export** — Download any chat as JSON.
- **Full-Text Search** — Search across all chat messages in all sessions.
- **GDPR Retention** — Automatic cleanup of old sessions (30-day retention).
- **Chat History Filter** — System messages and raw tool results no longer shown as chat bubbles.

**Flutter UI**
- **Mobile optimized** — Bottom nav reduced from 8 to 5 items. iPhone Pro Max responsive layout.
- **Light mode fixed** — Theme-aware text colors and code block backgrounds.
- **Incognito badge** — Purple indicator in AppBar + drawer button.
- **Search bar** — Live search in session drawer.
- **Project sidebar** — Sessions grouped by folder with ExpansionTile.
- **Device permissions** — Toggles work on both native and web.

**Infrastructure**
- **Docker Real-Life Test Suite** — 22 scenario tests (pipeline + live Ollama).
- **WebSocket stability** — Fixed reconnection storms, rate-limiting, Windows semaphore errors.
- **CI/CD** — iOS + Android builds green, GitHub Release uploads working.
- **106 MCP tools** (was 91), **11,769+ tests** (was 10,904).

### v0.47.1-beta — Sci-Fi UI, GEPA Pipeline, Robot Office Pathfinding

**Sci-Fi Flutter Command Center**
- **Sci-Fi aesthetic overhaul** — Dark translucent panels, neon accent glows, holographic card effects, particle background animations
- **GEPA pipeline visualization** — Real-time Goal-Evaluate-Plan-Act pipeline status with animated phase indicators and timing metrics
- **Robot Office pathfinding** — Interactive office map with A* pathfinding visualization, room navigation, and agent location tracking

**GEPA (Goal-Evaluate-Plan-Act)**
- **4-phase cognitive pipeline** — Goal extraction, Evaluation (context + memory retrieval), Planning (tool selection + sequencing), Action (sandboxed execution)
- **Pipeline observability** — Each phase emits timing, token count, and status events visible in the Flutter Command Center's Observe panel

**Tool Expansion**
- **MCP tools: 53 → 94** — New tools across filesystem, automation, code analysis, and agent coordination modules

### v0.42.0-beta — Premium UI, Complete Learning System, Issue #35/#36

**World-Class Flutter UI**
- **Responsive 3-tier navigation** — Desktop: animated side rail (220px expand/collapse), Tablet: compact rail with hover-expand, Mobile: bottom bar
- **Glassmorphism cards** — `BackdropFilter` frosted glass with gradient highlight edges
- **Micro-animations everywhere** — `StaggeredList` (cascading entrance, 50ms/item), `AnimatedCounter` (smooth number tweens), `ShimmerLoading` (gradient sweep skeleton), `AnimatedIndexedStack` (fade+slide page transitions)
- **Gradient background** — Subtle rotating accent glow (60s cycle, 3-5% opacity)
- **Theme contrast fix** — 30+ hardcoded dark-mode colors replaced with theme-aware `cardColor`/`dividerColor` — light mode fully usable
- **Centralized design system** — 30+ colors in `jarvis_theme.dart`: entity colors, phase colors, code block colors, Hermes-style semi-transparent accents, Google Fonts Inter typography
- **Admin Hub** — Master-detail layout (30/70 split) instead of grid, responsive
- **Dashboard** — Real-time API data (System Status, Performance Metrics, Model Info, Events, Activity Chart), 15s auto-refresh, animated counters
- **Custom toast system** — Top-of-screen styled toasts with type icons and accent borders
- **Global keyboard shortcuts** — Ctrl+1-5 for tabs (from any screen), Ctrl+S save, Ctrl+K search
- **Config export** — Browser file download (not just clipboard)

**Complete Learning System (Issue #36)**
- **ExplorationExecutor** — Autonomously researches knowledge gaps via memory search
- **KnowledgeQAStore** — SQLite Q&A knowledge base with confidence tracking and verification
- **KnowledgeLineageTracker** — Provenance tracking per entity (file/web/conversation/feedback/exploration)
- **Gateway integration** — ActiveLearner starts on boot, CuriosityEngine scans every 5min, ConfidenceManager decays daily
- **Confidence persistence** — Feedback API reads/writes actual entity confidence in DB
- **14 API endpoints** under `/api/v1/learning/*` — stats, gaps, Q&A CRUD, lineage, exploration
- **Flutter Learning Dashboard** — 5 tabs (Overview, Gaps, Queue, Q&A, Lineage) with directory config

**Issue #35 Bug Fixes**
- PDF upload, version display, provider clarity, observe panel, search button, markdown rendering, Ollama timeout
- Identity auto-install in `start_cognithor.bat` and `install.sh`
- React UI deprecated, Flutter auto-download from GitHub release

**Code Quality**
- `ruff check src/cognithor/` — 0 errors (was 204)
- `flutter analyze` — 0 issues
- All tests passing

### v0.41.0-beta — Flutter UI, Active Learning, Knowledge Curiosity Engine

**Flutter Command Center — Full Feature Parity (React UI now deprecated)**
- **48 new Flutter files** — Complete cross-platform UI replacing React for mobile/tablet/desktop
- **18 editable config pages** — General, Language, Providers, Models, Planner, Executor, Memory, Channels, Security, Web, MCP, Cron, Database, Logging, Prompts, Agents, Bindings, System
- **Form widget library** — 12 custom widgets: Text, Number, Slider, Select, Toggle, List, DomainList, TextArea, JSON Editor, ReadOnly, CollapsibleCard
- **ConfigProvider** — Deep dot-path set, JSON snapshot dirty-tracking, parallel save, resilient loading with defaults
- **Observe Panel** — 4-tab side panel (Agent Log, Kanban, DAG, Plan) with phase icons, elapsed time, pipeline status indicator
- **Knowledge Graph** — Force-directed layout with 6 entity-type colors, node click details, search + type filter
- **Voice Mode** — 5-state machine (OFF/LISTENING/CONVERSATION/PROCESSING/SPEAKING), German phonetic wake-word detection, speech_to_text + just_audio TTS
- **Global Search** — Ctrl+K modal, 50 indexed terms across 18 pages
- **Theme Toggle** — Light/Dark mode with SharedPreferences persistence
- **Runtime Locale Switching** — LocaleProvider with 4 languages (EN/DE/ZH/AR), instant UI update
- **Keyboard shortcuts** — Ctrl+1-0 for page navigation, Ctrl+S to save

**Active Learning System (Issue #36)**
- **CuriosityEngine** — Detects knowledge gaps from low-confidence (<0.5) and stale (>90 days) entities, proposes prioritized exploration tasks
- **KnowledgeConfidenceManager** — Exponential time decay (180-day half-life), feedback-based adjustment (positive/negative/correction), verification boost, full audit history
- **ActiveLearner** — Background file watcher for ~/Documents and ~/Downloads, idle-time processing, content-hash deduplication, configurable learning rate
- **7 new API endpoints** — `/api/v1/learning/stats`, `/gaps`, `/gaps/{id}/dismiss`, `/confidence/history`, `/confidence/{id}/feedback`, `/queue`, `/explore`
- **Flutter Learning Dashboard** — 3-tab screen (Overview, Knowledge Gaps, Exploration Queue) with stats cards, activity chart, confidence history

**Issue #35 Bug Fixes**
- **PDF Upload** — File picker with explicit extensions, upload spinner, error handling
- **Version Display** — Reads from backend config, fallback "Unknown"
- **Provider Clarity** — Active provider at top with "ACTIVE PROVIDER" badge, inactive dimmed
- **Observe Panel** — Phase icons (brain/shield/play/refresh), elapsed time per entry, pipeline status indicator
- **Search Button** — Moved from FAB to clean AppBar buttons
- **Response Formatting** — Markdown with tappable links, styled code blocks
- **Ollama Timeout** — 10s health check timeout, clear "Backend nicht erreichbar" message

### v0.36.0-beta — 9 New Features: Roles, Delegation, Resume, Context Windows, Parallel Tools

- **Create / Operate / Live Role System** — Agents now have explicit roles: `orchestrator` (extended thinking, can spawn), `worker` (full MCP tool access), `monitor` (read-only). Default: `worker` for backward compat
- **Direction-based Delegation** — A2A messages gain a `direction` field: `remember` (memory-write), `act` (execute task), `notes` (fire-and-forget log). Role-based send permissions
- **Resume-as-Tool-Call** — Persistent checkpoints saved to disk (`~/.jarvis/checkpoints/`). Sessions can be resumed from last checkpoint via `cognithor_resume`
- **Per-Agent Context Windows** — Each agent owns an isolated `ContextWindow` with time-weighted trimming. System messages and tool results are never trimmed
- **Parallel Tool Calls** — Read-only MCP tools fire simultaneously via `asyncio.gather()`. Write tools remain sequential. Per-tool timeout (30s default)
- **Thinking / Execution Split** — Orchestrators think privately (Extended Thinking ON, not logged). Workers execute (logged). Cost tracking counts thinking tokens
- **Tab-as-Context-Window** — Browser tabs mapped to agent context via `TabContextBridge`. Tab state persists through checkpoints
- **Multi-Session Cognitive Base** — Persistent session management with cross-session Core Memory (max 2048 tokens, never auto-trimmed)
- **Priority-based Agent Scheduling** — Min-heap priority queue (1-10), 50/50 orchestrator/worker quota, platform-aware concurrency limits
- **Cross-platform Utilities** — `jarvis.utils.platform` module: `get_platform_name()`, `get_user_data_dir()`, `get_max_concurrent_agents()`, `supports_curses()`
- **11,769+ tests passing** (90 new feature tests + 10,814 existing, 0 regressions)

### v0.35.6-beta — Community-Reported Fixes (#26, #29, #33)

- **Search Button CSS Fix** — Global search trigger in legacy React Control Center was invisible (same background as header). Now uses `--bg3` for proper contrast in both light and dark themes (#26)
- **i18n Prompt Presets** — System prompts now load curated translations from `prompt_presets.py` (de/en/zh) instead of falling back to hardcoded German. Priority chain: Disk file -> i18n Preset -> Hardcoded (#33)
- **CORE.md Tool Deduplication** — Tool descriptions no longer dumped into CORE.md AND the Planner prompt. CORE.md now shows a one-line tool count reference; Planner gets localized, categorized descriptions via `ToolRegistryDB` (#29)
- **Prompt Evolution Guard** — Tool descriptions are now protected against mutation by the PromptEvolutionEngine. `locked` column on tools table + post-evolution validation rejects variants that remove `{tools_section}` (#29)
- **11,769+ tests passing** (0 failures)

### v0.34.4-beta — A2A Delegation, Sandbox Enforcement, Lint Zero

- **A2A Planner Delegation** — 2 new MCP tools (`list_remote_agents`, `delegate_to_remote_agent`) let the Planner autonomously discover and delegate tasks to remote A2A agents. Auto-discovery via `/.well-known/agent.json`
- **Sandbox Config Enforcement** — UI settings for `max_memory_mb`, `max_cpu_seconds`, and `network_access` now actually propagate to the execution sandbox (were previously ignored)
- **Proportional Iteration Caps** — `max_iterations` setting now scales coding task caps proportionally (80% for iteration cap, 30% for success threshold) instead of hardcoded limits
- **Auto-Update on Startup** — `plugins.auto_update` and `marketplace.auto_update` now trigger community registry sync at gateway startup
- **Lint Zero** — 393 lint errors (F401, F541, F841, E501, E741, E402) cleaned to zero across the entire codebase
- **MCP tools: 51 → 53** (added A2A delegation tools)
- **11,769+ tests passing** (0 failures)

### v0.34.3-beta — REPLAN Loop Fix, Full English UI, Tool Schemas

- **REPLAN Loop Fix** — Deep architectural fix to PGE loop: detects bare REPLAN text, consecutive no-tool iterations, and coding task caps to prevent infinite replanning
- **Full English UI** — All remaining German strings in legacy React Control Center translated (cron.js, A2A descriptions, icons, prompts)
- **Tool Schemas in CORE.md** — Auto-inventory now shows full parameter signatures (`tool(param: type *)`) instead of bare names
- **Chrome Autofill Defense** — GlobalSearch and provider filter inputs protected against Chrome autofill interference
- **Renamed "MCP & A2A" → "Integrations"** — Clearer page label in Command Center
- **Backend Startup Fix** — Vite now verifies jarvis importability before selecting a Python interpreter; bootstrap auto-repairs broken venvs

### v0.33.0-beta — i18n Language Packs, 4 Critical Bug Fixes

- **i18n Language Pack System** — JSON-based internationalization with dot-notation keys, SHA-256 integrity verification, fallback chain (locale → EN → raw key), thread-safe locale switching. Ships with German and English packs (~250 keys each)
- **Language Switcher in UI** — Command Center header quick-toggle (DE/EN) + General page dropdown. Language changes are live — no restart needed
- **Bug Fix: Planner JSON Parse Retry** — When the LLM returns malformed JSON, the planner now automatically retries with format hints instead of silently failing ("task failed successfully")
- **Bug Fix: LLM Timeout Wiring** — Embedding timeouts now respect the configured `timeout_seconds` instead of hardcoded values. LLM timeout is now visible on the Executor page for all backends
- **Bug Fix: WebSocket Race Condition** — All 12 `send_json()` calls in the WebSocket handler are now protected against disconnection errors via `_ws_safe_send()`. No more "Cannot call send" crashes
- **Bug Fix: GlobalSearch** — Added missing pages (Executor, Workflows, Knowledge Graph) to FIELD_INDEX and PAGE_LABELS. Search now finds all 19 config pages
- **11,769+ tests passing** (0 failures)

### v0.30.0 — mTLS, Document Reading, DB Retry

- **Document Reading** — 3 new MCP tools: `read_pdf` (PyMuPDF), `read_ppt` (python-pptx), `read_docx` (python-docx) with structured output, formatting, tables, images, metadata
- **mTLS for WebUI API** — Mutual TLS with auto-generated CA/server/client certificates; prevents unauthorized API access (`security.mtls.enabled`)
- **DB Retry Logic** — SQLite retries "database is locked" with exponential backoff + jitter (configurable)
- **MCP tools: 48 → 51**

**Previous Releases**

- **v0.29.1** — CI sandbox test fix, `pysqlcipher3` dependency fix, encryption extras fix
- **v0.29.0** — QA fixes: UI wiggle, unsaved changes, keyboard shortcuts, token tracking, SQLite encryption

- **UI Stability** — Layout wiggle fixed (`scrollbar-gutter: stable`), unsaved-changes false positives eliminated, keyboard shortcuts made sequential (Cmd+1..0)
- **Token Tracking** — `WorkingMemory.add_message()` now updates `token_count` live (was always 0)
- **SQLite Encryption** — Optional SQLCipher support with OS keyring key storage (`pip install cognithor[encryption]`)
- **Speed field removed** — Was a metadata-only field with no runtime effect; removed from Models UI

**Previous Releases**

- **v0.28.0** — Vite IPv6 fix, Qwen3-Coder model update, Python 3.15 locale compat
- **v0.27.5** "BugHunt" — 60+ CodeQL fixes, CI stability, thread-safe EpisodicStore, 11,769+ tests
- **v0.27.3** — CWE-22 Path Traversal fix in TTS API, multi-GPU installer fix, `--init-only` hang fix
- **v0.27.1** — Community Skill Marketplace, ToolEnforcer runtime sandboxing, 5-check validation pipeline, 13 autonomy fixes
- **v0.27.0** — Full Audit, Installer Overhaul: 80-item audit, XSS fix, CORS hardening, rate limiting, auto-install Python/Ollama
- **v0.26.7** — Wiring: DAG-based parallel executor, http_request tool with SSRF protection, sub-agent depth guard, live config reload
- **v0.26.6** — Chat & Voice: Integrated chat page, voice mode with wake word, Piper TTS, 15 agent infrastructure subsystems
- **v0.26.5** — Human Feel: Personality Engine, sentiment detection, user preferences, status callbacks, friendly error messages
- **v0.26.0-v0.26.4** — Security hardening, Docker prod, LM Studio backend, scaling, coverage & skills

---
