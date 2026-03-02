<p align="center">
  <h1 align="center">Cognithor &middot; Agent OS</h1>
  <p align="center">
    <strong>The local-first, autonomous agent operating system that replaces your entire AI toolchain.</strong>
  </p>
  <p align="center">
    <em>Cognition + Thor — Intelligence with Power</em>
  </p>
  <p align="center">
    <a href="#llm-providers">16 LLM Providers</a> &middot; <a href="#channels">17 Channels</a> &middot; <a href="#5-tier-cognitive-memory">5-Tier Memory</a> &middot; <a href="#knowledge-vault">Knowledge Vault</a> &middot; <a href="#security">Enterprise Security</a> &middot; <a href="LICENSE">Apache 2.0</a>
  </p>
  <p align="center">
    <a href="#quick-start"><img src="https://img.shields.io/badge/python-%3E%3D3.12-blue?style=flat-square" alt="Python"></a>
    <a href="#tests"><img src="https://img.shields.io/badge/tests-8%2C306%20passing-brightgreen?style=flat-square" alt="Tests"></a>
    <a href="#tests"><img src="https://img.shields.io/badge/coverage-89%25-brightgreen?style=flat-square" alt="Coverage"></a>
    <a href="#tests"><img src="https://img.shields.io/badge/lint-0%20errors-brightgreen?style=flat-square" alt="Lint"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue?style=flat-square" alt="License"></a>
  </p>
</p>

---

## Why Cognithor?

- **Your data never leaves your machine.** Runs 100% locally with Ollama or LM Studio — no cloud, no API keys required, full GDPR compliance. Cloud providers are optional, not mandatory.
- **One system, not twenty tools.** 17 channels, 18+ MCP tool servers, 5-tier memory, knowledge vault, cron, voice, browser automation, distributed locking, durable message queues — integrated from day one. No glue code, no plugin hell.
- **Production-grade, not a demo.** 8,306 tests, 89% coverage, 4-level sandbox, SHA-256 audit chain, runtime token encryption, Prometheus metrics, and battle-tested deployment options from one-click Windows launcher to Docker Compose to bare-metal servers.

---

> **Cognithor** is a fully local, Ollama/LM Studio-powered, autonomous agent operating system that acts as your personal AI assistant. All data stays on your machine — no cloud, no mandatory API keys, full GDPR compliance. It supports tasks ranging from research, project management, and knowledge organization to file management and automated workflows. Optional cloud LLM providers (OpenAI, Anthropic, Gemini, and 11 more) can be enabled with a single API key. Users can add custom skills and rules to tailor the agent to their needs.

<p align="center">
  <img src="demo.svg" alt="Cognithor Demo" width="100%">
</p>

## Table of Contents

- [Why Cognithor?](#why-cognithor)
- [What's New in v0.26](#whats-new-in-v026)
- [Highlights](#highlights)
- [Architecture](#architecture)
- [LLM Providers](#llm-providers)
- [Channels](#channels)
- [Quick Start](#quick-start) (under 5 minutes)
- [Configuration](#configuration)
- [Security](#security)
- [MCP Tools](#mcp-tools)
- [Tests](#tests)
- [Code Quality](#code-quality)
- [Project Structure](#project-structure)
- [Deployment](#deployment)
- [Roadmap](#roadmap)
- [Recording a Demo](#recording-a-demo)
- [License](#license)

---

## What's New in v0.26

This release focuses on **production scaling** and **operational maturity**:

- **Distributed Locking** — Redis-backed (with file-based fallback) distributed locks for safe multi-instance deployments. Prevents concurrent plan execution and resource contention across nodes.
- **Durable Message Queue** — SQLite-backed persistent message queue with priority levels, dead-letter queue (DLQ), and automatic retry. Messages survive restarts and crashes.
- **Telegram Webhook Mode** — In addition to polling, Telegram now supports webhook mode with sub-100ms latency. Automatically configures TLS when certificates are provided.
- **Prometheus /metrics Endpoint** — Full Prometheus-compatible metrics export: request counts, latency histograms, memory usage, queue depths, LLM token consumption. Ships with a ready-made Grafana dashboard JSON.
- **Skill Marketplace Persistence** — Skill marketplace now backed by SQLite with full CRUD, ratings, search, and a REST API for marketplace operations.
- **Auto-Dependency Loading** — On startup, Cognithor inspects enabled features and installs missing optional dependencies automatically, eliminating manual `pip install` steps for feature groups.
- **2,700+ new tests** covering all scaling features, security hardening, and comprehensive coverage improvements, bringing the total to **8,306 tests** at **89% coverage**.
- **LM Studio Support** — LM Studio als zweites lokales Backend neben Ollama. OpenAI-kompatible API auf `localhost:1234`, kein API-Key nötig. Einfach `llm_backend_type: "lmstudio"` setzen.

---

## Highlights

- **16 LLM Providers** — Ollama (local), LM Studio (local), OpenAI, Anthropic, Google Gemini, Groq, DeepSeek, Mistral, Together AI, OpenRouter, xAI (Grok), Cerebras, GitHub Models, AWS Bedrock, Hugging Face, Moonshot/Kimi
- **17 Communication Channels** — CLI, Web UI, REST API, Telegram, Discord, Slack, WhatsApp, Signal, iMessage, Microsoft Teams, Matrix, Google Chat, Mattermost, Feishu/Lark, IRC, Twitch, Voice (STT/TTS)
- **5-Tier Cognitive Memory** — Core identity, episodic logs, semantic knowledge graph, procedural skills, working memory
- **3-Channel Hybrid Search** — BM25 full-text + vector embeddings + knowledge graph traversal with score fusion
- **PGE Architecture** — Planner (LLM) -> Gatekeeper (deterministic policy engine) -> Executor (sandboxed)
- **Enterprise Security** — 4-level sandbox, SHA-256 audit chain, EU AI Act compliance, credential vault, red-teaming, runtime token encryption (Fernet AES-256), TLS support, file-size limits
- **Knowledge Vault** — Obsidian-compatible Markdown vault with YAML frontmatter, tags, `[[backlinks]]`, full-text search
- **Document Analysis** — LLM-powered structured analysis of PDF/DOCX/HTML (summary, risks, action items, decisions)
- **Model Context Protocol (MCP)** — 18+ tool servers (filesystem, shell, memory, web, browser, media, vault, synthesis)
- **Distributed Locking** — Redis-backed (with file-based fallback) locks for multi-instance deployments
- **Durable Message Queue** — SQLite-backed persistent queue with priorities, DLQ, and automatic retry
- **Prometheus Metrics** — /metrics endpoint with Grafana dashboard for production observability
- **Skill Marketplace** — SQLite-persisted skill marketplace with ratings, search, and REST API
- **Telegram Webhook** — Polling + webhook mode with sub-100ms latency
- **Auto-Dependency Loading** — Missing optional packages detected and installed at startup
- **Agent-to-Agent Protocol (A2A)** — Linux Foundation RC v1.0 for inter-agent communication
- **React Control Center** — Full web dashboard (React 19 + Vite 7) with integrated backend launcher, live config editing, agent management, prompt editing, cron jobs, MCP servers, and A2A settings
- **Auto-Detect Channels** — Channels activate automatically when tokens are present in `.env` — no manual config flags needed
- **Knowledge Synthesis** — Meta-analysis across Memory + Vault + Web with LLM fusion: `knowledge_synthesize` (full synthesis with confidence ratings), `knowledge_contradictions` (fact-checking), `knowledge_timeline` (causal chains), `knowledge_gaps` (completeness score + research suggestions)
- **Adaptive Context Pipeline** — Automatic context enrichment before every Planner call: BM25 memory search + vault full-text search + recent episodes, injected into WorkingMemory in <50ms
- **Security Hardening** — Runtime token encryption (Fernet AES-256) across all channels, TLS support for webhook servers, file-size limits on all upload/processing paths, persistent session mappings in SQLite
- **One-Click Start** — Double-click `start_cognithor.bat` -> browser opens -> click **Power On** -> done
- **Enhanced Web Research** — 4-provider search fallback (SearXNG -> Brave -> Google CSE -> DuckDuckGo), Jina AI Reader for JS-heavy sites, domain filtering, source cross-checking
- **Procedural Learning** — Reflector auto-synthesizes reusable skills from successful sessions
- **8,306 tests** · **89% coverage** · **0 lint errors**

## Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│            Control Center UI (React 19 + Vite 7)                  │
│   Config · Agents · Prompts · Cron · MCP · A2A · Status          │
├───────────────────────────────────────────────────────────────────┤
│         Prometheus /metrics · Grafana Dashboard                    │
├───────────────────────────────────────────────────────────────────┤
│           REST API (FastAPI, 20+ endpoints, port 8741)            │
├───────────────────────────────────────────────────────────────────┤
│                       Channels (17)                                │
│   CLI · Web · Telegram (poll+webhook) · Discord · Slack           │
│   WhatsApp · Signal · iMessage · Teams · Matrix · Voice · ...     │
├───────────────────────────────────────────────────────────────────┤
│                     Gateway Layer                                  │
│      Session Management · Agent Loop · Distributed Lock            │
├───────────────────────────────────────────────────────────────────┤
│        Durable Message Queue (SQLite, priorities, DLQ)             │
├───────────────────────────────────────────────────────────────────┤
│           Context Pipeline (Memory · Vault · Episodes)             │
├─────────────┬──────────────┬──────────────────────────────────────┤
│  Planner    │  Gatekeeper  │  Executor                            │
│  (LLM)      │  (Policy)    │  (Sandbox)                           │
├─────────────┴──────────────┴──────────────────────────────────────┤
│                   MCP Tool Layer (18+)                              │
│   Filesystem · Shell · Memory · Web · Browser · Media · Vault      │
│   Synthesis · Skills Marketplace                                    │
├───────────────────────────────────────────────────────────────────┤
│               Multi-LLM Backend Layer (16)                         │
│   Ollama · OpenAI · Anthropic · Gemini · Groq · DeepSeek           │
│   Mistral · Together · OpenRouter · xAI · Cerebras · ...           │
├───────────────────────────────────────────────────────────────────┤
│               5-Tier Cognitive Memory                               │
│   Core · Episodic · Semantic · Procedural · Working                 │
├───────────────────────────────────────────────────────────────────┤
│         Infrastructure: Redis/File Distributed Lock                 │
│         SQLite Durable Queue · Prometheus Telemetry                 │
└───────────────────────────────────────────────────────────────────┘
```

### PGE Trinity (Planner -> Gatekeeper -> Executor)

Every user request passes through three stages:

1. **Planner** — LLM-based understanding and planning. Analyzes the request, searches memory for relevant context, creates structured action plans with tool calls. Supports re-planning on failures.
2. **Gatekeeper** — Deterministic policy engine. Validates every planned tool call against security rules (risk levels GREEN/YELLOW/ORANGE/RED, sandbox policy, parameter validation). No LLM, no hallucinations, no exceptions.
3. **Executor** — Executes approved actions in a multi-level sandbox. Shell commands run isolated (Process -> Namespace -> Container), file access restricted to allowed paths.

### 5-Tier Cognitive Memory

| Tier | Name | Persistence | Purpose |
|------|------|------------|---------|
| 1 | **Core** | `CORE.md` | Identity, rules, personality |
| 2 | **Episodic** | Daily log files | What happened today/yesterday |
| 3 | **Semantic** | Knowledge graph + SQLite | Customers, products, facts, relations |
| 4 | **Procedural** | Markdown + frontmatter | Learned skills and workflows |
| 5 | **Working** | RAM (volatile) | Active session context |

Memory search uses a 3-channel hybrid approach: **BM25** (full-text search with FTS5, optimized for German compound words) + **Vector Search** (Ollama embeddings, cosine similarity) + **Graph Traversal** (entity relations). Score fusion with configurable weights and recency decay.

### Knowledge Vault

In addition to the 5-tier memory, Cognithor includes an **Obsidian-compatible Knowledge Vault** (`~/.jarvis/vault/`) for persistent, human-readable notes:

- **Folder structure**: `recherchen/`, `meetings/`, `wissen/`, `projekte/`, `daily/`
- **Obsidian format**: YAML frontmatter (title, tags, sources, dates), `[[backlinks]]`
- **6 tools**: `vault_save`, `vault_search`, `vault_list`, `vault_read`, `vault_update`, `vault_link`
- Open the vault folder directly in [Obsidian](https://obsidian.md) for graph visualization

### Reflection & Procedural Learning

After completed sessions, the Reflector evaluates results, extracts facts for semantic memory, and identifies repeatable patterns as procedure candidates. Learned procedures are automatically suggested for future similar requests.

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
| **CLI** | Terminal REPL | Rich formatting, streaming, `/commands` |
| **Web UI** | WebSocket | Real-time streaming, voice recording, file upload, dark theme |
| **REST API** | FastAPI + SSE | Programmatic access, server-sent events |
| **Telegram** | Bot API (poll + webhook) | Text, voice messages (Whisper STT), photos, documents, webhook mode (<100ms) |
| **Discord** | Gateway + REST | Embeds, reactions, thread support |
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
| **Voice** | Whisper + Piper + ElevenLabs | STT, TTS, wake word, talk mode |

## Demo

```bash
python demo.py           # Full experience (~3 minutes)
python demo.py --fast    # Speed run (~15 seconds)
```

## Quick Start

**Time: under 5 minutes from clone to running agent.**

### Prerequisites

- Python >= 3.12
- **LLM Backend** (one of):
  - [Ollama](https://ollama.ai) — local, free, GDPR-compliant (recommended)
  - [LM Studio](https://lmstudio.ai) — local, OpenAI-compatible API on port 1234
  - Any of the 14 cloud providers listed above (just an API key)
- Optional: `playwright` for browser automation, `faster-whisper` for voice

### Step 1: Clone and Install (~2 min)

```bash
# Clone the repository
git clone https://github.com/Alex8791-cyber/cognithor.git
cd cognithor

# Recommended: Interactive installation (venv, Ollama check, systemd, smoke test)
./install.sh

# Or: Manual installation
pip install -e ".[all,dev]"

# Individual feature groups (install only what you need)
pip install -e ".[telegram,voice,web,cron]"
```

The installer offers four modes: `--minimal` (core only), `--full` (all features), `--systemd` (+ service installation), `--uninstall` (removal). Without flags, it starts in interactive mode.

### Step 2: Pull an LLM (~2 min)

```bash
ollama pull qwen3:32b           # Planner (20 GB VRAM)
ollama pull qwen3:8b            # Executor (6 GB VRAM)
ollama pull nomic-embed-text    # Embeddings (300 MB VRAM)
# Optional:
ollama pull qwen3-coder:32b     # Code tasks (20 GB VRAM)
```

No GPU? Use smaller models (`qwen3:8b` for both) or a cloud provider — just set one API key.

### Step 3: Start (~10 sec)

**Option A: One-Click (Windows)**

```
Double-click  start_cognithor.bat  ->  Browser opens  ->  Click "Power On"  ->  Done.
```

**Option B: CLI**

```bash
cognithor                          # Interactive CLI
# or: python -m jarvis

python -m jarvis --no-cli          # Headless mode (API only)
JARVIS_HOME=~/my-cognithor cognithor  # Custom home directory
```

**Option C: Control Center UI (Development)**

```bash
cd ui
npm install
npm run dev    # -> http://localhost:5173
```

Click **Power On** to start the backend directly from the UI. The Vite dev server automatically spawns and manages the Python backend process on port 8741 — including orphan detection, clean shutdown, and process lifecycle management.

All configuration — agents, prompts, cron jobs, MCP servers, A2A settings — can be edited and saved through the dashboard. Changes persist to YAML files under `~/.jarvis/`.

> **Windows users:** A desktop shortcut named **Cognithor** is included for convenience.

### Channel Auto-Detection

Channels start automatically when their tokens are found in `~/.jarvis/.env`:

```bash
# ~/.jarvis/.env — just add your tokens, channels activate automatically
JARVIS_TELEGRAM_TOKEN=your-bot-token
JARVIS_TELEGRAM_ALLOWED_USERS=123456789
JARVIS_DISCORD_TOKEN=your-discord-token
JARVIS_SLACK_TOKEN=xoxb-your-slack-token
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
├── mcp/
│   └── config.yaml      # MCP server configuration
├── queue/
│   └── messages.db      # Durable message queue (SQLite)
└── logs/
    └── cognithor.log    # Structured logs (JSON)
```

## Configuration

Cognithor is configured via `~/.cognithor/config.yaml`. All values can be overridden with environment variables using the `JARVIS_` prefix (legacy) or `COGNITHOR_` prefix.

```yaml
# Example: ~/.cognithor/config.yaml
owner_name: "Alex"

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
  path: "~/.jarvis/vault"
  # auto_save_research: false     # Auto-save web research results

channels:
  cli_enabled: true
  # Channels auto-detect from tokens in ~/.jarvis/.env
  # Set to false only to explicitly disable a channel:
  # telegram_enabled: false

security:
  allowed_paths:
    - "~/.cognithor"
    - "~/Documents"

# Scaling (new in v0.26)
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

Cognithor implements enterprise-grade security:

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
| **EU AI Act** | Compliance module, impact assessments, transparency reports |
| **Red-Teaming** | Automated offensive security tests (1,425 LOC) |

## MCP Tools

| Tool Server | Tools | Description |
|-------------|-------|-------------|
| **Filesystem** | read, write, edit, list, delete | Path-sandboxed file operations |
| **Shell** | exec_command | Sandboxed command execution with timeout |
| **Memory** | search, save, get_entity, add_entity, ... | 10 memory tools across all 5 tiers |
| **Web** | web_search, web_fetch, search_and_read, web_news_search | 4-provider search (SearXNG -> Brave -> Google CSE -> DDG), Jina Reader fallback, domain filtering, cross-check |
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

Current status: **8,306 tests** · **100% pass rate** · **89% coverage** · **~97,000 LOC source** · **~79,000 LOC tests**

| Area | Tests | Description |
|------|-------|-------------|
| Core | 1,422 | Planner, Gatekeeper, Executor, Config, Models, Reflector, Distributed Lock, Model Router, Unified LLM, LLM Backend |
| Integration | 1,314 | End-to-end tests, phase wiring, entrypoint, A2A protocol |
| Channels | 1,360 | CLI, Telegram (incl. Webhook), Discord, Slack, WhatsApp, API, WebUI, Voice, iMessage, Signal, Teams |
| MCP | 825 | Client, filesystem, shell, memory server, web, media, synthesis, vault, browser, bridge, resources |
| Memory | 658 | All 5 tiers, indexer, hybrid search, chunker, watcher, token estimation, integrity, hygiene |
| Skills | 498 | Skill registry, generator, marketplace, persistence, API, CLI tools, scaffolder, linter, BaseSkill |
| Security | 327 | Audit, credentials, token store, TLS, policies, sandbox, sanitizer, agent vault |
| Gateway | 252 | Session management, agent loop, context pipeline, phase init, approval flow |
| A2A | 158 | Agent-to-Agent protocol, client, HTTP handler, streaming |
| Telemetry | 120 | Cost tracking, metrics, tracing, Prometheus export, instrumentation |
| Other | 247 | HITL, governance, learning, proactive, config manager |
| Tools | 103 | Refactoring agent, code analyzer, skill CLI developer tools |
| Utils | 72 | Logging, helper functions |
| Cron | 63 | Engine, job store, scheduling |
| UI API | 55 | Control Center endpoints (config, agents, prompts, cron, MCP, A2A) |

## Code Quality

```bash
make lint        # Ruff linting (0 errors)
make format      # Ruff formatting
make typecheck   # MyPy strict type checking
make check       # All combined (lint + typecheck + tests)
make smoke       # Installation validation (26 checks)
make health      # Runtime check (Ollama, disk, memory, audit)
```

## Project Structure

```
cognithor/
├── src/jarvis/                    # Python backend
│   ├── config.py                  # Configuration system (YAML + env vars)
│   ├── config_manager.py          # Runtime config management (read/update/save)
│   ├── models.py                  # Pydantic data models (58+ classes)
│   ├── core/
│   │   ├── planner.py             # LLM planner with re-planning
│   │   ├── gatekeeper.py          # Deterministic policy engine (no LLM)
│   │   ├── executor.py            # Sandboxed tool executor with audit trail
│   │   ├── model_router.py        # Model selection by task type
│   │   ├── llm_backend.py         # Multi-provider LLM abstraction (16 backends)
│   │   ├── orchestrator.py        # High-level agent orchestration
│   │   ├── reflector.py           # Reflection, fact extraction, skill synthesis
│   │   └── distributed_lock.py    # Redis/file-based distributed locking
│   ├── memory/
│   │   ├── manager.py             # Central memory API (all 5 tiers)
│   │   ├── core_memory.py         # Tier 1: CORE.md management
│   │   ├── episodic.py            # Tier 2: Daily logs (Markdown)
│   │   ├── semantic.py            # Tier 3: Knowledge graph (entities + relations)
│   │   ├── procedural.py          # Tier 4: Skills (YAML frontmatter + Markdown)
│   │   ├── working.py             # Tier 5: Session context (RAM)
│   │   ├── indexer.py             # SQLite index (FTS5 + entities + vectors)
│   │   ├── search.py              # 3-channel hybrid search (BM25 + vector + graph)
│   │   ├── embeddings.py          # Embedding client with LRU cache
│   │   ├── chunker.py             # Markdown-aware sliding window chunker
│   │   └── watcher.py             # Auto-reindexing (watchdog/polling)
│   ├── mcp/
│   │   ├── client.py              # Multi-server MCP client (stdio + builtin)
│   │   ├── server.py              # Jarvis as MCP server
│   │   ├── filesystem.py          # File tools (path sandbox)
│   │   ├── shell.py               # Shell execution (timeout, sandbox)
│   │   ├── memory_server.py       # Memory as 10 MCP tools
│   │   ├── web.py                 # Enhanced web search (4 providers) and URL fetch (Jina fallback)
│   │   ├── vault.py               # Knowledge Vault (Obsidian-compatible, 6 tools)
│   │   ├── synthesis.py           # Knowledge Synthesis (4 tools: synthesize, contradictions, timeline, gaps)
│   │   ├── browser.py             # Browser automation (Playwright, 6 tools)
│   │   └── media.py               # Media pipeline (STT, TTS, image, PDF, document analysis, 8 tools)
│   ├── gateway/
│   │   ├── gateway.py             # Agent loop, session management, subsystem init
│   │   └── message_queue.py       # Durable SQLite-backed message queue (priorities, DLQ)
│   ├── channels/                  # 17 communication channels + Control Center API
│   │   ├── base.py                # Abstract channel interface
│   │   ├── config_routes.py       # REST API for Control Center (20+ endpoints)
│   │   ├── cli.py, api.py         # Core channels
│   │   ├── telegram.py            # Telegram (polling + webhook mode)
│   │   ├── discord.py             # Discord
│   │   ├── whatsapp.py, signal.py # Encrypted messaging
│   │   ├── voice.py               # Voice I/O (STT/TTS)
│   │   └── ...                    # Teams, Matrix, IRC, Twitch, Mattermost, etc.
│   ├── security/
│   │   ├── audit.py               # Append-only audit trail (SHA-256 chain)
│   │   ├── credentials.py         # Credential store (Fernet encrypted)
│   │   ├── token_store.py         # Runtime token encryption (ephemeral Fernet) + TLS helper
│   │   ├── sandbox.py             # Multi-level sandbox (L0-L2)
│   │   ├── policies.py            # Security policies (path, command, network)
│   │   └── sanitizer.py           # Input sanitization (injection protection)
│   ├── cron/                      # Cron engine with APScheduler
│   ├── a2a/                       # Agent-to-Agent protocol (Linux Foundation RC v1.0)
│   ├── skills/                    # Skill registry, generator, marketplace (SQLite persistence)
│   ├── graph/                     # Knowledge graph engine
│   ├── telemetry/                 # Cost tracking, metrics, tracing, Prometheus export
│   └── utils/
│       └── logging.py             # Structured logging (structlog + Rich)
├── ui/                            # Control Center (React 19 + Vite 7)
│   ├── vite.config.js             # Dev server with backend launcher plugin
│   ├── package.json               # Dependencies (react, vite)
│   ├── index.html                 # Entry point
│   └── src/
│       ├── CognithorControlCenter.jsx  # Main dashboard (1,700 LOC)
│       ├── App.jsx                # App shell
│       └── main.jsx               # React entry
├── tests/                         # 8,306 tests, ~77,000 LOC
│   ├── test_core/                 # Planner, Gatekeeper, Executor, Distributed Lock
│   ├── test_memory/               # All 5 memory tiers, hybrid search
│   ├── test_mcp/                  # MCP tools and client
│   ├── test_channels/             # All channel implementations (incl. Webhook)
│   ├── test_security/             # Audit, sandbox, policies
│   ├── test_integration/          # End-to-end tests
│   ├── test_skills/               # Skills, marketplace, persistence
│   ├── test_telemetry/            # Metrics, Prometheus export
│   ├── test_config_manager.py     # Config manager + API routes
│   └── test_ui_api_integration.py # 55 Control Center API integration tests
├── skills/                        # Built-in skill definitions
├── scripts/                       # Backup, deployment, utilities
├── deploy/                        # Docker, systemd, nginx, Caddy, bare-metal installer
├── apps/                          # PWA app (legacy)
├── start_cognithor.bat            # One-click launcher (Windows)
├── config.yaml.example            # Example configuration
├── pyproject.toml                 # Python project metadata
├── Makefile                       # Build, test, lint commands
├── Dockerfile                     # Container image
├── docker-compose.yml             # Development compose
├── docker-compose.prod.yml        # Production compose (5 services + profiles)
└── install.sh                     # Interactive installer
```

## Deployment

### One-Click (Windows)

Double-click `start_cognithor.bat` -> browser opens -> click **Power On** -> done.

### Docker (Development)

```bash
docker compose up -d                         # Core backend
docker compose --profile webui up -d         # + Web UI
```

### Docker (Production)

```bash
cp .env.example .env   # Edit: set JARVIS_API_TOKEN, etc.
docker compose -f docker-compose.prod.yml up -d

# With optional services
docker compose -f docker-compose.prod.yml --profile postgres up -d   # + PostgreSQL
docker compose -f docker-compose.prod.yml --profile nginx up -d      # + Nginx TLS
docker compose -f docker-compose.prod.yml --profile monitoring up -d # + Prometheus + Grafana
```

Services: Jarvis (headless) + WebUI + Ollama + optional PostgreSQL (pgvector) + optional Nginx reverse proxy + optional Prometheus/Grafana monitoring. GPU support via nvidia-container-toolkit (uncomment in compose file).

### Bare-Metal Server (Ubuntu/Debian)

```bash
sudo bash deploy/install-server.sh --domain jarvis.example.com --email admin@example.com
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
curl http://localhost:8741/api/v1/health     # Control Center API
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

## Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 1** | Foundation (PGE trinity, MCP, CLI) | Done |
| **Phase 2** | Memory (5-tier, hybrid search, MCP tools) | Done |
| **Phase 3** | Reflection & procedural learning | Done |
| **Phase 4** | Channels, cron, web tools, model router | Done |
| **Phase 5** | Multi-agent & security hardening | Done |
| **Phase 6** | Web UI & voice | Done |
| **Phase 7** | Control Center UI, API integration, channel auto-detect | Done |
| **Phase 8** | UI integration into repo, backend launcher, orphan management | Done |
| **Phase 9** | Security hardening: token encryption, TLS, file-size limits, session persistence | Done |
| **Phase 10** | Server deployment: Docker prod, bare-metal installer, Nginx/Caddy, health endpoints | Done |
| **Phase 11** | Scaling: distributed locking, durable message queue, Prometheus metrics, Telegram webhook, skill marketplace persistence, auto-dependency loading | Done |
| **Deploy** | Installer, systemd, Docker, backup, smoke test, one-click launcher | Done |

### What's Next

- **Phase 12** — Horizontal scaling: multi-node Gateway with Redis Streams, auto-sharding of memory tiers, load-balanced LLM backend pools
- **Phase 13** — Advanced agents: multi-agent collaboration workflows, shared memory spaces, agent specialization profiles
- **Phase 14** — Mobile: native Android/iOS apps via Capacitor, push notifications, offline mode with local LLM

## Recording a Demo

To create a terminal recording for your README or documentation:

```bash
# Install asciinema
pip install asciinema

# Record a session
asciinema rec demo.cast

# Convert to GIF (requires agg)
# https://github.com/asciinema/agg
agg demo.cast demo.gif
```

Alternatively, use [terminalizer](https://github.com/faressoft/terminalizer) for customizable terminal GIFs, or [VHS](https://github.com/charmbracelet/vhs) for scripted recordings.

---

**Metrics:** ~97,000 LOC source · ~79,000 LOC tests · 8,306 tests · 89% coverage · 0 lint errors

## License

Apache 2.0 — see [LICENSE](LICENSE)

Copyright 2026 Alexander Soellner
