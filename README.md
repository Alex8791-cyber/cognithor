<p align="center">
  <h1 align="center">Cognithor · Agent OS</h1>
  <p align="center">
    <strong>Local-first, autonomous agent operating system for personal AI assistance.</strong>
  </p>
  <p align="center">
    <em>Cognition + Thor — Intelligence with Power</em>
  </p>
  <p align="center">
    <a href="#llm-providers">15 LLM Providers</a> · <a href="#channels">17 Channels</a> · <a href="#5-tier-cognitive-memory">5-Tier Memory</a> · <a href="#knowledge-vault">Knowledge Vault</a> · <a href="#security">Enterprise Security</a> · <a href="LICENSE">Apache 2.0</a>
  </p>
  <p align="center">
    <a href="#quick-start"><img src="https://img.shields.io/badge/python-%3E%3D3.12-blue?style=flat-square" alt="Python"></a>
    <a href="#tests"><img src="https://img.shields.io/badge/tests-4%2C746%20passing-brightgreen?style=flat-square" alt="Tests"></a>
    <a href="#tests"><img src="https://img.shields.io/badge/coverage-89%25-brightgreen?style=flat-square" alt="Coverage"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue?style=flat-square" alt="License"></a>
  </p>
</p>

---

> **Cognithor** is a fully local, Ollama-powered, optional API-dependant, autonomous agent operating system that acts as your personal AI assistant. All data stays on your machine — no cloud, no mandatory API keys, full GDPR compliance. It supports tasks ranging from research, project management, and knowledge organization to file management and automated workflows. Users can add custom skills and rules to tailor the agent to their needs.

<p align="center">
  <img src="demo.svg" alt="Cognithor Demo" width="100%">
</p>

## Highlights

- **15 LLM Providers** — Ollama (local), OpenAI, Anthropic, Google Gemini, Groq, DeepSeek, Mistral, Together AI, OpenRouter, xAI (Grok), Cerebras, GitHub Models, AWS Bedrock, Hugging Face, Moonshot/Kimi
- **17 Communication Channels** — CLI, Web UI, REST API, Telegram, Discord, Slack, WhatsApp, Signal, iMessage, Microsoft Teams, Matrix, Google Chat, Mattermost, Feishu/Lark, IRC, Twitch, Voice (STT/TTS)
- **5-Tier Cognitive Memory** — Core identity, episodic logs, semantic knowledge graph, procedural skills, working memory
- **3-Channel Hybrid Search** — BM25 full-text + vector embeddings + knowledge graph traversal with score fusion
- **PGE Architecture** — Planner (LLM) → Gatekeeper (deterministic policy engine) → Executor (sandboxed)
- **Enterprise Security** — 4-level sandbox, SHA-256 audit chain, EU AI Act compliance, credential vault, red-teaming
- **Knowledge Vault** — Obsidian-compatible Markdown vault with YAML frontmatter, tags, `[[backlinks]]`, full-text search
- **Document Analysis** — LLM-powered structured analysis of PDF/DOCX/HTML (summary, risks, action items, decisions)
- **Model Context Protocol (MCP)** — 18+ tool servers (filesystem, shell, memory, web, browser, media, vault, synthesis)
- **Agent-to-Agent Protocol (A2A)** — Linux Foundation RC v1.0 for inter-agent communication
- **React Control Center** — Full web dashboard (React 19 + Vite 7) with integrated backend launcher, live config editing, agent management, prompt editing, cron jobs, MCP servers, and A2A settings
- **Auto-Detect Channels** — Channels activate automatically when tokens are present in `.env` — no manual config flags needed
- **Knowledge Synthesis** — Meta-analysis across Memory + Vault + Web with LLM fusion: `knowledge_synthesize` (full synthesis with confidence ratings), `knowledge_contradictions` (fact-checking), `knowledge_timeline` (causal chains), `knowledge_gaps` (completeness score + research suggestions)
- **Enhanced Web Research** — 4-provider search fallback (SearXNG → Brave → Google CSE → DuckDuckGo), Jina AI Reader for JS-heavy sites, domain filtering, source cross-checking
- **Procedural Learning** — Reflector auto-synthesizes reusable skills from successful sessions
- **4,746 tests** · **89% coverage** · **0 lint errors**

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│           Control Center UI (React 19 + Vite 7)              │
│  Config · Agents · Prompts · Cron · MCP · A2A · Status       │
├──────────────────────────────────────────────────────────────┤
│          REST API (FastAPI, 20+ endpoints, port 8741)        │
├──────────────────────────────────────────────────────────────┤
│                      Channels (17)                           │
│  CLI · Web · Telegram · Discord · Slack · WhatsApp · Signal  │
│  iMessage · Teams · Matrix · Voice · IRC · Twitch · ...      │
├──────────────────────────────────────────────────────────────┤
│                    Gateway Layer                              │
│            Session Management · Agent Loop                    │
├────────────┬──────────────┬──────────────────────────────────┤
│  Planner   │  Gatekeeper  │  Executor                        │
│  (LLM)     │  (Policy)    │  (Sandbox)                       │
├────────────┴──────────────┴──────────────────────────────────┤
│                  MCP Tool Layer (18+)                         │
│  Filesystem · Shell · Memory · Web · Browser · Media · Vault  │
│  Synthesis                                                     │
├──────────────────────────────────────────────────────────────┤
│              Multi-LLM Backend Layer (15)                     │
│  Ollama · OpenAI · Anthropic · Gemini · Groq · DeepSeek      │
│  Mistral · Together · OpenRouter · xAI · Cerebras · ...       │
├──────────────────────────────────────────────────────────────┤
│              5-Tier Cognitive Memory                          │
│  Core · Episodic · Semantic · Procedural · Working            │
└──────────────────────────────────────────────────────────────┘
```

### PGE Trinity (Planner → Gatekeeper → Executor)

Every user request passes through three stages:

1. **Planner** — LLM-based understanding and planning. Analyzes the request, searches memory for relevant context, creates structured action plans with tool calls. Supports re-planning on failures.
2. **Gatekeeper** — Deterministic policy engine. Validates every planned tool call against security rules (risk levels GREEN/YELLOW/ORANGE/RED, sandbox policy, parameter validation). No LLM, no hallucinations, no exceptions.
3. **Executor** — Executes approved actions in a multi-level sandbox. Shell commands run isolated (Process → Namespace → Container), file access restricted to allowed paths.

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
```

## Channels

| Channel | Protocol | Features |
|---------|----------|----------|
| **CLI** | Terminal REPL | Rich formatting, streaming, `/commands` |
| **Web UI** | WebSocket | Real-time streaming, voice recording, file upload, dark theme |
| **REST API** | FastAPI + SSE | Programmatic access, server-sent events |
| **Telegram** | Bot API | Text, voice messages (Whisper STT), photos, documents |
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
│   │   ├── llm_backend.py         # Multi-provider LLM abstraction (15 backends)
│   │   ├── orchestrator.py        # High-level agent orchestration
│   │   └── reflector.py           # Reflection, fact extraction, skill synthesis
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
│   │   └── gateway.py             # Agent loop, session management, subsystem init
│   ├── channels/                  # 17 communication channels + Control Center API
│   │   ├── base.py                # Abstract channel interface
│   │   ├── config_routes.py       # REST API for Control Center (20+ endpoints)
│   │   ├── cli.py, api.py         # Core channels
│   │   ├── telegram.py, discord.py # Chat platforms
│   │   ├── whatsapp.py, signal.py # Encrypted messaging
│   │   ├── voice.py               # Voice I/O (STT/TTS)
│   │   └── ...                    # Teams, Matrix, IRC, Twitch, Mattermost, etc.
│   ├── security/
│   │   ├── audit.py               # Append-only audit trail (SHA-256 chain)
│   │   ├── credentials.py         # Credential store (Fernet encrypted)
│   │   ├── sandbox.py             # Multi-level sandbox (L0–L2)
│   │   ├── policies.py            # Security policies (path, command, network)
│   │   └── sanitizer.py           # Input sanitization (injection protection)
│   ├── cron/                      # Cron engine with APScheduler
│   ├── a2a/                       # Agent-to-Agent protocol (Linux Foundation RC v1.0)
│   ├── skills/                    # Skill registry, generator, marketplace
│   ├── graph/                     # Knowledge graph engine
│   ├── telemetry/                 # Cost tracking, metrics, tracing
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
├── tests/                         # 4,746 tests
│   ├── test_core/                 # Planner, Gatekeeper, Executor
│   ├── test_memory/               # All 5 memory tiers
│   ├── test_mcp/                  # MCP tools and client
│   ├── test_channels/             # All channel implementations
│   ├── test_security/             # Audit, sandbox, policies
│   ├── test_integration/          # End-to-end tests
│   ├── test_config_manager.py     # Config manager + API routes
│   └── test_ui_api_integration.py # 55 Control Center API integration tests
├── skills/                        # Built-in skill definitions
├── scripts/                       # Backup, deployment, utilities
├── deploy/                        # Docker, systemd, nginx configs
├── apps/                          # PWA app (legacy)
├── config.yaml.example            # Example configuration
├── pyproject.toml                 # Python project metadata
├── Makefile                       # Build, test, lint commands
├── Dockerfile                     # Container image
├── docker-compose.yml             # Multi-service deployment
└── install.sh                     # Interactive installer
```

## Quick Start

### Prerequisites

- Python >= 3.12
- **LLM Backend** (one of):
  - [Ollama](https://ollama.ai) — local, free, GDPR-compliant (recommended)
  - Any of the 14 cloud providers listed above
- Optional: `playwright` for browser automation, `faster-whisper` for voice

### Installation

```bash
# Clone the repository
git clone https://github.com/Alex8791-cyber/cognithor.git
cd cognithor

# Recommended: Interactive installation (venv, Ollama check, systemd, smoke test)
./install.sh

# Or: Manual installation
pip install -e ".[all,dev]"

# Individual feature groups
pip install -e ".[telegram,voice,web,cron]"

# Control Center UI (optional)
cd ui && npm install
```

The installer offers four modes: `--minimal` (core only), `--full` (all features), `--systemd` (+ service installation), `--uninstall` (removal). Without flags, it starts in interactive mode.

### Load Ollama Models

```bash
ollama pull qwen3:32b           # Planner (20 GB VRAM)
ollama pull qwen3:8b            # Executor (6 GB VRAM)
ollama pull qwen3-coder:32b     # Code tasks (20 GB VRAM)
ollama pull nomic-embed-text    # Embeddings (300 MB VRAM)
```

### Start

```bash
# Interactive CLI
cognithor
# or: python -m jarvis

# Headless mode (API only, for Control Center UI)
python -m jarvis --no-cli

# With custom home directory
JARVIS_HOME=~/my-cognithor cognithor
```

### Control Center UI

The React-based Control Center provides a full web dashboard for managing Cognithor:

```bash
cd ui
npm install
npm run dev    # → http://localhost:5173
```

Click **Power On** to start the backend directly from the UI. The Vite dev server automatically spawns and manages the Python backend process on port 8741 — including orphan detection, clean shutdown, and process lifecycle management. Power Off reliably kills the backend even if it was started externally.

All configuration — agents, prompts, cron jobs, MCP servers, A2A settings — can be edited and saved through the dashboard. Changes persist to YAML files under `~/.jarvis/`.

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

On first start, Cognithor automatically creates the directory structure under `~/.cognithor/`:

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

ollama:
  base_url: "http://localhost:11434"
  timeout_seconds: 120

web:
  # Search providers (all optional, fallback chain: SearXNG → Brave → Google CSE → DDG)
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
```

## Security

Cognithor implements enterprise-grade security:

| Feature | Description |
|---------|-------------|
| **Gatekeeper** | Deterministic policy engine (no LLM). 4 risk levels: GREEN (auto) → YELLOW (inform) → ORANGE (approve) → RED (block) |
| **Sandbox** | 4 isolation levels: Process → Linux Namespaces (nsjail) → Docker → Windows Job Objects |
| **Audit Trail** | Append-only JSONL with SHA-256 chain. Tamper-evident. Credentials masked before logging |
| **Credential Vault** | Fernet-encrypted, per-agent secret storage |
| **Input Sanitization** | Injection protection for shell commands and file paths |
| **EU AI Act** | Compliance module, impact assessments, transparency reports |
| **Red-Teaming** | Automated offensive security tests (1,425 LOC) |

## MCP Tools

| Tool Server | Tools | Description |
|-------------|-------|-------------|
| **Filesystem** | read, write, edit, list, delete | Path-sandboxed file operations |
| **Shell** | exec_command | Sandboxed command execution with timeout |
| **Memory** | search, save, get_entity, add_entity, ... | 10 memory tools across all 5 tiers |
| **Web** | web_search, web_fetch, search_and_read, web_news_search | 4-provider search (SearXNG → Brave → Google CSE → DDG), Jina Reader fallback, domain filtering, cross-check |
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

Current status: **4,746 tests** · **100% pass rate** · **89% coverage**

| Area | Tests | Description |
|------|-------|-------------|
| Core | 239 | Planner, Gatekeeper, Executor, Config, Models, Reflector |
| Memory | 187 | All 5 tiers, indexer, hybrid search, chunker, watcher |
| MCP | 177 | Client, filesystem, shell, memory server, web tools |
| Channels | 163 | CLI, Telegram, Discord, Slack, WhatsApp, API, WebUI, Voice |
| Security | 133 | Audit, credentials, policies, sandbox, sanitizer |
| UI API | 55 | Control Center endpoints (config, agents, prompts, cron, MCP, A2A) |
| Cron | 57 | Engine, job store, scheduling |
| Integration | 46 | End-to-end tests, entrypoint |

## Code Quality

```bash
make lint        # Ruff linting (0 errors)
make format      # Ruff formatting
make typecheck   # MyPy strict type checking
make check       # All combined (lint + typecheck + tests)
make smoke       # Installation validation (26 checks)
make health      # Runtime check (Ollama, disk, memory, audit)
```

## Deployment

### As a systemd service

```bash
./install.sh --systemd
systemctl --user enable --now cognithor
journalctl --user -u cognithor -f    # Logs
```

### With Docker

```bash
docker compose up -d                         # Core backend
docker compose --profile webui up -d         # + Web UI
docker compose --profile full up -d          # Everything
```

### With Control Center UI (Development)

```bash
cd ui
npm install
npm run dev          # Starts Vite on :5173, manages backend on :8741
```

The Vite dev server includes a launcher plugin that automatically spawns, monitors, and stops the Python backend. Use **Power On** / **Power Off** in the dashboard header to control the backend lifecycle.

### Backup

```bash
./scripts/backup.sh                    # Create backup
./scripts/backup.sh --list             # List backups
./scripts/backup.sh --restore latest   # Restore
```

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

## Development Status

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
| **Deploy** | Installer, systemd, Docker, backup, smoke test | Done |

**Metrics:** ~85,000 LOC source · 53,000+ LOC tests · 4,746 tests · 89% coverage · 0 lint errors

## License

Apache 2.0 — see [LICENSE](LICENSE)

Copyright 2026 Alexander Soellner
