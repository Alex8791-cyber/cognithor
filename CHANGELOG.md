# Changelog

All notable changes to Cognithor are documented in this file.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

## [0.26.2] – 2026-03-02

### Added
- **LM Studio Backend** — Full support for LM Studio as a local LLM provider (OpenAI-compatible API on `localhost:1234`). Like Ollama, no API key required, operation mode stays OFFLINE. Includes:
  - `LLMBackendType.LMSTUDIO` enum value and `create_backend()` factory case
  - `lmstudio_api_key` and `lmstudio_base_url` config fields
  - Vision dispatch for OpenAI-compatible image format (`_OPENAI_VISION_BACKENDS` frozenset)
  - Startup banner shows LM Studio URL
  - Specific warning when LM Studio server is unreachable
  - 5 new tests (factory, config, operation mode)

### Changed
- LLM Provider count: 15 → 16 (Ollama, LM Studio, OpenAI, Anthropic, Gemini, Groq, DeepSeek, Mistral, Together, OpenRouter, xAI, Cerebras, GitHub, Bedrock, Hugging Face, Moonshot)
- Vision `format_for_backend()` now uses a `_OPENAI_VISION_BACKENDS` frozenset instead of hardcoded `"openai"` check — all OpenAI-compatible backends (including LM Studio) get proper image support

## [0.26.1] – 2026-03-01

### Added
- **Production Docker Compose** (`docker-compose.prod.yml`) — 5-service stack: Jarvis (headless), WebUI (`create_app` factory), Ollama, optional PostgreSQL (pgvector, `--profile postgres`), optional Nginx reverse proxy (`--profile nginx`). GPU support via nvidia-container-toolkit (commented). Health checks on all services
- **Bare-Metal Installer** (`deploy/install-server.sh`) — One-command bootstrap for Ubuntu 22.04/24.04 + Debian 12. Flags: `--domain`, `--email`, `--no-ollama`, `--no-nginx`, `--self-signed`, `--uninstall`. Installs to `/opt/cognithor/`, data in `/var/lib/cognithor/`, creates `cognithor` user, systemd services, Nginx with TLS, ufw firewall
- **Nginx Reverse Proxy** (`deploy/nginx.conf`) — HTTP→HTTPS redirect, TLS 1.2+1.3, WebSocket upgrade for `/ws/`, prefix-strip `/control/` → jarvis:8741, `/health` passthrough, security headers, 55 MB upload, 5 min read timeout
- **Caddy Config** (`deploy/Caddyfile`) — Auto-TLS alternative via Let's Encrypt with same routing as Nginx
- **`.dockerignore`** — Excludes `.git/`, `tests/`, `node_modules/`, `__pycache__/`, docs, `.env` from Docker builds
- **`create_app()` Factory** (`channels/webui.py`) — ASGI factory for standalone deployment via `uvicorn --factory`. Reads config from env vars (`JARVIS_WEBUI_HOST`, `JARVIS_API_TOKEN`, `JARVIS_WEBUI_CORS_ORIGINS`, TLS). Required by `docker-compose.yml` and systemd service
- **Health Endpoint** (`__main__.py`) — `GET /api/v1/health` on Control Center API (port 8741) returning status, version, and uptime
- **`--api-host` CLI argument** — Bind Control Center API to custom host. Default `127.0.0.1` (unchanged), server mode uses `0.0.0.0`
- **CORS restriction** — When `JARVIS_API_TOKEN` is set, CORS origins are restricted to `JARVIS_API_CORS_ORIGINS` instead of `*`
- **TLS passthrough** — Control Center API passes `ssl_certfile`/`ssl_keyfile` to uvicorn for direct HTTPS

### Fixed
- **`_ssl_cert` UnboundLocalError** — Variables referenced before assignment in `__main__.py` API server block. Moved `_session_store`, `_ssl_cert`, `_ssl_key` definitions before the try block
- **`start_cognithor.bat` crash** — Batch file closed immediately due to: (1) unescaped `|` `)`  `<` in ASCII art echo statements, (2) `::` comments inside `if` blocks (must use `REM`), (3) missing `call` before `npm run dev` (CMD transfers control to `.cmd` without return). All fixed
- **CRLF line endings** — `start_cognithor.bat` had Unix LF line endings; converted to Windows CRLF

### Changed
- `deploy/jarvis.service` — Rewritten for system-level deployment (`/opt/cognithor/venv/bin/jarvis --no-cli --api-host 0.0.0.0`), `User=cognithor`, security hardening, sed instructions for user-level adaptation
- `deploy/jarvis-webui.service` — Rewritten for system-level deployment with `create_app` factory
- `deploy/README.md` — Complete rewrite: Docker Quick Start, Bare-Metal Quick Start, Config Reference, Docker Profiles, TLS (Nginx/Caddy/Direct), Reverse Proxy Endpoints, Monitoring, Troubleshooting, VRAM Profiles
- `.env.example` — Added Server Deployment, WebUI Channel, TLS, and PostgreSQL sections
- `Dockerfile` — Version label updated from `0.1.0` to `0.26.0`

## [0.26.0] – 2026-03-01

### Added
- **Security Hardening** — Comprehensive runtime security improvements across the entire codebase:
  - **SecureTokenStore** (`security/token_store.py`) — Ephemeral Fernet (AES-256) encryption for all channel tokens in memory. Tokens are never stored as plaintext in RAM. Base64 fallback when `cryptography` is not installed
  - **Runtime Token Encryption** — All 9 channel classes (Telegram, Discord, Slack, Teams, WhatsApp, API, WebUI, Matrix, Mattermost) now store tokens encrypted via `SecureTokenStore` with `@property` access for backward compatibility
  - **TLS Support** — Optional SSL/TLS for webhook servers (Teams, WhatsApp) and HTTP servers (API, WebUI). `ssl_certfile`/`ssl_keyfile` config fields in `SecurityConfig`. Minimum TLS 1.2 enforced. Warning logged for non-localhost without TLS
  - **File-Size Limits** — Upload/processing limits on all paths: 50 MB documents (`media.py`), 100 MB audio (`media.py`), 1 MB code execution (`code_tools.py`), 50 MB WebUI uploads (`webui.py`), 50 MB Telegram documents (`telegram.py`)
  - **Session Persistence** — Channel-to-session mappings (`_session_chat_map`, `_user_chat_map`, `_session_users`) stored in SQLite via `SessionStore.channel_mappings` table. Survives restarts — Telegram, Discord, Teams, WhatsApp sessions are restored on startup
- **One-Click Launcher** — `start_cognithor.bat` for Windows: double-click → browser opens → click Power On → Jarvis runs. Desktop shortcut included
- 38 new tests for token store, TLS config, session persistence, file-size limits, document size validation

### Fixed
- Matrix channel constructor mismatch in `__main__.py` (`token=` → `access_token=`)
- Teams channel constructor in `__main__.py` now uses correct parameter names (`app_id`, `app_password`)

### Changed
- `SessionStore` gains `channel_mappings` table with idempotent migration, CRUD methods, and cleanup
- `SecurityConfig` gains `ssl_certfile` and `ssl_keyfile` fields
- Version bumped to 0.26.0
- Test count: 4,841 → 4,879

## [0.25.0] – 2026-03-01

### Added
- **Adaptive Context Pipeline** — Automatic pre-Planner context enrichment from Memory (BM25), Vault (full-text search), and Episodes (recent days). Injects relevant knowledge into WorkingMemory before the Planner runs, so Jarvis no longer "forgets" between sessions.
- **ContextPipelineConfig** — New configuration model with `enabled`, `memory_top_k`, `vault_top_k`, `episode_days`, `min_query_length`, `max_context_chars`, `smalltalk_patterns`
- Smalltalk detection to skip unnecessary context searches for greetings and short messages
- `vault_tools` exposed in tools.py PhaseResult for dependency injection into Context Pipeline

### Changed
- Gateway initializes Context Pipeline after tools phase and calls `enrich()` before PGE loop
- Architecture diagram updated with Context Pipeline layer
- Version bumped to 0.25.0

## [0.24.0] – 2026-03-01

### Added
- **Knowledge Synthesis** — Meta-analysis engine that orchestrates Memory, Vault, Web and LLM to build coherent understanding. 4 new MCP tools:
  - `knowledge_synthesize` — Full synthesis with confidence-rated findings (★★★), source comparison, contradiction detection, timeline, gap analysis
  - `knowledge_contradictions` — Compares stored knowledge (Memory + Vault) with current web information, identifies outdated facts and discrepancies
  - `knowledge_timeline` — Builds chronological timelines with causal chains (X → Y → Z) and trend analysis
  - `knowledge_gaps` — Completeness scoring (1–10), prioritized research suggestions with concrete search terms
- **Wissens-Synthese Skill** — New procedure (`data/procedures/wissens-synthese.md`) for guided knowledge synthesis workflow
- 3 depth levels: `quick` (Memory + Vault only), `standard` (+ 3 web results), `deep` (+ 5 web results, detailed analysis)
- Synthesis results can be saved directly to Knowledge Vault (`save_to_vault: true`)

### Changed
- tools.py captures return values from `register_web_tools` and `register_memory_tools` for dependency injection into synthesizer
- tools.py registers synthesis tools and wires LLM, Memory, Vault, and Web dependencies
- MCP Tool Layer expanded from 15+ to 18+ tools
- Version bumped to 0.24.0

## [0.23.0] – 2026-03-01

### Added
- **Knowledge Vault** — Obsidian-compatible Markdown vault (`~/.jarvis/vault/`) with YAML frontmatter, tags, `[[backlinks]]`, and full-text search. 6 new MCP tools: `vault_save`, `vault_search`, `vault_list`, `vault_read`, `vault_update`, `vault_link`
- **Document Analysis Pipeline** — LLM-powered structured analysis of PDF/DOCX/TXT/HTML documents via `analyze_document` tool. Analysis modes: full (6 sections), summary, risks, todos. Optional vault storage
- **Google Custom Search Engine** — 3rd search provider in the fallback chain (SearXNG → Brave → **Google CSE** → DuckDuckGo). Config: `google_cse_api_key`, `google_cse_cx`
- **Jina AI Reader Fallback** — Automatic fallback for JS-heavy sites where trafilatura extracts <200 chars. New `reader_mode` parameter (`auto`/`trafilatura`/`jina`) on `web_fetch`
- **Domain Filtering** — `domain_blocklist` and `domain_allowlist` in WebConfig for controlled web access
- **Source Cross-Check** — `cross_check` parameter on `search_and_read` appends a source comparison section
- **Dokument-Analyse Skill** — New procedure (`data/procedures/dokument-analyse.md`) for structured document analysis workflow
- **VaultConfig** — New Pydantic config model with `enabled`, `path`, `auto_save_research`, `default_folders`

### Changed
- Web search fallback chain now includes 4 providers (was 3)
- `web_fetch` uses auto reader mode with Jina fallback by default
- `search_and_read` supports optional source comparison
- MediaPipeline supports LLM and Vault injection for document analysis
- tools.py registers vault tools and wires LLM/vault into media pipeline
- Detailed German error messages when all search providers fail (instead of empty results)

## [0.22.0] – 2026-02-28

### Added
- **Control Center UI** — React 19 + Vite 7 dashboard integrated into repository (`ui/`)
- **Backend Launcher Plugin** — Vite plugin manages Python backend lifecycle (start/stop/orphan detection)
- **20+ REST API Endpoints** — Config CRUD, agents, bindings, prompts, cron jobs, MCP servers, A2A settings
- **55 UI API Integration Tests** — Full round-trip testing for every Control Center endpoint
- **Prompts Fallback** — Empty prompt files fall back to built-in Python constants
- **Health Endpoint** — `GET /api/v1/health` for backend liveness checks

### Fixed
- Agents GET returned hardcoded path instead of config's `jarvis_home`
- Bindings GET created ephemeral in-memory instances (always empty)
- MCP servers response format mismatch between backend and UI
- FastAPI route ordering: `/config/presets` captured by `/config/{section}`
- Prompts returned empty strings when 0-byte files existed on disk
- `policyYaml` round-trip stripped trailing whitespace

## [0.21.0] – 2026-02-27

### Added
- **Channel Auto-Detection** — Channels activate automatically when tokens are present in `.env`
- Removed manual `telegram_enabled`, `discord_enabled` etc. config flags
- All 10 channel types (Telegram, Discord, Slack, WhatsApp, Signal, Matrix, Teams, iMessage, IRC, Twitch) use token-based auto-detect

### Fixed
- Telegram not receiving messages when started via Control Center UI
- Config flag `telegram_enabled: false` blocked channel registration even when token was set

## [0.20.0] – 2026-02-26

### Added
- **15 LLM Providers** — Moonshot/Kimi, Cerebras, GitHub Models, AWS Bedrock, Hugging Face added
- **Cross-request context** — Vision results and tool outputs persist across conversation turns
- **Autonomous code toolkit** — `run_python` and `analyze_code` MCP tools
- **Document export** — PDF, DOCX generation from Markdown
- **Dual vision model** — Orchestration between primary and fallback vision models
- **Web search overhaul** — DuckDuckGo fallback, presearch bypass, datetime awareness

### Fixed
- JSON parse failures in planner responses
- Cross-request context loss for vision and tool results
- Telegram photo analysis path and intent forwarding
- Whisper voice transcription CPU mode enforcement
- Telegram approval deadlock for web tool classifications

## [0.10.0] – 2026-02-24

### Added
- **17 Communication Channels** — Discord, Slack, WhatsApp, Signal, iMessage, Teams, Matrix, Google Chat, Mattermost, Feishu/Lark, IRC, Twitch, Voice (STT/TTS) added to existing CLI, Web UI, REST API, Telegram
- **Agent-to-Agent Protocol (A2A)** — Linux Foundation RC v1.0 implementation
- **MCP Server Mode** — Jarvis as MCP server (stdio + HTTP)
- **Browser Automation** — Playwright-based tools (navigate, screenshot, click, fill, execute JS)
- **Media Pipeline** — STT (Whisper), TTS (Piper/ElevenLabs), image analysis, PDF extraction
- **Enterprise Security** — EU AI Act compliance module, red-teaming suite (1,425 LOC)
- **Cost Tracking** — Per-request cost estimation, daily/monthly budgets

## [0.5.0] – 2026-02-23

### Added
- **Multi-LLM Backend** — OpenAI, Anthropic, Gemini, Groq, DeepSeek, Mistral, Together, OpenRouter, xAI support
- **Model Router** — Automatic model selection by task type (planning, execution, coding, embedding)
- **Cron Engine** — APScheduler-based recurring tasks with YAML configuration
- **Procedural Learning** — Reflector auto-synthesizes reusable skills from successful sessions
- **Knowledge Graph** — Entity-relation graph with traversal queries
- **Skill Marketplace** — Skill registry, generator, import/export

## [0.1.0] – 2026-02-22

### Added

**Core Architecture**
- PGE Trinity: Planner → Gatekeeper → Executor agent loop
- Multi-model router (Planner/Executor/Coder routing)
- Reflector for post-execution analysis and learning loops
- Gateway as central message bus with session management

**5-Tier Cognitive Memory**
- Core Memory (CORE.md): Identity, rules, personality
- Episodic Memory: Daily logs with append-only writing
- Semantic Memory: Knowledge graph with entities + relations
- Procedural Memory: Learned workflows with trigger matching
- Working Memory: Session context with auto-compaction
- Hybrid search: BM25 + vector embeddings + graph queries
- Markdown-aware sliding window chunker

**Security**
- Gatekeeper with 4-level risk classification (GREEN/YELLOW/ORANGE/RED)
- 6 built-in security policies
- Input sanitizer against prompt injection
- Credential store with Fernet encryption (AES-256)
- Audit trail with SHA-256 hash chain
- Filesystem sandbox with path whitelist

**MCP Tools**
- Filesystem: read_file, write_file, edit_file, list_directory
- Shell: exec_command (with Gatekeeper protection)
- Web: web_search, web_fetch, search_and_read
- Memory: memory_search, memory_write, entity_create

**Channels**
- CLI channel with Rich terminal UI
- API channel (FastAPI REST)
- WebUI channel with WebSocket support
- Telegram bot channel
- Voice channel (Whisper STT + Piper TTS)

**Deployment**
- Interactive installer (`install.sh`)
- Systemd services (user-level)
- Docker + Docker Compose
- Smoke test and health check scripts
- Backup/restore with rotation management

**Quality**
- 1,060 automated tests
- Structured logging with structlog + Rich
- Python 3.12+, Pydantic v2, SQLite + sqlite-vec
- 100% local — no cloud dependencies required
