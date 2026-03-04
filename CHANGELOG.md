# Changelog

All notable changes to Cognithor are documented in this file.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

## [1.1.0] – 2026-03-05

### Agent Infrastructure Release — 15 New Subsystems

Major infrastructure release adding 15 new subsystems with 1,000+ new tests,
bringing Cognithor from a production-ready agent to a full enterprise Agent OS
with distributed workers, policy governance, benchmarking, and GDPR compliance.

### Added

**DAG Workflow Engine** (`core/dag_engine.py`)
- Directed Acyclic Graph workflow execution with topological sort
- 4 node states (PENDING, RUNNING, COMPLETED, FAILED), parallel branch execution
- Cycle detection, conditional edges, automatic retry on failure
- 53 tests

**Execution Graph UI** (`core/execution_graph.py`)
- Real-time visualization data for agent execution flows
- Node types: PLAN, GATE, TOOL, REPLAN, RESPONSE, ERROR
- Mermaid diagram export for documentation
- 37 tests

**Agent Delegation Engine** (`core/delegation.py`)
- Typed contracts for inter-agent task delegation
- DelegationContract with SLA guarantees, timeout, priority
- Result aggregation and escalation on failure
- 44 tests

**Policy-as-Code Governance** (`security/policy_store.py`)
- Versioned policy store with YAML persistence
- Policy simulation (dry-run), rollback to any version
- Diff between versions, audit trail for all changes
- 41 tests

**Knowledge Graph Layer** (`graph/`)
- Named Entity Recognition with configurable extractors
- Entity deduplication with fuzzy matching
- Graph serialization for UI visualization
- 46 tests

**Memory Consolidation Pipeline** (`memory/consolidation.py`)
- Importance scoring for memory entries (recency, frequency, emotional weight)
- Automatic deduplication with similarity detection
- Configurable retention thresholds and decay functions
- 48 tests

**Multi-Agent Collaboration** (`core/collaboration.py`)
- Debate pattern: agents argue positions, moderator synthesizes
- Voting pattern: democratic decision-making with quorum
- Pipeline pattern: sequential agent processing with context passing
- 52 tests

**Agent SDK** (`core/agent_sdk.py`)
- Decorator-based agent registration (`@agent`, `@tool`, `@hook`)
- Agent registry with capability discovery
- Project scaffolding (generate agent boilerplate)
- 38 tests

**Plugin Marketplace Remote Registry** (`skills/remote_registry.py`)
- Remote plugin manifests with SHA-256 checksums and signatures
- Dependency resolver with topological sort and circular detection
- Install, update, rollback, uninstall with local JSON persistence
- 36 tests

**Tool Sandbox Hardening** (`security/resource_limits.py`)
- Per-tool sandbox profiles with CPU, memory, disk, network limits
- ResourceWatchdog for runtime monitoring (timeout, output size)
- NetworkGuard with per-tool allow/block/restrict rules
- EscapeDetector with 8 attack categories (path traversal, command injection, etc.)
- 93 tests

**Distributed Worker Runtime** (`core/worker.py`)
- WorkerNode with capability-based job assignment
- 4 routing strategies: round-robin, least-loaded, capability-based, random
- HealthMonitor with heartbeat-based failure detection
- FailoverManager with automatic re-queue and dead-letter handling
- 64 tests

**Deterministic Replay System** (`telemetry/recorder.py`, `telemetry/replay.py`)
- ExecutionRecorder capturing full agent runs (13 event types)
- ReplayEngine with override support for what-if analysis
- Replay from specific iterations, diff computation, match rate metrics
- JSONL export/import
- 55 tests

**Agent Benchmark Suite** (`benchmark/suite.py`)
- 14 builtin tasks across 7 categories (research, automation, knowledge, policy, collaboration, reasoning, tool_use)
- BenchmarkScorer with composite scoring (keywords 40%, tools 30%, efficiency 15%, latency 15%)
- BenchmarkRunner with category/difficulty/tag filtering
- BenchmarkReport (JSON + Markdown), RegressionDetector
- 48 tests

**Installer Modernization** (`utils/installer.py`)
- uv auto-detection with transparent pip fallback (10x faster installs)
- `install.sh --use-uv` flag with auto-install of uv
- `bootstrap_windows.py` auto-detects and prefers uv
- Command abstraction for both pip and uv backends
- 36 tests

**GDPR Compliance Toolkit** (`security/gdpr.py`)
- Data Processing Log per Art. 30 (who, what, when, why, legal basis)
- Model Usage Records (LLM invocation tracking with PII flags)
- 6 default retention policies with configurable enforcement
- Right-to-Erasure (Art. 17) with pluggable external handlers
- Data Subject Access Reports (Art. 15)
- Audit export as JSON + Markdown
- GDPRComplianceManager orchestrating all functions
- 49 tests

### Changed
- Version: 1.0.0 → **1.1.0**
- Test count: 8,411 → **9,251** (+840 new tests across 15 new test files)
- `install.sh`: Added `--use-uv` flag, `detect_installer()` function, uv-aware venv creation
- `bootstrap_windows.py`: Added `_detect_python_installer()` with uv fast-path
- LOC source: ~98,000 → ~106,000
- LOC tests: ~80,000 → ~90,000

---

## [1.0.0] – 2026-03-04

### 🎉 Major Release — Chat, Voice Mode, TTS

Cognithor graduates from beta to **v1.0.0**. This release adds a fully integrated chat experience
with voice conversation mode, German TTS, and natural language responses — completing the
transition from a CLI-first tool to a full Agent OS.

### Added

**Chat-Seite im Control Center**
- **ChatPage** (`ui/src/pages/ChatPage.jsx`) — Full chat integration in the React UI with WebSocket streaming, auto-scroll, session management
- **MessageList** (`ui/src/components/chat/MessageList.jsx`) — Threaded message display with Markdown rendering, timestamps, sender avatars
- **ChatInput** (`ui/src/components/chat/ChatInput.jsx`) — Rich input bar with send button, Enter-to-submit, multiline support
- **ChatCanvas** (`ui/src/components/chat/ChatCanvas.jsx`) — Side panel for canvas artifacts (code, tables, diagrams)
- **ToolIndicator** (`ui/src/components/chat/ToolIndicator.jsx`) — Real-time tool execution indicators during agent processing
- **ApprovalBanner** (`ui/src/components/chat/ApprovalBanner.jsx`) — Inline approval/deny banner for ORANGE-risk actions
- **useJarvisChat** (`ui/src/hooks/useJarvisChat.js`) — React hook for WebSocket connection, message state, streaming
- Chat is now the default start page in the Control Center

**Voice Mode**
- **VoiceIndicator** (`ui/src/components/chat/VoiceIndicator.jsx`) — Visual feedback for listening/speaking/processing states
- **useVoiceMode** (`ui/src/hooks/useVoiceMode.js`) — React hook for wake word detection, STT, TTS playback
- **Wake Word** — "Jarvis" detection with Levenshtein distance + phonetic normalization for robust matching despite Chrome STT errors
- **Konversationsmodus** — Continuous listening after wake word until "Jarvis Ende" dismissal
- **TTS-Wiedergabe** — Automatic text-to-speech playback of agent responses in voice mode

**Piper TTS (Thorsten Emotional)**
- German speech synthesis with `de_DE-thorsten_emotional-medium` voice model
- Multi-speaker support for future voice variants
- Automatic model download on first use (~80 MB)

**Natürliche Sprache**
- System prompt tuned for spoken, human responses (no bullet points, flowing sentences)
- Adapted `formulate_response()` in Planner for voice-optimized output

### Changed
- 16 existing files modified (planner.py, config.py, __main__.py, CognithorControlCenter.jsx, vite.config.js, gateway.py, executor.py, telegram.py, indexer.py, memory manager, search, ttl_dict, bootstrap_windows.py, icons.jsx, test_search_coverage.py, pyproject.toml)
- ~776 lines added, ~118 removed
- Version: 0.26.6 → **1.0.0**
- Development Status: Beta → **Production/Stable**

---

## [0.26.5] – 2026-03-03

### Added — Human Feel

**Personality & Sentiment**
- **Personality Engine** (`core/personality.py`) — Configurable personality injection into SYSTEM_PROMPT. Time-of-day greetings (Morgen/Nachmittag/Abend/Nacht), warmth/humor scaling, follow-up questions, success celebration. `PersonalityConfig` with `warmth`, `humor`, `greeting_enabled`, `follow_up_questions`, `success_celebration`. 13 tests
- **Sentiment Detection** (`core/sentiment.py`) — Lightweight keyword/regex-based sentiment detection for German text. 5 categories: FRUSTRATED, URGENT, CONFUSED, POSITIVE, NEUTRAL. Confidence scoring, priority-ordered pattern matching. Automatic system-message injection to adapt response style. No ML dependencies. 40 tests
- **User Preference Store** (`core/user_preferences.py`) — SQLite-backed per-user preference persistence. Auto-learned verbosity (terse/normal/verbose) from message length via exponential moving average. Fields: `greeting_name`, `formality`, `verbosity`, `avg_message_length`, `interaction_count`. Verbosity hint injection into working memory. 16 tests
- **User-Friendly Error Messages** (`utils/error_messages.py`) — German error message templates replacing raw exceptions across all channels. `classify_error_for_user(exc)` maps Timeout/Connection/Permission/RateLimit/Memory to empathetic messages. `gatekeeper_block_message()` explains why actions were blocked with suggestions. `retry_exhausted_message()` with tool-specific context. `all_actions_blocked_message()` with per-action reasons. `_friendly_tool_name()` mapping for 22+ tools. 18 tests

**Status Callback System**
- **StatusType Enum** (`channels/base.py`) — 6 status types: THINKING, SEARCHING, EXECUTING, RETRYING, PROCESSING, FINISHING. Default no-op `send_status()` on base Channel class
- **Gateway Status Callbacks** (`gateway/gateway.py`) — Fire-and-forget status callbacks with 2s timeout in PGE loop. Tool-specific status messages via `_TOOL_STATUS_MAP` (22 mappings). "Denke nach..." before planner, tool-specific before executor, "Formuliere Antwort..." before response
- **Executor Retry Visibility** (`core/executor.py`) — "Versuch 2 von 3..." status callbacks during retry loop
- **CLI send_status()** — Rich-formatted italic status messages
- **Telegram send_status()** — `send_chat_action(typing)` indicator
- **Discord send_status()** — `channel.typing()` context manager
- **WebUI send_status()** — WebSocket `STATUS_UPDATE` event with status type and text
- 6 tests for status callback system

### Fixed
- **test_voice VAD fallback** — `test_load_fallback` no longer hardcodes `assert not vad._use_silero`. Now environment-agnostic: accepts both Silero and energy-based fallback. Added separate `test_load_fallback_without_torch` with mocked torch for deterministic fallback testing
- **Executor retry messages** — Now uses `retry_exhausted_message()` instead of raw error strings
- **Channel error handling** — CLI and Telegram now show `classify_error_for_user()` messages instead of raw `f"Fehler: {exc}"`
- **Gateway all-blocked message** — Replaced generic "Alle geplanten Aktionen wurden vom Gatekeeper blockiert" with per-action `all_actions_blocked_message()`

### Changed
- `PersonalityConfig` added to `JarvisConfig` with sensible defaults (warmth=0.7, humor=0.3)
- `Planner.__init__()` accepts optional `personality_engine` parameter
- SYSTEM_PROMPT template gains `{personality_section}` placeholder
- `gateway/phases/pge.py` wires `PersonalityEngine` and `UserPreferenceStore` into initialization
- `gateway/gateway.py` integrates sentiment detection, user preferences, and status callbacks into `handle_message()` and `_run_pge_loop()`
- Test count: 8,306 → 8,411 (+105 new tests across 5 new test files)
- LOC source: ~97,000 → ~98,000
- LOC tests: ~79,000 → ~80,000

## [0.26.4] – 2026-03-02

### Added — Coverage & Skills Infrastructure

**Skills Infrastructure**
- **BaseSkill Abstract Class** (`skills/base.py`) — Abstract base class for all Jarvis skills with `NAME`, `DESCRIPTION`, `VERSION`, `CRON`, `REQUIRES_NETWORK`, `API_BASE` class attributes and abstract `execute()` method. Properties: `name`, `description`, `version`, `is_automated`, `is_network_skill`, `validate_params()`. Exported from `jarvis.skills` package
- **Skill `__init__.py` Files** — Added package init files to all 5 skill directories (test, test_skill, backup, gmail_sync, wetter_abfrage) enabling correct relative imports
- **Fixed `wetter_abfrage` Manifest** — Added missing `network` permission and `weather`/`api` tags

**Test Coverage Deep Push (+255 tests, 8,051 → 8,306)**
- **Planner Tests** (7 → 32) — LLM error handling, native tool_calls parsing, replan with multiple/error results, formulate_response with search vs. non-search results, core_memory injection, OllamaError fallbacks, cost tracking (with/without tracker, exception handling), prompt loading from .md/.txt files, JSON sanitization, _try_parse_json 4-strategy fallback, _format_results truncation
- **LLM Backend Tests** (24 → 63) — OllamaBackend: chat, tool_calls, HTTP errors, timeouts, embed, is_available, list_models, close. GeminiBackend: chat, functionCall, HTTP errors, embed, is_available, list_models, multi-part content. AnthropicBackend: tool_use blocks, HTTP errors, is_available, close. Factory: mistral, together, openrouter, xai, cerebras
- **Executor Tests** (10 → 25) — Retry/backoff with retryable errors (ConnectionError, TimeoutError), non-retryable errors (ValueError), all retries exhausted, output truncation, MASK/INFORM gate status, no MCP client, RuntimeMonitor security block, audit logger success/failure, gap detector (unknown tool, repeated failure), workspace injection
- **Reflector Tests** (14 → 27) — apply() with session summary (episodic), extracted facts (semantic), procedure candidate (procedural), all types combined, memory manager errors. _write_semantic with entities, relations, injection sanitization. reflect() with episodic_store, causal_analyzer. _extract_json with markdown fences, raw JSON, no JSON
- **Shell Tests** (9 → 19) — Timeout behavior, truncated output, stderr handling, successful execution, sandbox overrides, multiple path traversals, safe file commands, different sandbox levels

**Coverage Consolidation**
- Removed 6 trivial tests (pure `is not None`/`hasattr` checks) from `test_secondary_coverage.py`
- Cleaned unused imports across `test_final_coverage.py`, `test_deep_coverage.py`, `test_secondary_coverage.py`

### Changed
- Test count: 8,051 → 8,306 (+255 new tests)
- LOC tests: ~77,000 → ~79,000
- Coverage estimate: 87% → 89%
- `skills/__init__.py` now exports `BaseSkill` and `SkillError`

## [0.26.3] – 2026-03-02

### Added — Scaling & Quality

**Scaling (Skalierung)**
- **Distributed Locking** (`core/distributed_lock.py`) — Abstract `DistributedLock` interface with 3 backends: `LocalLockBackend` (asyncio.Lock), `FileLockBackend` (cross-process file locking with msvcrt/fcntl), `RedisLockBackend` (SET NX EX + Lua release). Automatic fallback from Redis → File when redis package unavailable. `create_lock(config)` factory, `lock_backend` and `redis_url` config fields. 39 tests
- **Durable Message Queue** (`core/message_queue.py`) — SQLite-backed async message queue with priority levels (LOW/NORMAL/HIGH/CRITICAL), FIFO within priority, retry with exponential backoff, dead-letter queue (DLQ), configurable TTL and max size. `QueueConfig` with `enabled`, `max_size`, `ttl_hours`, `max_retries`. Gateway integration (Phase D.2). 34 tests
- **Telegram Webhook Support** (`channels/telegram.py`) — Webhook mode for <100ms latency alongside existing polling. aiohttp server with `/telegram/webhook` and `/telegram/health` endpoints. Optional TLS. Config fields: `telegram_use_webhook`, `telegram_webhook_url`, `telegram_webhook_port`, `telegram_webhook_host`. Automatic fallback to polling when webhook URL empty. 16 tests
- **Prometheus Metrics** (`telemetry/prometheus.py`) — Zero-dependency Prometheus text exposition format exporter. Exports counters, gauges, histograms from MetricsProvider + MetricCollector. `GET /metrics` endpoint on Control Center API. 10 standard metrics (requests_total, request_duration_ms, errors_total, tokens_used_total, active_sessions, queue_depth, tool_calls_total, tool_duration_ms, memory_usage_bytes, uptime_seconds). Gateway PGE loop instrumentation. 49 tests
- **Grafana Dashboard** (`deploy/grafana-dashboard.json`) — 14-panel dashboard (3 rows: Overview, System Health, Tool Execution) with channel/model template variables, 30s auto-refresh
- **Skill Marketplace Persistence** (`skills/persistence.py`) — SQLite-backed store for marketplace listings, reviews, reputation, install history. 6 tables with indexes. CRUD, search (fulltext + category + rating + sort), featured/trending, reputation scoring. REST API (`skills/api.py`) with 12 endpoints under `/api/v1/skills`. Seed data from built-in procedures. `MarketplaceConfig` with `enabled`, `db_path`, `auto_seed`, `require_signatures`. 71 tests
- **Auto-Dependency Loading** (`core/startup_check.py`) — Comprehensive startup checker that auto-installs missing Python packages, auto-starts Ollama, auto-pulls missing LLM models, verifies directory structure. Integrated into `__main__.py` for seamless startup experience

**Quality**
- **Magic Numbers → Config** — 30+ hardcoded constants extracted to typed Pydantic config classes (`BrowserConfig`, `FilesystemConfig`, `ShellConfig`, `MediaConfig`, `SynthesisConfig`, `CodeConfig`, `ExecutorConfig`, extended `WebConfig`). Safe config access with `getattr()` fallback pattern
- **Parametrized Channel Tests** — 122 cross-channel tests covering all 11 channel types with consistent interface validation
- **Windows Path Handling** — 34 new tests, `tempfile.gettempdir()` instead of hardcoded `/tmp/jarvis/`
- **Vault Frontmatter → PyYAML** — Replaced 4 regex-based frontmatter methods with `yaml.safe_load()` for Obsidian-compatible parsing. 47 vault tests
- **Token Estimation** — Language-aware token counting using `_estimate_tokens()` from chunker (word-based + German compound correction) instead of naive `len/4`. Configurable budget allocation via `MemoryConfig`. Auto-compaction in Gateway PGE loop. 8 new tests

### Changed
- Test count: 4,879 → 5,304+ (425+ new tests across all scaling and quality features)
- LOC tests: ~53,000 → ~56,000+
- Version `JarvisConfig.version` fixed: 0.25.0 → 0.26.0

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
