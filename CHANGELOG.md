# Changelog

All notable changes to Cognithor are documented in this file.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

## [0.27.3-beta] – 2026-03-07

### Security Fix & Installer Bug-Fixes

Closes a high-severity Path Traversal vulnerability (CWE-22) in the TTS/Voice API and fixes three installer bugs reported by QA.

### Fixed

- **CWE-22 Path Traversal in TTS API** — Malicious `voice` parameter in `POST /api/v1/tts` could escape the voices directory via `../../../../etc/passwd`. Added `validate_voice_name()` whitelist (regex + null-byte + length check) and `validate_model_path_containment()` defense-in-depth across all 4 TTS entry points (`__main__.py`, `mcp/media.py`, `voice_ws_bridge.py`)
- **Multi-GPU detection crash** (install.sh) — `nvidia-smi` on multi-GPU systems (e.g. 2x Tesla M40) returned multi-line output causing `bash: [[: 12288\n0: syntax error`. Now parses all lines individually and sums VRAM across GPUs
- **`--init-only` hangs indefinitely** — `StartupChecker.check_and_fix_all()` ran before the `--init-only` exit, attempting model pulls (30min timeout) and pip installs (5min timeout). Moved `--init-only` exit before StartupChecker. Added `timeout 30` safety net in install.sh

### Added

- `validate_voice_name()` in `security/sanitizer.py` — Central voice/model name validation against path traversal
- `validate_model_path_containment()` — Defense-in-depth path containment check (resolve + relative_to)
- AMD GPU detection via `rocm-smi` in install.sh
- Node.js missing: distro-specific installation instructions (Ubuntu, Fedora, Arch)
- 96 new security tests in `test_voice_path_traversal.py`

## [0.27.1] – 2026-03-07

### Community Skill Marketplace & Autonomy Hardening

Introduces the Community Skill Marketplace with full trust chain, plus 13 autonomy fixes across the PGE loop.

### Added

- **Community Skill Marketplace** — Install, search, rate, and report community skills from a GitHub-hosted registry. Publisher verification with trust levels (unknown/community/verified/official). 5-check validation pipeline (syntax, injection, tools, safety, hash)
- **ToolEnforcer** — Runtime tool-allowlist enforcement for community skills. Skills can only invoke tools declared in `tools_required`
- **SkillValidator** — 5-stage validation: syntax check, prompt-injection scan, tool whitelist, safety audit, SHA-256 hash verification
- **CommunityRegistryClient** — Async client for fetching, verifying, and installing skills from remote registries with aiohttp + urllib fallback
- **RegistrySync** — Periodic background sync with recall checks for deactivated/recalled skills
- **PublisherVerifier** — Publisher identity verification with 4 trust levels and GPG signature support
- **Community REST API** — 5 endpoints: search, detail, install, report, publisher info (`/api/v1/skills/community/`)
- **3 New MCP Tools** — `install_community_skill`, `search_community_skills`, `report_skill` (total: 5 in skill_tools.py)
- **Thread-safe Caches** — `asyncio.Lock` protection on all community module caches (client, sync, publisher)
- **aiohttp Fallback** — All HTTP clients gracefully fall back to `urllib` if aiohttp raises RuntimeError

### Fixed

- **Presearch Skip Patterns** — Removed trailing `\b` so "Erstelle/erstellst/erstellen" are recognized as action verbs (no longer misrouted to web search)
- **subprocess Differentiation** — `subprocess.run()` and `subprocess.check_output()` now ALLOWED in run_python; `subprocess.Popen/call/getoutput/getstatusoutput` remain BLOCKED
- **Socket Pattern Narrowing** — Changed from blanket `socket.` block to specific `socket.socket()` and `socket.create_connection()`
- **Project Dir in allowed_paths** — `allow_project_dir: true` (default) auto-adds project root so Cognithor can write to its own codebase
- **Multi-step Plan Early Exit** — PGE loop no longer breaks on first success for multi-step plans; coding tools always continue iteration
- **Failure Threshold** — Smart exit at `max(5, max_iterations // 2)` consecutive failures instead of immediate abort
- **Presearch Truncation** — Increased from 4000 to 8000 chars for better fact-question coverage
- **Planner Circuit Breaker** — Tuned: `failure_threshold=3->5`, `recovery_timeout=30->15s`, `half_open_max_calls=1->2`
- **Replan Retry** — 2 attempts with 1s pause before giving up on replan LLM calls
- **formulate_response Fallback** — On LLM failure, returns raw tool results instead of empty error
- **JSON Confidence** — Lowered from 0.8 to 0.5 for direct answers without JSON parsing
- **try-finally Cleanup** — Skill state (ToolEnforcer, active_skill) cleaned up via single `_cleanup_skill_state()` in finally block
- **Evidence Field Wiring** — Community skill reports now properly pass evidence to persistence layer
- **API Error Handling** — All community endpoints wrapped in try-except with proper HTTPException and logging

### Changed

- `response_token_budget`: 3000 -> 4000
- `memory_top_k`: 4 -> 8
- `vault_top_k`: 3 -> 5
- `max_context_chars`: 3000 -> 8000
- `compaction_keep_last_n`: 4 -> 8
- `budget_injected_memories`: 1500 -> 2500
- Context pipeline failures now log as WARNING (was DEBUG)
- Presearch failures now log as WARNING (was DEBUG)
- Community skill exports: 13 public classes from `skills.community` package

---

## [0.27.0] – 2026-03-07

### Installer & UX Overhaul

Non-technical user capability upgraded from 5-6/10 to 10/10. Full project audit with 80+ findings, critical fixes applied.

### Added

- **Python Auto-Install (Windows)** — `start_cognithor.bat` detects missing Python and offers winget install with PATH refresh
- **Ollama Auto-Install (Windows)** — `bootstrap_windows.py` offers `winget install Ollama.Ollama` during first boot
- **Ollama Auto-Install (Linux)** — `install.sh` offers `curl -fsSL https://ollama.com/install.sh | sh`
- **Distro-specific Python Hints (Linux)** — Ubuntu deadsnakes PPA, Fedora dnf, Arch pacman, openSUSE zypper, Debian pyenv
- **Locale-based Language Detection** — Auto-sets `language: "de"` or `"en"` in config.yaml based on system locale
- **Hardware Tier Display** — Shows VRAM, RAM, tier (minimal/standard/power/enterprise), and model recommendations before pull
- **LLM Smoke Test** — Post-install HTTP test to verify LLM responds ("Sage kurz Hallo.")
- **Linux .desktop Files** — `cognithor.desktop` (CLI) and `cognithor-webui.desktop` in `~/.local/share/applications/`
- **Pre-built UI Support** — Node.js no longer required if `ui/dist/` exists; FastAPI `StaticFiles` mount at "/"
- **GitHub Beta Release Workflow** — `.github/workflows/beta-release.yml` with lint, test, changelog generation, GitHub pre-release

### Fixed

- **XSS in MessageList.jsx** — Added `escapeHtml()` before `dangerouslySetInnerHTML` (CRITICAL)
- **CORS + Credentials** — `allow_credentials` now only `true` when origins are explicitly restricted (was always true with `*`)
- **Python Version Check Bug** — `deploy/install-server.sh` checked `(3, 11)` instead of `(3, 12)`
- **Unicode Crash (Windows)** — `first_boot.py` replaced Unicode symbols with ASCII-safe `[OK]`/`[FEHLER]`/`[WARNUNG]`
- **Missing curl Timeouts** — All `curl` calls in `install.sh` now have `--max-time` (3s checks, 30s uv, 60s Ollama)
- **Version Consistency** — Synced 0.27.0 across pyproject.toml, __init__.py, config.py, Dockerfile, demo.py, bootstrap_windows.py, test_config.py

### Changed

- `.env.example` expanded from 30 to 100+ variables (all channels, search providers, models, personality)
- `CONTRIBUTING.md` updated with beta branch strategy and conventional commits
- CI workflow now triggers on `beta` branch
- `start_cognithor.bat` supports 3-tier UI launch: Vite Dev → Pre-built UI → CLI fallback
- `bootstrap_windows.py` steps renumbered 13 → 14 (new smoke test step)

---

## [0.26.7] – 2026-03-07

### Wiring & Hardening

Closes 7 wiring gaps identified by capability-matrix analysis. Full suite at 9,596 tests (0 failures).

### Added

- **DAG-based Parallel Executor** — `execute()` now builds a `PlanGraph` from actions and runs independent tool calls concurrently in waves via `asyncio.gather()` + `asyncio.Semaphore`. Replaces sequential `for i, ...` loop. Backwards-compatible for linear dependencies
- **http_request Tool** (`mcp/web.py`) — Full HTTP method support (GET/POST/PUT/PATCH/DELETE/HEAD/OPTIONS) with SSRF protection (`_is_private_ip()`), body-size limit (1 MB), timeout clamping, and domain validation. Classified as ORANGE in Gatekeeper
- **Workflow Adapter** (`core/workflow_adapter.py`) — Bridge function `action_plan_to_workflow()` converts `ActionPlan` to `WorkflowDefinition`, making DAG WorkflowEngine usable from Gateway via `execute_action_plan_as_workflow()`
- **Sub-Agent Depth Guard** — `max_sub_agent_depth` field in `SecurityConfig` (default: 3, range 1–10). `handle_message()` checks depth from `msg.metadata` and rejects if exceeded. `_agent_runner` increments depth per call
- **Live Config Reload** — `reload_config()` methods on Executor and WebTools. PATCH routes in `config_routes.py` call `gateway.reload_components(config=True)` to propagate changes immediately without restart
- **DomainListInput UI Component** — Regex-validated domain input in CognithorControlCenter. Rejects schemes, paths, wildcards, spaces. Used for `domain_blocklist` and `domain_allowlist`
- **Secret Masking Verification** — Explicit tests confirming `google_cse_api_key`, `jina_api_key`, `brave_api_key` are correctly masked by `_is_secret_field()` pattern matching

### Changed

- Blocked actions now count as "completed" in DAG dependency resolution, allowing their dependents to proceed
- `_dag_workflow_engine` attribute declared and initialized in `advanced.py` phase
- Orchestrator runner wired in Gateway (creates `IncomingMessage` with `channel="sub_agent"`)
- DAG WorkflowEngine wired with `_mcp_client` and `_gatekeeper` in Gateway `apply_phase()`
- `SecurityPage` in UI gains `max_sub_agent_depth` NumberInput
- Test count: 9,357 → **9,596** (+239 tests across 9 test files)
- LOC source: ~106,000 → ~109,000
- MCP tool count: 47 → **48** (added http_request)

---

## [0.26.6] – 2026-03-05

### Chat, Voice, Agent Infrastructure & Security Hardening

Comprehensive release bringing integrated chat, voice mode, 15 new enterprise subsystems,
and deep security hardening. Full suite at 9,357 tests (0 failures).

### Added

**Chat & Voice**
- **ChatPage** (`ui/src/pages/ChatPage.jsx`) — Full chat integration in the React UI with WebSocket streaming
- **MessageList**, **ChatInput**, **ChatCanvas**, **ToolIndicator**, **ApprovalBanner** — Chat UI components
- **VoiceIndicator** + **useVoiceMode** — Voice mode with wake word ("Jarvis"), Levenshtein matching, Konversationsmodus
- **Piper TTS (Thorsten Emotional)** — German speech synthesis, automatic model download
- **Natural Language Responses** — System prompt for spoken, human responses

**Agent Infrastructure (15 Subsystems)**
- **DAG Workflow Engine** — Parallel branch execution, conditional edges, cycle detection (53 tests)
- **Execution Graph UI** — Real-time visualization data with Mermaid export (37 tests)
- **Agent Delegation Engine** — Typed contracts with SLA guarantees (44 tests)
- **Policy-as-Code Governance** — Versioned policy store, simulation, rollback (41 tests)
- **Knowledge Graph Layer** — NER, entity deduplication, graph visualization (46 tests)
- **Memory Consolidation** — Importance scoring, deduplication, retention (48 tests)
- **Multi-Agent Collaboration** — Debate, voting, pipeline patterns (52 tests)
- **Agent SDK** — Decorator-based registration, scaffolding (38 tests)
- **Plugin Marketplace Remote Registry** — Remote manifests, dependency resolution (36 tests)
- **Tool Sandbox Hardening** — Per-tool resource limits, escape detection (93 tests)
- **Distributed Worker Runtime** — Job routing, failover, dead-letter queue (64 tests)
- **Deterministic Replay** — Record/replay with what-if analysis (55 tests)
- **Agent Benchmark Suite** — 14 tasks, composite scoring, regression detection (48 tests)
- **Installer Modernization** — uv auto-detection, 10x faster installs (36 tests)
- **GDPR Compliance Toolkit** — Art. 15-17, 30, retention enforcement (49 tests)

**Security & Performance Hardening**
- Path traversal prevention in vault.py, memory_server.py, code_tools.py
- run_python Gatekeeper bypass protection (14 pattern regex)
- WebSocket authentication with token validation
- ModelRouter race condition fix (ContextVar per-task isolation)
- Embedding memory optimization (batched SQL queries)
- Graph traversal cycle guard (iterative BFS)
- Blocking I/O elimination (WAL mode, buffered audit, run_in_executor)
- CircuitBreaker HALF_OPEN race fix with inflight counter
- Unicode normalization (NFKC) + zero-width stripping for injection defense
- HMAC-based vault key derivation (replaces simple concatenation)
- 3 new credential masking patterns (AWS AKIA, PEM keys, generic secrets)
- Atomic policy rollback with backup/restore mechanism
- Thread-safe session store with double-check locking
- SQLite synchronous=NORMAL for WAL mode performance

### Changed
- **Beta/Experimental label** — README clearly marks Cognithor as Beta (#4)
- **Internationalization (i18n)** — Error messages support English via `JARVIS_LANGUAGE=en` (#4)
- **Status & Maturity** — README includes component maturity matrix (#4)
- **Shutdown audit** — Gatekeeper registers `atexit` handler for audit buffer flush
- **ContextVar propagation** — Fixed redundant set_coding_override() in create_task()
- Test count: 4,879 → **9,357** (+4,478 tests across all features)
- LOC source: ~53,000 → ~106,000
- LOC tests: ~56,000 → ~90,000

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
