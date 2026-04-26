# Changelog

All notable changes to Cognithor are documented in this file.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `cognithor.crew.trace_bus.TraceBus` — in-process pub/sub for crew audit events. Hooks `compiler.append_audit()` to live-broadcast crew_* events via lifecycle stream + per-trace topic subscriptions. Backpressure: drop-oldest at 1000-event queue cap (WP1 of v0.95.0 Trace-UI).
- `cognithor.security.owner.require_owner()` — owner-token gating for Trace-UI surfaces. Reads `COGNITHOR_OWNER_USER_ID` env var with `pyproject.toml` author-name fallback.

## [0.94.1] — 2026-04-26

### Fixed — v0.94.0 hotfix

- `[autogen]` extra (root + `cognithor_bench`) now also pins
  `autogen-ext[openai]==0.7.5`. Without it, `cognithor-bench --adapter autogen`
  raised a misleading ImportError because `OpenAIChatCompletionClient` lives
  in `autogen-ext`, not `autogen-agentchat`. Updated the adapter's error hint
  accordingly. Surfaced by post-release dry-run audit (`docs/superpowers/reports/2026-04-26-v094-dry-run-audit.md` — BUG-1, HIGH).
- `docs/integrations/catalog.json` no longer lists `sevdesk_get_invoice` and
  `sevdesk_list_contacts`. The sevDesk MCP module uses a no-op `@mcp_tool`
  marker decorator and is not yet wired into the live MCP server, so the
  catalog over-promised capability. The generator script
  (`scripts/generate_integrations_catalog.py`) now has an explicit
  `NOT_YET_REGISTERED_PREFIXES` filter — re-add `cognithor.mcp.sevdesk` to
  the catalog automatically when the connector lands. (BUG-2, MEDIUM.)
- `insurance-agent-pack run --interview` no longer renders `§34d` as
  mojibake on Windows consoles. CLI now reconfigures `sys.stdout`/`sys.stderr`
  to UTF-8 on `win32` at entry. (BUG-3, LOW.)

## [0.94.0] — 2026-04-25

### Added — AutoGen Strategy Adoption

- `cognithor.compat.autogen` — source-compatibility shim for
  `autogen-agentchat==0.7.5` (WP2). Search-and-replace import
  migration from AutoGen-AgentChat to Cognithor; 1-shot path uses
  `cognithor.crew`, multi-round path uses a 250-LOC `_RoundRobinAdapter`.
  Supported: `AssistantAgent`, `RoundRobinGroupChat`, message + termination
  classes, `OpenAIChatCompletionClient` wrapper. Not supported by design:
  `SelectorGroupChat`, `Swarm`, `MagenticOneGroupChat` (see ADR 0001).
- `cognithor_bench/` — reproducible Multi-Agent benchmark scaffold (WP4).
  CLI: `cognithor-bench run|tabulate`. Default adapter: Cognithor; opt-in
  AutoGen adapter via `pip install cognithor[autogen]`. Bundled smoke
  scenarios under `cognithor_bench/src/cognithor_bench/scenarios/`.
- `examples/insurance-agent-pack/` — DACH insurance pre-advisory reference
  pack (WP3). Standalone `pip install ./examples/insurance-agent-pack/`.
  4 agents (NeedsAssessor, PolicyAnalyst with PDF tool-use, ComplianceGatekeeper
  as visible PGE-demo, ReportGenerator). §34d-NEUTRAL — see DISCLAIMER.md.
- `docs/competitive-analysis/` — comparison docs for AutoGen, MAF, LangGraph, CrewAI (WP1).
- `docs/adr/0001-pge-trinity-vs-group-chat.md` — first Architecture Decision Record (WP5).
- `NOTICE` — AutoGen-MIT attribution under "Third-party attributions".
- `pyproject.toml` — new `[autogen]` extra (`autogen-agentchat==0.7.5`)
  as the single pin-point for v0.94.0's source-compat shim and bench adapter.
- `pyproject.toml` `[dev]` extra — registers `insurance-agent-pack` editable.

## [0.93.0] -- 2026-04-24

### Added
- **`cognithor.crew` — Crew-Layer (Feature 1 of v1.0 adoption)** — high-level
  declarative Multi-Agent API on top of PGE-Trinity. `CrewAgent`, `CrewTask`,
  `Crew`, `CrewProcess` (SEQUENTIAL + HIERARCHICAL), plus async kickoff,
  YAML loader, and `@agent` / `@task` / `@crew` method decorators. Every
  execution routes through the existing Planner → Gatekeeper → Executor
  pipeline — no new LLM entry point, no bypass. Audit events emit via the
  Hashline-Guard chain with PII redaction. Trilingual error messages
  (en/de/zh) via `cognithor.i18n`. Spec at
  `docs/superpowers/specs/2026-04-23-cognithor-crew-v1-adoption.md`.
- **`cognithor.crew.guardrails` — Task-Level Guardrails (Feature 4)** — function-
  based + string-based validators, built-in `hallucination_check`, `word_count`,
  `no_pii` (DE-focused), `schema` (Pydantic), plus `chain()` combinator. Failures
  trigger retry-with-feedback up to `task.max_retries`, then raise
  `GuardrailFailure`. Every verdict is recorded in the Hashline-Guard audit chain
  with PII-detection flag.
- **`cognithor init` scaffolder + 5 first-party templates (Feature 3)** —
  `cognithor init <name> --template <name> [--dir PATH] [--lang de|en]`
  generates a runnable Crew project from Jinja2 templates. Templates: `research`,
  `customer-support`, `data-analyst`, `content`, `versicherungs-vergleich`
  (DACH-differentiator, fully offline-capable, §34d-neutral guardrails).
  `cognithor init --list-templates` prints the catalog with DE/EN descriptions.
  CI scaffolds every template on every PR.
- **Integrations Catalog (Feature 7)** — `docs/integrations/catalog.json`
  auto-generated from MCP tool definitions by
  `scripts/generate_integrations_catalog.py`. CI fails on drift. Includes
  a new DACH-specific sevDesk REST connector (v1.0 Launch).
- **Quickstart Documentation (Feature 2)** — `docs/quickstart/` — 8-page
  bilingual (DE primary + EN) walkthrough from install to production. Includes
  5 runnable examples under `examples/quickstart/` (first-crew, first-tool,
  first-skill, guardrails, PKV report). CI smoke-tests every example on every
  PR via `.github/workflows/quickstart-examples.yml`.

### Breaking Changes
None. The Crew-Layer is strictly additive — no existing public API changes.

## [0.92.7] -- 2026-04-23

### Added
- **Video input via vLLM** — attach a local video (`.mp4` / `.webm` / `.mov` / `.mkv` /
  `.avi`) or paste a direct video URL in chat; Qwen3.6-27B (or any vLLM-served
  video-capable VLM) analyzes it end-to-end. Native vLLM `video_url` content
  type — no frame-extraction workarounds. Adaptive frame sampling based on
  duration (fps=3 for clips under 10 s, num_frames=32 for videos over 5 min)
  via `ffprobe`. Single video per chat turn. Local uploads served to vLLM over
  a 127.0.0.1-only HTTP file server. Videos are cleaned up when the chat
  session closes and auto-expire after 24 h. Video requests on a DEGRADED vLLM
  produce a hard error — no silent fallback to Ollama (Ollama has no vision).
  Windows installer now bundles an LGPL-licensed ffmpeg build. See
  `docs/vllm-user-guide.md` and `docs/superpowers/specs/2026-04-23-video-input-vllm-design.md`.
- **Flutter URL-einfügen dialog** — paperclip → "URL einfügen" opens an
  AlertDialog for a direct video URL (adds to Decision 1 of the video-input
  spec). Delegates validation to `ChatProvider.handlePastedTextForVideoUrl`.
- **Structured logging for docker run** — `VLLMOrchestrator.start_container`
  emits `log.info("vllm_docker_run_starting", cmd=..., model=..., port=...)`
  before `subprocess.run` and `log.error("vllm_docker_run_failed", ...)` on
  non-zero exit. HF_TOKEN values are redacted. Makes fragile-startup
  debugging possible without a local debugger.

### Changed
- Cognithor now requires Docker Engine ≥ 20.10 when using vLLM on Linux
  (host-gateway flag for `--add-host` was added in 20.10). Docker Desktop
  versions are all fine.
- Flutter paperclip in the chat input is now a popup menu with explicit
  entries for Image / Video / File / URL instead of a single file picker.
- Default `vllm/vllm-openai` image flipped from `v0.19.1` to `cu130-nightly`
  because the tagged release crashes Qwen3.6-27B-NVFP4 at warmup on SM120;
  the nightly ships the `FlashInferCutlassNvFp4LinearKernel` fix. See
  `docs/superpowers/spikes/2026-04-23-video-input-vllm-spike-findings.md`.
- `VLLMOrchestrator` now threads the live `VLLMConfig` through both
  construction sites (`backends_api._get_orchestrator` and
  `Gateway.__init__`) so user overrides of `max_model_len`,
  `gpu_memory_utilization`, `cpu_offload_gb`, `enforce_eager` actually
  reach the `docker run` command. Previously both sites fell back to
  `VLLMConfig()` defaults silently.
- Backend code paths unify on a single `VLLMOrchestrator` instance via
  `app.state.vllm_orchestrator`. Previously `backends_api` had its own
  cache, and the UI's "Start vLLM" button hit that second instance which
  never had `media_url` wired — leaving the container unable to fetch
  uploads. Fallback cache retained for standalone-API mode.

### Fixed
- **Video helper preserves text on multi-modal turns** — both
  `_attach_video_to_last_user` and `_attach_images_to_last_user` now
  handle list-form content (a prior image/video attachment in the same
  turn). Previously the text was silently dropped to `""` and the
  model received only the media with no question.
- **Flutter send-button race during upload** — `_pickVideo` now tracks
  `_isUploading`, the Send button becomes a progress indicator, and
  `_submit` short-circuits until the multipart POST completes. Also
  defers the `_isUploading = false` reset to a `WidgetsBinding`
  post-frame callback so the Send tree has rebuilt before the guard
  lifts.
- **ffprobe / ffmpeg no longer block the event loop** — the FastAPI
  `/api/media/upload` handler and the Gateway per-turn handler both
  offload the blocking `subprocess.run` calls via `asyncio.to_thread`.
  Concurrent uploads stop serialising on the uvicorn worker.
- **Path-traversal guard hardened** — `/media/{filename}` endpoints on
  both the Flutter-facing `/api/media/thumb` and the vLLM-facing
  MediaUploadServer now verify `resolved.is_relative_to(media_dir)`
  instead of substring-checking `"/" in filename or ".." in filename`.
  The old guard let `C:%5CWindows%5Csystem32%5Ccmd.exe` through on
  Windows because `pathlib` resolves absolute paths as replacements.
- **Quota TOCTOU** — `MediaUploadServer.save_upload` is now protected by
  a `threading.Lock` over the evict-and-write critical section. Two
  concurrent uploads can no longer both pass the quota check
  independently, leave the dir above quota, and silently accept a
  third party's eviction.
- **Session-close cleanup wired** — `Gateway._cleanup_stale_sessions`
  now dispatches `VideoCleanupWorker.on_session_close(session_id)`
  via `loop.create_task` when a running loop is available. Previously
  the 24 h TTL sweep was the only deletion path; videos from closed
  sessions persisted until then.
- **VideoCleanupWorker.start() is idempotent** — second calls without
  an intervening `stop()` no longer orphan the first sweep task.
- **URL paste filename strips query + fragment** — `clip.mp4?token=abc`
  now derives `filename="clip.mp4"` while the full URL (including
  query/fragment for fetching) remains intact in the pending
  attachment. The regex also now accepts URLs with `?` or `#` after
  the extension — previously it rejected them entirely.
- **chat_stream accepts video kwarg** — `VLLMBackend.chat_stream` now
  threads `video=` through the same way `chat()` does. A future caller
  routing video through the streaming path no longer silently drops
  the attachment.
- **URL-dialog TextEditingController lifecycle** — moved into a
  `_UrlInputDialog` StatefulWidget so its `dispose` runs after the
  dialog's exit animation.
- **KeyboardListener FocusNode lifecycle** — promoted from inline
  `FocusNode()` in `build()` (leaked every rebuild) to a `late final`
  state field initialised in `initState`, disposed in `dispose`.
- **Quota-exceeded recovery hint** — the 507 response for
  `MediaUploadQuotaExceededError` now includes `recovery_hint` so the
  client can surface actionable guidance. Previously the generic
  `MediaUploadError` branch dropped it.

## [0.92.6] -- 2026-04-23

### Added
- **vLLM as opt-in LLM backend** with full Flutter-driven lifecycle (install,
  pull-image, start, stop, hot-switch) — spec at
  `docs/superpowers/specs/2026-04-22-vllm-opt-in-backend-design.md`,
  plan at `docs/superpowers/plans/2026-04-22-vllm-opt-in-backend.md`. Ollama
  remains the default — vLLM is purely additive. Unlocks native FP4 on
  Blackwell GPUs (RTX 50xx), FP8 on Ada, AWQ-INT4 on Ampere. Model registry
  carries per-quantization entries with `min_compute_capability` and
  `vram_gb_min` so the UI can disable models that don't fit the detected GPU
  and auto-recommend the best fit. New `LlmBackendsScreen` and
  `VllmSetupScreen` in Flutter. New FastAPI endpoints under
  `/api/backends/vllm/*` including SSE-streamed pull progress. User guide
  at `docs/vllm-user-guide.md`, manual test recipe at
  `docs/vllm-manual-test.md`.

### Fixed
- **Language change in Settings now actually changes the chat response
  language** (closes #136). Reported by @PCAssistSoftware: switching
  Administration → Configuration → Language from German to English still
  produced German chat replies. Root cause: the config-save handler in
  `config_routes.py` was calling `gateway.reload_components(config=True)`
  only — which reloads the config itself but leaves the `Planner`'s cached
  `_system_prompt_template` untouched. The planner had already built its
  template from the German i18n preset (or the hardcoded German fallback
  which literally contains `"Du sprichst Deutsch."`) and continued using
  it for every subsequent LLM call. Now the handler additionally passes
  `prompts=True` when the language field is part of the update, which
  triggers `planner.reload_prompts()` and re-resolves the template from
  the preset chain for the new language. No restart required.

## [0.92.5] -- 2026-04-22

### Added
- **Qwen3.6 model registry + installer** (`cognithor models install <name>`).
  New CLI commands:
    - `cognithor models list` — print all known models grouped by provider.
    - `cognithor models install <name>` — install an Ollama tag (e.g.
      `qwen3.6:35b`) or a community HuggingFace GGUF (e.g.
      `unsloth/Qwen3.6-27B-GGUF`). HF GGUFs are downloaded via
      `huggingface_hub` and imported into the local Ollama via
      `ollama create -f Modelfile`, ending up under the registry's
      `import_as` tag (e.g. `qwen3.6:27b`, matching upstream).
  Model registry gained a `community_gguf` provider section + Qwen3.6
  entries. Registry updated timestamp: 2026-04-22.
- **Vision routing for the Planner**. When `WorkingMemory.image_attachments`
  is non-empty, `Planner.formulate_response()` routes the LLM call through
  `config.vision_model_detail` and passes the image paths to
  `OllamaClient.chat(images=...)`. The Ollama client encodes paths (or
  pre-encoded base64 strings) and attaches them to the last user message
  per Ollama's multimodal chat API.
- **Flutter WebUI image uploads → VLM**. When the user attaches an image
  in the chat input (`chat_input.dart` already supports PNG/JPG/WEBP/...),
  the WebUI backend now:
    - Detects the image MIME from the filename extension.
    - Saves the raw bytes to `~/.cognithor/workspace/uploads/`.
    - Populates `IncomingMessage.attachments` with the path (instead of
      forcing a text-extraction pass that would lose visual context).
  The Gateway filters `msg.attachments` for image extensions and sets
  `WorkingMemory.image_attachments` on the current turn, triggering the
  Planner's vision routing. The chat bubble renders a thumbnail preview
  of uploaded images via `metadata.image_base64`.
- **Latent file-upload bug fix**. The WebUI WebSocket file-upload handler
  previously only fired when `text.startswith("[file_upload]")` — a legacy
  format the current Flutter client has never sent. File uploads from
  Flutter were silently dropped. The gate is now `metadata.file_base64`
  presence, covering the real message shape.

### Fixed
- **`config.yaml` with legacy keys no longer hard-crashes at startup**
  (closes #131). Upgrading users whose `config.yaml` still contains fields
  the current `CognithorConfig` no longer recognizes (e.g. `max_agents`,
  `max_concurrent`, `memory_limit_mb`, `rag`) would hit
  `pydantic_core.ValidationError: extra_forbidden` inside `load_config`
  and be completely unable to launch. `load_config` now catches
  `extra_forbidden` errors from the on-disk YAML read path, strips the
  offending keys, logs a clear WARN listing them, and re-validates once.
  `CognithorConfig` keeps `extra="forbid"` on the model itself so
  programmatic construction in tests still catches typos — only the
  user-supplied YAML is self-healing.

- **Flutter version mismatch overlay blocked the app on every 0.92.x
  installer** (reported on #131 by @PCAssistSoftware). The Flutter
  frontend hard-coded `kFrontendVersion = '0.91.0'` in
  `connection_provider.dart` and `version: 0.91.0+1` in `pubspec.yaml`;
  neither was bumped during the 0.92.x release cadence, so every fresh
  0.92.x install reported itself as 0.91 to the backend, tripped the
  major.minor compatibility check, and wedged users behind the "Version
  Mismatch" overlay with no way out. Both constants are now bumped to
  match `cognithor.__version__` and a new
  `tests/test_flutter_version_sync.py` cross-checks the Flutter version
  against the Python `__version__` so CI fails loudly if a future
  release bump forgets Flutter.

### Deferred
- **Video input** is explicitly deferred. Qwen3.6-27B (VLM) can process
  video frames, but Ollama's `/api/chat` endpoint does not support video
  input. End-to-end video would require a direct Transformers/vLLM
  backend — tracked separately.

## [0.92.4] -- 2026-04-22

### Added
- **Local PII redactor** (closes #122). Opt-in regex-based redaction of
  outbound LLM messages. Seven categories — `email`, `phone`, `api_key`
  (OpenAI/Anthropic/GitHub/AWS/Google/Slack/HF), `credit_card` (Luhn-
  validated), `ssn`, `iban`, `private_key` (PEM blocks). Runs inside
  `OllamaClient.chat()` so every LLM call is covered (Planner, Observer,
  Reflector, browser vision). Default-off for backward compat; enable via
  `security.pii_redactor.enabled`. Zero external calls, zero telemetry —
  matches Cognithor's local-first promise. Optional spaCy NER mode reserved
  for later (names/orgs/locations). Co-designed with @teodorofodocrispin-cmyk.

## [0.92.3] -- 2026-04-21

### Added
- **Observer Audit Layer** (#118). New LLM-based response quality check that runs
  after the existing regex-based `ResponseValidator`. Audits every response
  across four dimensions — Hallucination, Sycophancy, Laziness, Tool-Ignorance
  — with per-dimension retry strategies:
    - Hallucination failures trigger response-regeneration inside the Planner.
    - Tool-Ignorance failures trigger a full PGE re-loop via the Gateway.
    - Sycophancy and Laziness are advisory (logged, non-blocking).
  Exhausted retries deliver the response with a `[Quality check flagged issues]`
  prefix. Config: `observer.*` section + `models.observer`. See
  `CONFIG_REFERENCE.md` for all options. Circuit breaker disables the Observer
  after consecutive failures; audit records persist to
  `~/.cognithor/db/observer_audits.db`. Dedicated `qwen3:32b` audit model with
  graceful degraded-mode fallback to the planner model when observer model
  is not installed. 96%+ test coverage on observer modules.
- **Structured skill documentation** (#117, #123, #124). All 5 core SKILL.md
  files upgraded from 3-line stubs to structured docs with YAML frontmatter,
  Steps, Examples, Error Handling, and Troubleshooting tables. External
  contribution from @rohan-tessl.

### Changed
- **Breaking**: `Planner.formulate_response()` now returns `ResponseEnvelope`
  (with optional `PGEReloopDirective`) instead of a plain `str`. All in-tree
  callers updated. Downstream integrations must dereference
  `envelope.content`.
- **Complete Jarvis → Cognithor rebrand** (#121, #125, #128):
    - Python: `JarvisConfig` → `CognithorConfig`, `jarvis_home` → `cognithor_home`
      (202 call sites, 159 attribute accesses). Backward-compat aliases added
      in #121 then removed in #125 after confirming zero external consumers.
    - Flutter: `JarvisApp`, `JarvisTheme`, and 15 other widget/service classes
      renamed to `Cognithor*`. 28 source files renamed (`jarvis_*.dart` →
      `cognithor_*.dart`). All 4 ARB i18n files (en/de/ar/zh) + regenerated
      l10n updated. 1635 → 19 remaining refs (backend-contract identifiers
      that require coordinated backend+frontend migration).
    - YAML auto-migration for existing user configs with legacy `jarvis_home:` key.
    - `JARVIS_*` env var prefix still accepted as legacy alias.
- **Documentation sweep** (#119, #126, #129). README, QUICKSTART, FAQ,
  ARCHITECTURE, CHANNELS_GUIDE, DEVELOPER, CONFIG_REFERENCE, TROUBLESHOOTING,
  DATABASE, IDENTITY, and install scripts now show `COGNITHOR_*` env vars,
  paths, and identifiers as primary. Observer mentioned in README status
  table, ARCHITECTURE PGE section, and FAQ.

### Fixed
- **install.sh silently stopped during Flutter check on Arch Linux** (#120,
  closes Reddit feedback item 1). Root cause was `python -c "import jarvis"`
  post-install verify, which failed since the package rename. The subsequent
  `fatal "Installation failed"` exit scrolled past the Flutter output, making
  it look like a Flutter problem.
- **`JARVIS_HOME` / `COGNITHOR_HOME` env vars crashed config load** (#119,
  closes Reddit feedback item 3). Config's `_apply_env_overrides` mapped the
  single-word env var to a rejected `home` field under Pydantic
  `extra="forbid"`. Now correctly aliases to the real `cognithor_home` field.
- **`/config` CLI command crashed with asyncio error** (#120, closes Reddit
  feedback item 4). `prompt_toolkit.prompt()` was called synchronously from
  an async handler. Now wrapped in `asyncio.to_thread()`.
- **No auto-config creation on first run** (#120, closes Reddit feedback
  item 5). Transitive fix from item 1: `setup_directories()` runs `cognithor
  --init-only` to materialize `config.yaml`, but only after
  `install_cognithor()` succeeds. Fixing item 1 fixes item 5.
- **`pysqlcipher3` install failed on macOS** (#127, closes #116). Generic
  helper passed Debian package name `libsqlcipher-dev` to brew, which
  fuzzy-matched to the unrelated `libserdes` formula. Now resolves per-PM:
  `libsqlcipher-dev` (apt), `sqlcipher-devel` (dnf), `sqlcipher`
  (pacman/brew).
- **Flaky `test_full_lifecycle` on Windows** (#123). Windows `time.time()` has
  ~15.6ms resolution; a fast-executing recorder lifecycle produces
  `started_at == finished_at`. Changed strict `>` assertion to `>=`.
- **`_call_llm_audit` used wrong kwarg name** (in #121 live tests). Passed
  `format="json"` but `OllamaClient.chat()` expects `format_json=True`.
  Observer was silently failing-open in production. Fixed before first ship.
- **Pre-existing `_migrate_jarvis_home` bug** where `src` and `dst` pointed
  at the same path (migration no-op for users with legacy `~/.jarvis/`).
  Fixed as side effect of the rebrand.

## [0.92.2] -- 2026-04-19

See GitHub release notes for details.

## [0.92.1] -- 2026-04-18

See GitHub release notes for details.

## [0.92.0] -- 2026-04-16

See GitHub release notes for details.

## [0.91.0] -- 2026-04-12

### Fixed
- **i18n language selection (#109)** — Installer post-processes `agents.yaml` based on chosen language, planner uses `_FORMULATE_TEMPLATES` dict (de/en) for all response prompts, voice STT channels (Telegram, Signal, WhatsApp, voice_ws_bridge) honor `config.language`
- **Approval flow** — `_WebUIBridge.request_approval()` no longer auto-approves; sends real `approval_request` over WebSocket and waits for client response (30min timeout)
- **Approval timeout** — Extended from 5min to 30min to accommodate user review time
- **Conversation tree empty** — `conversation_id` and `active_leaf_id` now persisted in session store; `/chat/tree/latest` endpoint accepts session_id query param
- **Chat dark background** — Chat body wrapped in Container with `scaffoldBackgroundColor` to prevent white background in tree view
- **Kanban sqlcipher crash** — `sqlite3.Row` row_factory incompatible with sqlcipher3 cursor; use `sqlcipher3.Row` with `_dict_row_factory` fallback
- **ORANGE approval routing** — Session ID mismatch fixed: gateway's internal session key vs WS client-facing session ID
- **Duplicate evolution plans** — Loop checked for existing plans by goal before creating new ones; cleaned up 323 duplicates
- **Evolution build TypeError** — `SkillGap()` missing required `id` argument prevented all build cycles
- **Gateway evolution goal progress endpoint** — Now reads from GoalManager (live progress) instead of config.yaml (static)
- **Planner refusal on action requests** — Retry with explicit tool-plan hint when LLM self-censors on file/delete operations
- **Config auto-swap** — `model_post_init` no longer overrides explicit `llm_backend_type` setting
- **DB migration** — Auto-migrate `~/.jarvis/` data to `~/.cognithor/` on first start; keyring key fallback; corrupt session DB recovery
- **Chat header "Jarvis"** — Renamed to "Cognithor" in all 4 locales (en/de/zh/ar) and planner system prompt
- **Agent roster** — System prompt lists all 6 available agents (Cognithor, Researcher, Coder, Office, Operator, Frontier)

### Added
- **Live Logs tab** in MonitoringScreen — polls `/api/v1/monitoring/events` every 5s, severity filter chips, auto-scroll with "new events" button
- **Invisible backend** — `cognithor.bat` uses `pythonw.exe` in `--ui` mode, no console window
- **ConnectionGuard overlay** — Blocks UI with non-dismissible overlay when backend unreachable, 15s health polling
- **Agent delegation visibility** — Backend broadcasts delegation status, Flutter shows agent badge on messages
- **Evolution goal edit/delete** — New popup menu entries with confirmation dialog
- **GitHub model list update** — Cognithor installer recommender updated to qwen3.5, llama4, gemma3, devstral (from obsolete qwen2.5, llama3.1)
- **Backend migration on upgrade** — Auto-copies data from legacy `~/.jarvis/` location

## [0.90.0] -- 2026-04-11

### Added
- **Cross-Platform Social Listening** — Hacker News + Discord scanners join existing Reddit system
  - `src/cognithor/social/hn_scanner.py` — HackerNewsScanner: Firebase API for story IDs, Algolia for search, HN-culture-aware LLM scoring, zero auth required
  - `src/cognithor/social/discord_scanner.py` — DiscordScanner: REST API v10 via httpx, bot token auth, message history fetch, 1s rate limiting between channels
  - `src/cognithor/mcp/social_tools.py` — 2 unified MCP tools: `social_scan` (dispatches to any/all platforms), `social_leads` (unified listing with platform filter)
  - Lead model gains `platform`, `platform_id`, `platform_url` fields; store migration with platform-aware queries
  - `RedditLeadService` extended with `scan_hackernews()`, `scan_discord()`, `scan_all()` methods
  - Config: `hn_enabled`, `hn_categories`, `hn_min_score`, `hn_scan_interval_minutes`, `discord_scanner_enabled`, `discord_scan_channels`, `discord_min_score`, `discord_scan_interval_minutes`
  - Gateway wiring: HN/Discord scanners initialized in post-init block
  - Flutter: HN + Discord config sections in Social Listening page
  - Gatekeeper: `social_scan` and `social_leads` classified as GREEN
- **Hierarchical Document Reasoning** — 4th retrieval channel (tree-based, vectorless)
  - `src/cognithor/memory/hierarchical/` — 8 modules: 5 parsers (markdown, pdf, docx, html, plaintext), `tree_builder`, `tree_store` (SQLite), `node_selector` (LLM-navigated), `retrieval`, `manager`
  - Builds heading-based document trees; LLM navigates tree structure to find relevant sections
  - No embeddings required — structural understanding through document hierarchy
  - 136 tests
- **CAG Layer (Cache-Augmented Generation)** — KV-cache prefix reuse for LLM acceleration
  - `src/cognithor/memory/cag/` — 7 modules: `content_normalizer`, `cache_store`, `selectors`, `metrics`, `builders/prefix_builder`, `builders/native_builder`, `manager`
  - Deterministic prefix generation from memory/vault/episodes context
  - Hooks into Planner for automatic prefix injection
  - Hit-rate metrics and cache eviction
  - 71 tests
- **CLI Config TUI** — Interactive terminal config editor
  - `src/cognithor/cli/config_tui.py` — rich + prompt_toolkit based TUI
  - `src/cognithor/cli/model_registry.py` — Dynamic model discovery from live LLM providers
  - Section navigation, validation, model selection from available models
  - Launched via `cognithor --config-tui`
- Pre-release smoke test in publish workflow (wheel build + install + startup verification)
- Version consistency check in CI (pyproject.toml vs __init__.py)
- `scripts/prepare_release.py` for local pre-release validation
- `scripts/verify_readme_claims.py` for README claims verification
- 17 new tests (release smoke, env overrides, security regression)

### Changed
- **Package rename**: `jarvis` → `cognithor` across 1,265 files (5,770 replacements)
  - Both `cognithor` and `jarvis` entry points preserved for backward compatibility
  - All internal imports, config paths, env vars updated
- **AST-Based Security** — Python `ast.NodeVisitor` + `bashlex` shell parser replace regex guards
  - `src/cognithor/security/python_ast_guard.py` (40 tests)
  - `src/cognithor/security/shell_ast_guard.py` (48 tests, bashlex + regex fallback)
- **`_safe_call()` pattern** — 79 silent `except Exception: pass` replaced with tracked error handling
  - `src/cognithor/core/safe_call.py` — `_safe_call()` + `_safe_call_async()` with failure registry
- **Installer overhaul** — `install.bat` + `install.sh` rewritten
  - GPU detection, model auto-pull, health checks, uv support
  - `installer/auto_upgrade.py` — Syncs source tree to installed version on every launch
  - Inno Setup `{%USERPROFILE}` fix for runtime env var resolution
- **Vault search** upgraded from substring matching to word-level scoring with title boost
- **Evolution deep learner** — Fixed infinite loop (hardcoded queries → LLM-generated), max rounds limit

### Fixed
- **i18n language selection** (#109) — English language selection now fully applied during installation; previously agents, planner formulate-response prompts, and voice channels defaulted to German even when English was selected. `installer/first_run.py` post-processes `agents.yaml` after wizard; `planner._build_formulate_messages` uses locale-aware `_FORMULATE_TEMPLATES` dict; telegram/whatsapp/signal/voice_ws_bridge accept `stt_language` from config.
- Environment variable overrides: JARVIS_* prefix now supported for backward compatibility
- Multi-word top-level config keys (owner_name, llm_backend_type) now correctly overridden via env vars
- Makefile venv path updated from .jarvis to .cognithor
- README numeric claims verified and corrected
- `await` outside async function in gateway.py (CAG hook in sync function)
- `cag_prefix` MagicMock crash in tests (isinstance check)
- `_BootstrapRequest` missing at runtime (TYPE_CHECKING import)
- Reddit tools `parameters` vs `input_schema` keyword mismatch
- Reddit tools missing JSON Schema `"type": "object"` wrapper
- `ModelsConfig.default` AttributeError (field doesn't exist)
- Reddit LLM scoring missing `model` parameter
- `WindowsPath` concatenation in `encrypted_db.py`
- Evolution goals dict vs string format mismatch
- Multiple CI failures from rename leftovers (python -m jarvis, PREREQUISITES.md, first_boot.py)

## [0.84.0] -- 2026-04-09

### Added
- **Reddit Lead Hunter** — Full social listening system integrated into Cognithor
  - `src/jarvis/social/` package: models, store, scanner, reply, service (5 modules)
  - `src/jarvis/mcp/reddit_tools.py` — 3 MCP tools: `reddit_scan`, `reddit_leads`, `reddit_reply`
  - `data/procedures/reddit-lead-hunter.md` — Skill file with interactive chat flow (asks for product + subreddits)
  - `SocialConfig` in config.py — 8 fields for Reddit scanning configuration
  - 6 REST endpoints: `/api/v1/leads/scan`, `/leads`, `/leads/{id}`, `/leads/{id}/reply`, `/leads/stats`
  - Cron job registration for automatic scanning (configurable interval)
  - No Reddit API key needed — uses public JSON feeds
  - Hybrid reply posting: clipboard (default), browser open, Playwright auto-post (opt-in)
  - SQLCipher-encrypted leads database with status workflow (new → reviewed → replied → archived)
  - 34 tests across 6 test files
- **Reddit Leads Flutter Tab** — 7th navigation tab (Ctrl+7)
  - `LeadsScreen` with stats bar (New/Reviewed/Replied), filter row (SegmentedButton), lead list
  - `LeadCard` widget with score badge, subreddit, status chip, action buttons
  - `LeadDetailSheet` bottom sheet with editable reply editor, Post Reply / Copy / Open on Reddit / Archive
  - `RedditLeadsProvider` with 30s polling, filtering, scan trigger
  - Scan Now FAB with loading spinner
  - 6 API client methods, 21 i18n keys in EN/DE/ZH/AR
- **Social Listening Config Page** — Configuration > System > Social Listening
  - Product name, description, subreddits, min score, scan interval, reply tone, auto-post
  - Orange warning banner when required fields empty

### Fixed
- **3 showstopper bugs** found by deep code audit:
  - MCP tools never registered (Phase D ran before Phase F created the service) — moved to post-init
  - LLM function always None in scanner — wired `self._ollama.chat` post-init
  - Gatekeeper blocked all reddit tools at ORANGE risk — added to GREEN/YELLOW sets
- Default subreddits from SocialConfig now propagated to scan path
- MCP tool `min_score=0` no longer silently overridden to 60
- Flutter social config default min_score aligned to 60 (was 50)

## [0.83.0] -- 2026-04-09

### Added
- **Reddit Lead Hunter Backend** — 7 backend modules (Plan A):
  - Data models (Lead, LeadStatus, ScanResult, LeadStats)
  - LeadStore (SQLite persistence with encryption)
  - RedditScanner (JSON feed fetch + LLM scoring + reply drafting)
  - ReplyPoster (clipboard/browser/auto posting)
  - RedditLeadService (orchestrator)
  - MCP tools + gateway wiring + skill file
  - REST API (6 endpoints) + cron registration
- **Social Listening Config Page** in Flutter
- **Skill chat flow** — asks user for product/subreddits before scanning

## [0.82.0] -- 2026-04-09

### Added
- **Robot Office Live Wiring** (#84) — 5 components:
  - `RobotOfficeProvider` — aggregates WS events + REST polling (10s)
  - Dynamic robots from real configured agents + PGE Trinity always present
  - Real-time PGE state sync (Planner types when planning, Executor works when running tools)
  - System metrics driving server rack LEDs, ceiling lights
  - Kanban board with real colored dots + hover tooltips showing task titles
  - System glow on PGE Trinity robots
- **Deep Learning Upload Pipeline** (#89) — 6 components:
  - Priority queue (High/Normal/Low) with background KnowledgeBuilder pipeline
  - PDF vision (image-heavy pages analyzed via vision model)
  - OCR fallback for scanned PDFs (Tesseract, when text < 100 chars)
  - YouTube frame extraction (yt-dlp + ffmpeg for HIGH priority)
  - Evolution Loop integration (`inject_user_material`)
  - Flutter: priority dropdown on Teach screen, deep-learn queue panel
  - Structured progress events in deep-learn pipeline
- **Windows Uninstaller** (#78) — `uninstall.bat` with code-only or full removal modes

### Fixed
- **#79** — Removed duplicate Evolution settings page (1,199 lines)
- **#80** — Translate Prompts via Ollama now actually sends prompts to backend
- **#81/#82** — Missing i18n in admin pages, sidebar, toolbar (7 new keys)
- **#83** — Logo fallback improved from plain "C" to brain icon with gradient
- **#86** — Operation Mode now has description explaining each mode
- **#87** — QR pairing screen fetches payload from server instead of manual input
- **#88** — Robot Office task messages replaced with clearly playful text

## [0.81.0] -- 2026-04-08

### Added
- **OpenRouter** in backend switch dialog + API status check + switch endpoint
- **Installer downgrade protection** — bootstrap and Inno Setup detect version downgrades

### Fixed
- **299 Ruff lint errors** resolved across 143 files (all rules, zero remaining)
- **19 Flutter analyze issues** resolved (unused code, deprecated APIs, null-safety)
- **Fail-fast backend routing** — non-Ollama backends no longer silently fall back to Ollama
- **Audit verify crash** (#66) — `config_manager` not passed to monitoring routes
- **Search navigation** (#72) — Ctrl+K search navigates to correct config tab
- **Save error detail** (#71) — shows actual error message per section
- **API key display** (#64) — 24 bullet chars instead of "***", no eye button
- **Incognito flag** (#76) — `sqlite3.Row` value check fix, schema + ON CONFLICT fix
- **Duplicate pages removed** — Models (#62), Agents (#65)
- **Backend selection unified** (#63) — all 17 providers, derived from config Literal
- **Kanban agents dynamic** (#73) — dropdown from AdminProvider, not hardcoded
- **Active models shown** (#74) — provider card shows planner/executor model names
- **Kanban i18n** (#75) — 22 new keys, column names, config dialog
- **Vault i18n** (#68) — 10 new keys
- **Device pairing i18n** — 35 missing ZH/AR translations
- **Incognito exit** (#67) — clickable badge + drawer toggle
- **Agent delete** (#69) — delete button in Administration > Agents
- **Token display** (#70) — model, backend, tokens, duration on chat messages
- **Version fixes** (#85) — `__init__.py` synced, `first_run.py` reads dynamically
- **Circuit breaker warning** (#76) — `check_open()` before coroutine creation

### Changed
- `check_before_push.sh` runs full ruff lint on `src/` and `tests/`
- Gateway keepalive uses `asyncio.Event` instead of mutable boolean flag
- Valid backends derived from `JarvisConfig.llm_backend_type` Literal type at runtime
- Legacy React/Preact UI removed (`ui/`, `apps/pwa/`) — Flutter Migration Phase 4 complete
- LOC counts updated: ~199k source, ~162k tests, ~55k Flutter

## [0.80.1] -- 2026-04-08

### Fixed
- **Full Ruff lint cleanup** — resolved all 299 lint errors across 143 files
  - E501 (86): line-too-long — shortened or added noqa for string literals
  - F841 (27): unused variables — removed or prefixed with underscore
  - B904 (6): `raise ... from err` in except clauses
  - SIM102 (7): collapsible nested if statements
  - SIM105 (3): `contextlib.suppress()` instead of try/except/pass
  - B007 (4): unused loop variables prefixed with underscore
  - B023 (2): loop variable capture — replaced with `asyncio.Event` in gateway keepalive
  - RUF006 (1): dangling asyncio task — stored reference in `_background_tasks`
  - N817 (5): `Path as P` renamed to direct `Path` import
  - F401 (3): removed unused imports
  - TC001/TC002/TC003 (88): moved imports into/out of TYPE_CHECKING blocks
  - E731 (22): replaced lambda assignments with def
  - B015 (1): removed pointless comparison
  - Plus RUF005, UP036, UP038, SIM118, RUF015, RUF046
- **CI test fix** — `test_prints_manual_pull_commands` now accepts `if`-guarded ollama pull (user-confirmed)
- **check_before_push.sh** — aligned with CI: scopes to `src/` and `tests/`, ignores voice_ws_bridge

### Changed
- `check_before_push.sh` now runs full ruff lint (all rules) instead of just critical errors
- Gateway keepalive uses `asyncio.Event` instead of mutable boolean flag (race-condition safe)

## [0.74.0] -- 2026-04-05

### Added
- **SmartExplorer** (`src/jarvis/arc/smart_explorer.py`) — systematic state-action graph exploration
  - Tracks tested/untested actions per state node in a directed graph
  - Navigates to nearest frontier state (with untested actions) via BFS on known transitions
  - Prunes actions that don't change state (no wasted exploration budget)
  - Click targets detected per-state via connected components (small salient objects first)
  - Inspired by [3rd-place ARC-AGI-3 Preview solution](https://arxiv.org/abs/2512.24156)
  - **7 new games solved**: TR87, BP35, CD82, TU93, KA59, SU15, TN36
- **VisionAgent** (`src/jarvis/arc/vision_agent.py`) — qwen3-vl guided step-by-step gameplay prototype
  - Per-step vision calls with action parsing and strategy context
  - Action history tracking to prevent navigation loops
- **Incremental click-DFS** for deep click puzzles (LP85 L2)
- **Click path shortening** — removes redundant clicks from solutions
- **Clicks as regular DFS actions** — enables multi-click sequences in mixed games

### Changed
- SmartExplorer runs before KeyboardSolver DFS as faster alternative
- Action 7 now included in keyboard solver (was completely dropped)
- Click positions scanned and passed to KeyboardSolver for mixed games
- BFS depth raised to 15, timeout to 80% of budget

### Fixed
- Action 7 dropped from keyboard DFS filter (affected BP35, SK48, SB26, AR25, LF52)
- Negative action indices (encoded clicks) crashing env.step() in double-step/undo
- Click-only at (32,32) instead of detected object positions
- GameState import missing in _execute_sequence_click path shortening

### Benchmark Results (24 levels across 13/25 games)
```
FT09: 10/10  VC33: 2/7   LP85: 2/8   SP80: 1/6   CN04: 1/6
M0R0: 1/6   TR87: 1/6   BP35: 1/9   CD82: 1/6   TU93: 1/9
KA59: 1/7   SU15: 1/9   TN36: 1/7
```

## [0.73.0] -- 2026-04-05

### Added
- **ClickSequenceSolver** — BFS through click sequences with sub-level detection
  - Effective position scanner (2px grid, grouping, max-diff representative)
  - Pump-then-trigger architecture for water-routing puzzles
  - Simulation A* with real env.step() calls for state-dependent effects
  - Height-space container detection, target marker recognition (multi-color)
  - Greedy effect-matrix fallback with round-robin valve cycling
  - VC33: 3/7 levels solved (water routing, Score 21.43)
  - LP85: 1/8 levels solved (repeated click)
- **KeyboardSolver** (`src/jarvis/arc/keyboard_solver.py`) — new file
  - Incremental DFS: env.step() forward, reset only on backtrack (~50x faster than BFS)
  - Undo optimization: tries reverse direction before full reset
  - Double-step for delayed-render games (G50T FPS fix)
  - Smart action ordering: avoids immediate reversal
  - Path shortening: iterative step removal (20-45% reduction)
  - LS20: 1/7 levels (maze), SP80: 1/6, CN04: 1/6, M0R0: 1/6
- **`has_toggles` field** in GameProfile for solver routing
  - Toggle-detected games route to cluster_click first
  - Non-toggle games route to sequence_click first
- **False positive detection** — verifies levels_completed before counting solved
- **Full 25-game benchmark** — all games tested and classified
- **20 new tests** (257 total ARC tests)

### Changed
- Click budget raised to 200 (was 20)
- Per-level timeout raised to 300s (was 120s)
- Game timeout raised to 1200s (was 300s)
- Valve grouping uses exact diff match + Manhattan <= 4 (was 10% + 8)
- Group representative uses max-diff pixel (was centroid)
- BFS always runs before sim-A* (was skipped for >=4 valves)

### Fixed
- R11L false positive (was reporting 10/10, actually 0)
- BFS skip bug (LP85 missed because >=4 valves skipped straight to sim-A*)
- Non-numeric vision target_color (e.g. "green") handled gracefully
- Double-step for games with delayed grid rendering

## [0.72.0] -- 2026-04-04

### Added
- **ARC-AGI-3 GameAnalyzer** — automated per-game analysis pipeline
  - `GameAnalyzer` (`src/jarvis/arc/game_analyzer.py`): sacrifice-level exploration + 2 qwen3-vl:32b vision calls
  - `GameProfile` (`src/jarvis/arc/game_profile.py`): persistent JSON profiles with strategy metrics, learning across runs
  - `PerGameSolver` (`src/jarvis/arc/per_game_solver.py`): budget-based strategy mix (cluster_click, targeted_click, keyboard_explore, keyboard_sequence, hybrid)
  - Smart elimination search: poison-cluster detection, progressive skip, O(n) for common cases
  - Toggle-pair detection from sacrifice level (fallback when vision unavailable)
  - **FT09: 10/10 levels solved** (reproducible, ~1s per level)
- **CLI**: `--mode analyzer` + `--reanalyze` flags for `python -m jarvis.arc`
- **61 new tests** across 4 test files (237 total ARC tests)

### Changed
- `env.reset()` replaces `arcade.make()` for combo testing — 760x faster (0.5ms vs 380ms)
- Click game budget raised from 20 to 200 actions
- Game timeout raised to 20 minutes, per-level timeout 120s

### Fixed
- Cluster_click strategy now replays winning clicks on main env (env sync bug)
- Sacrifice level checks for WIN state (not just GAME_OVER)
- `initial_levels` tracking uses None sentinel instead of faulty `or` idiom
- `keyboard_sequence` now distinct from `keyboard_explore` (4x repeat per direction)
- Non-numeric vision `target_color` (e.g. "green") handled gracefully

## [0.68.0] -- 2026-03-30

### Added
- **Document System Overhaul** — 7 document tools across all major formats
  - `read_xlsx`: Excel files as Markdown tables (openpyxl)
  - `document_create`: structured JSON input → DOCX, PDF, PPTX, XLSX
  - `typst_render`: Typst markup → high-quality PDFs
  - `template_list` + `template_render`: fill Typst templates → PDF
  - 3 bundled templates: Brief, Rechnung, Bericht
  - New `src/jarvis/documents/` package with TemplateManager
- **Skill Lifecycle Manager** (`src/jarvis/skills/lifecycle.py`)
  - Hot-loading: skills immediately available after creation
  - Startup scan of `~/.jarvis/skills/generated/` directory
  - `audit_all()`, `repair_skill()`, `suggest_skills()`, `prune_unused()`
  - Auto-repair of broken skills at gateway startup
  - ARC-AGI-3 skill with 8 trigger keywords
- **Tactical Memory (Tier 6)** — tool outcome tracking
  - Hybrid RAM+SQLite persistence
  - Avoidance rules after 3 consecutive failures (24h TTL decay)
  - Context Pipeline Wave 3 injection for Planner
  - Executor post-execution outcome recording
- **Dependencies**: openpyxl>=3.1, typst (Python Typst compiler)
- **MCP tools: 111 → 122** (+11 new tools)

### Changed
- Skill registry: `register_skill()` now public with immediate index rebuild
- Gateway: scans `~/.jarvis/skills/generated/` at startup
- MemoryTier enum: added TACTICAL (6 tiers total)
- Working Memory: 400-token tactical budget added

## [0.67.0] -- 2026-03-29

### Added
- **ARC-AGI-3 Benchmark Integration** — new `src/jarvis/arc/` module (14 files)
  - `CognithorArcAgent` — hybrid agent (algorithmic + optional LLM + optional CNN)
  - `EpisodeMemory` — in-session short-term learning with MD5 state hashing
  - `GoalInferenceModule` — autonomous win-condition detection (4 strategies)
  - `HypothesisDrivenExplorer` — 3-phase exploration (Discovery → Hypothesis → Exploitation)
  - `VisualStateEncoder` — 64x64 grid-to-text conversion for LLM context
  - `MechanicsModel` — cross-level rule abstraction with EMA consistency
  - `ArcSwarmOrchestrator` — parallel game runs via asyncio
  - `ArcAuditTrail` — SHA-256 hash-chain audit for reproducible runs
  - `ActionPredictor` (CNN) — optional online-learning action predictor (torch)
  - `ArcEnvironmentAdapter` — ARC SDK bridge with safe frame extraction
- **3 new MCP tools**: `arc_play`, `arc_status`, `arc_replay` (111 total)
- **CLI**: `python -m jarvis.arc --game ls20 [--mode benchmark|swarm]`
- **ArcConfig** Pydantic model integrated into JarvisConfig
- **105 new tests** in `tests/test_arc/` (11,978+ total)
- **Dependency groups**: `arc` (arc-agi + numpy), `arc-gpu` (+ torch)

### Fixed
- SQLCipher `VirtualLock` quota exhaustion — `PRAGMA cipher_memory_security = OFF`
- `UserPreferenceStore` auto-reconnect on DB corruption (MemoryError/DatabaseError)
- Heartbeat/agent consent bypass — added user_id prefix detection
- Wikipedia navigation garbage entities filtered in knowledge_builder
- `search_and_read` blocked domains filter (zhihu, baidu, yandex etc.)
- `vault_delete` correctly classified as RED (was ORANGE in test)
- `encrypted_connect()` timeout parameter support
- Ollama timeout_seconds test aligned to 180s default

## [0.66.1] -- 2026-03-29

### Added

#### LLM Provider Expansion — 16 Providers (4 Local + 12 Cloud)
- **vLLM** backend — high-throughput local inference via OpenAI-compatible API (default: `localhost:8000/v1`)
- **llama-cpp-python** backend — lightweight local inference via OpenAI-compatible API (default: `localhost:8080/v1`)
- Both backends reuse the existing `OpenAIBackend` — zero new code, full feature parity (streaming, tool calling, JSON mode)
- Flutter UI: backend dropdown updated with `vllm` and `llama_cpp` options
- Config: `vllm_base_url`, `vllm_api_key`, `llama_cpp_base_url`, `llama_cpp_api_key` fields

Complete provider list: Ollama, LM Studio, vLLM, llama-cpp-python, OpenAI, Anthropic, Google Gemini, Groq, DeepSeek, Mistral, Together AI, OpenRouter, xAI (Grok), Cerebras, AWS Bedrock, HuggingFace. Plus any custom OpenAI-compatible endpoint.

### Fixed
- CI: pinned `ruff==0.12.12` to prevent format drift between local and CI environments
- ARCHITECTURE.md: encryption and GDPR sections that were missed in v0.66.0 release commit

## [0.66.0] -- 2026-03-29

### Added

#### Encryption at Rest — Full Disk Clone Protection
- **SQLCipher**: All 33 SQLite databases encrypted with AES-256 via `sqlcipher3` (pre-compiled Windows wheel)
- **OS Keyring**: Encryption key stored in Windows Credential Locker / macOS Keychain / Linux SecretService — never on disk
- **EncryptedFileIO**: Transparent Fernet (AES-256) encryption for memory files (CORE.md, episodes, procedures, learning plans)
- **Auto-migration**: Unencrypted databases automatically migrated to SQLCipher on first startup
- **Vault toggle**: `vault.encrypt_files` config (default: off for Obsidian compatibility, on for max security)
- **compatible_row_factory()**: Cross-compatible row factory for sqlite3 and sqlcipher3

#### Vault Dual-Backend
- **VaultBackend ABC**: Pluggable storage interface with FileBackend and DBBackend
- **VaultDBBackend**: SQLCipher-encrypted SQLite with FTS5 full-text search (when `encrypt_files=true`)
- **VaultFileBackend**: Obsidian-compatible .md files (when `encrypt_files=false`)
- **Bidirectional migration**: Automatic data transfer when toggling between file and DB mode
- **Flutter Vault Page**: Config page with encryption toggle, security info boxes, BitLocker/LUKS recommendation

#### GDPR User Rights — 100% Coverage
- **Art. 15 (Access)**: Complete export across 11 tiers (sessions, vault with content, entities, relations, episodic memories, procedures, core memory, preferences, processing logs, model usage, consents). JSON + CSV formats.
- **Art. 16 (Correct)**: `PATCH /api/v1/user/data` for entities, preferences, vault notes
- **Art. 17 (Delete)**: 7 erasure handlers covering all data tiers including vault notes
- **Art. 18/21 (Restrict)**: Per-purpose restriction (evolution, cloud_llm, memory, osint) via REST API + ComplianceEngine enforcement
- **Art. 20 (Portability)**: Export format v2.0 `cognithor_portable` + `POST /api/v1/user/data/import`
- Delete methods added to: SessionStore, UserPreferenceStore, ConversationTree, FeedbackStore, CorrectionMemory

### Fixed
- Cron jobs blocked by GDPR consent (`channel=cli` with `user_id=cron` not recognized as system)
- `session_id=None` crash in OutgoingMessage (Pydantic validation)
- `sqlite3.Row` incompatible with `sqlcipher3.Cursor` (TypeError on startup)
- `sqlite3.OperationalError` not caught for `sqlcipher3.dbapi2.OperationalError` (migration crashes)
- `sqlite3.IntegrityError` not caught for `sqlcipher3.dbapi2.IntegrityError` (vault DB)
- Knowledge synthesis tools timing out at 30s (increased to 120s)
- Cron `day_of_week=7` invalid (normalized to 0 = Sunday)
- Non-relevant language domains in research (Zhihu, Baidu, Yandex filtered)
- 5 i18n test failures (bilingual assertions)
- 4 Ruff lint errors (F821 undefined names)
- Generated skills now encrypted (reveals user interests)
- `pysqlcipher3` installation on Windows (switched to `sqlcipher3` pre-compiled binary)

### Changed
- Vault refactored from 1017 to 736 lines (delegates to pluggable backend)
- `start_cognithor.bat`: checks sqlcipher3 + keyring, auto-installs
- `start_cognithor.sh`: new Linux startup script with SQLCipher + SearXNG
- `install.sh`: pysqlcipher3/sqlcipher3 in install pipeline
- Tool timeouts: knowledge_contradictions 120s, deep_research 180s, investigate_* 120s
- 74 files reformatted with ruff

## [0.65.0] -- 2026-03-28

### Added

#### Evolution Engine v2 — Deep Autonomous Learning (Phase 5)
- **Iterative Deep Research Loop** — No max rounds; runs until 90% coverage target is reached.
- **6 Intelligence Improvements** — Coverage fix, chat feedback integration, progressive depth, auto-skills generation, spaced repetition, source quality scoring.
- **Hybrid Search System** — SearXNG + Brave + DuckDuckGo in parallel for broader, more reliable results.
- **Knowledge Validation** — Claims extraction from research results with cross-referencing against multiple sources.
- **Quality Self-Examination** — Semantic grading of generated knowledge to ensure factual accuracy.
- **Goal-Scoped Indexes** — Per-plan isolated SQLite databases with FTS5 full-text search for research data.

#### HIM — Human Investigation Module (OSINT)
- **New module** `src/jarvis/osint/` — Structured OSINT for persons, projects, and organizations.
- **3 MCP tools** — `investigate_person`, `investigate_project`, `investigate_org`.
- **7 Collectors** — GitHub (full), Web (full), arXiv (full), 4 stubs (Scholar, LinkedIn, Crunchbase, Social).
- **Evidence Aggregator** — Cross-verification of findings, claim classification, contradiction detection.
- **Trust Scorer** — 5-dimension weighted scoring (0–100): Claim Accuracy, Source Diversity, Technical Substance, Transparency, Activity Recency.
- **GDPR Gatekeeper** — Public figure detection, scope limits, depth restrictions for private persons.
- **HIM Reporter** — Markdown, JSON, quick summary output with SHA-256 signatures.
- **Planner skill** `human-investigation.md` — Triggers on "wer ist", "Investigation", "Due Diligence".
- **32 tests** for the OSINT module.

#### GDPR Compliance Layer — Privacy by Design
- **ConsentManager** — Per-channel, per-type consent with versioning (SQLite-backed).
- **ComplianceEngine** — Runtime enforcement with fail-closed semantics; blocks processing without consent.
- **System channel exemptions** — cron, sub_agent, evolution exempt via LEGITIMATE_INTEREST legal basis.
- **Privacy Mode** — Runtime toggle disabling all persistent storage.
- **OSINT consent** — Requires explicit "osint" consent type before investigation.
- **Data erasure** — `erase_all()` across processing logs, model usage, consents, and registered handlers.
- **New MCP tools** — `vault_delete`, `delete_entity`, `delete_relation` (RED gatekeeper classification).
- **REST API** — `DELETE /api/v1/user/data` (Art. 17 right to erasure), `GET /api/v1/user/data` (Art. 15/20 data portability).
- **TTL enforcement** — Daily cron job (03:00) with configurable retention periods.
- **ComplianceAuditLog** — Append-only JSONL with SHA-256 hash chain, tamper detection, pseudonymization.
- **EncryptedDB wrapper** — SQLCipher with graceful fallback to standard sqlite3.
- **Privacy notices** — German + English templates.
- **Processing Register (Art. 30)** — 13 activities with purpose, legal basis, retention, risk level.
- **DPIA Template (Art. 35)** — Automated risk scoring with DPIARiskLevel enum (LOW/MEDIUM/HIGH/CRITICAL).
- **pysqlcipher3 check** — Added to `start_cognithor.bat`, `start_cognithor.sh`, `install.sh`.
- **Telegram consent flow** — Consent gate for text, voice, photo, document messages.
- **WebUI consent flow** — Consent gate via `handle_message`.
- **Policy version re-consent** — Triggers re-consent when privacy notice version changes.
- **35 GDPR tests**.

#### Config Additions
- **OsintConfig** — `enabled`, `github_token`, `collector_timeout`, `report_ttl_days`.
- **ComplianceConfig** — `consent_required`, `compliance_engine_enabled`, `privacy_mode`, `privacy_notice_version`.
- **RetentionConfig** — `episodic_days=90`, `processing_log_days=90`, `model_usage_log_days=180`, `him_report_days=30`, `vault_osint_days=30`, `session_days=180`.

#### New Files
- `start_cognithor.sh` — Linux/macOS startup script.

### Changed
- **11,779+ tests** (was 11,712).
- **MCP tools** — 108 (was 102): `investigate_person`, `investigate_project`, `investigate_org`, `vault_delete`, `delete_entity`, `delete_relation`.
- **Skills** — 16 (was 15): added `human-investigation`.
- **Gatekeeper** — `investigate_*` classified as ORANGE, delete/erasure tools classified as RED.

### Fixed
- **Evolution Engine** — Infinite re-test loop eliminated.
- **Evolution Engine** — Garbage entity filter prevents low-quality entities from polluting indexes.
- **Evolution Engine** — Quality assessor logger bug resolved.
- **Evolution Engine** — Timeout protection added for quality tests.
- **Evolution Engine** — Status persistence across restarts.
- **Evolution Engine** — Cooldown reduced from 300s to 60s for faster iteration cycles.

## [0.60.0] -- 2026-03-27

### Added

#### Evolution Engine Phase 3 — Per-Agent Budget + Resource Monitor
- **ResourceMonitor** (`src/jarvis/system/resource_monitor.py`) — Async real-time CPU/RAM/GPU sampling via psutil + nvidia-smi subprocess. Configurable thresholds (CPU 80%, RAM 90%, GPU 80%), result caching (5s), `should_yield()` for cooperative scheduling.
- **Per-Agent Cost Tracking** — `CostTracker` extended with `agent_name` column (auto-migration), `get_agent_costs(days)` for breakdown, `check_agent_budget(agent, limit)` with 80% warning threshold. German budget messages.
- **AgentBudgetStatus** model — Per-agent daily cost, limit, ok/warning status.
- **Cooperative Scheduling** — `EvolutionLoop` checks `ResourceMonitor` before each step (scout/research/build). Pauses when system busy (CPU/RAM/GPU over thresholds). Respects per-agent budget limits from `EvolutionConfig.agent_budgets`.
- **REST API** — `GET /api/v1/budget/agents` (per-agent costs today/week/month + budget status), `GET /api/v1/system/resources` (live CPU/RAM/GPU snapshot).
- **Flutter Budget Dashboard** — New config page showing per-agent cost table, resource usage bars (color-coded), evolution status. Auto-refresh every 10s.
- **36 new tests** (17 ResourceMonitor, 8 per-agent CostTracker, 6 cooperative scheduling, 5 existing enriched).

#### Evolution Engine Phase 4 — Checkpoint/Resume Engine
- **EvolutionCheckpoint** (`src/jarvis/evolution/checkpoint.py`) — Step-level state persistence with delta snapshots. Tracks cycle_id, step_name, step_index, accumulated data (gaps, research, skills). JSON serialization with `to_dict()`/`from_dict()`.
- **EvolutionResumer** (`src/jarvis/evolution/resume.py`) — Loads last checkpoint, determines resume point (`ResumeState`), lists/clears cycles. Supports resume after any interrupted step (scout/research/build/reflect).
- **Step-Level Checkpointing** — `EvolutionLoop` saves checkpoint after each completed step via `CheckpointStore`. Stores as `PersistentCheckpoint` under `evolution-{cycle_id}` session.
- **REST API** — `GET /api/v1/evolution/stats` enriched with checkpoint + resume state, `POST /api/v1/evolution/resume` triggers manual resume of interrupted cycles.
- **Flutter Evolution Dashboard** — New config page with 4-step stepper (Scout→Research→Build→Reflect), resume button for interrupted cycles, resource status bars, recent activity table. Auto-refresh every 15s.
- **12 new tests** for EvolutionResumer (resume after each step, completion detection, cycle listing/clearing).

#### Gateway Wiring
- **ResourceMonitor** initialized at startup (lightweight psutil-based, always available).
- **CheckpointStore** initialized with `~/.jarvis/checkpoints/` path.
- **EvolutionLoop** now receives `resource_monitor`, `cost_tracker`, and `checkpoint_store` via constructor.

#### Flutter i18n
- `configPageBudget` and `configPageEvolution` added to all 4 language packs (en/de/ar/zh).
- Generated localization files regenerated via `flutter gen-l10n`.

### Changed
- **EvolutionConfig** — Added `agent_budgets: dict[str, float]` field for per-agent daily limits.
- **CostReport** — Added `cost_by_agent` field to aggregated cost reports.
- **CostRecord** — Added `agent_name` field (backward-compatible, defaults to empty string).
- **11,712+ tests** (was 11,649).
- **Flutter app** — Version bumped to 0.51.0.

## [0.59.0] -- 2026-03-27

### Added

#### Evolution Engine Phase 1 — Hardware-Aware System Profile
- **SystemDetector** (`src/jarvis/system/detector.py`) — 8 detection targets: OS, CPU (psutil), RAM, GPU (nvidia-smi + Apple Silicon fallback), Disk, Network connectivity, Ollama status + models, LM Studio status.
- **SystemProfile** — Structured result with tier classification (minimal/standard/power/enterprise), mode recommendation (offline/hybrid/online), save/load to `~/.jarvis/system_profile.json`.
- **Gateway integration** — Quick-scan on startup, cached stable results, fresh volatile scans.
- **REST API** — `GET /api/v1/system/profile` returns full hardware profile, `POST /api/v1/system/rescan` forces full re-scan.
- **Flutter Hardware Page** — New config page under System > Hardware showing tier badge, recommended mode, all detection results with color-coded status badges and expandable raw data.
- **22 tests** for SystemDetector and SystemProfile.

#### Frontier Agent
- **Cloud-powered escalation agent** — Priority 8 (highest), triggered by keywords "komplex", "architektur", "refactoring", "advanced", "expert". Uses Cloud model (Claude/GPT) for tasks exceeding local model capabilities. 5-minute timeout, 2GB RAM limit, delegates to jarvis/researcher/coder.

#### Chat Branching Wire-up
- **ConversationTree nodes stored** — Every user message + assistant response saved as tree nodes in SQLite.
- **Tree sidebar data loading** — TreeProvider loads via `GET /api/v1/chat/tree/latest` REST endpoint after each message.
- **Resizable tree sidebar** — Drag right edge (180-450px), tooltips on hover, click to navigate, fork point badges.

### Changed
- **i18n** — Transliterated German umlauts in 1568 docstring lines + all comment lines across 210+ files. Code comments/docstrings now English, user-facing strings remain German.
- **11,649+ tests** (was 11,627).
- **0.0.0.0 API binding** — Default API host changed from 127.0.0.1 to 0.0.0.0 for mobile/network access.
- **Tailscale + AltServer** — Auto-start/stop alongside Cognithor in start_cognithor.bat.

### Fixed
- Test assertion "Einzelanfrage" → "Single request" (translated production code).
- Test assertion "gekuerzt"/"truncated" synced with production output.
- Test assertion "Pfad" → "Path" (translated gatekeeper message).
- Test vault_delete moved YELLOW→ORANGE in test fixture.
- Stray `_status_cb` reference removed from gateway tree_update block.

## [0.58.0] -- 2026-03-27

### Added

#### Smart Recovery & Transparency System
- **Pre-Flight Plan Preview** — Before complex plans (2+ steps), Cognithor shows a compact plan card with 3-second auto-execute countdown. User can cancel but doesn't have to. Agentic-first: never blocks.
- **Live Correction Detection** — User types "Nein", "Stopp", "Stattdessen X" during execution → system cancels current plan, injects correction context, replans. No edit/rewind needed.
- **Post-Correction Learning** — `CorrectionMemory` (SQLite) stores user corrections. After 3 similar corrections, Cognithor proactively asks before acting. Reminders injected into Planner context via ContextPipeline.
- **RecoveryConfig** — Configurable: pre_flight_enabled, timeout (3s), min_steps (2), correction_learning, proactive_threshold (3).

#### Chat Branching — Full Conversation Tree
- **ConversationTree** — Every message is a node with parentId/childIds. SQLite persistence, fork detection, path computation, replay support. 11 tests.
- **Branch-Aware Gateway** — `switch_branch()` replays message history into fresh WorkingMemory. Session-lock protected.
- **WebSocket branch_switch** — Real-time branch switching from Flutter UI.
- **REST API** — `GET /api/v1/chat/tree/{id}`, `GET /api/v1/chat/path/{id}/{leaf}`, `POST /api/v1/chat/branch`.
- **TreeProvider** (Flutter) — Active path tracking, fork point detection, branch navigation.
- **BranchNavigator** — Inline `< 1/3 >` controls at fork points.
- **TreeSidebar** — Optional collapsible tree overview panel (toggle via toolbar).

#### Chat UX Improvements
- **Claude-style Edit** — Edit rewrites conversation from that point. Old + new versions preserved.
- **Version Navigator** — `< 1/2 >` arrows to switch between edit versions.
- **Copy Message** — Copy icon on all messages (user + assistant).
- **Retry** — Refresh icon on last assistant message to regenerate.
- **Edit + Copy + Retry icons** — Visible on every message bubble.

### Fixed

#### Security Audit (8 Fixes)
- **Browser tools classified** — `browse_click`, `browse_fill`, `browse_execute_js` now ORANGE in Gatekeeper (were unclassified → accidental ORANGE).
- **SQL injection fixed** — `persistence.py` ORDER BY clause now uses strict allowlist instead of f-string interpolation.
- **HMAC/Ed25519 in record_event()** — `AuditTrail.record_event()` now signs entries like `record()` does. Previously unsigned entries created gaps in the cryptographic chain.
- **Key file permissions** — HMAC and Ed25519 key files now `chmod 0o600` after generation.
- **Path traversal regex** — `shell.py` regex now catches single `../` (was `{2,}`, now `{1,}`).
- **vault_delete → ORANGE** — Destructive operation moved from YELLOW (inform) to ORANGE (approve).
- **TSA fallback URLs → HTTPS** — DigiCert and Apple timestamp servers now use HTTPS.
- **BreachDetector thread-safe** — `_entries` access wrapped in `list()` copy.

#### Gateway Audit (10 Fixes)
- **shutdown() cancels background tasks** — Infinite loops (retention, breach, curiosity) now properly cancelled before resource cleanup.
- **switch_branch() session-locked** — `_working_memories` write protected by `_session_lock`.
- **OpenAI empty choices guard** — `data.get("choices") or [{}]` prevents IndexError on content-filtered responses.
- **Pre-flight timeout capped** — Hard upper bound of 30 seconds prevents indefinite blocking.
- **_pattern_record_timestamps → instance var** — Was ClassVar shared across instances, now per-instance.
- **Delegation session_id** — `execute_delegation()` now passes `session_id` to `set_agent_context()`.
- **Presearch LLM guard** — `_answer_from_presearch()` returns early if `self._llm` is None.
- **_keepalive_task tracked** — Added to `_background_tasks` set for proper cleanup.
- **database_tools fetchone() guard** — All 8 `fetchone()[0]` patterns now check for None.
- **Silent except → log.debug** — Pattern search `except Exception: pass` now logs.

#### Config + Flutter Audit (8 Fixes)
- **`/api/v1/health` unauthenticated** — Removed auth dependency (was breaking bootstrap flow).
- **factory-reset endpoint** — `POST /api/v1/config/factory-reset` now exists.
- **Flutter browser defaults** — Fixed field names (was headless/timeout/max_pages → correct BrowserConfig fields).
- **Flutter email defaults** — `password` → `password_env`, `smtp_port` 587 → 465.
- **LMStudio defaults** — `lmstudio_api_key` and `lmstudio_base_url` added to Flutter.
- **German tooltips → English** — Edit/Copy/Retry tooltips now English (locale-neutral).
- **TreeProvider registered** — Added to `main.dart` MultiProvider (was missing → crash).

### Changed
- **11,627+ tests** (was 11,609).

## [0.57.0] -- 2026-03-26

### Fixed

#### Gatekeeper Tool Classification (Critical)
- **4 Knowledge Synthesis tools** (`knowledge_synthesize`, `knowledge_contradictions`, `knowledge_timeline`, `knowledge_gaps`) were missing from the GREEN tools list. They defaulted to ORANGE, causing the Executor to skip them with `GatekeeperBlock` error — resulting in **0% success rate**. Now classified as GREEN (read-only analysis tools).
- **3 Vault tools** (`vault_read`, `vault_update`, `vault_link`) were missing from the gatekeeper. `vault_read` added to GREEN, `vault_update` and `vault_link` added to YELLOW.

#### Auto-Learned Procedure Naming
- Procedures auto-generated by the Causal Learning system were saved with cryptic names like `pattern-6e8e62b97567.md`. Now uses **human-readable slugs** from the top 3 trigger keywords (e.g., `schreibe-schachbot-starte.md`). Added duplicate detection: if a file with the same name exists, appends a 6-char hash; if that also exists, skips (true duplicate).

#### Config Editability
- **5 config sections** (`browser`, `calendar`, `email`, `identity`, `personality`) were not in `_EDITABLE_SECTIONS` — changes from the Flutter UI were silently rejected. Now editable via the config PATCH API.
- **Flutter save() list** was missing `improvement`, `prompt_evolution`, and the 5 sections above. All now included with proper defaults in `_defaults()`.

#### Code Quality
- **11 E501 lines** (> 100 chars) broken across 6 files: config.py, autonomous_orchestrator.py, llm_backend.py, planner.py, trace_optimizer.py, skill_tools.py.
- **5 unused imports** removed (dead code in token_budget.py, evolution_orchestrator.py, execution_trace.py, reflexion.py).
- **3 planned imports** restored with `# noqa: F401` annotation (shutil in editor.py, CausalFinding in trace_optimizer.py, asyncio in deep_research_v2.py) — referenced in docstrings or needed for upcoming features.
- **Ruff lint status**: 0 errors on F401/F811/F821/E501. Only 33 E402 remain (intentional re-exports in `__init__.py` files).

## [0.56.0] -- 2026-03-26

### Added

#### Skill Marketplace & Community Registry
- **15 builtin skills published** to [skill-registry](https://github.com/Alex8791-cyber/skill-registry) on GitHub — productivity (7), research (2), analysis (2), development (2), legal (1), automation (1).
- **`publish_skill` MCP tool** — Publish local skills to the community registry directly from chat. Validates frontmatter, computes SHA-256 hash, uploads to GitHub, updates registry.json. Uses git credential manager for auth.
- **`publish_builtin_skills.py` script** — Batch-publishes all builtin skills with manifest.json generation and registry index update.
- **Full marketplace loop**: create_skill → publish_skill → search_community_skills → install_community_skill (5-stage validation) → use.

#### 5 New Builtin Skills
- **Meeting-Protokoll** (priority 6) — Audio transcription → structured protocol with decisions, action items, deadlines, follow-up emails. Tools: media_transcribe_audio, write_file, email_send, vault_save.
- **Tages-Report** (priority 6) — End-of-day summary from episodes + memory + calendar. Erledigt / Offen / Morgen format. Tools: get_recent_episodes, search_memory, calendar_upcoming.
- **Code Review** (priority 6) — 5-dimension review (correctness, security, performance, maintainability, tests) with line-number references. Routed to coder agent. Tools: git_diff, analyze_code, read_file.
- **Vertrags-Pruefer** (priority 6) — 7-dimension contract analysis (parties, obligations, costs, termination, liability, privacy, red flags) with risk rating and legal disclaimer. Tools: media_extract_text, analyze_document, vault_save.
- **Workflow-Recorder** (priority 7) — "Show me once, then do it forever." User describes a process → Cognithor creates a reusable skill with trigger keywords and tool mappings. Tools: create_skill, list_skills.

#### Thumbs Up/Down Feedback System
- **FeedbackStore** (`core/feedback.py`) — SQLite-backed storage for user ratings with session/message/agent tracking.
- **4 REST endpoints**: POST /feedback, PATCH /feedback/{id}, GET /feedback/stats, GET /feedback/recent.
- **WebSocket integration** — `feedback` and `feedback_comment` message types for real-time rating from chat.
- **Follow-up on negative feedback** — On thumbs down, system asks "Was hat nicht gepasst?" via `feedback_followup` message. User's comment is stored for self-improvement.
- **Flutter FeedbackButtons widget** — Thumbs up/down icons below every assistant message, with confirmation state and inline follow-up banner.

#### Interactive DAG Visualization
- **InteractiveDagGraph widget** — Pan/zoom (pinch gesture + drag) and node tap detection (30px hit radius) for the workflow DAG view.
- **DagNodeDetail panel** — Shows node status, duration, retries, type, tool name, output (truncated), and errors when a node is tapped.
- **Node selection highlight** — Selected node gets glowing border in the graph.
- **Node detail API** — `GET /api/v1/workflows/dag/runs/{run_id}/nodes/{node_id}` returns single node execution data.

### Changed
- **123 MCP tools** (was 112). New: publish_skill + 6 background task tools + feedback tools.
- **15 builtin skills** (was 10). New: meeting-protokoll, tages-report, code-review, vertrag-pruefer, workflow-recorder.
- **11,609+ tests** (was 11,595).

### Fixed
- CI lint: F821 `config_manager` scope in audit endpoints — extracted to local variable.
- CI lint: ruff format on 19 files (feedback.py, gateway.py, skill_tools.py, etc.).
- Gatekeeper test: `screenshot_desktop` and `computer_screenshot` need `ToolsConfig(enabled=True)` in test fixtures since desktop tools are OFF by default.
- GitHub Pages landing page: updated from v0.47.1 to v0.55.0 (now v0.56.0), test count badge fixed.

## [0.55.0] -- 2026-03-26

### Added

#### Audit & Compliance (EU AI Act + GDPR)
- **HMAC-SHA256 Audit Signatures** — Every audit trail entry is cryptographically signed. Auto-generates 256-bit key at `~/.jarvis/audit_key` on first use.
- **Ed25519 Asymmetric Signatures** — Optional upgrade: private key signs, public key verifies. Key pair auto-generated at `~/.jarvis/audit_ed25519.key` + `.pub`. Requires `cryptography` package.
- **Blockchain Audit Anchoring** — `AuditTrail.get_anchor()` returns chain head hash + entry count for external blockchain/Arweave anchoring.
- **RFC 3161 TSA Timestamps** — Daily timestamp on audit anchor hash via OpenSSL CLI + FreeTSA.org. Proves audit log existed at specific time. Raw ASN.1 DER fallback when OpenSSL unavailable.
- **S3/MinIO WORM Backend** — Optional append-only storage: `audit.worm_backend: "s3"` or `"minio"`. Object Lock Compliance Mode (SEC 17a-4(f) compatible). Requires `pip install cognithor[worm]`.
- **GDPR Art. 15 User Data Export** — `GET /api/v1/user/audit-data` with channel/hours filter. Returns sanitized audit entries without internal IDs.
- **GDPR Art. 33 Breach Notification** — `BreachDetector` scans every 5 minutes for SECURITY events with CRITICAL/ERROR severity. Configurable cooldown. Logs `gdpr_breach_notification` at CRITICAL level.
- **Audit Hash-Chain Verification** — `GET /api/v1/audit/verify` reads `gatekeeper.jsonl` and validates the full SHA-256 hash chain.
- **Audit Timestamps API** — `GET /api/v1/audit/timestamps` lists all RFC 3161 `.tsr` files.
- **Daily Audit Retention Cleanup** — Background task removes entries older than `audit.retention_days` (default 90) and background job logs older than 7 days.
- **Flutter Audit Page** — New config page under Security > Audit with chain verification button, TSA timestamp list, and GDPR data export.

#### Background Process Manager
- **6 new MCP tools**: `start_background`, `list_background_jobs`, `check_background_job`, `read_background_log`, `stop_background_job`, `wait_background_job`.
- **ProcessMonitor** with 5 verification methods: process-alive, exit-code, output-stall detection, timeout enforcement, resource check (optional psutil).
- **SQLite persistence** (`~/.jarvis/background_jobs.db`) — survives restarts, orphan detection on boot.
- **Log management** — Per-job log files at `~/.jarvis/workspace/background_logs/`, tail/head/grep support, 10MB limit, auto-cleanup.
- **Status notifications** — Channel-aware callbacks when background jobs complete/fail/timeout.

#### Multi-Agent System
- **5 Specialized Agents** — jarvis (generalist), researcher (web research), coder (programming), office (email/calendar), operator (DevOps). Each with custom system prompt, trigger keywords/patterns, and tool restrictions.
- **Agent Model Overrides** — `preferred_model`, `temperature`, `top_p` per agent, wired through gateway to planner. Coder uses `qwen3-coder:30b`, Office/Operator use `qwen3:8b`.
- **Agent Delegation Chain** — jarvis delegates to all 4 specialists; each can delegate back. Delegation path also respects agent LLM overrides.
- **`top_p` field** added to AgentProfile and agent CRUD API.

#### Tool Configuration
- **Computer Use Toggle** — `config.tools.computer_use_enabled` (default OFF). Desktop automation only when explicitly enabled.
- **Desktop Tools Toggle** — `config.tools.desktop_tools_enabled` (default OFF). Clipboard/screenshot access gated.
- **Flutter Tools Page** — New config page under Security > Tools with warning banner and toggles.
- **Gatekeeper enforcement** — Disabled tool groups blocked as RED even if somehow registered.
- **Runtime reload** — Toggle changes take effect immediately via `reload_disabled_tools()`.

#### MCP Server for VSCode
- **`--mcp-server` CLI flag** — Starts Jarvis as pure stdio MCP server for VSCode/Claude Desktop/Cursor. Only workspace-safe tools exposed (37 tools, no shell/desktop/docker).
- **`MCP_WORKSPACE_SAFE_TOOLS`** allowlist in `bridge.py`.
- **`mode_override`** parameter on `MCPBridge.setup()` to force stdio mode.

### Changed
- **Planner prompt** — Capability questions ("was kannst du?") answered as text from tool list, no web search. Agents answer from tool knowledge, not via `list_skills` tool plan.
- **11,595+ tests** (was 11,509).

### Fixed
- **start_cognithor.bat** — CRLF line endings (was LF, caused cmd.exe parse errors on Windows).
- **Flutter Ctrl+S** — Prevented browser Save-As dialog; now triggers Flutter config save.
- **Flutter config_provider.dart** — `tools` section added to save() list and defaults (toggle was reverting).
- **config_manager.py** — `tools` and `audit` added to `_EDITABLE_SECTIONS`.
- **Icons** — `Icons.mouse` replaced with `Icons.desktop_windows` (mouse was tree-shaken).
- **Planner log file handle leak** — Closed parent's fd copy after subprocess spawn in `BackgroundProcessManager`.

## [0.54.0] -- 2026-03-23

### Added
- **Computer Use** — GPT-5.4-style desktop automation: 6 new MCP tools (`computer_screenshot`, `computer_click`, `computer_type`, `computer_hotkey`, `computer_scroll`, `computer_drag`). Takes screenshots, analyzes with vision model, clicks at pixel coordinates. Uses PyAutoGUI + mss.
- **Deep Research v2** — Perplexity-style iterative search engine: up to 25 search rounds, automatic query decomposition (official → github → community → lateral), LLM-powered result evaluation, cross-verification across independent sources, confidence scoring, and final synthesis with source attribution. Registered as `deep_research_v2` MCP tool.
- **`[desktop]` dependency group** — `pip install cognithor[desktop]` for Computer Use (pyautogui, mss, pyperclip, Pillow).
- **106 MCP tools** total (was 99).

### Fixed
- Computer Use: hotkey string splitting, Unicode clipboard fallback, primary monitor selection, graceful degradation when deps missing.
- Gatekeeper tests updated for GREEN tool classification.

### VS Code Extension
- **cognithor-vscode/** — Complete VS Code Extension with Chat sidebar, Code Lens, 11 commands, WebSocket streaming, context-aware code assistance.
- **Backend**: `POST /api/v1/chat/completions` endpoint with CodeContext support.
- **VSIX**: Built and installable (`cognithor-vscode-0.1.0.vsix`, 26KB).

## [0.53.0] -- 2026-03-22

### Added
- **Truly Autonomous Coding**: 50 iterations (was 20), failure threshold 70% (was 50%). Cognithor writes code, tests it, debugs errors, rewrites, and re-tests autonomously until it works.
- **Autonomous Task Orchestrator**: Complex/recurring tasks auto-decomposed with self-evaluation and quality scoring.
- **Deep Research by Default**: Complex queries auto-escalate to `deep_research` with multi-source consensus.
- **Periodic Thinking Status**: UI shows "Denke nach... (15s)" during long planning phases.
- **Auto-Extract Code**: When planner returns code in permission text, auto-creates `write_file` plan.

### Changed
- **Ollama/Qwen3:32b as default planner** — local-first, no API key needed, no permission issues.
- **`operation_mode: offline` now respected** — API keys no longer override backend when offline mode set.
- **Planner prefix in English** — more authoritative for Claude, explicitly blocks permission keywords.
- **REPLAN "Fehler fixen?"** — first option when code fails, before "Fertig?".
- **No external software rule** — planner must use only Python libraries, never Stockfish/ffmpeg/ImageMagick.
- **GREEN for core tools** — `write_file`, `edit_file`, `run_python`, `exec_command` no longer require approval.
- **Auto-approve all ORANGE tools** — for fully autonomous operation.
- **Claude Code timeout 600s** (was 120s, was overridden by Ollama config).

### Fixed
- WebSocket reconnection storm (don't close stale connections).
- WebSocket accept before close (Windows semaphore error).
- CancelledError in keepalive task (BaseException, not Exception).
- Splash screen race condition (StatefulWidget with guard).
- Approval handler signature mismatch (`action` vs `tool` param).

## [0.52.0] -- 2026-03-21

### Added
- **SSH Remote Shell Backend**: Execute commands on remote hosts via SSH. 3 MCP tools (`remote_exec`, `remote_list_hosts`, `remote_test_connection`). Host registration, command validation, dangerous-pattern blocking. `remote_exec` requires ORANGE gatekeeper approval.
- **GEPA Robustness Overhaul**: Longer evaluation windows (MIN_TRACES 10→20, MIN_SESSIONS_FOR_EVAL 5→15), user approval for high-impact proposals (prompt_patch, guardrail, strategy_change), LLM-powered patch generation (was stub), cascade failure auto-detection. New `GET /api/v1/learning/gepa/status` endpoint.
- **Docker Real-Life Test Suite**: 15 scenario tests across 7 categories (web research, file ops, remote exec, memory, tool coverage, GEPA safety, session management). Dockerfile + docker-compose for isolated testing.
- **Research Auto-Escalation**: REPLAN_PROMPT now includes quality self-assessment — checks source count, agreement, gaps. Auto-escalates to `deep_research`/`search_and_read` when results are thin. SYSTEM_PROMPT adds "Gruendlichkeit" principle.
- **Flutter UI: Incognito Badge**: Purple badge in AppBar when active session is incognito. "Inkognito Chat" button in session drawer.
- **Flutter UI: Search Bar**: Live full-text search across all chats in session drawer.
- **Flutter UI: Project Sidebar**: Sessions grouped by folder with ExpansionTile. Incognito sessions show purple icon.

### Fixed
- **Mobile bottom nav**: Reduced from 8 items to 5 on phones. Removed Search/Light/Office action buttons from mobile bottom bar (remain on tablet/desktop).
- **Splash screen session reuse**: App no longer creates a new session on every launch. Uses `autoSessionOnStartup()` to resume recent sessions or create new after 30 min inactivity.
- **Light mode**: `textPrimary`, `textSecondary`, `textTertiary` and `codeBlockBg` are now theme-aware. White text on white background and dark code blocks on light background fixed.

### Testing
- Full suite: 11,475 tests passing (was 11,447)
- 15 new real-life scenario tests
- 11 new tests (remote shell, GEPA robustness, research quality)
- 0 regressions

## [0.51.0] -- 2026-03-21

### Added
- **Auto-New-Session**: Automatically creates a fresh chat session after 30 minutes of inactivity instead of resuming stale conversations. Configurable via `config.session.inactivity_timeout_minutes`.
- **Project Folders**: Group chat sessions into projects/folders. Sidebar groups sessions by folder with filtered API listing (`GET /api/v1/sessions/by-folder/{folder}`).
- **Incognito Mode**: Sessions that skip memory enrichment (context pipeline) and don't persist chat history. No long-term memory reads or writes. `POST /api/v1/sessions/new-incognito`.
- **Session Export**: Download any chat session as JSON with full metadata (`GET /api/v1/sessions/{id}/export`).
- **Full-Text Search**: Search across all chat messages in all sessions (`GET /api/v1/sessions/search?q=...`). Flutter search bar wired.
- **GDPR Retention Enforcement**: `cleanup_old_sessions()` and `cleanup_channel_mappings()` now run periodically via gateway cron (30-day retention for inactive sessions).
- **SessionConfig**: New config section with `inactivity_timeout_minutes` (default 30) and `chat_history_limit` (default 100).
- 19 new tests in `tests/test_session_management/` covering all features.

### Fixed
- **Chat history leak**: System messages (identity axioms, trust boundary, context pipeline output) and raw tool results (web search HTML) were displayed as chat bubbles when switching sessions. Now only `user` and `assistant` messages are persisted and returned.
- **Claude Code permission loop**: Planner asked "Ich brauche Freigabe für WebSearch" instead of generating JSON tool plans. Fixed with `_PLANNER_PREFIX` framing and SYSTEM_PROMPT restructuring.
- **Chat history limit**: Increased from hardcoded 20 to configurable 100 messages on session resume.

### Changed
- `save_chat_history()` now filters to `_VISIBLE_ROLES = {"user", "assistant"}` — system/tool messages excluded.
- `get_session_history()` SQL includes `AND role IN ('user', 'assistant')` for legacy data compat.
- `ClaudeCodeBackend._messages_to_prompt()` prepends planner-framing prefix.

### Testing
- Full suite: 11,447 tests passing (was 10,904)
- 19 new session management tests
- 0 regressions

## [0.50.0] -- 2026-03-21

### Added
- **97 End-to-End Scenario Tests**: Real user interaction simulations through full PGE pipeline (greeting, factual questions, file ops, code generation, memory, documents, web research, shell, conversation context, error handling, language/tone, skills, safety, performance, sentiment, channels)
- **WebSocket Token Streaming**: Real-time token-by-token response delivery + tool_start/tool_result events during PGE execution
- **Property-Based Testing** (Hypothesis): 650 fuzz cases for hash determinism, parse roundtrips, budget invariants, signature consistency

### Changed
- **Prompts completely rewritten** — SYSTEM_PROMPT 274→50 lines (-61%), character-first design, casual German tone
- REPLAN_PROMPT 50→15 lines (-70%), three clear options
- ESCALATION_PROMPT uses first person ("Ich wollte...")
- Prompt presets synced: English + Chinese translations match new style
- formulate_response() prompts condensed (7 REGELN → 1 sentence)
- Personality directives shorter and more natural
- Greetings: "Morgen!" / "" / "Hey, guten Abend!" / "Na, auch noch wach?"
- Sentiment messages: removed "HINWEIS:" prefix, shorter

### Fixed
- trace_optimizer: `get()` → `get_trace()`, `get_recent()` → `get_recent_traces()`
- Personality test assertions updated to match new wording
- Chat bubble light mode contrast (dark text on light background)
- Hashline read_file pre-caches only, doesn't change output format

### Testing
- 97 E2E scenario tests (all passing)
- 8 automated test methods: mypy, bandit, API contract, SQLite schemas, Hypothesis, stress test, dependency audit, config fuzzing
- 0 bugs found in automated testing
- Full suite: 5,500+ tests passing

## [0.49.0] -- 2026-03-21

### Added
- **Hashline Guard** — hash-anchored file edit system preventing race conditions and stale-line errors:
  - 11 new modules in `src/jarvis/hashline/` (~1,500 lines)
  - xxHash64-based line hashing with 2-char Base62 display tags
  - Format: `1#aK| import yaml` — compact, LLM-parseable
  - Hash validation before every edit (always checks disk, never just cache)
  - Thread-safe LRU cache (100 files, OrderedDict + RLock)
  - 4 edit operations: replace, insert_after, insert_before, delete
  - Atomic writes (tempfile + os.replace), preserves permissions/encoding/newline style
  - Auto-recovery on hash mismatch: reread + fuzzy line matching (±5 lines, difflib)
  - Append-only JSONL audit trail with SHA-256
  - Binary/encoding detection, file size limits, excluded/protected paths
  - 14 configurable parameters via `config.yaml` hashline section
  - 119 tests, all passing
- **Voice Mode wired** — mic button in chat input toggles VoiceProvider, transcriptions auto-send
- **Chat typing indicator** — "Thinking..." label with waveform, triggers during streaming start
- **Dashboard idle state** — gauges show "Idle" instead of "0%" when backend is idle
- **Agent Router live reload** — `reload_from_yaml()` after agent CRUD operations
- **Kubernetes Helm Chart** — complete chart under `deploy/helm/cognithor/` with Ollama sidecar, GPU support
- **Integration tests** — 9 new tests for SuperClaude + Chat History features
- **demo.svg** — new animated terminal SVG with PGE pipeline visualization

### Changed
- FLUTTER_API_CONTRACT.md updated to v0.48.0 (25 new endpoints documented)
- PWA Capacitor config points to Flutter web build
- `xxhash>=3.0` added as dependency

## [0.48.0] -- 2026-03-20

### Added
- **SuperClaude Integration (8 features)**:
  - Reflexion-Based Error Learning (`learning/reflexion.py`): JSONL error memory with root cause, prevention rules, recurrence tracking (35 tests)
  - Pre-Execution Confidence Check (`core/confidence.py`): 3-stage assessment (clarity/mistakes/context) in Gatekeeper (20 tests)
  - Four Questions Response Validator (`core/response_validator.py`): Anti-hallucination checks in formulate_response() (22 tests)
  - Token Budget Manager (`core/token_budget.py`): Complexity-based allocation with channel multipliers (24 tests)
  - Parallel Wave Context Pipeline: asyncio.gather for memory+vault+episodes (13 tests)
  - Self-Correction Prevention Rules: Auto-generated from GEPA trace analysis (17 tests)
  - Channel-Specific Behavioral Flags (`core/channel_flags.py`): 11 channel profiles (18 tests)
  - Post-Execution Pattern Documentation: Auto-captures successful tool sequences (8 tests)
- **Chat History** (like ChatGPT/Claude):
  - Session sidebar with past conversations, auto-titled from first message
  - Folder/Project system: organize chats into project folders
  - Rename, move to folder, delete via 3-dot context menu
  - Session switching with WebSocket reconnect
  - 5 new REST endpoints (list, history, create, delete, rename)
- **Skill Editor (Full CRUD)**:
  - Create, edit, delete skills from Flutter UI
  - Monospace body editor, trigger keywords as chips, category dropdown
  - Export as SKILL.md (agentskills.io format)
  - Built-in skills protected with lock banner
  - 7 backend API endpoints under /skill-registry/
- **Agent Editor (Full CRUD)**:
  - Create, edit, delete agent profiles
  - System prompt editor, searchable model picker dialog
  - Temperature slider, allowed/blocked tools, sandbox settings
  - Default "jarvis" agent protected from deletion
  - 4 backend API endpoints
- **Interactive Model Selection**: Tap configured models to change via searchable picker dialog
- **First-Run Setup Wizard**: 3-step onboarding (provider selection, config, connection test)

### Changed
- GEPA enabled by default (opt-out instead of opt-in)
- Desktop breakpoint: 1024px → 800px (sidebar stays expanded on smaller screens)
- Neon visual intensity doubled across all UI elements
- Config sidebar width 200→220, labels with ellipsis
- Channels page: compact toggle grid instead of full-width rows
- Robot Office PiP: 50% larger (420×270 / 700×450)
- Robot pathfinding around desks via corridor waypoints
- System monitor in Robot Office: CPU/GPU/RAM/LOAD bars
- Matrix rain 4x brighter (0.35 opacity, 40 columns)
- Identity auto-unfreezes on startup when Genesis Anchors exist
- Password eye toggle disabled when value is backend-masked (***)
- All 80+ hardcoded UI strings localized (EN/DE/ZH/AR)
- Provider error handling: partial error tracking, errors cleared only on success

### Fixed
- Chat messages disappearing (ChatProvider moved to app-level)
- BackdropFilter causing invisible content on Flutter web (NeonCard replaces GlassPanel in lists)
- Security/Models screens crashing (all unsafe type casts replaced)
- Skills not showing (API path conflict with marketplace catch-all route)
- Admin hub gray background (explicit scaffoldBackgroundColor)
- Monitoring screen 404 (hardcoded API paths → proper methods)
- discord_channel_id int→str coercion in config
- Python version check was empty function
- persistence.py row.get() on sqlite3.Row
- Vite-specific tests skipped when React UI not present
- WebSocket session switch race condition (300ms delay)

### Testing
- 157 new tests for SuperClaude features (all passing)
- 71 GEPA tests (all passing)
- Full suite: 5,063+ passed, 0 failed
- flutter analyze: "No issues found!"
- ruff format: 713 files conformant

## [0.47.1] -- 2026-03-19

### Added
- **English documentation suite**: Rewrote `QUICKSTART.md`, `FIRST_BOOT.md` in English; created `CONFIG_REFERENCE.md` (complete configuration reference with all 30+ config classes, every field documented), `DATABASE.md` (all 19+ SQLite databases with full schema), `FAQ.md` (35 frequently asked questions)
- **Flutter app README**: Replaced placeholder with proper documentation covering architecture, project structure, development workflow, and key files

### Changed
- All user-facing documentation now in English
- `FIRST_BOOT.md`: Updated "Jarvis" references to "Cognithor" throughout
- `CHANGELOG.md`: Added v0.47.0 and v0.47.1 entries

## [0.47.0] -- 2026-03-19

### Added
- **Flutter Web UI**: Complete cross-platform UI built with Flutter 3.41, replacing React+Preact. Features Sci-Fi Command Center aesthetic, chat with markdown/voice/hacker mode, Robot Office dashboard, 12 admin sub-screens, skills marketplace, identity management, i18n (EN/DE/ZH/AR)
- **15 LLM Backend Providers**: Added support for Ollama, OpenAI, Anthropic, Gemini, Groq, DeepSeek, Mistral, Together, OpenRouter, xAI, Cerebras, GitHub Models, AWS Bedrock, Hugging Face, Moonshot/Kimi, and LM Studio. Auto-detection from API keys, automatic model name adaptation per provider
- **Community Skill Marketplace**: Public skill registry with publisher verification, trust levels, recall checks, tool enforcement. 3 new MCP tools: `install_community_skill`, `search_community_skills`, `report_skill`
- **GEPA (Guided Evolution through Pattern Analysis)**: Execution trace recording, optimization proposals, auto-rollback on performance regression
- **Prompt Evolution**: A/B-test-based prompt optimization with statistical significance testing
- **Identity Layer (Immortal Mind Protocol)**: Cognitive identity with checkpoints, narrative self-reflection, reality checks, optional blockchain anchoring
- **Cost Tracking**: Per-request LLM cost tracking with daily/monthly budget limits
- **Durable Message Queue**: SQLite-backed message queue with priority boost, TTL, and retry logic
- **53 MCP tools** across 10 modules (filesystem, shell, web, media, memory, vault, synthesis, code, skills, browser)
- **16 communication channels**: CLI, WebUI, Telegram, Discord, Slack, WhatsApp, Signal, Matrix, Teams, Google Chat, Mattermost, Feishu, IRC, Twitch, iMessage, Voice

### Changed
- Version bumped to 0.47.0
- Config system expanded to 30+ Pydantic config classes with full validation
- Test suite: 10,800+ tests

## [0.33.0-beta] – 2026-03-11

### Added
- **i18n Language Pack System**: JSON-based internationalization with dot-notation keys (`t("error.timeout")`), SHA-256 integrity verification, thread-safe locale switching, fallback chain (locale → EN → raw key). Ships with German and English packs (~250 keys each). New module: `src/jarvis/i18n/`
- **Language Switcher in UI**: Control Center header quick-toggle button (DE/EN) + General page dropdown. Language changes are live via `reload_components(config=True)` — no restart needed
- **Locales API Endpoint**: `GET /api/v1/locales` returns available language packs and active locale
- **English locale tests**: New test class `TestEnglishLocale` validates error messages in both locales

### Fixed
- **Planner JSON Parse Retry** (critical): When the LLM returns malformed JSON, the planner now detects the failure (via `parse_failed` flag on `ActionPlan`), automatically retries with a format hint and lower temperature, and provides a clear error message to the user if both attempts fail. Previously, malformed JSON was silently converted to a direct response ("task failed successfully")
- **LLM Timeout Wiring**: `embed()` and `embed_batch()` in `model_router.py` now use `self._timeout` from config instead of hardcoded 30s/60s. LLM timeout field added to Executor page in UI (visible for all backends, not just Ollama)
- **WebSocket Race Condition** (critical): New `_ws_safe_send()` helper wraps all 12 `websocket.send_json()` calls in `__main__.py`. Returns `False` on disconnection errors, breaking the message loop cleanly instead of crashing with "Cannot call 'send'"
- **GlobalSearch "Einstellung suchen"** (broken since v29): Added 3 missing pages to `FIELD_INDEX` and `PAGE_LABELS` in `GlobalSearch.jsx`: Executor, Workflows, Knowledge Graph. Search now finds all 19 config pages
- **Pre-existing `_validate_url` async bug**: Fixed 14 test methods in `test_web.py` and `test_web_coverage.py` that called `async _validate_url()` without `await`
- **German umlaut encoding in de.json**: Replaced ASCII-safe substitutions (ue/oe/ae) with proper UTF-8 characters (ü/ö/ä) across ~80 occurrences

### Changed
- `ActionPlan` model: New `parse_failed: bool` field (default `False`)
- `config_manager.py`: Added `language` to `_EDITABLE_TOP_LEVEL` set
- `gateway.py`: `reload_components(config=True)` now calls `set_locale()` for live language switching
- `conftest.py`: Global `set_locale("de")` autouse fixture for test backwards compatibility
- Test count: 10,165 → **10,208** (43 new tests)

## [0.30.0] – 2026-03-08

### Added
- **Dokument-Lese-Tools**: 3 neue MCP-Tools (`read_pdf`, `read_ppt`, `read_docx`) fuer strukturiertes Lesen von PDF, PowerPoint und Word-Dokumenten mit Formatierung, Tabellen, Bilderextraktion und Metadaten
- **mTLS fuer WebUI-API**: Mutual TLS mit automatischer CA/Server/Client-Zertifikatsgenerierung; Malware kann sich nicht mehr als Frontend ausgeben (`security.mtls.enabled`)
- **DB Retry-Logik**: SQLite-Backend wiederholt bei "database is locked" automatisch mit exponentiellem Backoff und Jitter (konfigurierbar via `database.sqlite_max_retries`)
- **PPTX-Textextraktion**: `media_extract_text` unterstuetzt jetzt auch `.pptx`-Dateien

### Changed
- MCP-Tool-Anzahl: 48 → **51** (3 neue Dokument-Lese-Tools)
- Dependencies: `pymupdf>=1.23` und `python-pptx>=0.6` in `[documents]` Extras

## [0.29.1] – 2026-03-08

### Fixed
- **CI sandbox test on Windows**: Assertion now accepts `container`/`timeout` keywords in stderr (not just `docker`), fixing false failure on GitHub Actions Windows runners without Docker
- **Encryption dependency**: Changed `sqlcipher3-binary` (non-existent on PyPI) to `pysqlcipher3==1.2.0` — the only cross-platform SQLCipher binding that works on Linux and Windows
- **Encryption import**: Updated `open_sqlite()` to import from `pysqlcipher3.dbapi2` instead of `sqlcipher3`
- **Install safety**: Removed `encryption` from `[all]` extras to prevent install failures for users without native SQLCipher build dependencies; encryption remains available via `pip install cognithor[encryption]` or `cognithor[full]`

## [0.29.0] – 2026-03-08

### Fixed
- **UI layout wiggle**: Added `scrollbar-gutter: stable` to prevent horizontal content shift when scrollbar appears/disappears on page navigation
- **Unsaved changes false positives**: Snapshot now captured directly from fetched API data in `loadAllConfig()`, eliminating React batching race condition; removed redundant SPA navigation guard (page switch preserves state)
- **Keyboard shortcuts inconsistent**: Sequential `Cmd+1`..`Cmd+0` mapping for first 10 pages; Executor now accessible via `Cmd+6`; key lookup by field instead of array index
- **token_estimate always 0**: `WorkingMemory.add_message()` now updates `token_count` with word-based token estimation (compound-aware); `clear_for_compaction()` recalculates after pruning
- **SkillTester subprocess fails**: Safe environment now includes `PYTHONPATH`, `APPDATA`, and `VIRTUAL_ENV` so pytest subprocess can find installed packages

### Added
- **SQLite encryption (optional)**: SQLCipher support with OS keyring key storage (`pip install cognithor[encryption]`); new `database.encryption_enabled` toggle in UI; `src/jarvis/db/encryption.py` module with `open_sqlite()`, `init_encryption()`, `get_encryption_key()`, `remove_encryption_key()`

### Removed
- **Speed field from Models UI**: Was a metadata-only field with no runtime effect; removed to avoid user confusion

## [0.28.0] – 2026-03-08

### Fixed
- **Vite dev server unreachable**: Explicit `host: '127.0.0.1'` binding prevents IPv6/IPv4 mismatch on newer Node.js versions where `localhost` resolves to `::1`
- **Deprecated `locale.getdefaultlocale()`**: Replaced with `getlocale()` in bootstrap to fix Python 3.13+ deprecation warning (removal in 3.15)

### Changed
- **Coder model updated**: `qwen3-coder:32b` (non-existent) → `qwen3-coder:30b` (official Qwen3-Coder MoE, 18 GB) across all configs, docs, and bootstrap tiers

## [0.27.5] "BugHunt" – 2026-03-08

### CodeQL Security Sweep & CI Stability

Systematic elimination of all GitHub CodeQL security alerts (60+), cross-platform CI stability fixes, and thread-safety hardening. Test suite expanded to 10,165 tests.

### Fixed

- **CWE-209 Information Exposure** — 60+ instances of `str(exc)` in API responses replaced with generic error messages + server-side logging across `config_routes.py`, `teams.py`, `__main__.py`
- **CWE-22 Path Traversal** — All user-supplied paths validated with `os.path.normpath()` + `startswith()` (CodeQL-recognized pattern) in `__main__.py` (voice models, downloads) and `sanitizer.py`
- **CWE-1333 ReDoS** — Simplified SemVer regex pre-release part in `validator.py` to eliminate exponential backtracking
- **CWE-312 Cleartext Storage** — Renamed `known_secret` to `known_key_data` in test to avoid false positive
- **Workflow Permissions** — Added `permissions: contents: read` to `ci.yml` and `publish.yml` for least-privilege CI
- **Windows CI** — Removed `| head -60` pipe (unavailable in PowerShell), fixed `\a` path escape in `test_production_readiness.py`
- **aiohttp Mock Leakage** — Scoped aiohttp mocks in `test_teams.py` with `patch.dict` to prevent polluting `test_telegram_webhook.py`
- **nio Mock Consistency** — `test_matrix.py` now uses `MagicMock(return_value=...)` so shared mock client has `add_event_callback`
- **Checkpoint Ordering** — Added monotonic `_seq` counter to `Checkpoint` as tiebreaker for same-timestamp sorting
- **EpisodicStore Thread Safety** — All SQLite read methods now serialized with `_write_lock` to prevent corruption under concurrent multi-thread access
- **URL Exact Match** — Groq/DeepSeek URL checks in tests changed from `startswith()` to exact `==` match to satisfy CodeQL

### Changed

- Version bumped to 0.27.5 "BugHunt"
- GitHub Stars badge added to README and docs (dynamic, shields.io)
- Test count updated: 10,165 tests, ~118,000 LOC source, ~108,000 LOC tests

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
