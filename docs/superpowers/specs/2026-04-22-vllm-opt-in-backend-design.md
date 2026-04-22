# vLLM as Opt-In Backend (Windows-functional) — Design Spec

**Status:** Brainstorming approved 2026-04-22, ready for implementation plan.

**Goal:** Add vLLM as a first-class opt-in LLM backend alongside the existing Ollama default. Users with an NVIDIA GPU (≥16 GB VRAM) can install, start, stop, and switch to vLLM entirely from the Flutter UI, without leaving Cognithor. Ollama remains the default — vLLM is purely additive.

**Non-goal for this spec:** Replacing Ollama. Video-input end-to-end (unlocked by vLLM but tracked separately in `project_video_input_deferred.md`). Linux/macOS installer integration (vLLM runs fine there natively; users who install via `pip` and start manually are handled via the Custom-HF-repo path, but no dedicated wizard).

---

## Approved Decisions (Brainstorming Outcomes)

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | **Scope:** Flutter in-app wizard with full container lifecycle (Option B, not point-at-existing) | "Funktional anbieten" requires install + start + stop, not just a config field |
| 2 | **Install tech:** Docker Desktop + official vLLM image | Cleanest lifecycle, officially supported, fast image pull vs. hour-long `pip install vllm` |
| 3 | **Docker bootstrap:** Link + detect (Option B) — we don't install Docker Desktop automatically | Docker install needs reboot; silent third-party install is a trust issue |
| 4 | **Hardware gate:** Strict — NVIDIA vendor + ≥16 GB VRAM (fits the smallest curated VLM, Qwen2.5-VL-7B at FP16) checked before vLLM is selectable | 30-min setup ending in OOM is worse UX than an upfront "not supported" message; override via config flag. Passing the gate only means "can run the smallest curated model." The UI shows a per-model VRAM badge and disables models whose `vram_gb_min` exceeds detected VRAM |
| 5 | **Model scope:** Hybrid curated + custom (Option Y+Z) | Curated dropdown with tested models, last entry is "Custom (HF repo id…)" text field with disclaimer |
| 6 | **Flutter UX:** Dedicated sub-screen "LLM Backends" (Option C) | Room for status, metrics, model management per backend; extensible for future backends |
| 7 | **Setup-page layout:** Status cards on one page (Option B) | All prerequisites visible at once, each card has its own action button, recoverable after partial failure |
| 8 | **Fail-mode:** Banner + situational fallback (Option Z) | Text-requests fall back to Ollama silently-with-banner; vision-requests hard-error because Ollama can't do vision |
| 9 | **Container lifecycle:** Smart with toggle (Option Z) | Default: stop on app close (no VRAM leak). Toggle: "keep running" for power users. Always: reuse existing container on app start if present |

---

## Architecture

vLLM integrates as a second local backend alongside Ollama via the existing `LLMBackend` ABC in `src/cognithor/core/llm_backend.py`. Both share the `UnifiedLLMClient` dispatch layer. No changes to Planner/Gatekeeper/Executor — they already go through `UnifiedLLMClient.chat()`.

Three new modules:

- **`VLLMBackend(LLMBackend)`** — protocol adapter. Talks OpenAI-compatible `/v1/chat/completions` via `httpx.AsyncClient`. Handles VLM image-payload conversion.
- **`vllm_orchestrator.py`** — stateful lifecycle manager. Wraps `docker`/`nvidia-smi` CLIs via `subprocess`. No Docker-SDK dependency.
- **Flutter `LlmBackendsScreen` + `VllmSetupScreen`** — the user-facing opt-in surface.

Backend switching is live — the existing `gateway.py:1968+` re-init path for `UnifiedLLMClient` already handles `llm_backend_type` changes and just needs a FastAPI endpoint to trigger it.

---

## Components

### Backend Layer (`src/cognithor/core/`)

**`VLLMBackend(LLMBackend)` — ~250 LOC, template: `OpenAIBackend`**
- `chat()`, `chat_stream()`, `embed()`, `is_available()`, `list_models()`, `close()`
- VLM-aware image-payload conversion: `images: list[str]` path-arg → OpenAI-vision format `{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}`
- `httpx.AsyncClient` with connection pooling, configurable timeout (default 60s)

**`vllm_orchestrator.py` — ~400 LOC, no new deps**
- State: `VLLMState` dataclass — `hardware_ok`, `docker_ok`, `image_pulled`, `container_running`, `current_model`, `last_error`
- Methods:
  - `check_hardware() -> HardwareInfo` — parses `nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits`
  - `check_docker() -> DockerInfo` — `docker version --format json`
  - `pull_image(tag, progress_callback) -> None` — streams `docker pull --progress=auto` stdout, parses layer-progress JSON. Typical image size ~10.5 GB (vllm/vllm-openai includes CUDA runtime + torch + vllm wheels).
  - `start_container(model, port=8000) -> ContainerInfo` — constructs `docker run` with `--gpus all`, `-v cognithor-hf-cache:/root/.cache/huggingface`, `-e HF_TOKEN=$token`, label `cognithor.managed=true`; auto-falls-back 8000→8009 on port conflict
  - `stop_container() -> None` — `docker stop` + `docker rm` on labeled container
  - `reuse_existing() -> Optional[ContainerInfo]` — scans `docker ps --filter "label=cognithor.managed=true"`
  - `status() -> VLLMState` — aggregates all above
- Ring-buffer last 500 lines of container stdout/stderr for diagnostics

### API Layer (`src/cognithor/channels/api.py` — existing FastAPI)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/backends` | GET | List all backends with status |
| `/api/backends/vllm/status` | GET | `VLLMState` as JSON |
| `/api/backends/vllm/check-hardware` | POST | Trigger hardware detection |
| `/api/backends/vllm/pull-image` | POST | SSE-stream (FastAPI `StreamingResponse` with `text/event-stream`) for pull progress. Flutter consumes via `EventSource`-equivalent. Line format: `data: {"layer":"sha256:...","current":1234567,"total":10000000}\n\n` |
| `/api/backends/vllm/start` | POST | Body: `{model: str}` — starts container |
| `/api/backends/vllm/stop` | POST | Stops container |
| `/api/backends/vllm/logs` | GET | Ring-buffer of last container logs |
| `/api/backends/active` | POST | Body: `{backend: "vllm"}` — triggers `UnifiedLLMClient` re-init |

### Flutter Layer (`flutter_app/lib/screens/`)

- **`llm_backends_screen.dart`** — list view of all backends with status dots (Settings → LLM Backends)
- **`vllm_setup_screen.dart`** — status-card page: Hardware · Docker · Image · Model. Each card has its own action button when the step is pending.
- **`llm_backend_provider.dart`** — `ChangeNotifier` with 2-second polling of `/api/backends/vllm/status` while the detail page is mounted. No polling from the list view.

### Config Layer (`src/cognithor/config.py`)

New Pydantic sub-model:

```python
class VLLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    model: str = "Qwen/Qwen2.5-VL-7B-Instruct"  # see model-registry note below
    docker_image: str = "vllm/vllm-openai:v0.19.1"  # stable at brainstorm time
    port: int = 8000
    auto_stop_on_close: bool = False  # default: stop on close; user opts in to persistent
    skip_hardware_check: bool = False  # override for edge cases (wrong detection, smaller models)
    request_timeout_seconds: int = 60
    # HF token is NOT stored here — it's read from the top-level
    # `config.huggingface_api_key` which the existing SecretStore keyring-backs
    # via `_SECRET_FIELDS` in config.py. The orchestrator passes it to the
    # container as `-e HF_TOKEN=$value`.
```

Embedded in existing `LLMBackendsConfig`. Loaded via the existing legacy-key-tolerant `load_config()` path. **HF-token handling reuses the existing `config.huggingface_api_key` field (already in `_SECRET_FIELDS`, keyring-backed via `SecretStore`) rather than introducing a new nested secret.**

### Model Registry (`src/cognithor/cli/model_registry.json`)

New provider section. VRAM values account for model weights at the given quantization **plus** KV cache + vLLM scheduler overhead (~1.3× weight size for typical context lengths):

```json
"vllm": {
  "description": "Vision-language models tested against vLLM on consumer NVIDIA GPUs. Focus is on vision — text-only models are supported via the Custom path.",
  "models": [
    {"id": "Qwen/Qwen2.5-VL-7B-Instruct", "vram_gb_min": 16, "capability": "vision", "tested": true,  "min_vllm_version": "0.7.0",  "quantization": "bf16", "notes": "Default. Well-supported in vLLM, fits on a 16 GB card."},
    {"id": "QuantTrio/Qwen3.6-35B-A3B-AWQ", "vram_gb_min": 24, "capability": "vision", "tested": false, "min_vllm_version": "pending", "quantization": "AWQ-4bit", "notes": "Community 4-bit AWQ. Awaiting vLLM Qwen3.6 architecture support."},
    {"id": "Qwen/Qwen3.6-27B-FP8", "vram_gb_min": 32, "capability": "vision", "tested": false, "min_vllm_version": "pending", "quantization": "FP8", "notes": "Official FP8 weights. Awaiting vLLM Qwen3.6 architecture support."},
    {"id": "Qwen/Qwen3.6-35B-A3B-FP8", "vram_gb_min": 40, "capability": "vision", "tested": false, "min_vllm_version": "pending", "quantization": "FP8", "notes": "Official FP8 MoE. Awaiting vLLM Qwen3.6 architecture support."}
  ]
}
```

**vLLM-version compatibility**: vLLM v0.19.1 (current stable as of brainstorm) does **not** yet support the Qwen3.6 architecture — Qwen3.6 was released on 2026-04-21, three days after v0.19.1. Until a vLLM release adds `qwen3_6`/`qwen3_5_vl` architecture support (expected v0.19.2 or v0.20.0), Qwen3.6 entries are marked `tested: false` with `min_vllm_version: "pending"`. The Flutter dropdown surfaces them as disabled with a tooltip "Requires vLLM ≥X — pull image first". Users who want to bleed-edge can set `docker_image: "vllm/vllm-openai:latest"` or `:nightly` manually via config.

Flutter reads this list for the curated dropdown; the last option is always a "Custom (HF repo id…)" text field with a disclaimer. The registry's `min_vllm_version` is used at implementation time to gate curated entries against the resolved Docker image.

---

## Data Flows

### Flow A — First-Time Setup (Cold Start)

1. User opens Flutter → Settings → LLM Backends → clicks "vLLM"
2. `vllm_setup_screen.dart` mounts, `LlmBackendProvider` begins polling `GET /api/backends/vllm/status`
3. Backend runs `orchestrator.check_hardware()` + `check_docker()` synchronously → Cards 1 + 2 render immediately (✓ or ✗)
4. User clicks "Pull image now" on Card 3 → `POST /api/backends/vllm/pull-image` (SSE stream) → Flutter renders progress bar from layer events
5. After successful pull, Card 4 "Select & load model" enables → dropdown from `model_registry.json.providers.vllm.models` + Custom text field
6. User picks a model, clicks "Start vLLM" → `POST /api/backends/vllm/start {model}` → `orchestrator.start_container()` → waits for vLLM `/health` ping (timeout 120 s) → response
7. User clicks "Make active" → `POST /api/backends/active {backend:"vllm"}` → `UnifiedLLMClient` re-init → hot-switch without app restart

### Flow B — Chat Request with Vision

1. User types prompt + attaches image → Flutter sends via WebSocket to Gateway
2. Gateway → Planner → `working_memory.image_attachments = [path]`
3. Planner selects model: `config.vision_model_detail` (e.g., `Qwen/Qwen2.5-VL-7B-Instruct` — the curated default; `Qwen/Qwen3.6-27B-FP8` once vLLM ships Qwen3.6 support) → `unified_llm.chat(images=[path])`
4. `UnifiedLLMClient` dispatches to `VLLMBackend` (active backend)
5. `VLLMBackend.chat()` converts image paths to OpenAI-vision format (`data:image/png;base64,...`), POSTs to `http://localhost:8000/v1/chat/completions`
6. vLLM container processes, streams response back → Planner → Gateway → WebSocket → Flutter

### Flow C — Fail Flow (vLLM Offline Mid-Chat)

1. `VLLMBackend.chat()` raises `VLLMNotReadyError` (timeout or connection refused)
2. `UnifiedLLMClient` catches, marks `backend_status = DEGRADED`, notifies Gateway via event
3. Gateway sends WebSocket event `backend_status_changed` → Flutter renders banner "⚠ vLLM offline — fallback to Ollama active"
4. **If text-request** (`working_memory.image_attachments` is empty): `UnifiedLLMClient` transparent-fallback to `OllamaBackend` with the same prompt
5. **If image-request** (`working_memory.image_attachments` is non-empty — same trigger the Planner uses for vision routing): `VLLMNotReadyError` propagates as error bubble in chat: "vLLM offline — cannot process image". No silent fallback because Ollama cannot do vision
6. Next request: `VLLMBackend.is_available()` checked → success? → `backend_status = OK`, banner dismisses

### Flow D — App Close / Re-Open

1. Flutter app closes → Python backend receives SIGTERM from launcher
2. Shutdown hook in backend: `config.vllm.auto_stop_on_close == true` → `orchestrator.stop_container()`. Otherwise: do nothing.
3. Next app start: backend init calls `orchestrator.reuse_existing()` → if a container with label `cognithor.managed=true` is running, mark as `ready` directly, no restart sequence

---

## Error Handling

### Error Hierarchy (`src/cognithor/core/llm_backend.py`)

- `LLMBackendError` — base (exists)
- `VLLMNotReadyError` — container not running or model not loaded
- `VLLMHardwareError` — NVIDIA not detected, VRAM insufficient
- `DockerError` — Docker Desktop unreachable

All exceptions carry a `recovery_hint: str` field which Flutter renders alongside the error message.

### Setup-Time Errors (Orchestrator Level)

- `check_hardware()`: `nvidia-smi` returns empty → `VLLMHardwareError("NVIDIA GPU not detected")`, Card 1 stays red, other cards disabled
- `docker version` return code ≠ 0 → `DockerError("Docker Desktop not running")`, Card 2 shows "Start Docker Desktop" hint
- `docker pull` timeout (10 min default) → error with retry button; partial layers stay in cache
- `start_container()` port 8000 busy → automatic fallback 8001 … 8009. Beyond 8010 → error
- vLLM `/health` doesn't answer within 120 s → last 50 lines of container logs shown in error panel (`docker logs`)

### Runtime Errors (Request Level)

- `chat()` timeout 60 s default → configurable via `vllm.request_timeout_seconds`
- HTTP 5xx from vLLM → `VLLMNotReadyError`, triggers fail flow (Flow C)
- HTTP 400 (e.g. context too long) → `LLMBackendError` propagates directly, **no fallback** (real user error; Ollama wouldn't solve it either)
- Connection refused → same as 5xx → fail flow

### Circuit Breaker (reuses existing `cognithor.utils.circuit_breaker.CircuitBreaker`)

The repo already has a production-grade async circuit breaker with a full `CLOSED → OPEN → HALF_OPEN → CLOSED` state machine (`src/cognithor/utils/circuit_breaker.py`). We reuse it — do not reinvent.

**Integration:** `UnifiedLLMClient` owns one `CircuitBreaker` instance **per backend** (`vllm_breaker`, `ollama_breaker`, …). Per-backend scope prevents a flaky vLLM from tripping Ollama's breaker.

**Parameters** for the vLLM breaker:
- `name="llm_backend_vllm"`
- `failure_threshold=3` — three consecutive failures open the circuit
- `recovery_timeout=60.0` — 60 s in OPEN, then HALF_OPEN for a probe request
- `half_open_max_calls=1` — one probe at a time in HALF_OPEN
- `excluded_exceptions=(LLMBadRequestError,)` — HTTP 400 errors are user/context problems, **not** backend faults, so they never count toward the breaker threshold

**Behavior:**
- `CLOSED` (healthy): normal dispatch
- `OPEN` (after 3 fails): `CircuitBreakerOpen` raised immediately → `UnifiedLLMClient` treats this as DEGRADED, routes per the fail-flow rules (text → Ollama, image → error). Banner shows.
- `HALF_OPEN` (after recovery_timeout): next request goes through as a probe. Success → `CLOSED`, banner dismisses. Failure → back to `OPEN` with fresh 60 s countdown.

No separate health-check thread — the existing breaker heals itself via the HALF_OPEN probe on the next real request. Saves a timer thread + 30 s polling cost.

### Logging

- All orchestrator actions log structured: `{"component":"vllm_orchestrator","action":"start_container","model":"...","duration_ms":...,"outcome":"ok|error"}`
- Container stdout/stderr kept in ring buffer of last 500 lines (in memory), retrievable via `GET /api/backends/vllm/logs` → Flutter can show a "Show logs" button on the setup page when things stall.

### No Backwards-Compatibility Traps

- If vLLM config is absent or disabled → `VLLMBackend` is never instantiated, zero overhead
- Existing non-vLLM users notice nothing about this module

---

## Testing Strategy

### Constraints

CI runners have no GPU and no way to actually start vLLM. No Docker-in-Docker, no 8 GB image pull. All integration testing uses mocks or fakes.

### Unit Layer (GitHub Actions, free runners)

- **`tests/test_core/test_vllm_backend.py`** — `VLLMBackend` against `httpx_mock`
  - `chat()` formats OpenAI-compatible payload correctly
  - Image-payload conversion (path → `data:image/…` URL)
  - `chat_stream()` parses SSE chunks correctly
  - `is_available()` against 200-ok and connection-refused
  - Error propagation (5xx → `VLLMNotReadyError`, 400 → `LLMBackendError`)
- **`tests/test_core/test_vllm_orchestrator.py`** — orchestrator with `subprocess.run` mocked
  - `check_hardware()` parses `nvidia-smi` output correctly (real + empty + garbled)
  - `check_docker()` handles missing Docker gracefully
  - `pull_image()` parses Docker-progress JSON
  - `start_container()` constructs the `docker run` command correctly (including `--gpus all`, volume, label, HF token env)
  - `reuse_existing()` filters by label
  - Port-fallback logic (8000 busy → 8001)
- **`tests/test_core/test_unified_llm_circuit_breaker.py`** — circuit-breaker state machine
  - 3 fails in 60 s → `DEGRADED`
  - Health-ping recovered → `OK`
  - Fail-flow dispatch (text → Ollama, image → error)

### Integration Layer (GitHub Actions, free runners)

- **`tests/test_integration/test_vllm_fake_server.py`** — a mini FastAPI app that impersonates vLLM's OpenAI API (started in an `asyncio` fixture, not via Docker). `VLLMBackend` communicates end-to-end with it.
  - Sends real request, receives real response
  - Tests streaming, image payloads, error responses
  - No GPU, no Docker, runs in < 1 second

### Flutter Layer (`flutter test`, already in CI)

- **`test/widgets/llm_backends_screen_test.dart`** — widget test with `LlmBackendProvider` mock. Status cards render correctly for every state.
- **`test/widgets/vllm_setup_screen_test.dart`** — buttons trigger correct API calls (with `http_mock_adapter`).
- Goldens: optional, only for the status-card page (clear visual layout, worth it).

### Cross-Repo Guard

- **`tests/test_vllm_registry_sync.py`** — cross-check that `model_registry.json.providers.vllm.models` and Flutter's curated list (if mirrored as a Dart constant) stay in sync. Prevents drift on model-list updates. Same pattern as `test_flutter_version_sync.py`.

### Manual Smoke Tests

Documented in `docs/vllm-manual-test.md`. Run once on a dev machine with real NVIDIA GPU + Docker Desktop:

- Full setup flow (click through cards, pull image, start Qwen2.5-VL-7B-Instruct as the curated default)
- Chat with text + image
- App close with/without auto-stop toggle
- Manually stop vLLM container mid-session → verify fail flow

### Coverage Target

≥ 90 % on `vllm_backend.py` and `vllm_orchestrator.py` via unit + integration. Flutter coverage analogous to existing screens (~70 %).

---

## Dependencies & Prerequisites

**Python (in-repo):**
- No new packages — uses existing `httpx`, `pydantic`, `structlog`, stdlib `subprocess`
- `huggingface_hub` (already optional for the community-GGUF path from PR #132) is **not** required for vLLM itself — vLLM downloads HF models internally when it starts

**User environment (documented, not installed by us):**
- Docker Desktop (user installs manually per Decision 3)
- NVIDIA driver with CUDA runtime (any modern driver from the last 2 years works)
- NVIDIA GPU with ≥ 16 GB VRAM (enforced by the hardware gate)

**CI environment:** unchanged — no GPU runners, no Docker-in-Docker. All tests use mocks.

---

## Scope Boundaries

**In scope:**
- `VLLMBackend` class implementing `LLMBackend` ABC
- `vllm_orchestrator.py` for container lifecycle
- FastAPI endpoints for the Flutter UI, including SSE streaming for pull progress
- New Flutter screens (`LlmBackendsScreen`, `VllmSetupScreen`)
- Config extension (`VLLMConfig` Pydantic model)
- Model registry additions (curated VLMs + per-model vLLM-version gating)
- `UnifiedLLMClient` integration of the existing `cognithor.utils.circuit_breaker.CircuitBreaker` per backend
- Banner + situational fallback in the fail flow
- **User-facing documentation** at `docs/vllm-user-guide.md` — install prerequisites, enable-vLLM walkthrough, troubleshooting
- Manual smoke-test instructions at `docs/vllm-manual-test.md`
- Tests per the testing strategy above

**Out of scope (tracked separately):**
- Video-frame end-to-end support (unlocked by vLLM but needs its own prompt-engineering work) — stays in `project_video_input_deferred.md`
- Migration from Ollama to vLLM as *default* — Ollama stays default forever
- Embedding-endpoint parity with Ollama (vLLM's embedding support is model-specific; we accept that `embed()` on vLLM may only work for models that explicitly support it)
- Multi-GPU / tensor-parallel setups (single-GPU only; flag `--tensor-parallel-size` not exposed in v1)
- Windows-native vLLM builds (we commit to the Docker Desktop path only)

---

## Estimate

**1.5–2 calendar weeks, single engineer.** Breakdown:

- `VLLMBackend` class + unit tests: 1 day
- `vllm_orchestrator.py` + unit tests: 2 days
- FastAPI endpoints including SSE streaming for pull progress: 1 day (was underestimated at 0.5 day — SSE server-side + client-side `EventSource` in Flutter is not trivial)
- Flutter screens + widget tests + SSE progress consumption: 2 days (was underestimated at 1.5 days)
- Config extension + `VLLMConfig` + `huggingface_api_key` env-var wiring: 0.5 day
- Wiring existing `CircuitBreaker` into `UnifiedLLMClient` per-backend + tests: 0.5 day (lighter than "new circuit breaker" because code is reused)
- User-facing guide (`docs/vllm-user-guide.md`) + manual-test doc: 1 day
- Manual smoke test on real NVIDIA hardware + bug fixes from it: 1 day
- Buffer / polish / PR cycle / spec-reviewer loops: 1 day

**Total: ~10 working days = 2 calendar weeks** if everything lines up. Best-case 1.5 weeks if the SSE and Flutter widget work goes smoothly. Revised upward from the original 1-week estimate because of (a) SSE non-triviality on both ends, (b) user-docs budgeted explicitly, (c) real-hardware smoke test typically surfaces 1-2 bugs worth a day.

---

## Open Questions Deferred to Plan

- Exact default for `request_timeout_seconds` — needs one benchmark run on Qwen2.5-VL-7B first turn with a modest prompt + image on the target hardware
- Whether to ship a Dart-side constant mirror of `model_registry.json` or fetch at runtime from the backend — affects `test_vllm_registry_sync.py` but not user-visible behavior
- When Qwen3.6 architecture support lands in vLLM: do we bump the default `docker_image` pin, default the `model` to `Qwen/Qwen3.6-27B-FP8`, and retire Qwen2.5-VL-7B from the curated default? Probably yes, but needs one more release cycle to evaluate stability

## Resolved During Research (2026-04-22)

- ~~vLLM Docker image tag to pin~~ → `vllm/vllm-openai:v0.19.1` at brainstorm time, user-overridable via `VLLMConfig.docker_image`
- ~~HF-token storage~~ → reuse existing top-level `config.huggingface_api_key` (already keyring-backed via `SecretStore._SECRET_FIELDS`), no new nested secret field
- ~~Circuit breaker implementation~~ → reuse existing `cognithor.utils.circuit_breaker.CircuitBreaker`, per-backend instance, do not reinvent
- ~~Model-registry VRAM math~~ → original draft used bf16 sizes which fail on consumer hardware; curated list now uses FP8/AWQ quantized variants with realistic per-model VRAM minima
- ~~Qwen3.6 vLLM support status~~ → vLLM v0.19.1 does **not** yet support the Qwen3.6 architecture (Qwen3.6 released 3 days after v0.19.1). Curated Qwen3.6 entries ship with `tested: false` and `min_vllm_version: "pending"`. Default curated model is Qwen2.5-VL-7B until vLLM ships Qwen3.6 support
