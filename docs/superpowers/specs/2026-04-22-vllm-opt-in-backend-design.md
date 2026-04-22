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
| 4 | **Hardware gate:** Strict — NVIDIA vendor + ≥16 GB VRAM + detected CUDA compute capability. Each curated model specifies a `min_compute_capability` (SM version) and `vram_gb_min`. The UI disables entries whose requirements exceed the detected GPU | 30-min setup ending in OOM or "no kernel for this arch" is worse UX than upfront "not supported." Blackwell (SM 12.0, RTX 50xx) unlocks NVFP4; Ada (SM 8.9, RTX 40xx) unlocks FP8; Ampere (SM 8.0-8.6, RTX 30xx) falls back to AWQ/INT8. Override via `skip_hardware_check` config flag |
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

**`vllm_orchestrator.py` — ~450 LOC, no new deps**
- State: `VLLMState` dataclass — `hardware_ok`, `hardware_info` (GPU name, VRAM GB, compute capability as tuple like `(12, 0)`), `docker_ok`, `image_pulled`, `container_running`, `current_model`, `last_error`
- Methods:
  - `check_hardware() -> HardwareInfo` — parses `nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader,nounits`. Returns name, VRAM GB, and compute capability as `(major, minor)` tuple for model-compatibility filtering (Blackwell SM 12.0, Ada SM 8.9, Ampere SM 8.x)
  - `check_docker() -> DockerInfo` — `docker version --format json`
  - `pull_image(tag, progress_callback) -> None` — streams `docker pull --progress=auto` stdout, parses layer-progress JSON. Typical image size ~10.5 GB (vllm/vllm-openai includes CUDA runtime + torch + vllm wheels).
  - `start_container(model, port=8000, health_timeout=None) -> ContainerInfo` — constructs `docker run` with `--gpus all`, `-v cognithor-hf-cache:/root/.cache/huggingface`, `-e HF_TOKEN=$token`, label `cognithor.managed=true`; auto-falls-back 8000→8009 on port conflict. `health_timeout` defaults to **120 s** for models under 20 GB weights, **300 s** for larger models — first-load includes HF download + vLLM weight-mapping which is slow for big models
  - `stop_container() -> None` — `docker stop` + `docker rm` on labeled container
  - `reuse_existing() -> Optional[ContainerInfo]` — scans `docker ps --filter "label=cognithor.managed=true"`
  - `status() -> VLLMState` — aggregates all above
  - `recommend_model(hardware: HardwareInfo, registry: list[ModelEntry], prefer: Literal["vision","text"]="vision") -> ModelEntry` — returns the best curated model that fits the detected GPU. Ranking: (1) `tested==true` beats `tested==false`, (2) matching capability beats non-matching, (3) higher priority (`premium > standard > fallback`) within VRAM limits, (4) tie-break by lower VRAM-to-weights ratio (headroom for KV cache)
  - `filter_registry(hardware: HardwareInfo, registry: list[ModelEntry]) -> list[ModelEntry]` — returns all entries that pass the hardware gate, used by the Flutter dropdown to show enabled/disabled state
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
    model: str = ""  # empty → orchestrator.recommend_model() picks the best fit for detected hardware on first start
    docker_image: str = "vllm/vllm-openai:v0.19.1"  # stable at brainstorm time; set to ":nightly" for bleed-edge Qwen3.6
    port: int = 8000
    auto_stop_on_close: bool = False  # default: stop on close; user opts in to persistent
    skip_hardware_check: bool = False  # override for edge cases (wrong detection, smaller models, unlisted GPUs)
    request_timeout_seconds: int = 60
    # HF token is NOT stored here — it's read from the top-level
    # `config.huggingface_api_key` which the existing SecretStore keyring-backs
    # via `_SECRET_FIELDS` in config.py. The orchestrator passes it to the
    # container as `-e HF_TOKEN=$value`.
```

**Recommendation flow on empty `model`:** orchestrator calls `recommend_model(hardware_info, registry)` and caches the result back to `config.vllm.model` so subsequent starts are deterministic. User can override via the Flutter dropdown at any time.

Embedded in existing `LLMBackendsConfig`. Loaded via the existing legacy-key-tolerant `load_config()` path. **HF-token handling reuses the existing `config.huggingface_api_key` field (already in `_SECRET_FIELDS`, keyring-backed via `SecretStore`) rather than introducing a new nested secret.**

### Model Registry (`src/cognithor/cli/model_registry.json`)

New provider section. The registry is **per-quantization**, not per-base-model — different quants of the same base model become separate entries so the orchestrator can pick the best fit per detected GPU. VRAM values account for model weights **plus** KV cache + vLLM scheduler overhead (~1.3× weight size for typical context lengths):

```json
"vllm": {
  "description": "Vision-language models tested against vLLM. Entries include the quantization format, minimum CUDA compute capability and minimum vLLM version required to load the model.",
  "models": [
    {
      "id": "mmangkad/Qwen3.6-27B-NVFP4",
      "display_name": "Qwen3.6-27B · NVFP4 (Blackwell-native)",
      "base_model": "Qwen/Qwen3.6-27B",
      "quantization": "NVFP4",
      "vram_gb_min": 14,
      "min_compute_capability": "12.0",
      "min_vllm_version": "pending",
      "capability": "vision",
      "priority": "premium",
      "tested": false,
      "notes": "Fastest option on RTX 50xx / Blackwell. Uses native FP4 tensor cores (~2× FP8 throughput). Built with NVIDIA's ModelOpt toolkit. Recommended default for RTX 5090."
    },
    {
      "id": "Qwen/Qwen3.6-27B-FP8",
      "display_name": "Qwen3.6-27B · FP8 (official)",
      "base_model": "Qwen/Qwen3.6-27B",
      "quantization": "FP8",
      "vram_gb_min": 32,
      "min_compute_capability": "8.9",
      "min_vllm_version": "pending",
      "capability": "vision",
      "priority": "premium",
      "tested": false,
      "notes": "Official Qwen FP8 block-128 quantization. Near-identical quality to bf16. Needs RTX 4090 (24 GB) with tight KV budget, RTX 5090 (32 GB), or workstation GPU."
    },
    {
      "id": "cyankiwi/Qwen3.6-27B-AWQ-INT4",
      "display_name": "Qwen3.6-27B · AWQ-INT4 (community)",
      "base_model": "Qwen/Qwen3.6-27B",
      "quantization": "AWQ-INT4",
      "vram_gb_min": 16,
      "min_compute_capability": "8.0",
      "min_vllm_version": "pending",
      "capability": "vision",
      "priority": "standard",
      "tested": false,
      "notes": "Community 4-bit AWQ. Runs on RTX 3090, 4070 Ti Super, 4080 (16 GB cards). Quality trade ~3-5% vs. bf16."
    },
    {
      "id": "Qwen/Qwen3.6-35B-A3B-FP8",
      "display_name": "Qwen3.6-35B-A3B · FP8 (MoE, official)",
      "base_model": "Qwen/Qwen3.6-35B-A3B",
      "quantization": "FP8",
      "vram_gb_min": 40,
      "min_compute_capability": "8.9",
      "min_vllm_version": "pending",
      "capability": "vision",
      "priority": "standard",
      "tested": false,
      "notes": "MoE with 3B active params — very fast inference. Needs workstation GPU (A6000, RTX 6000 Ada)."
    },
    {
      "id": "Qwen/Qwen2.5-VL-7B-Instruct",
      "display_name": "Qwen2.5-VL-7B · BF16 (fallback)",
      "base_model": "Qwen/Qwen2.5-VL-7B-Instruct",
      "quantization": "bf16",
      "vram_gb_min": 16,
      "min_compute_capability": "7.5",
      "min_vllm_version": "0.7.0",
      "capability": "vision",
      "priority": "fallback",
      "tested": true,
      "notes": "Current default until vLLM ships Qwen3.6 architecture support. Well-tested, runs on any 16 GB NVIDIA GPU including RTX 3090 / 4060 Ti 16 GB."
    }
  ]
}
```

**Field semantics:**
- `priority`: `premium` > `standard` > `fallback`. Orchestrator's `recommend_model()` prefers higher priority within hardware limits. `premium` marks the expected-best choice for each GPU class.
- `min_compute_capability`: CUDA SM version as `"major.minor"`. `12.0` = Blackwell, `8.9` = Ada, `8.0` = Ampere-A100, `7.5` = Turing. The orchestrator converts `nvidia-smi`'s reported capability to a tuple and compares.
- `tested: true` means we've verified the model actually loads and produces sensible output on real hardware. `false` = included based on registry-level research but not yet smoke-tested.
- `min_vllm_version: "pending"` = Qwen3.6 architecture is not yet supported by vLLM stable (v0.19.1 as of 2026-04-22). Will be flipped to the actual version when support lands.

**Qwen3.6 architecture status (as of 2026-04-22):** vLLM v0.19.1 (released 2026-04-18) predates Qwen3.6 (released 2026-04-21). Qwen3.6 NVFP4/FP8/AWQ weights have already been uploaded to HF by Qwen and the community, but the `qwen3_5_vl` / `qwen3_6` architecture loader is not yet in vLLM's main branch. Curated Qwen3.6 entries stay `tested: false` until a vLLM release lands — expected v0.19.2 or v0.20.0. Users can set `docker_image: "vllm/vllm-openai:nightly"` to bleed-edge it now.

**Flutter behavior:**
- Dropdown groups entries by `base_model`, shows quant as the differentiator
- Orchestrator's `recommend_model()` output is rendered with a "⭐ Recommended for your GPU" badge
- Entries failing `min_compute_capability` or `vram_gb_min` render disabled with a tooltip showing *why* ("Requires Blackwell GPU" / "Needs 32 GB VRAM, you have 16 GB" / "Requires vLLM ≥ 0.20.0")
- The last dropdown slot is always "Custom (HF repo id…)" with a disclaimer about untested models

---

## Data Flows

### Flow A — First-Time Setup (Cold Start)

1. User opens Flutter → Settings → LLM Backends → clicks "vLLM"
2. `vllm_setup_screen.dart` mounts, `LlmBackendProvider` begins polling `GET /api/backends/vllm/status`
3. Backend runs `orchestrator.check_hardware()` + `check_docker()` synchronously → Cards 1 + 2 render immediately (✓ or ✗)
4. User clicks "Pull image now" on Card 3 → `POST /api/backends/vllm/pull-image` (SSE stream) → Flutter renders progress bar from layer events
5. After successful pull, Card 4 "Select & load model" enables → dropdown sourced from `orchestrator.filter_registry(hardware, registry)` (entries failing `min_compute_capability` or `vram_gb_min` render disabled with a "why" tooltip). The entry matching `orchestrator.recommend_model(hardware)` is marked with a "⭐ Recommended for your GPU" badge. Last dropdown slot is always "Custom (HF repo id…)"
6. User picks a model (or accepts the recommendation), clicks "Start vLLM" → `POST /api/backends/vllm/start {model}` → `orchestrator.start_container()` → waits for vLLM `/health` ping (timeout 120 s, bumped to 300 s for first-load of models ≥ 20 GB since HF download + vLLM weight loading is slow) → response
7. User clicks "Make active" → `POST /api/backends/active {backend:"vllm"}` → `UnifiedLLMClient` re-init → hot-switch without app restart

### Flow B — Chat Request with Vision

1. User types prompt + attaches image → Flutter sends via WebSocket to Gateway
2. Gateway → Planner → `working_memory.image_attachments = [path]`
3. Planner selects model: `config.vision_model_detail` (e.g., `Qwen/Qwen2.5-VL-7B-Instruct` — the curated default; `Qwen/Qwen3.6-27B-FP8` once vLLM ships Qwen3.6 support) → `unified_llm.chat(images=[path])`
4. `UnifiedLLMClient` dispatches to `VLLMBackend` (active backend)
5. `VLLMBackend.chat()` converts image paths to OpenAI-vision format (`data:image/png;base64,...`), POSTs to `http://localhost:8000/v1/chat/completions`
6. vLLM container processes, streams response back → Planner → Gateway → WebSocket → Flutter

### Flow C — Fail Flow (vLLM Offline Mid-Chat)

1. `VLLMBackend.chat()` (wrapped in the vLLM `CircuitBreaker.call()`) raises `VLLMNotReadyError` (timeout or connection refused). After 3 consecutive failures the breaker transitions `CLOSED → OPEN`
2. `UnifiedLLMClient` catches the `VLLMNotReadyError` / `CircuitBreakerOpen` and marks its own public-facing `backend_status = DEGRADED`, notifies Gateway via event
3. Gateway sends WebSocket event `backend_status_changed` → Flutter renders banner "⚠ vLLM offline — fallback to Ollama active"
4. **If text-request** (`working_memory.image_attachments` is empty): `UnifiedLLMClient` transparent-fallback to `OllamaBackend` with the same prompt
5. **If image-request** (`working_memory.image_attachments` is non-empty — same trigger the Planner uses for vision routing): the error propagates as a red bubble in chat: "vLLM offline — cannot process image". No silent fallback because Ollama cannot do vision
6. **Recovery is automatic via the breaker**, not via a separate `is_available()` poll: after `recovery_timeout` (60 s) the breaker enters `HALF_OPEN` and lets the next real request through as a probe. Probe success → breaker `CLOSED` → `UnifiedLLMClient` flips `backend_status = OK` → banner dismisses. Probe failure → breaker back to `OPEN`, banner stays

### Flow D — App Close / Re-Open

1. Flutter app closes → Python backend receives SIGTERM from launcher
2. Shutdown hook in backend: `config.vllm.auto_stop_on_close == true` → `orchestrator.stop_container()`. Otherwise: do nothing.
3. Next app start: backend init calls `orchestrator.reuse_existing()` → if a container with label `cognithor.managed=true` is running, mark as `ready` directly, no restart sequence

---

## Error Handling

### Error Hierarchy (`src/cognithor/core/llm_backend.py`)

- `LLMBackendError` — base (exists)
- `LLMBadRequestError(LLMBackendError)` — wraps HTTP 400 responses (context too long, malformed request, unsupported model field). **Marked as `excluded_exceptions` on the `CircuitBreaker`** so these never count toward opening the circuit — they're user/context problems, not backend faults
- `VLLMNotReadyError(LLMBackendError)` — container not running or model not loaded
- `VLLMHardwareError(LLMBackendError)` — NVIDIA not detected, VRAM insufficient
- `VLLMDockerError(LLMBackendError)` — Docker Desktop unreachable

All exceptions carry a `recovery_hint: str` field which Flutter renders alongside the error message.

### Setup-Time Errors (Orchestrator Level)

- `check_hardware()`: `nvidia-smi` returns empty → `VLLMHardwareError("NVIDIA GPU not detected")`, Card 1 stays red, other cards disabled
- `docker version` return code ≠ 0 → `VLLMDockerError("Docker Desktop not running")`, Card 2 shows "Start Docker Desktop" hint
- `docker pull` timeout (10 min default) → error with retry button; partial layers stay in cache
- `start_container()` port 8000 busy → automatic fallback 8001 … 8009. Beyond 8010 → error
- vLLM `/health` doesn't answer within `start_container(...)`'s `health_timeout` (120 s for models < 20 GB, 300 s for larger) → last 50 lines of container logs shown in error panel (`docker logs`)

### Runtime Errors (Request Level)

- `chat()` timeout 60 s default → configurable via `vllm.request_timeout_seconds`
- HTTP 5xx from vLLM → `VLLMNotReadyError`, triggers fail flow (Flow C) and counts toward the breaker
- HTTP 400 (e.g. context too long) → `LLMBadRequestError` propagates directly, **no fallback** (real user error; Ollama wouldn't solve it either). Excluded from breaker counting
- Connection refused → `VLLMNotReadyError` → fail flow

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

CI runners have no GPU and no way to actually start vLLM. No Docker-in-Docker, no ~10.5 GB image pull. All integration testing uses mocks or fakes.

### Unit Layer (GitHub Actions, free runners)

- **`tests/test_core/test_vllm_backend.py`** — `VLLMBackend` against `httpx_mock`
  - `chat()` formats OpenAI-compatible payload correctly
  - Image-payload conversion (path → `data:image/…` URL)
  - `chat_stream()` parses SSE chunks correctly
  - `is_available()` against 200-ok and connection-refused
  - Error propagation (5xx → `VLLMNotReadyError`, 400 → `LLMBadRequestError`)
- **`tests/test_core/test_vllm_orchestrator.py`** — orchestrator with `subprocess.run` mocked
  - `check_hardware()` parses `nvidia-smi` output correctly (real + empty + garbled), extracts compute capability tuple (tested matrix: Blackwell 12.0, Ada 8.9, Ampere 8.6, Turing 7.5)
  - `check_docker()` handles missing Docker gracefully
  - `pull_image()` parses Docker-progress JSON
  - `start_container()` constructs the `docker run` command correctly (including `--gpus all`, volume, label, HF token env)
  - `reuse_existing()` filters by label
  - Port-fallback logic (8000 busy → 8001)
- **`tests/test_core/test_vllm_recommend_model.py`** — model-recommendation logic against the full registry
  - Blackwell SM 12.0 + 32 GB → picks `mmangkad/Qwen3.6-27B-NVFP4` (premium, native FP4)
  - Ada SM 8.9 + 24 GB → picks `Qwen/Qwen3.6-27B-FP8` (premium, fits)
  - Ada SM 8.9 + 16 GB → picks `cyankiwi/Qwen3.6-27B-AWQ-INT4` (fits budget, drops to standard tier)
  - Ampere SM 8.0 + 24 GB → picks AWQ-INT4 (no FP8 tensor cores on Ampere, vLLM emulates badly)
  - Turing SM 7.5 + 16 GB → picks `Qwen2.5-VL-7B-Instruct` (fallback, only tested model that matches)
  - Override: when `prefer="text"` and no text model is curated, returns `None` → UI shows "No curated text model — use Custom path"
  - `filter_registry()` correctly disables entries whose `min_compute_capability` exceeds detected capability
- **`tests/test_core/test_unified_llm_circuit_breaker.py`** — `UnifiedLLMClient`'s integration of the existing `CircuitBreaker` per backend. Does NOT retest the breaker's state machine itself (that has its own tests in `tests/test_utils/test_circuit_breaker.py`), only the wiring:
  - 3 consecutive `VLLMNotReadyError` → breaker `OPEN` → `UnifiedLLMClient.backend_status = DEGRADED`
  - `LLMBadRequestError` thrown by `VLLMBackend.chat()` → breaker counter does NOT increment (excluded_exception), `backend_status` stays `OK`
  - Breaker `HALF_OPEN` probe succeeds → breaker `CLOSED` → `backend_status = OK`, banner dismisses
  - Fail-flow dispatch: with `DEGRADED` set, text-request auto-routes to `OllamaBackend`, image-request raises to the caller

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

**~2.2 calendar weeks (11 working days), single engineer.** Breakdown:

- `VLLMBackend` class + unit tests: 1 day
- `vllm_orchestrator.py` + unit tests (including `recommend_model()` + `filter_registry()` GPU-awareness): 2.5 days
- FastAPI endpoints including SSE streaming for pull progress: 1 day (was underestimated at 0.5 day — SSE server-side + client-side `EventSource` in Flutter is not trivial)
- Flutter screens + widget tests + SSE progress consumption + per-model enable/disable logic + "⭐ Recommended" badge rendering: 2.5 days
- Config extension + `VLLMConfig` + `huggingface_api_key` env-var wiring: 0.5 day
- Wiring existing `CircuitBreaker` into `UnifiedLLMClient` per-backend + tests: 0.5 day (lighter than "new circuit breaker" because code is reused)
- User-facing guide (`docs/vllm-user-guide.md`) + manual-test doc: 1 day
- Manual smoke test on real NVIDIA hardware + bug fixes from it: 1 day
- Buffer / polish / PR cycle / spec-reviewer loops: 1 day

**Total: ~11 working days = 2.2 calendar weeks** if everything lines up. Revised upward from the original 1-week estimate because of (a) SSE non-triviality on both ends, (b) user-docs budgeted explicitly, (c) real-hardware smoke test typically surfaces 1-2 bugs worth a day, (d) added `recommend_model()` + `filter_registry()` hardware-awareness (0.5 day orchestrator + 0.5 day Flutter).

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
- ~~Qwen3.6 vLLM support status~~ → vLLM v0.19.1 does **not** yet support the Qwen3.6 architecture (Qwen3.6 released 3 days after v0.19.1). Curated Qwen3.6 entries ship with `tested: false` and `min_vllm_version: "pending"`. Until support lands, orchestrator's `recommend_model()` falls back to Qwen2.5-VL-7B-Instruct (the only `tested: true` entry)
- ~~Hardware-aware model recommendation~~ → registry entries now include `min_compute_capability` (CUDA SM version) and `priority` tier. `orchestrator.recommend_model()` picks the best entry per detected GPU — Blackwell users get NVFP4 as default, Ada users get FP8, Ampere users get AWQ-INT4, Turing falls back to Qwen2.5-VL-7B. Flutter dropdown shows "⭐ Recommended for your GPU" badge on the top pick and disables entries failing the detected GPU's capabilities with actionable tooltips. Unblocks RTX 5090 users to hit native NVFP4 throughput (~2× FP8) on day one
