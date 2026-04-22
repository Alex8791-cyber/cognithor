# vLLM Opt-In Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add vLLM as a first-class opt-in LLM backend alongside Ollama, with full Flutter-driven container lifecycle, GPU-aware model recommendation, and circuit-breaker-guarded fail-flow. Ollama stays default — vLLM is purely additive.

**Architecture:** New `VLLMBackend` implements the existing `LLMBackend` ABC (template: `OpenAIBackend`). New `vllm_orchestrator.py` wraps `docker`/`nvidia-smi` subprocesses for lifecycle + hardware detection + model recommendation. `UnifiedLLMClient` gets per-backend `CircuitBreaker` wrapping. Flutter gets a dedicated `LlmBackendsScreen` + `VllmSetupScreen` with status-card UX and SSE progress streaming.

**Tech Stack:** Python 3.12, Pydantic v2, pytest-asyncio, FastAPI (existing), httpx, Flutter 3.41.4 with `ChangeNotifier` state management, Inno Setup + Docker Desktop (user-managed). No new Python dependencies.

**Spec:** `docs/superpowers/specs/2026-04-22-vllm-opt-in-backend-design.md` — read it before starting. All nine key decisions are locked.

---

## File Structure

**New Python files:**
- `src/cognithor/core/vllm_orchestrator.py` — ~450 LOC — hardware/docker/container lifecycle + recommendation
- `src/cognithor/core/vllm_backend.py` — ~250 LOC — LLMBackend adapter for vLLM's OpenAI API
- `tests/test_core/test_vllm_orchestrator.py` — orchestrator unit tests
- `tests/test_core/test_vllm_backend.py` — backend unit tests
- `tests/test_core/test_vllm_recommend_model.py` — recommendation-matrix tests
- `tests/test_core/test_unified_llm_circuit_breaker.py` — UnifiedLLMClient + CB wiring
- `tests/test_integration/test_vllm_fake_server.py` — fake vLLM-compatible server
- `tests/test_vllm_registry_sync.py` — cross-repo guard
- `docs/vllm-user-guide.md` — user-facing install + enable guide
- `docs/vllm-manual-test.md` — smoke-test recipe for real hardware

**New Flutter files:**
- `flutter_app/lib/providers/llm_backend_provider.dart` — state + 2s polling
- `flutter_app/lib/screens/llm_backends_screen.dart` — list view
- `flutter_app/lib/screens/vllm_setup_screen.dart` — status cards + SSE progress
- `flutter_app/test/widgets/llm_backends_screen_test.dart` — widget tests
- `flutter_app/test/widgets/vllm_setup_screen_test.dart` — widget tests

**Modified Python files:**
- `src/cognithor/core/llm_backend.py` — add `LLMBadRequestError`, `VLLMNotReadyError`, `VLLMHardwareError`, `VLLMDockerError`; extend `LLMBackendType` enum with `VLLM`
- `src/cognithor/core/unified_llm.py` — per-backend `CircuitBreaker`, fail-flow dispatch, `backend_status` notification
- `src/cognithor/config.py` — add `VLLMConfig` sub-model, embed on `CognithorConfig`
- `src/cognithor/cli/model_registry.json` — new `vllm` provider section
- `src/cognithor/channels/api.py` — 7 new `/api/backends/*` endpoints including SSE streaming
- `CHANGELOG.md` — feature note

---

## TDD Contract

Every task follows: write failing test → run (red) → implement → run (green) → commit. No implementation without a test first. Commit after every green.

**Test infrastructure already in place:** `tests/conftest.py` has shared fixtures. Use `pytest-asyncio`'s `asyncio_mode=auto` (already in `pyproject.toml`). Ruff + Ruff format are enforced in CI (`.github/workflows/ci.yml:ruff check src/ tests/` + `ruff format --check src/ tests/`).

**Before EVERY commit:**
```bash
python -m ruff check src/cognithor/core/vllm_orchestrator.py src/cognithor/core/vllm_backend.py tests/
python -m ruff format --check src/cognithor/core/vllm_orchestrator.py src/cognithor/core/vllm_backend.py tests/
```
Run them first, then commit. Remember PR #135 lesson — CI has a separate `ruff format --check` step.

---

## Task 1: Error Hierarchy Extensions

**Files:**
- Modify: `src/cognithor/core/llm_backend.py` (add 4 exception classes after the existing `LLMBackendError`)
- Test: `tests/test_core/test_llm_backend_errors.py` (new)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_core/test_llm_backend_errors.py
from __future__ import annotations

import pytest

from cognithor.core.llm_backend import (
    VLLMDockerError,
    LLMBackendError,
    LLMBadRequestError,
    VLLMHardwareError,
    VLLMNotReadyError,
)


class TestErrorHierarchy:
    def test_all_vllm_errors_inherit_from_llm_backend_error(self):
        assert issubclass(LLMBadRequestError, LLMBackendError)
        assert issubclass(VLLMNotReadyError, LLMBackendError)
        assert issubclass(VLLMHardwareError, LLMBackendError)
        assert issubclass(VLLMDockerError, LLMBackendError)

    def test_errors_carry_recovery_hint(self):
        err = VLLMNotReadyError("container down", recovery_hint="Run: docker start vllm")
        assert err.recovery_hint == "Run: docker start vllm"
        assert str(err) == "container down"

    def test_recovery_hint_defaults_to_empty(self):
        err = VLLMDockerError("Docker not found")
        assert err.recovery_hint == ""

    def test_status_code_preserved_from_base(self):
        err = LLMBadRequestError("context too long", status_code=400)
        assert err.status_code == 400
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_core/test_llm_backend_errors.py -v
```
Expected: `ImportError: cannot import name 'LLMBadRequestError' from 'cognithor.core.llm_backend'`

- [ ] **Step 3: Add the exception classes**

Add below the existing `LLMBackendError` class in `src/cognithor/core/llm_backend.py` (around line 80-85):

```python
class LLMBackendError(Exception):
    """Error communicating with the LLM backend."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        recovery_hint: str = "",
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.recovery_hint = recovery_hint


class LLMBadRequestError(LLMBackendError):
    """Wraps HTTP 400 responses — user/context problem, not a backend fault.

    Excluded from circuit-breaker failure counting via ``excluded_exceptions``
    when the breaker is wired in ``UnifiedLLMClient``.
    """


class VLLMNotReadyError(LLMBackendError):
    """vLLM container not running or model not loaded."""


class VLLMHardwareError(LLMBackendError):
    """NVIDIA GPU not detected, VRAM insufficient, or unsupported compute capability."""


class VLLMDockerError(LLMBackendError):
    """Docker Desktop unreachable or wrong version."""
```

(Modify the existing `LLMBackendError.__init__` to accept `recovery_hint` — that's the only change to the base class. All four new subclasses inherit without override.)

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_core/test_llm_backend_errors.py -v
```
Expected: `4 passed`

- [ ] **Step 5: Ruff + format + commit**

```bash
python -m ruff check src/cognithor/core/llm_backend.py tests/test_core/test_llm_backend_errors.py
python -m ruff format --check src/cognithor/core/llm_backend.py tests/test_core/test_llm_backend_errors.py
git add src/cognithor/core/llm_backend.py tests/test_core/test_llm_backend_errors.py
git commit -m "feat(llm): add LLMBadRequestError + VLLM/Docker error subclasses"
```

---

## Task 2: LLMBackendType Enum Extension

**Files:**
- Modify: `src/cognithor/core/llm_backend.py:38-46` (add `VLLM` member to `LLMBackendType` StrEnum)
- Test: `tests/test_core/test_llm_backend_errors.py` (add test to existing file)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_core/test_llm_backend_errors.py`:

```python
class TestBackendTypeEnum:
    def test_vllm_is_a_backend_type(self):
        from cognithor.core.llm_backend import LLMBackendType
        assert LLMBackendType.VLLM == "vllm"
        assert LLMBackendType.VLLM.value == "vllm"

    def test_vllm_value_matches_config_literal(self):
        """config.CognithorConfig.llm_backend_type accepts "vllm" — keep enum aligned."""
        from cognithor.core.llm_backend import LLMBackendType
        assert "vllm" in {t.value for t in LLMBackendType}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_core/test_llm_backend_errors.py::TestBackendTypeEnum -v
```
Expected: `AttributeError: VLLM`

- [ ] **Step 3: Add enum member**

In `src/cognithor/core/llm_backend.py`, modify the `LLMBackendType` StrEnum:

```python
class LLMBackendType(StrEnum):
    """Supported LLM backends."""

    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    LMSTUDIO = "lmstudio"
    CLAUDE_CODE = "claude-code"
    VLLM = "vllm"
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_core/test_llm_backend_errors.py -v
```
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
python -m ruff check src/cognithor/core/llm_backend.py tests/test_core/test_llm_backend_errors.py && python -m ruff format --check src/cognithor/core/llm_backend.py tests/test_core/test_llm_backend_errors.py
git add src/cognithor/core/llm_backend.py tests/test_core/test_llm_backend_errors.py
git commit -m "feat(llm): register VLLM in LLMBackendType enum"
```

---

## Task 3: VLLMConfig Pydantic Sub-Model

**Files:**
- Modify: `src/cognithor/config.py` (add `VLLMConfig` class; embed on `CognithorConfig`)
- Test: `tests/config/test_vllm_config.py` (new)

- [ ] **Step 1: Write the failing test**

```python
# tests/config/test_vllm_config.py
from __future__ import annotations

import pytest
from pydantic import ValidationError

from cognithor.config import CognithorConfig, VLLMConfig


class TestVLLMConfig:
    def test_defaults(self):
        c = VLLMConfig()
        assert c.enabled is False
        assert c.model == ""
        assert c.docker_image == "vllm/vllm-openai:v0.19.1"
        assert c.port == 8000
        assert c.auto_stop_on_close is False
        assert c.skip_hardware_check is False
        assert c.request_timeout_seconds == 60

    def test_rejects_unknown_fields(self):
        with pytest.raises(ValidationError):
            VLLMConfig(unknown_field=1)

    def test_cognithor_config_has_vllm_sub_model(self):
        c = CognithorConfig()
        assert hasattr(c, "vllm")
        assert isinstance(c.vllm, VLLMConfig)

    def test_vllm_override(self):
        c = CognithorConfig(vllm={"enabled": True, "port": 8042})
        assert c.vllm.enabled is True
        assert c.vllm.port == 8042
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/config/test_vllm_config.py -v
```
Expected: `ImportError: cannot import name 'VLLMConfig' from 'cognithor.config'`

- [ ] **Step 3: Add VLLMConfig class + embed on CognithorConfig**

In `src/cognithor/config.py`, add a new class BEFORE `class CognithorConfig(BaseModel):`:

```python
class VLLMConfig(BaseModel):
    """Configuration for the optional vLLM backend.

    HF token is NOT stored here — read from top-level
    ``config.huggingface_api_key`` which is keyring-backed via
    ``SecretStore._SECRET_FIELDS``. Orchestrator passes it to
    the container as ``-e HF_TOKEN=$value``.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(default=False)
    model: str = Field(default="")
    docker_image: str = Field(default="vllm/vllm-openai:v0.19.1")
    port: int = Field(default=8000, ge=1024, le=65535)
    auto_stop_on_close: bool = Field(default=False)
    skip_hardware_check: bool = Field(default=False)
    request_timeout_seconds: int = Field(default=60, ge=5, le=600)
```

Then add the field inside `CognithorConfig` (search for `llm_backend_type` and place near it):

```python
vllm: VLLMConfig = Field(default_factory=VLLMConfig)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/config/test_vllm_config.py -v
```
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
python -m ruff check src/cognithor/config.py tests/config/test_vllm_config.py && python -m ruff format --check src/cognithor/config.py tests/config/test_vllm_config.py
git add src/cognithor/config.py tests/config/test_vllm_config.py
git commit -m "feat(config): add VLLMConfig sub-model"
```

---

## Task 4: Model Registry — vLLM Provider Section

**Files:**
- Modify: `src/cognithor/cli/model_registry.json` (add `providers.vllm` section)
- Test: `tests/test_core/test_model_registry_vllm.py` (new)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_core/test_model_registry_vllm.py
from __future__ import annotations

import json
from pathlib import Path

REGISTRY_PATH = (
    Path(__file__).resolve().parents[2]
    / "src" / "cognithor" / "cli" / "model_registry.json"
)


class TestVLLMRegistrySection:
    def setup_method(self):
        with open(REGISTRY_PATH, encoding="utf-8") as f:
            self.registry = json.load(f)

    def test_vllm_provider_exists(self):
        assert "vllm" in self.registry["providers"]

    def test_vllm_has_curated_models(self):
        models = self.registry["providers"]["vllm"]["models"]
        assert len(models) >= 5

    def test_each_model_has_required_fields(self):
        models = self.registry["providers"]["vllm"]["models"]
        required = {
            "id", "display_name", "base_model", "quantization",
            "vram_gb_min", "min_compute_capability", "min_vllm_version",
            "capability", "priority", "tested", "notes",
        }
        for m in models:
            missing = required - set(m.keys())
            assert not missing, f"Model {m.get('id')} missing fields: {missing}"

    def test_priority_values_are_valid(self):
        models = self.registry["providers"]["vllm"]["models"]
        for m in models:
            assert m["priority"] in ("premium", "standard", "fallback")

    def test_compute_capability_is_parseable(self):
        models = self.registry["providers"]["vllm"]["models"]
        for m in models:
            parts = m["min_compute_capability"].split(".")
            assert len(parts) == 2
            assert int(parts[0]) >= 7
            assert int(parts[1]) >= 0

    def test_vram_is_positive_integer(self):
        models = self.registry["providers"]["vllm"]["models"]
        for m in models:
            assert isinstance(m["vram_gb_min"], int)
            assert m["vram_gb_min"] > 0

    def test_at_least_one_tested_model(self):
        models = self.registry["providers"]["vllm"]["models"]
        assert any(m["tested"] for m in models)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_core/test_model_registry_vllm.py -v
```
Expected: `KeyError: 'vllm'`

- [ ] **Step 3: Add vllm provider section to model_registry.json**

In `src/cognithor/cli/model_registry.json`, find the existing `providers` object and add the `vllm` section with all 5 curated model entries EXACTLY as specified in the spec file `docs/superpowers/specs/2026-04-22-vllm-opt-in-backend-design.md` under "Model Registry" (lines 111-181). The entries are:

1. `mmangkad/Qwen3.6-27B-NVFP4` — NVFP4, vram 14, cc 12.0, priority premium, tested false
2. `Qwen/Qwen3.6-27B-FP8` — FP8, vram 32, cc 8.9, priority premium, tested false
3. `cyankiwi/Qwen3.6-27B-AWQ-INT4` — AWQ-INT4, vram 16, cc 8.0, priority standard, tested false
4. `Qwen/Qwen3.6-35B-A3B-FP8` — FP8, vram 40, cc 8.9, priority standard, tested false
5. `Qwen/Qwen2.5-VL-7B-Instruct` — bf16, vram 16, cc 7.5, priority fallback, tested TRUE

Copy the full JSON blocks from the spec verbatim. Also bump the registry's top-level `"updated"` timestamp to `"2026-04-22"`.

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_core/test_model_registry_vllm.py -v
```
Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
python -m ruff check tests/test_core/test_model_registry_vllm.py && python -m ruff format --check tests/test_core/test_model_registry_vllm.py
git add src/cognithor/cli/model_registry.json tests/test_core/test_model_registry_vllm.py
git commit -m "feat(registry): add vLLM provider section with 5 curated model entries"
```

---

## Task 5: Orchestrator Dataclasses + Module Skeleton

**Files:**
- Create: `src/cognithor/core/vllm_orchestrator.py` (dataclasses only; methods added in later tasks)
- Test: `tests/test_core/test_vllm_orchestrator.py` (new, initial dataclass tests)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_core/test_vllm_orchestrator.py
from __future__ import annotations

import pytest

from cognithor.core.vllm_orchestrator import (
    ContainerInfo,
    DockerInfo,
    HardwareInfo,
    ModelEntry,
    VLLMOrchestrator,
    VLLMState,
)


class TestDataclasses:
    def test_hardware_info_fields(self):
        h = HardwareInfo(gpu_name="RTX 5090", vram_gb=32, compute_capability=(12, 0))
        assert h.gpu_name == "RTX 5090"
        assert h.vram_gb == 32
        assert h.compute_capability == (12, 0)

    def test_hardware_info_sm_as_string(self):
        h = HardwareInfo(gpu_name="RTX 4090", vram_gb=24, compute_capability=(8, 9))
        assert h.sm_string == "8.9"

    def test_docker_info_fields(self):
        d = DockerInfo(available=True, version="26.0.0", server_running=True)
        assert d.available is True
        assert d.version == "26.0.0"

    def test_model_entry_from_dict(self):
        m = ModelEntry.from_dict({
            "id": "mmangkad/Qwen3.6-27B-NVFP4",
            "display_name": "Qwen3.6-27B · NVFP4",
            "base_model": "Qwen/Qwen3.6-27B",
            "quantization": "NVFP4",
            "vram_gb_min": 14,
            "min_compute_capability": "12.0",
            "min_vllm_version": "pending",
            "capability": "vision",
            "priority": "premium",
            "tested": False,
            "notes": "",
        })
        assert m.id == "mmangkad/Qwen3.6-27B-NVFP4"
        assert m.min_cc_tuple == (12, 0)
        assert m.vram_gb_min == 14
        assert m.priority == "premium"

    def test_vllm_state_initial(self):
        s = VLLMState()
        assert s.hardware_ok is False
        assert s.docker_ok is False
        assert s.container_running is False
        assert s.current_model is None
        assert s.hardware_info is None

    def test_container_info(self):
        c = ContainerInfo(container_id="abc123", port=8000, model="Qwen/Qwen3.6-27B-FP8")
        assert c.container_id == "abc123"
        assert c.port == 8000


class TestOrchestratorInit:
    def test_orchestrator_constructs_with_config(self):
        orch = VLLMOrchestrator(
            docker_image="vllm/vllm-openai:v0.19.1",
            port=8000,
            hf_token="hf_test",
        )
        assert orch.docker_image == "vllm/vllm-openai:v0.19.1"
        assert orch.port == 8000
        assert orch._hf_token == "hf_test"
        assert orch.state.hardware_ok is False
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_core/test_vllm_orchestrator.py -v
```
Expected: `ModuleNotFoundError: No module named 'cognithor.core.vllm_orchestrator'`

- [ ] **Step 3: Create vllm_orchestrator.py skeleton**

Create `src/cognithor/core/vllm_orchestrator.py`:

```python
"""vLLM lifecycle orchestrator — wraps docker/nvidia-smi subprocesses.

Stateful manager: hardware detection, Docker readiness, image pull,
container start/stop, model recommendation. No Docker-SDK dependency —
pure `subprocess` calls.

See spec: docs/superpowers/specs/2026-04-22-vllm-opt-in-backend-design.md
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Any, Literal

from cognithor.utils.logging import get_logger

log = get_logger(__name__)

Priority = Literal["premium", "standard", "fallback"]
Capability = Literal["vision", "text"]


@dataclass
class HardwareInfo:
    """NVIDIA GPU detection result."""

    gpu_name: str
    vram_gb: int
    compute_capability: tuple[int, int]

    @property
    def sm_string(self) -> str:
        """Returns the compute capability as 'major.minor' string."""
        return f"{self.compute_capability[0]}.{self.compute_capability[1]}"


@dataclass
class DockerInfo:
    """Docker Desktop readiness."""

    available: bool
    version: str = ""
    server_running: bool = False


@dataclass
class ContainerInfo:
    """A running/started vLLM container."""

    container_id: str
    port: int
    model: str


@dataclass
class ModelEntry:
    """One row from the model_registry.json vllm provider section."""

    id: str
    display_name: str
    base_model: str
    quantization: str
    vram_gb_min: int
    min_compute_capability: str  # "12.0" / "8.9" etc.
    min_vllm_version: str
    capability: Capability
    priority: Priority
    tested: bool
    notes: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelEntry:
        return cls(
            id=data["id"],
            display_name=data["display_name"],
            base_model=data["base_model"],
            quantization=data["quantization"],
            vram_gb_min=int(data["vram_gb_min"]),
            min_compute_capability=data["min_compute_capability"],
            min_vllm_version=data["min_vllm_version"],
            capability=data["capability"],
            priority=data["priority"],
            tested=bool(data["tested"]),
            notes=data.get("notes", ""),
        )

    @property
    def min_cc_tuple(self) -> tuple[int, int]:
        """Returns min_compute_capability as (major, minor) tuple."""
        parts = self.min_compute_capability.split(".")
        return (int(parts[0]), int(parts[1]))


@dataclass
class VLLMState:
    """Aggregate state snapshot for UI rendering."""

    hardware_ok: bool = False
    hardware_info: HardwareInfo | None = None
    docker_ok: bool = False
    docker_info: DockerInfo | None = None
    image_pulled: bool = False
    container_running: bool = False
    current_model: str | None = None
    last_error: str | None = None


class VLLMOrchestrator:
    """Stateful vLLM lifecycle manager. Methods added in later tasks."""

    def __init__(
        self,
        *,
        docker_image: str = "vllm/vllm-openai:v0.19.1",
        port: int = 8000,
        hf_token: str = "",
        log_ring_size: int = 500,
    ) -> None:
        self.docker_image = docker_image
        self.port = port
        self._hf_token = hf_token
        self.state = VLLMState()
        self._log_ring: collections.deque[str] = collections.deque(maxlen=log_ring_size)

    def get_logs(self) -> list[str]:
        """Snapshot of the container-log ring buffer."""
        return list(self._log_ring)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_core/test_vllm_orchestrator.py -v
```
Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
python -m ruff check src/cognithor/core/vllm_orchestrator.py tests/test_core/test_vllm_orchestrator.py && python -m ruff format --check src/cognithor/core/vllm_orchestrator.py tests/test_core/test_vllm_orchestrator.py
git add src/cognithor/core/vllm_orchestrator.py tests/test_core/test_vllm_orchestrator.py
git commit -m "feat(vllm): orchestrator dataclasses + module skeleton"
```

---

## Task 6: Orchestrator `check_hardware()`

**Files:**
- Modify: `src/cognithor/core/vllm_orchestrator.py` (add method)
- Modify: `tests/test_core/test_vllm_orchestrator.py` (add test class)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_core/test_vllm_orchestrator.py`:

```python
from unittest.mock import patch, MagicMock

from cognithor.core.llm_backend import VLLMHardwareError


class TestCheckHardware:
    def _mk_orch(self):
        return VLLMOrchestrator()

    def test_detects_rtx_5090(self):
        # nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader,nounits
        # sample output: "NVIDIA GeForce RTX 5090, 32768, 12.0"
        mock_result = MagicMock(returncode=0, stdout="NVIDIA GeForce RTX 5090, 32768, 12.0\n")
        with patch("subprocess.run", return_value=mock_result):
            info = self._mk_orch().check_hardware()
        assert info.gpu_name == "NVIDIA GeForce RTX 5090"
        assert info.vram_gb == 32
        assert info.compute_capability == (12, 0)

    def test_detects_rtx_4090(self):
        mock_result = MagicMock(returncode=0, stdout="NVIDIA GeForce RTX 4090, 24564, 8.9\n")
        with patch("subprocess.run", return_value=mock_result):
            info = self._mk_orch().check_hardware()
        assert info.gpu_name == "NVIDIA GeForce RTX 4090"
        assert info.vram_gb == 24  # rounds down
        assert info.compute_capability == (8, 9)

    def test_raises_when_nvidia_smi_missing(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            with pytest.raises(VLLMHardwareError) as exc:
                self._mk_orch().check_hardware()
            assert "nvidia-smi" in str(exc.value).lower()

    def test_raises_when_no_gpu_detected(self):
        mock_result = MagicMock(returncode=0, stdout="")
        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(VLLMHardwareError):
                self._mk_orch().check_hardware()

    def test_raises_when_nvidia_smi_fails(self):
        mock_result = MagicMock(returncode=9, stdout="", stderr="NVIDIA-SMI has failed")
        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(VLLMHardwareError):
                self._mk_orch().check_hardware()

    def test_picks_first_gpu_when_multiple(self):
        mock_result = MagicMock(
            returncode=0,
            stdout="NVIDIA GeForce RTX 5090, 32768, 12.0\nNVIDIA GeForce RTX 3060, 12288, 8.6\n",
        )
        with patch("subprocess.run", return_value=mock_result):
            info = self._mk_orch().check_hardware()
        assert "5090" in info.gpu_name

    def test_state_updated_after_success(self):
        mock_result = MagicMock(returncode=0, stdout="NVIDIA GeForce RTX 4080, 16380, 8.9\n")
        orch = self._mk_orch()
        with patch("subprocess.run", return_value=mock_result):
            orch.check_hardware()
        assert orch.state.hardware_ok is True
        assert orch.state.hardware_info is not None
        assert orch.state.hardware_info.compute_capability == (8, 9)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_core/test_vllm_orchestrator.py::TestCheckHardware -v
```
Expected: `AttributeError: 'VLLMOrchestrator' object has no attribute 'check_hardware'`

- [ ] **Step 3: Implement `check_hardware()`**

Add to `VLLMOrchestrator` class in `src/cognithor/core/vllm_orchestrator.py`:

```python
import subprocess

from cognithor.core.llm_backend import VLLMHardwareError


def check_hardware(self) -> HardwareInfo:
    """Detect NVIDIA GPU. Raises VLLMHardwareError on any failure.

    Parses ``nvidia-smi --query-gpu=name,memory.total,compute_cap
    --format=csv,noheader,nounits``.
    """
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except FileNotFoundError as exc:
        raise VLLMHardwareError(
            "nvidia-smi not found — NVIDIA driver not installed?",
            recovery_hint="Install the NVIDIA GPU driver from nvidia.com.",
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise VLLMHardwareError(
            "nvidia-smi timed out",
            recovery_hint="Check GPU driver health.",
        ) from exc

    if result.returncode != 0:
        raise VLLMHardwareError(
            f"nvidia-smi failed: {result.stderr.strip() or 'unknown error'}",
        )

    first_line = result.stdout.strip().split("\n")[0] if result.stdout.strip() else ""
    if not first_line:
        raise VLLMHardwareError("No NVIDIA GPU detected")

    parts = [p.strip() for p in first_line.split(",")]
    if len(parts) < 3:
        raise VLLMHardwareError(f"Unexpected nvidia-smi output: {first_line!r}")

    gpu_name = parts[0]
    try:
        vram_mib = int(parts[1])
        cc_parts = parts[2].split(".")
        compute_capability = (int(cc_parts[0]), int(cc_parts[1]))
    except (ValueError, IndexError) as exc:
        raise VLLMHardwareError(f"Cannot parse nvidia-smi output: {first_line!r}") from exc

    info = HardwareInfo(
        gpu_name=gpu_name,
        vram_gb=vram_mib // 1024,
        compute_capability=compute_capability,
    )
    self.state.hardware_info = info
    self.state.hardware_ok = True
    return info
```

Put the method inside the `VLLMOrchestrator` class. Move `import subprocess` and `from cognithor.core.llm_backend import VLLMHardwareError` to module-level imports.

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_core/test_vllm_orchestrator.py::TestCheckHardware -v
```
Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
python -m ruff check src/cognithor/core/vllm_orchestrator.py tests/test_core/test_vllm_orchestrator.py && python -m ruff format --check src/cognithor/core/vllm_orchestrator.py tests/test_core/test_vllm_orchestrator.py
git add src/cognithor/core/vllm_orchestrator.py tests/test_core/test_vllm_orchestrator.py
git commit -m "feat(vllm): orchestrator check_hardware() via nvidia-smi"
```

---

## Task 7: Orchestrator `check_docker()`

**Files:**
- Modify: `src/cognithor/core/vllm_orchestrator.py`
- Modify: `tests/test_core/test_vllm_orchestrator.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_core/test_vllm_orchestrator.py`:

```python
class TestCheckDocker:
    def test_docker_running(self):
        # docker version --format json
        mock_stdout = '{"Client":{"Version":"26.0.0"},"Server":{"Version":"26.0.0"}}'
        mock_result = MagicMock(returncode=0, stdout=mock_stdout)
        with patch("subprocess.run", return_value=mock_result):
            info = VLLMOrchestrator().check_docker()
        assert info.available is True
        assert info.server_running is True
        assert info.version == "26.0.0"

    def test_docker_installed_but_server_down(self):
        # Client OK, Server missing (Docker Desktop not started)
        mock_stdout = '{"Client":{"Version":"26.0.0"}}'
        mock_result = MagicMock(returncode=0, stdout=mock_stdout)
        with patch("subprocess.run", return_value=mock_result):
            info = VLLMOrchestrator().check_docker()
        assert info.available is True
        assert info.server_running is False

    def test_docker_cli_missing(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            info = VLLMOrchestrator().check_docker()
        assert info.available is False
        assert info.server_running is False

    def test_docker_cmd_fails(self):
        mock_result = MagicMock(returncode=1, stdout="", stderr="daemon not running")
        with patch("subprocess.run", return_value=mock_result):
            info = VLLMOrchestrator().check_docker()
        assert info.available is True  # CLI exists
        assert info.server_running is False

    def test_state_updated(self):
        mock_stdout = '{"Client":{"Version":"26.0.0"},"Server":{"Version":"26.0.0"}}'
        orch = VLLMOrchestrator()
        with patch("subprocess.run", return_value=MagicMock(returncode=0, stdout=mock_stdout)):
            orch.check_docker()
        assert orch.state.docker_ok is True
```

- [ ] **Step 2: Run test**

```bash
python -m pytest tests/test_core/test_vllm_orchestrator.py::TestCheckDocker -v
```
Expected: `AttributeError: check_docker`

- [ ] **Step 3: Implement `check_docker()`**

Add method to `VLLMOrchestrator`:

```python
import json as _json


def check_docker(self) -> DockerInfo:
    """Detect Docker Desktop. Never raises — returns DockerInfo with flags."""
    try:
        result = subprocess.run(
            ["docker", "version", "--format", "json"],
            capture_output=True, text=True, timeout=10,
        )
    except FileNotFoundError:
        info = DockerInfo(available=False)
        self.state.docker_ok = False
        self.state.docker_info = info
        return info
    except subprocess.TimeoutExpired:
        info = DockerInfo(available=True, server_running=False)
        self.state.docker_ok = False
        self.state.docker_info = info
        return info

    if result.returncode != 0:
        info = DockerInfo(available=True, server_running=False)
        self.state.docker_ok = False
        self.state.docker_info = info
        return info

    try:
        parsed = _json.loads(result.stdout)
    except _json.JSONDecodeError:
        info = DockerInfo(available=True, server_running=False)
        self.state.docker_ok = False
        self.state.docker_info = info
        return info

    server = parsed.get("Server")
    version = (server or parsed.get("Client", {})).get("Version", "")
    info = DockerInfo(
        available=True,
        version=version,
        server_running=server is not None,
    )
    self.state.docker_ok = info.server_running
    self.state.docker_info = info
    return info
```

Add `import json as _json` to module imports.

- [ ] **Step 4: Run test**

```bash
python -m pytest tests/test_core/test_vllm_orchestrator.py::TestCheckDocker -v
```
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
python -m ruff check src/cognithor/core/vllm_orchestrator.py tests/test_core/test_vllm_orchestrator.py && python -m ruff format --check src/cognithor/core/vllm_orchestrator.py tests/test_core/test_vllm_orchestrator.py
git add src/cognithor/core/vllm_orchestrator.py tests/test_core/test_vllm_orchestrator.py
git commit -m "feat(vllm): orchestrator check_docker() non-throwing detection"
```

---

## Task 8: Orchestrator `recommend_model()` + `filter_registry()`

**Files:**
- Modify: `src/cognithor/core/vllm_orchestrator.py`
- Create: `tests/test_core/test_vllm_recommend_model.py` (new)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_core/test_vllm_recommend_model.py
from __future__ import annotations

import json
from pathlib import Path

import pytest

from cognithor.core.vllm_orchestrator import (
    HardwareInfo,
    ModelEntry,
    VLLMOrchestrator,
)


REGISTRY_PATH = (
    Path(__file__).resolve().parents[2]
    / "src" / "cognithor" / "cli" / "model_registry.json"
)


@pytest.fixture
def registry() -> list[ModelEntry]:
    data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    return [ModelEntry.from_dict(m) for m in data["providers"]["vllm"]["models"]]


@pytest.fixture
def orch() -> VLLMOrchestrator:
    return VLLMOrchestrator()


class TestRecommendModel:
    def test_blackwell_32gb_picks_nvfp4(self, orch, registry):
        hw = HardwareInfo(gpu_name="RTX 5090", vram_gb=32, compute_capability=(12, 0))
        best = orch.recommend_model(hw, registry, prefer="vision")
        assert best.quantization == "NVFP4"
        assert "Qwen3.6-27B" in best.base_model

    def test_ada_24gb_falls_back_to_qwen25(self, orch, registry):
        # All Qwen3.6 entries have min_vllm_version="pending" → tested=false.
        # Until one is tested, the only tested entry (Qwen2.5-VL-7B) wins.
        hw = HardwareInfo(gpu_name="RTX 4090", vram_gb=24, compute_capability=(8, 9))
        best = orch.recommend_model(hw, registry, prefer="vision")
        assert best.tested is True
        assert "Qwen2.5-VL" in best.id

    def test_ampere_24gb_falls_back_to_qwen25(self, orch, registry):
        hw = HardwareInfo(gpu_name="RTX 3090", vram_gb=24, compute_capability=(8, 0))
        best = orch.recommend_model(hw, registry, prefer="vision")
        assert best.tested is True

    def test_turing_16gb_falls_back_to_qwen25(self, orch, registry):
        hw = HardwareInfo(gpu_name="RTX 2080 Ti", vram_gb=11, compute_capability=(7, 5))
        # 11 GB < 16 GB → fails all entries. Returns None.
        best = orch.recommend_model(hw, registry, prefer="vision")
        assert best is None

    def test_no_vision_text_preference_returns_none_when_none_curated(self, orch, registry):
        hw = HardwareInfo(gpu_name="RTX 5090", vram_gb=32, compute_capability=(12, 0))
        best = orch.recommend_model(hw, registry, prefer="text")
        # Registry has NO text-capability entries
        assert best is None

    def test_ignores_entries_exceeding_vram(self, orch, registry):
        hw = HardwareInfo(gpu_name="RTX 4080", vram_gb=16, compute_capability=(8, 9))
        results = orch.filter_registry(hw, registry)
        for m in results:
            assert m.vram_gb_min <= 16

    def test_ignores_entries_requiring_newer_compute_capability(self, orch, registry):
        hw = HardwareInfo(gpu_name="RTX 4090", vram_gb=24, compute_capability=(8, 9))
        results = orch.filter_registry(hw, registry)
        # NVFP4 (12.0) must NOT pass
        for m in results:
            assert m.min_cc_tuple <= (8, 9)


class TestRecommendationAfterQwen36Support:
    """Scenario: vLLM ships Qwen3.6 support, we flip tested=True on the FP8 entry."""

    def test_tested_nvfp4_beats_tested_fallback_on_blackwell(self, orch):
        # Synthetic registry — NVFP4 marked tested=true (simulating future release)
        entries = [
            ModelEntry.from_dict({
                "id": "mmangkad/Qwen3.6-27B-NVFP4",
                "display_name": "Qwen3.6-27B NVFP4",
                "base_model": "Qwen/Qwen3.6-27B",
                "quantization": "NVFP4",
                "vram_gb_min": 14,
                "min_compute_capability": "12.0",
                "min_vllm_version": "0.20.0",
                "capability": "vision",
                "priority": "premium",
                "tested": True,
                "notes": "",
            }),
            ModelEntry.from_dict({
                "id": "Qwen/Qwen2.5-VL-7B-Instruct",
                "display_name": "Qwen2.5-VL-7B",
                "base_model": "Qwen/Qwen2.5-VL-7B-Instruct",
                "quantization": "bf16",
                "vram_gb_min": 16,
                "min_compute_capability": "7.5",
                "min_vllm_version": "0.7.0",
                "capability": "vision",
                "priority": "fallback",
                "tested": True,
                "notes": "",
            }),
        ]
        hw = HardwareInfo(gpu_name="RTX 5090", vram_gb=32, compute_capability=(12, 0))
        best = orch.recommend_model(hw, entries, prefer="vision")
        assert best.id == "mmangkad/Qwen3.6-27B-NVFP4"
```

- [ ] **Step 2: Run test**

```bash
python -m pytest tests/test_core/test_vllm_recommend_model.py -v
```
Expected: `AttributeError: 'VLLMOrchestrator' object has no attribute 'recommend_model'`

- [ ] **Step 3: Implement `filter_registry()` + `recommend_model()`**

Add to `VLLMOrchestrator`:

```python
_PRIORITY_ORDER = {"premium": 0, "standard": 1, "fallback": 2}


def filter_registry(
    self,
    hardware: HardwareInfo,
    registry: list[ModelEntry],
) -> list[ModelEntry]:
    """Returns entries that fit the detected GPU (VRAM + compute capability)."""
    return [
        m for m in registry
        if m.vram_gb_min <= hardware.vram_gb
        and m.min_cc_tuple <= hardware.compute_capability
    ]


def recommend_model(
    self,
    hardware: HardwareInfo,
    registry: list[ModelEntry],
    *,
    prefer: Literal["vision", "text"] = "vision",
) -> ModelEntry | None:
    """Pick the best curated model for detected hardware.

    Ranking:
      1. Matches requested ``capability`` (vision/text)
      2. ``tested==True`` beats ``tested==False``
      3. Higher priority (premium > standard > fallback)
      4. Tie-break: larger VRAM headroom (lower vram_gb_min for same tier)

    Returns None if no entry fits.
    """
    candidates = [
        m for m in self.filter_registry(hardware, registry)
        if m.capability == prefer
    ]
    if not candidates:
        return None

    def sort_key(m: ModelEntry) -> tuple[int, int, int, int]:
        return (
            0 if m.tested else 1,          # tested first
            self._PRIORITY_ORDER[m.priority],  # then priority tier
            m.vram_gb_min,                 # then smaller VRAM (more headroom)
            0 if m.min_vllm_version != "pending" else 1,  # stable before pending
        )

    candidates.sort(key=sort_key)
    return candidates[0]
```

Add `from typing import Literal` to imports if not present.

- [ ] **Step 4: Run test**

```bash
python -m pytest tests/test_core/test_vllm_recommend_model.py -v
```
Expected: `8 passed`

- [ ] **Step 5: Commit**

```bash
python -m ruff check src/cognithor/core/vllm_orchestrator.py tests/test_core/test_vllm_recommend_model.py && python -m ruff format --check src/cognithor/core/vllm_orchestrator.py tests/test_core/test_vllm_recommend_model.py
git add src/cognithor/core/vllm_orchestrator.py tests/test_core/test_vllm_recommend_model.py
git commit -m "feat(vllm): orchestrator recommend_model() + filter_registry() with GPU-aware ranking"
```

---

## Task 9: Orchestrator `pull_image()`

**Files:**
- Modify: `src/cognithor/core/vllm_orchestrator.py`
- Modify: `tests/test_core/test_vllm_orchestrator.py`

- [ ] **Step 1: Write the failing test**

```python
class TestPullImage:
    def test_pull_emits_progress_events(self):
        """docker pull --progress=auto outputs JSON lines when TTY-off."""
        # Simulate streaming stdout from docker
        json_lines = [
            '{"status":"Pulling from vllm/vllm-openai","id":"latest"}\n',
            '{"status":"Downloading","progressDetail":{"current":1000000,"total":10000000},"id":"abc123"}\n',
            '{"status":"Download complete","id":"abc123"}\n',
            '{"status":"Status: Downloaded newer image for vllm/vllm-openai:v0.19.1"}\n',
        ]
        mock_proc = MagicMock()
        mock_proc.stdout = iter(json_lines)
        mock_proc.wait.return_value = 0
        mock_proc.returncode = 0

        events: list[dict] = []
        def cb(ev): events.append(ev)

        with patch("subprocess.Popen", return_value=mock_proc):
            VLLMOrchestrator().pull_image("vllm/vllm-openai:v0.19.1", progress_callback=cb)

        # Expect at least one "Downloading" event with progressDetail
        assert any(e.get("status") == "Downloading" for e in events)
        assert any("current" in (e.get("progressDetail") or {}) for e in events)

    def test_pull_failure_raises(self):
        mock_proc = MagicMock()
        mock_proc.stdout = iter(['{"status":"error"}\n'])
        mock_proc.wait.return_value = 1
        mock_proc.returncode = 1

        with patch("subprocess.Popen", return_value=mock_proc):
            with pytest.raises(Exception):
                VLLMOrchestrator().pull_image("bad/image:tag", progress_callback=None)

    def test_pull_sets_image_pulled_flag(self):
        mock_proc = MagicMock()
        mock_proc.stdout = iter(['{"status":"Pulling"}\n'])
        mock_proc.wait.return_value = 0
        mock_proc.returncode = 0
        orch = VLLMOrchestrator()
        with patch("subprocess.Popen", return_value=mock_proc):
            orch.pull_image(orch.docker_image, progress_callback=None)
        assert orch.state.image_pulled is True
```

- [ ] **Step 2: Run test**

```bash
python -m pytest tests/test_core/test_vllm_orchestrator.py::TestPullImage -v
```
Expected: `AttributeError: pull_image`

- [ ] **Step 3: Implement `pull_image()`**

Add to `VLLMOrchestrator`:

```python
from collections.abc import Callable


ProgressCallback = Callable[[dict[str, Any]], None] | None


def pull_image(
    self,
    tag: str,
    *,
    progress_callback: ProgressCallback = None,
) -> None:
    """Run ``docker pull`` streaming JSON progress to the callback.

    Raises:
        VLLMDockerError: if the pull exits non-zero.
    """
    cmd = ["docker", "pull", "--progress=auto", tag]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    try:
        for line in proc.stdout or []:
            line = line.strip()
            if not line:
                continue
            try:
                event = _json.loads(line)
            except _json.JSONDecodeError:
                # Docker's --progress=auto sometimes mixes plain text with JSON
                event = {"status": line}
            if progress_callback is not None:
                progress_callback(event)
    finally:
        proc.wait()

    if proc.returncode != 0:
        raise VLLMDockerError(
            f"docker pull {tag} failed with exit {proc.returncode}",
            recovery_hint="Check Docker Desktop is running and you have network access.",
        )

    self.state.image_pulled = True
```

Import `VLLMDockerError` at module top:

```python
from cognithor.core.llm_backend import VLLMDockerError, VLLMHardwareError
```

- [ ] **Step 4: Run test**

```bash
python -m pytest tests/test_core/test_vllm_orchestrator.py::TestPullImage -v
```
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
python -m ruff check src/cognithor/core/vllm_orchestrator.py tests/test_core/test_vllm_orchestrator.py && python -m ruff format --check src/cognithor/core/vllm_orchestrator.py tests/test_core/test_vllm_orchestrator.py
git add src/cognithor/core/vllm_orchestrator.py tests/test_core/test_vllm_orchestrator.py
git commit -m "feat(vllm): orchestrator pull_image() with streaming JSON progress"
```

---

## Task 10: Orchestrator `start_container()`

**Files:**
- Modify: `src/cognithor/core/vllm_orchestrator.py`
- Modify: `tests/test_core/test_vllm_orchestrator.py`

- [ ] **Step 1: Write the failing test**

```python
class TestStartContainer:
    def test_constructs_docker_run_command(self):
        # Mock: port-check OK, docker run OK, health-check OK
        with patch.object(VLLMOrchestrator, "_port_available", return_value=True), \
             patch("subprocess.run") as run_mock, \
             patch.object(VLLMOrchestrator, "_wait_for_health", return_value=True):

            run_mock.return_value = MagicMock(returncode=0, stdout="abc123def456")
            orch = VLLMOrchestrator(docker_image="vllm/vllm-openai:v0.19.1", port=8000, hf_token="hf_x")
            info = orch.start_container("Qwen/Qwen3.6-27B-FP8")

        args = run_mock.call_args[0][0]
        # Verify key flags in the docker run invocation
        assert "run" in args
        assert "-d" in args
        assert "--gpus" in args and "all" in args
        assert any("HF_TOKEN=hf_x" in a for a in args)
        assert any("cognithor.managed=true" in a for a in args)
        assert any("vllm-openai:v0.19.1" in a for a in args)
        assert "Qwen/Qwen3.6-27B-FP8" in args
        assert info.port == 8000
        assert info.model == "Qwen/Qwen3.6-27B-FP8"

    def test_port_fallback_when_busy(self):
        orch = VLLMOrchestrator(port=8000)
        # 8000 and 8001 busy, 8002 free
        with patch.object(orch, "_port_available", side_effect=[False, False, True]), \
             patch("subprocess.run", return_value=MagicMock(returncode=0, stdout="cid")), \
             patch.object(orch, "_wait_for_health", return_value=True):
            info = orch.start_container("Qwen/Qwen2.5-VL-7B-Instruct")
        assert info.port == 8002

    def test_raises_when_all_ports_busy(self):
        orch = VLLMOrchestrator(port=8000)
        with patch.object(orch, "_port_available", return_value=False):
            with pytest.raises(VLLMNotReadyError) as exc:
                orch.start_container("Qwen/Qwen2.5-VL-7B-Instruct")
            assert "port" in str(exc.value).lower()

    def test_health_timeout_scales_with_model_size(self):
        """Models with vram >= 20 GB get 300 s timeout instead of 120 s."""
        orch = VLLMOrchestrator(port=8000)
        # No registry lookup done here — timeout is passed explicitly
        with patch.object(orch, "_port_available", return_value=True), \
             patch("subprocess.run", return_value=MagicMock(returncode=0, stdout="cid")), \
             patch.object(orch, "_wait_for_health", return_value=True) as wait_mock:
            orch.start_container("x", health_timeout=300)
        # Verify _wait_for_health called with 300
        assert wait_mock.call_args.kwargs.get("timeout") == 300 \
            or (wait_mock.call_args.args and wait_mock.call_args.args[-1] == 300)

    def test_default_health_timeout_120(self):
        orch = VLLMOrchestrator(port=8000)
        with patch.object(orch, "_port_available", return_value=True), \
             patch("subprocess.run", return_value=MagicMock(returncode=0, stdout="cid")), \
             patch.object(orch, "_wait_for_health", return_value=True) as wait_mock:
            orch.start_container("x")
        assert wait_mock.call_args.kwargs.get("timeout", 120) == 120
```

Add import at top of test file:

```python
from cognithor.core.llm_backend import VLLMNotReadyError
```

- [ ] **Step 2: Run test**

```bash
python -m pytest tests/test_core/test_vllm_orchestrator.py::TestStartContainer -v
```
Expected: `AttributeError: start_container`

- [ ] **Step 3: Implement `start_container()`**

Add to `VLLMOrchestrator`:

```python
import socket
import time
import httpx

from cognithor.core.llm_backend import VLLMNotReadyError

_MAX_PORT_FALLBACKS = 10


def _port_available(self, port: int) -> bool:
    """True if nothing is listening on localhost:port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def _wait_for_health(self, port: int, *, timeout: int = 120) -> bool:
    """Poll vLLM /health until 200 or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with httpx.Client(timeout=2.0) as client:
                r = client.get(f"http://localhost:{port}/health")
                if r.status_code == 200:
                    return True
        except Exception:
            pass
        time.sleep(2.0)
    return False


def start_container(
    self,
    model: str,
    *,
    health_timeout: int | None = None,
) -> ContainerInfo:
    """Start a vLLM container. Auto-falls-back 8000→8009 on port conflict."""
    # Resolve port
    port = self.port
    for offset in range(_MAX_PORT_FALLBACKS):
        candidate = self.port + offset
        if self._port_available(candidate):
            port = candidate
            break
    else:
        raise VLLMNotReadyError(
            f"All ports {self.port}..{self.port + _MAX_PORT_FALLBACKS - 1} are busy",
            recovery_hint="Stop other services or change config.vllm.port.",
        )

    # Construct docker run
    cmd = [
        "docker", "run", "-d",
        "--gpus", "all",
        "-v", "cognithor-hf-cache:/root/.cache/huggingface",
        "-e", f"HF_TOKEN={self._hf_token}",
        "-p", f"{port}:8000",
        "--label", "cognithor.managed=true",
        self.docker_image,
        "--model", model,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise VLLMNotReadyError(
            f"docker run failed: {result.stderr.strip()}",
            recovery_hint="Check Docker Desktop logs.",
        )

    container_id = result.stdout.strip().split("\n")[-1][:12]

    # Wait for /health
    timeout = health_timeout if health_timeout is not None else 120
    if not self._wait_for_health(port, timeout=timeout):
        raise VLLMNotReadyError(
            f"vLLM /health did not respond within {timeout}s",
            recovery_hint="Check `docker logs <id>` for model-loading errors.",
        )

    info = ContainerInfo(container_id=container_id, port=port, model=model)
    self.state.container_running = True
    self.state.current_model = model
    return info
```

Import `socket`, `time`, `httpx` at module top. Import `VLLMNotReadyError` alongside the others.

- [ ] **Step 4: Run test**

```bash
python -m pytest tests/test_core/test_vllm_orchestrator.py::TestStartContainer -v
```
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
python -m ruff check src/cognithor/core/vllm_orchestrator.py tests/test_core/test_vllm_orchestrator.py && python -m ruff format --check src/cognithor/core/vllm_orchestrator.py tests/test_core/test_vllm_orchestrator.py
git add src/cognithor/core/vllm_orchestrator.py tests/test_core/test_vllm_orchestrator.py
git commit -m "feat(vllm): orchestrator start_container() with port fallback + health wait"
```

---

## Task 11: Orchestrator `stop_container()` + `reuse_existing()`

**Files:**
- Modify: `src/cognithor/core/vllm_orchestrator.py`
- Modify: `tests/test_core/test_vllm_orchestrator.py`

- [ ] **Step 1: Write the failing test**

```python
class TestStopAndReuse:
    def test_stop_container_via_label(self):
        # docker ps --filter "label=cognithor.managed=true" --format "{{.ID}}"
        find_result = MagicMock(returncode=0, stdout="abc123def456\n")
        stop_result = MagicMock(returncode=0)
        rm_result = MagicMock(returncode=0)

        orch = VLLMOrchestrator()
        orch.state.container_running = True

        with patch("subprocess.run", side_effect=[find_result, stop_result, rm_result]):
            orch.stop_container()

        assert orch.state.container_running is False

    def test_stop_when_no_container_is_noop(self):
        find_result = MagicMock(returncode=0, stdout="")
        orch = VLLMOrchestrator()
        with patch("subprocess.run", return_value=find_result):
            orch.stop_container()  # must not raise

    def test_reuse_existing_returns_info(self):
        # ps --filter label=... --format json
        ps_stdout = '{"ID":"abc123def456","Ports":"0.0.0.0:8000->8000/tcp","Image":"vllm/vllm-openai:v0.19.1","Command":"... --model Qwen/Qwen3.6-27B-FP8 ..."}\n'
        with patch("subprocess.run", return_value=MagicMock(returncode=0, stdout=ps_stdout)):
            info = VLLMOrchestrator().reuse_existing()
        assert info is not None
        assert info.container_id == "abc123def456"
        assert info.port == 8000

    def test_reuse_existing_returns_none_when_nothing_running(self):
        with patch("subprocess.run", return_value=MagicMock(returncode=0, stdout="")):
            info = VLLMOrchestrator().reuse_existing()
        assert info is None
```

- [ ] **Step 2: Run test**

```bash
python -m pytest tests/test_core/test_vllm_orchestrator.py::TestStopAndReuse -v
```
Expected: `AttributeError: stop_container`

- [ ] **Step 3: Implement both methods**

Add to `VLLMOrchestrator`:

```python
import re


def stop_container(self) -> None:
    """Stop and remove the cognithor-managed vLLM container. Noop if none."""
    find = subprocess.run(
        ["docker", "ps", "-q", "--filter", "label=cognithor.managed=true"],
        capture_output=True, text=True, timeout=10,
    )
    container_id = find.stdout.strip().split("\n")[0] if find.stdout.strip() else ""
    if not container_id:
        self.state.container_running = False
        return

    subprocess.run(["docker", "stop", container_id], capture_output=True, timeout=30)
    subprocess.run(["docker", "rm", container_id], capture_output=True, timeout=10)
    self.state.container_running = False
    self.state.current_model = None


def reuse_existing(self) -> ContainerInfo | None:
    """If a cognithor-managed container is already running, return its info.

    Used at app-start to pick up a container left running across sessions.
    """
    result = subprocess.run(
        [
            "docker", "ps",
            "--filter", "label=cognithor.managed=true",
            "--format", "json",
        ],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None

    # Docker outputs one JSON object per line
    first_line = result.stdout.strip().split("\n")[0]
    try:
        row = _json.loads(first_line)
    except _json.JSONDecodeError:
        return None

    container_id = row.get("ID", "").strip()
    ports = row.get("Ports", "")
    cmd = row.get("Command", "")

    port_match = re.search(r"0\.0\.0\.0:(\d+)->8000/tcp", ports)
    port = int(port_match.group(1)) if port_match else self.port

    model_match = re.search(r"--model\s+(\S+)", cmd)
    model = model_match.group(1) if model_match else ""

    if not container_id:
        return None

    info = ContainerInfo(container_id=container_id, port=port, model=model)
    self.state.container_running = True
    self.state.current_model = model
    return info
```

Add `import re` to imports.

- [ ] **Step 4: Run test**

```bash
python -m pytest tests/test_core/test_vllm_orchestrator.py::TestStopAndReuse -v
```
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
python -m ruff check src/cognithor/core/vllm_orchestrator.py tests/test_core/test_vllm_orchestrator.py && python -m ruff format --check src/cognithor/core/vllm_orchestrator.py tests/test_core/test_vllm_orchestrator.py
git add src/cognithor/core/vllm_orchestrator.py tests/test_core/test_vllm_orchestrator.py
git commit -m "feat(vllm): orchestrator stop_container() + reuse_existing() via label filter"
```

---

## Task 12: Orchestrator `status()` Aggregator

**Files:**
- Modify: `src/cognithor/core/vllm_orchestrator.py`
- Modify: `tests/test_core/test_vllm_orchestrator.py`

- [ ] **Step 1: Write the failing test**

```python
class TestStatusAggregator:
    def test_status_returns_current_state(self):
        orch = VLLMOrchestrator()
        orch.state.hardware_ok = True
        orch.state.docker_ok = True
        snapshot = orch.status()
        assert snapshot.hardware_ok is True
        assert snapshot.docker_ok is True
        # Must return a copy — mutating the returned object doesn't leak
        snapshot.hardware_ok = False
        assert orch.state.hardware_ok is True
```

- [ ] **Step 2: Run test**

```bash
python -m pytest tests/test_core/test_vllm_orchestrator.py::TestStatusAggregator -v
```
Expected: `AttributeError: status`

- [ ] **Step 3: Implement `status()`**

```python
import dataclasses


def status(self) -> VLLMState:
    """Return a snapshot copy of the current state (safe to mutate)."""
    return dataclasses.replace(self.state)
```

- [ ] **Step 4: Run test**

```bash
python -m pytest tests/test_core/test_vllm_orchestrator.py -v
```
Expected: all orchestrator tests pass

- [ ] **Step 5: Commit**

```bash
python -m ruff check src/cognithor/core/vllm_orchestrator.py tests/test_core/test_vllm_orchestrator.py && python -m ruff format --check src/cognithor/core/vllm_orchestrator.py tests/test_core/test_vllm_orchestrator.py
git add src/cognithor/core/vllm_orchestrator.py tests/test_core/test_vllm_orchestrator.py
git commit -m "feat(vllm): orchestrator status() snapshot aggregator"
```

---

## Task 13: `VLLMBackend` Class Skeleton + `is_available()` + `list_models()` + `close()`

**Files:**
- Create: `src/cognithor/core/vllm_backend.py` (new)
- Create: `tests/test_core/test_vllm_backend.py` (new)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_core/test_vllm_backend.py
from __future__ import annotations

import pytest
from pytest_httpx import HTTPXMock

from cognithor.core.llm_backend import (
    LLMBackendType,
    LLMBadRequestError,
    VLLMNotReadyError,
)
from cognithor.core.vllm_backend import VLLMBackend


BASE_URL = "http://localhost:8000/v1"


@pytest.fixture
def backend() -> VLLMBackend:
    return VLLMBackend(base_url=BASE_URL, timeout=5)


class TestVLLMBackendBasics:
    def test_backend_type(self, backend):
        assert backend.backend_type == LLMBackendType.VLLM

    @pytest.mark.asyncio
    async def test_is_available_true_on_200(self, backend, httpx_mock: HTTPXMock):
        httpx_mock.add_response(
            url="http://localhost:8000/health",
            status_code=200,
        )
        assert await backend.is_available() is True

    @pytest.mark.asyncio
    async def test_is_available_false_on_connection_refused(self, backend):
        # No mock registered → httpx gets connection refused
        assert await backend.is_available() is False

    @pytest.mark.asyncio
    async def test_list_models_from_openai_endpoint(self, backend, httpx_mock):
        httpx_mock.add_response(
            url=f"{BASE_URL}/models",
            status_code=200,
            json={"data": [{"id": "Qwen/Qwen3.6-27B-FP8"}]},
        )
        models = await backend.list_models()
        assert "Qwen/Qwen3.6-27B-FP8" in models

    @pytest.mark.asyncio
    async def test_close_is_idempotent(self, backend):
        await backend.close()
        await backend.close()  # second call must not raise
```

Install pytest-httpx if not already: check with `pip show pytest-httpx`. If missing, install via `pip install pytest-httpx` and add to dev-dependencies in `pyproject.toml`.

- [ ] **Step 2: Run test**

```bash
python -m pytest tests/test_core/test_vllm_backend.py -v
```
Expected: `ModuleNotFoundError: No module named 'cognithor.core.vllm_backend'`

- [ ] **Step 3: Create `vllm_backend.py` skeleton**

```python
"""vLLM backend — OpenAI-compatible LLMBackend adapter.

vLLM serves an OpenAI-compatible ``/v1/chat/completions`` endpoint.
This class adapts it to Cognithor's LLMBackend ABC with image-payload
conversion for vision models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import httpx

from cognithor.core.llm_backend import (
    ChatResponse,
    EmbedResponse,
    LLMBackend,
    LLMBackendError,
    LLMBackendType,
    LLMBadRequestError,
    VLLMNotReadyError,
)
from cognithor.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

log = get_logger(__name__)


class VLLMBackend(LLMBackend):
    """vLLM OpenAI-compat adapter."""

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:8000/v1",
        timeout: int = 60,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    @property
    def backend_type(self) -> LLMBackendType:
        return LLMBackendType.VLLM

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def is_available(self) -> bool:
        """Ping /health (NOT /v1/health — vLLM exposes /health at server root)."""
        health_url = self._base_url.rsplit("/v1", 1)[0] + "/health"
        client = await self._ensure_client()
        try:
            r = await client.get(health_url)
            return r.status_code == 200
        except Exception:
            return False

    async def list_models(self) -> list[str]:
        client = await self._ensure_client()
        try:
            r = await client.get(f"{self._base_url}/models")
            r.raise_for_status()
            data = r.json()
            return [m["id"] for m in data.get("data", [])]
        except httpx.HTTPStatusError as exc:
            raise LLMBackendError(f"vLLM /models failed: {exc}") from exc
        except httpx.RequestError as exc:
            raise VLLMNotReadyError(f"vLLM not reachable: {exc}") from exc

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # chat() / chat_stream() / embed() added in later tasks.
    async def chat(self, *args: Any, **kwargs: Any) -> ChatResponse:
        raise NotImplementedError

    async def chat_stream(self, *args: Any, **kwargs: Any) -> AsyncIterator[str]:
        raise NotImplementedError
        yield  # pragma: no cover

    async def embed(self, *args: Any, **kwargs: Any) -> EmbedResponse:
        raise NotImplementedError
```

- [ ] **Step 4: Run test**

```bash
python -m pytest tests/test_core/test_vllm_backend.py -v
```
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
python -m ruff check src/cognithor/core/vllm_backend.py tests/test_core/test_vllm_backend.py && python -m ruff format --check src/cognithor/core/vllm_backend.py tests/test_core/test_vllm_backend.py
git add src/cognithor/core/vllm_backend.py tests/test_core/test_vllm_backend.py
git commit -m "feat(vllm): VLLMBackend skeleton with is_available/list_models/close"
```

---

## Task 14: `VLLMBackend.chat()` with Image-Payload Conversion

**Files:**
- Modify: `src/cognithor/core/vllm_backend.py`
- Modify: `tests/test_core/test_vllm_backend.py`

- [ ] **Step 1: Write the failing test**

```python
class TestVLLMBackendChat:
    @pytest.mark.asyncio
    async def test_chat_sends_openai_payload(self, backend, httpx_mock):
        httpx_mock.add_response(
            url=f"{BASE_URL}/chat/completions",
            status_code=200,
            json={
                "choices": [{"message": {"content": "Hello!"}}],
                "model": "Qwen/Qwen2.5-VL-7B-Instruct",
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            },
        )
        resp = await backend.chat(
            model="Qwen/Qwen2.5-VL-7B-Instruct",
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.7,
        )
        assert resp.content == "Hello!"
        assert resp.model == "Qwen/Qwen2.5-VL-7B-Instruct"
        assert resp.usage == {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}

        request = httpx_mock.get_requests()[0]
        import json as _j
        body = _j.loads(request.content)
        assert body["model"] == "Qwen/Qwen2.5-VL-7B-Instruct"
        assert body["temperature"] == 0.7
        assert body["messages"] == [{"role": "user", "content": "Hi"}]

    @pytest.mark.asyncio
    async def test_chat_converts_image_paths_to_openai_vision_format(
        self, backend, httpx_mock, tmp_path
    ):
        img = tmp_path / "pic.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")  # minimal PNG-ish bytes

        httpx_mock.add_response(
            url=f"{BASE_URL}/chat/completions",
            status_code=200,
            json={"choices": [{"message": {"content": "ok"}}], "model": "x"},
        )
        await backend.chat(
            model="Qwen/Qwen2.5-VL-7B-Instruct",
            messages=[{"role": "user", "content": "what is this?"}],
            images=[str(img)],
        )

        import json as _j
        body = _j.loads(httpx_mock.get_requests()[0].content)
        last = body["messages"][-1]
        assert isinstance(last["content"], list)
        assert any(c.get("type") == "text" for c in last["content"])
        assert any(
            c.get("type") == "image_url" and c["image_url"]["url"].startswith("data:image/png;base64,")
            for c in last["content"]
        )

    @pytest.mark.asyncio
    async def test_chat_raises_bad_request_on_400(self, backend, httpx_mock):
        httpx_mock.add_response(
            url=f"{BASE_URL}/chat/completions",
            status_code=400,
            json={"error": {"message": "context too long"}},
        )
        with pytest.raises(LLMBadRequestError):
            await backend.chat(model="x", messages=[{"role": "user", "content": "a"}])

    @pytest.mark.asyncio
    async def test_chat_raises_not_ready_on_5xx(self, backend, httpx_mock):
        httpx_mock.add_response(
            url=f"{BASE_URL}/chat/completions",
            status_code=503,
            json={"error": "model loading"},
        )
        with pytest.raises(VLLMNotReadyError):
            await backend.chat(model="x", messages=[{"role": "user", "content": "a"}])

    @pytest.mark.asyncio
    async def test_chat_raises_not_ready_on_connection_refused(self, backend):
        # No mock → connection refused
        with pytest.raises(VLLMNotReadyError):
            await backend.chat(model="x", messages=[{"role": "user", "content": "a"}])
```

- [ ] **Step 2: Run test**

```bash
python -m pytest tests/test_core/test_vllm_backend.py::TestVLLMBackendChat -v
```
Expected: `NotImplementedError`

- [ ] **Step 3: Implement `chat()` + image conversion**

Replace the stub `chat` in `VLLMBackend` with:

```python
import base64
from pathlib import Path


def _encode_image_to_data_url(path: str) -> str | None:
    """Read an image file, return OpenAI-vision data-URL string. None if unreadable."""
    try:
        p = Path(path)
        if not p.is_file():
            return None
        data = p.read_bytes()
    except OSError:
        return None

    suffix = p.suffix.lower().lstrip(".")
    mime_map = {"png": "png", "jpg": "jpeg", "jpeg": "jpeg", "webp": "webp", "gif": "gif", "bmp": "bmp"}
    mime = mime_map.get(suffix, "png")
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/{mime};base64,{b64}"


def _attach_images_to_last_user(
    messages: list[dict[str, Any]],
    images: list[str],
) -> list[dict[str, Any]]:
    """Return a NEW messages list with images attached to the last user message in
    OpenAI-vision format. Never mutates the caller's list."""
    if not images:
        return list(messages)

    encoded = [e for e in (_encode_image_to_data_url(p) for p in images) if e]
    if not encoded:
        return list(messages)

    new_messages = [dict(m) for m in messages]
    for i in range(len(new_messages) - 1, -1, -1):
        if new_messages[i].get("role") == "user":
            existing = new_messages[i].get("content")
            text_part = existing if isinstance(existing, str) else ""
            content_list: list[dict[str, Any]] = []
            if text_part:
                content_list.append({"type": "text", "text": text_part})
            for url in encoded:
                content_list.append({"type": "image_url", "image_url": {"url": url}})
            new_messages[i] = {**new_messages[i], "content": content_list}
            break
    else:
        content_list = [{"type": "image_url", "image_url": {"url": u}} for u in encoded]
        new_messages.append({"role": "user", "content": content_list})
    return new_messages


async def chat(
    self,
    model: str,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    format_json: bool = False,
    images: list[str] | None = None,
) -> ChatResponse:
    """Send a chat-completion request to vLLM. Raises LLMBadRequestError
    on 400 (excluded from circuit breaker), VLLMNotReadyError on 5xx or
    connection failure (counts toward breaker)."""
    if images:
        messages = _attach_images_to_last_user(messages, images)

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
    }
    if tools:
        payload["tools"] = tools
    if format_json:
        payload["response_format"] = {"type": "json_object"}

    client = await self._ensure_client()
    try:
        r = await client.post(f"{self._base_url}/chat/completions", json=payload)
    except httpx.RequestError as exc:
        raise VLLMNotReadyError(
            f"vLLM not reachable: {exc}",
            recovery_hint="Check vLLM container is running.",
        ) from exc

    if r.status_code == 400:
        raise LLMBadRequestError(
            f"vLLM rejected the request: {r.text[:200]}",
            status_code=400,
        )
    if r.status_code >= 500:
        raise VLLMNotReadyError(
            f"vLLM returned {r.status_code}: {r.text[:200]}",
            status_code=r.status_code,
            recovery_hint="vLLM may still be loading the model.",
        )
    if r.status_code >= 400:
        raise LLMBackendError(
            f"vLLM returned {r.status_code}: {r.text[:200]}",
            status_code=r.status_code,
        )

    data = r.json()
    first_choice = data.get("choices", [{}])[0]
    content = first_choice.get("message", {}).get("content", "")
    tool_calls = first_choice.get("message", {}).get("tool_calls")
    return ChatResponse(
        content=content,
        tool_calls=tool_calls,
        model=data.get("model", model),
        usage=data.get("usage"),
        raw=data,
    )
```

Move the helper functions (`_encode_image_to_data_url`, `_attach_images_to_last_user`) to module level. Add `import base64` and `from pathlib import Path` at module top.

- [ ] **Step 4: Run test**

```bash
python -m pytest tests/test_core/test_vllm_backend.py::TestVLLMBackendChat -v
```
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
python -m ruff check src/cognithor/core/vllm_backend.py tests/test_core/test_vllm_backend.py && python -m ruff format --check src/cognithor/core/vllm_backend.py tests/test_core/test_vllm_backend.py
git add src/cognithor/core/vllm_backend.py tests/test_core/test_vllm_backend.py
git commit -m "feat(vllm): VLLMBackend.chat() with OpenAI-vision image payload conversion"
```

---

## Task 15: `VLLMBackend.chat_stream()`

**Files:**
- Modify: `src/cognithor/core/vllm_backend.py`
- Modify: `tests/test_core/test_vllm_backend.py`

- [ ] **Step 1: Write the failing test**

```python
class TestVLLMBackendChatStream:
    @pytest.mark.asyncio
    async def test_stream_yields_content_chunks(self, backend, httpx_mock):
        # vLLM SSE format: 'data: {"choices":[{"delta":{"content":"..."}}]}\n\n'
        sse_lines = (
            b'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n'
            b'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
            b'data: [DONE]\n\n'
        )
        httpx_mock.add_response(
            url=f"{BASE_URL}/chat/completions",
            status_code=200,
            content=sse_lines,
            headers={"content-type": "text/event-stream"},
        )

        chunks: list[str] = []
        async for piece in backend.chat_stream(
            model="x",
            messages=[{"role": "user", "content": "hi"}],
        ):
            chunks.append(piece)

        assert "".join(chunks) == "Hello"

    @pytest.mark.asyncio
    async def test_stream_raises_on_5xx(self, backend, httpx_mock):
        httpx_mock.add_response(
            url=f"{BASE_URL}/chat/completions",
            status_code=503,
        )
        with pytest.raises(VLLMNotReadyError):
            async for _ in backend.chat_stream(
                model="x",
                messages=[{"role": "user", "content": "hi"}],
            ):
                pass
```

- [ ] **Step 2: Run test**

```bash
python -m pytest tests/test_core/test_vllm_backend.py::TestVLLMBackendChatStream -v
```
Expected: `NotImplementedError`

- [ ] **Step 3: Implement `chat_stream()`**

Replace the stub with:

```python
import json as _json


async def chat_stream(
    self,
    model: str,
    messages: list[dict[str, Any]],
    *,
    temperature: float = 0.7,
    top_p: float = 0.9,
    images: list[str] | None = None,
) -> AsyncIterator[str]:
    """Stream response tokens from vLLM. Parses OpenAI SSE format."""
    if images:
        messages = _attach_images_to_last_user(messages, images)

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "stream": True,
    }
    client = await self._ensure_client()
    try:
        async with client.stream(
            "POST",
            f"{self._base_url}/chat/completions",
            json=payload,
        ) as r:
            if r.status_code >= 500:
                raise VLLMNotReadyError(f"vLLM streaming returned {r.status_code}")
            if r.status_code == 400:
                raise LLMBadRequestError(f"vLLM rejected stream request: {r.status_code}")
            if r.status_code >= 400:
                raise LLMBackendError(f"vLLM streaming returned {r.status_code}")

            async for line in r.aiter_lines():
                line = line.strip()
                if not line or not line.startswith("data:"):
                    continue
                payload_str = line[5:].strip()
                if payload_str == "[DONE]":
                    return
                try:
                    event = _json.loads(payload_str)
                except _json.JSONDecodeError:
                    continue
                choices = event.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                piece = delta.get("content")
                if piece:
                    yield piece
    except httpx.RequestError as exc:
        raise VLLMNotReadyError(f"vLLM stream not reachable: {exc}") from exc
```

- [ ] **Step 4: Run test**

```bash
python -m pytest tests/test_core/test_vllm_backend.py::TestVLLMBackendChatStream -v
```
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
python -m ruff check src/cognithor/core/vllm_backend.py tests/test_core/test_vllm_backend.py && python -m ruff format --check src/cognithor/core/vllm_backend.py tests/test_core/test_vllm_backend.py
git add src/cognithor/core/vllm_backend.py tests/test_core/test_vllm_backend.py
git commit -m "feat(vllm): VLLMBackend.chat_stream() parsing OpenAI SSE"
```

---

## Task 16: `VLLMBackend.embed()`

**Files:**
- Modify: `src/cognithor/core/vllm_backend.py`
- Modify: `tests/test_core/test_vllm_backend.py`

- [ ] **Step 1: Write the failing test**

```python
class TestVLLMBackendEmbed:
    @pytest.mark.asyncio
    async def test_embed_returns_vector(self, backend, httpx_mock):
        httpx_mock.add_response(
            url=f"{BASE_URL}/embeddings",
            status_code=200,
            json={"data": [{"embedding": [0.1, 0.2, 0.3]}], "model": "embed-model"},
        )
        resp = await backend.embed(model="embed-model", text="hello")
        assert resp.embedding == [0.1, 0.2, 0.3]
        assert resp.model == "embed-model"

    @pytest.mark.asyncio
    async def test_embed_raises_when_model_doesnt_support(self, backend, httpx_mock):
        # vLLM returns 400 when the loaded model has no embedding head
        httpx_mock.add_response(
            url=f"{BASE_URL}/embeddings",
            status_code=400,
            json={"error": "not an embedding model"},
        )
        with pytest.raises(LLMBadRequestError):
            await backend.embed(model="qwen-chat-only", text="hello")
```

- [ ] **Step 2: Run test**

```bash
python -m pytest tests/test_core/test_vllm_backend.py::TestVLLMBackendEmbed -v
```
Expected: `NotImplementedError`

- [ ] **Step 3: Implement `embed()`**

```python
async def embed(self, model: str, text: str) -> EmbedResponse:
    client = await self._ensure_client()
    try:
        r = await client.post(
            f"{self._base_url}/embeddings",
            json={"model": model, "input": text},
        )
    except httpx.RequestError as exc:
        raise VLLMNotReadyError(f"vLLM embed not reachable: {exc}") from exc

    if r.status_code == 400:
        raise LLMBadRequestError(f"vLLM embed rejected: {r.text[:200]}")
    if r.status_code >= 500:
        raise VLLMNotReadyError(f"vLLM embed 5xx: {r.status_code}")
    if r.status_code >= 400:
        raise LLMBackendError(f"vLLM embed: {r.status_code}")

    data = r.json()
    items = data.get("data", [])
    if not items:
        raise LLMBackendError("vLLM embed returned no data")
    return EmbedResponse(embedding=items[0].get("embedding", []), model=data.get("model", model))
```

- [ ] **Step 4: Run test**

```bash
python -m pytest tests/test_core/test_vllm_backend.py -v
```
Expected: all VLLMBackend tests pass

- [ ] **Step 5: Commit**

```bash
python -m ruff check src/cognithor/core/vllm_backend.py tests/test_core/test_vllm_backend.py && python -m ruff format --check src/cognithor/core/vllm_backend.py tests/test_core/test_vllm_backend.py
git add src/cognithor/core/vllm_backend.py tests/test_core/test_vllm_backend.py
git commit -m "feat(vllm): VLLMBackend.embed() for embedding-capable models"
```

---

## Task 17: Register `VLLMBackend` in `create_backend()` Factory

**Files:**
- Modify: `src/cognithor/core/llm_backend.py` (add vllm branch to `create_backend()`)
- Create: `tests/test_core/test_create_backend_vllm.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_core/test_create_backend_vllm.py
from __future__ import annotations

from cognithor.config import CognithorConfig, VLLMConfig
from cognithor.core.llm_backend import create_backend
from cognithor.core.vllm_backend import VLLMBackend


class TestCreateVLLMBackend:
    def test_returns_vllm_backend_when_config_says_vllm(self):
        cfg = CognithorConfig(
            llm_backend_type="vllm",
            vllm=VLLMConfig(enabled=True, port=8000),
        )
        backend = create_backend(cfg)
        assert isinstance(backend, VLLMBackend)
        assert backend.backend_type.value == "vllm"

    def test_vllm_backend_uses_configured_port(self):
        cfg = CognithorConfig(
            llm_backend_type="vllm",
            vllm=VLLMConfig(enabled=True, port=8042),
        )
        backend = create_backend(cfg)
        assert backend._base_url == "http://localhost:8042/v1"

    def test_vllm_backend_uses_request_timeout(self):
        cfg = CognithorConfig(
            llm_backend_type="vllm",
            vllm=VLLMConfig(enabled=True, request_timeout_seconds=30),
        )
        backend = create_backend(cfg)
        assert backend._timeout == 30
```

- [ ] **Step 2: Run test**

```bash
python -m pytest tests/test_core/test_create_backend_vllm.py -v
```
Expected: something like `ValueError: Unsupported backend: vllm` — `create_backend()` doesn't know vllm yet.

- [ ] **Step 3: Add vllm branch to `create_backend()`**

Find `create_backend(config)` in `src/cognithor/core/llm_backend.py` (near the bottom). Add a branch:

```python
def create_backend(config: CognithorConfig) -> LLMBackend:
    btype = config.llm_backend_type
    # ... existing branches ...
    if btype == "vllm":
        from cognithor.core.vllm_backend import VLLMBackend
        return VLLMBackend(
            base_url=f"http://localhost:{config.vllm.port}/v1",
            timeout=config.vllm.request_timeout_seconds,
        )
    # ... fallback / raise ...
```

Place the branch before the "raise unsupported" tail. Import inside the function (lazy) to avoid circular imports.

- [ ] **Step 4: Run test**

```bash
python -m pytest tests/test_core/test_create_backend_vllm.py -v
```
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
python -m ruff check src/cognithor/core/llm_backend.py tests/test_core/test_create_backend_vllm.py && python -m ruff format --check src/cognithor/core/llm_backend.py tests/test_core/test_create_backend_vllm.py
git add src/cognithor/core/llm_backend.py tests/test_core/test_create_backend_vllm.py
git commit -m "feat(llm): wire VLLMBackend into create_backend() factory"
```

---

## Task 18: `UnifiedLLMClient` — Per-Backend CircuitBreaker + `backend_status`

**Files:**
- Modify: `src/cognithor/core/unified_llm.py`
- Create: `tests/test_core/test_unified_llm_circuit_breaker.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_core/test_unified_llm_circuit_breaker.py
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from cognithor.core.llm_backend import (
    ChatResponse,
    LLMBadRequestError,
    VLLMNotReadyError,
)
from cognithor.core.unified_llm import BackendStatus, UnifiedLLMClient
from cognithor.utils.circuit_breaker import CircuitState


@pytest.fixture
def mock_vllm_backend() -> AsyncMock:
    mock = AsyncMock()
    mock.backend_type = "vllm"
    return mock


@pytest.fixture
def mock_ollama_client() -> AsyncMock:
    return AsyncMock()


class TestBreakerWiring:
    @pytest.mark.asyncio
    async def test_three_consecutive_failures_open_breaker(
        self, mock_vllm_backend, mock_ollama_client
    ):
        mock_vllm_backend.chat.side_effect = VLLMNotReadyError("down")
        client = UnifiedLLMClient(
            ollama_client=mock_ollama_client,
            backend=mock_vllm_backend,
        )
        # Reject images → trigger errors
        for _ in range(3):
            try:
                await client.chat(model="x", messages=[{"role": "user", "content": "hi"}])
            except Exception:
                pass
        assert client.vllm_breaker.state == CircuitState.open
        assert client.backend_status == BackendStatus.DEGRADED

    @pytest.mark.asyncio
    async def test_bad_request_error_is_excluded_from_breaker(
        self, mock_vllm_backend, mock_ollama_client
    ):
        mock_vllm_backend.chat.side_effect = LLMBadRequestError("context too long")
        client = UnifiedLLMClient(
            ollama_client=mock_ollama_client,
            backend=mock_vllm_backend,
        )
        for _ in range(5):
            try:
                await client.chat(model="x", messages=[{"role": "user", "content": "hi"}])
            except LLMBadRequestError:
                pass
        assert client.vllm_breaker.state == CircuitState.closed
        assert client.backend_status == BackendStatus.OK

    @pytest.mark.asyncio
    async def test_half_open_probe_success_closes_breaker(
        self, mock_vllm_backend, mock_ollama_client
    ):
        mock_vllm_backend.chat.side_effect = VLLMNotReadyError("down")
        client = UnifiedLLMClient(
            ollama_client=mock_ollama_client,
            backend=mock_vllm_backend,
            _breaker_recovery_timeout=0.05,  # fast for test
        )
        # Open breaker
        for _ in range(3):
            try:
                await client.chat(model="x", messages=[{"role": "user", "content": "hi"}])
            except Exception:
                pass
        assert client.vllm_breaker.state == CircuitState.open

        # Wait past recovery_timeout, reset mock to success
        import asyncio
        await asyncio.sleep(0.1)
        mock_vllm_backend.chat.side_effect = None
        mock_vllm_backend.chat.return_value = ChatResponse(content="ok", model="x")

        await client.chat(model="x", messages=[{"role": "user", "content": "hi"}])
        assert client.vllm_breaker.state == CircuitState.closed
        assert client.backend_status == BackendStatus.OK
```

- [ ] **Step 2: Run test**

```bash
python -m pytest tests/test_core/test_unified_llm_circuit_breaker.py -v
```
Expected: various attribute errors (`vllm_breaker`, `backend_status`, `BackendStatus`)

- [ ] **Step 3: Extend `UnifiedLLMClient` with breaker + status**

In `src/cognithor/core/unified_llm.py`:

1. Add imports at the top:

```python
from enum import StrEnum

from cognithor.core.llm_backend import LLMBadRequestError
from cognithor.utils.circuit_breaker import CircuitBreaker, CircuitBreakerOpen, CircuitState
```

2. Add enum:

```python
class BackendStatus(StrEnum):
    """Public-facing health state surfaced to the UI."""

    OK = "ok"
    DEGRADED = "degraded"
```

3. Extend `UnifiedLLMClient.__init__` to accept `_breaker_recovery_timeout` kwarg (test hook) and create per-backend breakers:

```python
def __init__(
    self,
    ollama_client: OllamaClient | None,
    backend: Any | None = None,
    config: CognithorConfig | None = None,
    *,
    _breaker_recovery_timeout: float = 60.0,
) -> None:
    self._ollama = ollama_client
    self._backend = backend
    self._config = config
    self._backend_type: str = "ollama"
    self._backend_cache: dict[str, Any] = {}
    # Per-backend circuit breakers
    self.vllm_breaker = CircuitBreaker(
        name="llm_backend_vllm",
        failure_threshold=3,
        recovery_timeout=_breaker_recovery_timeout,
        half_open_max_calls=1,
        excluded_exceptions=(LLMBadRequestError,),
    )
    self.ollama_breaker = CircuitBreaker(
        name="llm_backend_ollama",
        failure_threshold=3,
        recovery_timeout=_breaker_recovery_timeout,
        half_open_max_calls=1,
        excluded_exceptions=(LLMBadRequestError,),
    )
    self.backend_status: BackendStatus = BackendStatus.OK
    if backend is not None:
        self._backend_type = getattr(backend, "backend_type", "unknown")
```

4. Wrap the `chat()` dispatch method to use the relevant breaker AND to update `backend_status`:

```python
def _breaker_for(self, backend_type: str) -> CircuitBreaker:
    if backend_type == "vllm":
        return self.vllm_breaker
    return self.ollama_breaker


def _refresh_status(self) -> None:
    """Recompute backend_status from breaker state."""
    breaker = self._breaker_for(self._backend_type)
    if breaker.state == CircuitState.closed:
        self.backend_status = BackendStatus.OK
    else:
        self.backend_status = BackendStatus.DEGRADED


async def chat(self, *args, **kwargs):
    """Dispatch chat with per-backend circuit breaker protection."""
    if self._backend is None:
        # Legacy Ollama path — wrap too
        breaker = self.ollama_breaker
        try:
            result = await breaker.call(self._ollama.chat(*args, **kwargs))
        finally:
            self._refresh_status()
        return result

    breaker = self._breaker_for(self._backend_type)
    try:
        result = await breaker.call(self._backend.chat(*args, **kwargs))
    finally:
        self._refresh_status()
    return result
```

(Update any existing `chat()` implementation to go through the breaker; preserve the existing response-normalization logic if any.)

- [ ] **Step 4: Run test**

```bash
python -m pytest tests/test_core/test_unified_llm_circuit_breaker.py -v
```
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
python -m ruff check src/cognithor/core/unified_llm.py tests/test_core/test_unified_llm_circuit_breaker.py && python -m ruff format --check src/cognithor/core/unified_llm.py tests/test_core/test_unified_llm_circuit_breaker.py
git add src/cognithor/core/unified_llm.py tests/test_core/test_unified_llm_circuit_breaker.py
git commit -m "feat(unified-llm): per-backend CircuitBreaker + BackendStatus enum"
```

---

## Task 19: `UnifiedLLMClient` — Situational Fail-Flow Dispatch

**Files:**
- Modify: `src/cognithor/core/unified_llm.py`
- Modify: `tests/test_core/test_unified_llm_circuit_breaker.py`

- [ ] **Step 1: Write the failing test**

```python
class TestFailFlowDispatch:
    @pytest.mark.asyncio
    async def test_text_request_falls_back_to_ollama_when_vllm_degraded(
        self, mock_vllm_backend, mock_ollama_client
    ):
        mock_vllm_backend.chat.side_effect = VLLMNotReadyError("down")
        # Ollama answers successfully
        mock_ollama_client.chat = AsyncMock(return_value={"message": {"content": "fallback answer"}})

        client = UnifiedLLMClient(
            ollama_client=mock_ollama_client,
            backend=mock_vllm_backend,
            _breaker_recovery_timeout=60.0,
        )
        # Trip the breaker
        for _ in range(3):
            try:
                await client.chat(model="x", messages=[{"role": "user", "content": "hi"}])
            except Exception:
                pass

        # Now a pure text request — no images — must fall back silently
        result = await client.chat(model="x", messages=[{"role": "user", "content": "hi"}])
        # result shape is Ollama-style dict
        assert "fallback answer" in str(result)

    @pytest.mark.asyncio
    async def test_image_request_hard_errors_when_vllm_degraded(
        self, mock_vllm_backend, mock_ollama_client, tmp_path
    ):
        mock_vllm_backend.chat.side_effect = VLLMNotReadyError("down")
        img = tmp_path / "pic.png"
        img.write_bytes(b"\x89PNG")

        client = UnifiedLLMClient(
            ollama_client=mock_ollama_client,
            backend=mock_vllm_backend,
            _breaker_recovery_timeout=60.0,
        )
        # Trip the breaker
        for _ in range(3):
            try:
                await client.chat(model="x", messages=[{"role": "user", "content": "hi"}])
            except Exception:
                pass

        # Image request — Ollama can't do vision → must raise
        with pytest.raises(VLLMNotReadyError):
            await client.chat(
                model="x",
                messages=[{"role": "user", "content": "what is this?"}],
                images=[str(img)],
            )
```

- [ ] **Step 2: Run test**

```bash
python -m pytest tests/test_core/test_unified_llm_circuit_breaker.py::TestFailFlowDispatch -v
```
Expected: test fails because `chat()` doesn't dispatch between text / image yet

- [ ] **Step 3: Add situational fallback to `chat()`**

Replace the `chat()` method from Task 18 with:

```python
async def chat(self, *args, images: list[str] | None = None, **kwargs):
    """Dispatch chat. Text-requests may fall back to Ollama when vLLM is
    DEGRADED; image-requests raise because Ollama cannot do vision."""
    is_image_request = bool(images)

    # Try primary backend
    breaker = self._breaker_for(self._backend_type)
    try:
        if self._backend is not None:
            result = await breaker.call(
                self._backend.chat(*args, images=images, **kwargs)
            )
        else:
            result = await breaker.call(self._ollama.chat(*args, **kwargs))
        self._refresh_status()
        return result
    except (VLLMNotReadyError, CircuitBreakerOpen) as exc:
        self._refresh_status()
        if is_image_request:
            # No silent fallback — Ollama can't see images
            if isinstance(exc, CircuitBreakerOpen):
                raise VLLMNotReadyError(
                    "vLLM offline — cannot process image",
                    recovery_hint="Start vLLM from LLM Backends settings.",
                ) from exc
            raise
        if self._backend_type == "vllm" and self._ollama is not None:
            log.warning("vllm_fallback_to_ollama")
            return await self._ollama.chat(*args, **kwargs)
        raise
```

Import `VLLMNotReadyError` at module top.

- [ ] **Step 4: Run test**

```bash
python -m pytest tests/test_core/test_unified_llm_circuit_breaker.py -v
```
Expected: all circuit-breaker tests pass

- [ ] **Step 5: Commit**

```bash
python -m ruff check src/cognithor/core/unified_llm.py tests/test_core/test_unified_llm_circuit_breaker.py && python -m ruff format --check src/cognithor/core/unified_llm.py tests/test_core/test_unified_llm_circuit_breaker.py
git add src/cognithor/core/unified_llm.py tests/test_core/test_unified_llm_circuit_breaker.py
git commit -m "feat(unified-llm): situational fail-flow — text→Ollama, image→hard error"
```

---

## Task 20: Integration Test — Fake vLLM Server

**Files:**
- Create: `tests/test_integration/test_vllm_fake_server.py`

- [ ] **Step 1: Write the integration test**

```python
# tests/test_integration/test_vllm_fake_server.py
"""End-to-end test: VLLMBackend against a FastAPI impersonating vLLM's OpenAI API.

No GPU, no Docker, no real vLLM. Verifies the full request-response round-trip
on the HTTP layer with real (mocked-server) sockets.
"""

from __future__ import annotations

import asyncio
import json as _json

import pytest
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import uvicorn

from cognithor.core.vllm_backend import VLLMBackend


def _build_fake_vllm_app() -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    async def chat(body: dict):
        if body.get("stream"):
            async def gen():
                for chunk in ["Hel", "lo ", "world"]:
                    payload = _json.dumps({"choices": [{"delta": {"content": chunk}}]})
                    yield f"data: {payload}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(gen(), media_type="text/event-stream")
        return {
            "choices": [{"message": {"content": "echo: " + body["messages"][-1]["content"]}}],
            "model": body["model"],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    @app.get("/v1/models")
    async def models():
        return {"data": [{"id": "fake-model"}]}

    return app


@pytest.fixture
async def fake_server():
    """Start the fake server on an ephemeral port."""
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()

    config = uvicorn.Config(
        _build_fake_vllm_app(),
        host="127.0.0.1",
        port=port,
        log_level="error",
    )
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    # Wait for startup
    for _ in range(50):
        if server.started:
            break
        await asyncio.sleep(0.05)
    yield port
    server.should_exit = True
    await task


class TestVLLMBackendEndToEnd:
    @pytest.mark.asyncio
    async def test_full_chat_roundtrip(self, fake_server):
        backend = VLLMBackend(base_url=f"http://127.0.0.1:{fake_server}/v1")
        try:
            resp = await backend.chat(
                model="fake-model",
                messages=[{"role": "user", "content": "hello"}],
            )
            assert resp.content == "echo: hello"
            assert resp.model == "fake-model"
        finally:
            await backend.close()

    @pytest.mark.asyncio
    async def test_full_stream_roundtrip(self, fake_server):
        backend = VLLMBackend(base_url=f"http://127.0.0.1:{fake_server}/v1")
        try:
            chunks = []
            async for p in backend.chat_stream(
                model="fake-model",
                messages=[{"role": "user", "content": "stream me"}],
            ):
                chunks.append(p)
            assert "".join(chunks) == "Hello world"
        finally:
            await backend.close()

    @pytest.mark.asyncio
    async def test_is_available_roundtrip(self, fake_server):
        backend = VLLMBackend(base_url=f"http://127.0.0.1:{fake_server}/v1")
        try:
            assert await backend.is_available() is True
        finally:
            await backend.close()

    @pytest.mark.asyncio
    async def test_list_models_roundtrip(self, fake_server):
        backend = VLLMBackend(base_url=f"http://127.0.0.1:{fake_server}/v1")
        try:
            models = await backend.list_models()
            assert "fake-model" in models
        finally:
            await backend.close()
```

- [ ] **Step 2: Run test**

```bash
python -m pytest tests/test_integration/test_vllm_fake_server.py -v
```
Expected: `4 passed` in < 5 seconds

- [ ] **Step 3: Commit**

```bash
python -m ruff check tests/test_integration/test_vllm_fake_server.py && python -m ruff format --check tests/test_integration/test_vllm_fake_server.py
git add tests/test_integration/test_vllm_fake_server.py
git commit -m "test(vllm): integration test against a fake OpenAI-compatible server"
```

---

## Task 21: FastAPI — `GET /api/backends` + `GET /api/backends/vllm/status`

**Files:**
- Modify: `src/cognithor/channels/api.py` (add router section for backends)
- Create: `tests/test_channels/test_api_backends.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_channels/test_api_backends.py
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client_with_vllm_enabled():
    """TestClient with a Cognithor API where vLLM is enabled in config."""
    # Import here to avoid module-level side effects
    from cognithor.channels.api import build_app
    from cognithor.config import CognithorConfig, VLLMConfig

    cfg = CognithorConfig(
        llm_backend_type="ollama",  # Ollama stays default
        vllm=VLLMConfig(enabled=True),
    )
    app = build_app(config=cfg)
    return TestClient(app), cfg


class TestBackendsList:
    def test_lists_all_backends_with_status(self, client_with_vllm_enabled):
        client, _ = client_with_vllm_enabled
        r = client.get("/api/backends")
        assert r.status_code == 200
        data = r.json()
        # Response shape: {"active": "ollama", "backends": [...]}
        assert data["active"] == "ollama"
        names = {b["name"] for b in data["backends"]}
        assert "ollama" in names
        assert "vllm" in names


class TestVLLMStatus:
    def test_status_returns_current_vllm_state(self, client_with_vllm_enabled):
        client, _ = client_with_vllm_enabled
        # Patch orchestrator.status() to return a known snapshot
        with patch(
            "cognithor.core.vllm_orchestrator.VLLMOrchestrator.status"
        ) as mock:
            from cognithor.core.vllm_orchestrator import (
                DockerInfo, HardwareInfo, VLLMState,
            )
            mock.return_value = VLLMState(
                hardware_ok=True,
                hardware_info=HardwareInfo("RTX 5090", 32, (12, 0)),
                docker_ok=True,
                docker_info=DockerInfo(True, "26.0.0", True),
                image_pulled=False,
                container_running=False,
                current_model=None,
            )
            r = client.get("/api/backends/vllm/status")
        assert r.status_code == 200
        data = r.json()
        assert data["hardware_ok"] is True
        assert data["hardware_info"]["gpu_name"] == "RTX 5090"
        assert data["hardware_info"]["vram_gb"] == 32
        assert data["hardware_info"]["compute_capability"] == "12.0"
        assert data["docker_ok"] is True
        assert data["container_running"] is False
```

- [ ] **Step 2: Run test**

```bash
python -m pytest tests/test_channels/test_api_backends.py -v
```
Expected: fails — endpoints don't exist.

- [ ] **Step 3: Add endpoints to `src/cognithor/channels/api.py`**

Find where the existing `build_app(config)` function lives. Add a new router:

```python
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from cognithor.core.vllm_orchestrator import VLLMOrchestrator


backends_router = APIRouter(prefix="/api/backends", tags=["backends"])
_orchestrator: VLLMOrchestrator | None = None


def _get_orchestrator(config) -> VLLMOrchestrator:
    """Lazy-initialized module-level singleton for the vLLM orchestrator."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = VLLMOrchestrator(
            docker_image=config.vllm.docker_image,
            port=config.vllm.port,
            hf_token=config.huggingface_api_key,
        )
    return _orchestrator


@backends_router.get("")
async def list_backends(request):
    config = request.app.state.config
    backends = []
    # Ollama — assume always available; caller will probe if needed
    backends.append({
        "name": "ollama",
        "enabled": config.llm_backend_type == "ollama",
        "status": "ready",
    })
    # vLLM — read orchestrator state
    orch = _get_orchestrator(config)
    st = orch.status()
    vllm_status = "ready" if st.container_running else (
        "configured" if config.vllm.enabled else "disabled"
    )
    backends.append({
        "name": "vllm",
        "enabled": config.vllm.enabled,
        "status": vllm_status,
    })
    return {
        "active": config.llm_backend_type,
        "backends": backends,
    }


@backends_router.get("/vllm/status")
async def vllm_status(request):
    config = request.app.state.config
    orch = _get_orchestrator(config)
    st = orch.status()
    hw = None
    if st.hardware_info:
        hw = {
            "gpu_name": st.hardware_info.gpu_name,
            "vram_gb": st.hardware_info.vram_gb,
            "compute_capability": st.hardware_info.sm_string,
        }
    return {
        "hardware_ok": st.hardware_ok,
        "hardware_info": hw,
        "docker_ok": st.docker_ok,
        "image_pulled": st.image_pulled,
        "container_running": st.container_running,
        "current_model": st.current_model,
        "last_error": st.last_error,
    }
```

In `build_app()`, include the router: `app.include_router(backends_router)`. Also store the config on app state: `app.state.config = config`.

The `request` parameter uses FastAPI dependency injection — adjust signatures to the existing style of `channels/api.py` (may need `Request` object import from `fastapi`).

- [ ] **Step 4: Run test**

```bash
python -m pytest tests/test_channels/test_api_backends.py -v
```
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
python -m ruff check src/cognithor/channels/api.py tests/test_channels/test_api_backends.py && python -m ruff format --check src/cognithor/channels/api.py tests/test_channels/test_api_backends.py
git add src/cognithor/channels/api.py tests/test_channels/test_api_backends.py
git commit -m "feat(api): GET /api/backends + /api/backends/vllm/status"
```

---

## Task 22: FastAPI — `POST /api/backends/vllm/check-hardware`, `/start`, `/stop`

**Files:**
- Modify: `src/cognithor/channels/api.py`
- Modify: `tests/test_channels/test_api_backends.py`

- [ ] **Step 1: Write the failing tests**

```python
class TestVLLMActions:
    def test_check_hardware_delegates_to_orchestrator(
        self, client_with_vllm_enabled
    ):
        client, _ = client_with_vllm_enabled
        from cognithor.core.vllm_orchestrator import HardwareInfo

        with patch(
            "cognithor.core.vllm_orchestrator.VLLMOrchestrator.check_hardware",
            return_value=HardwareInfo("RTX 5090", 32, (12, 0)),
        ):
            r = client.post("/api/backends/vllm/check-hardware")
        assert r.status_code == 200
        assert r.json()["gpu_name"] == "RTX 5090"
        assert r.json()["vram_gb"] == 32
        assert r.json()["compute_capability"] == "12.0"

    def test_check_hardware_returns_503_on_no_gpu(self, client_with_vllm_enabled):
        client, _ = client_with_vllm_enabled
        from cognithor.core.llm_backend import VLLMHardwareError

        with patch(
            "cognithor.core.vllm_orchestrator.VLLMOrchestrator.check_hardware",
            side_effect=VLLMHardwareError("No GPU", recovery_hint="Install NVIDIA driver"),
        ):
            r = client.post("/api/backends/vllm/check-hardware")
        assert r.status_code == 503
        body = r.json()
        assert "No GPU" in body["detail"]["message"]
        assert body["detail"]["recovery_hint"] == "Install NVIDIA driver"

    def test_start_container_accepts_model(self, client_with_vllm_enabled):
        client, _ = client_with_vllm_enabled
        from cognithor.core.vllm_orchestrator import ContainerInfo

        with patch(
            "cognithor.core.vllm_orchestrator.VLLMOrchestrator.start_container",
            return_value=ContainerInfo("abc123", 8000, "Qwen/Qwen3.6-27B-FP8"),
        ):
            r = client.post(
                "/api/backends/vllm/start",
                json={"model": "Qwen/Qwen3.6-27B-FP8"},
            )
        assert r.status_code == 200
        data = r.json()
        assert data["container_id"] == "abc123"
        assert data["port"] == 8000
        assert data["model"] == "Qwen/Qwen3.6-27B-FP8"

    def test_stop_container(self, client_with_vllm_enabled):
        client, _ = client_with_vllm_enabled
        with patch(
            "cognithor.core.vllm_orchestrator.VLLMOrchestrator.stop_container"
        ) as stop_mock:
            r = client.post("/api/backends/vllm/stop")
        assert r.status_code == 200
        stop_mock.assert_called_once()

    def test_logs_endpoint(self, client_with_vllm_enabled):
        client, _ = client_with_vllm_enabled
        with patch(
            "cognithor.core.vllm_orchestrator.VLLMOrchestrator.get_logs",
            return_value=["line1", "line2"],
        ):
            r = client.get("/api/backends/vllm/logs")
        assert r.status_code == 200
        assert r.json()["lines"] == ["line1", "line2"]
```

- [ ] **Step 2: Run tests**

```bash
python -m pytest tests/test_channels/test_api_backends.py::TestVLLMActions -v
```
Expected: endpoints missing.

- [ ] **Step 3: Add endpoints**

```python
class StartRequest(BaseModel):
    model: str


@backends_router.post("/vllm/check-hardware")
async def check_hardware_endpoint(request):
    config = request.app.state.config
    orch = _get_orchestrator(config)
    try:
        info = orch.check_hardware()
    except Exception as exc:
        from cognithor.core.llm_backend import VLLMHardwareError
        if isinstance(exc, VLLMHardwareError):
            raise HTTPException(
                status_code=503,
                detail={"message": str(exc), "recovery_hint": exc.recovery_hint},
            ) from exc
        raise HTTPException(status_code=500, detail={"message": str(exc)}) from exc
    return {
        "gpu_name": info.gpu_name,
        "vram_gb": info.vram_gb,
        "compute_capability": info.sm_string,
    }


@backends_router.post("/vllm/start")
async def vllm_start(request, body: StartRequest):
    config = request.app.state.config
    orch = _get_orchestrator(config)
    try:
        info = orch.start_container(body.model)
    except Exception as exc:
        from cognithor.core.llm_backend import VLLMNotReadyError
        if isinstance(exc, VLLMNotReadyError):
            raise HTTPException(
                status_code=503,
                detail={"message": str(exc), "recovery_hint": getattr(exc, "recovery_hint", "")},
            ) from exc
        raise HTTPException(status_code=500, detail={"message": str(exc)}) from exc
    return {"container_id": info.container_id, "port": info.port, "model": info.model}


@backends_router.post("/vllm/stop")
async def vllm_stop(request):
    config = request.app.state.config
    orch = _get_orchestrator(config)
    orch.stop_container()
    return {"status": "stopped"}


@backends_router.get("/vllm/logs")
async def vllm_logs(request):
    config = request.app.state.config
    orch = _get_orchestrator(config)
    return {"lines": orch.get_logs()}
```

- [ ] **Step 4: Run test**

```bash
python -m pytest tests/test_channels/test_api_backends.py -v
```
Expected: all passed

- [ ] **Step 5: Commit**

```bash
python -m ruff check src/cognithor/channels/api.py tests/test_channels/test_api_backends.py && python -m ruff format --check src/cognithor/channels/api.py tests/test_channels/test_api_backends.py
git add src/cognithor/channels/api.py tests/test_channels/test_api_backends.py
git commit -m "feat(api): vLLM check-hardware / start / stop / logs endpoints"
```

---

## Task 23: FastAPI — SSE `/api/backends/vllm/pull-image`

**Files:**
- Modify: `src/cognithor/channels/api.py`
- Modify: `tests/test_channels/test_api_backends.py`

- [ ] **Step 1: Write the failing test**

```python
class TestPullImageSSE:
    def test_pull_image_streams_sse_events(self, client_with_vllm_enabled):
        client, _ = client_with_vllm_enabled

        # Fake orchestrator.pull_image() — invoke the callback with progress events
        def fake_pull(tag, progress_callback=None):
            if progress_callback:
                progress_callback({"status": "Pulling", "id": "layer1"})
                progress_callback({"status": "Downloading", "id": "layer1",
                                   "progressDetail": {"current": 500, "total": 1000}})
                progress_callback({"status": "Download complete", "id": "layer1"})

        with patch(
            "cognithor.core.vllm_orchestrator.VLLMOrchestrator.pull_image",
            side_effect=fake_pull,
        ):
            with client.stream("POST", "/api/backends/vllm/pull-image") as r:
                assert r.status_code == 200
                assert r.headers["content-type"].startswith("text/event-stream")
                lines = list(r.iter_lines())

        # SSE format: 'data: {...}\n\n' — blank lines separate events
        data_lines = [l for l in lines if l.startswith("data:")]
        assert len(data_lines) >= 3
        assert any("Downloading" in l for l in data_lines)
```

- [ ] **Step 2: Run test**

```bash
python -m pytest tests/test_channels/test_api_backends.py::TestPullImageSSE -v
```
Expected: fails — no endpoint.

- [ ] **Step 3: Implement SSE streaming endpoint**

```python
import asyncio
from fastapi.responses import StreamingResponse


@backends_router.post("/vllm/pull-image")
async def vllm_pull_image(request):
    """Stream docker-pull progress to the client as SSE."""
    config = request.app.state.config
    orch = _get_orchestrator(config)

    queue: asyncio.Queue[dict | None] = asyncio.Queue()

    def progress_cb(event: dict) -> None:
        queue.put_nowait(event)

    async def worker() -> None:
        """Run the blocking pull_image in a thread, enqueue events, sentinel at end."""
        await asyncio.to_thread(
            orch.pull_image, config.vllm.docker_image, progress_callback=progress_cb
        )
        queue.put_nowait(None)  # sentinel

    async def event_stream():
        task = asyncio.create_task(worker())
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield f"data: {_json.dumps(event)}\n\n"
        finally:
            await task

    import json as _json
    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

Move `import json as _json` and `import asyncio` to module level if not already there.

- [ ] **Step 4: Run test**

```bash
python -m pytest tests/test_channels/test_api_backends.py::TestPullImageSSE -v
```
Expected: passes

- [ ] **Step 5: Commit**

```bash
python -m ruff check src/cognithor/channels/api.py tests/test_channels/test_api_backends.py && python -m ruff format --check src/cognithor/channels/api.py tests/test_channels/test_api_backends.py
git add src/cognithor/channels/api.py tests/test_channels/test_api_backends.py
git commit -m "feat(api): SSE streaming endpoint for docker-pull progress"
```

---

## Task 24: FastAPI — `POST /api/backends/active` (UnifiedLLMClient re-init)

**Files:**
- Modify: `src/cognithor/channels/api.py`
- Modify: `tests/test_channels/test_api_backends.py`
- Modify: `src/cognithor/gateway/gateway.py:1968+` (extract helper for re-init)

- [ ] **Step 1: Write the failing test**

```python
class TestSetActiveBackend:
    def test_switch_to_vllm_reinits_unified_client(self, client_with_vllm_enabled):
        client, cfg = client_with_vllm_enabled
        # Gateway has a `rebuild_llm_client(new_backend_type)` hook
        with patch(
            "cognithor.gateway.gateway.Gateway.rebuild_llm_client"
        ) as rebuild:
            r = client.post("/api/backends/active", json={"backend": "vllm"})
        assert r.status_code == 200
        rebuild.assert_called_once_with("vllm")
        assert r.json()["active"] == "vllm"

    def test_rejects_unknown_backend(self, client_with_vllm_enabled):
        client, _ = client_with_vllm_enabled
        r = client.post("/api/backends/active", json={"backend": "unicorn"})
        assert r.status_code == 400
```

- [ ] **Step 2: Run test**

```bash
python -m pytest tests/test_channels/test_api_backends.py::TestSetActiveBackend -v
```
Expected: fails.

- [ ] **Step 3: Extract `rebuild_llm_client` helper on Gateway + add endpoint**

In `src/cognithor/gateway/gateway.py` near line 1968, extract the re-init logic into a method:

```python
def rebuild_llm_client(self, new_backend_type: str) -> None:
    """Re-init UnifiedLLMClient for a new backend type. Called from the
    FastAPI /api/backends/active endpoint when the user switches in UI."""
    self._config.llm_backend_type = new_backend_type
    from cognithor.core.unified_llm import UnifiedLLMClient
    self._llm = UnifiedLLMClient.create(self._config)
```

In `src/cognithor/channels/api.py`:

```python
from typing import Literal


class SetActiveRequest(BaseModel):
    backend: Literal["ollama", "vllm", "openai", "anthropic", "gemini", "lmstudio", "claude-code"]


@backends_router.post("/active")
async def set_active_backend(request, body: SetActiveRequest):
    gateway = request.app.state.gateway
    gateway.rebuild_llm_client(body.backend)
    return {"active": body.backend}
```

Ensure `build_app()` stores the gateway: `app.state.gateway = gateway`.

- [ ] **Step 4: Run test**

```bash
python -m pytest tests/test_channels/test_api_backends.py -v
```
Expected: all passed

- [ ] **Step 5: Commit**

```bash
python -m ruff check src/cognithor/channels/api.py src/cognithor/gateway/gateway.py tests/test_channels/test_api_backends.py && python -m ruff format --check src/cognithor/channels/api.py src/cognithor/gateway/gateway.py tests/test_channels/test_api_backends.py
git add src/cognithor/channels/api.py src/cognithor/gateway/gateway.py tests/test_channels/test_api_backends.py
git commit -m "feat(api): POST /api/backends/active hot-switches UnifiedLLMClient"
```

---

## Task 25: Flutter — `LlmBackendProvider` (State + 2s Polling)

**Files:**
- Create: `flutter_app/lib/providers/llm_backend_provider.dart`
- Create: `flutter_app/test/providers/llm_backend_provider_test.dart`

- [ ] **Step 1: Write the failing test**

```dart
// flutter_app/test/providers/llm_backend_provider_test.dart
import 'package:cognithor_ui/providers/llm_backend_provider.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('LlmBackendProvider', () {
    test('initial state has empty backends and not polling', () {
      final p = LlmBackendProvider(apiBaseUrl: 'http://localhost:8741');
      expect(p.backends, isEmpty);
      expect(p.vllmStatus, isNull);
      expect(p.isPolling, isFalse);
    });

    test('startPolling sets isPolling true', () {
      final p = LlmBackendProvider(apiBaseUrl: 'http://localhost:8741');
      p.startPolling();
      expect(p.isPolling, isTrue);
      p.stopPolling();
    });

    test('stopPolling resets', () {
      final p = LlmBackendProvider(apiBaseUrl: 'http://localhost:8741');
      p.startPolling();
      p.stopPolling();
      expect(p.isPolling, isFalse);
    });

    test('VLLMStatus.fromJson parses API payload', () {
      final status = VLLMStatus.fromJson({
        'hardware_ok': true,
        'hardware_info': {
          'gpu_name': 'RTX 5090',
          'vram_gb': 32,
          'compute_capability': '12.0',
        },
        'docker_ok': true,
        'image_pulled': false,
        'container_running': false,
        'current_model': null,
        'last_error': null,
      });
      expect(status.hardwareOk, isTrue);
      expect(status.hardwareInfo?.gpuName, 'RTX 5090');
      expect(status.hardwareInfo?.vramGb, 32);
      expect(status.hardwareInfo?.computeCapability, '12.0');
    });
  });
}
```

- [ ] **Step 2: Run test**

```bash
cd flutter_app && flutter test test/providers/llm_backend_provider_test.dart
```
Expected: fails — file doesn't exist.

- [ ] **Step 3: Create the provider**

```dart
// flutter_app/lib/providers/llm_backend_provider.dart
import 'dart:async';
import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;

class HardwareInfo {
  final String gpuName;
  final int vramGb;
  final String computeCapability;

  HardwareInfo({
    required this.gpuName,
    required this.vramGb,
    required this.computeCapability,
  });

  factory HardwareInfo.fromJson(Map<String, dynamic> j) => HardwareInfo(
        gpuName: j['gpu_name'] as String,
        vramGb: j['vram_gb'] as int,
        computeCapability: j['compute_capability'] as String,
      );
}

class VLLMStatus {
  final bool hardwareOk;
  final HardwareInfo? hardwareInfo;
  final bool dockerOk;
  final bool imagePulled;
  final bool containerRunning;
  final String? currentModel;
  final String? lastError;

  VLLMStatus({
    required this.hardwareOk,
    required this.hardwareInfo,
    required this.dockerOk,
    required this.imagePulled,
    required this.containerRunning,
    required this.currentModel,
    required this.lastError,
  });

  factory VLLMStatus.fromJson(Map<String, dynamic> j) => VLLMStatus(
        hardwareOk: j['hardware_ok'] as bool,
        hardwareInfo: j['hardware_info'] == null
            ? null
            : HardwareInfo.fromJson(j['hardware_info'] as Map<String, dynamic>),
        dockerOk: j['docker_ok'] as bool,
        imagePulled: j['image_pulled'] as bool,
        containerRunning: j['container_running'] as bool,
        currentModel: j['current_model'] as String?,
        lastError: j['last_error'] as String?,
      );
}

class BackendEntry {
  final String name;
  final bool enabled;
  final String status;
  BackendEntry({required this.name, required this.enabled, required this.status});
  factory BackendEntry.fromJson(Map<String, dynamic> j) => BackendEntry(
        name: j['name'] as String,
        enabled: j['enabled'] as bool,
        status: j['status'] as String,
      );
}

class LlmBackendProvider extends ChangeNotifier {
  final String apiBaseUrl;
  final http.Client _http;
  Timer? _pollTimer;

  List<BackendEntry> backends = [];
  String active = 'ollama';
  VLLMStatus? vllmStatus;
  String? error;

  bool get isPolling => _pollTimer != null;

  LlmBackendProvider({required this.apiBaseUrl, http.Client? httpClient})
      : _http = httpClient ?? http.Client();

  Future<void> refreshList() async {
    try {
      final r = await _http.get(Uri.parse('$apiBaseUrl/api/backends'));
      if (r.statusCode != 200) return;
      final body = jsonDecode(r.body) as Map<String, dynamic>;
      active = body['active'] as String;
      backends = (body['backends'] as List)
          .map((b) => BackendEntry.fromJson(b as Map<String, dynamic>))
          .toList();
      notifyListeners();
    } catch (e) {
      error = e.toString();
      notifyListeners();
    }
  }

  Future<void> refreshVllmStatus() async {
    try {
      final r = await _http.get(Uri.parse('$apiBaseUrl/api/backends/vllm/status'));
      if (r.statusCode != 200) return;
      vllmStatus = VLLMStatus.fromJson(jsonDecode(r.body) as Map<String, dynamic>);
      notifyListeners();
    } catch (e) {
      error = e.toString();
      notifyListeners();
    }
  }

  /// Start polling `/api/backends/vllm/status` every 2 seconds.
  /// Call from VllmSetupScreen.initState, stop in dispose.
  void startPolling() {
    stopPolling();
    refreshVllmStatus();
    _pollTimer = Timer.periodic(const Duration(seconds: 2), (_) => refreshVllmStatus());
  }

  void stopPolling() {
    _pollTimer?.cancel();
    _pollTimer = null;
  }

  @override
  void dispose() {
    stopPolling();
    _http.close();
    super.dispose();
  }

  Future<void> setActive(String backend) async {
    final r = await _http.post(
      Uri.parse('$apiBaseUrl/api/backends/active'),
      headers: {'content-type': 'application/json'},
      body: jsonEncode({'backend': backend}),
    );
    if (r.statusCode == 200) {
      active = backend;
      notifyListeners();
    } else {
      throw Exception('Backend switch failed: ${r.statusCode}');
    }
  }
}
```

- [ ] **Step 4: Run test**

```bash
cd flutter_app && flutter test test/providers/llm_backend_provider_test.dart
```
Expected: `All tests passed`

- [ ] **Step 5: Commit**

```bash
cd flutter_app && flutter analyze lib/providers/llm_backend_provider.dart test/providers/llm_backend_provider_test.dart
cd ..
git add flutter_app/lib/providers/llm_backend_provider.dart flutter_app/test/providers/llm_backend_provider_test.dart
git commit -m "feat(flutter): LlmBackendProvider with 2s polling + VLLMStatus model"
```

---

## Task 26: Flutter — `LlmBackendsScreen` (List View)

**Files:**
- Create: `flutter_app/lib/screens/llm_backends_screen.dart`
- Create: `flutter_app/test/widgets/llm_backends_screen_test.dart`

- [ ] **Step 1: Write the failing test**

```dart
// flutter_app/test/widgets/llm_backends_screen_test.dart
import 'package:cognithor_ui/providers/llm_backend_provider.dart';
import 'package:cognithor_ui/screens/llm_backends_screen.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:provider/provider.dart';

LlmBackendProvider _mkProvider(List<BackendEntry> backends, String active) {
  final p = LlmBackendProvider(apiBaseUrl: 'http://test');
  p.backends = backends;
  p.active = active;
  return p;
}

void main() {
  testWidgets('renders all backends with status dots', (tester) async {
    final provider = _mkProvider([
      BackendEntry(name: 'ollama', enabled: true, status: 'ready'),
      BackendEntry(name: 'vllm', enabled: false, status: 'disabled'),
    ], 'ollama');

    await tester.pumpWidget(MaterialApp(
      home: ChangeNotifierProvider<LlmBackendProvider>.value(
        value: provider,
        child: const LlmBackendsScreen(),
      ),
    ));

    expect(find.text('Ollama'), findsOneWidget);
    expect(find.text('vLLM'), findsOneWidget);
  });

  testWidgets('active backend has a visual marker', (tester) async {
    final provider = _mkProvider([
      BackendEntry(name: 'ollama', enabled: true, status: 'ready'),
      BackendEntry(name: 'vllm', enabled: true, status: 'ready'),
    ], 'vllm');

    await tester.pumpWidget(MaterialApp(
      home: ChangeNotifierProvider<LlmBackendProvider>.value(
        value: provider,
        child: const LlmBackendsScreen(),
      ),
    ));

    // "Active" label near the vllm row
    expect(find.byKey(const ValueKey('backend-vllm-active')), findsOneWidget);
  });
}
```

- [ ] **Step 2: Run test**

```bash
cd flutter_app && flutter test test/widgets/llm_backends_screen_test.dart
```
Expected: fails — screen doesn't exist.

- [ ] **Step 3: Create the screen**

```dart
// flutter_app/lib/screens/llm_backends_screen.dart
import 'package:cognithor_ui/providers/llm_backend_provider.dart';
import 'package:cognithor_ui/screens/vllm_setup_screen.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

class LlmBackendsScreen extends StatefulWidget {
  const LlmBackendsScreen({super.key});

  @override
  State<LlmBackendsScreen> createState() => _LlmBackendsScreenState();
}

class _LlmBackendsScreenState extends State<LlmBackendsScreen> {
  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      context.read<LlmBackendProvider>().refreshList();
    });
  }

  @override
  Widget build(BuildContext context) {
    final p = context.watch<LlmBackendProvider>();
    return Scaffold(
      appBar: AppBar(title: const Text('LLM Backends')),
      body: ListView.builder(
        itemCount: p.backends.length,
        itemBuilder: (ctx, i) {
          final b = p.backends[i];
          final isActive = p.active == b.name;
          return ListTile(
            leading: Icon(
              b.status == 'ready' ? Icons.circle : Icons.circle_outlined,
              color: b.status == 'ready' ? Colors.green : Colors.grey,
              size: 14,
            ),
            title: Text(_displayName(b.name)),
            subtitle: Text(_statusLine(b)),
            trailing: isActive
                ? Container(
                    key: ValueKey('backend-${b.name}-active'),
                    padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                    decoration: BoxDecoration(
                      color: Theme.of(ctx).colorScheme.primary.withValues(alpha: 0.2),
                      borderRadius: BorderRadius.circular(4),
                    ),
                    child: const Text('Active', style: TextStyle(fontSize: 11)),
                  )
                : const Icon(Icons.chevron_right),
            onTap: () {
              if (b.name == 'vllm') {
                Navigator.of(ctx).push(MaterialPageRoute(
                  builder: (_) => const VllmSetupScreen(),
                ));
              }
            },
          );
        },
      ),
    );
  }

  static String _displayName(String name) {
    switch (name) {
      case 'ollama':
        return 'Ollama';
      case 'vllm':
        return 'vLLM';
      case 'openai':
        return 'OpenAI';
      case 'anthropic':
        return 'Anthropic';
      default:
        return name;
    }
  }

  static String _statusLine(BackendEntry b) {
    if (!b.enabled) return 'Disabled';
    return b.status;
  }
}
```

Note: this file imports `vllm_setup_screen.dart` which is created in the next task. To avoid a broken tree, use a placeholder:

```dart
// flutter_app/lib/screens/vllm_setup_screen.dart (temporary stub)
import 'package:flutter/material.dart';

class VllmSetupScreen extends StatelessWidget {
  const VllmSetupScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return const Scaffold(body: Center(child: Text('vLLM setup — stub')));
  }
}
```

This stub satisfies the import; it gets fully implemented in Task 27.

- [ ] **Step 4: Run test**

```bash
cd flutter_app && flutter test test/widgets/llm_backends_screen_test.dart
```
Expected: passed

- [ ] **Step 5: Commit**

```bash
cd flutter_app && flutter analyze lib/screens/llm_backends_screen.dart lib/screens/vllm_setup_screen.dart test/widgets/llm_backends_screen_test.dart
cd ..
git add flutter_app/lib/screens/llm_backends_screen.dart flutter_app/lib/screens/vllm_setup_screen.dart flutter_app/test/widgets/llm_backends_screen_test.dart
git commit -m "feat(flutter): LlmBackendsScreen list view + VllmSetupScreen stub"
```

---

## Task 27: Flutter — `VllmSetupScreen` (Status Cards)

**Files:**
- Modify: `flutter_app/lib/screens/vllm_setup_screen.dart` (replace stub)
- Create: `flutter_app/test/widgets/vllm_setup_screen_test.dart`

- [ ] **Step 1: Write the failing test**

```dart
// flutter_app/test/widgets/vllm_setup_screen_test.dart
import 'package:cognithor_ui/providers/llm_backend_provider.dart';
import 'package:cognithor_ui/screens/vllm_setup_screen.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:provider/provider.dart';

LlmBackendProvider _mkProvider(VLLMStatus? s) {
  final p = LlmBackendProvider(apiBaseUrl: 'http://test');
  p.vllmStatus = s;
  return p;
}

void main() {
  testWidgets('all four status cards are rendered', (tester) async {
    final provider = _mkProvider(VLLMStatus(
      hardwareOk: false,
      hardwareInfo: null,
      dockerOk: false,
      imagePulled: false,
      containerRunning: false,
      currentModel: null,
      lastError: null,
    ));

    await tester.pumpWidget(MaterialApp(
      home: ChangeNotifierProvider<LlmBackendProvider>.value(
        value: provider,
        child: const VllmSetupScreen(),
      ),
    ));

    expect(find.byKey(const ValueKey('card-hardware')), findsOneWidget);
    expect(find.byKey(const ValueKey('card-docker')), findsOneWidget);
    expect(find.byKey(const ValueKey('card-image')), findsOneWidget);
    expect(find.byKey(const ValueKey('card-model')), findsOneWidget);
  });

  testWidgets('hardware card shows GPU name when detected', (tester) async {
    final provider = _mkProvider(VLLMStatus(
      hardwareOk: true,
      hardwareInfo: HardwareInfo(gpuName: 'RTX 5090', vramGb: 32, computeCapability: '12.0'),
      dockerOk: true,
      imagePulled: false,
      containerRunning: false,
      currentModel: null,
      lastError: null,
    ));

    await tester.pumpWidget(MaterialApp(
      home: ChangeNotifierProvider<LlmBackendProvider>.value(
        value: provider,
        child: const VllmSetupScreen(),
      ),
    ));

    expect(find.textContaining('RTX 5090'), findsOneWidget);
    expect(find.textContaining('32 GB'), findsOneWidget);
  });

  testWidgets('image card shows pull button when pending', (tester) async {
    final provider = _mkProvider(VLLMStatus(
      hardwareOk: true,
      hardwareInfo: HardwareInfo(gpuName: 'RTX 5090', vramGb: 32, computeCapability: '12.0'),
      dockerOk: true,
      imagePulled: false,
      containerRunning: false,
      currentModel: null,
      lastError: null,
    ));

    await tester.pumpWidget(MaterialApp(
      home: ChangeNotifierProvider<LlmBackendProvider>.value(
        value: provider,
        child: const VllmSetupScreen(),
      ),
    ));

    expect(find.text('Pull image'), findsOneWidget);
  });
}
```

- [ ] **Step 2: Run test**

```bash
cd flutter_app && flutter test test/widgets/vllm_setup_screen_test.dart
```
Expected: fails (stub doesn't have keys).

- [ ] **Step 3: Implement VllmSetupScreen**

Replace the content of `flutter_app/lib/screens/vllm_setup_screen.dart`:

```dart
import 'package:cognithor_ui/providers/llm_backend_provider.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

class VllmSetupScreen extends StatefulWidget {
  const VllmSetupScreen({super.key});

  @override
  State<VllmSetupScreen> createState() => _VllmSetupScreenState();
}

class _VllmSetupScreenState extends State<VllmSetupScreen> {
  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      context.read<LlmBackendProvider>().startPolling();
    });
  }

  @override
  void dispose() {
    context.read<LlmBackendProvider>().stopPolling();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final p = context.watch<LlmBackendProvider>();
    final s = p.vllmStatus;

    return Scaffold(
      appBar: AppBar(title: const Text('Configure vLLM')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          _HardwareCard(status: s),
          const SizedBox(height: 12),
          _DockerCard(status: s),
          const SizedBox(height: 12),
          _ImageCard(status: s),
          const SizedBox(height: 12),
          _ModelCard(status: s),
        ],
      ),
    );
  }
}

enum _CardState { ok, todo, pending, error }

Color _colorFor(_CardState st, BuildContext ctx) {
  switch (st) {
    case _CardState.ok:
      return Colors.green;
    case _CardState.todo:
      return Colors.orange;
    case _CardState.pending:
      return Theme.of(ctx).colorScheme.outline;
    case _CardState.error:
      return Colors.red;
  }
}

IconData _iconFor(_CardState st) {
  switch (st) {
    case _CardState.ok:
      return Icons.check_circle;
    case _CardState.todo:
      return Icons.radio_button_unchecked;
    case _CardState.pending:
      return Icons.more_horiz;
    case _CardState.error:
      return Icons.error;
  }
}

class _StatusCard extends StatelessWidget {
  final String title;
  final String subtitle;
  final _CardState state;
  final Widget? action;
  final ValueKey cardKey;

  const _StatusCard({
    required this.title,
    required this.subtitle,
    required this.state,
    required this.cardKey,
    this.action,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      key: cardKey,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(8),
        side: BorderSide(color: _colorFor(state, context), width: 1),
      ),
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Row(
          children: [
            Icon(_iconFor(state), color: _colorFor(state, context)),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(title, style: const TextStyle(fontWeight: FontWeight.w600)),
                  const SizedBox(height: 2),
                  Text(subtitle, style: const TextStyle(fontSize: 12, color: Colors.grey)),
                ],
              ),
            ),
            if (action != null) action!,
          ],
        ),
      ),
    );
  }
}

class _HardwareCard extends StatelessWidget {
  final VLLMStatus? status;
  const _HardwareCard({required this.status});

  @override
  Widget build(BuildContext context) {
    final hw = status?.hardwareInfo;
    final ok = status?.hardwareOk ?? false;
    return _StatusCard(
      cardKey: const ValueKey('card-hardware'),
      title: 'NVIDIA GPU',
      subtitle: hw == null
          ? 'Not detected'
          : '${hw.gpuName}, ${hw.vramGb} GB, SM ${hw.computeCapability}',
      state: ok ? _CardState.ok : _CardState.error,
    );
  }
}

class _DockerCard extends StatelessWidget {
  final VLLMStatus? status;
  const _DockerCard({required this.status});

  @override
  Widget build(BuildContext context) {
    final ok = status?.dockerOk ?? false;
    return _StatusCard(
      cardKey: const ValueKey('card-docker'),
      title: 'Docker Desktop',
      subtitle: ok ? 'Running' : 'Not running — start Docker Desktop',
      state: ok ? _CardState.ok : _CardState.todo,
    );
  }
}

class _ImageCard extends StatelessWidget {
  final VLLMStatus? status;
  const _ImageCard({required this.status});

  @override
  Widget build(BuildContext context) {
    final pulled = status?.imagePulled ?? false;
    return _StatusCard(
      cardKey: const ValueKey('card-image'),
      title: 'vLLM Docker image',
      subtitle: pulled
          ? 'vllm/vllm-openai:v0.19.1 ready'
          : 'Pull required (~10 GB, one-time)',
      state: pulled ? _CardState.ok : _CardState.todo,
      action: pulled
          ? null
          : FilledButton.tonal(
              onPressed: () {},  // wired up in Task 28 (SSE progress)
              child: const Text('Pull image'),
            ),
    );
  }
}

class _ModelCard extends StatelessWidget {
  final VLLMStatus? status;
  const _ModelCard({required this.status});

  @override
  Widget build(BuildContext context) {
    final running = status?.containerRunning ?? false;
    final model = status?.currentModel;
    return _StatusCard(
      cardKey: const ValueKey('card-model'),
      title: 'Model',
      subtitle: running
          ? 'Running: ${model ?? 'unknown'}'
          : 'Pick a model to start',
      state: running
          ? _CardState.ok
          : (status?.imagePulled ?? false)
              ? _CardState.todo
              : _CardState.pending,
    );
  }
}
```

- [ ] **Step 4: Run test**

```bash
cd flutter_app && flutter test test/widgets/vllm_setup_screen_test.dart
```
Expected: passed

- [ ] **Step 5: Commit**

```bash
cd flutter_app && flutter analyze lib/screens/vllm_setup_screen.dart test/widgets/vllm_setup_screen_test.dart
cd ..
git add flutter_app/lib/screens/vllm_setup_screen.dart flutter_app/test/widgets/vllm_setup_screen_test.dart
git commit -m "feat(flutter): VllmSetupScreen with 4 status cards"
```

---

## Task 28: Flutter — SSE Pull-Image Progress + Model Dropdown with Recommendation

**Files:**
- Modify: `flutter_app/lib/providers/llm_backend_provider.dart` (add `pullImage` stream)
- Modify: `flutter_app/lib/screens/vllm_setup_screen.dart` (wire Pull-button + model picker)

- [ ] **Step 1: Write the failing test**

Append to `flutter_app/test/providers/llm_backend_provider_test.dart`:

```dart
test('pullImage yields progress events from SSE stream', () async {
  // Use a stub http client that returns an SSE response
  final provider = LlmBackendProvider(apiBaseUrl: 'http://test');
  // This test verifies the public API surface only — actual SSE integration
  // is verified manually and by integration tests on the Python side.
  expect(provider.pullImage, isA<Function>());
});
```

The real SSE integration test lives on the Python side (Task 23). Flutter side only verifies the wiring.

- [ ] **Step 2: Run test**

```bash
cd flutter_app && flutter test test/providers/llm_backend_provider_test.dart
```
Expected: fails — `pullImage` method missing.

- [ ] **Step 3: Add SSE streaming method to provider**

Append to `LlmBackendProvider`:

```dart
/// Kick off docker-pull and yield progress events as parsed maps.
/// Events: {"status":"Downloading","progressDetail":{"current":N,"total":M},"id":"layer..."}
Stream<Map<String, dynamic>> pullImage() async* {
  final uri = Uri.parse('$apiBaseUrl/api/backends/vllm/pull-image');
  final request = http.Request('POST', uri);
  final streamed = await _http.send(request);
  if (streamed.statusCode != 200) {
    throw Exception('Pull failed: HTTP ${streamed.statusCode}');
  }
  String buffer = '';
  await for (final chunk in streamed.stream.transform(utf8.decoder)) {
    buffer += chunk;
    while (true) {
      final sep = buffer.indexOf('\n\n');
      if (sep == -1) break;
      final block = buffer.substring(0, sep);
      buffer = buffer.substring(sep + 2);
      for (final line in block.split('\n')) {
        if (line.startsWith('data:')) {
          final payload = line.substring(5).trim();
          if (payload.isEmpty) continue;
          try {
            yield jsonDecode(payload) as Map<String, dynamic>;
          } catch (_) {
            // Ignore malformed events
          }
        }
      }
    }
  }
  // Refresh full status after pull completes
  await refreshVllmStatus();
}

/// POST /api/backends/vllm/start — start a container for the given model.
Future<void> startContainer(String model) async {
  final r = await _http.post(
    Uri.parse('$apiBaseUrl/api/backends/vllm/start'),
    headers: {'content-type': 'application/json'},
    body: jsonEncode({'model': model}),
  );
  if (r.statusCode != 200) {
    final body = jsonDecode(r.body);
    throw Exception(body['detail']?['message'] ?? 'Start failed');
  }
  await refreshVllmStatus();
}
```

Wire the Pull-button in `VllmSetupScreen._ImageCard` — replace the `onPressed: () {}` with a handler that renders a `LinearProgressIndicator` while the SSE stream runs:

```dart
class _ImageCard extends StatefulWidget {
  final VLLMStatus? status;
  const _ImageCard({required this.status});

  @override
  State<_ImageCard> createState() => _ImageCardState();
}

class _ImageCardState extends State<_ImageCard> {
  double? _progress;
  String? _layer;
  bool _pulling = false;

  Future<void> _pull() async {
    setState(() {
      _pulling = true;
      _progress = null;
      _layer = null;
    });
    try {
      final p = context.read<LlmBackendProvider>();
      await for (final ev in p.pullImage()) {
        final detail = ev['progressDetail'] as Map<String, dynamic>?;
        final current = detail?['current'] as int?;
        final total = detail?['total'] as int?;
        if (current != null && total != null && total > 0) {
          setState(() {
            _progress = current / total;
            _layer = ev['id'] as String?;
          });
        }
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context)
            .showSnackBar(SnackBar(content: Text('Pull failed: $e')));
      }
    } finally {
      if (mounted) setState(() => _pulling = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final pulled = widget.status?.imagePulled ?? false;
    return _StatusCard(
      cardKey: const ValueKey('card-image'),
      title: 'vLLM Docker image',
      subtitle: pulled
          ? 'vllm/vllm-openai:v0.19.1 ready'
          : _pulling
              ? 'Downloading${_layer != null ? " layer ${_layer!.substring(0, 12)}..." : "…"}'
              : 'Pull required (~10 GB, one-time)',
      state: pulled
          ? _CardState.ok
          : _pulling
              ? _CardState.pending
              : _CardState.todo,
      action: pulled
          ? null
          : _pulling
              ? SizedBox(
                  width: 100,
                  child: LinearProgressIndicator(value: _progress),
                )
              : FilledButton.tonal(
                  onPressed: _pull,
                  child: const Text('Pull image'),
                ),
    );
  }
}
```

Model dropdown wiring for `_ModelCard`: fetch recommended model + full filter_registry via a new endpoint OR inline; simplest v1 — hardcode the display of the dropdown that POSTs to `/start`:

Extend the `_ModelCard` similarly to show a dropdown once the image is pulled (but no hardware-filtering on the client; the Python side already filters via `filter_registry()` — Task 29 adds an endpoint for the full list).

- [ ] **Step 4: Run test**

```bash
cd flutter_app && flutter test
```
Expected: all widget and provider tests pass.

- [ ] **Step 5: Commit**

```bash
cd flutter_app && flutter analyze lib/providers/llm_backend_provider.dart lib/screens/vllm_setup_screen.dart
cd ..
git add flutter_app/lib/providers/llm_backend_provider.dart flutter_app/lib/screens/vllm_setup_screen.dart flutter_app/test/providers/llm_backend_provider_test.dart
git commit -m "feat(flutter): SSE pull-image progress + start-container wiring"
```

---

## Task 29: Flutter — Model Dropdown with Recommendation Badge

**Files:**
- Modify: `src/cognithor/channels/api.py` — add `/api/backends/vllm/available-models` endpoint
- Modify: `flutter_app/lib/providers/llm_backend_provider.dart` — fetch available-models
- Modify: `flutter_app/lib/screens/vllm_setup_screen.dart` — render dropdown with recommendation

- [ ] **Step 1: Write the failing test (Python side)**

```python
class TestAvailableModels:
    def test_returns_filtered_registry_with_recommendation_flag(
        self, client_with_vllm_enabled
    ):
        client, _ = client_with_vllm_enabled
        from cognithor.core.vllm_orchestrator import HardwareInfo, ModelEntry

        with patch(
            "cognithor.core.vllm_orchestrator.VLLMOrchestrator.check_hardware",
            return_value=HardwareInfo("RTX 5090", 32, (12, 0)),
        ):
            r = client.get("/api/backends/vllm/available-models")
        assert r.status_code == 200
        data = r.json()
        # Shape: {"recommended_id": "...", "models": [{...entry..., "fits": bool}]}
        assert "recommended_id" in data
        assert "models" in data
        assert len(data["models"]) >= 1
        for m in data["models"]:
            assert "id" in m
            assert "fits" in m
```

- [ ] **Step 2: Run test**

```bash
python -m pytest tests/test_channels/test_api_backends.py::TestAvailableModels -v
```
Expected: fails.

- [ ] **Step 3: Implement endpoint (Python side)**

```python
@backends_router.get("/vllm/available-models")
async def vllm_available_models(request):
    config = request.app.state.config
    orch = _get_orchestrator(config)

    # Load registry
    import json as _json
    from pathlib import Path
    from cognithor.core.vllm_orchestrator import ModelEntry

    registry_path = (
        Path(__file__).resolve().parents[1]
        / "cli" / "model_registry.json"
    )
    registry_data = _json.loads(registry_path.read_text(encoding="utf-8"))
    entries = [
        ModelEntry.from_dict(m)
        for m in registry_data["providers"]["vllm"]["models"]
    ]

    # Detect hardware (cache on orchestrator state if already detected)
    hw = orch.state.hardware_info
    if hw is None:
        try:
            hw = orch.check_hardware()
        except Exception:
            hw = None

    recommended_id: str | None = None
    if hw is not None:
        best = orch.recommend_model(hw, entries)
        recommended_id = best.id if best else None

    # Build response
    fits_ids: set[str] = set()
    if hw is not None:
        fits_ids = {m.id for m in orch.filter_registry(hw, entries)}

    return {
        "recommended_id": recommended_id,
        "models": [
            {
                "id": e.id,
                "display_name": e.display_name,
                "quantization": e.quantization,
                "vram_gb_min": e.vram_gb_min,
                "min_compute_capability": e.min_compute_capability,
                "priority": e.priority,
                "tested": e.tested,
                "notes": e.notes,
                "fits": e.id in fits_ids,
            }
            for e in entries
        ],
    }
```

- [ ] **Step 4: Run Python test**

```bash
python -m pytest tests/test_channels/test_api_backends.py::TestAvailableModels -v
```
Expected: passed

- [ ] **Step 5: Flutter-side model dropdown**

Add to `LlmBackendProvider`:

```dart
List<Map<String, dynamic>> availableModels = [];
String? recommendedModelId;

Future<void> fetchAvailableModels() async {
  final r = await _http.get(Uri.parse('$apiBaseUrl/api/backends/vllm/available-models'));
  if (r.statusCode != 200) return;
  final body = jsonDecode(r.body) as Map<String, dynamic>;
  recommendedModelId = body['recommended_id'] as String?;
  availableModels = (body['models'] as List).cast<Map<String, dynamic>>();
  notifyListeners();
}
```

Update `_ModelCard` to show a dropdown when the image is pulled:

```dart
class _ModelCard extends StatefulWidget {
  final VLLMStatus? status;
  const _ModelCard({required this.status});
  @override
  State<_ModelCard> createState() => _ModelCardState();
}

class _ModelCardState extends State<_ModelCard> {
  String? _selected;
  bool _starting = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      context.read<LlmBackendProvider>().fetchAvailableModels();
    });
  }

  @override
  Widget build(BuildContext context) {
    final p = context.watch<LlmBackendProvider>();
    final running = widget.status?.containerRunning ?? false;
    final pulled = widget.status?.imagePulled ?? false;

    if (running) {
      return _StatusCard(
        cardKey: const ValueKey('card-model'),
        title: 'Model',
        subtitle: 'Running: ${widget.status?.currentModel ?? "unknown"}',
        state: _CardState.ok,
      );
    }

    if (!pulled) {
      return const _StatusCard(
        cardKey: ValueKey('card-model'),
        title: 'Model',
        subtitle: 'Available after image pull',
        state: _CardState.pending,
      );
    }

    final models = p.availableModels;
    final recommendedId = p.recommendedModelId;
    _selected ??= recommendedId ?? (models.isNotEmpty ? models[0]['id'] as String : null);

    return Card(
      key: const ValueKey('card-model'),
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text('Model', style: TextStyle(fontWeight: FontWeight.w600)),
            const SizedBox(height: 8),
            DropdownButton<String>(
              value: _selected,
              isExpanded: true,
              onChanged: (v) => setState(() => _selected = v),
              items: [
                for (final m in models)
                  DropdownMenuItem(
                    value: m['id'] as String,
                    enabled: m['fits'] as bool,
                    child: Row(
                      children: [
                        if (m['id'] == recommendedId)
                          const Padding(
                            padding: EdgeInsets.only(right: 6),
                            child: Icon(Icons.star, size: 14, color: Colors.amber),
                          ),
                        Expanded(child: Text(m['display_name'] as String)),
                        Text(
                          '${m['vram_gb_min']} GB',
                          style: TextStyle(
                            fontSize: 11,
                            color: (m['fits'] as bool) ? Colors.grey : Colors.red,
                          ),
                        ),
                      ],
                    ),
                  ),
              ],
            ),
            const SizedBox(height: 8),
            FilledButton(
              onPressed: _starting || _selected == null
                  ? null
                  : () async {
                      setState(() => _starting = true);
                      try {
                        await p.startContainer(_selected!);
                      } catch (e) {
                        if (mounted) {
                          ScaffoldMessenger.of(context).showSnackBar(
                            SnackBar(content: Text('Start failed: $e')),
                          );
                        }
                      } finally {
                        if (mounted) setState(() => _starting = false);
                      }
                    },
              child: Text(_starting ? 'Starting…' : 'Start vLLM'),
            ),
          ],
        ),
      ),
    );
  }
}
```

- [ ] **Step 6: Run full Flutter test suite**

```bash
cd flutter_app && flutter test && flutter analyze
```
Expected: all pass

- [ ] **Step 7: Commit**

```bash
python -m ruff check src/cognithor/channels/api.py tests/test_channels/test_api_backends.py && python -m ruff format --check src/cognithor/channels/api.py tests/test_channels/test_api_backends.py
git add src/cognithor/channels/api.py tests/test_channels/test_api_backends.py flutter_app/lib/providers/llm_backend_provider.dart flutter_app/lib/screens/vllm_setup_screen.dart
git commit -m "feat(vllm): GPU-aware model dropdown with recommendation badge + start button"
```

---

## Task 30: Gateway — Shutdown Hook + `reuse_existing()` on Startup

**Files:**
- Modify: `src/cognithor/gateway/gateway.py` (add vLLM lifecycle hooks)
- Create: `tests/test_gateway/test_gateway_vllm_lifecycle.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gateway/test_gateway_vllm_lifecycle.py
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from cognithor.config import CognithorConfig, VLLMConfig


class TestGatewayVLLMLifecycle:
    def test_shutdown_stops_container_when_toggle_on(self):
        cfg = CognithorConfig(vllm=VLLMConfig(enabled=True, auto_stop_on_close=True))
        from cognithor.gateway.gateway import Gateway
        gw = Gateway.__new__(Gateway)
        gw._config = cfg
        gw._vllm_orchestrator = MagicMock()
        gw.on_shutdown_vllm()
        gw._vllm_orchestrator.stop_container.assert_called_once()

    def test_shutdown_leaves_container_running_when_toggle_off(self):
        cfg = CognithorConfig(vllm=VLLMConfig(enabled=True, auto_stop_on_close=False))
        from cognithor.gateway.gateway import Gateway
        gw = Gateway.__new__(Gateway)
        gw._config = cfg
        gw._vllm_orchestrator = MagicMock()
        gw.on_shutdown_vllm()
        gw._vllm_orchestrator.stop_container.assert_not_called()

    def test_startup_picks_up_existing_container(self):
        cfg = CognithorConfig(vllm=VLLMConfig(enabled=True))
        from cognithor.gateway.gateway import Gateway
        from cognithor.core.vllm_orchestrator import ContainerInfo
        gw = Gateway.__new__(Gateway)
        gw._config = cfg
        gw._vllm_orchestrator = MagicMock()
        gw._vllm_orchestrator.reuse_existing.return_value = ContainerInfo(
            container_id="abc", port=8000, model="Qwen/Qwen2.5-VL-7B-Instruct"
        )
        result = gw.on_startup_vllm()
        assert result is not None
        assert result.model == "Qwen/Qwen2.5-VL-7B-Instruct"
```

- [ ] **Step 2: Run test**

```bash
python -m pytest tests/test_gateway/test_gateway_vllm_lifecycle.py -v
```
Expected: fails — `on_shutdown_vllm` / `on_startup_vllm` missing.

- [ ] **Step 3: Add lifecycle hooks to Gateway**

In `src/cognithor/gateway/gateway.py`, add two methods to the `Gateway` class:

```python
def on_startup_vllm(self):
    """Called during 6-phase init. If a cognithor-managed vLLM container
    is already running, reuse it. Otherwise do nothing — user starts via UI."""
    if not self._config.vllm.enabled:
        return None
    try:
        return self._vllm_orchestrator.reuse_existing()
    except Exception as exc:
        log.warning("vllm_reuse_existing_failed", error=str(exc))
        return None


def on_shutdown_vllm(self) -> None:
    """Called on Gateway.shutdown(). Stops the container only if the
    user has opted in via `auto_stop_on_close`."""
    if not self._config.vllm.enabled:
        return
    if self._config.vllm.auto_stop_on_close:
        try:
            self._vllm_orchestrator.stop_container()
        except Exception as exc:
            log.warning("vllm_shutdown_failed", error=str(exc))
```

Wire the hooks into the Gateway init (call `on_startup_vllm()` during the 6-phase init) and shutdown (`shutdown()` calls `on_shutdown_vllm()`). Also lazy-instantiate `self._vllm_orchestrator` in Gateway init when `config.vllm.enabled` is True:

```python
# In Gateway.__init__ or _init_llm phase
if self._config.vllm.enabled:
    from cognithor.core.vllm_orchestrator import VLLMOrchestrator
    self._vllm_orchestrator = VLLMOrchestrator(
        docker_image=self._config.vllm.docker_image,
        port=self._config.vllm.port,
        hf_token=self._config.huggingface_api_key,
    )
else:
    self._vllm_orchestrator = None
```

- [ ] **Step 4: Run test**

```bash
python -m pytest tests/test_gateway/test_gateway_vllm_lifecycle.py -v
```
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
python -m ruff check src/cognithor/gateway/gateway.py tests/test_gateway/test_gateway_vllm_lifecycle.py && python -m ruff format --check src/cognithor/gateway/gateway.py tests/test_gateway/test_gateway_vllm_lifecycle.py
git add src/cognithor/gateway/gateway.py tests/test_gateway/test_gateway_vllm_lifecycle.py
git commit -m "feat(gateway): vLLM shutdown hook + reuse_existing on startup"
```

---

## Task 31: Cross-Repo Registry-Sync Guard

**Files:**
- Create: `tests/test_vllm_registry_sync.py`

- [ ] **Step 1: Write the test**

```python
# tests/test_vllm_registry_sync.py
"""Guard test: the vLLM model registry entries must stay self-consistent
and load cleanly into ModelEntry. Prevents silent drift when someone
edits model_registry.json but forgets a field.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from cognithor.core.vllm_orchestrator import ModelEntry

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY = REPO_ROOT / "src" / "cognithor" / "cli" / "model_registry.json"


class TestVLLMRegistryLoads:
    def test_every_entry_parses_into_ModelEntry(self):
        data = json.loads(REGISTRY.read_text(encoding="utf-8"))
        for m in data["providers"]["vllm"]["models"]:
            # Must not raise
            ModelEntry.from_dict(m)

    def test_compute_capability_tuples_are_valid(self):
        data = json.loads(REGISTRY.read_text(encoding="utf-8"))
        for m in data["providers"]["vllm"]["models"]:
            entry = ModelEntry.from_dict(m)
            cc = entry.min_cc_tuple
            assert 7 <= cc[0] <= 20
            assert 0 <= cc[1] <= 9

    def test_min_vllm_version_is_valid(self):
        data = json.loads(REGISTRY.read_text(encoding="utf-8"))
        for m in data["providers"]["vllm"]["models"]:
            v = m["min_vllm_version"]
            # "pending" or semver-ish
            assert v == "pending" or re.match(r"^\d+\.\d+(\.\d+)?$", v)

    def test_priority_is_enum(self):
        data = json.loads(REGISTRY.read_text(encoding="utf-8"))
        for m in data["providers"]["vllm"]["models"]:
            assert m["priority"] in ("premium", "standard", "fallback")

    def test_capability_is_enum(self):
        data = json.loads(REGISTRY.read_text(encoding="utf-8"))
        for m in data["providers"]["vllm"]["models"]:
            assert m["capability"] in ("vision", "text")
```

- [ ] **Step 2: Run test**

```bash
python -m pytest tests/test_vllm_registry_sync.py -v
```
Expected: `5 passed` (assuming Tasks 4 & 5 done)

- [ ] **Step 3: Commit**

```bash
python -m ruff check tests/test_vllm_registry_sync.py && python -m ruff format --check tests/test_vllm_registry_sync.py
git add tests/test_vllm_registry_sync.py
git commit -m "test(vllm): registry-sync guard cross-validates every entry with ModelEntry"
```

---

## Task 32: User-Facing Guide `docs/vllm-user-guide.md`

**Files:**
- Create: `docs/vllm-user-guide.md`

- [ ] **Step 1: Write the guide**

Create `docs/vllm-user-guide.md` with this content:

```markdown
# Enabling the vLLM Backend

Cognithor ships with Ollama as the default local LLM backend — it "just works"
on any Windows/macOS/Linux machine without further setup. vLLM is an **opt-in**
alternative for users with an NVIDIA GPU who want faster inference, native FP4
support (Blackwell / RTX 50xx), or access to models not in the Ollama library.

## Prerequisites

You need:

1. **NVIDIA GPU** with at least 16 GB VRAM. For the best experience:
   - **RTX 5090 (32 GB)** — unlocks NVFP4 quantization, the fastest option
   - **RTX 4090 (24 GB)** — runs FP8 quantization
   - **RTX 4080 / 3090 / 4070 Ti Super (16 GB)** — runs AWQ-INT4 quantization
2. **NVIDIA driver** installed (any modern version from the last 2 years works)
3. **Docker Desktop** installed and running. Download from
   [docker.com](https://www.docker.com/products/docker-desktop). Cognithor will
   not install it for you — the installer needs admin rights and a reboot,
   which Cognithor does not handle.

## Enabling vLLM

1. Start Cognithor normally.
2. Open **Settings → LLM Backends**.
3. Tap **vLLM**. You will see four status cards:
   - **NVIDIA GPU** — detected automatically
   - **Docker Desktop** — detected automatically
   - **vLLM Docker image** — needs a one-time ~10 GB pull
   - **Model** — which model to load
4. If any card is red, fix the underlying issue first (install the missing
   driver, start Docker Desktop, etc.).
5. Tap **Pull image** on Card 3. Progress streams live.
6. Pick a model from Card 4. The star badge (⭐) marks the recommendation for
   your GPU. Models that don't fit your VRAM or require a newer GPU
   architecture are disabled with a tooltip explaining why.
7. Tap **Start vLLM**. First start takes 30–300 seconds depending on model
   size (weights download from HuggingFace).
8. Back in the list view, tap your vLLM row and select **Make active** to
   switch all future chat turns through vLLM.

## Switching Back to Ollama

Settings → LLM Backends → Ollama → **Make active**. The switch is live — no
restart required. vLLM keeps running in the background unless you enable
"Stop vLLM on app close" in settings.

## Troubleshooting

**"Version Mismatch" overlay on launch**: the installer bundled a stale
Flutter build. Install the newer Cognithor release.

**vLLM status card stays red with "No GPU detected"**: run `nvidia-smi` in a
terminal to confirm your driver works. On WSL2 you need the NVIDIA WSL driver
bundle from nvidia.com/drivers — not just the standard Windows driver.

**Docker card stays red**: open Docker Desktop, wait for the whale icon to
stop pulsing (that's the "ready" state).

**Pull fails mid-download**: partial layers are cached. Retry the pull —
Docker will resume from where it stopped.

**Container starts but /health never answers**: the model is probably still
loading. Qwen3.6-27B at FP8 takes ~60 s on an RTX 5090, up to 5 minutes on
slower cards. The setup page shows container logs below the status cards.

**Banner "vLLM offline — fallback to Ollama active"** appears mid-chat: vLLM
has crashed or become unresponsive for 3 consecutive requests. Text chats
transparently route through Ollama; image requests will error out until vLLM
recovers. Check `docker logs <container-id>` for the cause.

**I have a Qwen3.6 model selected but it fails to start**: vLLM stable
(v0.19.1) does not yet support the Qwen3.6 architecture. Workaround: set
`config.vllm.docker_image` to `vllm/vllm-openai:nightly` and restart. Cognithor
will adopt the new image on the next container start.

## Advanced Configuration

`~/.cognithor/config.yaml` section `vllm`:

| Field | Default | Purpose |
|-------|---------|---------|
| `enabled` | `false` | Master on/off |
| `model` | `""` (auto) | HF repo id. Empty → orchestrator picks best per GPU |
| `docker_image` | `vllm/vllm-openai:v0.19.1` | Override to bleed-edge |
| `port` | `8000` | Host port (falls back 8001..8009 if busy) |
| `auto_stop_on_close` | `false` | Stop container when Cognithor quits |
| `skip_hardware_check` | `false` | Override for unusual setups |
| `request_timeout_seconds` | `60` | Per-request timeout |

HF token for gated models: set `huggingface_api_key` at the top level of
`config.yaml` (or via the OS keyring) — Cognithor passes it to the container
automatically as `HF_TOKEN`.
```

- [ ] **Step 2: Commit**

```bash
git add docs/vllm-user-guide.md
git commit -m "docs(vllm): user-facing enable + troubleshooting guide"
```

---

## Task 33: Manual Smoke-Test Recipe `docs/vllm-manual-test.md`

**Files:**
- Create: `docs/vllm-manual-test.md`

- [ ] **Step 1: Write the recipe**

Create `docs/vllm-manual-test.md`:

```markdown
# vLLM Backend — Manual Smoke Test

The Python + Flutter unit tests and the fake-server integration test cover
everything except the real-hardware path. Before cutting a release that
touches the vLLM backend, run this recipe once on a dev machine with a real
NVIDIA GPU and Docker Desktop.

**Time:** ~30 minutes end-to-end.

## Test Matrix

Pick the row matching your dev hardware and run those steps:

| GPU | VRAM | Expected recommended model | Test sections |
|-----|------|----------------------------|---------------|
| RTX 5090 | 32 GB | `mmangkad/Qwen3.6-27B-NVFP4` | All |
| RTX 4090 | 24 GB | `cyankiwi/Qwen3.6-27B-AWQ-INT4` or fallback | 1–5 |
| RTX 4080 / 4070 Ti Super | 16 GB | `Qwen/Qwen2.5-VL-7B-Instruct` (tested fallback) | 1–4 |

## 1. Fresh install

- Uninstall any previous Cognithor.
- Wipe `~/.cognithor/config.yaml` (or rename it as backup).
- Run the new installer.
- Launch Cognithor. Expect: Flutter UI reaches the main screen. No version-
  mismatch overlay, no config crash.

## 2. Hardware + Docker detection

- Settings → LLM Backends → tap **vLLM**.
- Expect: Cards 1 and 2 turn green within 2 seconds (the polling interval).
  Card 1 shows the correct GPU name, VRAM, and compute capability string.
- If Docker Desktop is not running, start it and wait — Card 2 turns green
  when ready.

## 3. Image pull

- Tap **Pull image**.
- Expect: progress bar advances steadily. Total download ~10 GB.
- Expect: after completion, Card 3 turns green; Card 4 enables.

## 4. Model picker + start (quick path — Qwen2.5-VL-7B)

- Card 4: expect the model dropdown to populate. Qwen2.5-VL-7B is always
  marked "tested" and should show the star badge for any non-Blackwell GPU.
- Select it. Tap **Start vLLM**.
- Expect: within 120 seconds, Card 4 turns green with "Running: Qwen/
  Qwen2.5-VL-7B-Instruct".
- Back in the list view, tap vLLM, tap **Make active**.

## 5. End-to-end chat with vision

- Open the chat screen.
- Attach an image (any PNG) via the paperclip button.
- Ask: "What do you see in this image?"
- Expect: answer comes back within 5–10 seconds, describing the image.
- Close the chat screen and reopen — state persists.

## 6. Fail-flow verification

- In a terminal: `docker stop $(docker ps -q --filter label=cognithor.managed=true)`
- In Cognithor chat, send a text-only message: "hello"
  - Expect: within 2–3 requests, the "⚠ vLLM offline — fallback to Ollama
    active" banner appears. Reply comes from Ollama.
- Send an image request.
  - Expect: red error bubble "vLLM offline — cannot process image".
- In terminal: `docker start <container-id>`
  - Expect: within ~60 seconds (the CircuitBreaker `recovery_timeout`), the
    next text request goes through vLLM again; banner dismisses.

## 7. Lifecycle toggles

- Settings → LLM Backends → vLLM → enable "Keep vLLM running after app close"
- Close Cognithor entirely.
- Verify: `docker ps | grep cognithor.managed` — container still running.
- Reopen Cognithor → status shows "Running" immediately (no restart).
- Disable the toggle. Close Cognithor.
- Verify: `docker ps | grep cognithor.managed` — no result (container was
  stopped on shutdown).

## 8. Blackwell-specific (RTX 5090 only)

- Edit `~/.cognithor/config.yaml` → set `vllm.docker_image: "vllm/vllm-openai:nightly"`.
- Back in the setup screen, pull the nightly image (replaces the old one).
- Pick `mmangkad/Qwen3.6-27B-NVFP4` (star-badged on Blackwell).
- Start. Expect: model loads in ~30–60 seconds.
- Chat with vision. Expect: tokens stream noticeably faster than FP8 on the
  same hardware (NVFP4 uses native tensor cores).

## Reporting

If any step fails, capture:
- Contents of `~/.cognithor/log/cognithor.log` (last 200 lines)
- Output of `docker logs <container-id>`
- Screenshot of the relevant Flutter screen

File bugs against the `vllm-backend` label on GitHub.
```

- [ ] **Step 2: Commit**

```bash
git add docs/vllm-manual-test.md
git commit -m "docs(vllm): manual smoke-test recipe for release-gate validation"
```

---

## Task 34: CHANGELOG + Final Integration Run

**Files:**
- Modify: `CHANGELOG.md` (add [Unreleased] entry)

- [ ] **Step 1: Add CHANGELOG entry**

Add to the top of `CHANGELOG.md` under `## [Unreleased]` (create section if missing):

```markdown
## [Unreleased]

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
```

- [ ] **Step 2: Run the full regression locally**

```bash
python -m pytest tests/ -x -q --ignore=tests/test_integration/test_live_ollama.py
cd flutter_app && flutter analyze && flutter test
cd ..
```
Expected: every test green, zero analyze warnings.

- [ ] **Step 3: Final ruff sweep across the vLLM touch surface**

```bash
python -m ruff check src/cognithor/core/vllm_orchestrator.py \
                    src/cognithor/core/vllm_backend.py \
                    src/cognithor/core/unified_llm.py \
                    src/cognithor/core/llm_backend.py \
                    src/cognithor/channels/api.py \
                    src/cognithor/config.py \
                    src/cognithor/gateway/gateway.py \
                    tests/
python -m ruff format --check src/cognithor/ tests/
```
Expected: `All checks passed!` + `X files already formatted`.

- [ ] **Step 4: Commit**

```bash
git add CHANGELOG.md
git commit -m "chore(changelog): document vLLM opt-in backend feature"
```

- [ ] **Step 5: Open PR**

```bash
git push -u origin feat/vllm-opt-in-backend
```
Then open a PR with the title `feat(vllm): opt-in LLM backend with Flutter-driven lifecycle` and the body pulling from `docs/vllm-user-guide.md` summary + reference to spec + plan.

After CI is green, run the **manual smoke test from Task 33** before merging.

---

## Self-Review Checklist

After completing all tasks:

- [ ] All unit tests pass (`pytest tests/test_core/test_vllm_* tests/test_channels/test_api_backends.py tests/test_gateway/test_gateway_vllm_lifecycle.py tests/test_vllm_registry_sync.py tests/test_integration/test_vllm_fake_server.py -v`)
- [ ] `flutter test` green across `llm_backend_provider_test.dart`, `llm_backends_screen_test.dart`, `vllm_setup_screen_test.dart`
- [ ] `ruff check src/ tests/` clean
- [ ] `ruff format --check src/ tests/` clean (remember PR #135 lesson)
- [ ] `flutter analyze` clean
- [ ] Manual smoke test from Task 33 completed on real NVIDIA hardware
- [ ] CHANGELOG entry written
- [ ] No backward-incompatible changes to existing backends (Ollama/OpenAI/Anthropic paths unchanged)
- [ ] `config.vllm.enabled: false` path — verify Cognithor behaves identically to before this feature
