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
    Path(__file__).resolve().parents[2] / "src" / "cognithor" / "cli" / "model_registry.json"
)


@pytest.fixture
def registry() -> list[ModelEntry]:
    data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    return [ModelEntry.from_dict(m) for m in data["providers"]["vllm"]["models"]]


@pytest.fixture
def orch() -> VLLMOrchestrator:
    return VLLMOrchestrator()


class TestRecommendModel:
    def test_blackwell_32gb_picks_nvfp4_when_tested(self, orch):
        # Use synthetic registry where NVFP4 is marked tested=True (simulates future
        # vLLM Qwen3.6 support landing). Controller must pick it over fallback.
        entries = [
            ModelEntry.from_dict(
                {
                    "id": "mmangkad/Qwen3.6-27B-NVFP4",
                    "display_name": "NVFP4",
                    "base_model": "Qwen/Qwen3.6-27B",
                    "quantization": "NVFP4",
                    "vram_gb_min": 14,
                    "min_compute_capability": "12.0",
                    "min_vllm_version": "0.20.0",
                    "capability": "vision",
                    "priority": "premium",
                    "tested": True,
                    "notes": "",
                }
            ),
            ModelEntry.from_dict(
                {
                    "id": "Qwen/Qwen2.5-VL-7B-Instruct",
                    "display_name": "fallback",
                    "base_model": "Qwen/Qwen2.5-VL-7B-Instruct",
                    "quantization": "bf16",
                    "vram_gb_min": 16,
                    "min_compute_capability": "7.5",
                    "min_vllm_version": "0.7.0",
                    "capability": "vision",
                    "priority": "fallback",
                    "tested": True,
                    "notes": "",
                }
            ),
        ]
        hw = HardwareInfo(gpu_name="RTX 5090", vram_gb=32, compute_capability=(12, 0))
        best = orch.recommend_model(hw, entries, prefer="vision")
        assert best.id == "mmangkad/Qwen3.6-27B-NVFP4"

    def test_current_registry_ada_24gb_falls_back_to_qwen25(self, orch, registry):
        # All current Qwen3.6 entries have tested=False. Only Qwen2.5-VL-7B is tested.
        hw = HardwareInfo(gpu_name="RTX 4090", vram_gb=24, compute_capability=(8, 9))
        best = orch.recommend_model(hw, registry, prefer="vision")
        assert best.tested is True
        assert "Qwen2.5-VL" in best.id

    def test_current_registry_ampere_24gb_falls_back_to_qwen25(self, orch, registry):
        hw = HardwareInfo(gpu_name="RTX 3090", vram_gb=24, compute_capability=(8, 0))
        best = orch.recommend_model(hw, registry, prefer="vision")
        assert best.tested is True

    def test_low_vram_returns_none(self, orch, registry):
        hw = HardwareInfo(gpu_name="RTX 2080 Ti", vram_gb=11, compute_capability=(7, 5))
        best = orch.recommend_model(hw, registry, prefer="vision")
        assert best is None

    def test_text_preference_returns_none_when_none_curated(self, orch, registry):
        # Registry has no text-capability entries in v1
        hw = HardwareInfo(gpu_name="RTX 5090", vram_gb=32, compute_capability=(12, 0))
        best = orch.recommend_model(hw, registry, prefer="text")
        assert best is None


class TestFilterRegistry:
    def test_ignores_entries_exceeding_vram(self, orch, registry):
        hw = HardwareInfo(gpu_name="RTX 4080", vram_gb=16, compute_capability=(8, 9))
        results = orch.filter_registry(hw, registry)
        for m in results:
            assert m.vram_gb_min <= 16

    def test_ignores_entries_requiring_newer_compute_capability(self, orch, registry):
        hw = HardwareInfo(gpu_name="RTX 4090", vram_gb=24, compute_capability=(8, 9))
        results = orch.filter_registry(hw, registry)
        for m in results:
            assert m.min_cc_tuple <= (8, 9)
