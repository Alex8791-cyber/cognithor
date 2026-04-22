# tests/test_core/test_model_registry_vllm.py
from __future__ import annotations

import json
from pathlib import Path

REGISTRY_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "cognithor" / "cli" / "model_registry.json"
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
            "id",
            "display_name",
            "base_model",
            "quantization",
            "vram_gb_min",
            "min_compute_capability",
            "min_vllm_version",
            "capability",
            "priority",
            "tested",
            "notes",
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
