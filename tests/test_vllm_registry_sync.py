"""Guard test: the vLLM model registry entries must stay self-consistent
and load cleanly into ModelEntry.
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
            ModelEntry.from_dict(m)  # must not raise

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
            assert v == "pending" or re.match(r"^\d+\.\d+(\.\d+)?$", v)

    def test_priority_is_enum(self):
        data = json.loads(REGISTRY.read_text(encoding="utf-8"))
        for m in data["providers"]["vllm"]["models"]:
            assert m["priority"] in ("premium", "standard", "fallback")

    def test_capability_is_enum(self):
        data = json.loads(REGISTRY.read_text(encoding="utf-8"))
        for m in data["providers"]["vllm"]["models"]:
            assert m["capability"] in ("vision", "text")
