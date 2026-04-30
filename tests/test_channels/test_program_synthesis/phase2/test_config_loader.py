# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""YAML config-loader tests (Sprint-1 plan task 1)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from cognithor.channels.program_synthesis.integration.capability_tokens import (  # noqa: F401
    PSECapability as _PSECapability,
)
from cognithor.channels.program_synthesis.phase2 import (
    DEFAULT_HEURISTICS_PATH,
    ConfigLoadError,
    LoadedHeuristics,
    Phase2Config,
    load_heuristics,
)

# ---------------------------------------------------------------------------
# Default-path round-trip
# ---------------------------------------------------------------------------


class TestDefaultHeuristicsYaml:
    def test_default_path_resolves_to_repo_configs(self) -> None:
        # Plan anchors configs/synthesis/heuristics.yaml at the repo
        # root.
        assert DEFAULT_HEURISTICS_PATH.is_file(), (
            f"heuristics.yaml not at expected path: {DEFAULT_HEURISTICS_PATH}"
        )
        assert DEFAULT_HEURISTICS_PATH.parent.name == "synthesis", (
            f"expected configs/synthesis path, got {DEFAULT_HEURISTICS_PATH.parent}"
        )

    def test_load_returns_phase2_config_plus_raw(self) -> None:
        result = load_heuristics()
        assert isinstance(result, LoadedHeuristics)
        assert isinstance(result.phase2_config, Phase2Config)
        assert isinstance(result.raw, dict)

    def test_loaded_phase2_config_matches_spec_anchors(self) -> None:
        cfg = load_heuristics().phase2_config
        # F1 multipliers spec defaults.
        assert cfg.high_impact_multiplier == 3.0
        assert cfg.structural_abstraction_multiplier == 1.5
        assert cfg.regular_primitive_multiplier == 1.0
        # F2 zone thresholds spec defaults.
        assert cfg.repair_alpha_zone1_lower == 0.45
        assert cfg.repair_alpha_zone3_upper == 0.35
        assert cfg.refiner_hysteresis_window == 3
        # F3 reserves OFF by default.
        assert cfg.enable_argument_quality_factor is False
        assert cfg.enable_few_demos_dampening is False
        # α-bands.
        assert cfg.alpha_entropy_lower == 0.5
        assert cfg.alpha_entropy_upper == 0.85
        assert cfg.alpha_performance_lower == 0.5
        assert cfg.alpha_performance_upper == 1.0
        # Sample-size dampening.
        assert cfg.sample_size_dampening_n0 == 4

    def test_loaded_llm_config_uses_vllm_keys(self) -> None:
        # The repo config carries vLLM-style keys (HuggingFace name +
        # base URL); the loader prefers those over llama.cpp's
        # ``model_path``.
        cfg = load_heuristics().phase2_config
        assert cfg.llm_base_url == "http://localhost:8000/v1"
        assert cfg.llm_model_name == "Qwen/Qwen3.6-27B-Instruct"
        assert cfg.llm_fallback_model_name == "Qwen/Qwen3.6-27B-Instruct-AWQ"
        # llm.sampling.temperature_repair_cot / .temperature_repair_edit
        # populate the Phase2Config stage temperatures.
        assert cfg.llm_temperature_stage1 == 0.3
        assert cfg.llm_temperature_stage2 == 0.0

    def test_raw_dict_carries_full_yaml(self) -> None:
        # Sprint-2 modules read MCTS / CEGIS / score-weights from raw.
        raw = load_heuristics().raw
        assert "mcts" in raw
        assert raw["mcts"]["c_puct_default"] == 3.5
        assert "verifier" in raw
        assert raw["verifier"]["score_weights"]["demo_pass_rate"] == 0.55
        assert "budget" in raw
        partition = raw["budget"]["partition"]
        assert sum(partition.values()) == 1.0, f"budget partition does not sum to 1.0: {partition}"


# ---------------------------------------------------------------------------
# Schema-validation errors
# ---------------------------------------------------------------------------


class TestLoadHeuristicsErrors:
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigLoadError, match="not found"):
            load_heuristics(tmp_path / "absent.yaml")

    def test_invalid_yaml_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("not: yaml: at: all: ::: {", encoding="utf-8")
        with pytest.raises(ConfigLoadError, match="parse error"):
            load_heuristics(bad)

    def test_root_must_be_mapping(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("- just\n- a list\n", encoding="utf-8")
        with pytest.raises(ConfigLoadError, match="root must be a mapping"):
            load_heuristics(bad)

    def test_missing_section_raises(self, tmp_path: Path) -> None:
        # Section like ``verifier`` missing: surfaces with the path.
        bad = tmp_path / "bad.yaml"
        bad.write_text("alpha:\n  entropy_floor: 0.5\n", encoding="utf-8")
        with pytest.raises(ConfigLoadError, match=r"verifier"):
            load_heuristics(bad)

    def test_phase2_config_invariants_propagate(self, tmp_path: Path) -> None:
        # Build a YAML with valid structure but invalid ordering
        # (zone3_upper > zone1_lower). The loader hands that to
        # Phase2Config.__post_init__ which rejects it; the loader
        # surfaces the violation as a ConfigLoadError.
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            _MINIMAL_YAML.replace("full_llm_min_alpha: 0.45", "full_llm_min_alpha: 0.30").replace(
                "hybrid_min_alpha: 0.35", "hybrid_min_alpha: 0.40"
            ),
            encoding="utf-8",
        )
        with pytest.raises(ConfigLoadError, match="Phase2Config invariants"):
            load_heuristics(bad)


# ---------------------------------------------------------------------------
# Minimal valid YAML — used for the negative tests above
# ---------------------------------------------------------------------------


_MINIMAL_YAML = """\
verifier:
  syntactic_complexity:
    high_impact_multiplier: 3.0
    structural_abstraction_multiplier: 1.5
    regular_multiplier: 1.0
  score_weights:
    demo_pass_rate: 0.55
    partial_pixel_match: 0.13
    invariants_satisfied: 0.08
    triviality_score: 0.12
    suspicion_score: 0.12

refiner:
  mode_thresholds:
    full_llm_min_alpha: 0.45
    hybrid_min_alpha: 0.35
  mode_hysteresis:
    repairs_to_hold: 3

alpha:
  entropy_floor: 0.5
  entropy_base: 0.85
  performance_floor: 0.5
  performance_base: 1.0

symbolic_prior:
  sample_size:
    saturation_n: 4

reserved_fixes:
  few_demos_dampening:
    enabled: false
  argument_quality_factor:
    enabled: false

llm_prior:
  base_url: "http://localhost:8000/v1"
  model_name: "Qwen/Qwen3.6-27B-Instruct"
  inference: {}
  sampling:
    temperature_repair_cot: 0.3
    temperature_repair_edit: 0.0
"""


class TestMinimalYamlRoundTrip:
    def test_minimal_yaml_loads(self, tmp_path: Path) -> None:
        p = tmp_path / "ok.yaml"
        p.write_text(_MINIMAL_YAML, encoding="utf-8")
        result = load_heuristics(p)
        assert result.phase2_config.high_impact_multiplier == 3.0
        assert result.phase2_config.repair_alpha_zone1_lower == 0.45
