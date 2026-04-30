# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""YAML-backed loader for :class:`Phase2Config` (Sprint-1 plan task 1).

The Phase-2 Sprint-1 plan mandates a single source of truth for every
heuristic constant: ``configs/synthesis/heuristics.yaml``. This module
parses that YAML file and projects the relevant subset onto the typed
:class:`Phase2Config` dataclass that the rest of the channel reads.

The YAML carries more than `Phase2Config` currently models — MCTS
constants, CEGIS budgets, score-weights, FAISS cache settings, etc.
Those are Sprint-2 territory; the loader accepts the keys without
choking, projects only what is wired today, and exposes the raw YAML
dict for the future Sprint-2 modules to consume.

Design rules:

* **No silent defaults.** A missing or malformed value the loader
  understands surfaces as :class:`ConfigLoadError` with the offending
  YAML path. Spec rule "no magic numbers in source" extends to "no
  silent fallbacks at the loader boundary".
* **Phase2Config invariants still enforced.** The loader builds a
  :class:`Phase2Config` and lets its ``__post_init__`` validate.
* **Bridging only.** The loader is read-only: it does NOT mutate
  :data:`DEFAULT_PHASE2_CONFIG`. Callers receive a fresh
  :class:`Phase2Config` they can pass into the channel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from cognithor.channels.program_synthesis.phase2.config import Phase2Config

DEFAULT_HEURISTICS_PATH = (
    Path(__file__).resolve().parents[5] / "configs" / "synthesis" / "heuristics.yaml"
)
"""Path the Sprint-1 plan anchors. Override with an explicit argument
in tests or alternate deployments."""


class ConfigLoadError(ValueError):
    """Raised when the YAML config cannot be projected onto Phase2Config."""


@dataclass(frozen=True)
class LoadedHeuristics:
    """The structured result of loading ``heuristics.yaml``.

    ``phase2_config`` is the typed view the channel reads. ``raw`` is
    the entire parsed YAML — Sprint-2 modules (MCTS controller, CEGIS,
    Verifier score-weights) read their bits from here until they get
    typed dataclass mirrors of their own.
    """

    phase2_config: Phase2Config
    raw: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def load_heuristics(
    path: Path | str | None = None,
) -> LoadedHeuristics:
    """Load + validate the heuristics YAML at *path*.

    Defaults to :data:`DEFAULT_HEURISTICS_PATH`. Raises
    :class:`ConfigLoadError` on any schema violation.
    """
    p = Path(path) if path is not None else DEFAULT_HEURISTICS_PATH
    if not p.is_file():
        raise ConfigLoadError(f"heuristics YAML not found: {p}")
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigLoadError(f"YAML parse error in {p}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigLoadError(f"heuristics YAML root must be a mapping; got {type(raw).__name__}")
    config = _project_to_phase2_config(raw)
    return LoadedHeuristics(phase2_config=config, raw=raw)


# ---------------------------------------------------------------------------
# Projection: YAML → Phase2Config
# ---------------------------------------------------------------------------


def _project_to_phase2_config(raw: dict[str, Any]) -> Phase2Config:
    """Build a Phase2Config from the YAML structure.

    Reads only the keys Phase2Config currently models. Other YAML
    sections (mcts, refiner full body, verifier, budget, calibration,
    reserved_fixes) are ignored here — they're consumed by Sprint-2
    modules through the ``raw`` dict on :class:`LoadedHeuristics`.
    """
    verifier = _expect_section(raw, "verifier")
    sc = _expect_section(verifier, "syntactic_complexity", parent="verifier")
    refiner = _expect_section(raw, "refiner")
    thresholds = _expect_section(refiner, "mode_thresholds", parent="refiner")
    hysteresis = _expect_section(refiner, "mode_hysteresis", parent="refiner")
    alpha = _expect_section(raw, "alpha")
    symbolic = _expect_section(raw, "symbolic_prior")
    sample = _expect_section(symbolic, "sample_size", parent="symbolic_prior")
    reserved = _expect_section(raw, "reserved_fixes")
    few_demos = _expect_section(reserved, "few_demos_dampening", parent="reserved_fixes")
    arg_qual = _expect_section(reserved, "argument_quality_factor", parent="reserved_fixes")
    llm = _expect_section(raw, "llm_prior")
    llm_inference = _expect_section(llm, "inference", parent="llm_prior")
    llm_sampling = _expect_section(llm, "sampling", parent="llm_prior")

    try:
        return Phase2Config(
            high_impact_multiplier=_expect_float(
                sc, "high_impact_multiplier", parent="verifier.syntactic_complexity"
            ),
            structural_abstraction_multiplier=_expect_float(
                sc,
                "structural_abstraction_multiplier",
                parent="verifier.syntactic_complexity",
            ),
            regular_primitive_multiplier=_expect_float(
                sc, "regular_multiplier", parent="verifier.syntactic_complexity"
            ),
            repair_alpha_zone1_lower=_expect_float(
                thresholds, "full_llm_min_alpha", parent="refiner.mode_thresholds"
            ),
            repair_alpha_zone3_upper=_expect_float(
                thresholds, "hybrid_min_alpha", parent="refiner.mode_thresholds"
            ),
            refiner_hysteresis_window=_expect_int(
                hysteresis, "repairs_to_hold", parent="refiner.mode_hysteresis"
            ),
            enable_argument_quality_factor=_expect_bool(
                arg_qual,
                "enabled",
                parent="reserved_fixes.argument_quality_factor",
            ),
            enable_few_demos_dampening=_expect_bool(
                few_demos, "enabled", parent="reserved_fixes.few_demos_dampening"
            ),
            alpha_entropy_lower=_expect_float(alpha, "entropy_floor", parent="alpha"),
            alpha_entropy_upper=_expect_float(alpha, "entropy_base", parent="alpha"),
            alpha_performance_lower=_expect_float(alpha, "performance_floor", parent="alpha"),
            alpha_performance_upper=_expect_float(alpha, "performance_base", parent="alpha"),
            sample_size_dampening_n0=_expect_int(
                sample, "saturation_n", parent="symbolic_prior.sample_size"
            ),
            llm_base_url=_optional_str(llm, "base_url", default="http://localhost:8000/v1"),
            # Prefer the vLLM-style ``model_name`` (HuggingFace id)
            # over the llama.cpp-style ``model_path`` (file path).
            # Plan ships both side-by-side so a backend swap doesn't
            # require a YAML rewrite.
            llm_model_name=_first_str(
                llm,
                ("model_name", "model_path"),
                default="Qwen/Qwen3.6-27B-Instruct",
            ),
            llm_fallback_model_name=_first_str(
                llm,
                ("fallback_model_name", "fallback_model_path"),
                default="Qwen/Qwen3.6-27B-Instruct-AWQ",
            ),
            llm_temperature_stage1=_expect_float(
                llm_sampling, "temperature_repair_cot", parent="llm_prior.sampling"
            ),
            llm_temperature_stage2=_expect_float(
                llm_sampling, "temperature_repair_edit", parent="llm_prior.sampling"
            ),
            llm_json_max_retries=_optional_int(
                refiner.get("llm_repair", {}), "max_retries", default=1
            ),
            llm_top_k_default=_optional_int(llm_sampling, "top_k", default=5),
            llm_call_timeout_seconds=_optional_float(
                llm,
                "call_timeout_seconds",
                default=_optional_float(llm_inference, "call_timeout_seconds", default=8.0),
            ),
        )
    except ValueError as exc:
        # Phase2Config.__post_init__ raises ValueError; surface as a
        # config-load error so the caller sees one consistent
        # exception class.
        raise ConfigLoadError(f"Phase2Config invariants violated: {exc}") from exc


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------


def _expect_section(
    container: dict[str, Any], name: str, *, parent: str = "<root>"
) -> dict[str, Any]:
    value = container.get(name)
    if not isinstance(value, dict):
        raise ConfigLoadError(
            f"heuristics YAML: missing or malformed section "
            f"{parent}.{name} (expected mapping, got {type(value).__name__})"
        )
    return value


def _expect_float(d: dict[str, Any], key: str, *, parent: str) -> float:
    value = d.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ConfigLoadError(
            f"heuristics YAML: {parent}.{key} must be a number; "
            f"got {type(value).__name__} {value!r}"
        )
    return float(value)


def _expect_int(d: dict[str, Any], key: str, *, parent: str) -> int:
    value = d.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigLoadError(
            f"heuristics YAML: {parent}.{key} must be an integer; "
            f"got {type(value).__name__} {value!r}"
        )
    return value


def _expect_bool(d: dict[str, Any], key: str, *, parent: str) -> bool:
    value = d.get(key)
    if not isinstance(value, bool):
        raise ConfigLoadError(
            f"heuristics YAML: {parent}.{key} must be a bool; got {type(value).__name__} {value!r}"
        )
    return value


def _optional_str(d: dict[str, Any], key: str, *, default: str) -> str:
    value = d.get(key, default)
    if not isinstance(value, str):
        return default
    return value


def _first_str(d: dict[str, Any], keys: tuple[str, ...], *, default: str) -> str:
    """Return the first ``keys`` entry that's a non-empty string."""
    for k in keys:
        value = d.get(k)
        if isinstance(value, str) and value:
            return value
    return default


def _optional_int(d: dict[str, Any], key: str, *, default: int) -> int:
    value = d.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool):
        return default
    return value


def _optional_float(d: dict[str, Any], key: str, *, default: float) -> float:
    value = d.get(key, default)
    if not isinstance(value, int | float) or isinstance(value, bool):
        return default
    return float(value)


__all__ = [
    "DEFAULT_HEURISTICS_PATH",
    "ConfigLoadError",
    "LoadedHeuristics",
    "load_heuristics",
]
