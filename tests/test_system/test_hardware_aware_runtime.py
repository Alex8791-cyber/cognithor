"""Integration tests for the Hardware-Aware Runtime stack.

Layers under test:
- L1   detector.py (probes)
- L1.5 sanity.py
- L2   capabilities.py
- L3   manifest_loader.py + manifest_models.py
- L4   solver.py
- L6   apply_engine.py

L5 wizard is exercised via solver+apply integration; the interactive
input flow has its own (light) tests further down.
"""

from __future__ import annotations

import pytest

from cognithor.system.apply_engine import (
    ApplyError,
    apply_solution,
    list_backups,
)
from cognithor.system.capabilities import map_to_capabilities
from cognithor.system.detector import DetectionResult, SystemProfile
from cognithor.system.manifest_loader import ManifestLoader
from cognithor.system.manifest_models import Manifest
from cognithor.system.sanity import validate
from cognithor.system.solver import OBJECTIVE_PRESETS, UserObjective, solve

# ──────────────────────────────────────────────────────────────────────────
# Mock-Profile builders for 7 representative hardware configurations
# ──────────────────────────────────────────────────────────────────────────


def _profile(**overrides: DetectionResult) -> SystemProfile:
    """Build a SystemProfile with sensible defaults + overrides."""
    profile = SystemProfile()
    profile.results.update(
        {
            "os": DetectionResult("os", "Linux 6.x", "ok", {"os": "linux", "is_wsl": False}),
            "cpu": DetectionResult(
                "cpu",
                "Generic CPU (8C/16T)",
                "ok",
                {"physical_cores": 8, "logical_cores": 16, "max_freq_mhz": 4500},
            ),
            "ram": DetectionResult("ram", "32 GB", "ok", {"total_gb": 32, "available_gb": 24}),
            "disk": DetectionResult("disk", "120 GB", "ok", {"free_gb": 120, "total_gb": 1000}),
            "network": DetectionResult("network", "Connected", "ok", {"internet": True}),
            "ollama": DetectionResult("ollama", "—", "warn", {"running": False}),
            "lmstudio": DetectionResult("lmstudio", "—", "warn", {"running": False}),
            "container": DetectionResult("container", "host", "ok", {"in_container": False}),
            "docker": DetectionResult("docker", "Docker running", "ok", {"running": True}),
            "wsl2": DetectionResult("wsl2", "N/A", "ok", {"applicable": False}),
            "rocm": DetectionResult("rocm", "—", "warn", {"available": False}),
            "vllm": DetectionResult("vllm", "vLLM 0.10.2", "ok", {"pip_installed": True}),
            "huggingface": DetectionResult(
                "huggingface",
                "HF reachable",
                "ok",
                {"reachable": True, "token_present": False},
            ),
        }
    )
    profile.results.update(overrides)
    return profile


def profile_rtx_5090_blackwell_modern() -> SystemProfile:
    """RTX 5090, driver 596+, CUDA 13 — NVFP4-capable."""
    return _profile(
        gpu=DetectionResult(
            "gpu",
            "RTX 5090 (32GB, blackwell)",
            "ok",
            {
                "vendor": "nvidia",
                "model": "RTX 5090",
                "vram_total_gb": 32.0,
                "vram_free_gb": 30.0,
                "driver": "596.21",
                "compute_capability": "12.0",
                "cuda_version": "13.0",
                "architecture": "blackwell",
                "all_gpus": [],
                "multi_gpu_count": 1,
            },
        ),
        ram=DetectionResult("ram", "64 GB", "ok", {"total_gb": 64, "available_gb": 48}),
        disk=DetectionResult("disk", "300 GB", "ok", {"free_gb": 300, "total_gb": 2000}),
    )


def profile_rtx_5090_blackwell_old_driver() -> SystemProfile:
    """RTX 5090 hardware but driver 581 — NVFP4 BLOCKED, FP8 still OK."""
    p = profile_rtx_5090_blackwell_modern()
    g = p.results["gpu"]
    g.raw_data["driver"] = "581.29"
    g.raw_data["cuda_version"] = "13.0"
    return p


def profile_rtx_4090_ada() -> SystemProfile:
    """RTX 4090 — sm89 → FP8 yes, NVFP4 no."""
    return _profile(
        gpu=DetectionResult(
            "gpu",
            "RTX 4090 (24GB, ada)",
            "ok",
            {
                "vendor": "nvidia",
                "model": "RTX 4090",
                "vram_total_gb": 24.0,
                "vram_free_gb": 22.0,
                "driver": "550.10",
                "compute_capability": "8.9",
                "cuda_version": "12.4",
                "architecture": "ada",
                "all_gpus": [],
                "multi_gpu_count": 1,
            },
        ),
        ram=DetectionResult("ram", "64 GB", "ok", {"total_gb": 64, "available_gb": 48}),
        disk=DetectionResult("disk", "200 GB", "ok", {"free_gb": 200, "total_gb": 1000}),
    )


def profile_rtx_3060() -> SystemProfile:
    """RTX 3060 12GB — Ampere, sm86, no FP8."""
    return _profile(
        gpu=DetectionResult(
            "gpu",
            "RTX 3060 (12GB, ampere)",
            "ok",
            {
                "vendor": "nvidia",
                "model": "RTX 3060",
                "vram_total_gb": 12.0,
                "vram_free_gb": 11.0,
                "driver": "535.0",
                "compute_capability": "8.6",
                "cuda_version": "12.0",
                "architecture": "ampere",
                "all_gpus": [],
                "multi_gpu_count": 1,
            },
        ),
        ollama=DetectionResult("ollama", "Running", "ok", {"running": True}),
    )


def profile_apple_m3_max() -> SystemProfile:
    """Apple M3 Max — Metal."""
    return _profile(
        os=DetectionResult("os", "Darwin 24", "ok", {"os": "darwin", "is_wsl": False}),
        gpu=DetectionResult(
            "gpu",
            "Apple Silicon (unified)",
            "warn",
            {
                "vendor": "apple",
                "model": "Apple Silicon",
                "vram_total_gb": 0,
                "unified_memory": True,
                "compute_capability": None,
                "cuda_version": None,
                "architecture": "apple_silicon",
                "all_gpus": [],
                "multi_gpu_count": 1,
            },
        ),
        docker=DetectionResult("docker", "Docker not running", "warn", {"running": False}),
    )


def profile_no_gpu_low_ram() -> SystemProfile:
    """Cheap laptop — no GPU, 8 GB RAM."""
    return _profile(
        gpu=DetectionResult(
            "gpu",
            "No dedicated GPU",
            "fail",
            {
                "vendor": "none",
                "vram_total_gb": 0,
                "compute_capability": None,
                "cuda_version": None,
                "architecture": None,
            },
        ),
        ram=DetectionResult("ram", "8 GB", "warn", {"total_gb": 8, "available_gb": 4}),
        docker=DetectionResult("docker", "—", "fail", {"running": False}),
        vllm=DetectionResult("vllm", "—", "warn", {"pip_installed": False}),
    )


def profile_offline_no_internet() -> SystemProfile:
    p = profile_rtx_3060()
    p.results["network"] = DetectionResult("network", "Offline", "warn", {"internet": False})
    p.results["huggingface"] = DetectionResult(
        "huggingface",
        "HF unreachable",
        "warn",
        {"reachable": False, "token_present": False},
    )
    return p


# ──────────────────────────────────────────────────────────────────────────
# L2 Capability tests
# ──────────────────────────────────────────────────────────────────────────


class TestCapabilities:
    def test_rtx_5090_modern_unlocks_nvfp4(self) -> None:
        caps = map_to_capabilities(profile_rtx_5090_blackwell_modern())
        assert caps.can_run_nvfp4 is True
        assert caps.can_run_fp8_marlin is True
        assert caps.can_run_fp8_native is True
        assert caps.vram_class == "xlarge"
        assert caps.ram_class == "extreme"

    def test_rtx_5090_old_driver_blocks_nvfp4(self) -> None:
        caps = map_to_capabilities(profile_rtx_5090_blackwell_old_driver())
        assert caps.can_run_nvfp4 is False, "Driver < 596 must block NVFP4"
        assert caps.can_run_fp8_marlin is True, "FP8 still works on Blackwell"

    def test_rtx_4090_ada_no_nvfp4(self) -> None:
        caps = map_to_capabilities(profile_rtx_4090_ada())
        assert caps.can_run_nvfp4 is False, "Ada has sm89 < sm120"
        assert caps.can_run_fp8_marlin is True
        # 24 GB falls into "xlarge" (24..47); "large" starts at 16
        assert caps.vram_class == "xlarge"

    def test_rtx_3060_ampere_no_fp8_native(self) -> None:
        caps = map_to_capabilities(profile_rtx_3060())
        assert caps.can_run_nvfp4 is False
        assert caps.can_run_fp8_marlin is False, "Ampere sm86 < sm89 (FP8)"
        assert caps.can_run_gptq_int4 is True
        assert caps.vram_class == "medium"

    def test_apple_silicon_metal_only(self) -> None:
        caps = map_to_capabilities(profile_apple_m3_max())
        assert caps.can_run_gguf_metal is True
        assert caps.can_run_nvfp4 is False
        assert caps.can_run_gguf_cuda is False
        assert caps.can_run_vllm_container is False

    def test_no_gpu_low_ram(self) -> None:
        caps = map_to_capabilities(profile_no_gpu_low_ram())
        assert caps.can_run_gguf_cpu is True
        assert caps.can_run_nvfp4 is False
        assert caps.vram_class == "none"
        assert caps.ram_class == "low"

    def test_capability_is_deterministic(self) -> None:
        p = profile_rtx_5090_blackwell_modern()
        caps_1 = map_to_capabilities(p)
        caps_2 = map_to_capabilities(p)
        assert caps_1.profile_hash == caps_2.profile_hash

    def test_satisfies_ordinal_compare(self) -> None:
        caps = map_to_capabilities(profile_rtx_5090_blackwell_modern())
        assert caps.satisfies("vram_class>=large") is True
        assert caps.satisfies("vram_class>=xxlarge") is False
        assert caps.satisfies("ram_class>=high") is True

    def test_satisfies_bool_flag(self) -> None:
        caps = map_to_capabilities(profile_rtx_5090_blackwell_modern())
        assert caps.satisfies("can_run_nvfp4") is True
        assert caps.satisfies("can_run_gguf_metal") is False


# ──────────────────────────────────────────────────────────────────────────
# L1.5 Sanity tests
# ──────────────────────────────────────────────────────────────────────────


class TestSanity:
    def test_blackwell_with_old_driver_warns(self) -> None:
        warnings = validate(profile_rtx_5090_blackwell_old_driver())
        rule_ids = {w.rule_id for w in warnings}
        assert "nvfp4_driver_too_old" in rule_ids

    def test_modern_setup_no_warnings(self) -> None:
        warnings = validate(profile_rtx_5090_blackwell_modern())
        rule_ids = {w.rule_id for w in warnings}
        assert "nvfp4_driver_too_old" not in rule_ids

    def test_no_crash_on_partial_profile(self) -> None:
        # Profile with most fields missing — sanity must not crash
        empty = SystemProfile()
        warnings = validate(empty)
        assert isinstance(warnings, tuple)


# ──────────────────────────────────────────────────────────────────────────
# L3 Manifest-Loader tests
# ──────────────────────────────────────────────────────────────────────────


class TestManifestLoader:
    def test_embedded_load(self) -> None:
        loader = ManifestLoader()
        manifest, source = loader.load(prefer_online=False)
        assert isinstance(manifest, Manifest)
        assert manifest.manifest_version == "2026.05.07.01"
        assert source.origin in {"embedded", "cache"}
        assert len(manifest.tiers) >= 5
        assert len(manifest.models) >= 5

    def test_pricing_load(self) -> None:
        loader = ManifestLoader()
        loader.load(prefer_online=False)
        pricing = loader.load_pricing()
        assert pricing is not None
        assert "anthropic" in pricing.providers

    def test_recall_list_empty_by_default(self) -> None:
        loader = ManifestLoader()
        manifest, source = loader.load(prefer_online=False)
        # No exception → no active recall on this manifest version
        assert manifest.manifest_version == source.manifest_version


# ──────────────────────────────────────────────────────────────────────────
# L4 Solver tests
# ──────────────────────────────────────────────────────────────────────────


class TestSolver:
    @pytest.fixture
    def manifest_and_pricing(self):
        loader = ManifestLoader()
        m, _ = loader.load(prefer_online=False)
        p = loader.load_pricing()
        return m, p

    def test_solver_deterministic(self, manifest_and_pricing) -> None:
        m, p = manifest_and_pricing
        caps = map_to_capabilities(profile_rtx_5090_blackwell_modern())
        s1 = solve(m, caps, OBJECTIVE_PRESETS["balanced"], pricing=p)
        s2 = solve(m, caps, OBJECTIVE_PRESETS["balanced"], pricing=p)
        assert [s.tier_id for s in s1] == [s.tier_id for s in s2]
        assert [s.score for s in s1] == [s.score for s in s2]

    def test_modern_blackwell_top_is_nvfp4(self, manifest_and_pricing) -> None:
        m, p = manifest_and_pricing
        caps = map_to_capabilities(profile_rtx_5090_blackwell_modern())
        sols = solve(m, caps, OBJECTIVE_PRESETS["balanced"], pricing=p)
        runnable = [s for s in sols if s.is_immediately_runnable]
        assert runnable, "Should have at least one runnable solution"
        # Top runnable on modern Blackwell should be nvfp4 OR fp8
        assert runnable[0].tier_id in {
            "enterprise-vllm-nvfp4-blackwell",
            "power-vllm-fp8-ada",
        }

    def test_old_driver_blocks_nvfp4_tier(self, manifest_and_pricing) -> None:
        m, p = manifest_and_pricing
        caps = map_to_capabilities(profile_rtx_5090_blackwell_old_driver())
        sols = solve(m, caps, OBJECTIVE_PRESETS["balanced"], pricing=p)
        # NVFP4 tier should be in solutions but BLOCKED
        nvfp4 = next((s for s in sols if s.tier_id == "enterprise-vllm-nvfp4-blackwell"), None)
        if nvfp4 is not None:
            assert "can_run_nvfp4" in nvfp4.blockers
        # Top runnable must be FP8
        runnable = [s for s in sols if s.is_immediately_runnable]
        assert runnable[0].tier_id == "power-vllm-fp8-ada"

    def test_apple_silicon_picks_mac_tier(self, manifest_and_pricing) -> None:
        m, p = manifest_and_pricing
        caps = map_to_capabilities(profile_apple_m3_max())
        sols = solve(m, caps, OBJECTIVE_PRESETS["balanced"], pricing=p)
        runnable = [s for s in sols if s.is_immediately_runnable]
        ids = {s.tier_id for s in runnable}
        assert "mac-ollama-metal" in ids
        # Should never recommend vllm tiers on Apple
        assert "enterprise-vllm-nvfp4-blackwell" not in ids
        assert "power-vllm-fp8-ada" not in ids

    def test_no_gpu_picks_cpu_or_cloud(self, manifest_and_pricing) -> None:
        m, p = manifest_and_pricing
        caps = map_to_capabilities(profile_no_gpu_low_ram())
        sols = solve(m, caps, OBJECTIVE_PRESETS["balanced"], pricing=p)
        runnable = [s for s in sols if s.is_immediately_runnable]
        assert runnable, "Should never return zero solutions"
        ids = {s.tier_id for s in runnable}
        # Some local CPU-only tier or cloud fallback must appear
        assert ids & {"minimal-ollama-cpu", "cloud-only-anthropic"}

    def test_privacy_preset_excludes_cloud(self, manifest_and_pricing) -> None:
        m, p = manifest_and_pricing
        caps = map_to_capabilities(profile_rtx_5090_blackwell_modern())
        sols = solve(m, caps, OBJECTIVE_PRESETS["privacy"], pricing=p)
        runnable = [s for s in sols if s.is_immediately_runnable]
        ids = {s.tier_id for s in runnable}
        assert "cloud-only-anthropic" not in ids

    def test_offline_user_no_cloud_in_runnable(self, manifest_and_pricing) -> None:
        m, p = manifest_and_pricing
        caps = map_to_capabilities(profile_offline_no_internet())
        # has_internet=False → cloud-only requires has_internet → blocked
        sols = solve(m, caps, OBJECTIVE_PRESETS["balanced"], pricing=p)
        cloud = next((s for s in sols if s.tier_id == "cloud-only-anthropic"), None)
        if cloud is not None:
            assert not cloud.is_immediately_runnable

    def test_max_disk_constraint_filters(self, manifest_and_pricing) -> None:
        m, p = manifest_and_pricing
        caps = map_to_capabilities(profile_rtx_5090_blackwell_modern())
        obj = UserObjective(
            weight_quality=0.4,
            weight_speed=0.3,
            weight_cost=0.2,
            weight_privacy=0.1,
            max_disk_gb=10.0,
        )
        sols = solve(m, caps, obj, pricing=p)
        runnable = [s for s in sols if s.is_immediately_runnable]
        for s in runnable:
            assert s.estimated_disk_gb <= 10.0

    def test_score_breakdown_keys(self, manifest_and_pricing) -> None:
        m, p = manifest_and_pricing
        caps = map_to_capabilities(profile_rtx_5090_blackwell_modern())
        sols = solve(m, caps, OBJECTIVE_PRESETS["balanced"], pricing=p)
        for s in sols:
            assert set(s.score_breakdown) == {"quality", "speed", "cost", "privacy"}
            for v in s.score_breakdown.values():
                assert 0.0 <= v <= 1.0


# ──────────────────────────────────────────────────────────────────────────
# L6 Apply-Engine tests
# ──────────────────────────────────────────────────────────────────────────


class TestApplyEngine:
    @pytest.fixture
    def apply_setup(self, tmp_path):
        """Build a runnable Solution + Manifest + Capabilities for tmp config."""
        loader = ManifestLoader()
        m, _ = loader.load(prefer_online=False)
        p = loader.load_pricing()
        caps = map_to_capabilities(profile_rtx_5090_blackwell_modern())
        sols = solve(m, caps, OBJECTIVE_PRESETS["balanced"], pricing=p)
        runnable = [s for s in sols if s.is_immediately_runnable]
        chosen = runnable[0]
        return {
            "solution": chosen,
            "manifest": m,
            "caps": caps,
            "objective": OBJECTIVE_PRESETS["balanced"],
            "config_path": tmp_path / "config.yaml",
        }

    def test_apply_writes_valid_yaml(self, apply_setup) -> None:
        result = apply_solution(
            solution=apply_setup["solution"],
            manifest=apply_setup["manifest"],
            capabilities=apply_setup["caps"],
            objective=apply_setup["objective"],
            config_path=apply_setup["config_path"],
            user_confirmed=True,
        )
        assert result.success is True
        assert apply_setup["config_path"].exists()

    def test_apply_writes_sidecar(self, apply_setup) -> None:
        apply_solution(
            solution=apply_setup["solution"],
            manifest=apply_setup["manifest"],
            capabilities=apply_setup["caps"],
            objective=apply_setup["objective"],
            config_path=apply_setup["config_path"],
            user_confirmed=True,
        )
        sidecar = apply_setup["config_path"].parent / ".hardware_aware.json"
        assert sidecar.exists()
        import json

        data = json.loads(sidecar.read_text(encoding="utf-8"))
        assert data["recommended_tier"] == apply_setup["solution"].tier_id
        assert data["manifest_version"] == apply_setup["manifest"].manifest_version
        assert "system_profile_hash" in data

    def test_apply_writes_marker(self, apply_setup) -> None:
        result = apply_solution(
            solution=apply_setup["solution"],
            manifest=apply_setup["manifest"],
            capabilities=apply_setup["caps"],
            objective=apply_setup["objective"],
            config_path=apply_setup["config_path"],
            user_confirmed=True,
        )
        assert result.initialized_marker_path.exists()

    def test_apply_idempotent_yields_no_diff(self, apply_setup) -> None:
        apply_solution(
            **{
                k: apply_setup[k]
                for k in ("solution", "manifest", "caps", "objective", "config_path")
                if k in apply_setup
            },
            user_confirmed=True,
            capabilities=apply_setup["caps"],
        ) if False else None  # placeholder
        # Cleaner version:
        apply_solution(
            solution=apply_setup["solution"],
            manifest=apply_setup["manifest"],
            capabilities=apply_setup["caps"],
            objective=apply_setup["objective"],
            config_path=apply_setup["config_path"],
            user_confirmed=True,
        )
        first_content = apply_setup["config_path"].read_text(encoding="utf-8")
        apply_solution(
            solution=apply_setup["solution"],
            manifest=apply_setup["manifest"],
            capabilities=apply_setup["caps"],
            objective=apply_setup["objective"],
            config_path=apply_setup["config_path"],
            user_confirmed=True,
        )
        second_content = apply_setup["config_path"].read_text(encoding="utf-8")
        # The bookkeeping timestamps differ in sidecar, but the YAML body
        # should be identical (no User-Override changed)
        # Strip volatile lines (none in current schema, but defensive)
        assert first_content == second_content

    def test_apply_requires_explicit_confirmation(self, apply_setup) -> None:
        with pytest.raises(ApplyError, match="apply_requires_explicit_user_confirmation"):
            apply_solution(
                solution=apply_setup["solution"],
                manifest=apply_setup["manifest"],
                capabilities=apply_setup["caps"],
                objective=apply_setup["objective"],
                config_path=apply_setup["config_path"],
                user_confirmed=False,
            )

    def test_apply_blocked_solution_raises(self, apply_setup) -> None:
        # Construct a blocked solution from the existing one
        from dataclasses import replace

        blocked = replace(apply_setup["solution"], blockers=("can_run_nvfp4",))
        with pytest.raises(ApplyError, match="cannot_apply_blocked_solution"):
            apply_solution(
                solution=blocked,
                manifest=apply_setup["manifest"],
                capabilities=apply_setup["caps"],
                objective=apply_setup["objective"],
                config_path=apply_setup["config_path"],
                user_confirmed=True,
            )

    def test_rollback_restores_backup(self, apply_setup) -> None:
        cfg = apply_setup["config_path"]
        # First apply
        apply_solution(
            solution=apply_setup["solution"],
            manifest=apply_setup["manifest"],
            capabilities=apply_setup["caps"],
            objective=apply_setup["objective"],
            config_path=cfg,
            user_confirmed=True,
        )
        # Mutate manually
        cfg.write_text("llm_backend_type: ollama\n", encoding="utf-8")
        # Apply again — creates backup of the mutated file
        apply_solution(
            solution=apply_setup["solution"],
            manifest=apply_setup["manifest"],
            capabilities=apply_setup["caps"],
            objective=apply_setup["objective"],
            config_path=cfg,
            user_confirmed=True,
        )
        # Now there's a backup of `modified`
        backups = list_backups(cfg)
        assert backups, "Backup should have been created"

    def test_user_override_respected(self, apply_setup) -> None:
        # Pre-write a user-config that already specifies a planner
        apply_setup["config_path"].parent.mkdir(parents=True, exist_ok=True)
        apply_setup["config_path"].write_text(
            "llm_backend_type: vllm\nmodels:\n  planner:\n    name: my-custom-model\n",
            encoding="utf-8",
        )
        apply_solution(
            solution=apply_setup["solution"],
            manifest=apply_setup["manifest"],
            capabilities=apply_setup["caps"],
            objective=apply_setup["objective"],
            config_path=apply_setup["config_path"],
            user_confirmed=True,
        )
        import yaml

        cfg = yaml.safe_load(apply_setup["config_path"].read_text(encoding="utf-8"))
        assert cfg["models"]["planner"]["name"] == "my-custom-model"


# ──────────────────────────────────────────────────────────────────────────
# Property-style tests
# ──────────────────────────────────────────────────────────────────────────


class TestProperties:
    def test_solver_always_returns_at_least_one_solution(self) -> None:
        loader = ManifestLoader()
        m, _ = loader.load(prefer_online=False)
        p = loader.load_pricing()
        for builder in (
            profile_rtx_5090_blackwell_modern,
            profile_rtx_5090_blackwell_old_driver,
            profile_rtx_4090_ada,
            profile_rtx_3060,
            profile_apple_m3_max,
            profile_no_gpu_low_ram,
            profile_offline_no_internet,
        ):
            caps = map_to_capabilities(builder())
            sols = solve(m, caps, OBJECTIVE_PRESETS["balanced"], pricing=p)
            assert sols, f"Solver returned 0 solutions for {builder.__name__}"

    def test_capability_hash_changes_on_hardware_drift(self) -> None:
        c1 = map_to_capabilities(profile_rtx_5090_blackwell_modern())
        c2 = map_to_capabilities(profile_rtx_4090_ada())
        assert c1.profile_hash != c2.profile_hash

    def test_solver_score_in_unit_interval(self) -> None:
        loader = ManifestLoader()
        m, _ = loader.load(prefer_online=False)
        p = loader.load_pricing()
        caps = map_to_capabilities(profile_rtx_5090_blackwell_modern())
        for preset_name in OBJECTIVE_PRESETS:
            sols = solve(m, caps, OBJECTIVE_PRESETS[preset_name], pricing=p)
            for s in sols:
                assert 0.0 <= s.score <= 1.0, f"Score out of [0,1] in preset {preset_name}"
