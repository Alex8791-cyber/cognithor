"""Tests for ``scripts/launch_vllm_tier.py`` — the operator-side vLLM
launcher that bundles the 2026-05-09 smoke-test workarounds.

We don't import the script as a module (it lives outside ``src/``);
instead we ``runpy``-load it and call its helper functions directly
so we get coverage without needing pytest plugins for script discovery.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture(scope="module")
def launcher_module():
    """Load ``scripts/launch_vllm_tier.py`` as a module under a stable name."""
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "launch_vllm_tier.py"
    spec = importlib.util.spec_from_file_location("launch_vllm_tier", script_path)
    assert spec and spec.loader, "spec_from_file_location failed"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ───────────────────────────────────────────────────────────────────
# _build_env — env hardening
# ───────────────────────────────────────────────────────────────────


class TestBuildEnv:
    def test_path_starts_with_venv_bin(self, launcher_module) -> None:
        venv_bin = Path("/home/test/vllm-env/bin")
        env = launcher_module._build_env(venv_bin, force_arch=None)
        path_parts = env["PATH"].split(":")
        assert path_parts[0] == str(venv_bin), (
            "venv bin must be first in PATH for ninja JIT-compile to find the "
            "bundled ninja before any /mnt/c/... Windows interop binary"
        )

    def test_path_excludes_windows_mnt(self, launcher_module) -> None:
        venv_bin = Path("/home/test/vllm-env/bin")
        env = launcher_module._build_env(venv_bin, force_arch=None)
        for entry in env["PATH"].split(":"):
            assert not entry.startswith("/mnt/c/"), (
                f"Windows-mnt entry leaked into PATH: {entry!r}; flashinfer "
                "ninja PATH-lookup may hit a non-executable .exe under the mnt"
            )

    def test_hf_offline_set(self, launcher_module) -> None:
        env = launcher_module._build_env(Path("/x/bin"), force_arch=None)
        assert env["HF_HUB_OFFLINE"] == "1"
        assert env["TRANSFORMERS_OFFLINE"] == "1"

    def test_force_cuda_arch_only_when_explicit(self, launcher_module) -> None:
        e_off = launcher_module._build_env(Path("/x/bin"), force_arch=None)
        assert "FLASHINFER_CUDA_ARCH_LIST" not in e_off, (
            "force_arch=None must NOT set FLASHINFER_CUDA_ARCH_LIST — "
            "the override is RTX-50-series-specific and shouldn't be on by default"
        )
        e_on = launcher_module._build_env(Path("/x/bin"), force_arch="12.0f")
        assert e_on["FLASHINFER_CUDA_ARCH_LIST"] == "12.0f"


# ───────────────────────────────────────────────────────────────────
# _build_vllm_cmd — CLI assembly
# ───────────────────────────────────────────────────────────────────


def _make_cmd(launcher_module, **overrides: Any) -> list[str]:
    """Helper: build a default-ish CLI command and return it."""
    defaults: dict[str, Any] = {
        "python_bin": Path("/x/bin/python"),
        "model_path": "/snap/abc",
        "served_name": "org/repo",
        "port": 8000,
        "host": "0.0.0.0",
        "gpu_memory_utilization": 0.9,
        "max_model_len": None,
        "enforce_eager": False,
        "enable_prefix_caching": False,
        "speculative_config": None,
        "cpu_offload_gb": None,
        "extra_args": [],
    }
    defaults.update(overrides)
    return launcher_module._build_vllm_cmd(**defaults)


class TestBuildVllmCmd:
    def test_minimal_args(self, launcher_module) -> None:
        cmd = _make_cmd(launcher_module)
        # cmd[0] is str(python_bin); on Windows the separator differs but the
        # path components must round-trip through Path comparison.
        assert Path(cmd[0]) == Path("/x/bin/python")
        assert cmd[1:3] == ["-m", "vllm.entrypoints.openai.api_server"]
        assert "--model" in cmd
        assert Path(cmd[cmd.index("--model") + 1]) == Path("/snap/abc")
        assert "--host" in cmd
        assert cmd[cmd.index("--host") + 1] == "0.0.0.0"

    def test_speculative_config_emitted_as_json(self, launcher_module) -> None:
        """vLLM 0.20+ wants --speculative-config '{...}' (JSON), not the
        deprecated --num-speculative-tokens N. Bug #1 / Manifest migration."""
        spec = {"num_speculative_tokens": 1, "method": "ngram"}
        cmd = _make_cmd(launcher_module, speculative_config=spec)
        assert "--speculative-config" in cmd
        emitted = cmd[cmd.index("--speculative-config") + 1]
        # Must round-trip through JSON
        parsed = json.loads(emitted)
        assert parsed == spec

    def test_no_num_speculative_tokens_flag_ever_emitted(self, launcher_module) -> None:
        """The deprecated flag must never appear regardless of input shape."""
        cmd = _make_cmd(
            launcher_module,
            speculative_config={"num_speculative_tokens": 1},
        )
        assert "--num-speculative-tokens" not in cmd

    def test_cpu_offload_off_by_default(self, launcher_module) -> None:
        """Bug #4 / uvloop deadlock: cpu-offload must NOT be on the cmd line
        unless caller explicitly passed cpu_offload_gb > 0."""
        cmd = _make_cmd(launcher_module, cpu_offload_gb=None)
        assert "--cpu-offload-gb" not in cmd
        cmd_zero = _make_cmd(launcher_module, cpu_offload_gb=0)
        assert "--cpu-offload-gb" not in cmd_zero
        cmd_explicit = _make_cmd(launcher_module, cpu_offload_gb=4)
        assert "--cpu-offload-gb" in cmd_explicit
        assert cmd_explicit[cmd_explicit.index("--cpu-offload-gb") + 1] == "4"

    def test_enforce_eager_only_when_true(self, launcher_module) -> None:
        cmd_off = _make_cmd(launcher_module, enforce_eager=False)
        assert "--enforce-eager" not in cmd_off
        cmd_on = _make_cmd(launcher_module, enforce_eager=True)
        assert "--enforce-eager" in cmd_on

    def test_max_model_len_none_omits_flag(self, launcher_module) -> None:
        cmd = _make_cmd(launcher_module, max_model_len=None)
        assert "--max-model-len" not in cmd
        cmd_set = _make_cmd(launcher_module, max_model_len=16384)
        assert cmd_set[cmd_set.index("--max-model-len") + 1] == "16384"

    def test_extra_args_appended_verbatim(self, launcher_module) -> None:
        cmd = _make_cmd(
            launcher_module,
            extra_args=["--tensor-parallel-size", "2", "--quantization", "fp8"],
        )
        # Trailing args must appear in order after our default ones
        idx = cmd.index("--tensor-parallel-size")
        assert cmd[idx + 1] == "2"
        assert cmd[idx + 2] == "--quantization"
        assert cmd[idx + 3] == "fp8"


# ───────────────────────────────────────────────────────────────────
# _resolve_hf_snapshot — cache-aware HF id → path resolution
# ───────────────────────────────────────────────────────────────────


class TestResolveHfSnapshot:
    def test_returns_none_when_not_cached(self, launcher_module, tmp_path: Path) -> None:
        result = launcher_module._resolve_hf_snapshot("foo/bar", tmp_path)
        assert result is None

    def test_resolves_via_refs_main(self, launcher_module, tmp_path: Path) -> None:
        repo = tmp_path / "hub" / "models--foo--bar"
        snap = repo / "snapshots" / "abc123"
        snap.mkdir(parents=True)
        (repo / "refs").mkdir(parents=True)
        (repo / "refs" / "main").write_text("abc123\n", encoding="utf-8")
        result = launcher_module._resolve_hf_snapshot("foo/bar", tmp_path)
        assert result == snap

    def test_falls_back_to_first_snapshot(self, launcher_module, tmp_path: Path) -> None:
        repo = tmp_path / "hub" / "models--foo--bar"
        snap = repo / "snapshots" / "deadbeef"
        snap.mkdir(parents=True)
        # No refs/main → fallback to listing snapshots/ and returning one
        result = launcher_module._resolve_hf_snapshot("foo/bar", tmp_path)
        assert result == snap

    def test_handles_org_repo_with_dashes(self, launcher_module, tmp_path: Path) -> None:
        # HF transforms "qwen-team/super-mix-7b" → "models--qwen-team--super-mix-7b"
        repo = tmp_path / "hub" / "models--qwen-team--super-mix-7b"
        snap = repo / "snapshots" / "v1"
        snap.mkdir(parents=True)
        (repo / "refs").mkdir(parents=True)
        (repo / "refs" / "main").write_text("v1", encoding="utf-8")
        result = launcher_module._resolve_hf_snapshot("qwen-team/super-mix-7b", tmp_path)
        assert result == snap


# ───────────────────────────────────────────────────────────────────
# _load_sidecar — JSON robustness
# ───────────────────────────────────────────────────────────────────


class TestLoadSidecar:
    def test_missing_file_returns_empty(self, launcher_module, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(launcher_module, "_sidecar_path", lambda: tmp_path / "nope.json")
        assert launcher_module._load_sidecar() == {}

    def test_corrupt_json_returns_empty(self, launcher_module, tmp_path, monkeypatch) -> None:
        bad = tmp_path / "sidecar.json"
        bad.write_text("{not json", encoding="utf-8")
        monkeypatch.setattr(launcher_module, "_sidecar_path", lambda: bad)
        assert launcher_module._load_sidecar() == {}

    def test_valid_sidecar_parsed(self, launcher_module, tmp_path, monkeypatch) -> None:
        good = tmp_path / "sidecar.json"
        good.write_text(
            json.dumps({"recommended_tier": "x", "vllm_extras": {"a": 1}}),
            encoding="utf-8",
        )
        monkeypatch.setattr(launcher_module, "_sidecar_path", lambda: good)
        data = launcher_module._load_sidecar()
        assert data["recommended_tier"] == "x"
        assert data["vllm_extras"] == {"a": 1}
