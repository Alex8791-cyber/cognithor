#!/usr/bin/env python3
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Launch the vLLM OpenAI-compatible server with the safe defaults
encoded by the Hardware-Aware Runtime tier-apply (sidecar
``~/.cognithor/.hardware_aware.json``).

Bundles the workarounds discovered during the 2026-05-09 smoke-test of
the Enterprise NVFP4 tier on RTX 5090 + vLLM 0.20:

1. **HF Hub HEAD-bypass** — ``HF_HUB_OFFLINE=1`` +
   ``TRANSFORMERS_OFFLINE=1`` so a community-uploaded NVFP4 repo that
   lacks ``preprocessor_config.json`` upstream still loads from the
   already-cached snapshot (transformers' default behaviour: HEAD
   request to HF Hub *first*, fail before consulting local cache).

2. **flashinfer sm_120 CUDA-arch override** —
   ``FLASHINFER_CUDA_ARCH_LIST=12.0f``. flashinfer 0.6.x reads
   ``nvcc --version`` to decide whether to allow Blackwell sm_120; if
   the system nvcc is < 12.9 it errors out even though PyTorch is
   built against CUDA 13. The env var bypasses the version probe and
   pins the target arch directly with the ``f`` suffix vLLM 0.20+
   expects.

3. **PATH hygiene** — when the WSL ``$PATH`` includes the long chain
   of ``/mnt/c/...`` Windows interop directories, Python's ``execvp``
   for ``ninja`` (used by flashinfer JIT) can find a non-executable
   match in one of those mounts and raise ``PermissionError`` instead
   of falling through. We prepend the venv ``bin/`` and trim PATH to
   pure-Linux directories.

4. **No ``--cpu-offload-gb`` by default** — empirically triggers a
   uvloop-deadlock during the first incoming request on WSL2 + vLLM
   0.20; the API server stays bound but stops accepting connections.
   Pass ``--cpu-offload N`` explicitly only if VRAM is genuinely
   insufficient.

5. **``--host 0.0.0.0`` by default** — exposes the server on the WSL
   VM's network interface so Windows-host ``localhost:8000`` reaches
   it via WSL2's port-forward. Pass ``--host 127.0.0.1`` to keep it
   WSL-only.

6. **``--speculative-config`` JSON** instead of the deprecated
   ``--num-speculative-tokens N`` (vLLM ≥0.20 dropped the latter).

Resolution order for the model path:

* ``--model PATH`` — explicit path wins
* sidecar ``model_set_extras.planner.model_id`` — the L6 apply-engine
  emits the HF-format ID for the manifest-pinned model
* ``--model HF_ID`` — fallback to plain HF Hub ID

Usage::

    # offline-resolve from sidecar, defaults from Manifest
    python scripts/launch_vllm_tier.py

    # explicit model path + custom port
    python scripts/launch_vllm_tier.py \\
        --model ~/models/Qwen3.6-27B-NVFP4 \\
        --port 8000

    # dry-run — print the constructed command, exec nothing
    python scripts/launch_vllm_tier.py --print-only

The script does NOT attempt to download models from HuggingFace —
it expects the snapshot to already be in ``$HF_HOME/hub/...``. Use
``huggingface-cli download`` separately to pre-cache.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

# ───────────────────────────────────────────────────────────────────
# Sidecar resolution
# ───────────────────────────────────────────────────────────────────


def _sidecar_path() -> Path:
    return Path.home() / ".cognithor" / ".hardware_aware.json"


def _load_sidecar() -> dict[str, Any]:
    p = _sidecar_path()
    if not p.exists():
        return {}
    try:
        return dict(json.loads(p.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError):
        return {}


def _resolve_hf_snapshot(hf_id: str, hf_home: Path) -> Path | None:
    """Find the latest snapshot of ``org/repo`` under ``$HF_HOME/hub/``.

    HF caches as ``models--{org}--{repo}/snapshots/{revision}/`` with a
    ``refs/main`` symlink pointing at the active revision. We resolve
    that symlink rather than picking the first directory, in case
    multiple revisions are cached.
    """
    cache_root = hf_home / "hub" / f"models--{hf_id.replace('/', '--')}"
    if not cache_root.exists():
        return None

    refs_main = cache_root / "refs" / "main"
    if refs_main.exists():
        try:
            revision = refs_main.read_text(encoding="utf-8").strip()
            snapshot = cache_root / "snapshots" / revision
            if snapshot.exists():
                return snapshot
        except OSError:
            pass

    snapshots_dir = cache_root / "snapshots"
    if snapshots_dir.exists():
        revisions = list(snapshots_dir.iterdir())
        if revisions:
            return revisions[0]

    return None


# ───────────────────────────────────────────────────────────────────
# Environment hardening
# ───────────────────────────────────────────────────────────────────


def _build_env(venv_bin: Path, *, force_arch: str | None) -> dict[str, str]:
    """Construct the env dict that bypasses the 2026-05-09 smoke-test gotchas."""
    base_env = dict(os.environ)

    # Strip Windows-mnt entries from PATH; keep only Linux-standard dirs +
    # the venv bin (which has the bundled `ninja` for flashinfer JIT).
    cleaned_path = ":".join(
        [
            str(venv_bin),
            "/usr/local/sbin",
            "/usr/local/bin",
            "/usr/sbin",
            "/usr/bin",
            "/sbin",
            "/bin",
        ]
    )
    base_env["PATH"] = cleaned_path

    # HF Hub bypass — local cache only, no HEAD requests
    base_env["HF_HUB_OFFLINE"] = "1"
    base_env["TRANSFORMERS_OFFLINE"] = "1"

    # flashinfer Blackwell override (only applied if user explicitly asks
    # for it via --force-cuda-arch, since it's RTX 50-series-specific)
    if force_arch:
        base_env["FLASHINFER_CUDA_ARCH_LIST"] = force_arch

    return base_env


# ───────────────────────────────────────────────────────────────────
# vLLM CLI command builder
# ───────────────────────────────────────────────────────────────────


def _build_vllm_cmd(
    *,
    python_bin: Path,
    model_path: str,
    served_name: str | None,
    port: int,
    host: str,
    gpu_memory_utilization: float,
    max_model_len: int | None,
    enforce_eager: bool,
    enable_prefix_caching: bool,
    speculative_config: dict[str, Any] | None,
    cpu_offload_gb: int | None,
    extra_args: list[str],
) -> list[str]:
    cmd: list[str] = [
        str(python_bin),
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_path,
        "--host",
        host,
        "--port",
        str(port),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
    ]
    if served_name:
        cmd += ["--served-model-name", served_name]
    if max_model_len is not None:
        cmd += ["--max-model-len", str(max_model_len)]
    if enforce_eager:
        cmd += ["--enforce-eager"]
    if enable_prefix_caching:
        cmd += ["--enable-prefix-caching"]
    if cpu_offload_gb is not None and cpu_offload_gb > 0:
        cmd += ["--cpu-offload-gb", str(cpu_offload_gb)]
    if speculative_config:
        cmd += ["--speculative-config", json.dumps(speculative_config)]
    cmd.extend(extra_args)
    return cmd


# ───────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Launch vLLM with the Hardware-Aware tier sidecar's safe defaults."
    )
    parser.add_argument(
        "--model",
        help="HF id (org/repo) or local snapshot path. Resolves from sidecar by default.",
    )
    parser.add_argument(
        "--served-model-name",
        help=(
            "Public model ID surfaced via /v1/models. "
            "Defaults to --model when an HF id, else falls through."
        ),
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="0.0.0.0 (default, reachable from Windows host) or 127.0.0.1 (WSL-internal only).",
    )
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument(
        "--cpu-offload", type=int, default=0, help="Disabled by default (uvloop deadlock)."
    )
    parser.add_argument(
        "--force-cuda-arch",
        help="Set FLASHINFER_CUDA_ARCH_LIST (e.g. '12.0f' for Blackwell sm_120 with old nvcc).",
    )
    parser.add_argument("--no-eager", action="store_true", help="Override sidecar enforce_eager.")
    parser.add_argument("--no-prefix-caching", action="store_true")
    parser.add_argument(
        "--venv",
        default=str(Path.home() / "vllm-env"),
        help="Path to vLLM venv. Looked up at $venv/bin/python and $venv/bin (for ninja).",
    )
    parser.add_argument(
        "--hf-home",
        default=os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")),
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print the constructed command + env without executing.",
    )
    parser.add_argument(
        "vllm_args",
        nargs=argparse.REMAINDER,
        help="Trailing args passed verbatim to vllm.entrypoints (after --).",
    )
    args = parser.parse_args(argv)

    venv_path = Path(args.venv).expanduser()
    venv_bin = venv_path / "bin"
    python_bin = venv_bin / "python"
    # In --print-only mode the venv may not exist (e.g. running on the
    # Windows host to inspect the command before sshing into WSL).
    if not python_bin.exists() and not args.print_only:
        print(f"[ERROR] vLLM venv python not found at {python_bin}", file=sys.stderr)
        return 2

    sidecar = _load_sidecar()
    extras = sidecar.get("vllm_extras", {}) or {}
    model_set_extras = sidecar.get("model_set_extras", {}) or {}

    # Resolve model
    if args.model:
        model_arg: str = args.model
    else:
        planner = (model_set_extras.get("planner") or {}).get("name")
        if not planner:
            print(
                "[ERROR] No --model given and sidecar lacks model_set_extras.planner.name. "
                "Run `cognithor doctor --apply-recommendation` first.",
                file=sys.stderr,
            )
            return 3
        model_arg = planner

    # If model_arg looks like an HF id, try to resolve to a local snapshot path so
    # HF_HUB_OFFLINE doesn't fail trying to consult the registry for it.
    served_name = args.served_model_name or (model_arg if "/" in model_arg else None)
    if "/" in model_arg and not Path(model_arg).exists():
        snapshot = _resolve_hf_snapshot(model_arg, Path(args.hf_home).expanduser())
        if snapshot:
            print(f"[info] resolved {model_arg} → {snapshot}", file=sys.stderr)
            model_arg = str(snapshot)
        else:
            print(
                f"[warn] {model_arg} not in HF cache at {args.hf_home}/hub/. "
                "Server may fail with HF_HUB_OFFLINE=1.",
                file=sys.stderr,
            )

    # GPU memory util — sidecar wins if not overridden
    gpu_mem_util = args.gpu_memory_utilization
    if gpu_mem_util is None:
        gpu_mem_util = (sidecar.get("vllm") or {}).get("gpu_memory_utilization", 0.9)

    # max_model_len
    max_len = args.max_model_len
    if max_len is None:
        max_len = (sidecar.get("vllm") or {}).get("max_model_len")

    # enforce_eager
    sidecar_eager = (sidecar.get("vllm") or {}).get("enforce_eager", False)
    enforce_eager = sidecar_eager and not args.no_eager

    # prefix caching
    prefix_caching = bool(extras.get("enable_prefix_caching", True)) and not args.no_prefix_caching

    # speculative_config
    spec_cfg = extras.get("speculative_config")
    if not isinstance(spec_cfg, dict) or not spec_cfg:
        spec_cfg = None

    cpu_offload = args.cpu_offload if args.cpu_offload > 0 else None

    # Build env + command
    env = _build_env(venv_bin, force_arch=args.force_cuda_arch)
    env["HF_HOME"] = str(Path(args.hf_home).expanduser())

    cmd = _build_vllm_cmd(
        python_bin=python_bin,
        model_path=model_arg,
        served_name=served_name,
        port=args.port,
        host=args.host,
        gpu_memory_utilization=float(gpu_mem_util),
        max_model_len=int(max_len) if max_len else None,
        enforce_eager=bool(enforce_eager),
        enable_prefix_caching=prefix_caching,
        speculative_config=spec_cfg,
        cpu_offload_gb=cpu_offload,
        extra_args=[a for a in args.vllm_args if a != "--"],
    )

    if args.print_only:
        print("ENV (overrides):")
        for key in (
            "HF_HOME",
            "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE",
            "PATH",
            "FLASHINFER_CUDA_ARCH_LIST",
        ):
            if key in env:
                print(f"  {key}={env[key]}")
        print("\nCOMMAND:")
        # shlex.quote the parts so the user can copy-paste
        import shlex

        print("  " + " ".join(shlex.quote(p) for p in cmd))
        return 0

    # exec
    print(f"[launch] {' '.join(cmd[:6])} ...", file=sys.stderr)
    try:
        return subprocess.call(cmd, env=env)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 4
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    # ``shutil.which`` is imported but unused here; keeping the import so the
    # module exposes the helper for tests/imports that may need it.
    _ = shutil.which
    sys.exit(main())
