"""Unified model download + install for Cognithor.

Handles two install paths:

1. **Ollama-native tags** (``qwen3.6:35b`` etc.) → forwards to ``ollama pull``.
2. **Community GGUF** on HuggingFace (``unsloth/Qwen3.6-27B-GGUF``) →
   downloads the picked ``.gguf`` file via ``huggingface_hub`` and imports
   it into the local Ollama via ``ollama create -f Modelfile``.

Both paths return a uniform :class:`InstallResult` so the CLI / config TUI
can report success the same way.

The module avoids importing heavy deps at top level. ``httpx`` is imported
lazily (to probe Ollama) and ``huggingface_hub`` is imported only when a
community GGUF is being installed — it's not a mandatory dependency.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from cognithor.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

log = get_logger(__name__)


_REGISTRY_PATH = Path(__file__).parent.parent / "cli" / "model_registry.json"


@dataclass(frozen=True)
class InstallResult:
    """Outcome of an install attempt."""

    model_name: str
    status: Literal["installed", "already_present", "failed"]
    local_tag: str  # The name now usable by Ollama (e.g. "qwen3.6:27b")
    message: str
    bytes_downloaded: int = 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def install_model(
    name: str,
    *,
    ollama_base_url: str = "http://localhost:11434",
    progress_cb: Callable[[str], None] | None = None,
) -> InstallResult:
    """Install a model into the local Ollama, whatever its source.

    Args:
        name: Either an Ollama tag (``"qwen3.6:35b"``) or a HuggingFace
            repo id present in the community-GGUF section of the registry
            (``"unsloth/Qwen3.6-27B-GGUF"``).
        ollama_base_url: Ollama API endpoint. Used to check presence.
        progress_cb: Optional line-oriented callback for tool output —
            useful for streaming ``ollama pull`` progress into a CLI/UI.

    Returns:
        :class:`InstallResult` describing what happened. Never raises on
        expected failures — the caller checks ``.status``.
    """
    registry = _load_registry()
    community = registry.get("providers", {}).get("community_gguf", {}).get("entries", {})

    if name in community:
        return _install_community_gguf(name, community[name], progress_cb=progress_cb)

    # Treat everything else as an Ollama tag. Presence check first so we
    # can short-circuit "already_present" without kicking off a pull.
    if _ollama_has_tag(ollama_base_url, name):
        return InstallResult(
            model_name=name,
            status="already_present",
            local_tag=name,
            message=f"{name} already installed in local Ollama.",
        )
    return _install_ollama_tag(name, progress_cb=progress_cb)


def is_installed(
    name: str,
    *,
    ollama_base_url: str = "http://localhost:11434",
) -> bool:
    """Quick check: is ``name`` already pullable from local Ollama?

    For community GGUF names (``unsloth/...``) we compare against the
    ``import_as`` field of the registry entry.
    """
    registry = _load_registry()
    community = registry.get("providers", {}).get("community_gguf", {}).get("entries", {})
    if name in community:
        tag = community[name].get("import_as", name)
        return _ollama_has_tag(ollama_base_url, tag)
    return _ollama_has_tag(ollama_base_url, name)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _load_registry() -> dict:
    if not _REGISTRY_PATH.exists():
        return {}
    with open(_REGISTRY_PATH, encoding="utf-8") as f:
        return json.load(f)


def _ollama_has_tag(base_url: str, tag: str) -> bool:
    """Check via ``/api/tags`` whether the local Ollama already has ``tag``."""
    try:
        import httpx

        resp = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=3.0)
        if resp.status_code != 200:
            return False
        names = {m.get("name", "") for m in resp.json().get("models", [])}
        return tag in names or any(n.startswith(f"{tag}:") or n == tag for n in names)
    except Exception:
        return False


def _install_ollama_tag(
    tag: str,
    *,
    progress_cb: Callable[[str], None] | None,
) -> InstallResult:
    """Run ``ollama pull <tag>`` and stream progress."""
    ollama_bin = shutil.which("ollama")
    if ollama_bin is None:
        return InstallResult(
            model_name=tag,
            status="failed",
            local_tag=tag,
            message="ollama CLI not found on PATH. Install from https://ollama.com/download.",
        )
    try:
        proc = subprocess.Popen(
            [ollama_bin, "pull", tag],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            if progress_cb:
                progress_cb(line.rstrip())
        proc.wait()
        if proc.returncode != 0:
            return InstallResult(
                model_name=tag,
                status="failed",
                local_tag=tag,
                message=f"ollama pull exited {proc.returncode}",
            )
        return InstallResult(
            model_name=tag,
            status="installed",
            local_tag=tag,
            message=f"{tag} pulled successfully.",
        )
    except Exception as exc:
        return InstallResult(
            model_name=tag,
            status="failed",
            local_tag=tag,
            message=f"ollama pull crashed: {exc}",
        )


def _install_community_gguf(
    hf_repo: str,
    entry: dict,
    *,
    progress_cb: Callable[[str], None] | None,
) -> InstallResult:
    """Download a GGUF from HF and import it into local Ollama.

    Steps:
      1. ``huggingface_hub.snapshot_download`` or ``hf_hub_download`` for the
         exact file named in ``entry["file_hint"]``.
      2. Write a minimal ``Modelfile`` pointing at that GGUF.
      3. ``ollama create <import_as> -f Modelfile``.
    """
    import_as = entry.get("import_as")
    file_hint = entry.get("file_hint")
    if not import_as or not file_hint:
        return InstallResult(
            model_name=hf_repo,
            status="failed",
            local_tag="",
            message=(
                f"Registry entry for {hf_repo} is missing 'import_as' or "
                "'file_hint' — cannot install."
            ),
        )

    # Early-out if already imported.
    if _ollama_has_tag("http://localhost:11434", import_as):
        return InstallResult(
            model_name=hf_repo,
            status="already_present",
            local_tag=import_as,
            message=f"{import_as} already imported into local Ollama.",
        )

    try:
        from huggingface_hub import hf_hub_download  # type: ignore[import-untyped]
    except ImportError:
        return InstallResult(
            model_name=hf_repo,
            status="failed",
            local_tag=import_as,
            message=(
                "Community GGUF install requires the 'huggingface_hub' package. "
                "Install with: pip install huggingface_hub"
            ),
        )

    if progress_cb:
        progress_cb(f"Downloading {file_hint} from {hf_repo} ...")

    try:
        gguf_path = hf_hub_download(repo_id=hf_repo, filename=file_hint)
    except Exception as exc:
        return InstallResult(
            model_name=hf_repo,
            status="failed",
            local_tag=import_as,
            message=f"HuggingFace download failed: {exc}",
        )

    gguf_bytes = Path(gguf_path).stat().st_size if Path(gguf_path).exists() else 0
    modelfile = Path(gguf_path).parent / f"Modelfile-{import_as.replace(':', '_')}"
    modelfile.write_text(f"FROM {gguf_path}\n", encoding="utf-8")

    if progress_cb:
        progress_cb(f"Importing into Ollama as {import_as} ...")

    ollama_bin = shutil.which("ollama")
    if ollama_bin is None:
        return InstallResult(
            model_name=hf_repo,
            status="failed",
            local_tag=import_as,
            message="ollama CLI not found — download succeeded but import skipped.",
        )
    try:
        result = subprocess.run(
            [ollama_bin, "create", import_as, "-f", str(modelfile)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return InstallResult(
                model_name=hf_repo,
                status="failed",
                local_tag=import_as,
                message=(f"ollama create exited {result.returncode}: {result.stderr[:300]}"),
                bytes_downloaded=gguf_bytes,
            )
    except Exception as exc:
        return InstallResult(
            model_name=hf_repo,
            status="failed",
            local_tag=import_as,
            message=f"ollama create crashed: {exc}",
            bytes_downloaded=gguf_bytes,
        )

    return InstallResult(
        model_name=hf_repo,
        status="installed",
        local_tag=import_as,
        message=f"{hf_repo} → {import_as} imported successfully.",
        bytes_downloaded=gguf_bytes,
    )
