"""CLI commands for `cognithor models list` and `cognithor models install`.

Kept intentionally thin — real logic lives in
:mod:`cognithor.core.model_installer` and the JSON registry.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REGISTRY_PATH = Path(__file__).parent / "model_registry.json"


def cmd_list() -> int:
    """Print the known-models registry grouped by provider."""
    if not _REGISTRY_PATH.exists():
        print(f"registry not found at {_REGISTRY_PATH}", file=sys.stderr)
        return 1
    with open(_REGISTRY_PATH, encoding="utf-8") as f:
        data = json.load(f)
    print(f"Cognithor model registry (updated {data.get('updated', '?')})\n")
    for provider, block in data.get("providers", {}).items():
        if not block.get("models"):
            continue
        print(f"== {provider} ==")
        if block.get("description"):
            print(f"   {block['description']}")
        for m in block["models"]:
            entry = block.get("entries", {}).get(m)
            if entry:
                kind = entry.get("kind", "text")
                import_as = entry.get("import_as", m)
                print(f"   {m}  →  {import_as}  [{kind}]")
            else:
                print(f"   {m}")
        print()
    return 0


def cmd_install(name: str) -> int:
    """Install the model called ``name`` into local Ollama.

    Routes Ollama tags straight to ``ollama pull`` and HF repo ids through
    the community-GGUF path (download + ``ollama create``).
    """
    from cognithor.core.model_installer import install_model

    def _progress(line: str) -> None:
        # One line per Ollama progress update — let the user see what's
        # happening during a multi-GB download.
        print(line, flush=True)

    print(f"Installing {name} ...")
    result = install_model(name, progress_cb=_progress)

    status_icon = {
        "installed": "[OK]",
        "already_present": "[=]",
        "failed": "[X]",
    }.get(result.status, "[?]")

    print(f"\n{status_icon} {result.message}")
    if result.local_tag and result.local_tag != result.model_name:
        print(f"    Available in Ollama as: {result.local_tag}")
    if result.bytes_downloaded:
        mb = result.bytes_downloaded / 1024 / 1024
        print(f"    Downloaded: {mb:.0f} MB")

    return 0 if result.status in ("installed", "already_present") else 1
