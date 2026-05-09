"""Boot-path wiring for TRUST-7 BINARY fingerprints.

Auto-discovers the native binaries Cognithor depends on at runtime
(Ollama daemon, vLLM server, ffmpeg, piper TTS) and pins their on-disk
SHA-256 + best-effort version into the canonical
:data:`FINGERPRINT_LEDGER` so post-mortem reconstruction can answer
"which build of Ollama produced this trace" from the audit log alone.

Companion to :mod:`cognithor.security.fingerprint`, which ships the
``hash_native_binary`` / ``fingerprint_native_binary`` helpers. This
module owns the **discovery + boot-time registration** layer:

* Knows the canonical list of binaries to fingerprint and the
  version-probe flag each one uses.
* Resolves each binary via :func:`shutil.which` so cognithor doesn't
  depend on a hard-coded install path.
* Calls the per-binary helper, registers in the ledger, and returns
  the list of fingerprints captured.
* Best-effort: a missing binary is logged at ``debug`` level and
  skipped, never propagated. The boot path stays green even when
  Ollama isn't installed.

The ``init_tools`` gateway phase calls
:func:`register_runtime_binaries` once at boot. The result is a list
the caller can include in audit-log events ("the gateway booted with
these 3 binaries pinned").
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from cognithor.security.fingerprint import (
    FINGERPRINT_LEDGER,
    fingerprint_native_binary,
)
from cognithor.utils.logging import get_logger

if TYPE_CHECKING:
    from cognithor.security.fingerprint import (
        FingerprintLedger,
        ToolFingerprint,
    )

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Binary catalog
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _RuntimeBinarySpec:
    """One entry in the catalog of binaries Cognithor depends on at runtime."""

    name: str
    """Stable logical name surfaced in fingerprints + audit logs."""

    which_name: str
    """Argument passed to :func:`shutil.which`. Usually equals ``name``
    but can differ (e.g. ``"vllm"`` looks up ``"vllm"`` even though we
    call the resulting fingerprint ``"vllm-openai"``)."""

    version_flag: str = "--version"
    """CLI flag the binary accepts to print its version. Default
    ``--version`` covers Ollama, vLLM, ffmpeg, piper, etc."""

    upstream_url: str = ""
    """Best-effort upstream URL embedded into the fingerprint."""

    notes: str = ""
    """Free-text breadcrumb."""


# Canonical catalog. Adding a new binary is an explicit review point —
# the fingerprint becomes part of the audit-log surface, so we want
# intentional decisions about what runtime artefacts are pinned.
RUNTIME_BINARIES: tuple[_RuntimeBinarySpec, ...] = (
    _RuntimeBinarySpec(
        name="ollama",
        which_name="ollama",
        version_flag="--version",
        upstream_url="https://github.com/ollama/ollama",
        notes="Default LLM-serving daemon (cognithor.config.llm_backend_type='ollama').",
    ),
    _RuntimeBinarySpec(
        name="vllm",
        which_name="vllm",
        version_flag="--version",
        upstream_url="https://github.com/vllm-project/vllm",
        notes="Optional GPU-accelerated LLM server (Enterprise/Power tiers).",
    ),
    _RuntimeBinarySpec(
        name="ffmpeg",
        which_name="ffmpeg",
        version_flag="-version",  # ffmpeg uses single-dash -version
        upstream_url="https://ffmpeg.org/",
        notes="Media-tools dependency (audio analyse, video render).",
    ),
    _RuntimeBinarySpec(
        name="piper",
        which_name="piper",
        version_flag="--version",
        upstream_url="https://github.com/rhasspy/piper",
        notes="Local TTS backend used by the voice channel.",
    ),
)


# ---------------------------------------------------------------------------
# Boot-time registration
# ---------------------------------------------------------------------------


def register_runtime_binaries(
    ledger: FingerprintLedger | None = None,
    *,
    catalog: tuple[_RuntimeBinarySpec, ...] = RUNTIME_BINARIES,
) -> list[ToolFingerprint]:
    """Discover + fingerprint every runtime binary in ``catalog``.

    For each spec:

    1. Resolve the binary via ``shutil.which`` — skip if not on PATH.
    2. Compute SHA-256 + best-effort version via
       :func:`fingerprint_native_binary`.
    3. Register into ``ledger`` (defaults to the canonical
       :data:`FINGERPRINT_LEDGER`). Re-registering an unchanged binary
       is a no-op (ledger idempotency).

    Failure modes are absorbed individually so one missing or
    misbehaving binary doesn't block the rest:

    * Binary not on PATH → skipped, ``debug`` log.
    * Binary exists but hashing raises (permission error, deleted
      mid-read) → skipped, ``warning`` log.
    * Binary exists and hashes but version probe fails → registered
      with empty ``version`` field (ledger still gets the SHA).

    Returns:
        The fingerprints captured by THIS call (empty list when no
        binaries were found). Useful for the caller to surface in a
        boot-time audit event.
    """
    target_ledger = ledger if ledger is not None else FINGERPRINT_LEDGER
    captured: list[ToolFingerprint] = []

    for spec in catalog:
        resolved = shutil.which(spec.which_name)
        if resolved is None:
            log.debug(
                "runtime_binary_not_on_path",
                name=spec.name,
                which=spec.which_name,
            )
            continue

        path = Path(resolved)
        try:
            fingerprint = fingerprint_native_binary(
                name=spec.name,
                path=path,
                version_flag=spec.version_flag,
                upstream_url=spec.upstream_url,
                notes=spec.notes,
            )
        except (FileNotFoundError, IsADirectoryError, PermissionError, OSError) as exc:
            # `which` returned a path that vanished by the time we read
            # it (race) or whose bytes we can't access. Don't fail boot.
            log.warning(
                "runtime_binary_fingerprint_failed",
                name=spec.name,
                path=str(path),
                error=type(exc).__name__,
                error_msg=str(exc),
            )
            continue

        was_new = target_ledger.register(fingerprint)
        captured.append(fingerprint)
        log.debug(
            "runtime_binary_fingerprinted",
            name=spec.name,
            short_hash=fingerprint.short_hash,
            version=fingerprint.version or "<no-version>",
            new_to_ledger=was_new,
        )

    if captured:
        log.info(
            "runtime_binaries_registered",
            names=[f.name for f in captured],
            count=len(captured),
        )

    return captured
