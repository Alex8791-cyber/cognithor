"""``ToolFingerprint`` foundation (TRUST-7, operational-trust audit, 2026-05-04).

Reviewer-feedback gap: when an operator inspects a memory item, an
audit-log entry, or a run-receipt, they can reconstruct *what
happened* (TRUST-1 receipts) and *why a decision was made*
(TRUST-2 explanations) — but they cannot pin down **which exact
implementation** produced the result. Was it ``web_fetch`` v1.4.1
linked against urllib3 1.26.18, or v1.5.0 with the proxy-handling
patch? Was the Planner running ``qwen3:30b`` with the 2025-12 weight
hash or the 2026-04 retraining?

Without that pinning, post-mortem reconstruction has a blind spot:
"the same input produced different outputs" cannot be distinguished
from "the same input produced the same output but you ran it against
a different binary".

This module ships the **foundation**: a frozen ``ToolFingerprint``
dataclass + an in-memory ``FingerprintLedger`` indexed by content
hash, plus a ``BinaryKind`` enum so consumers can switch on the kind
of artefact (TOOL / MODEL / PACK / SCHEMA). Wiring this into the
gateway boot path (one snapshot per process, embedded in every run
receipt) is a deliberate follow-up — this layer stays storage-free.

Contract:

* Each fingerprint is keyed by ``content_hash`` (SHA-256 of the
  underlying artefact's canonical bytes — Python source, model
  weight file, pack zip, JSON schema).
* The same logical artefact produces the **same** content_hash
  across processes — fingerprints are deterministic, *not* keyed
  by process-local pointers.
* The ledger is **append-only**: re-registering the same hash is a
  no-op (idempotent), but a new hash for the same logical name
  appends a new entry, so an operator can spot
  "``web_fetch`` had three different SHA-256s during this audit
  window".
* Inspectors carry the responsibility to compute the hash; the
  module ships a ``hash_python_source(path)`` helper for the common
  case but does not auto-discover artefacts (that's gateway-boot
  territory).
* No DB. The follow-up PR persists fingerprints alongside the audit
  hash-chain; this layer stays memory-only for cheap testing.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from cognithor.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Artefact typing
# ---------------------------------------------------------------------------


class BinaryKind(StrEnum):
    """Kind of artefact a fingerprint pins.

    The set is closed so consumers can switch on it without an
    Unknown-fallback branch. Adding a new value is an intentional
    review point.
    """

    TOOL = "tool"  # MCP tool implementation (Python source)
    MODEL = "model"  # LLM weight file (qwen3, etc.)
    PACK = "pack"  # Agent Pack zip / installed-tree root
    SCHEMA = "schema"  # JSON-schema or pydantic model defining a contract
    BINARY = "binary"  # Generic native binary (Ollama server, vLLM, etc.)


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolFingerprint:
    """Immutable identity of a single artefact.

    Frozen so the audit log can hash it for de-duplication and embed
    it into run receipts without copy. ``content_hash`` is the
    primary key; everything else is metadata.

    Attributes
    ----------
    name:
        Stable logical name. ``"web_fetch"`` for an MCP tool,
        ``"qwen3:30b"`` for a model, ``"reddit-lead-hunter-pro"`` for
        a pack. Doesn't change when the implementation evolves.
    kind:
        Which artefact category — see :class:`BinaryKind`.
    content_hash:
        Lowercase hex SHA-256 of the artefact's canonical bytes.
        For Python source, the canonical bytes are the file content
        with normalised line endings; for model weights, the on-disk
        bytes; for pack zips, the zip bytes (not the unpacked tree).
        64 lowercase-hex chars; the constructor enforces the shape.
    version:
        Best-effort semantic version string (``"1.4.1"``,
        ``"2026.04.16"``, ``"v0.94.1"``). Empty when the artefact
        carries no version metadata.
    captured_at:
        UTC timestamp the fingerprint was *computed* (not when the
        artefact was created). Lets an operator distinguish "the
        binary changed" from "we just re-scanned the same binary".
    source_path:
        Filesystem path the bytes came from. Empty for in-memory
        artefacts. Useful for the audit log; not part of the identity
        contract — two fingerprints with the same hash but different
        source_paths are equal.
    upstream_url:
        Best-effort upstream URL (``"https://pypi.org/project/...``"``,
        ``"https://huggingface.co/Qwen/..."``). Empty when unknown.
    notes:
        Free-text breadcrumb. Keep it short.
    """

    name: str
    kind: BinaryKind
    content_hash: str
    version: str = ""
    captured_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    source_path: str = ""
    upstream_url: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            msg = "ToolFingerprint.name must be a non-empty string"
            raise ValueError(msg)
        if len(self.content_hash) != 64 or any(
            c not in "0123456789abcdef" for c in self.content_hash
        ):
            msg = (
                "ToolFingerprint.content_hash must be 64 lowercase-hex "
                f"chars (SHA-256), got {self.content_hash!r}"
            )
            raise ValueError(msg)

    @property
    def short_hash(self) -> str:
        """First 12 hex chars — readable in logs / Trace-UI columns."""
        return self.content_hash[:12]


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


class FingerprintLedger:
    """Append-only in-memory ledger of fingerprints.

    Two indices:

    * ``content_hash → ToolFingerprint`` for O(1) "have I seen this
      exact bytes before?" lookups.
    * ``name → tuple[ToolFingerprint, ...]`` for "show me every SHA
      I've seen for ``web_fetch``" (the audit-log query).

    Tests construct fresh ledgers; production code uses
    :data:`FINGERPRINT_LEDGER` (process-local default).
    """

    def __init__(self) -> None:
        self._by_hash: dict[str, ToolFingerprint] = {}
        self._by_name: dict[str, tuple[ToolFingerprint, ...]] = {}

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def register(self, fingerprint: ToolFingerprint) -> bool:
        """Register ``fingerprint``. Returns True if it was new.

        Idempotent: re-registering an identical hash is a no-op and
        returns False. Registering a *new* hash for an existing name
        appends to the name-index and returns True.
        """
        if fingerprint.content_hash in self._by_hash:
            return False
        self._by_hash[fingerprint.content_hash] = fingerprint
        existing = self._by_name.get(fingerprint.name, ())
        self._by_name[fingerprint.name] = (*existing, fingerprint)
        return True

    def remove(self, content_hash: str) -> bool:
        """Drop a fingerprint by hash. Returns True iff it existed.

        Removes the entry from both indices. Used by the test suite
        and by post-mortem replay; production code shouldn't normally
        forget fingerprints.
        """
        fp = self._by_hash.pop(content_hash, None)
        if fp is None:
            return False
        chain = self._by_name.get(fp.name, ())
        new_chain = tuple(f for f in chain if f.content_hash != content_hash)
        if new_chain:
            self._by_name[fp.name] = new_chain
        else:
            self._by_name.pop(fp.name, None)
        return True

    def clear(self) -> None:
        self._by_hash.clear()
        self._by_name.clear()

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, content_hash: str) -> ToolFingerprint | None:
        """Return the fingerprint with ``content_hash`` or ``None``."""
        return self._by_hash.get(content_hash)

    def history(self, name: str) -> tuple[ToolFingerprint, ...]:
        """Return every fingerprint registered under ``name`` (oldest first)."""
        return self._by_name.get(name, ())

    def names(self) -> list[str]:
        """Return the set of registered names, sorted."""
        return sorted(self._by_name)

    def __contains__(self, content_hash: object) -> bool:
        return content_hash in self._by_hash

    def __len__(self) -> int:
        return len(self._by_hash)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def filter_by_kind(self, kind: BinaryKind) -> list[ToolFingerprint]:
        """Return all fingerprints of a given kind, sorted by name then hash."""
        return sorted(
            (fp for fp in self._by_hash.values() if fp.kind == kind),
            key=lambda fp: (fp.name, fp.content_hash),
        )

    def divergent_names(self) -> list[str]:
        """Return names with more than one distinct hash registered.

        Lets the audit-log surface "tools that changed during this
        audit window" — the smoking-gun query when post-mortem
        reconstruction shows behaviour drift.
        """
        return sorted(name for name, chain in self._by_name.items() if len(chain) > 1)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def snapshot(self) -> list[dict[str, object]]:
        """JSON-serialisable snapshot.

        Stable ordering: by ``name`` then by ``captured_at``. Embedded
        in TRUST-1 run receipts so the Trace-UI can render
        "fingerprints in scope at run time".
        """
        return [
            {
                "name": fp.name,
                "kind": fp.kind.value,
                "content_hash": fp.content_hash,
                "short_hash": fp.short_hash,
                "version": fp.version,
                "captured_at": fp.captured_at.isoformat(),
                "source_path": fp.source_path,
                "upstream_url": fp.upstream_url,
                "notes": fp.notes,
            }
            for fp in sorted(
                self._by_hash.values(),
                key=lambda f: (f.name, f.captured_at, f.content_hash),
            )
        ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def hash_bytes(data: bytes) -> str:
    """Lowercase-hex SHA-256 of ``data``."""
    return hashlib.sha256(data).hexdigest()


def hash_python_source(path: Path | str) -> str:
    """Lowercase-hex SHA-256 of the file at ``path`` with normalised line endings.

    Reads as bytes, replaces ``b"\\r\\n"`` with ``b"\\n"`` so the same
    Python file produces the same hash on Windows and POSIX
    checkouts. Raises :class:`FileNotFoundError` if the file does not
    exist.
    """
    p = Path(path)
    raw = p.read_bytes()
    canonical = raw.replace(b"\r\n", b"\n")
    return hash_bytes(canonical)


def fingerprint_python_tool(
    *,
    name: str,
    path: Path | str,
    version: str = "",
    upstream_url: str = "",
    notes: str = "",
) -> ToolFingerprint:
    """Build a TOOL-kind fingerprint for a Python source file.

    Convenience wrapper for the gateway-boot path (one fingerprint
    per registered MCP tool). Reads the file, normalises line
    endings, computes SHA-256, builds the dataclass.
    """
    p = Path(path)
    return ToolFingerprint(
        name=name,
        kind=BinaryKind.TOOL,
        content_hash=hash_python_source(p),
        version=version,
        source_path=str(p),
        upstream_url=upstream_url,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Native-binary fingerprinting (TRUST-7 BINARY)
# ---------------------------------------------------------------------------
#
# Where ``fingerprint_python_tool`` covers Python-source artefacts, this
# section covers the executables Cognithor depends on at runtime: the
# Ollama daemon, the vLLM server, etc. The audit trail needs to pin
# *which build* of Ollama produced a planning trace so post-mortem
# reconstruction can distinguish "same input, different bytes" from
# "same input, different binary version".
#
# Native binaries are NOT line-ending-normalised — they're raw bytes.
# The hash is over the on-disk content as-is. Version is best-effort
# captured from ``<bin> --version`` with a hard timeout.


# Maximum bytes we'll read from a binary's --version output. Any
# real CLI produces << 1 KB; cap at 4 KB so a misbehaving binary
# can't lock up the gateway boot path.
_VERSION_PROBE_MAX_OUTPUT_BYTES = 4096

# Maximum seconds to wait for a --version probe. Fast tools answer
# in <100 ms; this generous cap protects against a mis-built binary
# that hangs on stdout.
_VERSION_PROBE_TIMEOUT_S = 5.0


def hash_native_binary(path: Path | str) -> str:
    """Lowercase-hex SHA-256 of the on-disk binary at ``path``.

    Unlike :func:`hash_python_source`, this does NOT normalise line
    endings — native binaries are raw bytes and any normalisation
    would produce a hash that diverges from what users get when they
    compute SHA-256 themselves (``sha256sum /usr/bin/ollama``).

    Streams the file in 64 KiB chunks so multi-GB executables (vLLM
    bundles ship with embedded model weights at times) don't pin the
    whole content into RAM. Raises :class:`FileNotFoundError` if the
    file does not exist; raises :class:`IsADirectoryError` if ``path``
    points at a directory.
    """
    p = Path(path)
    sha = hashlib.sha256()
    chunk_size = 64 * 1024
    with p.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            sha.update(chunk)
    return sha.hexdigest()


def _capture_binary_version(
    path: Path,
    *,
    version_flag: str = "--version",
    timeout_s: float = _VERSION_PROBE_TIMEOUT_S,
) -> str:
    """Best-effort: run ``<bin> --version`` and parse a single-line
    version string from stdout.

    Returns ``""`` (the empty string, which the ``ToolFingerprint``
    contract accepts as "no version metadata") on:

    * the binary not being executable from this process
    * the probe timing out
    * the binary writing nothing to stdout
    * the binary writing more than 4 KB (suspicious; treated as opaque)
    * any subprocess-level error

    The first stdout line is returned as-is, stripped of trailing
    whitespace. Callers that want a normalised semver should
    post-process; this function deliberately stays "best-effort raw"
    so it never blocks the boot path on quirky CLI output.
    """
    import subprocess

    try:
        proc = subprocess.run(
            [str(path), version_flag],
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError, FileNotFoundError):
        return ""

    raw = (proc.stdout or b"") + (proc.stderr or b"")
    if not raw:
        return ""
    if len(raw) > _VERSION_PROBE_MAX_OUTPUT_BYTES:
        return ""

    try:
        text = raw.decode("utf-8", errors="replace")
    except UnicodeDecodeError:
        return ""

    # First non-empty line, trimmed; bounded length so a 1-line CLI
    # banner with embedded shell-escapes can't spam the audit log.
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line[:200]
    return ""


def fingerprint_native_binary(
    *,
    name: str,
    path: Path | str,
    version: str | None = None,
    version_flag: str = "--version",
    upstream_url: str = "",
    notes: str = "",
) -> ToolFingerprint:
    """Build a BINARY-kind fingerprint for a native executable.

    Convenience wrapper for the gateway-boot path: when a detector
    finds an external binary Cognithor depends on (Ollama daemon,
    vLLM server, ffmpeg, ...), this captures the SHA-256 of its
    on-disk bytes plus a best-effort version string and pins it into
    a :class:`ToolFingerprint` carrying ``BinaryKind.BINARY``.

    Args:
        name: Stable logical name (``"ollama"``, ``"vllm-openai"``).
        path: Filesystem path to the executable.
        version: Explicit version string. When ``None`` (default), the
            function probes ``<path> --version`` with a 5 s timeout
            and uses the first stdout line. Pass ``""`` to skip the
            probe entirely.
        version_flag: CLI flag used for the version probe; default
            ``--version``. Override for binaries that need a
            different incantation (e.g. ``-v``).
        upstream_url: Best-effort upstream URL.
        notes: Free-text breadcrumb.

    Raises ``FileNotFoundError`` / ``IsADirectoryError`` if the path
    can't be hashed.
    """
    p = Path(path)
    captured_version = version
    if captured_version is None:
        captured_version = _capture_binary_version(p, version_flag=version_flag)
    return ToolFingerprint(
        name=name,
        kind=BinaryKind.BINARY,
        content_hash=hash_native_binary(p),
        version=captured_version,
        source_path=str(p),
        upstream_url=upstream_url,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Process-local default
# ---------------------------------------------------------------------------

# The gateway boot wires fingerprints into this instance via a
# follow-up PR; the ledger stays empty in production until that
# wiring lands. Tests construct their own :class:`FingerprintLedger`.
FINGERPRINT_LEDGER: FingerprintLedger = FingerprintLedger()


def _record_fingerprint_ledger_migration() -> None:
    """TRUST-10 self-audit: announce the fingerprint-ledger schema.

    Idempotent via the canonical ``MigrationLedger``'s duplicate-id
    check. Wrapped in ``suppress`` so import-time side effects NEVER
    raise. Test isolation is unaffected — tests that monkey-patch
    the canonical migration ledger see this step as a no-op since
    it already landed at first import.
    """
    from contextlib import suppress

    from cognithor.security.migration_ledger import (
        MIGRATION_LEDGER,
        MigrationChainError,
        MigrationDomain,
        MigrationStatus,
        MigrationStep,
    )

    with suppress(MigrationChainError, ValueError):
        MIGRATION_LEDGER.record(
            MigrationStep(
                domain=MigrationDomain.FINGERPRINT_LEDGER,
                source_version="v0-no-ledger",
                target_version="v1-dual-index-ledger",
                status=MigrationStatus.APPLIED,
                applied_by="system",
                item_count=-1,
                migration_id="fingerprint_ledger:v0-no-ledger:v1-dual-index-ledger",
                notes=(
                    "TRUST-7 FingerprintLedger schema active "
                    "(by-hash + by-name dual index, BinaryKind enum)"
                ),
            )
        )


_record_fingerprint_ledger_migration()
