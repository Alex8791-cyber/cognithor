"""Pack loader — discovers, validates, and imports installed packs.

Directory layout expected::

    <packs_dir>/
        <namespace>/
            <pack_id>/
                pack_manifest.json
                eula.md
                .eula_accepted
                pack.py          <- entrypoint (default)

Validation pipeline (per pack):
1. ``pack_manifest.json`` must exist and parse as a valid ``PackManifest``.
2. ``eula.md`` must exist and its SHA-256 must match ``manifest.eula_sha256``.
3. ``min_cognithor_version`` must be satisfied by the running version.
4. ``.eula_accepted`` must exist and its ``eula_sha256`` must match the manifest.
5. The entrypoint file (``pack.py`` by default) must exist.

A broken pack is **never** allowed to crash Core: all exceptions are caught,
logged, and swallowed.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import re
from typing import TYPE_CHECKING

from cognithor.packs.errors import PackLoadError
from cognithor.packs.interface import AgentPack, PackContext, PackManifest
from cognithor.utils.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

_log = get_logger(__name__)

# Operator prefix pattern for version specs like ">=1.0.0", "==0.9.0"
_VERSION_SPEC_RE = re.compile(r"^(?P<op>>=|<=|>|<|==)?(?P<ver>\d+\.\d+\.\d+.*)")


def _parse_version(s: str) -> tuple[int, int, int]:
    """Strip any operator prefix and parse X.Y.Z into a 3-tuple of ints.

    Pre-release / build metadata after the third component is ignored for
    comparison purposes (e.g. ``1.0.0-alpha`` -> ``(1, 0, 0)``).
    """
    m = _VERSION_SPEC_RE.match(s.strip())
    if not m:
        raise ValueError(f"Cannot parse version: {s!r}")
    raw = m.group("ver")
    parts = raw.split(".")
    major = int(parts[0])
    minor = int(parts[1]) if len(parts) > 1 else 0
    # Third part may have pre-release suffix — keep only numeric prefix
    patch_raw = parts[2] if len(parts) > 2 else "0"
    patch = int(re.match(r"\d+", patch_raw).group())  # type: ignore[union-attr]
    return (major, minor, patch)


def _version_satisfies(current: str, spec: str) -> bool:
    """Return True if *current* satisfies the version *spec*.

    Supported operators: ``>=``, ``>``, ``<=``, ``<``, ``==``.
    A bare version (no operator) is treated as ``>=``.
    """
    spec = spec.strip()
    m = _VERSION_SPEC_RE.match(spec)
    if not m:
        raise ValueError(f"Invalid version spec: {spec!r}")
    op = m.group("op") or ">="
    spec_ver = _parse_version(m.group("ver"))
    cur_ver = _parse_version(current)

    if op == ">=":
        return cur_ver >= spec_ver
    if op == ">":
        return cur_ver > spec_ver
    if op == "<=":
        return cur_ver <= spec_ver
    if op == "<":
        return cur_ver < spec_ver
    if op == "==":
        return cur_ver == spec_ver
    raise ValueError(f"Unknown operator {op!r} in spec {spec!r}")


class PackLoader:
    """Discovers and loads packs from a root directory.

    Parameters
    ----------
    packs_dir:
        Root directory that contains ``<namespace>/<pack_id>/`` sub-trees.
    cognithor_version:
        Running Cognithor version string (e.g. ``"0.92.0"``), used to check
        ``min_cognithor_version`` / ``max_cognithor_version`` constraints.
    """

    def __init__(self, *, packs_dir: Path, cognithor_version: str) -> None:
        self._root = packs_dir
        self._cognithor_version = cognithor_version
        self._loaded: dict[str, AgentPack] = {}
        # TRUST-10 backfill: record the pack-manifest schema lineage
        # in the canonical MIGRATION_LEDGER. Idempotent + best-effort
        # (see _record_pack_schema_migration). Pack loading MUST NEVER
        # fail because of this hook.
        self._record_pack_schema_migration()

    @staticmethod
    def _record_pack_schema_migration() -> None:
        """Record the pack_manifest schema lineage (v0 implicit → v1).

        v0 = pre-#395 era when packs shipped without an explicit
        ``schema_version`` field. v1 = current ``schema_version: int = 1``
        contract enforced by :class:`PackManifest`.

        Idempotent: duplicate ``migration_id`` is silently swallowed
        via :class:`MigrationChainError` suppression. Multiple
        :class:`PackLoader` instances in the same process share one
        ledger entry.
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
                    domain=MigrationDomain.PACK_MANIFEST,
                    source_version="v0-implicit",
                    target_version="v1-explicit-schema_version",
                    status=MigrationStatus.APPLIED,
                    applied_by="system",
                    item_count=-1,
                    migration_id=("pack_manifest:v0-implicit:v1-explicit-schema_version"),
                    notes=(
                        "PackManifest now requires an explicit schema_version "
                        "field (default=1) and forbids extra keys"
                    ),
                )
            )

        # TRUST-7: register a SCHEMA-kind fingerprint for the
        # PackManifest pydantic schema itself. The content_hash is
        # SHA-256 of PackManifest.model_json_schema() so any future
        # change to the schema (added field, type tightening, etc.)
        # produces a new fingerprint and divergent_names() can spot it.
        PackLoader._fingerprint_pack_manifest_schema()

    @staticmethod
    def _fingerprint_pack_manifest_schema() -> None:
        """Best-effort SCHEMA-kind fingerprint of ``PackManifest``.

        Idempotent via the ledger's content-hash de-dup: same schema
        bytes → same registration → no-op. Failures are silently
        logged and swallowed; pack discovery NEVER fails because of
        fingerprinting.
        """
        import hashlib
        import json as _json

        from cognithor.security.fingerprint import (
            FINGERPRINT_LEDGER,
            BinaryKind,
            ToolFingerprint,
        )

        try:
            schema = PackManifest.model_json_schema()
            canonical = _json.dumps(schema, sort_keys=True, ensure_ascii=False)
            digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            FINGERPRINT_LEDGER.register(
                ToolFingerprint(
                    name="pack_manifest_schema",
                    kind=BinaryKind.SCHEMA,
                    content_hash=digest,
                    version="v1",
                    notes="PackManifest pydantic schema",
                )
            )
        except (ValueError, TypeError, _json.JSONDecodeError) as exc:
            _log.debug(
                "pack.schema_fingerprint_skipped",
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def discover(self) -> list[PackManifest]:
        """Walk *packs_dir* and return every valid ``PackManifest``.

        Invalid or incomplete packs are logged and skipped — they do **not**
        raise.
        """
        manifests: list[PackManifest] = []
        if not self._root.exists():
            return manifests

        for namespace_dir in sorted(self._root.iterdir()):
            if not namespace_dir.is_dir():
                continue
            # Skip dot-prefixed directories (TRUST-4: ``.backups/`` etc.
            # are PackInstaller bookkeeping, not user-visible packs).
            if namespace_dir.name.startswith("."):
                continue
            for pack_dir in sorted(namespace_dir.iterdir()):
                if not pack_dir.is_dir():
                    continue
                if pack_dir.name.startswith("."):
                    continue
                manifest = self._validate_pack(pack_dir)
                if manifest is not None:
                    manifests.append(manifest)
        return manifests

    def load_all(self, context: PackContext) -> None:
        """Discover all valid packs and call ``register(context)`` on each.

        Exceptions from individual packs are caught, logged, and swallowed.
        """
        for manifest in self.discover():
            with contextlib.suppress(Exception):
                self._load_one(manifest, context)

    def unload_all(self, context: PackContext) -> None:
        """Call ``unregister(context)`` on every loaded pack in reverse order."""
        for qid in reversed(list(self._loaded)):
            pack = self._loaded.pop(qid)
            with contextlib.suppress(Exception):
                pack.unregister(context)

    def get(self, qualified_id: str) -> AgentPack | None:
        """Return the loaded ``AgentPack`` for *qualified_id*, or ``None``."""
        return self._loaded.get(qualified_id)

    def loaded(self) -> list[AgentPack]:
        """Return all successfully loaded packs."""
        return list(self._loaded.values())

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _validate_pack(self, pack_dir: Path) -> PackManifest | None:
        """Run the full validation pipeline for *pack_dir*.

        Returns a ``PackManifest`` on success, ``None`` on any failure.
        """
        qid = pack_dir.as_posix()  # human-readable in log messages

        # Step 1 — manifest file
        manifest_path = pack_dir / "pack_manifest.json"
        if not manifest_path.exists():
            _log.warning(
                "pack.manifest_missing",
                pack_dir=qid,
            )
            return None

        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest = PackManifest.model_validate(raw)
        except Exception as exc:
            _log.warning(
                "pack.manifest_invalid",
                pack_dir=qid,
                error=str(exc),
            )
            return None

        # Step 2 — EULA integrity
        eula_path = pack_dir / "eula.md"
        if not eula_path.exists():
            _log.warning(
                "pack.eula_missing",
                qualified_id=manifest.qualified_id,
            )
            return None

        actual_hash = hashlib.sha256(eula_path.read_bytes()).hexdigest()
        if actual_hash != manifest.eula_sha256:
            _log.warning(
                "pack.eula_hash_mismatch",
                qualified_id=manifest.qualified_id,
                expected=manifest.eula_sha256,
                actual=actual_hash,
            )
            return None

        # Step 3 — version range
        try:
            if not _version_satisfies(self._cognithor_version, manifest.min_cognithor_version):
                _log.warning(
                    "pack.version_too_low",
                    qualified_id=manifest.qualified_id,
                    requires=manifest.min_cognithor_version,
                    running=self._cognithor_version,
                )
                return None
            if manifest.max_cognithor_version is not None and not _version_satisfies(
                self._cognithor_version,
                f"<={manifest.max_cognithor_version}",
            ):
                _log.warning(
                    "pack.version_too_high",
                    qualified_id=manifest.qualified_id,
                    max_allowed=manifest.max_cognithor_version,
                    running=self._cognithor_version,
                )
                return None
        except ValueError as exc:
            _log.warning(
                "pack.version_spec_invalid",
                qualified_id=manifest.qualified_id,
                error=str(exc),
            )
            return None

        # Step 4 — EULA acceptance file
        accepted_path = pack_dir / ".eula_accepted"
        if not accepted_path.exists():
            _log.warning(
                "pack.eula_not_accepted",
                qualified_id=manifest.qualified_id,
            )
            return None

        try:
            accepted_data = json.loads(accepted_path.read_text(encoding="utf-8"))
            if accepted_data.get("eula_sha256") != manifest.eula_sha256:
                _log.warning(
                    "pack.eula_accepted_hash_mismatch",
                    qualified_id=manifest.qualified_id,
                )
                return None
        except Exception as exc:
            _log.warning(
                "pack.eula_accepted_invalid",
                qualified_id=manifest.qualified_id,
                error=str(exc),
            )
            return None

        # Step 5 — entrypoint file
        entrypoint = pack_dir / manifest.entrypoint
        if not entrypoint.exists():
            _log.warning(
                "pack.entrypoint_missing",
                qualified_id=manifest.qualified_id,
                entrypoint=manifest.entrypoint,
            )
            return None

        return manifest

    def _load_one(self, manifest: PackManifest, context: PackContext) -> None:
        """Import ``pack.py``, instantiate ``Pack``, and call ``register``.

        Audit-PR8 (CREW F1): the loaded module is inserted into
        ``sys.modules`` *before* ``exec_module`` runs. Without it,
        any relative import or import-time side-effect inside
        ``pack.py`` (e.g., ``logging.getLogger`` mutating root
        handlers, or class-level metaclass hooks that key off
        ``__name__`` lookups) re-fires on every reload — leaking
        global state across ``load_all`` calls. The matching cleanup
        on import failure prevents partial-load entries lingering.

        Audit-PR8 (CREW F7): when the inner ``register(context)``
        call raises (typically a ``ValueError`` from
        ``SourceRegistry.register`` flagging a duplicate
        ``source_id``), the original cause was lost inside the
        generic ``pack.load_failed`` log line. We now record the
        exception type alongside its message so operators can
        distinguish "pack code is broken" from "pack already
        registered, hot-reload is missing".
        """
        import sys

        qid = manifest.qualified_id
        pack_dir = self._root / manifest.namespace / manifest.pack_id
        entrypoint = pack_dir / manifest.entrypoint

        spec = importlib.util.spec_from_file_location(
            f"_cognithor_pack_{manifest.namespace}_{manifest.pack_id}",
            entrypoint,
        )
        if spec is None or spec.loader is None:
            raise PackLoadError(f"Could not create module spec for {entrypoint}")
        module_name = spec.name
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        # Snapshot the tool registry so we can detect tools that *this*
        # pack adds during ``register(context)`` and audit their risk
        # declarations afterwards. Built-in tools registered before pack
        # load are filtered out by name comparison.
        mcp = getattr(context, "mcp_client", None)
        registry = getattr(mcp, "_tool_registry", None) if mcp is not None else None
        pre_existing: frozenset[str] = (
            frozenset(registry.keys()) if isinstance(registry, dict) else frozenset()
        )
        try:
            spec.loader.exec_module(module)
            pack_cls = module.Pack
            instance: AgentPack = pack_cls(manifest)
            instance.register(context)
            self._register_tool_risks(manifest, context)
            self._warn_undeclared_tool_risks(manifest, context, pre_existing)
            self._loaded[qid] = instance
            _log.info("pack.loaded", qualified_id=qid, version=manifest.version)
            self._fingerprint_pack(manifest, entrypoint)
        except Exception as exc:
            # F1: drop the half-loaded module from sys.modules so a
            # subsequent reload doesn't import the broken state.
            sys.modules.pop(module_name, None)
            _log.warning(
                "pack.load_failed",
                qualified_id=qid,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            raise PackLoadError(f"Failed to load pack {qid!r}: {exc}") from exc

    @staticmethod
    def _fingerprint_pack(manifest: PackManifest, entrypoint: Path) -> None:
        """Best-effort PACK-kind fingerprint of the loaded pack's entrypoint.

        TRUST-7 hook (#398). The fingerprint registers the pack's
        qualified_id as the logical name and the entrypoint's
        SHA-256 (line-ending normalised) as the content hash. Lets
        the operational-trust receipt answer "which pack version
        was active during run X". Failures are silently logged and
        swallowed — pack loading NEVER fails because of fingerprinting.
        """
        from cognithor.security.fingerprint import (
            FINGERPRINT_LEDGER,
            BinaryKind,
            ToolFingerprint,
            hash_python_source,
        )

        try:
            content_hash = hash_python_source(entrypoint)
            FINGERPRINT_LEDGER.register(
                ToolFingerprint(
                    name=manifest.qualified_id,
                    kind=BinaryKind.PACK,
                    content_hash=content_hash,
                    version=manifest.version,
                    source_path=str(entrypoint),
                )
            )
        except (OSError, ValueError) as exc:
            _log.debug(
                "pack.fingerprint_skipped",
                qualified_id=manifest.qualified_id,
                error=str(exc),
            )

    @staticmethod
    def _register_tool_risks(manifest: PackManifest, context: PackContext) -> None:
        """Populate the MCP tool registry with risk levels declared in the manifest.

        The Gatekeeper reads ``risk_level`` from ``mcp_client._tool_registry``
        before falling back to its hardcoded lists (see
        ``Gatekeeper._classify_risk``). Packs therefore don't need to register
        every tool as an MCP handler to get correct risk classification —
        declaring ``tool_risks`` in the manifest is enough.

        If the pack already called ``mcp_client.register_builtin_handler``
        with its own risk_level, the existing entry is preserved (we only
        fill gaps with empty risk_level).
        """
        if not manifest.tool_risks:
            return
        mcp = getattr(context, "mcp_client", None)
        if mcp is None:
            return
        registry = getattr(mcp, "_tool_registry", None)
        if registry is None:
            return

        # Lazy import to avoid a hard dep on cognithor.models at module top
        from cognithor.models import MCPToolInfo

        pack_origin = f"pack:{manifest.qualified_id}"
        for tool_name, risk in manifest.tool_risks.items():
            existing = registry.get(tool_name)
            # Reject overrides of built-in tools — only pack-provided tools
            # may have their risk_level set via the manifest. A built-in's
            # ``server`` field never starts with ``"pack:"``; if a built-in
            # is already registered, refuse the override even when its
            # ``risk_level`` is empty (which is the default for every
            # ``register_builtin_handler`` call). Without this guard, a
            # malicious pack manifest with ``tool_risks: {delete_file:
            # "green"}`` would downgrade the Gatekeeper's RED-list to
            # GREEN and bypass risk classification entirely.
            if existing is not None:
                existing_server = getattr(existing, "server", "") or ""
                if not existing_server.startswith("pack:"):
                    _log.warning(
                        "pack_tool_risk_override_rejected",
                        pack=manifest.qualified_id,
                        tool=tool_name,
                        existing_server=existing_server,
                        attempted_risk=risk,
                    )
                    continue
                # Pack-on-pack override: keep first-pack's non-empty risk_level.
                if getattr(existing, "risk_level", ""):
                    continue
            registry[tool_name] = MCPToolInfo(
                name=tool_name,
                server=pack_origin,
                description=(existing.description if existing else ""),
                input_schema=(existing.input_schema if existing else {}),
                risk_level=risk,
            )

    @staticmethod
    def _warn_undeclared_tool_risks(
        manifest: PackManifest,
        context: PackContext,
        pre_existing: frozenset[str],
    ) -> None:
        """Surface pack tools that have no explicit risk-level declaration.

        Defense-in-depth for SEC-CRIT-2 / PACK-2: a pack can register
        builtin tools via ``mcp_client.register_builtin_handler`` without
        passing ``risk_level``. Such tools fall through to the
        Gatekeeper's hardcoded fallback, which classifies unknown tools
        as ORANGE — safe but stricter than intended and silent. Authors
        should either pass ``risk_level="..."`` to the registrar or
        declare ``tool_risks`` in ``pack_manifest.json``. Emit one
        WARNING per undeclared tool so operators notice the omission.
        """
        mcp = getattr(context, "mcp_client", None)
        registry = getattr(mcp, "_tool_registry", None) if mcp is not None else None
        if not isinstance(registry, dict):
            return
        declared = set((manifest.tool_risks or {}).keys())
        for tool_name in sorted(set(registry) - pre_existing):
            info = registry.get(tool_name)
            if info is None:
                continue
            risk = (getattr(info, "risk_level", "") or "").strip()
            if risk:
                continue
            if tool_name in declared:
                # Manifest declared a risk but ``_register_tool_risks``
                # left risk_level empty — already audited by the
                # built-in-override guard. Nothing to add here.
                continue
            _log.warning(
                "pack_tool_risk_undeclared",
                pack=manifest.qualified_id,
                tool=tool_name,
                fallback_risk="orange",
                hint=(
                    "Pass risk_level=... to register_builtin_handler "
                    "or add tool_risks in pack_manifest.json."
                ),
            )
