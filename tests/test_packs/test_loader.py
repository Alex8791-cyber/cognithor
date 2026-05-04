"""Tests for cognithor.packs.loader — pack discovery and lifecycle."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

import pytest

from cognithor.packs.interface import PackContext
from cognithor.packs.loader import PackLoader

if TYPE_CHECKING:
    from pathlib import Path


def _write_pack(
    base: Path,
    *,
    namespace: str = "cognithor-official",
    pack_id: str = "test-pack",
    version: str = "1.0.0",
    eula_text: str = "BY INSTALLING THIS PACK YOU AGREE...",
    min_version: str = ">=0.1.0",
    license_: str = "apache-2.0",
    pack_py_body: str = (
        "from cognithor.packs.interface import AgentPack\n\n"
        "class Pack(AgentPack):\n"
        "    def register(self, ctx): pass\n"
    ),
    write_eula_accepted: bool = True,
) -> Path:
    """Create a pack directory at base/<namespace>/<pack_id>/ with all files."""
    pack_dir = base / namespace / pack_id
    pack_dir.mkdir(parents=True, exist_ok=True)
    (pack_dir / "eula.md").write_text(eula_text, encoding="utf-8")
    eula_hash = hashlib.sha256(eula_text.encode("utf-8")).hexdigest()

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "namespace": namespace,
        "pack_id": pack_id,
        "version": version,
        "display_name": "Test Pack",
        "description": "test",
        "license": license_,
        "min_cognithor_version": min_version,
        "eula_sha256": eula_hash,
        "publisher": {"id": "cognithor-official", "display_name": "Cognithor"},
    }
    (pack_dir / "pack_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (pack_dir / "pack.py").write_text(pack_py_body, encoding="utf-8")

    if write_eula_accepted:
        (pack_dir / ".eula_accepted").write_text(
            json.dumps(
                {
                    "timestamp": 1234567890.0,
                    "user": "tester",
                    "eula_sha256": eula_hash,
                    "installer_version": "0.92.0",
                }
            ),
            encoding="utf-8",
        )
    return pack_dir


@pytest.fixture
def packs_dir(tmp_path: Path) -> Path:
    return tmp_path / "packs"


class TestPackLoaderDiscovery:
    def test_discover_empty_dir(self, packs_dir: Path) -> None:
        packs_dir.mkdir(parents=True)
        loader = PackLoader(packs_dir=packs_dir, cognithor_version="0.92.0")
        assert loader.discover() == []

    def test_discover_single_pack(self, packs_dir: Path) -> None:
        _write_pack(packs_dir)
        loader = PackLoader(packs_dir=packs_dir, cognithor_version="0.92.0")
        manifests = loader.discover()
        assert len(manifests) == 1
        assert manifests[0].qualified_id == "cognithor-official/test-pack"

    def test_discover_skips_pack_with_missing_manifest(self, packs_dir: Path) -> None:
        _write_pack(packs_dir, pack_id="good")
        broken = packs_dir / "cognithor-official" / "broken"
        broken.mkdir(parents=True)
        # No manifest file
        loader = PackLoader(packs_dir=packs_dir, cognithor_version="0.92.0")
        manifests = loader.discover()
        assert len(manifests) == 1
        assert manifests[0].pack_id == "good"

    def test_discover_skips_pack_with_bad_eula_hash(self, packs_dir: Path) -> None:
        pack_dir = _write_pack(packs_dir, pack_id="badeula")
        # Corrupt the EULA after writing
        (pack_dir / "eula.md").write_text("tampered!", encoding="utf-8")
        loader = PackLoader(packs_dir=packs_dir, cognithor_version="0.92.0")
        manifests = loader.discover()
        assert manifests == []

    def test_discover_skips_pack_with_missing_eula_accepted(self, packs_dir: Path) -> None:
        _write_pack(packs_dir, pack_id="unaccepted", write_eula_accepted=False)
        loader = PackLoader(packs_dir=packs_dir, cognithor_version="0.92.0")
        manifests = loader.discover()
        assert manifests == []


class TestPackLoaderLoadAll:
    def test_load_all_calls_register(self, packs_dir: Path) -> None:
        body = (
            "from cognithor.packs.interface import AgentPack\n\n"
            "class Pack(AgentPack):\n"
            "    def __init__(self, manifest):\n"
            "        super().__init__(manifest)\n"
            "        self.calls = 0\n"
            "    def register(self, ctx):\n"
            "        self.calls += 1\n"
        )
        _write_pack(packs_dir, pack_id="counted", pack_py_body=body)
        loader = PackLoader(packs_dir=packs_dir, cognithor_version="0.92.0")
        ctx = PackContext()
        loader.load_all(ctx)
        loaded = loader.get("cognithor-official/counted")
        assert loaded is not None
        assert loaded.calls == 1

    def test_broken_pack_does_not_stop_others(self, packs_dir: Path) -> None:
        body_good = (
            "from cognithor.packs.interface import AgentPack\n\n"
            "class Pack(AgentPack):\n"
            "    def register(self, ctx): pass\n"
        )
        body_broken = "raise RuntimeError('intentional')\n"
        _write_pack(packs_dir, pack_id="good", pack_py_body=body_good)
        _write_pack(packs_dir, pack_id="broken", pack_py_body=body_broken)
        loader = PackLoader(packs_dir=packs_dir, cognithor_version="0.92.0")
        ctx = PackContext()
        loader.load_all(ctx)
        assert loader.get("cognithor-official/good") is not None
        assert loader.get("cognithor-official/broken") is None


class TestPackLoaderVersionRange:
    def test_older_cognithor_rejects_newer_pack(self, packs_dir: Path) -> None:
        _write_pack(packs_dir, pack_id="future", min_version=">=1.0.0")
        loader = PackLoader(packs_dir=packs_dir, cognithor_version="0.92.0")
        assert loader.discover() == []

    def test_version_range_accepts_exact_min(self, packs_dir: Path) -> None:
        _write_pack(packs_dir, pack_id="exact", min_version=">=0.92.0")
        loader = PackLoader(packs_dir=packs_dir, cognithor_version="0.92.0")
        assert len(loader.discover()) == 1


class TestPackLoaderFingerprint:
    """TRUST-7 hook: every successfully loaded pack registers a
    PACK-kind fingerprint in the canonical FINGERPRINT_LEDGER so the
    operational-trust receipt can show which exact pack version was
    active during a run.
    """

    def test_load_registers_pack_fingerprint(self, packs_dir: Path) -> None:
        import cognithor.security.fingerprint as fp_mod
        from cognithor.security.fingerprint import BinaryKind, FingerprintLedger

        body = (
            "from cognithor.packs.interface import AgentPack\n\n"
            "class Pack(AgentPack):\n"
            "    def register(self, ctx): pass\n"
        )
        _write_pack(packs_dir, pack_id="fp-probe", pack_py_body=body)

        isolated = FingerprintLedger()
        original = fp_mod.FINGERPRINT_LEDGER
        fp_mod.FINGERPRINT_LEDGER = isolated  # type: ignore[misc]
        try:
            loader = PackLoader(packs_dir=packs_dir, cognithor_version="0.92.0")
            loader.load_all(PackContext())
            history = isolated.history("cognithor-official/fp-probe")
            assert len(history) == 1
            fp = history[0]
            assert fp.kind == BinaryKind.PACK
            assert fp.version == "1.0.0"  # default in _write_pack
            assert len(fp.content_hash) == 64
            assert fp.source_path.endswith("pack.py")
        finally:
            fp_mod.FINGERPRINT_LEDGER = original  # type: ignore[misc]

    def test_failed_pack_does_not_register_fingerprint(self, packs_dir: Path) -> None:
        # A pack whose register() blows up must not leave a stale
        # fingerprint behind — the load failed, so the ledger should
        # not pretend the pack is "in scope".
        import cognithor.security.fingerprint as fp_mod
        from cognithor.security.fingerprint import FingerprintLedger

        body_broken = (
            "from cognithor.packs.interface import AgentPack\n\n"
            "class Pack(AgentPack):\n"
            "    def register(self, ctx): raise RuntimeError('boom')\n"
        )
        _write_pack(packs_dir, pack_id="exploding", pack_py_body=body_broken)

        isolated = FingerprintLedger()
        original = fp_mod.FINGERPRINT_LEDGER
        fp_mod.FINGERPRINT_LEDGER = isolated  # type: ignore[misc]
        try:
            loader = PackLoader(packs_dir=packs_dir, cognithor_version="0.92.0")
            loader.load_all(PackContext())  # exception is swallowed
            assert isolated.history("cognithor-official/exploding") == ()
        finally:
            fp_mod.FINGERPRINT_LEDGER = original  # type: ignore[misc]


class TestPackLoaderMigrationBackfill:
    """TRUST-10 backfill: PackLoader construction records the
    pack_manifest schema lineage into the canonical MIGRATION_LEDGER.
    """

    def test_construction_records_migration_step(self, packs_dir: Path) -> None:
        import cognithor.security.migration_ledger as mig_mod
        from cognithor.security.migration_ledger import (
            MigrationDomain,
            MigrationLedger,
            MigrationStatus,
        )

        isolated = MigrationLedger()
        original = mig_mod.MIGRATION_LEDGER
        mig_mod.MIGRATION_LEDGER = isolated  # type: ignore[misc]
        try:
            PackLoader(packs_dir=packs_dir, cognithor_version="0.92.0")
            head = isolated.head_version(MigrationDomain.PACK_MANIFEST)
            assert head == "v1-explicit-schema_version"
            step = isolated.get("pack_manifest:v0-implicit:v1-explicit-schema_version")
            assert step is not None
            assert step.status == MigrationStatus.APPLIED
            assert step.applied_by == "system"
            assert step.domain == MigrationDomain.PACK_MANIFEST
        finally:
            mig_mod.MIGRATION_LEDGER = original  # type: ignore[misc]

    def test_multiple_constructions_are_idempotent(self, packs_dir: Path) -> None:
        import cognithor.security.migration_ledger as mig_mod
        from cognithor.security.migration_ledger import (
            MigrationDomain,
            MigrationLedger,
        )

        isolated = MigrationLedger()
        original = mig_mod.MIGRATION_LEDGER
        mig_mod.MIGRATION_LEDGER = isolated  # type: ignore[misc]
        try:
            PackLoader(packs_dir=packs_dir, cognithor_version="0.92.0")
            PackLoader(packs_dir=packs_dir, cognithor_version="0.92.0")
            PackLoader(packs_dir=packs_dir, cognithor_version="0.92.0")
            assert len(isolated) == 1
            assert (
                isolated.head_version(MigrationDomain.PACK_MANIFEST) == "v1-explicit-schema_version"
            )
        finally:
            mig_mod.MIGRATION_LEDGER = original  # type: ignore[misc]


class TestPackManifestSchemaFingerprint:
    """TRUST-7 SCHEMA-kind capture: PackLoader construction registers
    a SCHEMA fingerprint of PackManifest.model_json_schema().
    """

    def test_construction_registers_schema_fingerprint(self, packs_dir: Path) -> None:
        import cognithor.security.fingerprint as fp_mod
        from cognithor.security.fingerprint import BinaryKind, FingerprintLedger

        isolated = FingerprintLedger()
        original = fp_mod.FINGERPRINT_LEDGER
        fp_mod.FINGERPRINT_LEDGER = isolated  # type: ignore[misc]
        try:
            PackLoader(packs_dir=packs_dir, cognithor_version="0.92.0")
            history = isolated.history("pack_manifest_schema")
            assert len(history) == 1
            fp = history[0]
            assert fp.kind == BinaryKind.SCHEMA
            assert fp.version == "v1"
            assert len(fp.content_hash) == 64
        finally:
            fp_mod.FINGERPRINT_LEDGER = original  # type: ignore[misc]

    def test_multiple_constructions_share_one_fingerprint(self, packs_dir: Path) -> None:
        import cognithor.security.fingerprint as fp_mod
        from cognithor.security.fingerprint import FingerprintLedger

        isolated = FingerprintLedger()
        original = fp_mod.FINGERPRINT_LEDGER
        fp_mod.FINGERPRINT_LEDGER = isolated  # type: ignore[misc]
        try:
            PackLoader(packs_dir=packs_dir, cognithor_version="0.92.0")
            PackLoader(packs_dir=packs_dir, cognithor_version="0.92.0")
            PackLoader(packs_dir=packs_dir, cognithor_version="0.92.0")
            # Same content_hash → ledger de-dups, exactly one entry.
            assert len(isolated.history("pack_manifest_schema")) == 1
        finally:
            fp_mod.FINGERPRINT_LEDGER = original  # type: ignore[misc]
