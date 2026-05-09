"""Tests for ``register_runtime_binaries`` — the TRUST-7 BINARY
boot-path wiring that pins Ollama / vLLM / ffmpeg / piper into the
canonical fingerprint ledger at gateway init."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from cognithor.security.binary_runtime_fingerprint import (
    RUNTIME_BINARIES,
    _RuntimeBinarySpec,
    register_runtime_binaries,
)
from cognithor.security.fingerprint import (
    BinaryKind,
    FingerprintLedger,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def fake_ollama_binary(tmp_path):
    """A fake binary on disk + a stub `shutil.which` resolver."""
    binary = tmp_path / "ollama"
    binary.write_bytes(b"\x7fELFfake-ollama")
    return binary


@pytest.fixture
def fake_vllm_binary(tmp_path):
    binary = tmp_path / "vllm"
    binary.write_bytes(b"#!/usr/bin/env python\n# fake vllm shim\n")
    return binary


# ---------------------------------------------------------------------------
# Catalog sanity
# ---------------------------------------------------------------------------


class TestRuntimeBinaryCatalog:
    def test_catalog_has_at_least_ollama_and_vllm(self) -> None:
        names = {spec.name for spec in RUNTIME_BINARIES}
        assert "ollama" in names
        assert "vllm" in names

    def test_each_spec_has_non_empty_name_and_which_name(self) -> None:
        for spec in RUNTIME_BINARIES:
            assert spec.name, f"spec missing name: {spec}"
            assert spec.which_name, f"spec missing which_name: {spec}"

    def test_ffmpeg_uses_single_dash_version_flag(self) -> None:
        """ffmpeg quirk — uses ``-version`` (single dash), not ``--version``.
        If this assertion fails, the catalog regressed and ffmpeg's
        version probe will time out on every gateway boot."""
        ffmpeg = next((s for s in RUNTIME_BINARIES if s.name == "ffmpeg"), None)
        assert ffmpeg is not None
        assert ffmpeg.version_flag == "-version"

    def test_unique_logical_names(self) -> None:
        """Two specs claiming the same name would collide in the
        ledger — surface the catalog error rather than allow it."""
        names = [spec.name for spec in RUNTIME_BINARIES]
        assert len(names) == len(set(names))


# ---------------------------------------------------------------------------
# register_runtime_binaries — discovery + fingerprint
# ---------------------------------------------------------------------------


class TestRegisterRuntimeBinaries:
    def test_returns_empty_when_no_binaries_on_path(self, tmp_path: Path) -> None:
        """Cleanest case — no relevant binaries, no fingerprints."""
        ledger = FingerprintLedger()
        with patch(
            "cognithor.security.binary_runtime_fingerprint.shutil.which",
            return_value=None,
        ):
            captured = register_runtime_binaries(ledger)
        assert captured == []
        assert len(ledger) == 0

    def test_fingerprints_only_present_binaries(self, fake_ollama_binary: Path) -> None:
        """Only Ollama on PATH → only Ollama gets fingerprinted."""
        ledger = FingerprintLedger()

        def which_stub(name: str) -> str | None:
            if name == "ollama":
                return str(fake_ollama_binary)
            return None

        with (
            patch(
                "cognithor.security.binary_runtime_fingerprint.shutil.which",
                side_effect=which_stub,
            ),
            patch(
                "cognithor.security.binary_runtime_fingerprint.fingerprint_native_binary"
            ) as mock_fp,
        ):
            # Use the real fingerprint helper but skip the version probe
            # so the test stays deterministic + fast.
            from cognithor.security.fingerprint import (
                ToolFingerprint,
                hash_native_binary,
            )

            mock_fp.return_value = ToolFingerprint(
                name="ollama",
                kind=BinaryKind.BINARY,
                content_hash=hash_native_binary(fake_ollama_binary),
                version="ollama 0.4.7",
                source_path=str(fake_ollama_binary),
            )

            captured = register_runtime_binaries(ledger)

        assert len(captured) == 1
        assert captured[0].name == "ollama"
        assert captured[0].kind == BinaryKind.BINARY
        # Ledger received it
        assert captured[0].content_hash in ledger

    def test_skips_binary_when_hashing_raises(self, tmp_path: Path) -> None:
        """A binary that vanishes mid-boot (which → path, but path
        doesn't exist) must not crash the loop."""
        ledger = FingerprintLedger()
        ghost_path = str(tmp_path / "vanished")  # doesn't exist

        def which_stub(name: str) -> str | None:
            if name == "ollama":
                return ghost_path
            return None

        with patch(
            "cognithor.security.binary_runtime_fingerprint.shutil.which",
            side_effect=which_stub,
        ):
            captured = register_runtime_binaries(ledger)

        # Ollama was discovered but couldn't be hashed → skipped
        assert captured == []
        assert len(ledger) == 0

    def test_continues_after_one_binary_fails(self, fake_vllm_binary: Path, tmp_path: Path) -> None:
        """If Ollama fails to hash but vLLM is fine, vLLM still lands
        in the ledger. Single-binary failure can't sink the rest."""
        ledger = FingerprintLedger()
        ghost_ollama = str(tmp_path / "ghost-ollama")

        def which_stub(name: str) -> str | None:
            if name == "ollama":
                return ghost_ollama
            if name == "vllm":
                return str(fake_vllm_binary)
            return None

        with patch(
            "cognithor.security.binary_runtime_fingerprint.shutil.which",
            side_effect=which_stub,
        ):
            captured = register_runtime_binaries(ledger)

        assert len(captured) == 1
        assert captured[0].name == "vllm"
        assert "vllm" in {f.name for f in ledger.filter_by_kind(BinaryKind.BINARY)}

    def test_idempotent_on_repeat_call(self, fake_ollama_binary: Path) -> None:
        """Calling the function twice in a single boot (test fixture
        re-init) must NOT register duplicates — same hash → ledger
        no-op (returns False)."""
        ledger = FingerprintLedger()

        def which_stub(name: str) -> str | None:
            return str(fake_ollama_binary) if name == "ollama" else None

        with patch(
            "cognithor.security.binary_runtime_fingerprint.shutil.which",
            side_effect=which_stub,
        ):
            first = register_runtime_binaries(ledger)
            second = register_runtime_binaries(ledger)

        # Both calls report the fingerprint they captured…
        assert len(first) == 1
        assert len(second) == 1
        # …but the ledger has only ONE entry (same hash).
        assert len(ledger) == 1

    def test_uses_module_default_ledger_when_none(self, fake_ollama_binary: Path) -> None:
        """``register_runtime_binaries(None)`` writes into the canonical
        :data:`FINGERPRINT_LEDGER`. The boot path relies on this."""
        from cognithor.security import binary_runtime_fingerprint as mod

        scratch_ledger = FingerprintLedger()
        with (
            patch.object(mod, "FINGERPRINT_LEDGER", scratch_ledger),
            patch(
                "cognithor.security.binary_runtime_fingerprint.shutil.which",
                side_effect=lambda n: (str(fake_ollama_binary) if n == "ollama" else None),
            ),
        ):
            captured = register_runtime_binaries(None)

        assert len(captured) == 1
        assert captured[0].content_hash in scratch_ledger

    def test_custom_catalog_overrides_default(self, fake_ollama_binary: Path) -> None:
        """Passing a single-spec catalog limits discovery to that one
        binary — useful for tests + targeted re-pinning."""
        ledger = FingerprintLedger()
        spec = _RuntimeBinarySpec(name="my-tool", which_name="my-tool")

        def which_stub(name: str) -> str | None:
            return str(fake_ollama_binary) if name == "my-tool" else None

        with patch(
            "cognithor.security.binary_runtime_fingerprint.shutil.which",
            side_effect=which_stub,
        ):
            captured = register_runtime_binaries(ledger, catalog=(spec,))

        assert len(captured) == 1
        assert captured[0].name == "my-tool"

    def test_passes_version_flag_to_helper(self, fake_ollama_binary: Path) -> None:
        """The ffmpeg spec uses ``-version`` (single dash). The wiring
        must forward each spec's flag verbatim — otherwise version
        probes silently fail."""
        ledger = FingerprintLedger()
        ffmpeg_spec = _RuntimeBinarySpec(
            name="ffmpeg", which_name="ffmpeg", version_flag="-version"
        )

        def which_stub(name: str) -> str | None:
            return str(fake_ollama_binary) if name == "ffmpeg" else None

        with (
            patch(
                "cognithor.security.binary_runtime_fingerprint.shutil.which",
                side_effect=which_stub,
            ),
            patch(
                "cognithor.security.binary_runtime_fingerprint.fingerprint_native_binary"
            ) as mock_fp,
        ):
            from cognithor.security.fingerprint import (
                ToolFingerprint,
                hash_native_binary,
            )

            mock_fp.return_value = ToolFingerprint(
                name="ffmpeg",
                kind=BinaryKind.BINARY,
                content_hash=hash_native_binary(fake_ollama_binary),
            )

            register_runtime_binaries(ledger, catalog=(ffmpeg_spec,))

            mock_fp.assert_called_once()
            kwargs = mock_fp.call_args.kwargs
            assert kwargs["version_flag"] == "-version", (
                f"expected version_flag='-version' to be forwarded, got: "
                f"{kwargs.get('version_flag')!r}"
            )

    def test_preserves_upstream_url_and_notes_in_fingerprint(
        self, fake_ollama_binary: Path
    ) -> None:
        """Audit-log surface depends on the metadata fields — they
        must round-trip from the catalog to the captured fingerprint."""
        ledger = FingerprintLedger()
        spec = _RuntimeBinarySpec(
            name="my-tool",
            which_name="my-tool",
            upstream_url="https://example.com/my-tool",
            notes="critical-path dep",
        )

        def which_stub(name: str) -> str | None:
            return str(fake_ollama_binary) if name == "my-tool" else None

        with patch(
            "cognithor.security.binary_runtime_fingerprint.shutil.which",
            side_effect=which_stub,
        ):
            captured = register_runtime_binaries(ledger, catalog=(spec,))

        assert len(captured) == 1
        assert captured[0].upstream_url == "https://example.com/my-tool"
        assert captured[0].notes == "critical-path dep"


# ---------------------------------------------------------------------------
# Integration — the real boot path absorbs all errors
# ---------------------------------------------------------------------------


class TestRegisterRuntimeBinariesIntegration:
    def test_real_call_does_not_raise_in_clean_environment(self) -> None:
        """End-to-end smoke: calling against the real ``shutil.which``
        and the real :data:`FINGERPRINT_LEDGER` must complete without
        raising, regardless of which binaries happen to be installed
        on the runner. Any installed binary lands in the ledger; any
        missing one is silently skipped."""
        # Use a fresh ledger so we don't pollute the global one with
        # whatever the test runner happens to have installed.
        ledger = FingerprintLedger()
        captured = register_runtime_binaries(ledger)
        # `captured` is a list (possibly empty if no runtime binaries
        # are on PATH on the test runner). Type-check is the assertion.
        assert isinstance(captured, list)
        # Every captured entry is BINARY-kind and has a valid hash.
        for fp in captured:
            assert fp.kind == BinaryKind.BINARY
            assert len(fp.content_hash) == 64
