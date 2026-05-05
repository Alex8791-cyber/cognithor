"""PACK-4 — Tests for the Ed25519 registry verifier (TUF-Light).

Spec: ``docs/superpowers/specs/2026-05-05-pack4-registry-signing.md`` §9.

Each numbered scenario in the spec maps to a test below. The tests
construct payloads in-memory using ephemeral keypairs — never touch the
network, never depend on a deployed registry.
"""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

if TYPE_CHECKING:
    from pathlib import Path

from cognithor.skills.community import _pinned_keys
from cognithor.skills.community.signing import (
    RegistryKeyError,
    RegistryNotConfiguredError,
    RegistryReplayError,
    RegistrySignatureError,
    RegistryStaleError,
    RegistryVerifier,
    SignedPayload,
    verify_signed_payload,
)

# ---------------------------------------------------------------------------
# Helpers — ephemeral key generation + signed-payload builders
# ---------------------------------------------------------------------------


def _make_keypair() -> tuple[Ed25519PrivateKey, str]:
    """Return ``(private, public_key_b64)`` — fresh per call."""
    priv = Ed25519PrivateKey.generate()
    pub_raw = priv.public_key().public_bytes_raw()
    return priv, base64.b64encode(pub_raw).decode("ascii")


def _keyid(pub_b64: str) -> str:
    raw = base64.b64decode(pub_b64.encode("ascii"))
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _canonicalise(signed: dict) -> bytes:
    return json.dumps(signed, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _sign_envelope(priv: Ed25519PrivateKey, signed: dict, *, keyid: str) -> bytes:
    """Build a signed JSON envelope ready for ``verify_signed_payload``."""
    canonical = _canonicalise(signed)
    sig = priv.sign(canonical)
    envelope = {
        "signed": signed,
        "signatures": [
            {
                "keyid": keyid,
                "method": "ed25519",
                "sig": base64.b64encode(sig).decode("ascii"),
            }
        ],
    }
    return json.dumps(envelope).encode("utf-8")


def _now() -> datetime:
    return datetime(2026, 5, 5, 12, 0, 0, tzinfo=UTC)


def _build_registry_signed(
    *,
    version: int = 42,
    issued_offset: timedelta = timedelta(seconds=0),
    valid_offset: timedelta = timedelta(days=14),
    type_: str = "registry",
    extra: dict | None = None,
) -> dict:
    base = {
        "_type": type_,
        "version": version,
        "issued_at": (_now() + issued_offset).isoformat(),
        "valid_until": (_now() + valid_offset).isoformat(),
        "skills": [{"name": "demo", "version": "1.0"}],
    }
    if extra:
        base.update(extra)
    return base


def _build_root_signed(
    *,
    version: int,
    targets_pub_b64: str,
    valid_offset: timedelta = timedelta(days=365),
    min_client_version: str | None = None,
) -> dict:
    body = {
        "_type": "root",
        "version": version,
        "issued_at": _now().isoformat(),
        "valid_until": (_now() + valid_offset).isoformat(),
        "targets": {
            "keyid": _keyid(targets_pub_b64),
            "method": "ed25519",
            "public_key": targets_pub_b64,
        },
    }
    if min_client_version is not None:
        body["min_client_version"] = min_client_version
    return body


# ---------------------------------------------------------------------------
# Scenario 1 — Legitimate signed payload
# ---------------------------------------------------------------------------


class TestLegitimatePayload:
    def test_registry_payload_accepted(self) -> None:
        priv, pub_b64 = _make_keypair()
        body = _sign_envelope(
            priv,
            _build_registry_signed(version=10),
            keyid=_keyid(pub_b64),
        )
        result = verify_signed_payload(
            body,
            expected_type="registry",
            public_key=priv.public_key(),
            last_seen_version=0,
            now=_now(),
        )
        assert isinstance(result, SignedPayload)
        assert result.type_ == "registry"
        assert result.version == 10
        assert result.body["skills"][0]["name"] == "demo"


# ---------------------------------------------------------------------------
# Scenario 2 — Replay (version older than last seen)
# ---------------------------------------------------------------------------


class TestReplay:
    def test_older_version_rejected(self) -> None:
        priv, pub_b64 = _make_keypair()
        body = _sign_envelope(
            priv,
            _build_registry_signed(version=5),
            keyid=_keyid(pub_b64),
        )
        with pytest.raises(RegistryReplayError, match="replay rejected"):
            verify_signed_payload(
                body,
                expected_type="registry",
                public_key=priv.public_key(),
                last_seen_version=10,  # we've already seen version 10
                now=_now(),
            )

    def test_equal_version_accepted(self) -> None:
        """Same version is OK (idempotent re-sync)."""
        priv, pub_b64 = _make_keypair()
        body = _sign_envelope(
            priv,
            _build_registry_signed(version=10),
            keyid=_keyid(pub_b64),
        )
        result = verify_signed_payload(
            body,
            expected_type="registry",
            public_key=priv.public_key(),
            last_seen_version=10,
            now=_now(),
        )
        assert result.version == 10


# ---------------------------------------------------------------------------
# Scenario 3 — Stale (valid_until in the past)
# ---------------------------------------------------------------------------


class TestStale:
    def test_stale_rejected(self) -> None:
        priv, pub_b64 = _make_keypair()
        body = _sign_envelope(
            priv,
            _build_registry_signed(
                version=1,
                issued_offset=-timedelta(days=30),
                valid_offset=-timedelta(days=1),  # expired yesterday
            ),
            keyid=_keyid(pub_b64),
        )
        with pytest.raises(RegistryStaleError, match="stale"):
            verify_signed_payload(
                body,
                expected_type="registry",
                public_key=priv.public_key(),
                last_seen_version=0,
                now=_now(),
            )


# ---------------------------------------------------------------------------
# Scenarios 4 + 5 — Tampered signed dict and tampered signature
# ---------------------------------------------------------------------------


class TestTampering:
    def test_tampered_signed_body_rejected(self) -> None:
        priv, pub_b64 = _make_keypair()
        signed = _build_registry_signed(version=1)
        body = _sign_envelope(priv, signed, keyid=_keyid(pub_b64))
        # Flip a byte in the signed.skills field.
        envelope = json.loads(body.decode("utf-8"))
        envelope["signed"]["skills"][0]["name"] = "tampered"
        tampered = json.dumps(envelope).encode("utf-8")
        with pytest.raises(RegistrySignatureError, match="signature verification failed"):
            verify_signed_payload(
                tampered,
                expected_type="registry",
                public_key=priv.public_key(),
                last_seen_version=0,
                now=_now(),
            )

    def test_tampered_signature_rejected(self) -> None:
        priv, pub_b64 = _make_keypair()
        body = _sign_envelope(priv, _build_registry_signed(version=1), keyid=_keyid(pub_b64))
        envelope = json.loads(body.decode("utf-8"))
        good_sig = envelope["signatures"][0]["sig"]
        # Flip the last byte of the signature.
        flipped = base64.b64decode(good_sig.encode("ascii"))
        flipped = flipped[:-1] + bytes([flipped[-1] ^ 0x01])
        envelope["signatures"][0]["sig"] = base64.b64encode(flipped).decode("ascii")
        tampered = json.dumps(envelope).encode("utf-8")
        with pytest.raises(RegistrySignatureError):
            verify_signed_payload(
                tampered,
                expected_type="registry",
                public_key=priv.public_key(),
                last_seen_version=0,
                now=_now(),
            )

    def test_signed_by_wrong_key_rejected(self) -> None:
        priv_good, _ = _make_keypair()
        priv_evil, pub_evil_b64 = _make_keypair()
        # Sign with the evil key but envelope-claim the good key's id.
        signed = _build_registry_signed(version=1)
        body = _sign_envelope(priv_evil, signed, keyid=_keyid(pub_evil_b64))
        with pytest.raises(RegistrySignatureError):
            verify_signed_payload(
                body,
                expected_type="registry",
                public_key=priv_good.public_key(),
                last_seen_version=0,
                now=_now(),
            )


# ---------------------------------------------------------------------------
# Scenario 6 — ROOT_PUBLIC_KEY_B64 is None → NotConfigured
# ---------------------------------------------------------------------------


class TestNotConfigured:
    def test_dormant_marketplace_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(_pinned_keys, "ROOT_PUBLIC_KEY_B64", None)
        monkeypatch.setattr(_pinned_keys, "REQUIRE_SIGNED_REGISTRY", True)
        verifier = RegistryVerifier(state_path=tmp_path / "state.json")
        with pytest.raises(RegistryNotConfiguredError, match="dormant|pinned"):
            verifier.verify_root(b"{}")
        assert verifier.is_configured() is False


# ---------------------------------------------------------------------------
# Scenario 7 — Wrong _type
# ---------------------------------------------------------------------------


class TestWrongType:
    def test_registry_validated_as_recalls_rejected(self) -> None:
        priv, pub_b64 = _make_keypair()
        body = _sign_envelope(
            priv,
            _build_registry_signed(version=1, type_="registry"),
            keyid=_keyid(pub_b64),
        )
        with pytest.raises(RegistrySignatureError, match="expected _type='recalls'"):
            verify_signed_payload(
                body,
                expected_type="recalls",
                public_key=priv.public_key(),
                last_seen_version=0,
                now=_now(),
            )


# ---------------------------------------------------------------------------
# Scenario 8 — Future-dated issued_at beyond clock-skew
# ---------------------------------------------------------------------------


class TestClockSkew:
    def test_far_future_issued_at_rejected(self) -> None:
        priv, pub_b64 = _make_keypair()
        body = _sign_envelope(
            priv,
            _build_registry_signed(
                version=1,
                issued_offset=timedelta(hours=2),  # well beyond 5min tolerance
            ),
            keyid=_keyid(pub_b64),
        )
        with pytest.raises(RegistrySignatureError, match="future"):
            verify_signed_payload(
                body,
                expected_type="registry",
                public_key=priv.public_key(),
                last_seen_version=0,
                now=_now(),
            )

    def test_near_future_issued_at_accepted(self) -> None:
        """Within 5min tolerance — small NTP drift forgiven."""
        priv, pub_b64 = _make_keypair()
        body = _sign_envelope(
            priv,
            _build_registry_signed(version=1, issued_offset=timedelta(minutes=2)),
            keyid=_keyid(pub_b64),
        )
        result = verify_signed_payload(
            body,
            expected_type="registry",
            public_key=priv.public_key(),
            last_seen_version=0,
            now=_now(),
        )
        assert result.version == 1


# ---------------------------------------------------------------------------
# Scenario 9 — Targets-key rotation via root.json (full RegistryVerifier flow)
# ---------------------------------------------------------------------------


class TestKeyRotation:
    def test_rotation_via_root_json(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        # Step 1: generate root + initial targets.
        root_priv, root_pub_b64 = _make_keypair()
        targets_priv_v1, targets_pub_v1_b64 = _make_keypair()
        targets_priv_v2, targets_pub_v2_b64 = _make_keypair()

        monkeypatch.setattr(_pinned_keys, "ROOT_PUBLIC_KEY_B64", root_pub_b64)
        monkeypatch.setattr(_pinned_keys, "REQUIRE_SIGNED_REGISTRY", True)

        verifier = RegistryVerifier(state_path=tmp_path / "state.json")

        # Step 2: bootstrap with root.json v1 → caches targets v1.
        root_v1 = _sign_envelope(
            root_priv,
            _build_root_signed(version=1, targets_pub_b64=targets_pub_v1_b64),
            keyid=_keyid(root_pub_b64),
        )
        verifier.verify_root(root_v1, now=_now())

        # Step 3: registry.json signed by targets v1 verifies.
        reg_v1 = _sign_envelope(
            targets_priv_v1,
            _build_registry_signed(version=1),
            keyid=_keyid(targets_pub_v1_b64),
        )
        result_v1 = verifier.verify_targets_payload(
            reg_v1, expected_type="registry", channel_key="registry", now=_now()
        )
        assert result_v1.version == 1

        # Step 4: rotate — root.json v2 delegates to targets v2.
        root_v2 = _sign_envelope(
            root_priv,
            _build_root_signed(version=2, targets_pub_b64=targets_pub_v2_b64),
            keyid=_keyid(root_pub_b64),
        )
        verifier.verify_root(root_v2, now=_now())

        # Step 5: registry signed by old (rotated-out) targets v1 now fails.
        reg_old = _sign_envelope(
            targets_priv_v1,
            _build_registry_signed(version=2),
            keyid=_keyid(targets_pub_v1_b64),
        )
        with pytest.raises(RegistrySignatureError):
            verifier.verify_targets_payload(
                reg_old, expected_type="registry", channel_key="registry", now=_now()
            )

        # Step 6: registry signed by new targets v2 verifies.
        reg_new = _sign_envelope(
            targets_priv_v2,
            _build_registry_signed(version=2),
            keyid=_keyid(targets_pub_v2_b64),
        )
        result_new = verifier.verify_targets_payload(
            reg_new, expected_type="registry", channel_key="registry", now=_now()
        )
        assert result_new.version == 2


# ---------------------------------------------------------------------------
# Scenario 10 — min_client_version gate
# ---------------------------------------------------------------------------


class TestMinClientVersion:
    def test_running_version_below_required_rejected(self) -> None:
        priv, pub_b64 = _make_keypair()
        body = _sign_envelope(
            priv,
            _build_registry_signed(
                version=1,
                extra={"min_client_version": "0.99.0"},
            ),
            keyid=_keyid(pub_b64),
        )
        with pytest.raises(RegistryKeyError, match="min_client_version"):
            verify_signed_payload(
                body,
                expected_type="registry",
                public_key=priv.public_key(),
                last_seen_version=0,
                now=_now(),
                running_version="0.97.0",
            )

    def test_running_version_at_or_above_accepted(self) -> None:
        priv, pub_b64 = _make_keypair()
        body = _sign_envelope(
            priv,
            _build_registry_signed(version=1, extra={"min_client_version": "0.97.0"}),
            keyid=_keyid(pub_b64),
        )
        result = verify_signed_payload(
            body,
            expected_type="registry",
            public_key=priv.public_key(),
            last_seen_version=0,
            now=_now(),
            running_version="0.97.0",
        )
        assert result.version == 1


# ---------------------------------------------------------------------------
# Scenario 11 — State persistence across verifier restarts
# ---------------------------------------------------------------------------


class TestStatePersistence:
    def test_last_seen_survives_restart(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        root_priv, root_pub_b64 = _make_keypair()
        targets_priv, targets_pub_b64 = _make_keypair()
        monkeypatch.setattr(_pinned_keys, "ROOT_PUBLIC_KEY_B64", root_pub_b64)
        monkeypatch.setattr(_pinned_keys, "REQUIRE_SIGNED_REGISTRY", True)

        state_file = tmp_path / "state.json"
        v1 = RegistryVerifier(state_path=state_file)
        root = _sign_envelope(
            root_priv,
            _build_root_signed(version=1, targets_pub_b64=targets_pub_b64),
            keyid=_keyid(root_pub_b64),
        )
        v1.verify_root(root, now=_now())
        reg = _sign_envelope(
            targets_priv,
            _build_registry_signed(version=42),
            keyid=_keyid(targets_pub_b64),
        )
        v1.verify_targets_payload(reg, expected_type="registry", channel_key="registry", now=_now())

        # New instance reads the same file.
        v2 = RegistryVerifier(state_path=state_file)
        # Replay attempt (version 41 < persisted 42) must fail.
        replay = _sign_envelope(
            targets_priv,
            _build_registry_signed(version=41),
            keyid=_keyid(targets_pub_b64),
        )
        with pytest.raises(RegistryReplayError):
            v2.verify_targets_payload(
                replay,
                expected_type="registry",
                channel_key="registry",
                now=_now(),
            )


# ---------------------------------------------------------------------------
# Scenario 12 — Schema malformation
# ---------------------------------------------------------------------------


class TestMalformedEnvelopes:
    def test_not_json_rejected(self) -> None:
        priv, _ = _make_keypair()
        with pytest.raises(RegistrySignatureError, match="not valid"):
            verify_signed_payload(
                b"not json at all",
                expected_type="registry",
                public_key=priv.public_key(),
                last_seen_version=0,
                now=_now(),
            )

    def test_missing_signed_key_rejected(self) -> None:
        priv, _ = _make_keypair()
        with pytest.raises(RegistrySignatureError, match="signed"):
            verify_signed_payload(
                b'{"signatures": []}',
                expected_type="registry",
                public_key=priv.public_key(),
                last_seen_version=0,
                now=_now(),
            )

    def test_empty_signatures_rejected(self) -> None:
        priv, _ = _make_keypair()
        body = json.dumps({"signed": _build_registry_signed(version=1), "signatures": []}).encode(
            "utf-8"
        )
        with pytest.raises(RegistrySignatureError, match="non-empty"):
            verify_signed_payload(
                body,
                expected_type="registry",
                public_key=priv.public_key(),
                last_seen_version=0,
                now=_now(),
            )

    def test_negative_version_rejected(self) -> None:
        priv, pub_b64 = _make_keypair()
        body = _sign_envelope(
            priv,
            _build_registry_signed(version=-1),
            keyid=_keyid(pub_b64),
        )
        with pytest.raises(RegistrySignatureError, match="non-negative"):
            verify_signed_payload(
                body,
                expected_type="registry",
                public_key=priv.public_key(),
                last_seen_version=0,
                now=_now(),
            )


# ---------------------------------------------------------------------------
# Smoke: verify_root requires correct _type even with valid signature
# ---------------------------------------------------------------------------


class TestVerifyRootGuards:
    def test_root_with_wrong_type_rejected(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        root_priv, root_pub_b64 = _make_keypair()
        _, targets_pub_b64 = _make_keypair()
        monkeypatch.setattr(_pinned_keys, "ROOT_PUBLIC_KEY_B64", root_pub_b64)
        monkeypatch.setattr(_pinned_keys, "REQUIRE_SIGNED_REGISTRY", True)
        verifier = RegistryVerifier(state_path=tmp_path / "state.json")

        # Build a registry-typed envelope but feed it to verify_root.
        signed_as_registry = {
            "_type": "registry",
            "version": 1,
            "issued_at": _now().isoformat(),
            "valid_until": (_now() + timedelta(days=30)).isoformat(),
            "skills": [],
        }
        body = _sign_envelope(root_priv, signed_as_registry, keyid=_keyid(root_pub_b64))
        with pytest.raises(RegistrySignatureError, match="_type"):
            verifier.verify_root(body, now=_now())

    def test_targets_payload_without_root_bootstrap(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        priv, pub_b64 = _make_keypair()
        monkeypatch.setattr(_pinned_keys, "ROOT_PUBLIC_KEY_B64", pub_b64)
        monkeypatch.setattr(_pinned_keys, "REQUIRE_SIGNED_REGISTRY", True)
        verifier = RegistryVerifier(state_path=tmp_path / "state.json")
        body = _sign_envelope(
            priv,
            _build_registry_signed(version=1),
            keyid=_keyid(pub_b64),
        )
        with pytest.raises(RegistryNotConfiguredError, match="cached Targets"):
            verifier.verify_targets_payload(
                body,
                expected_type="registry",
                channel_key="registry",
                now=_now(),
            )

    def test_root_rejects_leading_attacker_signature_entry(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """BUG-1 regression: verify_root must select the signature whose
        keyid matches the pinned Root key, not just the first ed25519 entry.

        An attacker controlling the registry-host CDN can prepend their
        own signature block. Before the fix, the verifier picked the
        first ``method=ed25519`` entry and failed crypto-verify against
        the Root key — but the failure path leaked the attacker's keyid
        into the audit log. After the fix, ``expected_keyid`` is bound to
        the pinned Root keyid, so the attacker entry is skipped entirely
        and the legitimate one is verified.
        """
        root_priv, root_pub_b64 = _make_keypair()
        attacker_priv, attacker_pub_b64 = _make_keypair()
        targets_priv, targets_pub_b64 = _make_keypair()
        monkeypatch.setattr(_pinned_keys, "ROOT_PUBLIC_KEY_B64", root_pub_b64)
        monkeypatch.setattr(_pinned_keys, "REQUIRE_SIGNED_REGISTRY", True)

        signed = _build_root_signed(version=1, targets_pub_b64=targets_pub_b64)
        canonical = _canonicalise(signed)
        # Two ed25519 signatures: attacker's first, real Root second.
        envelope = {
            "signed": signed,
            "signatures": [
                {
                    "keyid": _keyid(attacker_pub_b64),
                    "method": "ed25519",
                    "sig": base64.b64encode(attacker_priv.sign(canonical)).decode("ascii"),
                },
                {
                    "keyid": _keyid(root_pub_b64),
                    "method": "ed25519",
                    "sig": base64.b64encode(root_priv.sign(canonical)).decode("ascii"),
                },
            ],
        }
        body = json.dumps(envelope).encode("utf-8")

        verifier = RegistryVerifier(state_path=tmp_path / "state.json")
        payload = verifier.verify_root(body, now=_now())
        # Returned keyid must match the pinned Root, NOT the attacker.
        assert payload.keyid == _keyid(root_pub_b64), (
            f"expected Root keyid {_keyid(root_pub_b64)}, got {payload.keyid}"
        )
        # Reuse with targets_priv (yet another key) — confirm targets-key
        # caching took effect and attacker injection didn't poison state.
        del targets_priv  # not used after this
