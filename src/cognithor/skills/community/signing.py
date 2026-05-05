"""Ed25519 signature verifier for the community registry (TUF-Light).

PACK-4. Spec: ``docs/superpowers/specs/2026-05-05-pack4-registry-signing.md``.

Two roles:
  * Root (offline) — pinned in :mod:`._pinned_keys`. Signs only ``root.json``.
  * Targets (online) — public key delegated via ``root.json``. Signs every
    other payload (registry, recalls, publisher profiles).

Hard-fail by design: every failure raises a :class:`RegistrySignatureError`
subclass. The caller is expected to let it propagate so the surrounding
sync turns into a no-op (recalls do not fire on a sync that failed
verification).

Threading
---------
Each :class:`RegistryVerifier` instance owns a state file and is **not**
thread-safe. ``RegistrySync`` and ``PublisherVerifier`` each construct
their own. Cross-process safety relies on atomic state writes
(``*.tmp`` + :func:`os.replace`).
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import threading
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from cognithor.skills.community import _pinned_keys
from cognithor.utils.logging import get_logger

log = get_logger(__name__)

# Window forgiven for clock-skew on issued_at. Stale data (valid_until in
# the past) is NOT forgiven — the kill-switch needs to fail-closed.
_CLOCK_SKEW_TOLERANCE = timedelta(minutes=5)

# Schema version of the on-disk state file. Bumped only on incompatible
# format changes; the verifier discards the file on mismatch (forces a
# fresh root.json fetch on next call).
_STATE_SCHEMA = 1


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class RegistrySignatureError(Exception):
    """Base class. Every failure path inherits from this."""


class RegistryReplayError(RegistrySignatureError):
    """``signed.version`` is older than the last verified version."""


class RegistryStaleError(RegistrySignatureError):
    """``signed.valid_until`` lies in the past (kill-switch must fail closed)."""


class RegistryKeyError(RegistrySignatureError):
    """Pinned key absent, root.json key mismatch, or ``min_client_version`` too high."""


class RegistryNotConfiguredError(RegistrySignatureError):
    """``_pinned_keys.ROOT_PUBLIC_KEY_B64`` is ``None`` and signed-registry is required.

    Default state for a Cognithor build that has not yet activated the
    marketplace. Callers should treat this as "marketplace dormant" rather
    than "marketplace broken".
    """


# ---------------------------------------------------------------------------
# Data shape
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignedPayload:
    """The validated ``signed`` block extracted from a verified envelope.

    ``body`` carries the type-specific fields: ``targets`` for root,
    ``skills`` for registry, ``recalls`` for recalls, ``reputation_score``
    etc. for publisher.
    """

    type_: str
    version: int
    issued_at: datetime
    valid_until: datetime
    body: dict[str, Any]
    keyid: str


# ---------------------------------------------------------------------------
# Canonicalisation
# ---------------------------------------------------------------------------


def _canonicalise(signed: dict[str, Any]) -> bytes:
    """Render the ``signed`` dict for signing/verifying.

    Stable across Python versions: sorted keys, no whitespace, no ASCII
    escaping. Switching to RFC 8785 in the future is a one-line change
    here — verifier and signer both call this.
    """
    return json.dumps(
        signed,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _keyid_from_pubkey_b64(pubkey_b64: str) -> str:
    """SHA-256 fingerprint of the raw key bytes, hex-encoded with prefix."""
    raw = base64.b64decode(pubkey_b64.encode("ascii"))
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _load_pubkey(pubkey_b64: str) -> Ed25519PublicKey:
    raw = base64.b64decode(pubkey_b64.encode("ascii"))
    if len(raw) != 32:
        raise RegistryKeyError(f"Ed25519 public key must be 32 bytes, got {len(raw)}")
    return Ed25519PublicKey.from_public_bytes(raw)


# ---------------------------------------------------------------------------
# Envelope parsing & verification (the only part that touches signatures)
# ---------------------------------------------------------------------------


def verify_signed_payload(
    body: bytes,
    *,
    expected_type: str,
    public_key: Ed25519PublicKey,
    last_seen_version: int,
    now: datetime,
    expected_keyid: str | None = None,
    running_version: str | None = None,
) -> SignedPayload:
    """Validate one signed envelope against ``public_key``.

    Steps run in strict order; each must pass before the next is reached.
    On any failure, raises a :class:`RegistrySignatureError` subclass and
    leaves no side effects.

    ``running_version``: if a payload carries ``min_client_version`` in its
    body and the running Cognithor version is lower, raises
    :class:`RegistryKeyError`. Pass the actual running version (e.g. from
    ``cognithor.__version__``).
    """
    # 1. Parse JSON.
    try:
        envelope = json.loads(body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RegistrySignatureError(f"envelope is not valid UTF-8 JSON: {exc}") from exc

    # 2. Schema check: top-level keys must be exactly {"signed", "signatures"}.
    if not isinstance(envelope, dict) or set(envelope.keys()) != {"signed", "signatures"}:
        raise RegistrySignatureError(
            "envelope must have exactly the top-level keys {'signed', 'signatures'}"
        )
    signed = envelope["signed"]
    signatures = envelope["signatures"]
    if not isinstance(signed, dict) or not isinstance(signatures, list) or not signatures:
        raise RegistrySignatureError(
            "malformed envelope: signed must be dict, signatures non-empty list"
        )

    # 3. _type check.
    payload_type = signed.get("_type")
    if payload_type != expected_type:
        raise RegistrySignatureError(f"expected _type={expected_type!r}, got {payload_type!r}")

    # 4. Re-canonicalise. We do NOT trust the wire-format byte stream — a
    # signer MUST emit canonical JSON, and we re-emit canonical here so
    # whitespace / ordering on the wire is irrelevant.
    canonical = _canonicalise(signed)

    # 5+6. Find a matching signature and verify.
    sig_match: dict[str, Any] | None = None
    for sig in signatures:
        if not isinstance(sig, dict):
            continue
        if sig.get("method") != "ed25519":
            continue
        if expected_keyid is not None and sig.get("keyid") != expected_keyid:
            continue
        sig_match = sig
        break
    if sig_match is None:
        raise RegistrySignatureError(
            "no signature with method=ed25519"
            + (f" and keyid={expected_keyid!r}" if expected_keyid else "")
        )
    sig_b64 = sig_match.get("sig")
    if not isinstance(sig_b64, str):
        raise RegistrySignatureError("signature.sig must be a base64 string")
    try:
        sig_bytes = base64.b64decode(sig_b64.encode("ascii"))
    except (ValueError, TypeError) as exc:
        raise RegistrySignatureError(f"signature.sig is not valid base64: {exc}") from exc
    try:
        public_key.verify(sig_bytes, canonical)
    except InvalidSignature as exc:
        raise RegistrySignatureError("Ed25519 signature verification failed") from exc

    # 7. Replay check.
    version = signed.get("version")
    if not isinstance(version, int) or version < 0:
        raise RegistrySignatureError("signed.version must be a non-negative int")
    if version < last_seen_version:
        raise RegistryReplayError(
            f"replay rejected: incoming version {version} < last_seen {last_seen_version}"
        )

    # 8. Stale check (valid_until in the past).
    valid_until_raw = signed.get("valid_until")
    issued_at_raw = signed.get("issued_at")
    if not isinstance(valid_until_raw, str) or not isinstance(issued_at_raw, str):
        raise RegistrySignatureError(
            "signed.issued_at and signed.valid_until must be ISO-8601 strings"
        )
    try:
        valid_until = datetime.fromisoformat(valid_until_raw.replace("Z", "+00:00"))
        issued_at = datetime.fromisoformat(issued_at_raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RegistrySignatureError(f"invalid ISO-8601 timestamp: {exc}") from exc
    # Coerce naive to UTC for comparison safety.
    if valid_until.tzinfo is None:
        valid_until = valid_until.replace(tzinfo=UTC)
    if issued_at.tzinfo is None:
        issued_at = issued_at.replace(tzinfo=UTC)
    if valid_until < now:
        raise RegistryStaleError(
            f"stale: valid_until {valid_until.isoformat()} < now {now.isoformat()}"
        )

    # 9. Clock-skew guard on issued_at.
    if issued_at > now + _CLOCK_SKEW_TOLERANCE:
        raise RegistrySignatureError(
            f"issued_at {issued_at.isoformat()} is more than {_CLOCK_SKEW_TOLERANCE} in the future"
        )

    # 10. Optional min_client_version gate.
    if running_version is not None:
        min_required = signed.get("min_client_version")
        if isinstance(min_required, str) and _version_lt(running_version, min_required):
            raise RegistryKeyError(
                f"running version {running_version} below min_client_version {min_required}"
            )

    body_dict = {
        k: v
        for k, v in signed.items()
        if not k.startswith("_")
        and k
        not in {
            "version",
            "issued_at",
            "valid_until",
        }
    }
    return SignedPayload(
        type_=payload_type,
        version=version,
        issued_at=issued_at,
        valid_until=valid_until,
        body=body_dict,
        keyid=str(sig_match.get("keyid", "")),
    )


def _version_lt(running: str, required: str) -> bool:
    """Compare semver-ish ``X.Y.Z`` strings. Pre-release suffixes ignored.

    Returns ``True`` iff ``running`` is strictly less than ``required``.
    """

    def _parts(v: str) -> tuple[int, int, int]:
        nums = v.split("-", 1)[0].split(".")
        while len(nums) < 3:
            nums.append("0")
        try:
            return (int(nums[0]), int(nums[1]), int(nums[2]))
        except ValueError as exc:
            raise RegistryKeyError(f"invalid version string {v!r}") from exc

    return _parts(running) < _parts(required)


# ---------------------------------------------------------------------------
# Stateful verifier
# ---------------------------------------------------------------------------


class RegistryVerifier:
    """Stateful verifier across multiple channels.

    Persists ``last_seen.version`` per channel + the cached Targets pubkey
    in ``state_path`` (defaults to
    ``~/.cognithor/community_registry_state.json``).

    The verifier never performs HTTP itself. The caller supplies a
    ``root_loader`` callable that returns ``root.json`` bytes when the
    cached Targets key is missing or stale.
    """

    def __init__(
        self,
        *,
        state_path: Path | None = None,
        running_version: str | None = None,
    ) -> None:
        self._state_path = state_path or (
            Path.home() / ".cognithor" / "community_registry_state.json"
        )
        self._running_version = running_version
        self._lock = threading.Lock()
        self._state = self._load_state()

    # -- state ----------------------------------------------------------

    def _load_state(self) -> dict[str, Any]:
        if not self._state_path.exists():
            return {"schema": _STATE_SCHEMA, "last_seen": {}, "cached_targets_key": None}
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("registry_state_unreadable", path=str(self._state_path), error=str(exc))
            return {"schema": _STATE_SCHEMA, "last_seen": {}, "cached_targets_key": None}
        if not isinstance(data, dict) or data.get("schema") != _STATE_SCHEMA:
            log.warning("registry_state_schema_mismatch", path=str(self._state_path))
            return {"schema": _STATE_SCHEMA, "last_seen": {}, "cached_targets_key": None}
        return data

    def _persist_state(self) -> None:
        """Atomic write: tmp → replace, mode 0o600."""
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._state_path.with_suffix(self._state_path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._state, sort_keys=True), encoding="utf-8")
        with suppress(OSError):
            os.chmod(tmp, 0o600)
        os.replace(tmp, self._state_path)

    def _last_seen_version(self, channel_key: str) -> int:
        return int(self._state["last_seen"].get(channel_key, {}).get("version", 0))

    def _record_seen(self, channel_key: str, version: int, now: datetime) -> None:
        self._state["last_seen"][channel_key] = {
            "version": version,
            "verified_at": now.isoformat(),
        }
        self._persist_state()

    # -- root.json ------------------------------------------------------

    def _pinned_root_pubkey(self) -> Ed25519PublicKey:
        if not _pinned_keys.REQUIRE_SIGNED_REGISTRY:
            # Caller has source-patched out the requirement. Production
            # builds NEVER reach this branch. We still demand a pinned
            # key to even attempt verification — the alternative would
            # be silent acceptance, which is worse.
            raise RegistryNotConfiguredError(
                "REQUIRE_SIGNED_REGISTRY is False — registry verification disabled."
            )
        b64 = _pinned_keys.ROOT_PUBLIC_KEY_B64
        if not b64:
            raise RegistryNotConfiguredError(
                "Community registry has no pinned Root public key in this build. "
                "Marketplace is dormant. See "
                "docs/superpowers/specs/2026-05-05-pack4-registry-signing.md."
            )
        return _load_pubkey(b64)

    def verify_root(self, body: bytes, *, now: datetime | None = None) -> SignedPayload:
        """Verify ``root.json`` against the pinned Root public key.

        On success, caches the delegated Targets public key in state.
        Subsequent ``verify_targets_payload`` calls use it without
        re-fetching ``root.json``.
        """
        with self._lock:
            now = now or datetime.now(UTC)
            pubkey = self._pinned_root_pubkey()
            payload = verify_signed_payload(
                body,
                expected_type="root",
                public_key=pubkey,
                last_seen_version=self._last_seen_version("root"),
                now=now,
                running_version=self._running_version,
            )
            targets = payload.body.get("targets")
            if not isinstance(targets, dict):
                raise RegistryKeyError("root.json signed.targets must be a dict")
            tk_b64 = targets.get("public_key")
            if not isinstance(tk_b64, str) or not tk_b64:
                raise RegistryKeyError(
                    "root.json signed.targets.public_key must be a non-empty base64 string"
                )
            # Validate it parses, throw away result.
            _load_pubkey(tk_b64)
            self._state["cached_targets_key"] = tk_b64
            self._record_seen("root", payload.version, now)
            log.info(
                "registry_root_verified",
                version=payload.version,
                keyid=payload.keyid,
                valid_until=payload.valid_until.isoformat(),
            )
            return payload

    # -- targets-signed payloads ---------------------------------------

    def verify_targets_payload(
        self,
        body: bytes,
        *,
        expected_type: str,
        channel_key: str,
        now: datetime | None = None,
    ) -> SignedPayload:
        """Verify a registry/recalls/publisher payload against the cached Targets key.

        Raises :class:`RegistryNotConfiguredError` if no Targets key is
        cached yet — caller must call :meth:`verify_root` first (typically
        the surrounding sync does this once at the top of every run).
        """
        with self._lock:
            now = now or datetime.now(UTC)
            tk_b64 = self._state.get("cached_targets_key")
            if not isinstance(tk_b64, str) or not tk_b64:
                raise RegistryNotConfiguredError(
                    "no cached Targets key — call verify_root(root.json) first"
                )
            pubkey = _load_pubkey(tk_b64)
            expected_keyid = _keyid_from_pubkey_b64(tk_b64)
            payload = verify_signed_payload(
                body,
                expected_type=expected_type,
                public_key=pubkey,
                last_seen_version=self._last_seen_version(channel_key),
                now=now,
                expected_keyid=expected_keyid,
                running_version=self._running_version,
            )
            self._record_seen(channel_key, payload.version, now)
            log.info(
                "registry_payload_verified",
                channel=channel_key,
                type=expected_type,
                version=payload.version,
                keyid=payload.keyid,
            )
            return payload

    # -- diagnostics ----------------------------------------------------

    def is_configured(self) -> bool:
        """Whether the build has a pinned Root key and a cached Targets key.

        ``False`` means subsequent verify-calls will hard-fail; useful for
        callers that want to short-circuit (e.g. ``RegistrySync.sync_once``
        skipping the whole flow when the marketplace is dormant).
        """
        if not _pinned_keys.REQUIRE_SIGNED_REGISTRY:
            return False
        if not _pinned_keys.ROOT_PUBLIC_KEY_B64:
            return False
        return True


__all__ = [
    "RegistryKeyError",
    "RegistryNotConfiguredError",
    "RegistryReplayError",
    "RegistrySignatureError",
    "RegistryStaleError",
    "RegistryVerifier",
    "SignedPayload",
    "verify_signed_payload",
]
