#!/usr/bin/env python3
"""Sign a community-registry payload with the Targets key.

PACK-4. Spec: docs/superpowers/specs/2026-05-05-pack4-registry-signing.md.

This script is invoked by the registry's CI workflow (GitHub Actions)
whenever a new ``registry.json``, ``recalls/active.json``, or
``publishers/{username}.json`` is published. The Targets private key
comes from the ``REGISTRY_TARGETS_PRIVATE_KEY`` secret.

The script:
  1. Reads the unsigned ``signed`` block from --in (a JSON file).
  2. Validates it has the required fields (_type, version, issued_at,
     valid_until).
  3. Canonicalises (sort_keys + minimal separators).
  4. Signs with the Targets key.
  5. Writes the full envelope ``{"signed": ..., "signatures": [...]}``
     to --out.

Usage::

    python scripts/registry_signing/sign_payload.py \\
        --in unsigned/registry.json \\
        --key targets_private.pem \\
        --out signed/registry.json
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

REQUIRED_FIELDS = ("_type", "version", "issued_at", "valid_until")
ALLOWED_TYPES = frozenset({"registry", "recalls", "publisher"})


def _canonicalise(signed: dict) -> bytes:
    return json.dumps(signed, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _keyid(pub: Ed25519PublicKey) -> str:
    raw = pub.public_bytes_raw()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _validate_signed(signed: dict) -> None:
    missing = [f for f in REQUIRED_FIELDS if f not in signed]
    if missing:
        raise SystemExit(f"signed payload missing required fields: {missing}")
    if signed["_type"] not in ALLOWED_TYPES:
        raise SystemExit(
            f"signed._type must be one of {sorted(ALLOWED_TYPES)}; got {signed['_type']!r}"
        )
    if not isinstance(signed["version"], int) or signed["version"] < 0:
        raise SystemExit("signed.version must be a non-negative int")
    # ISO-8601 sanity check (and reject naive datetimes — must carry tz).
    for field in ("issued_at", "valid_until"):
        try:
            dt = datetime.fromisoformat(str(signed[field]).replace("Z", "+00:00"))
        except ValueError as exc:
            raise SystemExit(f"signed.{field} is not ISO-8601: {exc}") from exc
        if dt.tzinfo is None:
            raise SystemExit(
                f"signed.{field} must include a timezone offset (e.g. ...+00:00 or ...Z)"
            )
    issued = datetime.fromisoformat(str(signed["issued_at"]).replace("Z", "+00:00"))
    valid_until = datetime.fromisoformat(str(signed["valid_until"]).replace("Z", "+00:00"))
    if valid_until <= issued:
        raise SystemExit("signed.valid_until must be strictly after signed.issued_at")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_path", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True, help="Targets private key (PEM)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    raw = args.in_path.read_text(encoding="utf-8")
    signed = json.loads(raw)
    _validate_signed(signed)

    pem = args.key.read_bytes()
    priv = serialization.load_pem_private_key(pem, password=None)
    if not isinstance(priv, Ed25519PrivateKey):
        print(f"Key {args.key} is not an Ed25519 private key", file=sys.stderr)
        return 2

    canonical = _canonicalise(signed)
    sig = priv.sign(canonical)
    envelope = {
        "signed": signed,
        "signatures": [
            {
                "keyid": _keyid(priv.public_key()),
                "method": "ed25519",
                "sig": base64.b64encode(sig).decode("ascii"),
            }
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(envelope, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    now = datetime.now(UTC).isoformat()
    print(f"[{now}] Signed {args.in_path} -> {args.out}")
    print(f"  type:    {signed['_type']}")
    print(f"  version: {signed['version']}")
    print(f"  keyid:   {envelope['signatures'][0]['keyid']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
