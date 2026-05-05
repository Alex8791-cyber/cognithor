#!/usr/bin/env python3
"""Sign the registry's ``root.json`` with the OFFLINE Root key.

PACK-4. Spec: docs/superpowers/specs/2026-05-05-pack4-registry-signing.md.

This script is run on the offline machine. The Root private key NEVER
leaves the offline machine. The output ``root.json`` is committed to
the registry repo.

Usage::

    python scripts/registry_signing/sign_root.py \\
        --root-key root_private.pem \\
        --targets-pubkey targets_public.b64 \\
        --version 2 \\
        --valid-days 365 \\
        --min-client-version 0.97.0 \\
        --out root.json
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)


def _canonicalise(signed: dict) -> bytes:
    return json.dumps(signed, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _keyid(pub: Ed25519PublicKey) -> str:
    raw = pub.public_bytes_raw()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-key", type=Path, required=True, help="Root private key PEM")
    parser.add_argument(
        "--targets-pubkey",
        type=Path,
        required=True,
        help="Targets public key (base64-encoded raw 32 bytes), file or string",
    )
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--valid-days", type=int, default=365)
    parser.add_argument("--min-client-version", type=str, default="0.97.0")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    pem = args.root_key.read_bytes()
    root_priv = serialization.load_pem_private_key(pem, password=None)
    if not isinstance(root_priv, Ed25519PrivateKey):
        print(f"Root key {args.root_key} is not Ed25519", file=sys.stderr)
        return 2

    targets_pub_b64 = args.targets_pubkey.read_text(encoding="utf-8").strip()
    # Validate it parses to 32 raw bytes.
    try:
        raw = base64.b64decode(targets_pub_b64.encode("ascii"))
    except Exception as exc:
        print(f"--targets-pubkey is not valid base64: {exc}", file=sys.stderr)
        return 2
    if len(raw) != 32:
        print(f"Targets public key must be 32 bytes; got {len(raw)}", file=sys.stderr)
        return 2
    targets_keyid = "sha256:" + hashlib.sha256(raw).hexdigest()

    now = datetime.now(UTC)
    valid_until = now + timedelta(days=args.valid_days)
    signed = {
        "_type": "root",
        "version": args.version,
        "issued_at": now.isoformat(),
        "valid_until": valid_until.isoformat(),
        "targets": {
            "keyid": targets_keyid,
            "method": "ed25519",
            "public_key": targets_pub_b64,
        },
        "min_client_version": args.min_client_version,
    }

    canonical = _canonicalise(signed)
    sig = root_priv.sign(canonical)
    envelope = {
        "signed": signed,
        "signatures": [
            {
                "keyid": _keyid(root_priv.public_key()),
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
    print(f"Signed root.json -> {args.out}")
    print(f"  version:               {args.version}")
    print(f"  valid_until:           {valid_until.isoformat()}")
    print(f"  Root keyid:            {envelope['signatures'][0]['keyid']}")
    print(f"  Delegated Targets key: {targets_keyid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
