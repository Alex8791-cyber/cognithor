#!/usr/bin/env python3
"""Generate a fresh Ed25519 keypair for the community registry Targets role.

PACK-4. Spec: docs/superpowers/specs/2026-05-05-pack4-registry-signing.md.

The Targets key signs ``registry.json``, ``recalls/active.json``, and
``publishers/*.json``. It rotates frequently — at least when compromise
is suspected, and recommended every 90 days.

This script is run during routine rotation. The resulting public key is
embedded into a new ``root.json`` (signed by the offline Root key — see
``sign_root.py``). The private key is then stored as a GitHub-Actions
secret (``REGISTRY_TARGETS_PRIVATE_KEY``).

Usage::

    python scripts/registry_signing/generate_targets_key.py --out-dir ./keys
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import os
import sys
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    out: Path = args.out_dir
    if out.exists():
        print(f"Refusing to overwrite existing directory {out}", file=sys.stderr)
        return 2
    out.mkdir(parents=True)

    priv = Ed25519PrivateKey.generate()
    pub = priv.public_key()

    pem = priv.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pub_raw = pub.public_bytes_raw()
    pub_b64 = base64.b64encode(pub_raw).decode("ascii")
    keyid = "sha256:" + hashlib.sha256(pub_raw).hexdigest()

    (out / "targets_private.pem").write_bytes(pem)
    with contextlib.suppress(OSError):
        os.chmod(out / "targets_private.pem", 0o600)
    (out / "targets_public.b64").write_text(pub_b64 + "\n", encoding="utf-8")
    (out / "targets_keyid.txt").write_text(keyid + "\n", encoding="utf-8")

    print(f"Wrote keypair to {out}")
    print(f"  Public key (base64):  {pub_b64}")
    print(f"  Keyid:                {keyid}")
    print()
    print("NEXT STEPS")
    print("  1. Set GitHub-Actions secret REGISTRY_TARGETS_PRIVATE_KEY to the")
    print("     contents of targets_private.pem (in the registry repo).")
    print("  2. Embed targets_public.b64 in a new root.json via sign_root.py")
    print("     (signed offline by your Root key).")
    print("  3. Push the new root.json to the registry repo.")
    print("  4. After clients pick it up (next sync), DELETE the old Targets")
    print("     private key from anywhere it lived.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
