#!/usr/bin/env python3
"""Generate a fresh Ed25519 keypair for the community registry Root role.

PACK-4. Spec: docs/superpowers/specs/2026-05-05-pack4-registry-signing.md.

This is a ONE-OFF operator action. Run it on an offline machine. Move
the resulting ``root_private.pem`` to a hardware token / encrypted USB.
The public key (printed and saved as ``root_public.b64``) is what gets
embedded into ``src/cognithor/skills/community/_pinned_keys.py`` for
the next Cognithor release.

Usage::

    python scripts/registry_signing/generate_root_key.py --out-dir ./keys

Outputs (all written to ``--out-dir``):
    root_private.pem  - PEM-encoded private key. Move offline immediately.
    root_public.b64   - Base64 raw public key, ready for _pinned_keys.py.
    root_keyid.txt    - SHA-256 fingerprint of the public key.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import os
import sys
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory to write the keypair into. Must not already exist.",
    )
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

    (out / "root_private.pem").write_bytes(pem)
    with contextlib_suppress_oserror():
        os.chmod(out / "root_private.pem", 0o600)
    (out / "root_public.b64").write_text(pub_b64 + "\n", encoding="utf-8")
    (out / "root_keyid.txt").write_text(keyid + "\n", encoding="utf-8")

    print(f"Wrote keypair to {out}")
    print(f"  Public key (base64):  {pub_b64}")
    print(f"  Keyid:                {keyid}")
    print()
    print("NEXT STEPS")
    print("  1. Move root_private.pem to an offline storage medium NOW.")
    print("  2. Edit src/cognithor/skills/community/_pinned_keys.py:")
    print(f"     ROOT_PUBLIC_KEY_B64 = {pub_b64!r}")
    print("  3. Cut a Cognithor release with the patched file.")
    print("  4. NEVER commit root_private.pem to git.")
    return 0


def contextlib_suppress_oserror():  # type: ignore[no-untyped-def]
    """Tiny helper so chmod on Windows is a no-op without needing import."""
    import contextlib

    return contextlib.suppress(OSError)


if __name__ == "__main__":
    raise SystemExit(main())
