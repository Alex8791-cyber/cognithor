"""Owner-side signing tool for the Hardware-Aware Manifest (TUF-Light).

Signs `manifest/v2/{tiers,models,pricing}.yaml` with an Ed25519 private
key and writes the signature to `manifest/v2/manifest.sig`.

Usage:
    python scripts/sign_manifest.py --key /secure/path/manifest_targets.key.pem
    python scripts/sign_manifest.py --verify              # verify existing sig

The pinned PUBLIC key MUST be patched into
`src/cognithor/_pinned_keys.py` as `HARDWARE_MANIFEST_TARGETS_KEY`
before clients can verify the signature.

Trust model:
- The Ed25519 PRIVATE key never leaves the operator's offline machine.
- Manifest updates (YAML PRs to `main`) are signed locally, the .sig
  file is committed alongside the YAML.
- Clients verify via `cognithor.system.manifest_loader.ManifestLoader._verify_signature`.
- Compromised targets-key: rotate via PACK-4-style root-signed delegation
  + `manifest/recalls/active.json` to force-fail compromised manifests.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import sys
from pathlib import Path

# We rely on the same crypto primitive PACK-4 uses (cryptography lib).
try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )
except ImportError:
    print(
        "ERROR: pip install cryptography",
        file=sys.stderr,
    )
    sys.exit(2)


REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_ROOT = REPO_ROOT / "manifest" / "v2"


def _payload_bytes() -> bytes:
    """The signed payload = concat of tiers.yaml + models.yaml.

    pricing.yaml is included via SHA-256 reference instead of inline so the
    targets-key holder can rotate prices independently if a separate
    pricing-key is introduced later.
    """
    tiers = (MANIFEST_ROOT / "tiers.yaml").read_bytes()
    models = (MANIFEST_ROOT / "models.yaml").read_bytes()
    pricing_hash = (
        hashlib.sha256((MANIFEST_ROOT / "pricing.yaml").read_bytes()).hexdigest().encode("ascii")
    )
    return tiers + b"\n--MANIFEST-DELIM--\n" + models + b"\n--PRICING-SHA256:" + pricing_hash


def _load_private_key(key_path: Path) -> Ed25519PrivateKey:
    raw = key_path.read_bytes()
    try:
        key = serialization.load_pem_private_key(raw, password=None)
    except ValueError:
        key = serialization.load_der_private_key(raw, password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise SystemExit("ERROR: key must be Ed25519")
    return key


def cmd_sign(key_path: Path) -> int:
    if not MANIFEST_ROOT.exists():
        print(f"ERROR: manifest root not found: {MANIFEST_ROOT}", file=sys.stderr)
        return 2
    priv = _load_private_key(key_path)
    payload = _payload_bytes()
    sig = priv.sign(payload)
    sig_b64 = base64.b64encode(sig).decode("ascii")

    sig_path = MANIFEST_ROOT / "manifest.sig"
    sig_path.write_text(sig_b64 + "\n", encoding="utf-8")

    pub = priv.public_key()
    pub_raw = pub.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    pub_b64 = base64.b64encode(pub_raw).decode("ascii")

    print(f"[OK] Signed: {sig_path}")
    print(f"     Payload-Hash (SHA-256): {hashlib.sha256(payload).hexdigest()}")
    print()
    print("Patch the following into src/cognithor/_pinned_keys.py:")
    print(f'    HARDWARE_MANIFEST_TARGETS_KEY = "ed25519:{pub_b64}"')
    print()
    print("Then commit:")
    print("    git add manifest/v2/manifest.sig src/cognithor/_pinned_keys.py")
    print('    git commit -m "manifest: rotate signature for v2026.05.07.01"')
    return 0


def cmd_verify(pubkey_b64: str | None) -> int:
    sig_path = MANIFEST_ROOT / "manifest.sig"
    if not sig_path.exists():
        print(f"ERROR: no signature at {sig_path}", file=sys.stderr)
        return 2

    sig = base64.b64decode(sig_path.read_text(encoding="utf-8").strip())
    payload = _payload_bytes()

    if pubkey_b64 is None:
        try:
            sys.path.insert(0, str(REPO_ROOT / "src"))
            from cognithor.system._pinned_keys import HARDWARE_MANIFEST_TARGETS_KEY
        except ImportError as exc:
            print(
                f"ERROR: --pubkey not given and pinned-keys module not loadable: {exc}",
                file=sys.stderr,
            )
            return 2
        if HARDWARE_MANIFEST_TARGETS_KEY is None:
            print(
                "ERROR: --pubkey not given and HARDWARE_MANIFEST_TARGETS_KEY is None.\n"
                "       Patch src/cognithor/system/_pinned_keys.py first, "
                "or pass --pubkey <base64>.",
                file=sys.stderr,
            )
            return 2
        if not HARDWARE_MANIFEST_TARGETS_KEY.startswith("ed25519:"):
            print("ERROR: pinned key not in 'ed25519:base64' format", file=sys.stderr)
            return 2
        pubkey_b64 = HARDWARE_MANIFEST_TARGETS_KEY.split(":", 1)[1]

    pub = Ed25519PublicKey.from_public_bytes(base64.b64decode(pubkey_b64))
    try:
        pub.verify(sig, payload)
    except Exception as exc:
        print(f"[FAIL] Signature INVALID: {exc}", file=sys.stderr)
        return 1
    print("[OK] Signature verified.")
    return 0


def cmd_genkey(out: Path) -> int:
    """Mint a fresh Ed25519 keypair. Owner-action only — store offline."""
    priv = Ed25519PrivateKey.generate()
    pem = priv.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    out.write_bytes(pem)
    out.chmod(0o600)

    pub_raw = priv.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    pub_b64 = base64.b64encode(pub_raw).decode("ascii")
    print(f"[OK] Private key written: {out}  (chmod 0600)")
    print()
    print("KEEP THIS PRIVATE KEY OFFLINE. Never commit, never email, never paste.")
    print()
    print("Public key (paste into src/cognithor/_pinned_keys.py):")
    print(f'    HARDWARE_MANIFEST_TARGETS_KEY = "ed25519:{pub_b64}"')
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sign_manifest",
        description="Owner-side TUF-Light signing for the Hardware-Aware Manifest.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sign = sub.add_parser("sign", help="Sign manifest/v2/")
    p_sign.add_argument("--key", required=True, type=Path, help="Ed25519 private key (PEM)")

    p_verify = sub.add_parser("verify", help="Verify existing manifest.sig")
    p_verify.add_argument(
        "--pubkey",
        default=None,
        help="Base64-Public-Key (default: read from cognithor/_pinned_keys.py)",
    )

    p_gen = sub.add_parser("genkey", help="Mint fresh Ed25519 keypair (offline use)")
    p_gen.add_argument("--out", required=True, type=Path, help="Output path for private key (PEM)")

    args = parser.parse_args(argv)
    if args.cmd == "sign":
        return cmd_sign(args.key)
    if args.cmd == "verify":
        return cmd_verify(args.pubkey)
    if args.cmd == "genkey":
        return cmd_genkey(args.out)
    return 2


if __name__ == "__main__":
    sys.exit(main())
