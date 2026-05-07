"""Pinned public keys for the Hardware-Aware Runtime manifest.

The hardware-aware tier/model manifest at ``manifest/v2/`` is signed
TUF-Light style. The signing payload + verification logic live in
:mod:`cognithor.system.manifest_loader._verify_signature` and
:mod:`scripts.sign_manifest`.

When ``HARDWARE_MANIFEST_TARGETS_KEY`` is ``None`` the manifest is
treated as **unsigned** — the loader logs `manifest_unsigned` and the
``signed`` field in :class:`ManifestSource` reports ``False``. The
runtime still functions (the embedded fallback ships with the wheel
and is trusted via the wheel's own signature), but the **online
refresh** path no longer has end-to-end integrity until the operator
mints a key per ``docs/runbooks/manifest-signing.md`` and patches the
base64 string here.

Key rotation:
- A compromised targets-key is rotated by minting a fresh keypair,
  patching this file, and shipping a new Cognithor release.
- The rotation event is also recorded in
  ``manifest/recalls/active.json`` so older manifests signed with the
  compromised key hard-fail at load time.
"""

from __future__ import annotations

# Base64-encoded 32-byte raw Ed25519 public key (no PEM wrapping).
# Format must be: "ed25519:<base64>" — the prefix is enforced by
# :func:`cognithor.system.manifest_loader.ManifestLoader._verify_signature`.
#
# Set to ``None`` until the operator has minted a keypair offline using
# ``python scripts/sign_manifest.py genkey``.
#
# Minted 2026-05-08 (private-key on external drive at
# F:\cognithor-keys\manifest_targets.key.pem). Rotation per
# docs/runbooks/manifest-signing.md.
HARDWARE_MANIFEST_TARGETS_KEY: str | None = "ed25519:3Ujk4fziqpXusVKVzp1yoPAr0RMXze+pAytYuPclgnA="
