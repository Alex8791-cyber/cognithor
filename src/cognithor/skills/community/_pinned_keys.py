"""Pinned Root public key for the community registry.

PACK-4 (deep-audit pass-2, 2026-05-05). See
``docs/superpowers/specs/2026-05-05-pack4-registry-signing.md``.

The Root key signs only ``root.json``, and ``root.json`` carries the
delegated Targets key that signs all other registry payloads. Rotating
the Targets key only requires the offline Root key — never a Cognithor
release. Rotating the Root key requires a new release (this file
patched).

When ``ROOT_PUBLIC_KEY_B64`` is ``None`` the registry is treated as
"not yet operational": the verifier hard-fails with
:class:`RegistryNotConfiguredError`. This is the default until the
operator generates the Root keypair offline (see
``scripts/registry_signing/generate_root_key.py``) and embeds the
public key here in a release.
"""

from __future__ import annotations

# Base64-encoded 32-byte raw Ed25519 public key (no PEM wrapping). ``None``
# means the marketplace is dormant — community-skill features hard-disable
# at the verifier boundary. Replace with the actual key string as part of
# the release that activates the marketplace.
ROOT_PUBLIC_KEY_B64: str | None = None

# Build-time guard. ALWAYS ``True`` in shipped releases. Source-patchable
# for Cognithor developers running an unsigned local mirror, but never
# togglable from the CLI — that would be a downgrade vector. Test code
# sets this via :func:`unittest.mock.patch` on this exact module
# attribute, NOT via env var.
REQUIRE_SIGNED_REGISTRY: bool = True

# Earliest Cognithor version that supports this signing format. Updated
# only when a backwards-incompatible change to the wire format ships;
# clients with a lower version see :class:`RegistryKeyError` from the
# verifier when the registry's ``min_client_version`` exceeds this.
SIGNING_FORMAT_VERSION: str = "1.0.0"
