# Registry Signing — Operator Toolkit (PACK-4)

Spec: [`docs/superpowers/specs/2026-05-05-pack4-registry-signing.md`](../../docs/superpowers/specs/2026-05-05-pack4-registry-signing.md).

These four scripts implement the offline + online operator workflow for the community registry's TUF-Light signing scheme.

| Script | Where it runs | Purpose |
|---|---|---|
| `generate_root_key.py` | Offline machine, **once** at marketplace launch | Mint the long-lived Root keypair. Public key gets pinned in source. |
| `generate_targets_key.py` | Offline machine, every rotation | Mint a fresh Targets keypair. Public key goes into a new `root.json`. |
| `sign_root.py` | Offline machine, every rotation | Sign `root.json` with the Root key, delegating to a Targets key. |
| `sign_payload.py` | GitHub Actions (online) | Sign `registry.json` / `recalls/active.json` / `publishers/*.json` with the Targets key. |

## One-time launch flow (marketplace activation)

```bash
# All on an offline machine.
python scripts/registry_signing/generate_root_key.py    --out-dir ./keys/root
python scripts/registry_signing/generate_targets_key.py --out-dir ./keys/targets

# Sign the initial root.json delegating to the brand-new Targets key.
python scripts/registry_signing/sign_root.py \
    --root-key ./keys/root/root_private.pem \
    --targets-pubkey ./keys/targets/targets_public.b64 \
    --version 1 \
    --valid-days 365 \
    --min-client-version 0.97.0 \
    --out ./registry/root.json

# Move the keys offline. Then:
#   1. Edit src/cognithor/skills/community/_pinned_keys.py:
#        ROOT_PUBLIC_KEY_B64 = "<contents of keys/root/root_public.b64>"
#   2. Cut a Cognithor release including that change.
#   3. Push registry/root.json to the registry repo.
#   4. Set GitHub-Actions secret REGISTRY_TARGETS_PRIVATE_KEY = contents of
#      keys/targets/targets_private.pem (in the registry repo).
#   5. Move keys/targets/targets_private.pem to encrypted offline backup;
#      its only live copy now lives in the GitHub-Actions secret.
#   6. Move keys/root/* to a hardware token / encrypted air-gapped drive.
```

## Routine sign — `registry.json` / `recalls/active.json` / `publishers/*.json`

Run this from a GitHub Actions workflow on every push to the registry repo:

```yaml
- name: Sign registry payloads
  env:
    REGISTRY_TARGETS_PRIVATE_KEY: ${{ secrets.REGISTRY_TARGETS_PRIVATE_KEY }}
  run: |
    echo "$REGISTRY_TARGETS_PRIVATE_KEY" > /tmp/targets.pem
    python scripts/registry_signing/sign_payload.py \
        --in unsigned/registry.json \
        --key /tmp/targets.pem \
        --out signed/registry.json
    python scripts/registry_signing/sign_payload.py \
        --in unsigned/recalls/active.json \
        --key /tmp/targets.pem \
        --out signed/recalls/active.json
    rm /tmp/targets.pem
```

Each unsigned input must include `_type`, `version`, `issued_at`, `valid_until`, plus the type-specific body. `sign_payload.py` validates these before signing.

## Targets-key rotation (every 90 days, or after suspected compromise)

```bash
# Offline machine.
python scripts/registry_signing/generate_targets_key.py --out-dir ./keys/targets-v2

# Sign root.json v2 delegating to the new Targets key.
# The version MUST be > the current root.json version.
python scripts/registry_signing/sign_root.py \
    --root-key ./keys/root/root_private.pem \
    --targets-pubkey ./keys/targets-v2/targets_public.b64 \
    --version 2 \
    --valid-days 365 \
    --min-client-version 0.97.0 \
    --out ./registry/root.json
```

Then:

1. Push the new `root.json` to the registry repo.
2. Update the GitHub-Actions secret `REGISTRY_TARGETS_PRIVATE_KEY` to the new private key.
3. Wait for clients to pick up the new `root.json` (next sync).
4. Securely destroy the old Targets private key from GitHub-Actions and any local copies.

## Root-key compromise (emergency)

There is no `code` recovery path — this is by design. The Root key lives on a hardware token / air-gapped drive precisely so this column has no online recovery option.

The only response is:

1. Generate a new Root keypair on a clean offline machine.
2. Cut a new Cognithor release with `_pinned_keys.py` updated.
3. Sign a fresh `root.json` with the new Root key (and a new Targets key).
4. Burn the old Root key. Old clients without the new release will continue trusting the old Root indefinitely — they need to update.

The corresponding owner runbook is in `docs/runbooks/registry_key_rotation.md`.

## NEVER

- Commit `*_private.pem` to **any** git repo.
- Run `generate_root_key.py` on a network-connected machine.
- Set `REGISTRY_TARGETS_PRIVATE_KEY` to anything other than the freshly minted Targets PEM.
- Add a CLI flag that disables signature verification on the client. The build-time `REQUIRE_SIGNED_REGISTRY` constant in `_pinned_keys.py` is intentionally not togglable from the CLI — that would be a downgrade vector.
