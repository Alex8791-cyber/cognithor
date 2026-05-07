# Manifest-Signing Runbook (Hardware-Aware Runtime)

> **Audience:** Repo-Owner. **Frequency:** Once for key-mint, once per
> manifest update afterwards. **Time:** ~5 min for sign, ~15 min for the
> initial mint + first roll-out.
> **Last reviewed:** 2026-05-07.

This document is the operational equivalent of
`docs/runbooks/registry_key_rotation.md` (PACK-4) for the
**Hardware-Aware Manifest** at `manifest/v2/`.

---

## 1. What is signed?

The signed payload is:

```
tiers.yaml + b"\n--MANIFEST-DELIM--\n" + models.yaml + b"\n--PRICING-SHA256:" + sha256(pricing.yaml)
```

Pricing is included by hash so a future Pricing-Operator-Role can rotate
prices without re-signing tiers/models. v2 still uses one key for all.

---

## 2. Phase 1: Initial Key Mint (one-time, offline)

### 2.1 Prepare an offline machine

- Air-gapped (or at minimum: no network during key-gen).
- Encrypted disk.
- USB-key for backup.

### 2.2 Mint the targets-keypair

```bash
# On the offline machine, in a freshly-cloned cognithor repo:
python scripts/sign_manifest.py genkey --out /secure/manifest_targets.key.pem
```

Output:
```
[OK] Private key written: /secure/manifest_targets.key.pem  (chmod 0600)

Public key (paste into src/cognithor/_pinned_keys.py):
    HARDWARE_MANIFEST_TARGETS_KEY = "ed25519:<base64>"
```

### 2.3 Pin the public key

Edit `src/cognithor/_pinned_keys.py`:

```python
# Existing PACK-4 root
PACK_REGISTRY_ROOT_KEY = "ed25519:..."

# NEW — hardware-manifest targets key (siehe docs/runbooks/manifest-signing.md)
HARDWARE_MANIFEST_TARGETS_KEY = "ed25519:<the base64 from genkey>"
```

### 2.4 Backup the private key

- 1× on encrypted USB → labeled, in a safe.
- 1× on a paper-printed QR-code → in a different safe.
- **NEVER** in the repo, NEVER in cloud-storage, NEVER in a password-manager
  that is itself cloud-synced.

### 2.5 Sign the current manifest

Still on the offline machine:

```bash
python scripts/sign_manifest.py sign --key /secure/manifest_targets.key.pem
```

This writes `manifest/v2/manifest.sig`.

### 2.6 Commit + push the public-pin and signature together

```bash
git add manifest/v2/manifest.sig src/cognithor/_pinned_keys.py
git commit -m "manifest: initial signing for v2026.05.07.01"
git push origin main
```

---

## 3. Phase 2: Routine Manifest Updates

### 3.1 Edit the YAML files

Most updates touch only `manifest/v2/models.yaml` (new model entries) or
`manifest/v2/tiers.yaml` (new tier definitions). Pricing updates go to
`manifest/v2/pricing.yaml`.

### 3.2 Bump the manifest version

In **all three files**, update:

```yaml
manifest_version: "2026.05.07.01"  # Datum + Index
```

Use today's date + a 2-digit index that increments on multiple updates
per day.

### 3.3 Re-sign

```bash
python scripts/sign_manifest.py sign --key /secure/manifest_targets.key.pem
```

### 3.4 Verify locally before commit

```bash
python scripts/sign_manifest.py verify
```

Expected: `[OK] Signature verified.`

### 3.5 Commit + push

```bash
git add manifest/v2/*.yaml manifest/v2/manifest.sig
git commit -m "manifest: <description of update>"
git push origin main
```

After the push, all running cognithor installations will pick up the new
manifest on next `cognithor doctor --refresh-manifest` (or after the
30-day auto-refresh fires).

---

## 4. Recall a Compromised Manifest

If a malicious actor obtains the targets-key OR if a manifest contains a
broken/dangerous tier:

### 4.1 Add to active recall list

Edit `manifest/recalls/active.json`:

```json
{
  "schema_version": 2,
  "recalls": [
    {
      "manifest_version": "2026.05.07.01",
      "reason": "Compromised targets-key — see GHSA-XXX",
      "recalled_at_utc": "2026-05-08T10:00:00Z",
      "severity": "critical"
    }
  ]
}
```

### 4.2 If the targets-key is compromised: rotate

1. Mint a new keypair (§2.2).
2. Update `src/cognithor/_pinned_keys.py` with the new `HARDWARE_MANIFEST_TARGETS_KEY`.
3. Sign the latest manifest with the new key.
4. Push as a **patch release** of cognithor (so end-users get the new
   pinned key via PyPI).

### 4.3 Communicate via SECURITY.md + GHSA

Recall events that involve security implications get a private GHSA
write-up + public SECURITY.md update.

---

## 5. Verifying client-side

End-user diagnostic to confirm signature was verified:

```bash
cognithor doctor
# Look for: "Manifest: 2026.05.07.01 (origin=cache, signed=True)"
```

If `signed=False` after a manifest update was supposed to be signed,
something is wrong (signature missing, wrong payload-format, key-rotation
not yet shipped to the user).

---

## 6. Verification protocol contract

The **payload format MUST stay stable** so that older clients can verify
newer manifests during the rollout window. Any change to the payload
construction in `scripts/sign_manifest.py` MUST also land in
`src/cognithor/system/manifest_loader.py` `_verify_signature` AND be
shipped in the same cognithor release. Mismatches across versions are a
hard fail (signature-invalid).

---

## 7. References

- Sign-tool: `scripts/sign_manifest.py`
- Verify-side: `src/cognithor/system/manifest_loader.py:_verify_signature`
- PACK-4 sibling runbook: `docs/runbooks/registry_key_rotation.md`
- Spec: `docs/superpowers/specs/2026-05-07-hardware-aware-runtime-spec-v2.md`
