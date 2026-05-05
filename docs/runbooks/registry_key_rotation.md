# Runbook — Community Registry Key Rotation

Audience: Cognithor maintainer (currently a single owner).
Last reviewed: 2026-05-05 — initial PACK-4 ship.

This runbook covers the two key-rotation procedures defined in [PACK-4 spec §2](../superpowers/specs/2026-05-05-pack4-registry-signing.md).

| Trigger | Procedure |
|---|---|
| Routine Targets-key rotation (every 90 days, recommended) | [§A](#a-routine-targets-rotation) |
| Suspected Targets-key compromise | [§A](#a-routine-targets-rotation) — same flow, immediate |
| Suspected Root-key compromise | [§B](#b-emergency-root-rotation) — release-bound, no online recovery |

The operator-side tooling lives in [`scripts/registry_signing/`](../../scripts/registry_signing/README.md).

---

## A — Routine Targets rotation

**Goal:** invalidate the current Targets key without taking the registry offline. Clients pick up the new key transparently on their next sync.

### Preconditions
- Air-gapped or offline machine with the Root private key on a hardware token / encrypted USB.
- A clean checkout of the Cognithor source tree on that machine (for the scripts).
- Access to the registry repo (push) and to GitHub-Actions secrets (write).

### Steps

1. **On the offline machine**, mint a fresh Targets keypair:
   ```bash
   python scripts/registry_signing/generate_targets_key.py --out-dir ./keys/targets-v$(date +%Y%m%d)
   ```

2. **Sign a new `root.json`** with the existing Root key, delegating to the new Targets pubkey. The version number MUST be exactly one higher than the current `root.json.signed.version`:
   ```bash
   python scripts/registry_signing/sign_root.py \
       --root-key /media/hardware-token/root_private.pem \
       --targets-pubkey ./keys/targets-vNEW/targets_public.b64 \
       --version <CURRENT + 1> \
       --valid-days 365 \
       --min-client-version 0.97.0 \
       --out ./root.json
   ```

3. **Move the signed `root.json`** to a network-connected machine via the same medium that holds the Root key (USB → registry-host).

4. **Push** to the registry repo:
   ```bash
   cd <registry-repo>
   cp /path/to/root.json root.json
   git add root.json && git commit -m "rotate: targets v$(date +%Y%m%d) → root.json v<N+1>" && git push
   ```

5. **Update the GitHub-Actions secret** `REGISTRY_TARGETS_PRIVATE_KEY` to the contents of `./keys/targets-vNEW/targets_private.pem`. Use `gh secret set`:
   ```bash
   gh secret set REGISTRY_TARGETS_PRIVATE_KEY < ./keys/targets-vNEW/targets_private.pem
   ```

6. **Wait for clients to pick up** the new `root.json`. The default check_interval is 1 hour, so within 1h the entire client base will have cached the new Targets pubkey via `RegistryVerifier.verify_root`.

7. **Burn the old Targets key**:
   - Securely delete the old `targets_private.pem` from the offline machine (`shred -u` on Linux, `Remove-Item -Force` + `Reset-FileTime` on Windows).
   - The previous GitHub-Actions secret is already overwritten in step 5 — no further action needed.

8. **Run a smoke check**: trigger a small change in the registry repo (e.g. bump a no-op field in `registry.json`), let CI sign and push, then on a developer machine run:
   ```bash
   python -c "from cognithor.skills.community.signing import RegistryVerifier; v = RegistryVerifier(); print(v.is_configured())"
   ```
   Should print `True`. Force-resync via the API or by deleting `~/.cognithor/community_registry_state.json` and re-running `RegistrySync.sync_once`.

### Rollback
If the new Targets key is faulty (e.g. PEM corrupted in transit), the old `root.json` with the old Targets key is still in git history. Revert the registry repo's `root.json` to the prior commit. **Do not** bump the version — that would create a replay-vulnerable window. Restore the old GitHub-Actions secret.

---

## B — Emergency Root rotation

**Goal:** invalidate a leaked Root key.

> **There is no online recovery path.** The Root key is offline by design precisely so that this is a release-bound, deliberate, all-hands operation rather than a cron job. Plan for ~2h of focused work plus a release cycle.

### Steps

1. **Stop signing** with the leaked Root key immediately. Lock the offline storage medium.

2. **On a clean offline machine** (NOT the one that may be compromised), mint a new Root keypair:
   ```bash
   python scripts/registry_signing/generate_root_key.py --out-dir ./keys/root-vNEW
   ```

3. **Mint a new Targets keypair** at the same time (the leaked Root could have been used to delegate fake Targets keys; rotate both):
   ```bash
   python scripts/registry_signing/generate_targets_key.py --out-dir ./keys/targets-vNEW
   ```

4. **Sign a fresh `root.json`** with the new Root key. Set `version: 1` because it's the first signing under the new authority. Older clients will not know about this Root, that is expected.
   ```bash
   python scripts/registry_signing/sign_root.py \
       --root-key ./keys/root-vNEW/root_private.pem \
       --targets-pubkey ./keys/targets-vNEW/targets_public.b64 \
       --version 1 \
       --valid-days 365 \
       --min-client-version 0.<X>.0 \
       --out ./root.json
   ```

5. **Patch source**:
   ```python
   # src/cognithor/skills/community/_pinned_keys.py
   ROOT_PUBLIC_KEY_B64 = "<contents of ./keys/root-vNEW/root_public.b64>"
   ```

6. **Cut a Cognithor release** with that change. Bump `pyproject.toml` patch version. Tag, build, push to PyPI.

7. **Push the new `root.json`** to the registry repo.

8. **Communicate** to operators: old clients will silently keep trusting the leaked Root until they upgrade. Send a security advisory naming the affected versions and the upgrade path. Consider yanking the affected PyPI versions if the leak is recent.

9. **Burn the old Root key**: physically destroy the hardware token / drive. There is no software-level revocation; this is a physical-control story.

### What old (un-upgraded) clients see
- They keep trusting the old (now-leaked) Root indefinitely.
- They will continue accepting payloads signed by the old Targets key. **This is the cost of offline-Root design.** The cost is bounded: clients that don't upgrade also don't get new recalls / new skills / new publisher data, so attack value erodes over time.
- Mitigation: the registry can publish a "honeypot" `root.json` v2 signed by the OLD Root that includes `min_client_version: 0.<X>.0` (the version where the new Root lands). Old clients then refuse the registry until upgraded.

### Why we accept this gap
A code-level recovery story would require either (a) embedding multiple Root keys (TUF-full) or (b) a key-revocation channel that itself needs auth. Both add roughly 10× the complexity for a Solo project where the Root key already lives on a hardware token. The release-bound recovery is acceptable IF the Root key is genuinely offline. **Verify offline-ness annually.**

---

## Calendar reminders

| Cadence | Action |
|---|---|
| Every 90 days | Routine Targets rotation (§A) |
| Every 365 days | `root.json` valid_until is hitting expiry — re-sign with the same Root + Targets, version+1. |
| Annually | Verify the Root private key is still on the documented offline storage; verify the offline machine's OS is patched. |

---

## Related

- Full spec: [`docs/superpowers/specs/2026-05-05-pack4-registry-signing.md`](../superpowers/specs/2026-05-05-pack4-registry-signing.md)
- Operator scripts: [`scripts/registry_signing/`](../../scripts/registry_signing/README.md)
- SECURITY policy: [`SECURITY.md`](../../SECURITY.md) — Registry Trust Model section
- Trust-model overview: [`docs/operational_trust.md`](../operational_trust.md)
