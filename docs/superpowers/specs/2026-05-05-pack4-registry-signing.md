# PACK-4 — Community Registry Signing (TUF-Light)

**Status**: Spec ratified 2026-05-05. Implementation = this PR.
**Audit-finding**: PACK-4 (REAL-CRIT, deep-audit pass-2, 2026-05-05).
**Affects**: `src/cognithor/skills/community/sync.py`, `publisher.py`.
**Depends on**: `cryptography>=46.0.5` (already a dep — Ed25519 via `cryptography.hazmat.primitives.asymmetric.ed25519`).

---

## 1. Threat model

The community-skill registry is hosted at `https://raw.githubusercontent.com/Alex8791-cyber/skill-registry/main/...` and ships three JSON resources that drive client-side actions:

| Path | Drives |
|---|---|
| `registry.json` | Skill catalogue / discovery |
| `recalls/active.json` | **Remote kill-switch** — names listed here trigger `_deactivate_skill` + `SkillRegistry.disable` on every client |
| `publishers/{username}.json` | Reputation score, VERIFIED status |

**Trust gap addressed:** plain HTTPS protects the bytes on the wire but says nothing about the operator. An adversary that controls the registry host (GitHub repo compromise, BGP-MITM on `raw.githubusercontent.com`, DNS hijack) can:

- Trigger arbitrary `_deactivate_skill` calls — denial-of-service on legitimate skills
- Elevate untrusted publishers to VERIFIED — supply-chain takeover
- **Replay** — serve an old, legitimately signed `registry.json` to **freeze a recall**, neutralising the kill-switch the moment it's needed most
- **Key compromise** — leaked Targets-key alone could sign updates that rotate the key

**Non-goals (for v1):**

- Federation / multiple operators — single root authority for now.
- Snapshot/timestamp roles à la full-fat TUF. The `valid_until` window in every signed payload replaces the timestamp role pragmatically.
- Online key rotation. Rotation requires the offline Root key.

---

## 2. Architecture: TUF-Light, two roles

```
       ┌────────────────────────┐
       │  Root (offline)        │       Hardware token / air-gapped machine.
       │  Ed25519               │       Signs ONLY root.json. Used for
       │  signs root.json only  │       Targets-key rotation.
       └─────────┬──────────────┘
                 │
                 │ delegates to
                 ▼
       ┌────────────────────────┐
       │  Targets (online)      │       Stored as a GitHub-Actions secret.
       │  Ed25519               │       Signs registry.json, recalls/*,
       │  signs all data        │       publishers/*. Rotated whenever
       └─────────┬──────────────┘       compromise is suspected.
                 │
                 │ verified by
                 ▼
       ┌────────────────────────┐
       │  Cognithor client      │
       │  ROOT_PUBLIC_KEY pinned│       Pinned in source. Loads root.json,
       │  in source             │       checks signature against pinned key,
       └────────────────────────┘       extracts current Targets pubkey.
```

**Compromise recovery:**

| Compromise | Effect | Recovery |
|---|---|---|
| Targets key leaks | Adversary can sign forged `registry.json` etc. | Owner generates new Targets key offline, signs a new `root.json` with Root key, deploys it. Clients pick it up via version monotony. |
| Root key leaks | Game over (theoretically). | Mitigation: Root key lives offline on hardware token. Single physical artifact, never on a server. The value of moving Root offline is exactly that this column has no `code` recovery — it requires a new release with a new pinned key. |

---

## 3. Wire format

### 3.1 Common envelope

Every signed payload uses the same envelope:

```json
{
  "signed": {
    "_type": "root" | "registry" | "recalls" | "publisher",
    "version": 17,
    "issued_at": "2026-05-05T09:00:00Z",
    "valid_until": "2026-05-19T09:00:00Z",
    "<type-specific fields>": "..."
  },
  "signatures": [
    {
      "keyid": "sha256:abc123...",
      "method": "ed25519",
      "sig": "<base64-encoded 64-byte signature>"
    }
  ]
}
```

**Canonicalisation rule:** signatures are computed over `json.dumps(signed, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")`. The verifier must use the **same** canonicalisation — never re-serialise the parsed dict in a different order. (Future-proofing: switching to canonical-JSON-RFC8785 is a one-line change.)

### 3.2 `root.json` — Targets-key delegation

```json
{
  "signed": {
    "_type": "root",
    "version": 1,
    "issued_at": "2026-05-05T09:00:00Z",
    "valid_until": "2027-05-05T09:00:00Z",
    "targets": {
      "keyid": "sha256:9f4a...",
      "method": "ed25519",
      "public_key": "<base64-encoded 32-byte raw Ed25519 public key>"
    },
    "min_client_version": "0.97.0"
  },
  "signatures": [
    {"keyid": "sha256:root1...", "method": "ed25519", "sig": "<base64>"}
  ]
}
```

`min_client_version` lets the operator cut off ancient clients during a forced migration (e.g. if a future spec change is incompatible).

### 3.3 `registry.json`

```json
{
  "signed": {
    "_type": "registry",
    "version": 42,
    "issued_at": "2026-05-12T09:00:00Z",
    "valid_until": "2026-05-26T09:00:00Z",
    "skills": [
      {"name": "sql-helper", "version": "1.2.0", "publisher": "alice", "...": "..."}
    ]
  },
  "signatures": [
    {"keyid": "sha256:9f4a...", "method": "ed25519", "sig": "<base64>"}
  ]
}
```

### 3.4 `recalls/active.json`

```json
{
  "signed": {
    "_type": "recalls",
    "version": 8,
    "issued_at": "2026-05-12T09:00:00Z",
    "valid_until": "2026-05-13T09:00:00Z",
    "recalls": [
      {"skill_name": "evil-skill", "reason": "credential exfil", "recalled_at": "..."}
    ]
  },
  "signatures": [...]
}
```

The shorter `valid_until` (1 day vs. 14 days for `registry.json`) means stale recall data is trusted for at most 24h before the client refuses to act. Keeps the kill-switch responsive without needing a separate timestamp role.

### 3.5 `publishers/{username}.json`

```json
{
  "signed": {
    "_type": "publisher",
    "version": 3,
    "issued_at": "2026-05-12T09:00:00Z",
    "valid_until": "2026-06-26T09:00:00Z",
    "github_username": "alice",
    "reputation_score": 87.5,
    "...": "..."
  },
  "signatures": [...]
}
```

Note: per-publisher `version` is independent — verified against per-publisher last-seen.

---

## 4. Client-side state

```
~/.cognithor/community_registry_state.json
{
  "schema": 1,
  "last_seen": {
    "root":             {"version": 1,  "verified_at": "..."},
    "registry":         {"version": 42, "verified_at": "..."},
    "recalls":          {"version": 8,  "verified_at": "..."},
    "publisher:alice":  {"version": 3,  "verified_at": "..."}
  },
  "cached_targets_key": "<base64 from latest verified root.json>"
}
```

Persisted atomically via `*.tmp` + `os.replace`. File mode `0o600` on POSIX.

---

## 5. Verifier algorithm

```python
def verify_signed_payload(
    body: bytes,                      # the raw JSON bytes off the wire
    expected_type: str,                # "registry", "recalls", "publisher", "root"
    public_key: Ed25519PublicKey,     # for non-root: the cached targets key; for root: the pinned key
    last_seen_version: int,            # from state
    now: datetime,
    *,
    expected_keyid: str | None = None,  # bind signature to a specific keyid
) -> SignedPayload:
    """Returns the validated `signed` dict, or raises.

    Steps (in order — each must pass before the next):
      1. Parse JSON. Reject on parse error.
      2. Schema check: top-level keys must be exactly {"signed", "signatures"}.
      3. signed._type must equal expected_type.
      4. Re-canonicalise signed: json.dumps(signed, sort_keys=True, separators=(",", ":"), ensure_ascii=False).
      5. Find a signature where method == "ed25519" and (expected_keyid is None or keyid == expected_keyid).
      6. Verify signature against canonicalised bytes via public_key.verify().
      7. Reject if signed.version < last_seen_version (REPLAY).
      8. Reject if datetime.fromisoformat(signed.valid_until) < now (STALE).
      9. Reject if datetime.fromisoformat(signed.issued_at) > now + 5min (CLOCK-SKEW guard).
     10. Return signed.
    """
```

The first six steps reject all forgeries. Steps 7-9 reject replays and stale data. The 5-minute clock-skew window is one-sided: future-dated payloads are slightly forgiven (network propagation), but stale payloads are not.

**Hard-fail behaviour:** every failure raises a `RegistrySignatureError` subclass. **No soft-fall-through path exists.** The caller in `RegistrySync._fetch_json` does not catch these — they propagate to `sync_once_inner`'s outer `except Exception`, which logs `registry_sync_failed` and returns `SyncResult(success=False)`. Recalls do not fire on a sync that failed verification.

---

## 6. Module layout

```
src/cognithor/skills/community/
├── _pinned_keys.py        # NEW. Pinned Root pubkey + REQUIRE_SIGNED_REGISTRY flag.
├── signing.py             # NEW. RegistryVerifier, SignedPayload, exceptions.
├── sync.py                # MODIFIED. Wires verifier into _fetch_json.
└── publisher.py           # MODIFIED. Wires verifier into _fetch_publisher_profile.
```

### 6.1 `_pinned_keys.py`

```python
"""Pinned Root public key for the community registry.

Embedded at build time. To rotate the Root key, ship a new Cognithor release
with this file updated. The Targets key (rotated more frequently) is NOT
pinned — it lives in root.json, signed by the Root key.

When ROOT_PUBLIC_KEY_B64 is None, the registry is treated as "not yet
operational": community-skill features hard-fail with a clear error. This
is the default until the operator generates a Root key offline and embeds
it here.
"""

# Base64-encoded 32-byte raw Ed25519 public key, OR None if the registry
# has not yet been deployed. Operators replace this string when they ship
# a release that activates the marketplace. See
# scripts/registry_signing/generate_root_key.py for the generation script.
ROOT_PUBLIC_KEY_B64: str | None = None

# Build-time flag. ALWAYS True in shipped releases. Source-patchable for
# Cognithor developers running an unsigned local-mirror; never CLI-toggleable.
REQUIRE_SIGNED_REGISTRY: bool = True
```

### 6.2 `signing.py` — public API

```python
class SignedPayload(BaseModel, frozen=True):
    type_: str         # alias of _type
    version: int
    issued_at: datetime
    valid_until: datetime
    body: dict[str, Any]    # the rest of the signed dict, type-specific


class RegistrySignatureError(Exception): ...
class RegistryReplayError(RegistrySignatureError): ...
class RegistryStaleError(RegistrySignatureError): ...
class RegistryKeyError(RegistrySignatureError): ...
class RegistryNotConfiguredError(RegistrySignatureError):
    """Raised when ROOT_PUBLIC_KEY_B64 is None and signed-registry is required."""


class RegistryVerifier:
    """Stateful verifier for a single registry channel.

    Threading: instances are NOT thread-safe. Each RegistrySync /
    PublisherVerifier instance owns one RegistryVerifier.
    """

    def __init__(self, *, state_path: Path | None = None) -> None: ...

    def verify_root(self, body: bytes) -> SignedPayload:
        """Verify root.json against pinned Root pubkey + last-seen root version.
        Updates cached_targets_key on success."""

    def verify_targets_payload(
        self, body: bytes, *, expected_type: str, channel_key: str
    ) -> SignedPayload:
        """Verify a registry/recalls/publisher payload against the cached
        Targets key + per-channel last-seen version. ``channel_key`` is the
        state-file key (e.g. "registry", "recalls", "publisher:alice").

        Calls verify_root once per process if cached_targets_key is missing
        (lazy bootstrap via http callback supplied by the caller)."""

    def attach_root_loader(self, loader: Callable[[], Awaitable[bytes]]) -> None:
        """Caller supplies a function that returns root.json bytes. The
        verifier calls it lazily when the Targets key isn't cached yet."""
```

### 6.3 Wiring example — `sync.py`

```python
# Before:
registry_data = await self._fetch_json(f"{self._registry_url}/registry.json")

# After:
verifier = RegistryVerifier(...)
verifier.attach_root_loader(lambda: self._fetch_text(f"{self._registry_url}/root.json"))

raw = await self._fetch_text(f"{self._registry_url}/registry.json")
payload = await verifier.verify_targets_payload_async(
    raw.encode("utf-8"),
    expected_type="registry",
    channel_key="registry",
)
registry_data = payload.body  # parsed, verified, version-checked
```

The verifier never touches HTTP itself — it asks for bytes via a callback. This keeps the test surface tiny: tests construct payloads in-memory.

---

## 7. Migration plan (no existing un-signed corpus)

The community marketplace **has not yet shipped publicly** (per `MEMORY.md`: "Q4 2026 community creator marketplace"). This means:

- No existing un-signed registry to re-sign — clean cut-over.
- No backward-compatibility code path needed in the verifier.
- v0.97.0 lands the verifier with `ROOT_PUBLIC_KEY_B64 = None` → community features hard-disabled.
- v0.97.x (or first marketplace launch) lands the operator-generated key.
- Until then, `RegistrySync` and `PublisherVerifier` are dormant: any call returns the `RegistryNotConfiguredError` cleanly.

---

## 8. Owner-side tooling

`scripts/registry_signing/` (operator-only):

```
generate_root_key.py     # one-off: emits root_public.b64 + root_private.pem
generate_targets_key.py  # rotation: emits targets_public.b64 + targets_private.pem
sign_payload.py          # CI-callable: --in body.json --key targets.pem --type registry
sign_root.py             # offline-only: --targets-pubkey ... --root-key root.pem --out root.json
```

`scripts/registry_signing/README.md` documents the runbook (also linked from `docs/runbooks/registry_key_rotation.md`).

---

## 9. Test matrix

`tests/test_skills/test_community/test_signing.py` covers:

| # | Scenario | Expected |
|---|---|---|
| 1 | Legitimate signed `registry.json` | Returns parsed `SignedPayload` |
| 2 | `version < last_seen_version` (replay) | Raises `RegistryReplayError` |
| 3 | `valid_until < now` (stale) | Raises `RegistryStaleError` |
| 4 | Tampered `signed` dict (bytes flipped) | Raises `RegistrySignatureError` |
| 5 | Tampered `signatures.sig` (bytes flipped) | Raises `RegistrySignatureError` |
| 6 | `ROOT_PUBLIC_KEY_B64 = None` + signed-required | Raises `RegistryNotConfiguredError` |
| 7 | Wrong `_type` (registry payload validated as recalls) | Raises `RegistrySignatureError` |
| 8 | Future-dated `issued_at` > now + 5min | Raises `RegistrySignatureError` |
| 9 | Targets-key rotation via new `root.json` (version 1→2) | Subsequent payloads verify against new key |
| 10 | `min_client_version` exceeds running version | Raises `RegistryKeyError` |
| 11 | Persisted `last_seen.version` survives verifier restart | New verifier instance reads same state |
| 12 | Race: two parallel `verify_targets_payload` calls | Both succeed, state file consistent (atomic write) |

Targets coverage: 100% on `signing.py`. Existing `sync.py` / `publisher.py` tests adjusted to inject a stub verifier.

---

## 10. Documentation deltas

This PR also ships:

| Doc | Change |
|---|---|
| `docs/operational_trust.md` | PACK-4 row: "known gap → addressed in v0.97.0" |
| `SECURITY.md` | New section "Registry Trust Model" with Threat-Model table |
| `README.md` | Trust-Modell paragraph mentions Ed25519 + TUF-Light + EU-sovereign positioning |
| `docs/runbooks/registry_key_rotation.md` | Operator runbook: routine Targets rotation + emergency Root rotation |
| `scripts/registry_signing/README.md` | CLI reference for signing scripts |
| `MEMORY.md` | PACK-4 pointer flipped to "addressed" |

---

## 11. Acceptance criteria

- [ ] All 12 test scenarios pass.
- [ ] `mypy --strict src/cognithor/skills/community/signing.py` clean.
- [ ] `mypy --strict src/cognithor/skills/community/sync.py` clean (after wiring).
- [ ] `mypy --strict src/cognithor/skills/community/publisher.py` clean (after wiring).
- [ ] `ruff check` + `ruff format --check` clean across all touched files.
- [ ] CI green.
- [ ] No new runtime dependency (cryptography is already pinned).
- [ ] All four signing scripts execute end-to-end on a clean checkout (smoke test in `tests/test_skills/test_community/test_signing_scripts.py`).
- [ ] `docs/operational_trust.md`, `SECURITY.md`, `README.md`, runbook and `MEMORY.md` all updated in the same PR.

---

## 12. Out of scope (explicit non-goals)

- Sigstore/cosign keyless signing. Considered, deferred — Sigstore-infra dep is heavier than the static-key approach for a Solo project, and the EU-sovereign positioning prefers self-managed keys over a third-party witness.
- Full TUF (snapshot + timestamp roles). Pragmatic `valid_until` covers the freeze attack at 1/100th the complexity.
- Federation / multi-operator trust. Single Root is enough for v1.
- Signed plugin/pack manifests. Different attack surface (already partly addressed by `eula_sha256`); separate sprint.

---

*End of spec.*
