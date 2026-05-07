# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.97.0+ | Yes (current: 0.99.0) — full TRUST-1..10 stack, hash-chained audit, TUF-Light registry, AuditCategory.REFLECTION |
| 0.85.0–0.96.x | Best-effort security patches; functional, but missing operational-trust ledgers and registry signatures |
| 0.78.2–0.84.x | Security patches on request only |
| 0.71–0.78.1 | Upgrade recommended (GHSA-cognithor-001) |
| < 0.71  | No        |

## Reporting a Vulnerability

If you discover a security vulnerability in Cognithor, please report it responsibly:

1. **Do NOT open a public issue.** Security vulnerabilities must be reported privately.
2. **Email:** Send a detailed report to the repository owner via GitHub's private vulnerability reporting feature (Security tab → "Report a vulnerability").
3. **Include:**
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

We aim to acknowledge reports within 48 hours and provide a fix within 7 days for critical issues.

## Security Architecture

Cognithor implements defense-in-depth with multiple security layers (supporting Ollama and LM Studio as local backends):

- **Gatekeeper** — Deterministic policy engine (no LLM). Every tool call is validated against security policies with 4 risk levels: GREEN (auto-approve) → YELLOW (inform) → ORANGE (require approval) → RED (block). Decisions emit a structured `rule_id` + `rule_source` + `matched_pattern` explanation (TRUST-2) for the receipt and the IDE Decision-Hover.
- **Sandbox** — Multi-level execution isolation: Process-level → Linux Namespaces (nsjail / bubblewrap) → Docker containers → Windows Job Objects.
- **Audit Trail** — Append-only JSONL log with **HMAC-SHA-256 hash chain** (`prev_hash` over canonical NFC-normalized JSON, SEC-HIGH-5 mitigation v0.97.0). Verify with `cognithor audit verify`. Reflector writes route through a dedicated `AuditCategory.REFLECTION` channel (Compliance-Spring v0.98.0); nine event types cover causal sequences, weight snapshots, episodic appends, semantic facts, and procedure auto-creation. Property-based Hypothesis tests + a nightly burn-in CI workflow (`.github/workflows/nightly-burn-in.yml`) keep the chain intact. Credentials are masked before logging.
- **Operational Trust (TRUST-1..10)** — Six append-only ledgers (Provenance, Permission-Scopes, Tool-Fingerprints, Cloud-Escalation, Cost in micro-USD, Migration) plus signed run-receipts (`cognithor receipt show / verify / list / export-all / diff`) and a 15-value `FailureMode` taxonomy. REST surface: `GET /api/crew/trace/{trace_id}/receipt`. See [`docs/operational_trust.md`](docs/operational_trust.md).
- **Resilient Workflow Engine (CRWE, v0.99.0)** — `cognithor task <manifest>` runs declarative batch workflows with JSONL streaming, atomic `.checkpoint.json` writes, file-locking (POSIX `fcntl.flock` / Windows `msvcrt.locking`), SIGINT/SIGTERM emergency-checkpoint between tasks, manifest-tamper detection on `--resume`, and audit-chain integration (`workflow_resumed` + `system_checkpoint_created` events). Crash-recovery uses `results.jsonl` line count as source-of-truth. Closes the gap-injection attack vector for offline batch operators.
- **Credential Vault** — Fernet-encrypted (AES-256) per-agent secret storage. Keys never appear in logs or API responses.
- **AST-Based Code Analysis** — Python `ast.NodeVisitor` guard detects dangerous imports, subprocess calls, eval, exec at the syntax tree level. Shell commands analyzed via `bashlex` parser with regex fallback. Replaces regex-based guards (v0.90.0+).
- **Input Sanitization** — Protection against shell injection, path traversal, and prompt injection attacks (incl. SEC-HIGH-3 indirect-prompt-injection guard against web content asserting trustworthiness).
- **Path Sandbox** — File operations restricted to explicitly allowed directories.
- **Red-Teaming + CB-wide bug sweeps** — Automated offensive security test suite (1,425 LOC). Pass-3 (PR #479) and Pass-4 (PRs #460–#469) closed 16 + 10 verified findings (CRITs / HIGHs / MEDs across CB) in two coordinated sweeps in May 2026.
- **Registry Trust Model** — Community-skill registry payloads (`registry.json`, `recalls/active.json`, `publishers/*.json`) are Ed25519-signed under a TUF-Light scheme: an offline Root key signs `root.json`, which delegates to a rotating online Targets key. See [Registry Trust Model](#registry-trust-model-pack-4) below.

## Registry Trust Model (PACK-4)

Cognithor's community-skill marketplace uses a self-managed TUF-Light signing scheme — no third-party witness, no Sigstore dependency, EU-sovereign by design. Spec: [`docs/superpowers/specs/2026-05-05-pack4-registry-signing.md`](docs/superpowers/specs/2026-05-05-pack4-registry-signing.md).

### Threat coverage

| Attack | Mitigation |
|---|---|
| Tampered registry JSON in transit (BGP-MITM, DNS hijack) | Ed25519 signature over canonical-JSON `signed` block. |
| Compromised CDN / GitHub-Pages serving the registry | Same — signed payloads are verified client-side regardless of origin. |
| Replay of an old, legit-signed `registry.json` to neutralise a recall | Monotonic `version` field; client persists `last_seen` per channel and refuses anything older. |
| Stale `recalls/active.json` served indefinitely after key rotation | `valid_until` field (1 day for recalls, 14 days for registry). Hard-fail when expired. |
| Targets-key compromise | Offline Root key signs a new `root.json` with a fresh Targets pubkey. Clients pick up the rotation transparently on next sync (key change is part of the signed-data version chain). |
| Root-key compromise | Release-bound rotation: new pinned key in source, new Cognithor release. **By design** — offline-Root is the trust anchor. See [`docs/runbooks/registry_key_rotation.md`](docs/runbooks/registry_key_rotation.md). |
| Confused-deputy: swap `publishers/alice.json` with `publishers/eve.json` | Verifier requires `payload.body.github_username` to match the requested user. |
| Downgrade: `--accept-unsigned-registry` flag | Does not exist. `REQUIRE_SIGNED_REGISTRY` is a build-time constant in `_pinned_keys.py`, source-patchable for developers but not togglable from the CLI. |

### Hard-fail behaviour

Every signature/freshness/replay failure raises `RegistrySignatureError` from `cognithor.skills.community.signing`. The surrounding `RegistrySync.sync_once` lets the exception propagate, which marks the sync as `success=False` and prevents recall application. **Soft-fail on a kill-switch mechanism would be a contradiction** — the entire point is that recalls reach clients reliably.

### Dormant marketplace (default)

Until the operator activates the marketplace by minting Root keys offline and embedding the Root pubkey in `_pinned_keys.py`, `RegistryVerifier.is_configured()` returns `False`. `RegistrySync.sync_once` short-circuits cleanly and `PublisherVerifier._fetch_publisher_profile` returns `None`. No network traffic, no errors.

## Runtime Token Protection (v0.26.0+)

All channel tokens (Telegram, Discord, Slack, Teams, WhatsApp, API, WebUI, Matrix, Mattermost) are encrypted in memory using ephemeral Fernet keys (AES-256). Tokens are never stored as plaintext in RAM after initialization.

- **Encryption**: `SecureTokenStore` generates a random Fernet key at startup. All tokens are encrypted immediately upon channel construction.
- **Access**: Tokens are decrypted on-demand via `@property` accessors. External callers see plaintext — internal storage is always ciphertext.
- **Fallback**: Without the `cryptography` package, Base64 obfuscation is used with a logged warning.
- **Scope**: Runtime protection against memory dumps. Does not replace disk-level encryption for config files.

## TLS Support (v0.26.0+)

Webhook servers (Teams, WhatsApp) and HTTP servers (API, WebUI) support optional TLS:

- Configure `ssl_certfile` and `ssl_keyfile` in `security` section of `config.yaml`
- Minimum TLS 1.2 enforced (`ssl.TLSVersion.TLSv1_2`)
- Non-localhost servers without TLS log a `WARNING` at startup

## File-Size Limits (v0.26.0+)

All upload and processing paths enforce size limits to prevent resource exhaustion:

| Path | Limit | Constant |
|------|-------|----------|
| Document extraction (`media.py`) | 50 MB | `MAX_EXTRACT_FILE_SIZE` |
| Audio transcription (`media.py`) | 100 MB | `MAX_AUDIO_FILE_SIZE` |
| Code execution (`code_tools.py`) | 1 MB | `MAX_CODE_SIZE` |
| WebUI file upload (`webui.py`) | 50 MB | `MAX_UPLOAD_SIZE` |
| Telegram document download (`telegram.py`) | 50 MB | `MAX_DOCUMENT_SIZE` |

## Credential Handling

- API keys in configuration are masked (`***`) in all API responses by default.
- The `.env` file (`~/.cognithor/.env`) is excluded from version control via `.gitignore`.
- The Control Center API never writes masked placeholder values (`***`) back to configuration files.

## Past Advisories

### GHSA-cognithor-001 — Unauthenticated Master Token Disclosure (CRITICAL)

- **CVSS**: 9.8 (AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H)
- **Affected**: <= 0.78.1
- **Fixed in**: 0.78.2
- **CWE**: CWE-306 (Missing Authentication for Critical Function), CWE-200 (Exposure of Sensitive Information)
- **Description**: The `/api/v1/bootstrap` endpoint returned the master bearer token without authentication. Combined with the default `0.0.0.0` bind, any network-reachable host could steal the token and access all protected API endpoints.
- **Fix**: Bootstrap endpoint restricted to loopback addresses only; default API bind changed from `0.0.0.0` to `127.0.0.1`.
- **Reported by**: [Offgrid Security](https://www.offgridsec.com/) — responsible disclosure

## Acknowledgments

We thank the following researchers for responsibly disclosing security issues:

- **[Offgrid Security](https://www.offgridsec.com/)** — GHSA-cognithor-001 (April 2026)

## Dependencies

We regularly review dependencies for known vulnerabilities. If you find a vulnerable dependency, please report it using the process above.
