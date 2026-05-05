# Operational Trust — TRUST-1..10 reference

Cognithor's operational-trust stack answers a single reviewer
question:

> *If something goes wrong, can an operator reconstruct exactly
> what the agent knew, what it decided, which tool it called, why
> it was allowed, what changed, and how to roll it back?*

The answer is yes. This document maps the question's clauses to
the production code, ledger schemas, and CLI / REST surfaces.

The stack is split across ten layers. TRUST-1..4 ship the
recovery-path basics (receipts, decision explanations,
failure-mode taxonomy, pack rollback). TRUST-5..10 ship the
forensic-detail layer (permission scopes, cost ledger, fingerprint
ledger, cloud-escalation log, memory provenance, schema-migration
chain).

---

## TRUST-1 — Run receipts

**Question answered:** *what did the agent do during run X?*

* Module: [`cognithor.audit.AuditLogger.run_receipt`](../src/cognithor/audit/__init__.py)
* Schema version: `RECEIPT_SCHEMA_VERSION = 1`
* Optional flag: `include_trust=True` folds the TRUST-5..10 bundle
  under the receipt's `"trust"` key (#403).
* Optional flag: `signing_key=` produces an HMAC-SHA-256 signature
  over the canonical (signature-less) form. Tamper detection
  works across audit entries AND the trust block.

CLI surface ([`cognithor.cli.receipt_cmd`](../src/cognithor/cli/receipt_cmd.py)):

| Subcommand | Purpose |
|---|---|
| `cognithor receipt show <session_id>` | dump one receipt to stdout / file |
| `cognithor receipt verify <bundle.json> --key K` | check the HMAC signature |
| `cognithor receipt list [--log-dir DIR]` | enumerate session_ids in audit log |
| `cognithor receipt export-all --log-dir DIR --out OUT` | bulk-export every session |
| `cognithor receipt diff <a> <b>` | surface trust-ledger deltas |

REST: `GET /api/crew/trace/{trace_id}/receipt[?include_trust=true]`
([`cognithor.api.crew_traces`](../src/cognithor/api/crew_traces.py),
owner-gated).

---

## TRUST-2 — Structured decision explanations

**Question answered:** *why was this tool call allowed or blocked?*

Every Gatekeeper block path emits a `DecisionExplanation` carrying
`rule_id` + `rule_source` + `matched_pattern`. The Trace-UI
renders them without parsing free-text reasons.

| Block path | rule_id |
|---|---|
| destructive shell command (AST) | `dest_cmd_ast` |
| destructive shell command (regex) | `dest_cmd_regex` |
| dangerous Python (AST) | `dangerous_python_ast` |
| dangerous Python (regex) | `dangerous_python_regex` |
| path unparseable | `path_unparseable` |
| path outside allowed | `path_outside_allowed` |
| credential mask | `credential_scan` |
| OFFLINE network block | `operation_mode_offline_network` |
| permission scope | `scope:<axis>:<identity>` |
| disabled-tool config | `tool_disabled_by_config` |
| capability matrix | `capability_matrix` |
| explicit policy match | `policy:<rule_name>` |

`matched_pattern` is truncated to 200 chars across the board so
no single block can bloat the audit chain.

---

## TRUST-3 — Failure-mode taxonomy + classifier

**Question answered:** *what kind of thing went wrong?*

Module: [`cognithor.models.FailureMode`](../src/cognithor/models.py)
+ [`cognithor.audit.AuditLogger.classify_failure`](../src/cognithor/audit/__init__.py).

The taxonomy is a closed StrEnum. Adding a value is a deliberate
review point.

| Group | Values |
|---|---|
| Plan-layer | `PLAN_PARSE_FAILED`, `PLAN_LLM_ERROR`, `PLAN_TIMEOUT` |
| Gatekeeper | `GATEKEEPER_BLOCK`, `GATEKEEPER_APPROVAL_DENIED` |
| Tool / Executor | `TOOL_TIMEOUT`, `TOOL_NOT_FOUND`, `TOOL_INVALID_PARAMS`, `TOOL_INTERNAL_ERROR`, `SANDBOX_REFUSED` |
| Environmental | `NETWORK_ERROR`, `AUTH_ERROR`, `QUOTA_EXCEEDED` |
| LLM | `LLM_HALLUCINATION`, `LLM_REFUSAL` |
| Operational-trust (TRUST-5..10) | `PERMISSION_SCOPE_DENIED`, `BUDGET_EXCEEDED`, `FINGERPRINT_DRIFT`, `CLOUD_ESCALATION_REJECTED`, `PROVENANCE_EXPIRED`, `MIGRATION_CHAIN_ERROR` |

Aggregator: `AuditLogger.failures_by_mode(hours=24)` returns a
sorted descending count dict. Empty dict means "no failures in
the window".

---

## TRUST-4 — Pack rollback

**Question answered:** *how do I undo a bad pack install?*

CLI: `cognithor pack rollback <qualified_id> [--to-version V]`.
Backups live under `<packs_dir>/.backups/`. Tested end-to-end in
`tests/test_packs/test_rollback_cli.py`.

---

## TRUST-5 — Permission scopes

**Question answered:** *who is allowed to call which tool from
which channel?*

Module: [`cognithor.security.permission_scope`](../src/cognithor/security/permission_scope.py).

* `ScopeAxis` (CHANNEL / USER / WORKFLOW / PACK) StrEnum.
* `PermissionScope` frozen dataclass: `tool_allowlist`,
  `tool_denylist`, `max_risk` ceiling.
* `ScopeRegistry.evaluate(scope_keys, tool, risk)` returns the
  most-restrictive verdict (denylist beats allowlist beats
  max_risk). `from_config()` loads from YAML.
* Production wiring: Gatekeeper Step 0.5 in `evaluate()` (between
  OperationMode and Credential scan).

---

## TRUST-6 — Cost ledger

**Question answered:** *how much have I spent today, broken down
by tool / domain / channel / run?*

Module: [`cognithor.security.cost_ledger`](../src/cognithor/security/cost_ledger.py).

* Integer micro-USD canonical cost unit (no float drift).
* `CostKind` StrEnum (LLM_INFERENCE / EMBEDDING / TOOL_API /
  STORAGE / NETWORK / OTHER).
* Multi-axis aggregation: `summarise()` returns `by_kind` /
  `by_tool` / `by_backend` / `by_channel` / `by_domain` / `by_run`.
* `BudgetReport` (UNDER / APPROACHING / EXCEEDED) is passive — the
  ledger doesn't enforce, it reports.
* `top_n("tool"|"backend"|...)` for "biggest spenders" tile.

---

## TRUST-7 — Tool / model / pack fingerprints

**Question answered:** *which exact code or weights produced this
result?*

Module: [`cognithor.security.fingerprint`](../src/cognithor/security/fingerprint.py).

* `BinaryKind` StrEnum: TOOL / MODEL / PACK / SCHEMA / BINARY.
* `ToolFingerprint` keyed by SHA-256 content hash (lowercase 64
  hex enforced).
* `FingerprintLedger` dual index: by-hash + by-name. `register()`
  is idempotent on hash; `divergent_names()` surfaces names with
  > 1 hash (the smoking gun).
* `hash_python_source()` normalises CRLF→LF so Windows + POSIX
  produce the same hash for the same Python file.

Capture sites:
| Kind | Capture site |
|---|---|
| `TOOL` | `JarvisMCPServer.register_tool` |
| `PACK` | `PackLoader._load_one` |
| `MODEL` | `OllamaBackend.fingerprint_model` (auto-wired on first chat) |
| `SCHEMA` | `PackLoader._fingerprint_pack_manifest_schema` |
| `BINARY` | (deferred — Ollama/vLLM server binaries) |

---

## TRUST-8 — Cloud-escalation log

**Question answered:** *did this query leave the machine?*

Module: [`cognithor.security.cloud_escalation`](../src/cognithor/security/cloud_escalation.py).

* `EscalationReason` StrEnum (8 values, all carry privacy
  implications: LOCAL_BACKEND_DOWN / CONTEXT_TOO_LARGE /
  MODEL_NOT_AVAILABLE_LOCALLY / OWNER_OVERRIDE / RATE_LIMITED_LOCAL
  / COST_BUDGET_DECISION / QUALITY_THRESHOLD / UNKNOWN).
* `EscalationEvent` frozen dataclass — **metadata only**, no
  prompt or response content. `from_backend` / `to_backend`,
  token counts, cost in micro-USD, owner_consented flag.
* `EscalationLedger.summarise()` works with full ledger or
  by_run / by_destination / by_reason / in_window subsets.
* Cross-wiring with TRUST-6: `record_escalation_with_cost()` in
  `cognithor.security.trust_wiring` records to BOTH ledgers on
  the same `run_id` so totals reconcile.

---

## TRUST-9 — Memory provenance

**Question answered:** *where did this remembered fact come from
and when does it stop being true?*

Module: [`cognithor.memory.provenance`](../src/cognithor/memory/provenance.py).

* `SourceType` StrEnum (9 closed values: CHAT_UTTERANCE /
  TOOL_OUTPUT / AGENT_INFERENCE / CONFIG_IMPORT /
  PACK_REGISTRATION / SCHEDULED_INGEST / MIGRATION /
  USER_DIRECTIVE / UNKNOWN).
* `ExpiryPolicy` StrEnum (PERMANENT / TTL / REPLACE_ON_NEW /
  MANUAL).
* `ProvenanceLedger` is append-only per `item_id`. Re-tagging
  appends to the chain rather than overwriting.
* `expired(now=...)` returns IDs whose head tag is past TTL;
  `superseded(item_id)` returns the prefix replaced by the head.

Memory-tier capture sites:
| Tier | Site |
|---|---|
| Semantic entities | `SemanticMemory.add_entity` |
| Knowledge-graph relations | `SemanticMemory.add_relation` |
| Episodic logs | `EpisodicMemory.append_entry` |
| Vault secrets | `AgentVault.store` |
| Session store | `IsolatedSessionStore.create_session` |
| Procedural / skills | `ProceduralMemory.save_procedure` |
| Core memory (Tier-1) | `CoreMemory.save` |
| Knowledge ingestion | `KnowledgeIngestService.ingest_file` |
| Indexed chunks | `MemoryIndex.upsert_chunk` |

All sites use the same opt-in contract: `provenance_source_type`
+ `provenance_source_id` keyword args required to tag, partial
args silently skip, unknown values coerce to `UNKNOWN`. **Tier
operations NEVER fail because of provenance tagging.**

---

## TRUST-10 — Migration chain

**Question answered:** *which schema versions are active and how
did we get here?*

Module: [`cognithor.security.migration_ledger`](../src/cognithor/security/migration_ledger.py).

* `MigrationDomain` StrEnum: per-persistence-layer chain (memory
  tiers, audit log, pack manifest, config schema, all six
  TRUST-5..10 ledgers).
* `MigrationStatus`: PENDING / APPLIED / FAILED / ROLLED_BACK.
  Only APPLIED + ROLLED_BACK move the head version.
* Per-domain chain integrity: `record()` raises
  `MigrationChainError` if `source_version` doesn't match the
  current head.
* `rollback_of` is only legal on a step with status=ROLLED_BACK
  AND the referenced step exists, is APPLIED, and is in the same
  domain.

Self-audit + backfill coverage:
| Domain | PR |
|---|---|
| `AUDIT_LOG` | #416 |
| `PACK_MANIFEST` | #417 |
| `PROVENANCE_LEDGER` | #432 |
| `FINGERPRINT_LEDGER` | #432 |
| `COST_LEDGER` | #434 |
| `ESCALATION_LEDGER` | #434 |
| `SCOPE_REGISTRY` | #434 |

---

## Trust bundle composition

[`cognithor.security.trust_bundle`](../src/cognithor/security/trust_bundle.py)
ties everything together. `build_trust_bundle(run_id)` returns:

```python
{
  "schema_version": 1,
  "run_id": "<run_id>",
  "permission_scopes": [...],     # full snapshot (TRUST-5)
  "cost": {"summary": ..., "entries": [...]},  # run-scoped (TRUST-6)
  "fingerprints": {"all": [...], "divergent_names": [...]},  # full snapshot (TRUST-7)
  "escalations": {"summary": ..., "entries": [...]},  # run-scoped (TRUST-8)
  "provenance": {<item_id>: [<tag>, ...]},  # full snapshot (TRUST-9)
  "migrations": {"head_version": {...}, "steps": [...]},  # full snapshot (TRUST-10)
}
```

Run-scoped sections (cost, escalations) filter by `run_id`. Full
snapshots (the rest) describe **the world the agent ran in**.

The composer is dependency-injectable via `TrustLedgers`; tests
construct fresh ledgers for isolation, production passes `None`
to use the canonical singletons.

---

## Cross-references

* Reviewer-feedback gap analysis: `memory/project_operational_trust_gap_analysis.md`
* TRUST-5..10 ship log: `memory/project_2026_05_04_trust_stack_complete.md`
* PR span: #395 → #435 (41 PRs across two days)

---

## PACK-4 — Community registry signing (addressed)

**Status (2026-05-05):** REAL-CRIT addressed. The community-skill
registry is now Ed25519-signed under a TUF-Light scheme (offline
Root key + rotating online Targets key), with monotonic-version
replay protection and `valid_until` freshness windows.

* Spec: [`docs/superpowers/specs/2026-05-05-pack4-registry-signing.md`](superpowers/specs/2026-05-05-pack4-registry-signing.md)
* Verifier module: [`cognithor.skills.community.signing`](../src/cognithor/skills/community/signing.py)
* Operator runbook: [`docs/runbooks/registry_key_rotation.md`](runbooks/registry_key_rotation.md)
* Trust-model summary: [`SECURITY.md`](../SECURITY.md#registry-trust-model-pack-4)

Deferred (not blockers): full TUF snapshot/timestamp roles,
Sigstore/cosign keyless signing, multi-operator federation. See
spec §12 for non-goals.
