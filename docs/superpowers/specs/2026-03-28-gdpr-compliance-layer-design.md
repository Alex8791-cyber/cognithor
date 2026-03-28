# GDPR Compliance Layer — Design Spec

> **For agentic workers:** Use superpowers:writing-plans to create the implementation plan from this spec.

**Goal:** Make Cognithor GDPR-compliant without breaking existing functionality. Additive changes only — nothing gets removed.

**Date:** 2026-03-28
**Status:** Approved

---

## 1. Consent Manager

**File:** `src/jarvis/security/consent.py`
**DB:** `~/.jarvis/index/consent.db` (SQLCipher encrypted)

### Schema
```sql
CREATE TABLE consent (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    channel TEXT NOT NULL,        -- telegram, webui, cli, discord, etc.
    consent_type TEXT NOT NULL,   -- data_processing, cloud_llm, osint
    status TEXT NOT NULL,         -- pending, accepted, rejected, withdrawn
    privacy_version TEXT,         -- version of privacy notice shown
    granted_at TEXT,
    withdrawn_at TEXT,
    ip_or_context TEXT,           -- e.g. telegram chat_id, session_id
    created_at TEXT NOT NULL
);
CREATE INDEX idx_consent_user ON consent(user_id, channel, consent_type);
```

### ConsentManager API
```python
class ConsentManager:
    def has_consent(self, user_id: str, channel: str, consent_type: str = "data_processing") -> bool
    def grant_consent(self, user_id: str, channel: str, consent_type: str, context: str = "") -> None
    def withdraw_consent(self, user_id: str, channel: str, consent_type: str = "data_processing") -> None
    def get_user_consents(self, user_id: str) -> list[dict]
    def requires_consent(self, user_id: str, channel: str) -> bool
```

### Channel Integration
- **Telegram** (`channels/telegram.py`): Before processing any message, check `consent_manager.has_consent(user_id, "telegram")`. If not, send privacy notice with InlineKeyboard buttons "Akzeptieren" / "Ablehnen". Store result. If rejected, respond with "Ich kann deine Nachrichten ohne Datenschutz-Einwilligung nicht verarbeiten."
- **WebUI** (`channels/webui.py`): WS message type `consent_check` on connection. Frontend shows banner. Backend waits for `consent_response` before processing.
- **CLI** (`__main__.py`): On first run (no consent in DB for cli channel), show privacy notice text and prompt "Einverstanden? (ja/nein)".
- **Other channels**: Same pattern — check before first message processing.

### Privacy Notice
- Stored in `data/legal/privacy_notice_de.md` and `data/legal/privacy_notice_en.md`
- Versioned (e.g. "1.0") — if version changes, re-consent required
- Content: What data is collected, how it's used, retention periods, user rights, contact info

---

## 2. Data Erasure (Right to be Forgotten)

**File:** Extend `src/jarvis/security/gdpr.py` ErasureManager

### Full Erasure Method
```python
async def erase_all(self, user_id: str, mcp_client: Any = None) -> dict:
    """Delete ALL personal data for user_id across every tier.

    Returns dict with counts per tier.
    """
```

### Tiers to erase (in order):
1. **Processing Logs** — `DataProcessingLog.delete_user_records(user_id)` (exists)
2. **Model Usage Logs** — `ModelUsageLog.delete_user_records(user_id)` (exists)
3. **Consent Records** — `ConsentManager.delete_user(user_id)`
4. **Chat Sessions** — Delete from sessions.db where user matches
5. **Episodic Memory** — Delete files matching user context
6. **Semantic Memory** — Delete indexed chunks by source_user
7. **Entities/Relations** — Delete entities where attributes contain user_id
8. **Vault Notes** — Delete notes tagged with user_id or in user folder
9. **HIM Reports** — Delete reports where target matches
10. **Audit Logs** — Anonymize (replace user_id with hash)

### REST API
- `DELETE /api/v1/user/data` — requires authentication, triggers `erase_all()`
- Returns: `{"erased": {"memories": 12, "entities": 5, "vault_notes": 3, ...}}`

### MCP Tool
- `erase_user_data(user_id, justification)` — RED classification in gatekeeper

---

## 3. Missing Delete Tools

### vault_delete
**File:** `src/jarvis/mcp/vault.py` — add method + register handler

```python
async def vault_delete(self, path: str) -> str:
    """Delete a vault note by path. Returns confirmation or error."""
```

- Validates path (no traversal)
- Deletes the .md file
- Removes from search index
- Gatekeeper: RED

### delete_entity / delete_relation
**File:** `src/jarvis/mcp/memory_server.py` — add methods + register handlers

```python
async def delete_entity(self, name: str) -> str:
    """Delete an entity and all its relations by name."""

async def delete_relation(self, source: str, target: str, relation: str = "") -> str:
    """Delete a specific relation between two entities."""
```

- Gatekeeper: RED for both

---

## 4. TTL Enforcement

### Cron Job
Register in gateway during cron engine setup:
- Job name: `retention_enforcer`
- Schedule: `0 3 * * *` (daily at 03:00)
- Handler: calls `RetentionEnforcer.enforce()` + HIM cleanup + episodic cleanup

### Default Retention Periods
Add to config with defaults:
```python
class RetentionConfig(BaseModel):
    episodic_days: int = 90
    processing_log_days: int = 90
    model_usage_log_days: int = 180
    him_report_days: int = 30
    session_days: int = 180
    vault_osint_days: int = 30
```

### HIM Report Cleanup
- Vault notes in `recherchen/osint/` older than `him_report_days` get deleted
- Uses `vault_delete()` internally

### Episodic Memory Cleanup
- Set `episodic_retention_days` default to 90 in MemoryManager config
- `prune_old()` already exists, just needs the default set

---

## 5. SQLCipher Encryption at Rest

### New Dependency
Add to `pyproject.toml`: `pysqlcipher3>=1.2.0` under `[security]` extra

### Encrypted Connection Wrapper
**File:** `src/jarvis/security/encrypted_db.py`

```python
def encrypted_connect(db_path: str, key: str | None = None) -> sqlite3.Connection:
    """Open a SQLCipher-encrypted SQLite database.

    If key is None, reads from JARVIS_DB_KEY env var.
    If env var is empty, auto-generates and stores in credential store.
    Falls back to unencrypted sqlite3 if pysqlcipher3 is not installed.
    """
```

### Migration Strategy
- On startup, check if each DB is encrypted (try opening with key)
- If unencrypted: create new encrypted DB, copy all data, replace original
- One-time migration, logged clearly
- Fallback: if `pysqlcipher3` not installed, warn and use unencrypted (graceful degradation)

### Affected Databases
- `~/.jarvis/index/memory_traces.db`
- `~/.jarvis/index/memory_proposals.db`
- `~/.jarvis/index/memory_governance.db`
- `~/.jarvis/index/memory_runs.db`
- `~/.jarvis/index/knowledge_claims.db`
- `~/.jarvis/index/consent.db` (new)
- `~/.jarvis/memory/sessions/sessions.db`
- `~/.jarvis/memory/knowledge_qa.db`
- `~/.jarvis/memory/knowledge_lineage.db`
- Evolution goal-scoped indexes

### Key Management
- Primary: `JARVIS_DB_KEY` environment variable
- Fallback: Auto-generated 32-byte key stored in credential store (`~/.jarvis/credentials/db_key.enc`)
- The credential store itself uses `JARVIS_CREDENTIAL_KEY` (already exists)

---

## 6. Cloud LLM Consent Flow

### Integration Point
**File:** `src/jarvis/core/model_router.py` — wrap the `_call_api()` method

Before any cloud API call:
1. Check `consent_manager.has_consent(user_id, "cloud_llm_" + provider)`
2. If not consented: raise `CloudConsentRequired(provider)`
3. Gateway catches this and prompts user: "Deine Anfrage wird an {provider} gesendet. Erlaubst du das? (ja/nein/immer)"
4. "immer" stores permanent consent for that provider

### Processing Log
Every cloud call logged:
```python
processing_log.record(
    user_id=user_id,
    category="llm_inference",
    purpose="query_processing",
    data_types=["user_query", "memory_context"],
    recipient=provider_name,
    legal_basis="consent",
)
```

### Config
```python
class CloudPrivacyConfig(BaseModel):
    consent_required: bool = True
    log_cloud_calls: bool = True
    allowed_providers: list[str] = Field(default_factory=list)  # empty = all with consent
```

---

## 7. Data Access / Export API

### REST Endpoint
`GET /api/v1/user/data?format=json` — authenticated

Returns structured JSON:
```json
{
    "user_id": "...",
    "exported_at": "...",
    "memories": [...],
    "entities": [...],
    "vault_notes": [...],
    "sessions": [...],
    "processing_log": [...],
    "consents": [...],
    "him_reports": [...]
}
```

### MCP Tool
`export_user_data(user_id)` — GREEN classification (read-only)

---

## 8. Legal Documents

### Privacy Notice (`data/legal/privacy_notice_de.md`)
German DSGVO-compliant notice covering:
- Identity of data controller (configurable)
- What data is collected and why
- Legal basis (consent)
- Retention periods
- User rights (access, erasure, portability, objection)
- Third-party recipients (cloud LLM providers)
- Contact information

### DPIA Template (`data/legal/dpia_template.md`)
Data Protection Impact Assessment template for:
- HIM/OSINT investigations
- Cloud LLM usage
- Memory/knowledge graph storage

---

## Integration Summary

| Existing File | Change |
|--------------|--------|
| `config.py` | Add RetentionConfig, CloudPrivacyConfig |
| `gatekeeper.py` | Add vault_delete, delete_entity, delete_relation, erase_user_data to RED |
| `gateway/gateway.py` | Wire ConsentManager, add retention cron |
| `channels/telegram.py` | Add consent check before message processing |
| `mcp/vault.py` | Add vault_delete method + handler |
| `mcp/memory_server.py` | Add delete_entity, delete_relation handlers |
| `security/gdpr.py` | Extend ErasureManager with erase_all() |
| `core/model_router.py` | Add cloud consent check before API calls |
| `config_routes.py` | Add /api/v1/user/data endpoints |

| New File | Purpose |
|----------|---------|
| `security/consent.py` | ConsentManager |
| `security/encrypted_db.py` | SQLCipher wrapper + migration |
| `data/legal/privacy_notice_de.md` | German privacy notice |
| `data/legal/privacy_notice_en.md` | English privacy notice |
| `data/legal/dpia_template.md` | DPIA template |

---

## 9. Legal Basis Tracking (Art. 6 — ChatGPT Review Gap A)

Not all processing can rely on consent. Each data processing activity must declare its legal basis.

### ProcessingContext
```python
class LegalBasis(str, Enum):
    CONSENT = "consent"                    # User explicitly agreed
    CONTRACT = "contract"                  # Necessary for service delivery
    LEGITIMATE_INTEREST = "legitimate_interest"  # Security, fraud detection
    LEGAL_OBLIGATION = "legal_obligation"  # Required by law
```

### Integration
- Every `DataProcessingLog.record()` call must include `legal_basis`
- Security monitoring / audit logs use `LEGITIMATE_INTEREST` (no consent needed)
- Chat processing uses `CONSENT`
- Erasure requests: only delete data based on CONSENT, preserve LEGITIMATE_INTEREST data (audit trail)

---

## 10. Purpose Limitation (Art. 5(1)(b) — ChatGPT Review Gap B)

Every stored data item must be tagged with its purpose.

### Purpose Tags
```python
class DataPurpose(str, Enum):
    CONVERSATION = "conversation"
    MEMORY = "memory"
    SECURITY = "security"
    ANALYTICS = "analytics"
    OSINT = "osint"
    EVOLUTION = "evolution"
```

### Implementation
- Memory entries: `purpose` field in metadata (default: "conversation")
- Vault notes: `purpose` tag in YAML frontmatter
- Entities: `purpose` in attributes dict
- Processing logs: already have `purpose` field — enforce non-empty

### Erasure Impact
- `erase_all()` only deletes data with purpose=conversation/memory/osint
- Data with purpose=security is anonymized (user_id replaced with hash), not deleted
- This prevents over-deletion of audit trails

---

## 11. Immutable Compliance Audit Log (Art. 5(2) — ChatGPT Review Gap C)

Append-only log for all compliance events. Cannot be deleted or modified.

**File:** `src/jarvis/security/compliance_audit.py`
**Storage:** `~/.jarvis/data/audit/compliance.jsonl` (append-only, no delete/truncate)

### Events logged:
- Consent granted/withdrawn (user_id, channel, timestamp)
- Erasure requested/executed (user_id, tiers affected, items deleted)
- Data export requested/delivered (user_id, format, size)
- Cloud data sent (provider, data types, legal basis)
- OSINT investigation started (target, justification, scope)

### Format
```json
{"ts": "2026-03-28T18:00:00Z", "event": "consent_granted", "user_id": "u123", "channel": "telegram", "consent_type": "data_processing", "policy_version": "1.0"}
{"ts": "2026-03-28T18:05:00Z", "event": "erasure_executed", "user_id": "u123", "tiers": ["memory", "vault"], "items_deleted": 15}
```

### Properties
- Append-only (open with mode="a")
- Never deleted by RetentionEnforcer
- Signed with SHA-256 chain (each line includes hash of previous line)
- Backed up to `~/.jarvis/data/audit/compliance.jsonl.bak` weekly

---

## 12. Erasure Authentication (ChatGPT Review Gap D)

The `DELETE /api/v1/user/data` endpoint must verify identity.

### Implementation
- Requires existing `_verify_cc_token` auth (same as all config routes)
- Additionally: `user_id` is extracted from the authenticated session, not from request body
- A user can only erase their OWN data
- Admin override: `JARVIS_ADMIN_TOKEN` env var can erase any user's data
- Re-authentication: for WebUI, require password/token re-entry before erasure

---

## 13. Data Processing Register (Art. 30 — ChatGPT Review Gap F)

Structured register of all processing activities.

**File:** `data/legal/processing_register.yaml`

```yaml
processing_activities:
  - name: chat_processing
    purpose: Respond to user queries
    legal_basis: consent
    data_categories: [user_queries, chat_history]
    recipients: [ollama_local]
    retention: 180 days
    erasure: full_delete

  - name: cloud_llm_inference
    purpose: Process complex queries via cloud AI
    legal_basis: consent
    data_categories: [user_queries, memory_context]
    recipients: [anthropic, openai, groq]
    retention: 90 days
    erasure: full_delete

  - name: osint_investigation
    purpose: Background research on persons/orgs
    legal_basis: consent + legitimate_interest
    data_categories: [public_profiles, claims, evidence]
    recipients: [github_api, arxiv_api, web_search]
    retention: 30 days
    erasure: full_delete

  - name: security_monitoring
    purpose: Detect abuse, credential leaks, injection
    legal_basis: legitimate_interest
    data_categories: [tool_calls, risk_assessments]
    recipients: [none]
    retention: 365 days
    erasure: anonymize

  - name: evolution_learning
    purpose: Autonomous knowledge building
    legal_basis: consent
    data_categories: [web_content, entities, claims]
    recipients: [searxng_local, brave_api, ddg]
    retention: none (permanent, domain knowledge)
    erasure: full_delete
```

---

## 14. Privacy Mode (ChatGPT "Next Level" Suggestion)

Runtime toggle that disables all persistent storage.

### Config
```python
class PrivacyConfig(BaseModel):
    privacy_mode: bool = False  # or JARVIS_PRIVACY_MODE env var
```

### When enabled:
- No memory storage (episodic, semantic, procedural all disabled)
- No vault writes
- No entity/relation creation
- No processing logs
- No session persistence
- Chat responses are stateless (no context from previous messages)
- Cloud LLM calls still require consent

### Implementation
- Check `config.privacy.privacy_mode` in gateway before each storage operation
- Existing tools still callable but storage calls become no-ops
- MCP tool handlers check privacy mode and skip persistence

---

## 15. Compliance Engine — Runtime Enforcement (Review Gap 1)

Central policy evaluation BEFORE every data processing operation. Not just logging — blocking.

**File:** `src/jarvis/security/compliance_engine.py`

```python
class ComplianceEngine:
    """Central GDPR policy enforcer. Called before every processing operation."""

    def __init__(self, consent_manager, config):
        self._consent = consent_manager
        self._config = config

    def check(self, user_id: str, channel: str, legal_basis: LegalBasis,
              purpose: DataPurpose, data_types: list[str]) -> None:
        """Raise ComplianceViolation if processing is not allowed."""
        # Rule 1: Consent-based processing requires actual consent
        if legal_basis == LegalBasis.CONSENT:
            if not self._consent.has_consent(user_id, channel):
                raise ComplianceViolation(
                    f"No consent for {purpose.value} on channel {channel}"
                )
        # Rule 2: Privacy mode blocks all persistent storage
        if self._config.privacy.privacy_mode and purpose != DataPurpose.SECURITY:
            raise ComplianceViolation("Privacy mode active — no persistent storage")
        # Rule 3: OSINT requires explicit OSINT consent
        if purpose == DataPurpose.OSINT:
            if not self._consent.has_consent(user_id, channel, "osint"):
                raise ComplianceViolation("OSINT investigation requires explicit consent")

class ComplianceViolation(Exception):
    pass
```

### Integration
- Gateway calls `compliance_engine.check()` before every `handle_message()`
- Memory tools check before `save_to_memory()`
- Vault tools check before `vault_save()`
- Cloud LLM router checks before API calls
- Gatekeeper can call compliance engine as additional gate

---

## 16. Active Data Minimization (Review Gap 2)

GDPR requires minimization at COLLECTION time, not just deletion after the fact.

### Input Filtering
Before storing ANY user message content:
1. Strip PII that is not needed for the response (emails, phone numbers, addresses in casual mentions)
2. Memory summarization: after 7 days, episodic memories are compressed to summaries (existing MemoryHygieneEngine can be extended)
3. Embedding-only mode: for semantic search, store only embeddings + source hash, not raw text

### Implementation
- Extend `MemoryHygieneEngine` with `minimize()` method
- Run `minimize()` in the retention_enforcer cron job
- Config: `data_minimization: bool = True` (default on)
- After 7 days: raw episodic text replaced with LLM-generated summary
- After 30 days: summaries reduced to key facts only

---

## 17. Complete Data Export (Review Gap 3)

Art. 15 export must include ALL data, including derived/computed data.

### Export Categories
The `/api/v1/user/data` export must include:
1. **Raw data**: messages, vault notes, memories
2. **Derived data**: entities, relations, knowledge graph connections
3. **Computed data**: trust scores, quality scores, claim verifications
4. **Embeddings metadata**: which texts were embedded, when, for what purpose (not raw vectors — those are not personally meaningful)
5. **Processing history**: every logged processing event
6. **Consent records**: all consent grants/withdrawals
7. **HIM reports**: any investigation reports involving this user

### Format
```json
{
    "export_version": "1.0",
    "user_id": "...",
    "exported_at": "...",
    "raw_data": { "messages": [...], "vault_notes": [...], "memories": [...] },
    "derived_data": { "entities": [...], "relations": [...], "claims": [...] },
    "computed_data": { "trust_scores": [...], "quality_scores": [...] },
    "embeddings_metadata": [{"text_hash": "...", "purpose": "...", "created_at": "..."}],
    "processing_history": [...],
    "consents": [...],
    "him_reports": [...]
}
```

---

## 18. Pseudonymized Audit Log (Review Gap 4)

The immutable compliance audit log must survive user erasure — but with pseudonymized user IDs.

### Implementation
- When `erase_all(user_id)` runs, the audit log is NOT deleted
- Instead, all occurrences of `user_id` in the audit log are replaced with `SHA-256(user_id + salt)`
- The salt is stored ONLY in the credential store
- This means: audit trail is preserved for accountability, but user cannot be re-identified without the salt
- If credential store is also wiped (full system reset), audit entries become permanently anonymous

### Pseudonymization Function
```python
def pseudonymize_user_id(user_id: str, salt: str) -> str:
    return hashlib.sha256(f"{user_id}:{salt}".encode()).hexdigest()[:16]
```

---

## 19. Dynamic Processing Register (Review Gap 5)

The `processing_register.yaml` must be auto-generated from actual system state, not manually maintained.

### Implementation
- On startup (or via API call), scan all registered MCP tools and processing activities
- Generate `processing_register.yaml` from:
  - Tool registry (what tools exist, what data they access)
  - Config (which cloud providers are enabled)
  - Consent records (which users have consented to what)
  - Retention config (what are the actual TTLs)
- REST endpoint: `GET /api/v1/compliance/register` returns current register as JSON
- Register is regenerated daily by the retention_enforcer cron job

---

## 20. DPIA Risk Scoring (Review Gap 6)

The DPIA must not just be a template — it needs automated risk classification.

### Risk Levels
```python
class DPIARiskLevel(str, Enum):
    LOW = "low"        # No PII, local only, standard retention
    MEDIUM = "medium"  # PII involved, but local processing, consent obtained
    HIGH = "high"      # PII sent to cloud, OSINT on persons, profiling
    CRITICAL = "critical"  # Sensitive data (health, political), cross-border transfer
```

### Automatic Classification
Each processing activity gets a risk score based on:
- Does it involve PII? (+1)
- Does it send data to cloud? (+2)
- Does it involve profiling/scoring persons? (+2)
- Does it process sensitive categories (Art. 9)? (+3)
- Is data transferred outside EU? (+2)
- Is retention > 180 days? (+1)

Score 0-1 = LOW, 2-3 = MEDIUM, 4-6 = HIGH, 7+ = CRITICAL

### Integration
- Risk level shown in processing register
- HIGH/CRITICAL activities logged in compliance audit
- CRITICAL activities require admin approval (similar to gatekeeper RED)

---

## Non-Breaking Guarantees

1. **Graceful degradation**: If `pysqlcipher3` not installed, falls back to unencrypted SQLite with WARNING
2. **Consent opt-out**: If `consent_required: false` in config, consent checks are skipped (for development/testing)
3. **Existing data preserved**: Migration copies data, does not delete originals until verified
4. **No API changes**: All existing MCP tools work exactly as before
5. **Performance**: SQLCipher adds ~10-15% overhead on DB operations; acceptable for background agent
6. **ComplianceEngine bypass**: If `compliance_engine_enabled: false` in config, all checks pass (for development)

---

*GDPR Compliance Layer Spec v1.2 | Apache 2.0*
