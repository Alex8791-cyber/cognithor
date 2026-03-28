# GDPR User Rights — Complete Implementation Spec

> **For agentic workers:** Use superpowers:writing-plans to create the implementation plan.

**Goal:** Close all remaining GDPR user rights gaps. Every user must be able to access, correct, delete, restrict, transfer, and understand ALL their data.

**Date:** 2026-03-29
**Status:** Approved

---

## 1. Complete Data Export (Art. 15 + Art. 20)

### Current gap
`GET /api/v1/user/data` only exports processing logs, consents, memories, and entities. Missing: vault notes, sessions, HIM reports, episodic memories, procedures, evolution data.

### Solution
Extend the export endpoint to collect from ALL data tiers:

```python
export = {
    "export_version": "2.0",
    "format": "cognithor_portable",  # machine-readable, importable
    "user_id": user_id,
    "exported_at": iso_timestamp,

    # Tier 1: Direct data
    "sessions": [...],           # from session_store
    "vault_notes": [...],        # from vault backend
    "episodic_memories": [...],  # from episodic .md files
    "procedures": [...],         # from procedural .md files

    # Tier 2: Derived data
    "entities": [...],           # from memory indexer
    "relations": [...],          # from memory indexer
    "claims": [...],             # from knowledge_validator

    # Tier 3: Processing data
    "processing_log": [...],     # from GDPR manager
    "model_usage_log": [...],    # from GDPR manager
    "consents": [...],           # from consent manager

    # Tier 4: Investigation data
    "him_reports": [...],        # from vault recherchen/osint/

    # Tier 5: Preferences
    "user_preferences": {...},   # from user_preferences store
    "core_memory": "...",        # CORE.md content
}
```

Also support `?format=csv` for tabular data (sessions, entities, processing log).

### MCP Tool
`export_user_data` — GREEN classification (read-only)

---

## 2. Complete Data Deletion (Art. 17)

### Current gap
`erase_all()` deletes processing logs, model usage, consents, and registered handlers. Missing: vault notes, sessions, entities by user, episodic memories, HIM reports, user preferences.

### Solution
Register erasure handlers for ALL remaining tiers during gateway init:

```python
# In gateway.py after GDPR + memory init:
erasure = gdpr_manager.erasure

# Vault: delete all notes by user (or all if single-user)
erasure.register_handler(lambda uid: vault_tools.delete_all_user_notes(uid))

# Sessions: delete user's sessions
erasure.register_handler(lambda uid: session_store.delete_user_sessions(uid))

# Entities: delete entities with user-related attributes
erasure.register_handler(lambda uid: memory_tools.delete_user_entities(uid))

# User preferences: clear
erasure.register_handler(lambda uid: pref_store.delete_user(uid))

# HIM reports: delete from vault recherchen/osint/
erasure.register_handler(lambda uid: vault_tools.delete_osint_reports(uid))

# Episodic: clear user's episodes
erasure.register_handler(lambda uid: memory_manager.episodic.clear_user(uid))

# Conversation tree: delete
erasure.register_handler(lambda uid: conversation_tree.delete_user(uid))

# Feedback + corrections: delete
erasure.register_handler(lambda uid: feedback_store.delete_user(uid))
erasure.register_handler(lambda uid: correction_memory.delete_user(uid))
```

Each handler returns count of items deleted. `erase_all()` calls all handlers and returns aggregate counts.

### New methods needed
- `SessionStore.delete_user_sessions(user_id) -> int`
- `UserPreferenceStore.delete_user(user_id) -> int`
- `ConversationTree.delete_user(user_id) -> int`
- `FeedbackStore.delete_user(user_id) -> int`
- `CorrectionMemory.delete_user(user_id) -> int`
- `EpisodicMemory.clear_user(user_id) -> int` (or clear_all for single-user)

These are simple SQL DELETE statements.

---

## 3. Data Correction (Art. 16)

### Current gap
No user-facing way to correct stored data. Vault notes can be updated, but entities, memories, and preferences cannot be corrected via API.

### Solution
New REST endpoint: `PATCH /api/v1/user/data`

```json
{
    "corrections": [
        {"type": "entity", "name": "Old Name", "field": "name", "new_value": "Corrected Name"},
        {"type": "preference", "key": "greeting_name", "new_value": "Alex"},
        {"type": "memory", "id": "chunk_123", "action": "delete"},
        {"type": "vault_note", "path": "wissen/note.md", "field": "content", "new_value": "..."}
    ]
}
```

New MCP tool: `correct_user_data` — YELLOW classification

Each correction is logged in the compliance audit log.

---

## 4. Data Portability / Transfer (Art. 20)

### Current gap
Export is JSON but not importable by another Cognithor instance.

### Solution
The export format (`cognithor_portable`) is designed to be importable:

New REST endpoint: `POST /api/v1/user/data/import`
- Accepts the JSON export from another instance
- Imports: vault notes, entities, relations, preferences, procedures
- Skips: processing logs, consents (instance-specific)
- Conflict resolution: skip duplicates (by title/path)

New MCP tool: `import_user_data` — RED classification (creates data)

---

## 5. Granular Restriction (Art. 18 + Art. 21)

### Current gap
Only all-or-nothing: Privacy Mode disables everything, consent withdrawal blocks everything. No per-purpose opt-out.

### Solution
Extend ConsentManager with purpose-specific consent:

```python
# User can consent/withdraw per purpose:
consent_mgr.grant_consent(user_id, channel, "conversation")
consent_mgr.grant_consent(user_id, channel, "memory")
consent_mgr.withdraw_consent(user_id, channel, "evolution")  # Stop learning about me
consent_mgr.withdraw_consent(user_id, channel, "cloud_llm")  # Don't send to cloud
```

ComplianceEngine checks purpose-specific consent:
```python
def check(self, user_id, channel, legal_basis, purpose):
    # Check general consent
    if not self._consent.has_consent(user_id, channel, "data_processing"):
        raise ComplianceViolation(...)
    # Check purpose-specific restriction
    if self._consent.is_restricted(user_id, channel, purpose.value):
        raise ComplianceViolation(f"User restricted {purpose.value} processing")
```

New ConsentManager methods:
- `restrict_purpose(user_id, channel, purpose) -> None`
- `unrestrict_purpose(user_id, channel, purpose) -> None`
- `is_restricted(user_id, channel, purpose) -> bool`
- `get_restrictions(user_id) -> list[str]`

Flutter UI: Settings page with per-purpose toggles.

---

## 6. Files to Modify/Create

| File | Change |
|------|--------|
| `channels/config_routes.py` | Extend GET export, add PATCH correct, add POST import |
| `security/gdpr.py` | Extend erase_all with more handler calls |
| `security/consent.py` | Add restrict/unrestrict purpose methods |
| `security/compliance_engine.py` | Check purpose-specific restrictions |
| `gateway/gateway.py` | Register all erasure handlers at init |
| `gateway/session_store.py` | Add delete_user_sessions method |
| `core/user_preferences.py` | Add delete_user method |
| `core/conversation_tree.py` | Add delete_user method |
| `core/feedback.py` | Add delete_user method |
| `core/correction_memory.py` | Add delete_user method |
| `mcp/memory_server.py` | Add delete_user_entities tool |
| `flutter_app/.../settings/` | Per-purpose restriction toggles |

---

## 7. Non-Breaking Guarantees

- All existing functionality unchanged
- New endpoints are additions, not modifications
- Consent model is backward-compatible (existing "data_processing" consent still works)
- Import skips duplicates (safe to run multiple times)
- Every correction/deletion logged in compliance audit

---

*GDPR User Rights Complete Spec v1.0 | Apache 2.0*
