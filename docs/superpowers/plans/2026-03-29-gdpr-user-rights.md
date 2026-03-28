# GDPR User Rights — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close all GDPR user rights gaps — access 100%, delete 100%, correct, transfer, restrict.

**Architecture:** Extend existing endpoints + add delete methods to all stores + register erasure handlers + per-purpose consent.

**Spec:** `docs/superpowers/specs/2026-03-29-gdpr-user-rights-complete-design.md`

---

### Task 1: Delete Methods for All Stores

Add `delete_user(user_id) -> int` to every data store that holds user data.

**Files to modify:**
- `src/jarvis/gateway/session_store.py` — add `delete_user_sessions(user_id)`
- `src/jarvis/core/user_preferences.py` — add `delete_user(user_id)`
- `src/jarvis/core/conversation_tree.py` — add `delete_user(user_id)`
- `src/jarvis/core/feedback.py` — add `delete_user(user_id)`
- `src/jarvis/core/correction_memory.py` — add `delete_user(user_id)`

Each method: `DELETE FROM table WHERE user_id = ?` (or `DELETE FROM table` for single-user systems). Return `cursor.rowcount`.

---

### Task 2: Register All Erasure Handlers

**File:** `src/jarvis/gateway/gateway.py`

After the GDPR init block (where `_gdpr_manager` is created), register erasure handlers for every tier. Find the existing handler registrations and add:

```python
# Session tier
if hasattr(self, "_session_store") and self._session_store:
    ss = self._session_store
    erasure.register_handler(lambda uid: ss.delete_user_sessions(uid))

# User preferences
pref = getattr(self, "_user_pref_store", None)
if pref and hasattr(pref, "delete_user"):
    erasure.register_handler(lambda uid: pref.delete_user(uid))

# Conversation tree
ct = getattr(self, "_conversation_tree", None)
if ct and hasattr(ct, "delete_user"):
    erasure.register_handler(lambda uid: ct.delete_user(uid))

# Feedback
fb = getattr(self, "_feedback_store", None)
if fb and hasattr(fb, "delete_user"):
    erasure.register_handler(lambda uid: fb.delete_user(uid))

# Corrections
cm = getattr(self, "_correction_memory", None)
if cm and hasattr(cm, "delete_user"):
    erasure.register_handler(lambda uid: cm.delete_user(uid))
```

---

### Task 3: Complete Data Export

**File:** `src/jarvis/channels/config_routes.py`

Replace the existing `GET /api/v1/user/data` with a comprehensive version that exports ALL tiers:

- Sessions (from session_store)
- Vault notes (from vault backend — list all, include title + path + tags)
- Episodic memories (from memory manager)
- Procedures (from memory manager)
- Entities + Relations (from memory indexer)
- User preferences (from user_pref_store)
- Processing log + Model usage (from GDPR manager)
- Consents (from consent manager)
- HIM reports (vault notes in recherchen/osint/)

Each section wrapped in try/except for resilience.

---

### Task 4: Data Correction Endpoint

**File:** `src/jarvis/channels/config_routes.py`

New endpoint: `PATCH /api/v1/user/data`

Accepts JSON with corrections array:
```json
{"corrections": [
    {"type": "entity", "name": "Old", "field": "name", "new_value": "New"},
    {"type": "preference", "key": "greeting_name", "new_value": "Alex"},
    {"type": "vault_note", "path": "...", "field": "content", "new_value": "..."}
]}
```

For each correction type:
- `entity`: find entity by name, update via memory indexer
- `preference`: update via user_pref_store
- `vault_note`: update via vault_update tool

Log each correction in compliance audit.

---

### Task 5: Per-Purpose Restriction

**File:** `src/jarvis/security/consent.py`

Add restriction tracking to ConsentManager:

```python
def restrict_purpose(self, user_id, channel, purpose):
    self.grant_consent(user_id, channel, f"restrict_{purpose}")

def unrestrict_purpose(self, user_id, channel, purpose):
    self.withdraw_consent(user_id, channel, f"restrict_{purpose}")

def is_restricted(self, user_id, channel, purpose):
    return self.has_consent(user_id, channel, f"restrict_{purpose}")

def get_restrictions(self, user_id):
    consents = self.get_user_consents(user_id)
    return [c["consent_type"].replace("restrict_", "")
            for c in consents if c["consent_type"].startswith("restrict_")]
```

**File:** `src/jarvis/security/compliance_engine.py`

Add restriction check after consent check:
```python
# After consent passes:
if self._consent and self._consent.is_restricted(user_id, channel, purpose.value):
    raise ComplianceViolation(f"User restricted {purpose.value} processing")
```

**File:** `src/jarvis/channels/config_routes.py`

New endpoints:
- `POST /api/v1/user/restrictions` — set restriction
- `GET /api/v1/user/restrictions` — list restrictions
- `DELETE /api/v1/user/restrictions` — remove restriction

---

### Task 6: Data Import (Portability)

**File:** `src/jarvis/channels/config_routes.py`

New endpoint: `POST /api/v1/user/data/import`

Accepts the JSON export format. For each section:
- `vault_notes`: call vault_save for each note (skip if path exists)
- `entities`: call add_entity for each (skip duplicates)
- `user_preferences`: merge preferences
- `procedures`: write procedure files

Skip: processing logs, consents, sessions (instance-specific).

---

### Task 7: Tests + Commit

**File:** `tests/test_security/test_user_rights.py`

```python
def test_complete_export_has_all_tiers()
def test_erase_all_deletes_sessions()
def test_erase_all_deletes_preferences()
def test_correction_updates_entity()
def test_restriction_blocks_evolution()
def test_restriction_allows_other_purposes()
def test_import_creates_vault_notes()
```
