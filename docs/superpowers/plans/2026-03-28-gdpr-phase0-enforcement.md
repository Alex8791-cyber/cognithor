# GDPR Phase 0: Enforcement Basis — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** No data flows without a compliance gate. Every processing operation must declare legal basis and purpose, and consent must be verified before processing.

**Architecture:** ConsentManager (SQLite) + ComplianceEngine (central gate) wired into Gateway.handle_message() and channel entry points. Builds on existing ProcessingBasis and DataCategory enums in security/gdpr.py.

**Tech Stack:** Python 3.12+, sqlite3, existing security/gdpr.py infrastructure

**Spec:** `docs/superpowers/specs/2026-03-28-gdpr-compliance-layer-design.md` (Sections 1, 9, 10, 15)

**Key constraint:** Existing ProcessingBasis enum already has CONSENT, CONTRACT, LEGITIMATE_INTEREST etc. DataCategory exists. Do NOT duplicate — extend and use.

---

## File Structure

```
CREATE src/jarvis/security/consent.py         — ConsentManager (SQLite-backed)
CREATE src/jarvis/security/compliance_engine.py — Central runtime enforcement
CREATE tests/test_security/test_consent.py     — Consent tests
CREATE tests/test_security/test_compliance_engine.py — Enforcement tests
CREATE data/legal/privacy_notice_de.md         — German privacy notice
CREATE data/legal/privacy_notice_en.md         — English privacy notice
MODIFY src/jarvis/security/gdpr.py             — Add DataPurpose enum
MODIFY src/jarvis/config.py                    — Add ComplianceConfig
MODIFY src/jarvis/gateway/gateway.py           — Wire ComplianceEngine + ConsentManager
MODIFY src/jarvis/channels/telegram.py         — Add consent check before processing
```

---

### Task 1: DataPurpose Enum + ComplianceConfig

**Files:**
- Modify: `src/jarvis/security/gdpr.py`
- Modify: `src/jarvis/config.py`

- [ ] **Step 1: Add DataPurpose enum to security/gdpr.py**

Find the existing enums (DataCategory, ProcessingBasis, etc.) and add after them:

```python
class DataPurpose(str, Enum):
    """Purpose tag for every stored data item (Art. 5(1)(b))."""
    CONVERSATION = "conversation"
    MEMORY = "memory"
    SECURITY = "security"
    ANALYTICS = "analytics"
    OSINT = "osint"
    EVOLUTION = "evolution"
```

- [ ] **Step 2: Add ComplianceConfig to config.py**

Find where other config classes are defined (near SecurityConfig) and add:

```python
class ComplianceConfig(BaseModel):
    """GDPR compliance configuration."""
    consent_required: bool = True
    compliance_engine_enabled: bool = True
    privacy_mode: bool = False
    privacy_notice_version: str = "1.0"
    cloud_consent_required: bool = True
```

Add to JarvisConfig:
```python
compliance: ComplianceConfig = Field(default_factory=ComplianceConfig)
```

- [ ] **Step 3: Commit**

```bash
git add src/jarvis/security/gdpr.py src/jarvis/config.py
git commit -m "feat(gdpr): add DataPurpose enum and ComplianceConfig"
```

---

### Task 2: Consent Manager

**Files:**
- Create: `src/jarvis/security/consent.py`
- Create: `tests/test_security/test_consent.py`

- [ ] **Step 1: Write tests**

Create `tests/test_security/test_consent.py`:

```python
"""Tests for GDPR Consent Manager."""
from __future__ import annotations

import pytest
from pathlib import Path
from jarvis.security.consent import ConsentManager


@pytest.fixture
def consent_mgr(tmp_path):
    db_path = tmp_path / "consent.db"
    return ConsentManager(db_path=str(db_path))


def test_no_consent_by_default(consent_mgr):
    assert consent_mgr.has_consent("user1", "telegram") is False


def test_grant_consent(consent_mgr):
    consent_mgr.grant_consent("user1", "telegram", "data_processing", context="chat_123")
    assert consent_mgr.has_consent("user1", "telegram") is True


def test_withdraw_consent(consent_mgr):
    consent_mgr.grant_consent("user1", "telegram", "data_processing")
    consent_mgr.withdraw_consent("user1", "telegram", "data_processing")
    assert consent_mgr.has_consent("user1", "telegram") is False


def test_consent_per_channel(consent_mgr):
    consent_mgr.grant_consent("user1", "telegram", "data_processing")
    assert consent_mgr.has_consent("user1", "telegram") is True
    assert consent_mgr.has_consent("user1", "webui") is False


def test_consent_type_specific(consent_mgr):
    consent_mgr.grant_consent("user1", "telegram", "data_processing")
    assert consent_mgr.has_consent("user1", "telegram", "data_processing") is True
    assert consent_mgr.has_consent("user1", "telegram", "cloud_llm") is False


def test_consent_versioning(consent_mgr):
    consent_mgr.grant_consent("user1", "telegram", "data_processing", policy_version="1.0")
    assert consent_mgr.has_consent("user1", "telegram", policy_version="1.0") is True
    assert consent_mgr.has_consent("user1", "telegram", policy_version="2.0") is False


def test_get_user_consents(consent_mgr):
    consent_mgr.grant_consent("user1", "telegram", "data_processing")
    consent_mgr.grant_consent("user1", "telegram", "cloud_llm")
    consents = consent_mgr.get_user_consents("user1")
    assert len(consents) == 2


def test_requires_consent_true_when_none(consent_mgr):
    assert consent_mgr.requires_consent("user1", "telegram") is True


def test_requires_consent_false_after_grant(consent_mgr):
    consent_mgr.grant_consent("user1", "telegram", "data_processing")
    assert consent_mgr.requires_consent("user1", "telegram") is False


def test_delete_user(consent_mgr):
    consent_mgr.grant_consent("user1", "telegram", "data_processing")
    consent_mgr.delete_user("user1")
    assert consent_mgr.has_consent("user1", "telegram") is False
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_security/test_consent.py -v
```

- [ ] **Step 3: Write `src/jarvis/security/consent.py`**

```python
"""GDPR Consent Manager — per-channel consent tracking with versioning."""
from __future__ import annotations

import sqlite3
import time
import uuid
from pathlib import Path

from jarvis.utils.logging import get_logger

log = get_logger(__name__)

__all__ = ["ConsentManager"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS consent (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    channel TEXT NOT NULL,
    consent_type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'accepted',
    policy_version TEXT DEFAULT '1.0',
    granted_at TEXT,
    withdrawn_at TEXT,
    context TEXT DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_consent_user
    ON consent(user_id, channel, consent_type);
CREATE INDEX IF NOT EXISTS idx_consent_status
    ON consent(user_id, status);
"""


class ConsentManager:
    """Track per-user, per-channel GDPR consent with versioning."""

    def __init__(self, db_path: str | None = None) -> None:
        if db_path is None:
            db_path = str(Path.home() / ".jarvis" / "index" / "consent.db")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def has_consent(
        self,
        user_id: str,
        channel: str,
        consent_type: str = "data_processing",
        policy_version: str | None = None,
    ) -> bool:
        """Check if user has active consent for this channel and type."""
        query = (
            "SELECT 1 FROM consent "
            "WHERE user_id = ? AND channel = ? AND consent_type = ? AND status = 'accepted'"
        )
        params: list = [user_id, channel, consent_type]
        if policy_version:
            query += " AND policy_version = ?"
            params.append(policy_version)
        query += " LIMIT 1"
        row = self._conn.execute(query, params).fetchone()
        return row is not None

    def grant_consent(
        self,
        user_id: str,
        channel: str,
        consent_type: str = "data_processing",
        context: str = "",
        policy_version: str = "1.0",
    ) -> None:
        """Record that user granted consent."""
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        # Upsert: withdraw any existing, then insert new
        self._conn.execute(
            "UPDATE consent SET status = 'superseded', withdrawn_at = ? "
            "WHERE user_id = ? AND channel = ? AND consent_type = ? AND status = 'accepted'",
            (now, user_id, channel, consent_type),
        )
        self._conn.execute(
            "INSERT INTO consent (id, user_id, channel, consent_type, status, "
            "policy_version, granted_at, context, created_at) "
            "VALUES (?, ?, ?, ?, 'accepted', ?, ?, ?, ?)",
            (uuid.uuid4().hex[:16], user_id, channel, consent_type,
             policy_version, now, context, now),
        )
        self._conn.commit()
        log.info("consent_granted", user_id=user_id[:8], channel=channel, type=consent_type)

    def withdraw_consent(
        self,
        user_id: str,
        channel: str,
        consent_type: str = "data_processing",
    ) -> None:
        """Record consent withdrawal."""
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._conn.execute(
            "UPDATE consent SET status = 'withdrawn', withdrawn_at = ? "
            "WHERE user_id = ? AND channel = ? AND consent_type = ? AND status = 'accepted'",
            (now, user_id, channel, consent_type),
        )
        self._conn.commit()
        log.info("consent_withdrawn", user_id=user_id[:8], channel=channel, type=consent_type)

    def requires_consent(self, user_id: str, channel: str) -> bool:
        """Check if user still needs to give consent for this channel."""
        return not self.has_consent(user_id, channel, "data_processing")

    def get_user_consents(self, user_id: str) -> list[dict]:
        """Return all consent records for a user."""
        cursor = self._conn.execute(
            "SELECT * FROM consent WHERE user_id = ? AND status = 'accepted' "
            "ORDER BY created_at DESC",
            (user_id,),
        )
        columns = [d[0] for d in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def delete_user(self, user_id: str) -> int:
        """Delete all consent records for a user (for erasure)."""
        cursor = self._conn.execute(
            "DELETE FROM consent WHERE user_id = ?", (user_id,)
        )
        self._conn.commit()
        return cursor.rowcount

    def close(self) -> None:
        self._conn.close()
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_security/test_consent.py -v
```
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/jarvis/security/consent.py tests/test_security/test_consent.py
git commit -m "feat(gdpr): ConsentManager with per-channel consent tracking and versioning"
```

---

### Task 3: Compliance Engine

**Files:**
- Create: `src/jarvis/security/compliance_engine.py`
- Create: `tests/test_security/test_compliance_engine.py`

- [ ] **Step 1: Write tests**

Create `tests/test_security/test_compliance_engine.py`:

```python
"""Tests for GDPR Compliance Engine."""
from __future__ import annotations

import pytest
from pathlib import Path
from jarvis.security.compliance_engine import ComplianceEngine, ComplianceViolation
from jarvis.security.consent import ConsentManager
from jarvis.security.gdpr import ProcessingBasis, DataPurpose


@pytest.fixture
def consent_mgr(tmp_path):
    return ConsentManager(db_path=str(tmp_path / "consent.db"))


@pytest.fixture
def engine(consent_mgr):
    return ComplianceEngine(consent_manager=consent_mgr)


def test_blocks_without_consent(engine):
    with pytest.raises(ComplianceViolation, match="No consent"):
        engine.check(
            user_id="user1",
            channel="telegram",
            legal_basis=ProcessingBasis.CONSENT,
            purpose=DataPurpose.CONVERSATION,
        )


def test_allows_with_consent(engine, consent_mgr):
    consent_mgr.grant_consent("user1", "telegram", "data_processing")
    # Should not raise
    engine.check(
        user_id="user1",
        channel="telegram",
        legal_basis=ProcessingBasis.CONSENT,
        purpose=DataPurpose.CONVERSATION,
    )


def test_legitimate_interest_no_consent_needed(engine):
    # Security monitoring doesn't need consent
    engine.check(
        user_id="user1",
        channel="telegram",
        legal_basis=ProcessingBasis.LEGITIMATE_INTEREST,
        purpose=DataPurpose.SECURITY,
    )


def test_privacy_mode_blocks_storage(engine):
    engine.set_privacy_mode(True)
    with pytest.raises(ComplianceViolation, match="[Pp]rivacy mode"):
        engine.check(
            user_id="user1",
            channel="telegram",
            legal_basis=ProcessingBasis.LEGITIMATE_INTEREST,
            purpose=DataPurpose.CONVERSATION,
        )


def test_privacy_mode_allows_security(engine):
    engine.set_privacy_mode(True)
    # Security purpose should still work in privacy mode
    engine.check(
        user_id="user1",
        channel="telegram",
        legal_basis=ProcessingBasis.LEGITIMATE_INTEREST,
        purpose=DataPurpose.SECURITY,
    )


def test_osint_requires_explicit_consent(engine, consent_mgr):
    consent_mgr.grant_consent("user1", "telegram", "data_processing")
    with pytest.raises(ComplianceViolation, match="[Oo]sint"):
        engine.check(
            user_id="user1",
            channel="telegram",
            legal_basis=ProcessingBasis.CONSENT,
            purpose=DataPurpose.OSINT,
        )


def test_osint_allowed_with_osint_consent(engine, consent_mgr):
    consent_mgr.grant_consent("user1", "telegram", "data_processing")
    consent_mgr.grant_consent("user1", "telegram", "osint")
    engine.check(
        user_id="user1",
        channel="telegram",
        legal_basis=ProcessingBasis.CONSENT,
        purpose=DataPurpose.OSINT,
    )


def test_disabled_engine_allows_everything(consent_mgr):
    engine = ComplianceEngine(consent_manager=consent_mgr, enabled=False)
    # Should not raise even without consent
    engine.check(
        user_id="user1",
        channel="telegram",
        legal_basis=ProcessingBasis.CONSENT,
        purpose=DataPurpose.CONVERSATION,
    )
```

- [ ] **Step 2: Write `src/jarvis/security/compliance_engine.py`**

```python
"""GDPR Compliance Engine — central runtime enforcement.

Every data processing operation must pass through this engine.
It enforces: consent requirements, legal basis validation,
purpose limitations, and privacy mode.
"""
from __future__ import annotations

from jarvis.security.consent import ConsentManager
from jarvis.security.gdpr import DataPurpose, ProcessingBasis
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

__all__ = ["ComplianceEngine", "ComplianceViolation"]


class ComplianceViolation(Exception):
    """Raised when a processing operation violates GDPR policy."""


class ComplianceEngine:
    """Central GDPR policy enforcer. Called before every processing operation.

    Rules:
    1. Consent-based processing requires actual consent
    2. Privacy mode blocks all persistent storage except security
    3. OSINT requires explicit OSINT consent
    4. Legitimate interest bypasses consent (security monitoring, audit)
    """

    def __init__(
        self,
        consent_manager: ConsentManager | None = None,
        enabled: bool = True,
    ) -> None:
        self._consent = consent_manager
        self._enabled = enabled
        self._privacy_mode = False

    def set_privacy_mode(self, enabled: bool) -> None:
        self._privacy_mode = enabled
        log.info("privacy_mode_changed", enabled=enabled)

    @property
    def privacy_mode(self) -> bool:
        return self._privacy_mode

    def check(
        self,
        user_id: str,
        channel: str,
        legal_basis: ProcessingBasis,
        purpose: DataPurpose,
        data_types: list[str] | None = None,
    ) -> None:
        """Verify that the processing operation is GDPR-compliant.

        Raises ComplianceViolation if not allowed.
        Does nothing if engine is disabled (development mode).
        """
        if not self._enabled:
            return

        # Rule 1: Privacy mode blocks everything except security
        if self._privacy_mode and purpose != DataPurpose.SECURITY:
            raise ComplianceViolation(
                f"Privacy mode active — {purpose.value} processing blocked"
            )

        # Rule 2: Consent-based processing requires actual consent
        if legal_basis == ProcessingBasis.CONSENT:
            if self._consent and not self._consent.has_consent(user_id, channel):
                raise ComplianceViolation(
                    f"No consent for {purpose.value} on channel {channel}. "
                    f"User {user_id[:8]} must accept the privacy notice first."
                )

        # Rule 3: OSINT requires explicit OSINT consent (above and beyond data_processing)
        if purpose == DataPurpose.OSINT:
            if self._consent and not self._consent.has_consent(user_id, channel, "osint"):
                raise ComplianceViolation(
                    f"OSINT investigation requires explicit osint consent from user {user_id[:8]}"
                )

        # Rule 4: Legitimate interest is allowed without consent
        # (security monitoring, audit trails, fraud detection)
        # No check needed — this is the bypass

        log.debug(
            "compliance_check_passed",
            user=user_id[:8],
            channel=channel,
            basis=legal_basis.value,
            purpose=purpose.value,
        )
```

- [ ] **Step 3: Run tests**

```bash
python -m pytest tests/test_security/test_compliance_engine.py -v
```
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add src/jarvis/security/compliance_engine.py tests/test_security/test_compliance_engine.py
git commit -m "feat(gdpr): ComplianceEngine — central runtime enforcement with consent+purpose+privacy checks"
```

---

### Task 4: Privacy Notices

**Files:**
- Create: `data/legal/privacy_notice_de.md`
- Create: `data/legal/privacy_notice_en.md`

- [ ] **Step 1: Write German privacy notice**

Create `data/legal/privacy_notice_de.md`:

```markdown
# Datenschutzhinweis — Cognithor Agent OS

**Version:** 1.0
**Stand:** 2026-03-28

## Was wird gespeichert?
- Deine Nachrichten und Gespraeche (fuer Kontext und Antwortqualitaet)
- Erstellte Erinnerungen, Notizen und Wissensgraph-Eintraege
- Verarbeitungsprotokolle (welche Tools verwendet wurden)

## Rechtsgrundlage
- **Einwilligung (Art. 6 Abs. 1 lit. a DSGVO)** fuer Nachrichtenverarbeitung und Speicherung
- **Berechtigtes Interesse (Art. 6 Abs. 1 lit. f DSGVO)** fuer Sicherheitsprotokollierung

## Drittanbieter
Im Hybrid-/Online-Modus koennen Anfragen an Cloud-KI-Dienste (z.B. Anthropic, OpenAI) weitergeleitet werden. Dafuer wird eine separate Einwilligung eingeholt.

## Speicherdauer
- Gespraeche: 180 Tage
- Erinnerungen: 365 Tage
- Sicherheitsprotokolle: 365 Tage
- OSINT-Berichte: 30 Tage

## Deine Rechte
- **Auskunft** (Art. 15 DSGVO): Du kannst jederzeit eine Kopie deiner Daten anfordern
- **Loeschung** (Art. 17 DSGVO): Du kannst die Loeschung aller deiner Daten verlangen
- **Widerruf** (Art. 7 Abs. 3 DSGVO): Du kannst deine Einwilligung jederzeit widerrufen
- **Datenportabilitaet** (Art. 20 DSGVO): Du kannst deine Daten im JSON-Format exportieren

## Kontakt
Verantwortlicher: [Wird in config.yaml konfiguriert]
```

- [ ] **Step 2: Write English privacy notice**

Create `data/legal/privacy_notice_en.md`:

```markdown
# Privacy Notice — Cognithor Agent OS

**Version:** 1.0
**Date:** 2026-03-28

## What data is stored?
- Your messages and conversations (for context and response quality)
- Created memories, notes, and knowledge graph entries
- Processing logs (which tools were used)

## Legal basis
- **Consent (Art. 6(1)(a) GDPR)** for message processing and storage
- **Legitimate interest (Art. 6(1)(f) GDPR)** for security logging

## Third parties
In hybrid/online mode, queries may be forwarded to cloud AI services (e.g. Anthropic, OpenAI). Separate consent is obtained for this.

## Retention periods
- Conversations: 180 days
- Memories: 365 days
- Security logs: 365 days
- OSINT reports: 30 days

## Your rights
- **Access** (Art. 15 GDPR): Request a copy of your data at any time
- **Erasure** (Art. 17 GDPR): Request deletion of all your data
- **Withdrawal** (Art. 7(3) GDPR): Withdraw your consent at any time
- **Portability** (Art. 20 GDPR): Export your data in JSON format

## Contact
Data controller: [Configured in config.yaml]
```

- [ ] **Step 3: Commit**

```bash
git add data/legal/
git commit -m "docs(gdpr): add privacy notices (German + English) for consent flow"
```

---

### Task 5: Gateway Integration

**Files:**
- Modify: `src/jarvis/gateway/gateway.py`

- [ ] **Step 1: Wire ConsentManager and ComplianceEngine into gateway**

In `gateway.py`, find the `init_compliance()` method (or wherever compliance components are initialized). Add:

```python
# At the top imports:
from jarvis.security.consent import ConsentManager
from jarvis.security.compliance_engine import ComplianceEngine

# In init or init_compliance:
self._consent_manager = ConsentManager(
    db_path=str(Path(config.jarvis_home) / "index" / "consent.db")
)
self._compliance_engine = ComplianceEngine(
    consent_manager=self._consent_manager,
    enabled=getattr(getattr(config, "compliance", None), "compliance_engine_enabled", True),
)
if getattr(getattr(config, "compliance", None), "privacy_mode", False):
    self._compliance_engine.set_privacy_mode(True)
```

- [ ] **Step 2: Add compliance check in handle_message()**

Find `handle_message()` and add at the very beginning (before any processing):

```python
# GDPR compliance gate — check consent before processing
from jarvis.security.gdpr import ProcessingBasis, DataPurpose
from jarvis.security.compliance_engine import ComplianceViolation

if hasattr(self, "_compliance_engine") and self._compliance_engine:
    try:
        self._compliance_engine.check(
            user_id=msg.user_id or msg.session_id,
            channel=msg.channel,
            legal_basis=ProcessingBasis.CONSENT,
            purpose=DataPurpose.CONVERSATION,
        )
    except ComplianceViolation as e:
        log.info("compliance_blocked", user=msg.user_id[:8] if msg.user_id else "unknown", reason=str(e))
        return OutgoingMessage(
            channel=msg.channel,
            text=str(e),
            session_id=msg.session_id,
            is_final=True,
        )
```

- [ ] **Step 3: Expose consent_manager for channels**

Add a property:
```python
@property
def consent_manager(self) -> ConsentManager | None:
    return getattr(self, "_consent_manager", None)
```

- [ ] **Step 4: Commit**

```bash
git add src/jarvis/gateway/gateway.py
git commit -m "feat(gdpr): wire ComplianceEngine into Gateway.handle_message()"
```

---

### Task 6: Telegram Consent Flow

**Files:**
- Modify: `src/jarvis/channels/telegram.py`

- [ ] **Step 1: Add consent check in _process_incoming()**

Find the `_process_incoming()` method. At the very beginning (after whitelist check), add:

```python
# GDPR consent check
consent_mgr = getattr(self._gateway, "consent_manager", None) if hasattr(self, "_gateway") else None
if consent_mgr is None and hasattr(self, "_handler") and hasattr(self._handler, "__self__"):
    consent_mgr = getattr(self._handler.__self__, "consent_manager", None)

if consent_mgr and consent_mgr.requires_consent(str(user_id), "telegram"):
    # Check if this IS the consent response
    if text and text.strip().lower() in ("akzeptieren", "accept", "ja", "yes"):
        consent_mgr.grant_consent(str(user_id), "telegram", "data_processing",
                                   context=str(chat_id))
        await self._reply(chat_id, "Datenschutz-Einwilligung erteilt. Ich bin jetzt bereit!")
        return
    elif text and text.strip().lower() in ("ablehnen", "decline", "nein", "no"):
        await self._reply(chat_id,
            "Ich kann deine Nachrichten ohne Datenschutz-Einwilligung nicht verarbeiten. "
            "Sende 'akzeptieren' wenn du es dir anders ueberlegst.")
        return
    else:
        # Show privacy notice
        await self._reply(chat_id,
            "Datenschutzhinweis: Ich speichere Nachrichten, Erinnerungen und "
            "Verarbeitungsprotokolle. Details: cognithor.dev/privacy\n\n"
            "Antworte mit 'akzeptieren' oder 'ablehnen'.")
        return
```

Note: `_reply` is a helper. If it doesn't exist, use the existing send mechanism:
```python
async def _reply(self, chat_id, text):
    """Send a simple text reply to a chat."""
    await self._app.bot.send_message(chat_id=chat_id, text=text)
```

- [ ] **Step 2: Store gateway reference**

In the Telegram channel's `start()` method, store the handler reference so we can access consent_manager:

```python
self._handler = handler
```

- [ ] **Step 3: Commit**

```bash
git add src/jarvis/channels/telegram.py
git commit -m "feat(gdpr): Telegram consent flow — privacy notice before processing"
```

---

### Task 7: Final Integration Test

**Files:**
- Run all existing tests to verify no regressions

- [ ] **Step 1: Run GDPR-specific tests**

```bash
python -m pytest tests/test_security/test_consent.py tests/test_security/test_compliance_engine.py -v
```

- [ ] **Step 2: Run full test suite (quick check for regressions)**

```bash
python -m pytest tests/ -x --timeout=60 -q 2>&1 | tail -10
```

- [ ] **Step 3: Final commit with all Phase 0 changes**

```bash
git add -A
git commit -m "feat(gdpr): Phase 0 complete — enforcement basis

GDPR Phase 0 Enforcement Basis:
- ConsentManager: per-channel consent with versioning
- ComplianceEngine: runtime enforcement (consent, purpose, privacy mode)
- Gateway integration: compliance check before every message
- Telegram consent flow: privacy notice + accept/decline
- Privacy notices: German + English
- DataPurpose enum + ComplianceConfig

No data flows without a compliance gate."
```

---

## Definition of Done

- [ ] ConsentManager tracks consent per user + channel + type with versioning
- [ ] ComplianceEngine blocks processing without consent (fail-closed)
- [ ] Legitimate interest bypasses consent (security logging still works)
- [ ] Privacy mode blocks all storage except security
- [ ] OSINT requires explicit osint consent
- [ ] Gateway.handle_message() runs compliance check first
- [ ] Telegram shows privacy notice on first contact
- [ ] Privacy notices exist in German + English
- [ ] All existing tests still pass (no regressions)
- [ ] Engine can be disabled via config (development mode)
