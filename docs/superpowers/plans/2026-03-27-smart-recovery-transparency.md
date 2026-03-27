# Smart Recovery & Transparency System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pre-flight plan previews (non-blocking, 3s auto-execute), live mid-execution correction via natural language, and post-correction learning that makes Cognithor smarter over time.

**Architecture:** Pre-flight inserts a brief WebSocket notification after the Planner produces an ActionPlan but does NOT block — it shows the plan and auto-executes after 3 seconds unless the user cancels. Live correction uses the existing `_cancelled_sessions` mechanism with correction-context injection into the Planner. CorrectionMemory (SQLite) stores patterns and injects reminders into the Context Pipeline.

**Tech Stack:** Python 3.12+ (asyncio, sqlite3), Flutter/Dart, pytest

**CRITICAL DESIGN PRINCIPLE:** The system is agentic-first. Pre-flight is a NOTIFICATION, not a gate. It shows what's coming and auto-proceeds. The user CAN intervene but doesn't HAVE to. Never make the user wait.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `src/jarvis/core/correction_memory.py` | CorrectionMemory: SQLite store + keyword matching |
| Modify | `src/jarvis/core/context_pipeline.py` | Inject correction reminders pre-planner |
| Modify | `src/jarvis/gateway/gateway.py` | Pre-flight notification + correction detection in PGE loop |
| Modify | `src/jarvis/config.py` | RecoveryConfig model |
| Modify | `src/jarvis/__main__.py` | WebSocket pre_flight_cancel message handling |
| Create | `flutter_app/lib/widgets/chat/pre_flight_card.dart` | Pre-flight UI card with countdown |
| Modify | `flutter_app/lib/providers/chat_provider.dart` | Handle pre_flight WS messages |
| Modify | `flutter_app/lib/screens/chat_screen.dart` | Render PreFlightCard |
| Create | `tests/unit/test_correction_memory.py` | Tests |

---

### Task 1: RecoveryConfig + CorrectionMemory

**Files:**
- Modify: `src/jarvis/config.py`
- Create: `src/jarvis/core/correction_memory.py`
- Create: `tests/unit/test_correction_memory.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_correction_memory.py`:

```python
"""Tests for CorrectionMemory — stores and retrieves user corrections."""

import pytest
from pathlib import Path


class TestCorrectionMemory:

    @pytest.fixture
    def mem(self, tmp_path):
        from jarvis.core.correction_memory import CorrectionMemory
        return CorrectionMemory(db_path=tmp_path / "corrections.db")

    def test_store_correction(self, mem):
        cid = mem.store(
            user_message="Analysiere den Vertrag",
            correction_text="Nein, fasse nur zusammen",
            original_plan="analyze_document + write_file",
        )
        assert cid.startswith("corr_")

    def test_find_similar(self, mem):
        mem.store(
            user_message="Analysiere den Vertrag",
            correction_text="Nur zusammenfassen, keine Risiken",
            original_plan="full_analysis",
        )
        matches = mem.find_similar("Pruefe diesen Vertrag")
        assert len(matches) >= 1
        assert "zusammenfassen" in matches[0]["correction_text"]

    def test_no_match_for_unrelated(self, mem):
        mem.store(
            user_message="Schreibe einen Schachbot",
            correction_text="Benutze kein Stockfish",
            original_plan="exec_command stockfish",
        )
        matches = mem.find_similar("Was ist das Wetter?")
        assert len(matches) == 0

    def test_increment_times_triggered(self, mem):
        mem.store(
            user_message="Recherchiere X",
            correction_text="Nutze nur deutsche Quellen",
            original_plan="web_search",
        )
        mem.store(
            user_message="Suche nach Y",
            correction_text="Nur deutsche Quellen bitte",
            original_plan="web_search",
        )
        # Similar corrections should be merged or both found
        matches = mem.find_similar("Finde Infos zu Z")
        assert len(matches) >= 1

    def test_should_ask_proactively(self, mem):
        for i in range(3):
            mem.store(
                user_message=f"Recherche {i}",
                correction_text="Nur deutsche Quellen",
                original_plan="web_search",
                keywords=["recherche", "quellen", "deutsch"],
            )
        assert mem.should_ask_proactively("recherche", ["quellen"]) is True

    def test_not_proactive_under_threshold(self, mem):
        mem.store(
            user_message="Recherche",
            correction_text="Nur deutsch",
            original_plan="web_search",
            keywords=["recherche"],
        )
        assert mem.should_ask_proactively("recherche", ["quellen"]) is False

    def test_get_reminder_text(self, mem):
        mem.store(
            user_message="Schreib eine E-Mail",
            correction_text="Immer in Du-Form, nie Sie",
            original_plan="email_send",
            keywords=["email", "schreib"],
        )
        reminder = mem.get_reminder("Verfasse eine E-Mail an Max")
        assert reminder is not None
        assert "Du-Form" in reminder
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "D:\Jarvis\jarvis complete v20" && python -m pytest tests/unit/test_correction_memory.py -v`
Expected: FAIL — `ImportError: cannot import name 'CorrectionMemory'`

- [ ] **Step 3: Add RecoveryConfig to config.py**

In `src/jarvis/config.py`, after AuditConfig, add:

```python
class RecoveryConfig(BaseModel):
    """Smart Recovery & Transparency Konfiguration."""

    pre_flight_enabled: bool = Field(
        default=True,
        description="Plan-Vorschau vor komplexen Aktionen anzeigen",
    )
    pre_flight_timeout_seconds: int = Field(
        default=3, ge=1, le=30,
        description="Auto-Execute nach N Sekunden (agentic-first)",
    )
    pre_flight_min_steps: int = Field(
        default=2, ge=1, le=10,
        description="Pre-Flight nur bei Plaenen mit N+ Schritten",
    )
    correction_learning_enabled: bool = Field(
        default=True,
        description="Aus User-Korrekturen lernen",
    )
    correction_proactive_threshold: int = Field(
        default=3, ge=2, le=10,
        description="Nach N gleichen Korrekturen proaktiv fragen",
    )
```

Wire into JarvisConfig:
```python
    recovery: RecoveryConfig = Field(default_factory=RecoveryConfig)
```

- [ ] **Step 4: Implement CorrectionMemory**

Create `src/jarvis/core/correction_memory.py`:

```python
"""Correction Memory — learns from user corrections to avoid repeating mistakes.

Stores corrections in SQLite, matches similar situations by keyword overlap,
and provides reminders for the Planner context.
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

from jarvis.utils.logging import get_logger

log = get_logger(__name__)

__all__ = ["CorrectionMemory"]

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS corrections (
    id TEXT PRIMARY KEY,
    user_message TEXT NOT NULL,
    correction_text TEXT NOT NULL,
    original_plan TEXT DEFAULT '',
    corrected_plan TEXT DEFAULT '',
    keywords TEXT DEFAULT '',
    times_triggered INTEGER DEFAULT 1,
    created_at REAL NOT NULL,
    last_triggered_at REAL
);
CREATE INDEX IF NOT EXISTS idx_corr_keywords ON corrections(keywords);
"""


class CorrectionMemory:
    """SQLite-backed correction store with keyword matching."""

    def __init__(self, db_path: Path | str, proactive_threshold: int = 3) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._proactive_threshold = proactive_threshold
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.executescript(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def store(
        self,
        user_message: str,
        correction_text: str,
        original_plan: str = "",
        corrected_plan: str = "",
        keywords: list[str] | None = None,
    ) -> str:
        """Store a correction. Returns correction ID."""
        corr_id = f"corr_{uuid.uuid4().hex[:12]}"
        if keywords is None:
            keywords = self._extract_keywords(user_message + " " + correction_text)
        kw_str = ",".join(keywords)

        # Check for similar existing correction (merge)
        with self._conn() as conn:
            existing = conn.execute(
                "SELECT id, times_triggered FROM corrections "
                "WHERE keywords LIKE ? LIMIT 1",
                (f"%{keywords[0]}%" if keywords else "",),
            ).fetchone()

            if existing and self._text_overlap(correction_text, existing) > 0.5:
                conn.execute(
                    "UPDATE corrections SET times_triggered = times_triggered + 1, "
                    "last_triggered_at = ? WHERE id = ?",
                    (time.time(), existing["id"]),
                )
                return existing["id"]

            conn.execute(
                "INSERT INTO corrections (id, user_message, correction_text, "
                "original_plan, corrected_plan, keywords, created_at, last_triggered_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (corr_id, user_message[:500], correction_text[:500],
                 original_plan[:500], corrected_plan[:500], kw_str,
                 time.time(), time.time()),
            )

        log.info("correction_stored", id=corr_id, keywords=kw_str[:60])
        return corr_id

    def find_similar(self, user_message: str, limit: int = 3) -> list[dict[str, Any]]:
        """Find corrections similar to the current user message."""
        keywords = self._extract_keywords(user_message)
        if not keywords:
            return []

        with self._conn() as conn:
            results = []
            for kw in keywords:
                rows = conn.execute(
                    "SELECT * FROM corrections WHERE keywords LIKE ? "
                    "ORDER BY times_triggered DESC, last_triggered_at DESC "
                    "LIMIT ?",
                    (f"%{kw}%", limit),
                ).fetchall()
                for r in rows:
                    d = dict(r)
                    if d not in results:
                        results.append(d)

        return results[:limit]

    def should_ask_proactively(self, query: str, keywords: list[str] | None = None) -> bool:
        """Check if we should proactively ask before acting (threshold reached)."""
        if keywords is None:
            keywords = self._extract_keywords(query)
        if not keywords:
            return False

        with self._conn() as conn:
            for kw in keywords:
                row = conn.execute(
                    "SELECT SUM(times_triggered) as total FROM corrections "
                    "WHERE keywords LIKE ?",
                    (f"%{kw}%",),
                ).fetchone()
                if row and (row["total"] or 0) >= self._proactive_threshold:
                    return True
        return False

    def get_reminder(self, user_message: str) -> str | None:
        """Get a reminder string for the Planner context, or None."""
        matches = self.find_similar(user_message, limit=2)
        if not matches:
            return None

        reminders = []
        for m in matches:
            reminders.append(
                f"- Bei \"{m['user_message'][:80]}\" hat der User korrigiert: "
                f"\"{m['correction_text'][:120]}\""
            )

        if self.should_ask_proactively(user_message):
            return (
                "WICHTIG — Der User hat bei aehnlichen Anfragen mehrfach korrigiert. "
                "Frage ZUERST ob dein Ansatz passt, bevor du handelst:\n"
                + "\n".join(reminders)
            )

        return (
            "ERINNERUNG — Der User hat bei aehnlichen Anfragen korrigiert:\n"
            + "\n".join(reminders)
            + "\nBeruecksichtige das in deinem Plan."
        )

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract meaningful keywords from text."""
        import re
        stopwords = {
            "der", "die", "das", "ein", "eine", "und", "oder", "ist", "sind",
            "hat", "haben", "wird", "werden", "mit", "von", "fuer", "auf",
            "den", "dem", "des", "im", "in", "an", "zu", "nicht", "nein",
            "ja", "bitte", "mal", "noch", "auch", "nur", "mir", "mich",
            "the", "and", "for", "with", "this", "that", "from", "not",
        }
        words = re.findall(r"\b[a-zA-ZäöüÄÖÜß]{3,}\b", text.lower())
        return [w for w in words if w not in stopwords][:10]

    @staticmethod
    def _text_overlap(text: str, row: Any) -> float:
        """Simple word overlap score between correction text and existing row."""
        try:
            existing_text = row["correction_text"] if hasattr(row, "__getitem__") else ""
            words_a = set(text.lower().split())
            words_b = set(existing_text.lower().split())
            if not words_a or not words_b:
                return 0.0
            return len(words_a & words_b) / max(len(words_a), len(words_b))
        except Exception:
            return 0.0
```

- [ ] **Step 5: Run tests**

Run: `cd "D:\Jarvis\jarvis complete v20" && python -m pytest tests/unit/test_correction_memory.py -v`
Expected: All 7 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/jarvis/config.py src/jarvis/core/correction_memory.py tests/unit/test_correction_memory.py
git commit -m "feat: CorrectionMemory + RecoveryConfig for smart recovery system"
```

---

### Task 2: Pre-Flight Notification in Gateway PGE Loop

**Files:**
- Modify: `src/jarvis/gateway/gateway.py`

- [ ] **Step 1: Add pre-flight logic after Planner, before Executor**

In `_run_pge_loop`, find the section AFTER the Planner returns the plan (around line 2169, after `all_plans.append(plan)`) and BEFORE the Gatekeeper evaluates. Add:

```python
            # ── Pre-Flight Notification (non-blocking) ──────────────
            # Show plan preview to user. Auto-execute after timeout.
            # User can cancel by sending "cancel" or "stop".
            _recovery_cfg = getattr(self._config, "recovery", None)
            if (
                _recovery_cfg
                and getattr(_recovery_cfg, "pre_flight_enabled", False)
                and plan.has_actions
                and len(plan.steps) >= getattr(_recovery_cfg, "pre_flight_min_steps", 2)
            ):
                _timeout = getattr(_recovery_cfg, "pre_flight_timeout_seconds", 3)
                _steps_summary = [
                    {"tool": s.tool, "rationale": (s.rationale or "")[:80]}
                    for s in plan.steps[:5]
                ]
                # Send pre-flight notification via WebSocket (non-blocking)
                await _status_cb("pre_flight", json.dumps({
                    "goal": plan.goal or msg.text[:100],
                    "steps": _steps_summary,
                    "timeout": _timeout,
                    "session_id": msg.session_id,
                }))
                # Wait for timeout — but check for cancellation every 0.5s
                _pf_start = time.monotonic()
                while (time.monotonic() - _pf_start) < _timeout:
                    if msg.session_id in self._cancelled_sessions:
                        self._cancelled_sessions.discard(msg.session_id)
                        log.info("pre_flight_cancelled", session=session.session_id[:8])
                        final_response = "Plan abgebrochen. Was soll ich stattdessen tun?"
                        break
                    await asyncio.sleep(0.5)
                else:
                    # Timeout reached — auto-execute (agentic-first)
                    log.debug("pre_flight_auto_execute", session=session.session_id[:8])
                if final_response:
                    break  # Exit PGE loop if cancelled
```

IMPORTANT: Add `import json` at the top of gateway.py if not already present. Also ensure `time` is imported.

- [ ] **Step 2: Verify syntax**

Run: `cd "D:\Jarvis\jarvis complete v20" && python -c "from jarvis.gateway.gateway import Gateway; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/jarvis/gateway/gateway.py
git commit -m "feat: non-blocking pre-flight plan notification in PGE loop (3s auto-execute)"
```

---

### Task 3: Live Correction Detection

**Files:**
- Modify: `src/jarvis/gateway/gateway.py`

- [ ] **Step 1: Add correction detection in handle_message**

Find the `handle_message` method in gateway.py. BEFORE the PGE loop is called, add correction detection for messages that arrive while the system is already processing:

```python
        # ── Live Correction Detection ─────────────────────────────
        # If the system is already processing and user sends a correction,
        # cancel current execution and inject correction context.
        _CORRECTION_TRIGGERS = frozenset({
            "nein", "stopp", "stop", "halt", "falsch", "nicht so",
            "stattdessen", "anders", "korrigier", "abbrech", "cancel",
            "wrong", "no", "lass das", "vergiss das", "mach anders",
        })
        _lower = msg.text.lower().strip()
        _is_correction = any(trigger in _lower for trigger in _CORRECTION_TRIGGERS)

        if _is_correction and session.iteration_count > 0:
            # This is a correction to an ongoing task
            log.info("live_correction_detected", text=msg.text[:80])
            # Cancel current PGE loop
            self.cancel_session(msg.session_id)
            # Store correction
            if hasattr(self, "_correction_memory") and self._correction_memory:
                self._correction_memory.store(
                    user_message=session.last_user_message or "",
                    correction_text=msg.text,
                    original_plan=str(getattr(session, "_last_plan_goal", "")),
                )
            # Inject correction into working memory
            wm.add_message(Message(
                role=MessageRole.SYSTEM,
                content=(
                    f"[KORREKTUR] Der User hat die vorherige Aktion korrigiert: "
                    f"\"{msg.text}\". Passe deinen Plan entsprechend an. "
                    f"Fuehre NICHT die vorherige Aktion erneut aus."
                ),
                channel=msg.channel,
            ))
```

- [ ] **Step 2: Initialize CorrectionMemory in gateway startup**

In the gateway startup section (near where FeedbackStore is initialized), add:

```python
        # Correction Memory (Smart Recovery)
        try:
            from jarvis.core.correction_memory import CorrectionMemory
            _proactive = 3
            if hasattr(self._config, "recovery"):
                _proactive = getattr(self._config.recovery, "correction_proactive_threshold", 3)
            self._correction_memory = CorrectionMemory(
                db_path=self._config.jarvis_home / "corrections.db",
                proactive_threshold=_proactive,
            )
            log.info("correction_memory_initialized")
        except Exception:
            log.debug("correction_memory_init_failed", exc_info=True)
            self._correction_memory = None
```

- [ ] **Step 3: Verify syntax**

Run: `cd "D:\Jarvis\jarvis complete v20" && python -c "from jarvis.gateway.gateway import Gateway; print('OK')"`

- [ ] **Step 4: Commit**

```bash
git add src/jarvis/gateway/gateway.py
git commit -m "feat: live correction detection with cancel + correction context injection"
```

---

### Task 4: Context Pipeline — Correction Reminder Injection

**Files:**
- Modify: `src/jarvis/core/context_pipeline.py`

- [ ] **Step 1: Inject correction reminders in enrich()**

Find the `enrich()` method. At the END, after Wave 2, before returning the ContextResult, add:

```python
        # ── Correction Reminders (Smart Recovery) ────────────────
        if hasattr(self, "_correction_memory") and self._correction_memory:
            try:
                reminder = self._correction_memory.get_reminder(user_message)
                if reminder and len(wm.injected_procedures) < 3:
                    wm.injected_procedures.append(reminder)
                    log.debug("correction_reminder_injected", length=len(reminder))
            except Exception:
                log.debug("correction_reminder_failed", exc_info=True)
```

Also add a method to set the correction memory:

```python
    def set_correction_memory(self, memory: Any) -> None:
        """Inject CorrectionMemory for correction reminders."""
        self._correction_memory = memory
```

And in the gateway, after creating CorrectionMemory, wire it:

```python
        if self._correction_memory and hasattr(self, "_context_pipeline") and self._context_pipeline:
            self._context_pipeline.set_correction_memory(self._correction_memory)
```

- [ ] **Step 2: Verify syntax**

Run: `cd "D:\Jarvis\jarvis complete v20" && python -c "from jarvis.core.context_pipeline import ContextPipeline; print('OK')"`

- [ ] **Step 3: Commit**

```bash
git add src/jarvis/core/context_pipeline.py src/jarvis/gateway/gateway.py
git commit -m "feat: inject correction reminders into Planner context via ContextPipeline"
```

---

### Task 5: Flutter Pre-Flight Card + WS Handling

**Files:**
- Create: `flutter_app/lib/widgets/chat/pre_flight_card.dart`
- Modify: `flutter_app/lib/providers/chat_provider.dart`
- Modify: `flutter_app/lib/screens/chat_screen.dart`

- [ ] **Step 1: Create PreFlightCard widget**

Create `flutter_app/lib/widgets/chat/pre_flight_card.dart`:

```dart
import 'dart:async';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:jarvis_ui/theme/jarvis_theme.dart';

/// Non-blocking plan preview card with auto-execute countdown.
class PreFlightCard extends StatefulWidget {
  const PreFlightCard({
    super.key,
    required this.goal,
    required this.steps,
    required this.timeoutSeconds,
    this.onCancel,
    this.onModify,
  });

  final String goal;
  final List<Map<String, dynamic>> steps;
  final int timeoutSeconds;
  final VoidCallback? onCancel;
  final void Function(String)? onModify;

  @override
  State<PreFlightCard> createState() => _PreFlightCardState();
}

class _PreFlightCardState extends State<PreFlightCard> {
  late int _remaining;
  Timer? _timer;
  bool _expanded = false;
  bool _executed = false;

  @override
  void initState() {
    super.initState();
    _remaining = widget.timeoutSeconds;
    _timer = Timer.periodic(const Duration(seconds: 1), (_) {
      if (_remaining <= 1) {
        _timer?.cancel();
        setState(() => _executed = true);
      } else {
        setState(() => _remaining--);
      }
    });
  }

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (_executed) {
      return Padding(
        padding: const EdgeInsets.symmetric(vertical: 4),
        child: Row(
          children: [
            Icon(Icons.play_arrow, size: 14, color: JarvisTheme.green),
            const SizedBox(width: 6),
            Text(
              'Plan gestartet: ${widget.goal}',
              style: TextStyle(fontSize: 12, color: JarvisTheme.textSecondary),
            ),
          ],
        ),
      );
    }

    final stepsSummary = widget.steps
        .map((s) => s['tool'] ?? '?')
        .join(' → ');

    return Container(
      margin: const EdgeInsets.symmetric(vertical: 6),
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: JarvisTheme.accent.withValues(alpha: 0.08),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: JarvisTheme.accent.withValues(alpha: 0.25)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.route, size: 16, color: JarvisTheme.accent),
              const SizedBox(width: 6),
              Expanded(
                child: Text(
                  '${widget.steps.length} Schritte: $stepsSummary',
                  style: TextStyle(fontSize: 12, color: JarvisTheme.textSecondary),
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                ),
              ),
              Text(
                '${_remaining}s',
                style: TextStyle(
                  fontSize: 11,
                  color: JarvisTheme.accent,
                  fontFamily: 'monospace',
                ),
              ),
            ],
          ),
          if (_expanded) ...[
            const SizedBox(height: 6),
            ...widget.steps.map((s) => Padding(
              padding: const EdgeInsets.only(left: 22, bottom: 2),
              child: Text(
                '${s['tool']}: ${s['rationale'] ?? ''}',
                style: TextStyle(fontSize: 11, color: JarvisTheme.textTertiary),
              ),
            )),
          ],
          const SizedBox(height: 6),
          Row(
            children: [
              InkWell(
                onTap: () => setState(() => _expanded = !_expanded),
                child: Text(
                  _expanded ? 'Weniger' : 'Details',
                  style: TextStyle(fontSize: 11, color: JarvisTheme.accent),
                ),
              ),
              const Spacer(),
              TextButton(
                onPressed: () {
                  _timer?.cancel();
                  widget.onCancel?.call();
                },
                style: TextButton.styleFrom(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                  minimumSize: Size.zero,
                ),
                child: Text('Abbrechen',
                    style: TextStyle(fontSize: 11, color: JarvisTheme.red)),
              ),
            ],
          ),
        ],
      ),
    );
  }
}
```

- [ ] **Step 2: Add pre_flight handling to ChatProvider**

In `flutter_app/lib/providers/chat_provider.dart`, add state:

```dart
  // Pre-flight state
  Map<String, dynamic>? preFlightData;

  void dismissPreFlight() {
    preFlightData = null;
    notifyListeners();
  }
```

Register WS listener for `pre_flight` status type — in the `_onStatusUpdate` handler, detect pre_flight:

```dart
  void _onStatusUpdate(Map<String, dynamic> msg) {
    final type = msg['type'] as String? ?? msg['status_type'] as String? ?? '';
    final text = msg['text'] as String? ?? '';

    if (type == 'pre_flight') {
      // Parse pre-flight data from status text (JSON encoded)
      try {
        preFlightData = json.decode(text) as Map<String, dynamic>;
      } catch (_) {
        preFlightData = {'goal': text, 'steps': [], 'timeout': 3};
      }
      notifyListeners();
      return;
    }

    statusText = text;
    // ... rest of existing handler
```

- [ ] **Step 3: Render PreFlightCard in chat_screen.dart**

In the message list builder, before the TypingIndicator, add:

```dart
if (chat.preFlightData != null) ...[
  PreFlightCard(
    goal: chat.preFlightData!['goal'] ?? '',
    steps: (chat.preFlightData!['steps'] as List?)
        ?.cast<Map<String, dynamic>>() ?? [],
    timeoutSeconds: chat.preFlightData!['timeout'] ?? 3,
    onCancel: () {
      chat.dismissPreFlight();
      // Send cancel to backend
      final sessions = context.read<SessionsProvider>();
      final api = context.read<ConnectionProvider>().api;
      api.post('system/cancel', {'session_id': sessions.activeSessionId ?? ''});
    },
  ),
],
```

- [ ] **Step 4: Add import and analyze**

Add import for `pre_flight_card.dart` at the top of chat_screen.dart.

Run: `cd "D:\Jarvis\jarvis complete v20\flutter_app" && flutter analyze lib/`
Expected: No issues

- [ ] **Step 5: Build**

Run: `cd "D:\Jarvis\jarvis complete v20\flutter_app" && flutter build web --release --no-tree-shake-icons`

- [ ] **Step 6: Commit**

```bash
git add flutter_app/lib/widgets/chat/pre_flight_card.dart flutter_app/lib/providers/chat_provider.dart flutter_app/lib/screens/chat_screen.dart
git commit -m "feat: PreFlightCard with countdown + cancel for plan preview in chat"
```

---

### Task 6: Add 'recovery' to editable config + Flutter

**Files:**
- Modify: `src/jarvis/config_manager.py`
- Modify: `flutter_app/lib/providers/config_provider.dart`

- [ ] **Step 1: Add to _EDITABLE_SECTIONS**

Add `"recovery"` to the `_EDITABLE_SECTIONS` frozenset in `config_manager.py`.

- [ ] **Step 2: Add to Flutter save() list + defaults()**

In `config_provider.dart`, add `'recovery'` to the save section list and add defaults:

```dart
    'recovery': {
      'pre_flight_enabled': true,
      'pre_flight_timeout_seconds': 3,
      'pre_flight_min_steps': 2,
      'correction_learning_enabled': true,
      'correction_proactive_threshold': 3,
    },
```

- [ ] **Step 3: Commit**

```bash
git add src/jarvis/config_manager.py flutter_app/lib/providers/config_provider.dart
git commit -m "feat: recovery config editable via UI"
```

---

### Task 7: Full Test Suite

- [ ] **Step 1: Run correction memory tests**

Run: `cd "D:\Jarvis\jarvis complete v20" && python -m pytest tests/unit/test_correction_memory.py -v`
Expected: All 7 PASS

- [ ] **Step 2: Run all unit tests**

Run: `cd "D:\Jarvis\jarvis complete v20" && python -m pytest tests/unit/ -v`
Expected: All PASS

- [ ] **Step 3: Ruff check**

Run: `cd "D:\Jarvis\jarvis complete v20" && python -m ruff check src/jarvis/ --select=F401,F811,F821,E501 --no-fix`
Expected: All checks passed

- [ ] **Step 4: Commit any fixes**

```bash
git add -A
git commit -m "fix: test adjustments for smart recovery system"
```
