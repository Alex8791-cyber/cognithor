# Cognithor Trace-UI — v0.95.0 Design Spec

**Status:** Draft — 2026-04-26
**Target Release:** v0.95.0
**Source Backlog:** `project_v0930_post_release_backlog.md` (Path C, Feature 5)
**Decided in Brainstorm:** 2026-04-26

---

## 1. Executive Summary

Cognithor v0.93.0 shipped `cognithor.crew` with a Hashline-Guard audit chain emitting deterministic events at every kickoff/task/guardrail boundary, but those events live only in `~/.cognithor/logs/audit.jsonl`. There is no UI for them. v0.95.0 ships **Trace-UI** — a Flutter live-debugging screen that visualises running and historical Crews against this existing audit chain.

The release is full-build (Path C / Scope A): WebSocket plumbing for live events + REST endpoints for replay/history + Flutter master-detail screens with timeline-log + per-agent stats. Owner-token gated. Single-instance (no multi-tenant).

This spec defines **five work packages (WP1–WP5)** delivered as **four PRs + Direct-Commit-on-main + tag** (v0.94.0 release-discipline pattern). Aufwand: ~10–14 Tage seriell, parallelisiert ~6–8 Tage.

---

## 2. Brainstorm-Decisions (binding)

| ID | Frage | Entscheidung |
|----|-------|--------------|
| D1 | Scope für v0.95.0 | **Vollausbau** — WS-Plumbing + REST + Flutter-Screen + History-Search |
| D2 | Multi-Crew-Sicht | **Hybrid Dashboard** — Liste aller (live+historisch) Runs + Detail-Pin auf Klick |
| D3 | Detail-View Visualisierung | **Vertical Timeline-Log + Stats-Sidebar** (mockup `timeline-layout.html` bestätigt) |
| D4 | WS-Broadcast-Pattern | **Hybrid C** — Lightweight Lifecycle-Stream fürs Dashboard + REST-Detail-Fetch + WS-Topic-Subscribe pro Trace |
| D5 | Authorization | **Owner-Only** — `require_owner(token)` gated; non-owner-Sessions sehen Tab nicht |
| D6 | AutoGen-Shim Coverage | **Mit-instrumentieren** — `cognithor.compat.autogen.AssistantAgent.run()` Trace-Emission verifiziert + ggf. gefixt |
| D7 | Performance-Bound | **Niedrig (≤10 ev/s)** — Buffer 1k pro Subscriber, drop-oldest on overflow, kein Streaming-Mode |

---

## 3. Validierte Faktenbasis

Aus dem Codebase-Audit-Brief (Sektion 4 in Brainstorm):

### 3.1 Event-Emission existiert bereits

`src/cognithor/crew/compiler.py` emittiert sieben Event-Types via `append_audit()` (line 93–124) durch `AuditTrail.record_event()`. Payload-Felder pro Event:

| Event | Felder |
|---|---|
| `crew_kickoff_started` | `trace_id`, `n_tasks`, `process` (`SEQUENTIAL`/`HIERARCHICAL`) |
| `crew_task_started` | `trace_id`, `task_id`, `agent_role` |
| `crew_task_completed` | `trace_id`, `task_id`, `duration_ms`, `tokens` |
| `crew_task_failed` | `trace_id`, `task_id`, `reason` |
| `crew_guardrail_check` | `trace_id`, `task_id`, `verdict` (`pass`/`fail`), `retry_count`, `pii_detected`, `feedback` |
| `crew_kickoff_completed` | `trace_id`, `n_tasks` |
| `crew_kickoff_failed` | `trace_id`, `reason` |

`trace_id` ist ein hex UUID4, generiert pro Kickoff (compiler.py L308, L523), als Korrelations-ID durchgereicht.

### 3.2 Persistenz

`~/.cognithor/logs/audit.jsonl`, append-only, SHA-256 Hashline-Chain (`prev_hash` + `hash` Felder). Optionale HMAC + Ed25519 Signatur. Existing `AuditTrail.query(session_id, tool, status, since, limit)` API für REST-Reads.

### 3.3 Was fehlt

- WebSocket-Broadcast — Events werden **nur** in JSONL geschrieben, nicht über WS verteilt.
- REST-Endpoints — keine `/api/crew/traces`, `/api/crew/trace/{id}`.
- Flutter Trace-Screen — `monitoring_screen.dart` ist closest analogue (REST-polling), kein Live-Crew-View.
- Owner-Gating auf Audit — aktuell darf jede authentifizierte Flutter-Session alles lesen.
- AutoGen-Shim-Trace-Coverage — `cognithor.compat.autogen.AssistantAgent.run()` muss verifiziert werden (sollte transparent emitten, da es durch `cognithor.crew.Crew` läuft, aber Tests fehlen).

---

## 4. Cognithor-Kontext (post-v0.94.1)

**Repo State nach v0.94.1:** `Alex8791-cyber/cognithor` auf `main` mit:

- v0.94.x ausgeliefert (5 WPs + 1 Hotfix), PyPI `cognithor==0.94.1`
- `cognithor.crew` Production Multi-Agent-API mit Audit-Chain bereits vorhanden
- `cognithor.compat.autogen` Source-Compat-Shim aktiv, routet 1-shot-path durch `Crew.kickoff_async()`
- Flutter Command Center 33 screens, Material 3 Dark Theme, Provider-State, WebSocket-Service
- cognithor.ai Site mit `/integrations` + `/quickstart` (DE+EN) live (Vercel Auto-Deploy)

**Hardware-Baseline:** RTX 5090, Ryzen 9 9950X3D, Windows 11, Ollama lokal.

---

## 5. Gesamtziel und Nicht-Ziele

### 5.1 Gesamtziel

Cognithor-Operator (Owner) bekommt eine **Live-Visibility-Schicht** auf laufende und historische Crew-Kickoffs:

1. Was passiert *jetzt* in einer laufenden Crew (welcher Agent, welcher Task, welche Tokens, welche Guardrail-Verdicts)
2. Was ist *passiert* in einem abgeschlossenen Run (vollständige Replay-History)
3. Welche Runs gibt es überhaupt (Dashboard mit Filter/Search)

Sichtbar zu machen: PGE-Trinity in Action — Hashline-Guard als Marketing-Story.

### 5.2 Nicht-Ziele

- **Kein** Multi-User / Multi-Tenant — Owner-Single-User-Modell
- **Keine** Editier- oder Eingriffsmöglichkeit (read-only Visibility)
- **Kein** Streaming-Mode (Token-für-Token-Events) — nur Boundary-Events; Streaming wäre v0.96.0
- **Keine** Performance-Benchmarks gegenüber AutoGen UI / Magentic-One UI
- **Keine** Hashline-Guard-Verifikation im UI (eigener CLI-Command `cognithor audit verify` existiert; Trace-UI surfaced nur Roh-Events)
- **Keine** Custom-Dashboards / Charting (KPI-Aggregation post-v0.95.0)
- **Keine** SQLite/Postgres-Migration der Audit-Storage — JSONL bleibt

---

## 6. Globale Randbedingungen

- **Python 3.12+**, Type Hints vollständig, Pydantic v2, `ruff` + `mypy --strict` müssen grün sein
- **Apache 2.0** bleibt Lizenz
- **Conventional Commits**: `feat:`, `docs:`, `test:`, `refactor:`, `chore:`, `fix:`, `style:`
- **Branching:** Jeder PR auf `feat/cognithor-trace-vN-<kurzname>` Branch, einzelne PRs gegen `main`. Cleanup nach Merge in **separater Turn** (Memory `feedback_pr_merge_never_chain_cleanup.md`)
- **No new mandatory runtime deps** — vorhandene `fastapi`, `pydantic`, `structlog` reichen für Backend; `provider` + bestehende WebSocket-Infrastruktur reichen für Flutter
- **Test coverage floors:** ≥89% Repo total (CI-Gate), ≥85% auf neue Backend-Module, ≥80% auf neue Flutter-Widgets
- **Owner-Gating Default:** `COGNITHOR_OWNER_USER_ID` env var; Default-Read aus `pyproject.toml` `[project] authors[0].name` falls env unset
- **Pre-WP1 Gate:** 24h v0.94.1-Stabilität ohne v0.94.2-Hotfix-Bedarf bevor PR 1 startet

---

## 7. Arbeitspaket-Übersicht

| WP  | Titel                                       | PR  | Branch                                   | Tasks | Aufwand   |
|-----|---------------------------------------------|-----|------------------------------------------|-------|-----------|
| WP1 | TraceBus + Owner-Gating                     | PR1 | `feat/cognithor-trace-v1-bus`            | 8     | 1.5 Tage  |
| WP2 | REST API for Traces                         | PR2 | `feat/cognithor-trace-v2-api`            | 10    | 2 Tage    |
| WP3 | WebSocket Live-Stream (lifecycle + topic)   | PR3 | `feat/cognithor-trace-v3-ws`             | 8     | 1.5 Tage  |
| WP4 | Flutter Trace-Screen (List + Detail)        | PR4 | `feat/cognithor-trace-v4-flutter`        | 16    | 4–5 Tage  |
| WP5 | AutoGen-Shim Trace-Coverage Verification    | PR4 | (gleiche PR)                             | 3     | 0.5 Tage  |
| —   | v0.95.0 Release-Bundle                      | DC* | `main` (direct commit + tag)             | 5     | 1 Tag     |

*DC = Direct Commit auf main (kein PR), pattern aus v0.93.0 + v0.94.0 reused.

**Total:** ~50 Tasks across 4 PRs + 1 Direct-Commit. Reihenfolge: PR1 → PR2 → PR3 → PR4 → Direct-Commit → Tag-Push.

**Aufwand seriell:** ~10–11 Tage. **Mit Subagent-Parallelisierung:** ~6–8 Tage realistisch.

---

## 8. WP-Details

### 8.1 WP1 — TraceBus + Owner-Gating

**Ziel:** In-process Pub/Sub-Bus, der von `compiler.append_audit()` befüllt wird; routet Lifecycle-Events an alle Owner-Sessions, Topic-Events an spezifische Subscriber. Plus Owner-Identifikation.

**Deliverables:**

```
src/cognithor/
├── crew/
│   └── trace_bus.py                    # NEW — TraceBus singleton
├── crew/
│   └── compiler.py                     # MODIFIED — 1 line in append_audit()
├── security/
│   └── owner.py                        # NEW — require_owner() + identification
└── tests/
    ├── test_crew/test_trace_bus.py     # NEW — pubsub unit tests
    └── test_security/test_owner.py     # NEW — owner-token tests
```

**`TraceBus` API (key signatures):**

```python
class TraceBus:
    """In-process pub/sub for crew audit events."""
    
    def publish(self, record: dict) -> None: ...
    def subscribe(self, topic: str, queue: asyncio.Queue) -> SubscriptionHandle: ...
    def subscribe_lifecycle(self, queue: asyncio.Queue) -> SubscriptionHandle: ...
    def unsubscribe(self, handle: SubscriptionHandle) -> None: ...

def get_trace_bus() -> TraceBus:
    """Process-wide singleton accessor."""
```

Backpressure: per-subscriber `asyncio.Queue(maxsize=1000)`. On overflow → `get_nowait` (drop oldest), `put_nowait` (insert newest), increment dropped-counter, rate-limited warn-log.

**`require_owner` API:**

```python
def require_owner(bootstrap_token: BootstrapToken) -> None:
    """Raises HTTPException(403) if token is not the owner.
    
    Owner = COGNITHOR_OWNER_USER_ID env var, or
            pyproject.toml [project] authors[0].name as fallback.
    """
```

**Acceptance Criteria:**
- [ ] `TraceBus.publish` <1ms im Hot-Path (assert in benchmark test)
- [ ] Subscriber-Queue overflow droppt oldest + emits one log per minute
- [ ] `compiler.append_audit()` ruft `get_trace_bus().publish(record)` AFTER `record_event()` (JSONL persistence garantiert)
- [ ] Lifecycle-only events (`crew_kickoff_*`) gehen an Lifecycle-Subscribers; non-lifecycle gehen an Topic-Subscribers
- [ ] `require_owner` returnt OK für Owner-Token, raised 403 für andere
- [ ] Coverage ≥85% auf `trace_bus.py` + `owner.py`

---

### 8.2 WP2 — REST API for Traces

**Ziel:** FastAPI-Router mit drei Endpoints zum Reading der historisch-persistierten Audit-Chain. Reuses `AuditTrail.query()` als Source.

**Deliverables:**

```
src/cognithor/
├── api/
│   ├── __init__.py                     # MODIFIED — router mount
│   └── crew_traces.py                  # NEW — endpoints
├── tests/
│   └── test_api/test_crew_traces.py    # NEW — endpoint tests
└── docs/
    └── api/crew-traces.md              # NEW — endpoint reference
```

**Endpoints:**

```
GET /api/crew/traces?status=&since=&limit=
  → Returns: [{trace_id, crew_label, status, started_at, ended_at, 
               duration_ms, agent_count, total_tokens, n_tasks, n_failed_guardrails}]
  Owner-only.

GET /api/crew/trace/{trace_id}
  → Returns: {trace_id, status, events: [...], meta: {skipped_lines: 0}}
  Owner-only. 404 if trace_id unknown.

GET /api/crew/trace/{trace_id}/stats
  → Returns: {total_tokens, total_duration_ms, agent_breakdown: {role: tokens},
              guardrail_summary: {pass: N, fail: N, retries: N}}
  Owner-only. 404 if unknown.
```

**Performance:** All queries scan JSONL once per request. For ~10k events JSONL (~10 MB), parse ≤200ms — acceptable for v0.95.0. SQLite-backed index post-v0.95.0 if needed.

**Acceptance Criteria:**
- [ ] `GET /api/crew/traces` returns sorted (newest first) trace list with derived status
- [ ] `GET /api/crew/trace/{id}` returns full event list, ordered by timestamp
- [ ] `GET /api/crew/trace/{id}/stats` returns derived aggregates
- [ ] Corrupt JSONL line → `meta.skipped_lines` incremented, line skipped, response still valid
- [ ] All endpoints 403 for non-owner
- [ ] OpenAPI schema auto-generated by FastAPI is correct
- [ ] Coverage ≥85%

---

### 8.3 WP3 — WebSocket Live-Stream

**Ziel:** Bestehenden Channel um zwei neue Message-Types erweitern. Owner-Session bekommt automatisch Lifecycle-Stream beim Connect; Topic-Subscribe ist explicit.

**Deliverables:**

```
src/cognithor/
├── channels/
│   └── webui.py                        # MODIFIED — add subscribe handlers
└── tests/
    └── test_channels/test_trace_ws.py  # NEW — WS integration tests
```

**New WS message types (server → client):**

```json
{"type": "crew_lifecycle", 
 "event": "crew_kickoff_started",
 "trace_id": "abc...", 
 "crew_label": "research-summarize", 
 "n_tasks": 4, 
 "started_at": "2026-04-26T..."}

{"type": "crew_event",
 "trace_id": "abc...",
 "event_type": "crew_task_completed",
 "task_id": "t-1",
 "agent_role": "researcher",
 "duration_ms": 4810,
 "tokens": 1234,
 "timestamp": "..."}
```

**New WS message types (client → server):**

```json
{"type": "crew_lifecycle_subscribe"}        # opt-in once per session
{"type": "crew_subscribe", "trace_id": "abc..."}
{"type": "crew_unsubscribe", "trace_id": "abc..."}
```

**Auto-cleanup:** WS disconnect → all topic-subscriptions for that session removed via `TraceBus.unsubscribe_all(session)`.

**Acceptance Criteria:**
- [ ] Owner-Session bekommt Lifecycle-Events nach `crew_lifecycle_subscribe`
- [ ] Topic-Subscribe routet Events nur an passende Sessions
- [ ] Non-owner WS-`crew_subscribe` → server schickt `{type: "error", code: "owner_only"}` und ignoriert
- [ ] Disconnect cleanup verified (no leak in TraceBus subscribers)
- [ ] Backpressure: slow client triggert drop-oldest; JSONL bleibt vollständig
- [ ] Coverage ≥85%

---

### 8.4 WP4 — Flutter Trace-Screen

**Ziel:** Master-Detail-View mit `TraceListScreen` (Dashboard) + `TraceDetailScreen` (Timeline+Sidebar). Owner-only sichtbar.

**Deliverables:**

```
flutter_app/lib/
├── models/
│   └── crew_trace.dart                 # NEW — CrewTraceMeta, CrewEvent, CrewTraceStats
├── services/
│   ├── trace_service.dart              # NEW — REST + WS API client
│   └── websocket_service.dart          # MODIFIED — add WsType.crewLifecycle, crewEvent
├── providers/
│   └── trace_provider.dart             # NEW — ChangeNotifier
├── screens/
│   ├── main_shell.dart                 # MODIFIED — owner-conditional Trace tab
│   └── trace/
│       ├── trace_list_screen.dart      # NEW — Dashboard with status pills
│       ├── trace_detail_screen.dart    # NEW — Timeline + Stats Sidebar
│       └── widgets/
│           ├── event_row.dart          # NEW — single event row
│           ├── stats_sidebar.dart      # NEW — derived stats panel
│           └── trace_card.dart         # NEW — list-row card
├── i18n/
│   ├── app_en.arb                      # MODIFIED — new keys
│   └── app_de.arb                      # MODIFIED — new keys
└── test/
    ├── screens/trace/trace_list_screen_test.dart   # NEW
    ├── screens/trace/trace_detail_screen_test.dart # NEW
    └── widgets/event_row_test.dart                 # NEW (golden-tests)
```

**TraceListScreen Layout:**

- Material 3 ListView, jede Card zeigt:
  - Status-Pill (running/done/failed) mit pulsierendem Indikator wenn running
  - `crew_label` als Title
  - `trace_id[:8]` + ISO-Timestamp als Subtitle
  - Right side: agent count icon + total tokens chip
- Filter chips top: All / Running / Failed (this hour) / Last 24h
- WebSocket auto-update: neue Crews erscheinen oben, Status-Pills updaten live

**TraceDetailScreen Layout** (per `timeline-layout.html` mockup):

- Header bar: trace_id, crew_label, process-type, status pulsing-pill, elapsed-counter
- 2-Spalten-Body:
  - Left (flex 1): Timeline-Log, monospace, `event_row` per Event
  - Right (280px): Stats-Sidebar mit Elapsed / Tokens / Guardrails / Per-Agent breakdown
- Auto-scroll-to-bottom on new event (with "scroll-lock" toggle if user scrolls up)

**Acceptance Criteria:**
- [ ] Trace-Tab erscheint in main_shell nur für Owner-Token
- [ ] TraceListScreen rendert REST-Daten + auto-updates via WS
- [ ] Click auf Card → Navigator.push zu TraceDetailScreen mit `trace_id`
- [ ] TraceDetailScreen lädt initialen State via REST + abonniert WS-Topic
- [ ] Live-Update: neuer Event erscheint < 100ms nach Backend-Emit
- [ ] Reconnect: WS-Abriss → Re-fetch via REST + Re-subscribe → keine Daten-Lücke
- [ ] Empty-State Card für 404 trace_id
- [ ] DE+EN i18n-Strings vollständig
- [ ] Coverage ≥80% on new widgets/services

---

### 8.5 WP5 — AutoGen-Shim Trace-Coverage Verification

**Ziel:** Verifizieren dass `cognithor.compat.autogen.AssistantAgent.run()` transparent crew_*-Events emittiert. Falls nein: minimal fix.

**Deliverables:**

```
tests/test_compat/test_autogen/
└── test_trace_emission.py              # NEW — integration tests
```

(Plus ggf. Fix in `src/cognithor/compat/autogen/_bridge.py` falls Verifikation einen Gap findet.)

**Test:**

```python
@pytest.mark.asyncio
async def test_assistant_agent_run_emits_crew_kickoff_events():
    """compat.autogen.AssistantAgent.run() should emit crew_* events transparently."""
    captured: list[dict] = []
    bus = get_trace_bus()
    handle = bus.subscribe_lifecycle(callback=captured.append)
    
    agent = AssistantAgent(name="test", model_client=MockClient())
    await agent.run(task="hello")
    
    bus.unsubscribe(handle)
    event_types = [e["event_type"] for e in captured]
    assert "crew_kickoff_started" in event_types
    assert "crew_kickoff_completed" in event_types
```

Plus 2 weitere Tests: `test_run_stream_emits_events`, `test_round_robin_emits_per_round`.

**Acceptance Criteria:**
- [ ] All 3 Tests grün (= Verifikation erfolgreich)
- [ ] Falls ein Test rot: minimal fix in `_bridge.py` ergänzt; Test wird grün
- [ ] AutoGen-Shim-Run erscheint in TraceListScreen als normaler Crew-Run mit `crew_label = "compat-autogen-{name}"` (oder ähnlich)

---

### 8.6 v0.95.0 Release-Bundle (Direct Commit + Tag, kein PR)

**Pattern reuse:** Identisch zu v0.94.0 release-discipline.

**Step 1 — Version-Bump-Commit auf main (5 files):**

```
+ pyproject.toml                                        version = "0.95.0"
+ src/cognithor/__init__.py                             __version__ = "0.95.0"
+ flutter_app/pubspec.yaml                              version: 0.95.0+1
+ flutter_app/lib/providers/connection_provider.dart    kFrontendVersion = '0.95.0'
+ CHANGELOG.md                                          [Unreleased] → [0.95.0] -- 2026-MM-DD
```

Plus: `README.md` Highlights bullet für Trace-UI v0.95.0.

**Step 2 — Tag + Push:**

```bash
git tag -a v0.95.0 -m "Cognithor v0.95.0 — Trace-UI"
git push origin v0.95.0
```

**Step 3 — Workflow-Triggers (parallel via REST API workflow_dispatch):**

- `publish.yml` (PyPI auto-publish)
- `build-windows-installer.yml`
- `build-deb.yml`
- `build-mobile.yml`
- `build-flutter-web.yml`

**Step 4 — Verify:**

- PyPI: `https://pypi.org/project/cognithor/0.95.0/` live
- GitHub Release: 6 platform artifacts attached
- Trace-UI screenshot ins Release-Body (curated, wie v0.94.0+v0.94.1)
- cognithor.ai changelog auto-syncs (1h ISR)

---

## 9. Test-Strategie

| WP | Test-Location | Coverage-Target | Spezial |
|----|--------------|-----------------|---------|
| WP1 | `tests/test_crew/test_trace_bus.py` + `tests/test_security/test_owner.py` | ≥85% | Backpressure stress test |
| WP2 | `tests/test_api/test_crew_traces.py` | ≥85% | Corrupt JSONL fixture |
| WP3 | `tests/test_channels/test_trace_ws.py` | ≥85% | Disconnect cleanup |
| WP4 | `flutter_app/test/screens/trace/` + `integration_test/trace_flow_test.dart` | ≥80% | Golden-tests für event_row icons |
| WP5 | `tests/test_compat/test_autogen/test_trace_emission.py` | n/a (additive) | 3 integration tests |

**Pre-PR-Closeout Template** (analog v0.94.x):
- `pytest tests/ -x -q --cov=src/cognithor --cov-fail-under=89`
- `ruff check .` + `ruff format --check .`
- `mypy --strict src/cognithor/{crew,api,security,channels}` (WP1-3)
- Flutter: `flutter test` + `flutter test integration_test/`
- Push + open PR + wait CI green + squash-merge + (separate turn) cleanup

---

## 10. Acceptance Criteria — Gesamt

v0.95.0 ist released wenn:

- [ ] Alle 4 PRs gemergt + cleanup done
- [ ] Direct-Commit-on-main mit Version-Bump in allen 5 Files
- [ ] Tag `v0.95.0` gepusht, alle 5 Release-Workflows grün
- [ ] PyPI hat `cognithor==0.95.0` (wheel + sdist)
- [ ] GitHub Release hat 6 fresh Artifacts (Win Installer, Launcher, Linux .deb, APK, IPA, Flutter Web)
- [ ] Keine stale Artifacts auf der Release
- [ ] `pip install cognithor==0.95.0` works
- [ ] Owner-Token: Trace-Tab erscheint in Flutter, listet aktuelle + historische Crews
- [ ] Live-Test: `cognithor` CLI startet eine Crew → Trace-UI zeigt sie live (Eyeball-Test mit Ollama)
- [ ] Detail-View zeigt korrekt: Events, Token-Counts, Guardrail-Verdicts, Per-Agent Breakdown
- [ ] Reconnect-Test: WS abreißen → wenn Crew weiter läuft, fängt Re-Connect die Events korrekt ab
- [ ] Non-Owner-Token: Trace-Tab unsichtbar; REST 403; WS subscribe-error
- [ ] AutoGen-Shim-Run erscheint im Dashboard
- [ ] `NOTICE` unverändert (kein neues third-party concept)
- [ ] CHANGELOG `[0.95.0]` Section hat alle 5 WPs gelistet

---

## 11. Sequencing & Dependencies

```
v0.94.1 (released 2026-04-26)
    ↓
[Pre-WP1 Gate: 24h v0.94.1-Stabilität ohne v0.94.2-Hotfix]
    ↓
PR 1 (WP1 TraceBus + Owner) — foundation, blocks all
    ↓
PR 2 (WP2 REST API) — depends on WP1's TraceBus + owner
    ↓
PR 3 (WP3 WebSocket) — depends on WP1's TraceBus + owner
    ↓
PR 4 (WP4 Flutter + WP5 AutoGen-Shim) — depends on WP2 + WP3
    ↓
Direct-Commit on main (version bump v0.94.1 → v0.95.0)
    ↓
Tag v0.95.0 + push → Release-Workflows
    ↓
v0.95.0 LIVE
```

**Hard Dependencies:**

- PR 2 (REST) braucht `require_owner` aus WP1
- PR 3 (WS) braucht `TraceBus` + `require_owner` aus WP1
- PR 4 (Flutter) braucht REST-Endpoints (PR 2) + WS-Messages (PR 3)
- WP5 (AutoGen-Shim Verification) ist additive in PR 4 — keine Cross-Dep

**Soft Dependencies (parallel-ok):**

- PR 2 + PR 3 könnten theoretisch parallelisiert werden nach PR 1 merge (verschiedene Files), aber Reviewer-Bandwidth bevorzugt sequentiell.

---

## 12. Out-of-Scope (für v0.95.0)

- Multi-User / Tenant-Isolation
- Editier-/Replay-/Restart-Funktionen
- Streaming Token-Events
- KPI-Charts / Dashboards / Trends
- SQLite-Migration der Audit-Storage
- Hashline-Chain-Verification im UI (separater CLI)
- Custom Color-Coding pro Crew-Label
- Export als Markdown/PDF (post-v0.95.0)
- Mobile-optimiertes Layout (Flutter ist desktop-first; mobile rendert aber funktional)

---

## 13. References

- v0.94.0 Spec (Pattern reused): `docs/superpowers/specs/2026-04-25-cognithor-autogen-strategy-design.md`
- v0.94.0 Plan: `docs/superpowers/plans/2026-04-25-cognithor-autogen-strategy.md`
- v0.94.0 Release-Pattern: commits `26d7cba0` → `c2f8b43d` → `672bca9a` → `ad983fb4` → `3b8ffa24` → `0abb9d26`
- v0.94.1 Hotfix: commit `3b71a796`
- v0.94.0 Audit-Report (uncovered the wiring gaps): `docs/superpowers/reports/2026-04-26-v094-dry-run-audit.md`
- Crew compiler event emission: `src/cognithor/crew/compiler.py:93-124, 308-362, 523-642`
- AuditTrail storage: `src/cognithor/security/audit.py:107-272`
- Existing WebSocket service: `flutter_app/lib/services/websocket_service.dart:24-58, 70-354`
- Material 3 theme: `flutter_app/lib/theme/cognithor_theme.dart`
- Mockup confirmed: `.superpowers/brainstorm/36095-1777212291/content/timeline-layout.html`

---

**End of Spec.**
