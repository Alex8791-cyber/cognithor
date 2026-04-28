# Stale-Module Triage — 2026-04-27

Audit of 14 backend modules under `src/cognithor/` with no commits since 2026-04-12. Performed during the v0.95.0 post-release wartung session as part of the comprehensive Cognithor audit.

## Summary

| Category | Count | Modules |
|---|---|---|
| **KEEP-ACTIVE** | 9 | `aacs`, `forensics`, `governance`, `graph`, `hashline`, `hitl`, `kanban`, `osint`, `system` |
| **KEEP-DOCUMENTED** | 3 | `documents`, `proactive`, `sdk` (all stability-docstring'd in `4573dc21`) |
| **VERIFY-WIRING** | 1 | `benchmark` |
| **ARCHIVED / DELETED** | 1 | ~~`ui`~~ (deleted PR #162) |

> **Update 2026-04-27 evening:** `ui` deleted, KEEP-DOCUMENTED stability docstrings written, ARCHITECTURE.md gains HITL + Forensics sections, false-positive TODO counts in `kanban/` + `proactive/` corrected. See per-module entries for trail.

## Per-module findings

### `aacs` — **KEEP-ACTIVE**

- **Purpose:** Advanced analytics & compliance suite — internal security/auth pipeline integration.
- **External import count:** 0 outside the module (security layer integrated within own boundary).
- **Test coverage:** 11 test refs.
- **Recommendation:** No action. Module is stable and integrated within the security layer.

### `benchmark` — **VERIFY-WIRING** → see [2026-04-28 archive recommendation](2026-04-28-benchmark-archive-recommendation.md)

- **Purpose:** Performance benchmarking & profiling utilities (2 files, 863 LOC).
- **External import count:** 0 (only test refs).
- **Test coverage:** 1 test ref.
- **Runtime relevance:** None — not wired into CLI / gateway / channels.
- **Recommendation:** Archive. The canonical benchmark home is the top-level `cognithor_bench/` package (own `pyproject.toml`, `cognithor-bench` CLI, Cognithor + AutoGen adapters). The internal `src/cognithor/benchmark/` is a parallel implementation with zero callers. See the linked recommendation doc for details + decision options.

### `documents` — **KEEP-DOCUMENTED**

- **Purpose:** Document parsing/chunking + Typst template wrapper (2 files, 215 LOC).
- **External import count:** 1 (used by MCP media tools).
- **Test coverage:** 1 test ref.
- **Recommendation:** Stability docstring added in commit `4573dc21`.

### `forensics` — **KEEP-ACTIVE**

- **Purpose:** Incident forensics, execution replay, post-mortem log analysis (3 files, 639 LOC).
- **External import count:** 2.
- **Test coverage:** 12 test refs.
- **Recommendation:** No code action. Consider a brief mention in `docs/ARCHITECTURE.md` so the replay capability is discoverable.

### `governance` — **KEEP-ACTIVE**

- **Purpose:** Policy engine, decision audit, governance rules (4 files, 795 LOC).
- **External import count:** 4 (integrated with cron + audit).
- **Test coverage:** 14 test refs.
- **Recommendation:** No action.

### `graph` — **KEEP-ACTIVE**

- **Purpose:** Knowledge graph, entity relations, semantic indexing (6 files, 2 207 LOC).
- **External import count:** 16 (heaviest non-self consumer in the stale set).
- **Test coverage:** 11 test refs; 78 doc refs.
- **Recommendation:** No action — actively imported by orchestration code.

### `hashline` — **KEEP-ACTIVE**

- **Purpose:** Session hash chains, integrity verification — also underpins safe-file-edit guard for MCP. (12 files, 2 013 LOC.)
- **External import count:** 4 (heavy use across MCP).
- **Test coverage:** 44 test refs.
- **Recommendation:** No action. Critical security primitive.

### `hitl` — **KEEP-ACTIVE**

- **Purpose:** Human-in-the-loop approval workflows (5 files, 1 447 LOC).
- **External import count:** 3.
- **Test coverage:** 8 test refs; 0 doc refs.
- **Recommendation:** Add a section to `docs/ARCHITECTURE.md` describing the HITL approval flow — currently undocumented despite live wiring.

### `kanban` — **KEEP-ACTIVE**

- **Purpose:** Kanban board state machine + workflows (6 files, 956 LOC).
- **External import count:** 8.
- **Test coverage:** 13 test refs; 133 doc refs.
- **Recommendation:** No archival, no follow-up needed. The "7 TODOs" reported in the original audit were a false positive — they were enum members named `TaskStatus.TODO` and references to it (e.g. `TaskStatus.TODO: {TaskStatus.IN_PROGRESS, ...}`), not `# TODO:` work-marker comments.

### `osint` — **KEEP-ACTIVE**

- **Purpose:** Open-source intelligence gathering, data enrichment (16 files, 1 234 LOC).
- **External import count:** 3.
- **Test coverage:** 13 test refs; 159 doc refs.
- **Recommendation:** No action.

### `proactive` — **KEEP-DOCUMENTED**

- **Purpose:** Event-driven heartbeat, autonomous task scheduling (1 file, 669 LOC). Single-file, feature-complete.
- **External import count:** 2 (used by cron engine).
- **Test coverage:** 3 test refs.
- **Recommendation:** Stability docstring added in commit `4573dc21`. The "2 TODOs" reported in the original audit were enum members (`EventType.TODO_REMINDER`) and references to it, not work-marker comments.

### `sdk` — **KEEP-DOCUMENTED**

- **Purpose:** Developer-facing decorators + scaffolding helpers (5 files, 649 LOC).
- **External import count:** 0 in production (decorator-only — developer-time API).
- **Test coverage:** 10 test refs.
- **Recommendation:** Stability docstring added in commit `4573dc21`.

### `system` — **KEEP-ACTIVE**

- **Purpose:** System-level utilities, kernel integration, resource monitoring (3 files, 569 LOC).
- **External import count:** 4.
- **Test coverage:** 17 test refs; 230 doc refs (highest doc density in the stale set).
- **Recommendation:** No action.

### `ui` — ~~VERIFY-WIRING~~ → **DELETED** (PR #162, commit `c95cd729`)

User confirmed deprecation. The package + its single experimental file
(`session_manager.py` — a 6th-tier Core Memory + SQLite persistence
experiment that was never wired in) plus the only consumer
(`tests/test_v036/test_sessions.py`) were removed. Full backend regression
green post-deletion (14 454 passed).

## Next-action checklist

- [x] ~~Add stability docstrings to `documents/__init__.py`, `proactive/__init__.py`, `sdk/__init__.py`~~ — done (`4573dc21`).
- [x] ~~Resolve `kanban/` (7 TODOs) and `proactive/` (2 TODOs) hot-spots~~ — false positive, the "TODOs" were enum members named `TODO`, not work markers.
- [x] ~~Decide `ui/`~~ — deleted (PR #162).
- [x] ~~Add brief sections to `docs/ARCHITECTURE.md` for `forensics/` and `hitl/`~~ — done (`adcd6314`).
- [ ] Decide `benchmark/`: archive vs. keep-and-wire — see [2026-04-28 archive recommendation](2026-04-28-benchmark-archive-recommendation.md). Recommended: archive. Pending User confirmation.

## Notes

- "Stale" here means *no commit since 2026-04-12*, which captures all modules untouched during the v0.93/0.94/0.95 release wave. It does NOT imply "abandoned" — most are simply feature-complete.
- The triage was produced via static import-graph analysis, not runtime tracing. A module with zero static imports could still be loaded dynamically (entry points, registry lookups). Treat VERIFY-WIRING as "look once before deciding".
- This report should be reviewed within 30 days; otherwise it becomes itself stale.
