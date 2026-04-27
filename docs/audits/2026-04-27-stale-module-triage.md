# Stale-Module Triage — 2026-04-27

Audit of 14 backend modules under `src/cognithor/` with no commits since 2026-04-12. Performed during the v0.95.0 post-release wartung session as part of the comprehensive Cognithor audit.

## Summary

| Category | Count | Modules |
|---|---|---|
| **KEEP-ACTIVE** | 9 | `aacs`, `forensics`, `governance`, `graph`, `hashline`, `hitl`, `kanban`, `osint`, `system` |
| **KEEP-DOCUMENTED** | 3 | `documents`, `proactive`, `sdk` |
| **VERIFY-WIRING** | 2 | `benchmark`, `ui` |
| **ARCHIVE-CANDIDATE** | 0 | — |

**No outright dead code.** Every stale module either has live production imports or carries an unanswered question.

## Per-module findings

### `aacs` — **KEEP-ACTIVE**

- **Purpose:** Advanced analytics & compliance suite — internal security/auth pipeline integration.
- **External import count:** 0 outside the module (security layer integrated within own boundary).
- **Test coverage:** 11 test refs.
- **Recommendation:** No action. Module is stable and integrated within the security layer.

### `benchmark` — **VERIFY-WIRING**

- **Purpose:** Performance benchmarking & profiling utilities (2 files, 863 LOC).
- **External import count:** 0 (only test refs).
- **Test coverage:** 1 test ref.
- **Runtime relevance:** Mentioned in CLI flags but not instantiated at runtime.
- **Question for the user:** *Archive now, or keep seeded for future ARC-AGI-3 integration?* If kept, document the intended consumer.

### `documents` — **KEEP-DOCUMENTED**

- **Purpose:** Document parsing/chunking + Typst template wrapper (2 files, 215 LOC).
- **External import count:** 1 (used by MCP media tools).
- **Test coverage:** 1 test ref.
- **Recommendation:** Add to `src/cognithor/documents/__init__.py`:
  > *"Stable Typst-template wrapper used by `cognithor.mcp.media`. Feature-complete; expand only when adding new template formats."*

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

- **Purpose:** Kanban board state machine + workflows (6 files, 956 LOC, 7 TODOs — highest TODO density in the stale set).
- **External import count:** 8.
- **Test coverage:** 13 test refs; 133 doc refs.
- **Recommendation:** No archival. **Resolve the 7 TODOs in `kanban/models.py` and `kanban/engine.py`** as a separate cleanup task — likely unblocks blocked work.

### `osint` — **KEEP-ACTIVE**

- **Purpose:** Open-source intelligence gathering, data enrichment (16 files, 1 234 LOC).
- **External import count:** 3.
- **Test coverage:** 13 test refs; 159 doc refs.
- **Recommendation:** No action.

### `proactive` — **KEEP-DOCUMENTED**

- **Purpose:** Event-driven heartbeat, autonomous task scheduling (1 file, 669 LOC). Single-file, feature-complete.
- **External import count:** 2 (used by cron engine).
- **Test coverage:** 3 test refs; 2 TODOs in `__init__.py`.
- **Recommendation:** Resolve the 2 TODOs (probably trivial), then add stability docstring:
  > *"Feature-complete. Used by `cognithor.cron.engine`. Expand only when adding new heartbeat strategies."*

### `sdk` — **KEEP-DOCUMENTED**

- **Purpose:** Developer-facing decorators + scaffolding helpers (5 files, 649 LOC).
- **External import count:** 0 in production (decorator-only — developer-time API).
- **Test coverage:** 10 test refs.
- **Recommendation:** Add docstring to `src/cognithor/sdk/__init__.py`:
  > *"Public developer SDK — `@tool`, `@skill`, scaffolding helpers. Used at definition time, not runtime. Stable API; bump major when changing decorator signatures."*

### `system` — **KEEP-ACTIVE**

- **Purpose:** System-level utilities, kernel integration, resource monitoring (3 files, 569 LOC).
- **External import count:** 4.
- **Test coverage:** 17 test refs; 230 doc refs (highest doc density in the stale set).
- **Recommendation:** No action.

### `ui` — **VERIFY-WIRING**

- **Purpose:** Near-empty package (`__init__.py` = `# Jarvis UI utilities`). 2 files, 168 LOC.
- **External import count:** 0 in production.
- **Test coverage:** 1 test ref.
- **Runtime relevance:** None detectable.
- **Questions for the user:**
  1. Is this a deprecated stub from the pre-Cognithor rebrand era?
  2. Should it host shared UI helper functions, or be removed?
  3. The single test file — should it migrate to `flutter_app/test/` or be deleted?

## Next-action checklist

- [ ] Add stability docstrings to `documents/__init__.py`, `proactive/__init__.py`, `sdk/__init__.py` (KEEP-DOCUMENTED).
- [ ] Resolve `kanban/` (7 TODOs) and `proactive/` (2 TODOs) hot-spots.
- [ ] Decide `benchmark/`: archive vs. keep-for-ARC-AGI-3.
- [ ] Decide `ui/`: stub for shared helpers, deprecated remnant, or delete?
- [ ] Add brief sections to `docs/ARCHITECTURE.md` for `forensics/` (replay capability) and `hitl/` (approval flow).

## Notes

- "Stale" here means *no commit since 2026-04-12*, which captures all modules untouched during the v0.93/0.94/0.95 release wave. It does NOT imply "abandoned" — most are simply feature-complete.
- The triage was produced via static import-graph analysis, not runtime tracing. A module with zero static imports could still be loaded dynamically (entry points, registry lookups). Treat VERIFY-WIRING as "look once before deciding".
- This report should be reviewed within 30 days; otherwise it becomes itself stale.
