# Evolution Engine: Content Quality Gate + ATL Auto-Persist

**Date:** 2026-03-31
**Status:** Approved
**Scope:** `knowledge_builder.py`, `loop.py` (ATL)

---

## Problem Statement

Two related bugs in the Evolution Engine:

1. **Deep Learner PDF Artifacts:** `web_fetch` / trafilatura sometimes extracts PDF metadata or boilerplate instead of real content. This produces garbage entities ("PDF", "Object", "FlateDecode"), worthless knowledge claims, and quality test scores of 0.00 despite 70+ chunks collected. The entity filter catches garbage entities but chunks and vault notes are created regardless.

2. **ATL 0% Progress:** The Autonomous Thinking Loop calls `search_and_read` repeatedly (90+ cycles) but never persists results. The tool return value is discarded. All 5 goals show 0% progress despite continuous research activity.

---

## Solution Overview

Two changes that reinforce each other:

1. **Content Quality Gate** in `KnowledgeBuilder.build()` — rejects garbage content before the triple-write pipeline (vault + chunks + entities) runs.

2. **ATL Auto-Persist** in `loop.py` — after each successful `search_and_read`, synthesizes key findings via LLM and writes them through `KnowledgeBuilder.build()` into the goal-scoped index.

The quality gate protects both the Deep Learner and ATL paths from indexing garbage.

---

## Design

### 1. Content Quality Gate

**File:** `src/jarvis/evolution/knowledge_builder.py`

**New function:** `_is_usable_content(text: str, min_chars: int = 200) -> tuple[bool, str]`

Checks:
- Minimum length: 200 chars after whitespace normalization
- PDF artifact ratio: if >30% of lines match PDF structural patterns (`/Type`, `/Filter`, `endobj`, `xref`, `FlateDecode`, `MediaBox`, `stream`, `%%EOF`, etc.), reject the source

Returns `(usable: bool, reason: str)`.

**Integration point:** Called in `build()` after the existing `if not fetch_result.text` check (line 224), before vault save begins. If content is rejected, log the reason and return early with an error in `BuildResult`. The entire triple-write pipeline is skipped — no partial indexing of garbage.

**PDF artifact pattern:**
```python
_PDF_ARTIFACT_RE = re.compile(
    r"(^\d+ \d+ obj|endobj|endstream|xref|trailer|"
    r"stream$|/Type|/Filter|/Length|/Pages|/Root|"
    r"FlateDecode|MediaBox|DeviceRGB|%%EOF)",
    re.MULTILINE,
)
```

This reuses the same domain knowledge as the existing `_GARBAGE_PATTERNS` regex for entity filtering, but applied at the source level.

### 2. ATL Auto-Persist

**File:** `src/jarvis/evolution/loop.py`

**Change:** In the action dispatch block (around line 431), capture the `call_tool()` return value for `search_and_read` results and run a persist pipeline.

**Flow:**
1. Capture tool result: `tool_result = await self._mcp_client.call_tool(tool_name, params)`
2. If `tool_name == "search_and_read"` and result has >200 chars:
   a. **Goal-match:** Find which goal this action relates to
   b. **Dedup:** Skip if this query was already persisted
   c. **Synthesis:** LLM summarizes key findings relevant to the goal
   d. **Persist:** `KnowledgeBuilder.build()` with synthesized text

#### 2a. Goal Matching

**New method:** `_match_goal_for_action(action: ATLAction, goals: list[Goal]) -> Goal | None`

Logic:
1. Check `action.params["goal_id"]` if present (explicit)
2. Word-overlap between `action.rationale + query` and each goal title
3. Require minimum 2 matching words
4. No match → no persist (safe default)

#### 2b. Dedup

Two layers:
- **Session-level:** `_persisted_queries: set[str]` — reset on ATL restart. Prevents same query from being persisted multiple times within a session.
- **Index-level:** `GoalScopedIndex.has_source(url)` — prevents re-indexing across sessions.

#### 2c. Synthesis

**New method:** `_synthesize_for_goal(research_text: str, goal_title: str, query: str) -> str | None`

Calls the local LLM with a focused prompt:
```
Du bist ein Wissensassistent der Recherche-Ergebnisse einordnet.

Ziel: {goal_title}
Suchanfrage: {query}

Recherche-Ergebnis:
{research_text[:3000]}

Aufgabe:
1. Extrahiere die 3-5 wichtigsten Fakten die für das Ziel relevant sind
2. Formuliere eine strukturierte Notiz (deutsch, sachlich)
3. Wenn nichts Relevantes gefunden wurde, antworte nur mit "KEINE_RELEVANZ"

Format:
## {Thema}
- Kernaussage 1 (Quelle: ...)
- Kernaussage 2
...
```

Returns `None` if LLM responds with "KEINE_RELEVANZ" or on error. This filters irrelevant search hits before they pollute the index.

**Why synthesis matters:** Cognithor should behave like an intelligent agent taking structured notes, not a web crawler dumping raw HTML extracts. The synthesis step produces readable, goal-relevant vault notes with clear attribution.

#### 2d. Persist

Construct a `FetchResult` from the synthesized text and call `KnowledgeBuilder.build()`:
```python
FetchResult(
    url=query,  # Search query as source identifier
    text=synthesis,
    title=f"ATL: {action.rationale[:80]}",
    source_type="atl_research",
)
```

Entity extraction is skipped (`skip_entity_extraction=True`) to avoid GPU load during idle-time ATL cycles. Chunks + vault are sufficient for progress tracking.

#### 2e. KnowledgeBuilder Instances

The ATL loop needs `KnowledgeBuilder` instances per goal. These are created lazily and cached in `_knowledge_builders: dict[str, KnowledgeBuilder]`. The ATL already has access to `self._deep_learner` for goal slugs and `self._mcp_client` for tool calls.

### 3. Data Flow

```
ATL Thinking Cycle (every 5 min)
  │
  ├─ LLM proposes actions (max 3)
  │
  ├─ Action type="research" → search_and_read(query)
  │     │
  │     ├─ Result comes back (raw text, ~3 pages)
  │     ├─ Goal-Match → no match? → log + skip
  │     ├─ Dedup → already persisted? → skip
  │     ├─ Synthesis (LLM) → "KEINE_RELEVANZ"? → skip
  │     │
  │     └─ KnowledgeBuilder.build(synthesized_text)
  │           │
  │           ├─ Content Quality Gate ← NEW
  │           │   └─ min 200 chars? <30% PDF artifacts?
  │           │   └─ Fail → log + return early
  │           │
  │           ├─ vault_save (structured note)
  │           ├─ chunk + save_to_memory (semantic tier)
  │           ├─ goal_index.add_chunk()
  │           └─ Entity extraction: SKIPPED
  │
  └─ Goal Progress Update (existing metric, unchanged)
      └─ 40% subgoal + 30% chunks + 20% entities + 10% sources
         → chunks now grow → progress increases
```

### 4. Files Changed

| File | Change |
|------|--------|
| `knowledge_builder.py` | Add `_is_usable_content()`, `_PDF_ARTIFACT_RE`. Call in `build()` before triple-write. |
| `loop.py` | Capture tool result. Add `_match_goal_for_action()`, `_synthesize_for_goal()`, persist logic, dedup set, lazy KnowledgeBuilder cache. |

### 5. Files NOT Changed

| File | Reason |
|------|--------|
| `web.py` | No blast radius — quality gate is in KnowledgeBuilder |
| `deep_learner.py` | Automatically benefits from quality gate in KnowledgeBuilder |
| `goal_index.py` | No changes needed |
| `action_queue.py` | No changes needed |
| `atl_prompt.py` | No prompt changes — persist is code-level, not LLM-dependent |

### 6. Testing

| Test | Type | What it verifies |
|------|------|------------------|
| `_is_usable_content` with real PDF dump text | Unit | Rejects PDF metadata, returns reason |
| `_is_usable_content` with real article text | Unit | Accepts good content |
| `_is_usable_content` with short text (<200 chars) | Unit | Rejects, returns "too_short" |
| `_is_usable_content` with borderline content (25-35% garbage) | Unit | Threshold behavior |
| `_match_goal_for_action` with matching rationale | Unit | Returns correct goal |
| `_match_goal_for_action` with no match | Unit | Returns None |
| `_synthesize_for_goal` with relevant content | Unit (mocked LLM) | Returns structured note |
| `_synthesize_for_goal` with irrelevant content | Unit (mocked LLM) | Returns None on KEINE_RELEVANZ |
| `build()` with garbage content | Unit | Returns early, no vault/chunk/entity writes |
| ATL cycle with mocked MCP | Integration | search_and_read result → chunks appear in GoalIndex |
| ATL dedup | Integration | Same query twice → only one persist |

### 7. LLM Cost

- Synthesis call: ~500 input tokens + ~200 output tokens per successful search_and_read
- Max 3 actions per cycle, 5 min interval = max 3 synthesis calls per 5 min
- Uses local qwen model, no API costs
- KEINE_RELEVANZ filter prevents unnecessary persist overhead
