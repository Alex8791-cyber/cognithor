# Computer Use Phase 2C: Planner Intelligence — Design Spec

**Date:** 2026-04-03
**Status:** Approved
**Depends on:** Phase 2A (Vision Engine), Phase 2B (Agent Loop) — both complete

## Goal

Make the CU agent handle complex multi-step desktop automation tasks by adding sub-task decomposition, structured content accumulation, file creation, error recovery, and completion reporting — all within the existing `CUAgentExecutor` architecture.

## Reference Scenario

"Oeffne die Anwendung 'Reddit' auf meinem Computer, gib oben in die Zeile bei 'Find Anything' ein: /locallama. Scrolle die letzten 10 Posts und erzeuge jeweils eine Zusammenfassung. Lege diese dann in der neu erstellten Datei 'Reddit_fetch_[heutiges Datum ohne Punkte]' ab und informiere mich darueber, wo die Datei abgelegt wurde — gib mir den Link dorthin."

## Architecture

**Approach: Decompose, Then Adapt (Hybrid)**

A `CUTaskDecomposer` breaks the goal into ordered sub-tasks before the loop starts. Each sub-task runs as a mini agent-loop with the existing screenshot-decide-act cycle. Completion hints enable the executor to detect when a sub-task is done without relying solely on the model saying "DONE". Content flows between sub-tasks via a shared content bag.

Key constraint: `CUAgentExecutor` already handles the inner loop. Phase 2C wraps it with sub-task management. The inner loop per sub-task is identical to Phase 2B.

## 1. Data Model

All new dataclasses live in `src/jarvis/core/cu_agent.py`.

### CUSubTask

A single phase of the decomposed goal:

```python
@dataclass
class CUSubTask:
    name: str                    # "navigate_to_subreddit"
    goal: str                    # "Klicke auf Find Anything und tippe /locallama"
    completion_hint: str         # "locallama erscheint in URL oder Titel"
    max_iterations: int = 10    # per-sub-task iteration budget
    available_tools: list[str] = field(default_factory=list)  # tools for this phase
    extract_content: bool = False  # whether to accumulate text
    content_key: str = ""        # key in content bag, e.g. "posts"
    output_file: str = ""        # resolved filename if this phase writes a file
    status: str = "pending"      # pending | running | done | failed | partial
```

### CUTaskPlan

The full decomposed plan:

```python
@dataclass
class CUTaskPlan:
    original_goal: str
    sub_tasks: list[CUSubTask]
    output_filename: str = ""    # e.g. "Reddit_fetch_20260403.txt"
    variables: dict[str, str] = field(default_factory=dict)  # {"date": "20260403"}
```

### Extended CUAgentResult

Two new fields added to the existing dataclass:

```python
@dataclass
class CUAgentResult:
    success: bool = False
    iterations: int = 0
    duration_ms: int = 0
    tool_results: list[ToolResult] = field(default_factory=list)
    final_screenshot_description: str = ""
    abort_reason: str = ""
    extracted_content: str = ""
    action_history: list[str] = field(default_factory=list)
    # Phase 2C additions:
    output_files: list[str] = field(default_factory=list)
    task_summary: str = ""
```

### Content Bag

Shared between sub-tasks, simple dict passed forward:

```python
content_bag: dict[str, list[str]] = {}
# e.g. {"posts": ["## Post 1/10\n...", "## Post 2/10\n..."]}
```

## 2. CUTaskDecomposer

New class in `cu_agent.py`. Runs once before the agent loop.

### Responsibility

Takes the user's goal string, resolves variables (date, paths), and produces a `CUTaskPlan` by calling qwen3-vl:32b with a structured prompt.

### Variable Resolution

```python
def _resolve_variables(self, goal: str) -> dict[str, str]:
    today = datetime.now()
    return {
        "date": today.strftime("%Y%m%d"),
        "date_dots": today.strftime("%d.%m.%Y"),
        "date_iso": today.isoformat()[:10],
        "user_home": str(Path.home()),
        "documents": str(Path.home() / "Documents"),
    }
```

Variables are resolved before the LLM call. The prompt includes `{date} = 20260403` etc. so the model can use them in `output_file`.

### Decomposition Prompt

`_CU_DECOMPOSE_PROMPT` instructs the model to:
- Break the goal into sequential phases
- Each phase has: name (snake_case), goal, completion_hint (visible on screen), max_iterations, tools, extract_content flag, content_key, output_file
- Includes a few-shot example (calculator scenario)
- Lists available tools: computer_screenshot, computer_click, computer_type, computer_hotkey, computer_scroll, exec_command, write_file, extract_text

### Parsing

Uses the existing 3-tier JSON parsing pattern:
1. Direct `json.loads`
2. Markdown code block extraction
3. Regex for JSON array

### Graceful Degradation

If parsing fails entirely, the decomposer returns a single sub-task with the original goal — identical to Phase 2B behavior. Phase 2C is strictly additive.

### Method Signature

```python
class CUTaskDecomposer:
    def __init__(self, planner: Any, config: CUAgentConfig) -> None: ...
    async def decompose(self, goal: str) -> CUTaskPlan: ...
```

## 3. Enhanced Agent Loop

`CUAgentExecutor.execute()` changes from a flat loop to a sub-task-driven loop. The inner loop per sub-task stays identical to Phase 2B.

### Flow

```
execute(goal, initial_plan)
  +-- decomposer.decompose(goal) -> CUTaskPlan
  +-- execute initial_plan steps (as before)
  +-- for each sub_task in plan.sub_tasks:
  |     +-- reset iteration counter, recent_actions
  |     +-- inject sub-task context into prompt
  |     +-- inner loop: screenshot -> decide -> act
  |     |     +-- on extract_text: label as "## {content_key} {n}"
  |     |     +-- on failed action: inject failure hint into next decide
  |     |     +-- check completion_hint against screenshot description
  |     |     +-- check sub-task max_iterations
  |     |     +-- check global abort conditions
  |     +-- on completion: store content in content_bag[content_key]
  |     +-- on sub-task failure: log, continue to next (with warning)
  +-- populate result.output_files, result.task_summary
  +-- return CUAgentResult
```

### Enhanced Prompt

The existing `_CU_DECIDE_PROMPT` gets a new context block prepended per sub-task:

```
--- Aktuelle Phase: {phase_name} ({phase_idx}/{phase_total}) ---
Phasenziel: {phase_goal}
Abschlusskriterium: {completion_hint}
{extraction_status}   # "Du hast 7/10 Eintraege extrahiert." or empty
{failure_hint}        # "Letzte Aktion fehlgeschlagen: ..." or empty
{content_preview}     # last 500 chars of accumulated content, or empty
---
```

### Completion Hint Matching

Fuzzy keyword overlap — 60% threshold:

```python
def _check_completion_hint(self, hint: str, screenshot_desc: str) -> bool:
    hint_lower = hint.lower()
    desc_lower = screenshot_desc.lower()
    keywords = [w for w in hint_lower.split() if len(w) > 3]
    if not keywords:
        return False
    matches = sum(1 for kw in keywords if kw in desc_lower)
    return matches / len(keywords) >= 0.6
```

Not exact matching. The model can still say DONE as a fallback path.

### Extraction Counter

The executor (not the model) tracks how many `extract_text` calls succeeded in the current sub-task. This count gets injected into the prompt so the model doesn't lose track:

```
Du hast 7 Eintraege extrahiert.
```

### Content Labeling

Each extracted text block gets labeled automatically by the executor:

```
## Post 3
[extracted text here]
```

The label uses the sub-task's `content_key` and the running extraction count.

## 4. Error Recovery

### Failure Hint Injection

When a tool action fails, the executor records the failure and injects a hint into the next `_decide_next_step` call:

```
Letzte Aktion fehlgeschlagen: computer_click(x=450, y=320) -> Element nicht gefunden.
Versuche eine Alternative: anderes Element, scrollen, oder warten.
```

### Escalation Levels

| Consecutive failures | Behavior |
|---------------------|----------|
| 1 | Hint: "Versuche eine Alternative" |
| 2 | Hint: "Versuche einen komplett anderen Ansatz" |
| 3+ | Hint: "Phase wird uebersprungen wenn naechste Aktion auch fehlschlaegt" |
| 4 | Sub-task marked failed, move to next |

### Screenshot Stale Detection

If the screenshot description is very similar to the previous one (>90% word overlap), the executor treats it as a no-op:

```python
def _screenshot_similarity(self, prev: str, curr: str) -> float:
    words_a = set(prev.lower().split())
    words_b = set(curr.lower().split())
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)
```

After 2 stale screens, a failure hint is injected: "Bildschirm hat sich nicht veraendert."

### No Re-Decomposition

If a sub-task fails and gets skipped, the remaining sub-tasks continue as planned. No mid-execution re-decomposition — if the plan is fundamentally wrong, global abort conditions (timeout, stuck) catch it.

## 5. File Creation

### How It Works

The decomposer adds a file-writing sub-task when the goal involves saving content. This sub-task:

1. Has `write_file` in its `available_tools`
2. Has `output_file` set to the resolved filename
3. Gets the accumulated content from the content bag injected into its prompt

### File Path Resolution

```python
def _resolve_output_path(self, filename: str, variables: dict) -> str:
    for key, val in variables.items():
        filename = filename.replace(f"{{{key}}}", val)
    return str(Path(variables["documents"]) / filename)
```

`Reddit_fetch_{date}.txt` -> `C:\Users\ArtiCall\Documents\Reddit_fetch_20260403.txt`

### Prompt for File-Writing Sub-Task

The enhanced decide prompt includes:

```
Gesammelter Inhalt ({content_key}, {count} Eintraege):
---
{content_bag_preview}
---

Schreibe diesen Inhalt mit write_file in die Datei: {output_file}
Zielverzeichnis: {documents_path}
```

The model calls `write_file`. The executor intercepts the result and records the full path in `result.output_files`.

## 6. Completion Reporting

### Enriched CUAgentResult

After all sub-tasks finish, the executor populates:

```python
completed = [st for st in sub_tasks if st.status == "done"]
failed = [st for st in sub_tasks if st.status == "failed"]

result.task_summary = (
    f"{len(completed)}/{len(sub_tasks)} Phasen abgeschlossen. "
    + (f"Fehlgeschlagen: {', '.join(f.name for f in failed)}. " if failed else "")
    + (f"Dateien erstellt: {', '.join(result.output_files)}. " if result.output_files else "")
    + f"Gesammelter Inhalt: {len(result.extracted_content)} Zeichen."
)
```

### Gateway Integration

The gateway's CU result block (3 lines added) includes:

```python
+ (f"\nZusammenfassung: {cu_result.task_summary}" if cu_result.task_summary else "")
+ (f"\nErstellte Dateien: {', '.join(cu_result.output_files)}" if cu_result.output_files else "")
```

The planner (qwen3.5:27b, back in main PGE loop) formulates the natural-language response from these enriched fields.

## 7. Files Changed

| File | Change |
|------|--------|
| `src/jarvis/core/cu_agent.py` | Add CUSubTask, CUTaskPlan, CUTaskDecomposer. Extend CUAgentResult. Refactor execute() to sub-task loop. Add prompts, hint matching, failure tracking, content bag, extraction counter. |
| `src/jarvis/gateway/gateway.py` | ~3 lines: include task_summary and output_files in CU result message. |
| `tests/test_core/test_cu_agent.py` | New tests for decomposer, sub-task transitions, hint matching, failure escalation, content bag, file paths, extraction counter, degradation fallback. |

### Unchanged

- `mcp/computer_use.py` — CU tools unchanged
- `browser/vision.py` — vision engine unchanged
- `core/planner.py` — no new CU workarounds
- `core/executor.py` — no changes
- `core/gatekeeper.py` — write_file already classified

## 8. Degradation Guarantee

If the decomposer fails (bad JSON, model error, timeout), the executor falls back to a single sub-task with the original goal. This is identical to Phase 2B behavior. Phase 2C is strictly additive — it cannot break existing CU functionality.
