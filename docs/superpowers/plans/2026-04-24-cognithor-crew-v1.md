# Cognithor Crew-Layer v1.0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the five v1.0-blocker features from `docs/superpowers/specs/2026-04-23-cognithor-crew-v1-adoption.md` (Features 1, 4, 3, 2, 7) — a CrewAI-inspired high-level Crew-API layer on top of PGE-Trinity, plus Guardrails, Scaffolding CLI + Templates, Quickstart documentation, and auto-generated Integrations catalog.

**Architecture:** New `cognithor.crew` package — Pydantic v2 dataclasses (`CrewAgent`, `CrewTask`, `Crew`, `CrewOutput`, `TaskOutput`) compile to sequential or hierarchical `PlanRequest`s that route through `Planner.formulate_response()` → `Gatekeeper.classify()` → Executor. No direct LLM calls from the Crew-Layer; it is a pure translation layer. Guardrails run as Gatekeeper post-execution hooks. `cognithor init` scaffolds from Jinja2 templates into new projects. All docs + examples live under `docs/quickstart/` and `examples/quickstart/` with a CI job that exercises every example against a mock-Ollama container.

**Tech Stack:** Python 3.12, Pydantic v2, pytest + pytest-asyncio, Jinja2 (new runtime dep — added in Feature 3 Task 34), ruff, mypy strict. No verbatim CrewAI code; API shape is concept-inspired only (MIT → Apache 2.0 bridge via NOTICE attribution).

---

## Sequencing and Dependencies

Per spec §9:

1. **Feature 1 (Tasks 1-20)** — Crew-Layer core (foundation for everything)
2. **Feature 4 (Tasks 21-32)** — Guardrails (builds on Feature 1)
3. **Feature 3 (Tasks 33-52)** — `cognithor init` + Templates (uses Features 1 + 4)
4. **Feature 7 (Tasks 67-78)** — Integrations catalog (parallel-safe with Feature 2; code-only so it ships BEFORE docs)
5. **Feature 2 (Tasks 53-66)** — Quickstart docs (documents 1 + 3 + 4 + 7)
6. **Final integration + PR prep (Tasks 79-82)** — split: 79 / 79b / 79c / 80a / 80b / 81 / 82

Features 5 (Trace-UI) and 6 (Flows) are explicitly out of plan scope — those are v1.x per spec §5.6 and §6.6.

---

## PR Strategy (Five Sequential PRs)

The 82 tasks ship as **five sequential PRs against `main`**, not one mega-PR. This keeps each review digestible and lets us ship incrementally while still gating v0.93.0 on the final PR. The final docs-only PR is split from the integrations-catalog PR because (a) doc review has different reviewers than code review, (b) docs benefit from preview builds, and (c) keeping the integrations/sevDesk-connector PR small and shippable lets it land before the release-blocking docs review finishes.

| PR | Feature | Tasks | Branch | Merge Target | Release? |
|----|---------|-------|--------|--------------|----------|
| **PR 1** | Feature 1 — Crew-Layer Core | 1-20 | `feat/cognithor-crew-v1-f1` | `main` | No |
| **PR 2** | Feature 4 — Guardrails | 21-32 | `feat/cognithor-crew-v1-f4` | `main` | No |
| **PR 3** | Feature 3 — CLI + Templates | 33-52 | `feat/cognithor-crew-v1-f3` | `main` | No |
| **PR 4a** | Feature 7 — Integrations Catalog + sevDesk connector (code + tests) | 67-78 | `feat/cognithor-crew-v1-f7` | `main` | No |
| **PR 4b** | Feature 2 — 8-page Quickstart Docs + version bump | 53-66 + 60b | `feat/cognithor-crew-v1-f2` | `main` | **Yes — v0.93.0** |
| **Site PR** | cognithor-site v0.93.0 integrations + changelog (Task 80c) | site-repo only | `release/v0.93.0` (cognithor-site) | cognithor-site/`main` | **Gates tag push** |

**Parallelism:** Task 80c (the cognithor-site PR) is drafted and opened IN PARALLEL with PR 4b review. The site-PR must be merged before Task 82 Step 3 (`git tag v0.93.0`). This is a hard gate on spec §12 AC 7 — published release links to live docs, never 404s.

**Branch strategy:** Each PR is its own feature branch cut from `main`. After each PR merges:

```bash
git checkout main && git pull
git checkout -b feat/cognithor-crew-v1-fN   # next feature's branch
```

No long-lived staging branch. If PR N's tasks depend on PR N-1 types, PR N-1 must be merged first — this is a hard dependency, not optional.

**Per-PR closeout sequence (applies to PRs 1, 2, 3, 4a):**
1. Full regression on the feature branch (`pytest tests/ -x -q --cov=src/cognithor`)
2. Ruff + Ruff-format check clean
3. Mypy --strict clean on new code
4. CHANGELOG `[Unreleased]` section shows the feature's entries (no version bump yet)
5. Open PR, wait all CI green, merge into `main`
6. **Do NOT** chain merge + cleanup via `&&` (per feedback memory — this has caused two branch-closure incidents)

**PR 4b (docs + release) closeout adds:**
7. CHANGELOG `[Unreleased]` → `[0.93.0]` bump (all feature entries consolidated from PRs 1-4a)
8. Version bump across 5 locations (see Task 80)
9. External-reader usability pass verified (spec §12 AC 4 — Task 62)
10. Merge PR 4b → open cognithor-site PR → site-PR merges → tag `v0.93.0` → release pipeline runs → PyPI publish

PR-specific merge-prep task groups are spelled out at the end of each feature block (see "Tasks 79-82 Restructured — Per-PR Merge-Prep" at the bottom of the plan).

**Cross-repo dependency (Spec §7.2.1 / §12 AC 7):** The `cognithor.ai/integrations` site page lives in the `cognithor-site` Vercel repo and is **not** part of any PR in this plan. This plan produces `docs/integrations/catalog.json` + the generator; the separate site PR consumes them. Spec §12 AC 7 is satisfied by the site PR landing before the v0.93.0 PyPI release completes — that is tracked as Task 73's site-integration note and as an external checklist item in the final release checklist.

---

## File Structure

### New package: `src/cognithor/crew/`

- `__init__.py` — public API exports
- `agent.py` — `CrewAgent` Pydantic model
- `task.py` — `CrewTask` Pydantic model
- `process.py` — `CrewProcess` enum (SEQUENTIAL, HIERARCHICAL)
- `output.py` — `CrewOutput`, `TaskOutput`, `TokenUsageDict`
- `crew.py` — `Crew` class with `kickoff()` / `kickoff_async()`
- `compiler.py` — Translates `Crew` to ordered `PlanRequest`s through the Planner
- `yaml_loader.py` — Load crews from `config/agents.yaml` + `config/tasks.yaml`
- `decorators.py` — `@cognithor.crew.agent`, `@cognithor.crew.task`, `@cognithor.crew.crew`
- `errors.py` — `CrewError`, `ToolNotFoundError`, `GuardrailFailure`, `CrewCompilationError`
- `guardrails/__init__.py` — public guardrail exports
- `guardrails/base.py` — `Guardrail` protocol, `GuardrailResult` dataclass
- `guardrails/function_guardrail.py` — `FunctionGuardrail` wrapper
- `guardrails/string_guardrail.py` — `StringGuardrail` LLM-validated
- `guardrails/builtin.py` — `hallucination_check`, `word_count`, `no_pii`, `schema`, `chain`
- `cli/__init__.py`
- `cli/init_cmd.py` — `cognithor init` subcommand
- `cli/run_cmd.py` — `cognithor run` subcommand (used inside scaffolded projects)
- `cli/list_templates_cmd.py`
- `cli/scaffolder.py` — Jinja2 render helper (shared with skills scaffolder)
- `templates/` — 5 directories (one per template), each with its Jinja2 tree

### New tests tree: `tests/test_crew/`

- `__init__.py`
- `test_agent.py`, `test_task.py`, `test_process.py`, `test_output.py`, `test_crew.py`
- `test_compiler.py`, `test_yaml_loader.py`, `test_decorators.py`, `test_errors.py`
- `test_sequential_kickoff.py`, `test_hierarchical_kickoff.py`, `test_async_kickoff.py`
- `test_tool_resolution.py`, `test_context_passing.py`, `test_idempotent_kickoff.py`
- `test_audit_chain.py`, `test_gatekeeper_integration.py`
- `test_guardrails/test_base.py`, `test_function.py`, `test_string.py`
- `test_guardrails/test_hallucination.py`, `test_word_count.py`, `test_no_pii.py`, `test_schema.py`, `test_chain.py`
- `test_cli/test_init.py`, `test_list_templates.py`, `test_run.py`
- `test_cli/test_scaffolder.py`
- `test_templates/test_research.py`, `test_customer_support.py`, `test_data_analyst.py`, `test_content.py`, `test_versicherungs_vergleich.py`
- `test_pkv_example.py` — spec §1.4 end-to-end

### New documentation: `docs/quickstart/`

Eight pages each in German (default) and English (`.en.md` suffix):

- `00-installation.md` / `.en.md`
- `01-first-crew.md` / `.en.md`
- `02-first-tool.md` / `.en.md`
- `03-first-skill.md` / `.en.md`
- `04-guardrails.md` / `.en.md`
- `05-deployment.md` / `.en.md`
- `06-next-steps.md` / `.en.md`
- `07-troubleshooting.md` / `.en.md`  (R3-NI9 / R4: FAQ page; see Task 60b)
- `README.md` — quickstart index

### New examples: `examples/quickstart/`

- `01_first_crew/main.py`, `requirements.txt`, `README.md`
- `02_first_tool/` — analogous
- `03_first_skill/` — analogous
- `04_guardrails/` — analogous
- `05_pkv_report/` — the spec's PKV example (§1.4)

### New integrations catalog:

- `docs/integrations/catalog.json` (generated)
- `docs/integrations/README.md`
- `scripts/generate_integrations_catalog.py`
- `tests/test_integrations_catalog.py`

### New CI workflows: `.github/workflows/`

- `quickstart-examples.yml` — runs every example against mock Ollama
- `integrations-catalog.yml` — regenerates catalog.json and fails if drift

### Modified files:

- `pyproject.toml` — adds `jinja2>=3.1,<4` to runtime deps (Feature 3)
- `src/cognithor/__init__.py` — re-exports `cognithor.crew.*` at package root for DX
- `src/cognithor/__main__.py` — wire `cognithor init`, `cognithor run` subcommands
- `NOTICE` — CrewAI concept attribution (new file if absent, update if exists)
- `CHANGELOG.md` — v0.93.0 entry (Crew-Layer is a semver minor bump — additive)
- `README.md` — Highlights entry for Crew-Layer + link to quickstart

---

## Scope Clarifications

- **Module import path:** `cognithor.crew` (lowercase, matches Python conventions).
- **No new runtime deps except Jinja2.** Everything else reuses existing dependencies (Pydantic v2, PyYAML, structlog).
- **Apache 2.0 only.** `NOTICE` gets an attribution line; no verbatim CrewAI code anywhere.
- **Backward compatibility:** Zero changes to the existing Agent SDK (`@agent`, `@tool`, `@hook`) or to PGE-Trinity internals. The Crew-Layer is strictly additive.
- **Test coverage floor:** Branch CI guards ≥ 89% total coverage. Each new module ships with ≥ 85% line coverage of its own.
- **DSGVO:** All defaults offline-capable (mock-Ollama container in CI). No new external HTTP calls in default code paths.

---

# FEATURE 1 — Crew-Layer Core (Tasks 1-20)

Implements spec §1: `cognithor.crew` package with `CrewAgent`, `CrewTask`, `Crew`, `CrewProcess`, `CrewOutput`, sequential + hierarchical processes, YAML loader, decorators, and full PGE-Trinity + audit-chain integration.

---

### Task 1: Package skeleton + public exports

**Files:**
- Create: `src/cognithor/crew/__init__.py`
- Create: `tests/test_crew/__init__.py`
- Create: `tests/test_crew/test_package_exports.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_crew/test_package_exports.py
def test_public_api_exports():
    from cognithor import crew
    assert hasattr(crew, "CrewAgent")
    assert hasattr(crew, "CrewTask")
    assert hasattr(crew, "Crew")
    assert hasattr(crew, "CrewProcess")
    assert hasattr(crew, "CrewOutput")
    assert hasattr(crew, "TaskOutput")
    assert hasattr(crew, "GuardrailFailure")
    assert hasattr(crew, "ToolNotFoundError")
```

- [ ] **Step 2: Run test — expect failure**

```bash
cd "D:/Jarvis/jarvis complete v20"
python -m pytest tests/test_crew/test_package_exports.py -v
```
Expected: `ModuleNotFoundError: No module named 'cognithor.crew'`

- [ ] **Step 3: Create `src/cognithor/crew/__init__.py`**

```python
"""Cognithor Crew-Layer — high-level Multi-Agent API on top of PGE-Trinity.

Concept inspired by CrewAI (MIT, crewAIInc/crewAI) — re-implementation in
Apache 2.0; no source-level copy.

See docs/superpowers/specs/2026-04-23-cognithor-crew-v1-adoption.md.
"""

from __future__ import annotations

from cognithor.crew.agent import CrewAgent
from cognithor.crew.crew import Crew
from cognithor.crew.errors import (
    CrewCompilationError,
    CrewError,
    GuardrailFailure,
    ToolNotFoundError,
)
from cognithor.crew.output import CrewOutput, TaskOutput, TokenUsageDict
from cognithor.crew.process import CrewProcess
from cognithor.crew.task import CrewTask

__all__ = [
    "Crew",
    "CrewAgent",
    "CrewCompilationError",
    "CrewError",
    "CrewOutput",
    "CrewProcess",
    "CrewTask",
    "GuardrailFailure",
    "TaskOutput",
    "TokenUsageDict",
    "ToolNotFoundError",
]
```

This will still fail until Tasks 2-7 add the referenced modules. For now create empty files so imports resolve:

```bash
touch src/cognithor/crew/{agent,task,crew,process,output,errors}.py
touch tests/test_crew/__init__.py
```

Add minimal placeholders that the tests will replace:

```python
# src/cognithor/crew/errors.py
from __future__ import annotations

from dataclasses import dataclass


class CrewError(Exception):
    """Base class for every Crew-Layer error."""


class CrewCompilationError(CrewError):
    """Raised when the Compiler cannot translate a Crew into PGE inputs."""


class ToolNotFoundError(CrewError):
    """Raised when an agent references a tool the registry does not expose."""


@dataclass
class GuardrailFailure(CrewError):
    """Raised when a guardrail rejects output after exhausting retries.

    The message says "after N attempt(s)" where N is the actual number of
    attempts made (initial try + retries). Avoids the "max_retries" off-by-one
    surprise where max_retries=2 meant 3 attempts.

    Includes a custom ``__reduce__`` so the exception can be pickled and
    unpickled correctly — a plain ``@dataclass`` subclass of Exception fails
    under ``ProcessPoolExecutor`` / ``multiprocessing.Queue`` / Celery because
    the dataclass-generated ``__init__`` signature does not match the
    single-arg unpickle path ``Exception.__init__`` uses by default.
    """

    task_id: str
    guardrail_name: str
    attempts: int
    reason: str

    def __str__(self) -> str:
        return (
            f"Guardrail '{self.guardrail_name}' rejected output from task "
            f"'{self.task_id}' after {self.attempts} attempt(s): {self.reason}"
        )

    def __post_init__(self) -> None:
        # Keep Exception.args in sync so stack traces show a useful repr.
        super().__init__(str(self))

    def __reduce__(self) -> tuple:
        """Support pickling across process boundaries.

        Without this, ``pickle.dumps(GuardrailFailure(...))`` succeeds but
        ``pickle.loads(...)`` raises a misleading
        ``TypeError: __init__() missing 3 required positional arguments``
        because Exception's default unpickle path calls ``__init__`` with a
        single positional arg (``self.args[0]``) which doesn't match the
        dataclass signature.
        """
        return (
            self.__class__,
            (self.task_id, self.guardrail_name, self.attempts, self.reason),
        )
```

Also add a regression test `test_guardrail_failure_pickle_roundtrip` in `tests/test_crew/test_errors.py`:

```python
import pickle

from cognithor.crew.errors import GuardrailFailure


def test_guardrail_failure_pickle_roundtrip():
    """GuardrailFailure must survive pickle.dumps -> pickle.loads intact.

    Regression: a plain @dataclass Exception subclass breaks multiprocessing /
    ProcessPoolExecutor / Celery because Exception's unpickle path passes a
    single arg to __init__, which the dataclass __init__ rejects. The custom
    __reduce__ fixes this by telling pickle to pass all 4 fields.
    """
    original = GuardrailFailure(
        task_id="t42",
        guardrail_name="no_pii",
        attempts=3,
        reason="email detected",
    )
    roundtripped = pickle.loads(pickle.dumps(original))

    assert roundtripped.task_id == "t42"
    assert roundtripped.guardrail_name == "no_pii"
    assert roundtripped.attempts == 3
    assert roundtripped.reason == "email detected"
    # Exception message preserved too
    assert str(roundtripped) == str(original)
```

```python
# src/cognithor/crew/process.py
from enum import Enum
class CrewProcess(Enum):
    SEQUENTIAL = "sequential"
    HIERARCHICAL = "hierarchical"
```

Leave `agent.py`, `task.py`, `crew.py`, `output.py` with `# stub` comments — they are filled in Tasks 2-7.

- [ ] **Step 4: Run — expect ImportError on CrewAgent etc.**

```bash
python -m pytest tests/test_crew/test_package_exports.py -v
```

Expected failure: `ImportError: cannot import name 'CrewAgent' from 'cognithor.crew.agent'` (stub module is empty).

- [ ] **Step 5: Add placeholder classes to satisfy the import test**

```python
# src/cognithor/crew/agent.py
from __future__ import annotations
class CrewAgent: ...  # Implementation in Task 2
```

Same pattern for `task.py` (`class CrewTask: ...`), `crew.py` (`class Crew: ...`), `output.py` (`class CrewOutput: ...`, `class TaskOutput: ...`, `TokenUsageDict = dict`).

- [ ] **Step 6: Run — expect PASS**

```bash
python -m pytest tests/test_crew/test_package_exports.py -v
```

- [ ] **Step 7: Ruff + commit**

```bash
python -m ruff check src/cognithor/crew tests/test_crew
python -m ruff format --check src/cognithor/crew tests/test_crew
git add src/cognithor/crew tests/test_crew
git commit -m "feat(crew): package skeleton + public API exports"
```

---

### Task 2: `CrewProcess` enum with full unit tests

**Files:**
- Modify: `src/cognithor/crew/process.py`
- Create: `tests/test_crew/test_process.py`

> **pytest-asyncio convention (R4-I2):** the repository root `pyproject.toml`
> sets `asyncio_mode = "auto"` — any `async def test_*` is automatically
> detected as an async test. **Do NOT add `@pytest.mark.asyncio` decorators to
> tests in this plan.** They are redundant under auto-mode and can mask
> config-drift if the mode is ever disabled. All `async def test_*` functions
> throughout Tasks 1-83 follow this convention.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_crew/test_process.py
import pytest
from cognithor.crew.process import CrewProcess


class TestCrewProcess:
    def test_has_sequential_and_hierarchical(self):
        assert CrewProcess.SEQUENTIAL.value == "sequential"
        assert CrewProcess.HIERARCHICAL.value == "hierarchical"

    def test_two_members_only(self):
        assert len(CrewProcess) == 2

    def test_from_string_roundtrip(self):
        assert CrewProcess("sequential") is CrewProcess.SEQUENTIAL
        assert CrewProcess("hierarchical") is CrewProcess.HIERARCHICAL

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            CrewProcess("parallel")

    def test_stringifies_for_logging(self):
        assert "SEQUENTIAL" in repr(CrewProcess.SEQUENTIAL)
```

- [ ] **Step 2: Run — expect first two pass (from stub), last three fail**

```bash
python -m pytest tests/test_crew/test_process.py -v
```

- [ ] **Step 3: Nothing to change — `Enum` already supports all these**

The stub from Task 1 is already complete. Go to Step 4.

- [ ] **Step 4: Run — expect 5 passed**

- [ ] **Step 5: Ruff + commit**

```bash
git add tests/test_crew/test_process.py
git commit -m "test(crew): CrewProcess enum contract tests"
```

---

### Task 3: `TokenUsageDict`, `TaskOutput`, `CrewOutput` dataclasses

**Files:**
- Modify: `src/cognithor/crew/output.py`
- Create: `tests/test_crew/test_output.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_crew/test_output.py
import pytest
from pydantic import ValidationError
from cognithor.crew.output import CrewOutput, TaskOutput, TokenUsageDict


class TestTokenUsageDict:
    def test_typed_keys(self):
        usage: TokenUsageDict = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
        assert usage["total_tokens"] == 120

    def test_missing_key_raises_at_runtime_on_strict_access(self):
        # TypedDict is advisory at runtime — this test just confirms the type
        # annotation exists and the factory helper sanitizes input.
        from cognithor.crew.output import empty_token_usage
        usage = empty_token_usage()
        assert usage == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


class TestTaskOutput:
    def test_minimal(self):
        out = TaskOutput(task_id="t1", agent_role="writer", raw="hello")
        assert out.task_id == "t1"
        assert out.raw == "hello"
        assert out.duration_ms == 0.0
        assert out.token_usage == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def test_structured_output(self):
        out = TaskOutput(
            task_id="t1",
            agent_role="analyst",
            raw='{"foo": 1}',
            structured={"foo": 1},
        )
        assert out.structured == {"foo": 1}

    def test_frozen_after_construction(self):
        out = TaskOutput(task_id="t1", agent_role="x", raw="y")
        with pytest.raises(ValidationError):
            out.raw = "mutated"  # type: ignore[misc]


class TestCrewOutput:
    def test_aggregates_tasks(self):
        t1 = TaskOutput(task_id="t1", agent_role="analyst", raw="A",
                       token_usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
        t2 = TaskOutput(task_id="t2", agent_role="writer", raw="B",
                       token_usage={"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28})
        out = CrewOutput(raw="B", tasks_output=[t1, t2], trace_id="trace-xyz")
        assert out.raw == "B"
        assert len(out.tasks_output) == 2
        assert out.token_usage == {"prompt_tokens": 30, "completion_tokens": 13, "total_tokens": 43}
        assert out.trace_id == "trace-xyz"

    def test_trace_id_required(self):
        with pytest.raises(ValidationError):
            CrewOutput(raw="x", tasks_output=[])  # trace_id omitted
```

- [ ] **Step 2: Run — expect `ImportError` or `ValidationError` mismatches**

- [ ] **Step 3: Implement `src/cognithor/crew/output.py`**

```python
"""Crew output dataclasses — immutable result objects."""

from __future__ import annotations

from typing import Any, TypedDict

from pydantic import BaseModel, ConfigDict, Field, computed_field


class TokenUsageDict(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


def empty_token_usage() -> TokenUsageDict:
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


class TaskOutput(BaseModel):
    """Result of one CrewTask execution."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    task_id: str
    agent_role: str
    raw: str
    structured: dict[str, Any] | None = None
    duration_ms: float = 0.0
    token_usage: TokenUsageDict = Field(default_factory=empty_token_usage)
    guardrail_verdict: str | None = None  # pass / fail / skipped


class CrewOutput(BaseModel):
    """Aggregate result of one Crew.kickoff()."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    raw: str
    tasks_output: list[TaskOutput]
    trace_id: str

    @computed_field  # type: ignore[prop-decorator]
    @property
    def token_usage(self) -> TokenUsageDict:
        prompt = sum(t.token_usage["prompt_tokens"] for t in self.tasks_output)
        completion = sum(t.token_usage["completion_tokens"] for t in self.tasks_output)
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        }
```

- [ ] **Step 4: Run — expect all pass**

- [ ] **Step 5: Ruff + commit**

```bash
git add src/cognithor/crew/output.py tests/test_crew/test_output.py
git commit -m "feat(crew): immutable TaskOutput + CrewOutput + TokenUsageDict"
```

---

### Task 4: `CrewAgent` Pydantic model

**Files:**
- Modify: `src/cognithor/crew/agent.py`
- Create: `tests/test_crew/test_agent.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_crew/test_agent.py
import pytest
from pydantic import ValidationError
from cognithor.crew.agent import CrewAgent


class TestCrewAgent:
    def test_minimal_construction(self):
        a = CrewAgent(role="writer", goal="produce drafts")
        assert a.role == "writer"
        assert a.goal == "produce drafts"
        assert a.backstory == ""
        assert a.tools == []
        assert a.llm is None
        assert a.allow_delegation is False
        assert a.max_iter == 20
        assert a.memory is True
        assert a.verbose is False

    def test_full_construction(self):
        a = CrewAgent(
            role="analyst",
            goal="analyze tarifs",
            backstory="veteran broker",
            tools=["web_search", "pdf_reader"],
            llm="ollama/qwen3:32b",
            allow_delegation=True,
            max_iter=5,
            memory=False,
            verbose=True,
        )
        assert a.tools == ["web_search", "pdf_reader"]
        assert a.llm == "ollama/qwen3:32b"
        assert a.max_iter == 5

    def test_role_and_goal_required(self):
        with pytest.raises(ValidationError):
            CrewAgent(goal="x")  # role missing
        with pytest.raises(ValidationError):
            CrewAgent(role="x")  # goal missing

    def test_max_iter_positive(self):
        with pytest.raises(ValidationError):
            CrewAgent(role="x", goal="y", max_iter=0)

    def test_tools_must_be_strings(self):
        with pytest.raises(ValidationError):
            CrewAgent(role="x", goal="y", tools=[123])  # type: ignore[list-item]

    def test_frozen(self):
        a = CrewAgent(role="x", goal="y")
        with pytest.raises(ValidationError):
            a.role = "z"  # type: ignore[misc]
```

- [ ] **Step 2: Run — expect failures**

- [ ] **Step 3: Implement `src/cognithor/crew/agent.py`**

```python
"""CrewAgent — declarative Pydantic model for a Crew participant."""

from __future__ import annotations

from typing import Any, TypeAlias

from pydantic import BaseModel, ConfigDict, Field


# Forward-compat stub (Spec §1.2): the spec allows `llm: str | LLMConfig`. For
# v1.0 LLMConfig is an opaque dict — concrete schema (temperature, seed, …)
# lands in a later minor release. Using a TypeAlias here keeps the public type
# stable so adding a real BaseModel later is a pure type-widening (non-breaking).
LLMConfig: TypeAlias = dict[str, Any]


class CrewAgent(BaseModel):
    """Declarative description of an agent participating in a Crew.

    Concept inspired by CrewAI's Agent; re-implementation in Apache 2.0.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    role: str = Field(..., min_length=1, description="Short role name, used in logs")
    goal: str = Field(..., min_length=1, description="What this agent is trying to accomplish")
    backstory: str = Field(default="", description="Context the Planner uses to shape the system prompt")
    tools: list[str] = Field(default_factory=list, description="Tool names resolved via MCP registry")
    # Spec §1.2 — widened to str | LLMConfig | None. LLMConfig is currently a
    # dict alias; a future BaseModel swap-in is non-breaking.
    llm: str | LLMConfig | None = Field(
        default=None,
        description="Model spec (e.g. 'ollama/qwen3:32b') or LLMConfig dict",
    )
    allow_delegation: bool = Field(default=False)
    max_iter: int = Field(default=20, ge=1, le=200)
    memory: bool = Field(default=True, description="Enable 6-Tier Cognitive Memory for this agent")
    verbose: bool = Field(default=False)
    metadata: dict[str, Any] = Field(default_factory=dict)
```

Add `LLMConfig` to `src/cognithor/crew/__init__.py` `__all__` alongside `CrewAgent`.

- [ ] **Step 4: Run — expect all 6 tests pass**

- [ ] **Step 5: Ruff + commit**

```bash
git add src/cognithor/crew/agent.py tests/test_crew/test_agent.py
git commit -m "feat(crew): CrewAgent Pydantic model"
```

---

### Task 5: `CrewTask` Pydantic model (guardrail field as Any for now)

**Files:**
- Modify: `src/cognithor/crew/task.py`
- Create: `tests/test_crew/test_task.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_crew/test_task.py
import pytest
from pydantic import BaseModel, ValidationError
from cognithor.crew.agent import CrewAgent
from cognithor.crew.task import CrewTask


@pytest.fixture
def agent() -> CrewAgent:
    return CrewAgent(role="writer", goal="draft")


class TestCrewTask:
    def test_minimal(self, agent: CrewAgent):
        t = CrewTask(description="Write something", expected_output="A sentence.", agent=agent)
        assert t.description == "Write something"
        assert t.agent.role == "writer"
        assert t.context == []
        assert t.tools == []
        assert t.guardrail is None
        assert t.async_execution is False

    def test_context_accepts_other_tasks(self, agent: CrewAgent):
        t1 = CrewTask(description="research", expected_output="facts", agent=agent)
        t2 = CrewTask(description="write", expected_output="text", agent=agent, context=[t1])
        assert len(t2.context) == 1
        assert t2.context[0] is t1

    def test_guardrail_callable_accepted(self, agent: CrewAgent):
        t = CrewTask(
            description="x", expected_output="y", agent=agent,
            guardrail=lambda out: (True, out),
        )
        assert t.guardrail is not None

    def test_guardrail_string_accepted(self, agent: CrewAgent):
        t = CrewTask(
            description="x", expected_output="y", agent=agent,
            guardrail="Output must be one sentence",
        )
        assert isinstance(t.guardrail, str)

    def test_output_json_must_be_pydantic_model(self, agent: CrewAgent):
        class Schema(BaseModel):
            name: str
        t = CrewTask(description="x", expected_output="y", agent=agent, output_json=Schema)
        assert t.output_json is Schema

    def test_description_required(self, agent: CrewAgent):
        with pytest.raises(ValidationError):
            CrewTask(expected_output="y", agent=agent)  # type: ignore[call-arg]

    def test_frozen(self, agent: CrewAgent):
        t = CrewTask(description="x", expected_output="y", agent=agent)
        with pytest.raises(ValidationError):
            t.description = "mutated"  # type: ignore[misc]
```

- [ ] **Step 2: Run — expect failures**

- [ ] **Step 3: Implement `src/cognithor/crew/task.py`**

```python
"""CrewTask — declarative description of a unit of work."""

from __future__ import annotations

import uuid as _uuid
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from cognithor.crew.agent import CrewAgent


# A function-based guardrail: takes the raw output string (to keep the public
# API decoupled from TaskOutput) plus a context dict, returns (ok, feedback).
# The detailed GuardrailResult structure lives in Feature 4.
GuardrailCallable = Callable[[Any], tuple[bool, Any]]


class CrewTask(BaseModel):
    """Declarative unit of work executed by a CrewAgent."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    task_id: str = Field(default_factory=lambda: _uuid.uuid4().hex)
    description: str = Field(..., min_length=1)
    expected_output: str = Field(..., min_length=1)
    agent: CrewAgent
    context: list[CrewTask] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    guardrail: GuardrailCallable | str | None = None
    output_file: str | None = None
    output_json: type[BaseModel] | None = None
    async_execution: bool = False
    max_retries: int = Field(default=2, ge=0, le=10)


# Resolve the self-reference after the class is defined.
CrewTask.model_rebuild()
```

- [ ] **Step 4: Run — expect all tests pass**

- [ ] **Step 5: Ruff + commit**

```bash
git add src/cognithor/crew/task.py tests/test_crew/test_task.py
git commit -m "feat(crew): CrewTask Pydantic model with context and guardrail fields"
```

---

### Task 6: `Crew` class — construction only (kickoff landing in Task 8)

**Files:**
- Modify: `src/cognithor/crew/crew.py`
- Create: `tests/test_crew/test_crew.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_crew/test_crew.py
import pytest
from pydantic import ValidationError
from cognithor.crew import Crew, CrewAgent, CrewProcess, CrewTask


@pytest.fixture
def agent() -> CrewAgent:
    return CrewAgent(role="writer", goal="draft")


@pytest.fixture
def task(agent: CrewAgent) -> CrewTask:
    return CrewTask(description="x", expected_output="y", agent=agent)


class TestCrewConstruction:
    def test_minimal(self, agent: CrewAgent, task: CrewTask):
        c = Crew(agents=[agent], tasks=[task])
        assert len(c.agents) == 1
        assert c.process is CrewProcess.SEQUENTIAL
        assert c.verbose is False
        assert c.planning is False
        assert c.manager_llm is None

    def test_full(self, agent: CrewAgent, task: CrewTask):
        c = Crew(
            agents=[agent], tasks=[task],
            process=CrewProcess.HIERARCHICAL, verbose=True,
            planning=True, manager_llm="ollama/qwen3:32b",
        )
        assert c.process is CrewProcess.HIERARCHICAL
        assert c.manager_llm == "ollama/qwen3:32b"

    def test_rejects_empty_agents(self, task: CrewTask):
        with pytest.raises(ValidationError):
            Crew(agents=[], tasks=[task])

    def test_rejects_empty_tasks(self, agent: CrewAgent):
        with pytest.raises(ValidationError):
            Crew(agents=[agent], tasks=[])

    def test_hierarchical_without_manager_llm_warns(self, agent: CrewAgent, task: CrewTask):
        # Hierarchical mode without manager_llm is supported but emits a warning
        # because delegation quality suffers without a dedicated router model.
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Crew(agents=[agent], tasks=[task], process=CrewProcess.HIERARCHICAL)
        assert any("manager_llm" in str(w.message) for w in caught)
```

- [ ] **Step 2: Run — expect failures**

- [ ] **Step 3: Implement `src/cognithor/crew/crew.py`** (kickoff stub; real implementation in Task 8)

```python
"""Crew — top-level orchestration object."""

from __future__ import annotations

import warnings
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from cognithor.crew.agent import CrewAgent
from cognithor.crew.output import CrewOutput
from cognithor.crew.process import CrewProcess
from cognithor.crew.task import CrewTask


class Crew(BaseModel):
    """A Crew is a declarative bundle of agents + tasks + process."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    agents: list[CrewAgent] = Field(..., min_length=1)
    tasks: list[CrewTask] = Field(..., min_length=1)
    process: CrewProcess = CrewProcess.SEQUENTIAL
    verbose: bool = False
    planning: bool = False
    manager_llm: str | None = None

    @model_validator(mode="after")
    def _warn_on_hierarchical_without_manager(self) -> Crew:
        if self.process is CrewProcess.HIERARCHICAL and self.manager_llm is None:
            warnings.warn(
                "CrewProcess.HIERARCHICAL without manager_llm falls back to the "
                "first agent's llm for routing decisions. For production, set "
                "manager_llm explicitly.",
                stacklevel=2,
            )
        return self

    def kickoff(self, inputs: dict[str, Any] | None = None) -> CrewOutput:
        """Synchronous kickoff. Implemented in Task 8."""
        raise NotImplementedError("Crew.kickoff landing in Task 8 — Sequential compiler wiring")

    async def kickoff_async(self, inputs: dict[str, Any] | None = None) -> CrewOutput:
        """Async kickoff. Implemented in Task 9."""
        raise NotImplementedError("Crew.kickoff_async landing in Task 9")
```

- [ ] **Step 4: Run — expect 5 pass**

- [ ] **Step 5: Ruff + commit**

```bash
git add src/cognithor/crew/crew.py tests/test_crew/test_crew.py
git commit -m "feat(crew): Crew class construction + hierarchical-without-manager warning"
```

---

### Task 7: Tool resolution via MCP registry + "did you mean" suggestions

**Files:**
- Create: `src/cognithor/crew/tool_resolver.py`
- Create: `tests/test_crew/test_tool_resolution.py`

- [ ] **Step 1: Scout the real ToolRegistryDB**

The real class lives at `src/cognithor/mcp/tool_registry_db.py:848`:
```python
class ToolRegistryDB:
    def __init__(self, db_path: Path) -> None: ...
    def get_tool(self, name: str) -> ToolInfo | None: ...
    def get_tools_for_role(self, role: str, language: str = "de") -> list[ToolInfo]: ...
    def upsert_tool(self, name: str, ...): ...
```

There is **no** `list_tool_names()` method and **no** `get_tool_registry()` factory. The tool resolver wraps the real registry with a thin adapter so we don't leak DB details into `cognithor.crew`.

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_crew/test_tool_resolution.py
from unittest.mock import MagicMock
import pytest
from cognithor.crew.errors import ToolNotFoundError
from cognithor.crew.tool_resolver import resolve_tools, did_you_mean, available_tool_names


class TestAvailableToolNames:
    def test_uses_get_tools_for_role_all(self):
        """available_tool_names() pulls every tool regardless of role."""
        registry = MagicMock()
        tool_a = MagicMock(name="tool_a"); tool_a.name = "web_search"
        tool_b = MagicMock(name="tool_b"); tool_b.name = "pdf_reader"
        registry.get_tools_for_role.return_value = [tool_a, tool_b]
        names = available_tool_names(registry)
        registry.get_tools_for_role.assert_called_once_with("all")
        assert names == ["web_search", "pdf_reader"]


class TestResolveTools:
    def _registry_with(self, names: list[str]) -> MagicMock:
        registry = MagicMock()
        tools = []
        for n in names:
            m = MagicMock()
            m.name = n
            tools.append(m)
        registry.get_tools_for_role.return_value = tools
        return registry

    def test_resolves_known_tools(self):
        registry = self._registry_with(["web_search", "pdf_reader", "shell_run"])
        resolved = resolve_tools(["web_search", "pdf_reader"], registry=registry)
        assert resolved == ["web_search", "pdf_reader"]

    def test_unknown_tool_raises_with_suggestion(self):
        registry = self._registry_with(["web_search", "pdf_reader"])
        with pytest.raises(ToolNotFoundError) as exc:
            resolve_tools(["web_seach"], registry=registry)
        assert "web_seach" in str(exc.value)
        assert "web_search" in str(exc.value)

    def test_unknown_tool_no_close_match(self):
        registry = self._registry_with(["completely_other"])
        with pytest.raises(ToolNotFoundError) as exc:
            resolve_tools(["totally_foreign"], registry=registry)
        assert "totally_foreign" in str(exc.value)
        assert "Meintest du" not in str(exc.value)


class TestDidYouMean:
    def test_close_match(self):
        assert did_you_mean("web_seach", ["web_search", "pdf_reader"]) == "web_search"

    def test_no_match(self):
        assert did_you_mean("xyz", ["web_search"]) is None

    def test_exact_match_returns_none(self):
        # No suggestion when exact match exists
        assert did_you_mean("web_search", ["web_search"]) is None
```

- [ ] **Step 3: Run — expect failures**

- [ ] **Step 4: Implement `src/cognithor/crew/tool_resolver.py`**

```python
"""Resolve CrewAgent / CrewTask tool names against the MCP registry.

Wraps `cognithor.mcp.tool_registry_db.ToolRegistryDB` with the one helper
the Crew-Layer needs: 'give me every tool name'. The real registry groups
tools by role (planner/executor/browser/…) — we ask for role='all' to flatten.

Provides friendly 'did you mean' suggestions via difflib (stdlib, no new deps).
"""

from __future__ import annotations

import difflib
from typing import Any

from cognithor.crew.errors import ToolNotFoundError


def available_tool_names(registry: Any) -> list[str]:
    """Return every tool name known to the registry, flat.

    `registry` must be a `ToolRegistryDB` (or any duck-compatible object that
    exposes `get_tools_for_role(role: str) -> list[ToolInfo]` where each item
    has a `.name` attribute).
    """
    tools = registry.get_tools_for_role("all")
    return [t.name for t in tools]


def did_you_mean(name: str, candidates: list[str], cutoff: float = 0.6) -> str | None:
    """Return the closest match above cutoff, or None when nothing is close
    or when `name` is already in candidates.
    """
    if name in candidates:
        return None
    matches = difflib.get_close_matches(name, candidates, n=1, cutoff=cutoff)
    return matches[0] if matches else None


def resolve_tools(tool_names: list[str], *, registry: Any) -> list[str]:
    """Verify every tool name exists in the registry.

    Raises ToolNotFoundError on first unknown name, with a 'Meintest du ...?'
    suggestion when a close match exists.
    """
    available = available_tool_names(registry)
    for name in tool_names:
        if name in available:
            continue
        suggestion = did_you_mean(name, available)
        hint = f" Meintest du '{suggestion}'?" if suggestion else ""
        raise ToolNotFoundError(f"Tool '{name}' nicht in der Registry.{hint}")
    return tool_names
```

- [ ] **Step 4: Run — expect all pass**

- [ ] **Step 5: Ruff + commit**

```bash
git add src/cognithor/crew/tool_resolver.py tests/test_crew/test_tool_resolution.py
git commit -m "feat(crew): tool resolver with did-you-mean suggestions"
```

---

### Task 8: `Crew.kickoff()` sequential happy-path

**Files:**
- Create: `src/cognithor/crew/compiler.py`
- Create: `src/cognithor/crew/compiler_hierarchical.py` (stub only — real impl in Task 10)
- Modify: `src/cognithor/crew/crew.py`
- Create: `tests/test_crew/test_sequential_kickoff.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_crew/test_sequential_kickoff.py
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from cognithor.crew import Crew, CrewAgent, CrewProcess, CrewTask
from cognithor.crew.output import TaskOutput


@pytest.fixture
def researcher() -> CrewAgent:
    return CrewAgent(role="researcher", goal="research", llm="ollama/qwen3:8b")


@pytest.fixture
def writer() -> CrewAgent:
    return CrewAgent(role="writer", goal="write", llm="ollama/qwen3:8b")


class TestSequentialKickoff:
    def test_two_tasks_run_in_order(self, researcher: CrewAgent, writer: CrewAgent):
        t1 = CrewTask(description="research topic", expected_output="facts", agent=researcher)
        t2 = CrewTask(description="write report", expected_output="report", agent=writer, context=[t1])
        crew = Crew(agents=[researcher, writer], tasks=[t1, t2], process=CrewProcess.SEQUENTIAL)

        fake_outputs = [
            TaskOutput(task_id=t1.task_id, agent_role="researcher", raw="FACTS ABOUT TOPIC"),
            TaskOutput(task_id=t2.task_id, agent_role="writer", raw="REPORT DRAFT"),
        ]

        with patch("cognithor.crew.compiler.execute_task", side_effect=fake_outputs) as mocked:
            result = crew.kickoff()

        assert result.raw == "REPORT DRAFT"
        assert len(result.tasks_output) == 2
        assert result.trace_id
        # Sequential ordering: first call is t1, second is t2
        assert mocked.call_args_list[0].args[0].task_id == t1.task_id
        assert mocked.call_args_list[1].args[0].task_id == t2.task_id

    def test_inputs_threaded_into_first_task(self, researcher: CrewAgent):
        t1 = CrewTask(description="research {topic}", expected_output="facts", agent=researcher)
        crew = Crew(agents=[researcher], tasks=[t1])

        captured: list = []

        def spy(task, *, context, inputs, registry):
            captured.append(inputs)
            return TaskOutput(task_id=task.task_id, agent_role=task.agent.role, raw="OK")

        with patch("cognithor.crew.compiler.execute_task", side_effect=spy):
            crew.kickoff(inputs={"topic": "PKV tariffs"})

        assert captured[0] == {"topic": "PKV tariffs"}
```

- [ ] **Step 2: Run — expect NotImplementedError from Task 6 stub**

- [ ] **Step 3: Create `src/cognithor/crew/compiler_hierarchical.py` as an import-resolvable stub**

Task 8's compiler `else` branch imports `order_tasks_hierarchical` from this module. Task 10 replaces it with the real implementation, but to keep imports resolvable in the meantime we ship a deterministic declaration-order fallback here:

```python
"""Hierarchical compiler stub.

Task 8-9 may route HIERARCHICAL-process Crews through this module's entry
point before Task 10 lands the real manager-LLM integration. The stub
returns declaration order so imports resolve and HIERARCHICAL Crews run
deterministically without a manager. Task 10 replaces this wholesale.
"""

from __future__ import annotations

from typing import Any

from cognithor.crew.agent import CrewAgent
from cognithor.crew.task import CrewTask


def order_tasks_hierarchical(
    tasks: list[CrewTask],
    agents: list[CrewAgent],
    **_: Any,
) -> list[CrewTask]:
    """Stub: declaration order. Real manager-LLM routing lands in Task 10."""
    return list(tasks)
```

- [ ] **Step 4: Implement `src/cognithor/crew/compiler.py`**

```python
"""Compiler translates Crew definitions into ordered execution steps that
route through the existing Planner/Gatekeeper pipeline.

The compiler itself is a pure function; the `execute_task` helper is where
the actual PGE integration happens (Task 11). For the happy path in Task 8
we only need ordered traversal.
"""

from __future__ import annotations

import uuid as _uuid
from typing import Any

from cognithor.crew.agent import CrewAgent
from cognithor.crew.output import CrewOutput, TaskOutput
from cognithor.crew.process import CrewProcess
from cognithor.crew.task import CrewTask


def order_tasks_sequential(tasks: list[CrewTask]) -> list[CrewTask]:
    """Sequential process: keep the declaration order."""
    return list(tasks)


def execute_task(
    task: CrewTask,
    *,
    context: list[TaskOutput],
    inputs: dict[str, Any] | None,
    registry: Any,
) -> TaskOutput:
    """Route one task through the PGE pipeline.

    Stub for Task 8 — the real PGE wiring lands in Task 11. The stub raises
    NotImplementedError so that the unit test at Task 8 is forced to patch
    this function (the test does). Integration happens in Task 11 where the
    patch target becomes a real call site."""
    raise NotImplementedError(
        "execute_task is stubbed in Task 8; real PGE wiring arrives in Task 11. "
        "Tests must patch 'cognithor.crew.compiler.execute_task' until then."
    )


# Guardrails land in Feature 4 (PR 2). Between PR 1 (this file) shipping and
# PR 2 landing on the user's install, a CrewTask with `guardrail=<anything>`
# would silently do nothing — the user gets no safety they expected. Guard
# against that foot-gun by probing the guardrails module at import time and
# emitting a UserWarning if a task declares a guardrail on a version that
# can't execute it. Removed in Task 21 when the real apply path lands.
try:
    from cognithor.crew.guardrails import base as _guardrails_base  # noqa: F401
    _guardrails_available = True
except ImportError:
    _guardrails_available = False


def _warn_if_guardrail_silently_ignored(task: CrewTask) -> None:
    """PR 1 → PR 2 bridge guard. Removed in Task 21."""
    import warnings
    if task.guardrail is not None and not _guardrails_available:
        warnings.warn(
            f"CrewTask '{task.task_id}' has a guardrail but "
            "cognithor.crew.guardrails is not available in this release. "
            "The guardrail will be IGNORED. Upgrade to cognithor>=0.93.1 "
            "(or install via `pip install cognithor[all]`) to enable guardrails.",
            UserWarning,
            stacklevel=3,
        )


def compile_and_run_sync(
    agents: list[CrewAgent],
    tasks: list[CrewTask],
    process: CrewProcess,
    inputs: dict[str, Any] | None,
    registry: Any,
) -> CrewOutput:
    """Synchronous compiler + runner.

    Sequential: straight linear order. Hierarchical: Task 10.
    """
    if process is CrewProcess.SEQUENTIAL:
        ordered = order_tasks_sequential(tasks)
    else:
        from cognithor.crew.compiler_hierarchical import order_tasks_hierarchical
        ordered = order_tasks_hierarchical(tasks, agents)

    trace_id = _uuid.uuid4().hex
    outputs: list[TaskOutput] = []
    for t in ordered:
        _warn_if_guardrail_silently_ignored(t)  # PR 1 → PR 2 bridge guard
        out = execute_task(t, context=outputs, inputs=inputs, registry=registry)
        outputs.append(out)
    return CrewOutput(raw=outputs[-1].raw, tasks_output=outputs, trace_id=trace_id)
```

- [ ] **Step 5: Wire into `Crew.kickoff()`**

The real `ToolRegistryDB(db_path: Path)` requires a DB file; there is no module-level singleton. We therefore introduce a tiny factory helper `cognithor.crew.runtime.get_default_tool_registry()` that builds one from config (Task 11 expands this helper to also return a Planner). For Task 8 it's minimal:

```python
# src/cognithor/crew/runtime.py  (initial version — expanded in Task 11)
"""Runtime helpers for Crew.kickoff() / kickoff_async()."""

from __future__ import annotations

import threading
from typing import Any

_registry_lock = threading.Lock()
_registry_singleton: Any = None


def get_default_tool_registry() -> Any:
    """Return a process-wide default ToolRegistryDB instance.

    Builds from `cognithor.config.load_config().cognithor_home / 'db' /
    'tool_registry.db'`. Implementers: if config loading fails (e.g. standalone
    test without ~/.cognithor/ present), fall back to a temp-dir DB and log a
    warning — never silently return None.
    """
    global _registry_singleton
    with _registry_lock:
        if _registry_singleton is not None:
            return _registry_singleton
        from pathlib import Path
        from cognithor.config import load_config
        from cognithor.mcp.tool_registry_db import ToolRegistryDB

        try:
            cfg = load_config()
            db_path = Path(cfg.cognithor_home) / "db" / "tool_registry.db"
        except Exception:
            import tempfile
            db_path = Path(tempfile.gettempdir()) / "cognithor_crew_registry.db"
        _registry_singleton = ToolRegistryDB(db_path=db_path)
        return _registry_singleton
```

Then `Crew.kickoff()`:

```python
def kickoff(self, inputs: dict[str, Any] | None = None) -> CrewOutput:
    import asyncio
    # Sync kickoff is NOT safe from inside a running event loop (pytest-asyncio
    # mode=auto, Gateway, etc). Detect and redirect to the explicit async entry.
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop — safe to asyncio.run().
        return asyncio.run(self.kickoff_async(inputs))
    raise RuntimeError(
        "Crew.kickoff() called from within a running event loop. "
        "Use `await crew.kickoff_async(inputs)` instead."
    )
```

(The async helper is wired in Task 9; for Task 8 alone, the sync wrapper can temporarily call a sync-only `compile_and_run_sync` — once Task 9 lands, the sync path becomes the `asyncio.run()` trampoline above. Document the interim behaviour in the Task 8 commit body.)

- [ ] **Step 6: Run — expect 2 pass**

- [ ] **Step 7: Ruff + commit**

```bash
git add src/cognithor/crew/compiler.py src/cognithor/crew/compiler_hierarchical.py src/cognithor/crew/crew.py src/cognithor/crew/runtime.py tests/test_crew/test_sequential_kickoff.py
git commit -m "feat(crew): sequential compile-and-run happy path + hierarchical import stub"
```

---

### Task 9: `Crew.kickoff_async()` and async execution

**Files:**
- Modify: `src/cognithor/crew/compiler.py` (add `compile_and_run_async`)
- Modify: `src/cognithor/crew/crew.py`
- Create: `tests/test_crew/test_async_kickoff.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_crew/test_async_kickoff.py
from unittest.mock import patch, AsyncMock
import pytest
from cognithor.crew import Crew, CrewAgent, CrewTask
from cognithor.crew.output import TaskOutput


async def test_kickoff_async_returns_same_as_sync():
    agent = CrewAgent(role="x", goal="y")
    task = CrewTask(description="a", expected_output="b", agent=agent)
    crew = Crew(agents=[agent], tasks=[task])

    fake = TaskOutput(task_id=task.task_id, agent_role="x", raw="DONE")

    with patch("cognithor.crew.compiler.execute_task_async", new=AsyncMock(return_value=fake)):
        result = await crew.kickoff_async()

    assert result.raw == "DONE"
    assert len(result.tasks_output) == 1


async def test_async_tasks_run_concurrently_when_no_dependency():
    agent = CrewAgent(role="x", goal="y")
    t1 = CrewTask(description="a", expected_output="b", agent=agent, async_execution=True)
    t2 = CrewTask(description="c", expected_output="d", agent=agent, async_execution=True)
    crew = Crew(agents=[agent], tasks=[t1, t2])

    import asyncio
    call_times: list[float] = []

    async def timed(task, context, inputs, registry):
        # asyncio.get_event_loop() is deprecated on Python 3.12 when there's a
        # running loop; inside an async fn we're guaranteed a running loop, so
        # get_running_loop() is the safe call.
        call_times.append(asyncio.get_running_loop().time())
        await asyncio.sleep(0.05)
        return TaskOutput(task_id=task.task_id, agent_role="x", raw="OK")

    with patch("cognithor.crew.compiler.execute_task_async", side_effect=timed):
        await crew.kickoff_async()

    # Two async-marked tasks with no dependency start within ~10 ms of each other
    assert abs(call_times[0] - call_times[1]) < 0.01
```

- [ ] **Step 2: Implement `compile_and_run_async` in `compiler.py`**

```python
import asyncio

async def execute_task_async(task, *, context, inputs, registry):
    """Async counterpart of execute_task. Real PGE wiring in Task 11."""
    raise NotImplementedError(
        "execute_task_async is stubbed in Task 9; real PGE wiring arrives in Task 11."
    )


async def compile_and_run_async(agents, tasks, process, inputs, registry):
    if process is CrewProcess.SEQUENTIAL:
        ordered = order_tasks_sequential(tasks)
    else:
        from cognithor.crew.compiler_hierarchical import order_tasks_hierarchical
        ordered = order_tasks_hierarchical(tasks, agents)

    trace_id = _uuid.uuid4().hex
    outputs: list[TaskOutput] = []
    # PR 1 → PR 2 bridge: warn on any silently-ignored guardrail before entering
    # the fan-out loop (single pass; warnings filter dedupes by call site).
    for t in ordered:
        _warn_if_guardrail_silently_ignored(t)
    i = 0
    while i < len(ordered):
        # Collect a fan-out group: consecutive tasks with async_execution=True
        # and no dependency on each other.
        group = [ordered[i]]
        j = i + 1
        while j < len(ordered) and ordered[j].async_execution:
            # Only group if the later task doesn't depend on earlier group members
            deps = {t.task_id for t in ordered[j].context}
            if deps.isdisjoint({t.task_id for t in group}):
                group.append(ordered[j])
                j += 1
            else:
                break
        if len(group) == 1:
            out = await execute_task_async(group[0], context=outputs, inputs=inputs, registry=registry)
            outputs.append(out)
        else:
            parallel_outs = await asyncio.gather(
                *[execute_task_async(t, context=outputs, inputs=inputs, registry=registry) for t in group]
            )
            outputs.extend(parallel_outs)
        i = j if len(group) > 1 else i + 1
    return CrewOutput(raw=outputs[-1].raw, tasks_output=outputs, trace_id=trace_id)
```

- [ ] **Step 3: Wire into `Crew.kickoff_async()`**

```python
async def kickoff_async(self, inputs: dict[str, Any] | None = None) -> CrewOutput:
    from cognithor.crew.compiler import compile_and_run_async
    from cognithor.crew.runtime import get_default_tool_registry
    return await compile_and_run_async(
        agents=self.agents,
        tasks=self.tasks,
        process=self.process,
        inputs=inputs,
        registry=get_default_tool_registry(),
    )
```

- [ ] **Step 4: Run + commit**

```bash
python -m pytest tests/test_crew/test_async_kickoff.py -v
git add src/cognithor/crew/compiler.py src/cognithor/crew/crew.py tests/test_crew/test_async_kickoff.py
git commit -m "feat(crew): async kickoff with parallel fan-out for async_execution=True"
```

---

### Task 10: Hierarchical process with manager_llm

**Files:**
- Create: `src/cognithor/crew/compiler_hierarchical.py`
- Create: `tests/test_crew/test_hierarchical_kickoff.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_crew/test_hierarchical_kickoff.py
from unittest.mock import patch
import pytest
from cognithor.crew import Crew, CrewAgent, CrewProcess, CrewTask
from cognithor.crew.output import TaskOutput


class TestHierarchical:
    def test_manager_agent_is_synthesized(self):
        """Hierarchical process injects a synthetic 'manager' agent that picks
        which worker handles each task. Worker order is NOT necessarily
        declaration order."""
        analyst = CrewAgent(role="analyst", goal="analyze")
        writer = CrewAgent(role="writer", goal="write")
        t1 = CrewTask(description="produce a PKV summary", expected_output="x", agent=analyst)
        t2 = CrewTask(description="polish the summary into a customer-facing report", expected_output="y", agent=writer)
        crew = Crew(
            agents=[analyst, writer], tasks=[t1, t2],
            process=CrewProcess.HIERARCHICAL, manager_llm="ollama/qwen3:32b",
        )

        # The manager decides order — we force it to pick writer before analyst
        # by stubbing the delegation module to return reversed order.
        from cognithor.crew.compiler_hierarchical import order_tasks_hierarchical
        reordered = order_tasks_hierarchical(crew.tasks, crew.agents, manager_llm="ollama/qwen3:32b")
        # The default fallback — no live LLM — returns declaration order.
        assert [t.task_id for t in reordered] == [t1.task_id, t2.task_id]
```

- [ ] **Step 2: Implement `compiler_hierarchical.py`**

```python
"""Hierarchical compiler: inserts a synthetic manager agent that picks
execution order for each task. When manager_llm is not available (offline
tests, no model set), falls back to declaration order to keep behaviour
deterministic.
"""

from __future__ import annotations

from cognithor.crew.agent import CrewAgent
from cognithor.crew.task import CrewTask


def order_tasks_hierarchical(
    tasks: list[CrewTask],
    agents: list[CrewAgent],
    *,
    manager_llm: str | None = None,
) -> list[CrewTask]:
    """Return tasks in the order the manager agent chose.

    Deterministic fallback: when no manager_llm is set or the delegation
    module is unavailable, return the declaration order. Production
    hierarchical routing uses the existing `cognithor.core.delegation`
    module — wiring arrives in Task 11 once the PGE integration lands.
    """
    if manager_llm is None:
        return list(tasks)

    # Placeholder — integration with cognithor.core.delegation lands in Task 11.
    # For now the offline default is identical to sequential. This keeps the
    # test contract tight while leaving the wiring-point explicit.
    return list(tasks)
```

- [ ] **Step 3: Run + commit**

```bash
python -m pytest tests/test_crew/test_hierarchical_kickoff.py -v
git add src/cognithor/crew/compiler_hierarchical.py tests/test_crew/test_hierarchical_kickoff.py
git commit -m "feat(crew): hierarchical compiler scaffolding with deterministic fallback"
```

---

### Task 11: PGE-Trinity integration — real `execute_task` via Planner

**Files:**
- Modify: `src/cognithor/crew/compiler.py` (replace stubbed `execute_task` + `execute_task_async`)
- Modify: `src/cognithor/crew/crew.py` (accept an explicit Planner instance)
- Modify: `src/cognithor/crew/runtime.py` (add `get_default_planner` factory)
- Create: `tests/test_crew/test_pge_integration.py`

- [ ] **Step 1: Scouted Planner API (verified against `src/cognithor/core/planner.py`)**

**Constructor (planner.py:482):**
```python
def __init__(
    self,
    config: CognithorConfig,
    ollama: Any,                    # OllamaClient or UnifiedLLMClient
    model_router: ModelRouter,
    audit_logger: AuditLogger | None = None,
    causal_analyzer: Any = None,
    task_profiler: Any = None,
    cost_tracker: Any = None,
    personality_engine: Any = None,
    prompt_evolution: Any = None,
) -> None: ...
```

All three of `config`, `ollama`, `model_router` are **required** positional arguments. The plan's earlier `Planner(config=cfg)` would raise `TypeError`.

**`formulate_response` (planner.py:1031):**
```python
async def formulate_response(
    self,
    user_message: str,
    results: list[ToolResult],
    working_memory: WorkingMemory,
) -> ResponseEnvelope: ...
```

Returns a `ResponseEnvelope` (from `cognithor.core.observer`) with fields `.content: str` and `.directive: PGEReloopDirective | None`. **There is no `.usage`.** Token counts come from the Ollama `chat` response dicts consumed internally (`prompt_eval_count` / `eval_count`) and are only exposed externally through `cost_tracker.record_llm_call()` — which the Planner invokes itself. The Crew-Layer reads token usage from a `CostTracker` passed into the Planner (if present), not from the envelope.

**`WorkingMemory` (models.py:478)** is a Pydantic model with `session_id`, `chat_history`, `tool_results`, etc. Minimum viable construction: `WorkingMemory()` — all fields have defaults.

**`ToolResult` (models.py:257)** is a frozen Pydantic model with `tool_name`, `content`, `is_error`, etc. To thread Crew-context as "prior tool results" we can synthesize `ToolResult` entries with `tool_name="crew_context"` and `content=prior_task_output`.

**There is no `get_running_gateway()` function** in `gateway.py`. The Crew-Layer avoids auto-discovery entirely: callers pass an explicit Planner instance (either from a live Gateway's `gateway._planner` attribute or constructed via the factory below).

- [ ] **Step 2: Write the failing integration test**

```python
# tests/test_crew/test_pge_integration.py
from unittest.mock import AsyncMock, MagicMock
import pytest
from cognithor.crew import Crew, CrewAgent, CrewTask
from cognithor.crew.compiler import execute_task_async
from cognithor.core.observer import ResponseEnvelope


async def test_execute_task_routes_through_planner():
    """The real execute_task_async must: (a) construct a user_message + WorkingMemory,
    (b) call Planner.formulate_response(user_message, results, working_memory),
    (c) return a TaskOutput with the planner's content."""
    agent = CrewAgent(role="writer", goal="write", llm="ollama/qwen3:8b")
    task = CrewTask(description="Write a haiku", expected_output="three lines", agent=agent)

    mock_planner = MagicMock()
    mock_planner.formulate_response = AsyncMock(
        return_value=ResponseEnvelope(
            content="First line / Second line / Third line",
            directive=None,
        )
    )

    # Registry adapter returning a stable tool list via get_tools_for_role("all")
    mock_registry = MagicMock()
    mock_registry.get_tools_for_role.return_value = []

    out = await execute_task_async(
        task, context=[], inputs=None, registry=mock_registry, planner=mock_planner,
    )
    assert out.task_id == task.task_id
    assert out.agent_role == "writer"
    assert out.raw == "First line / Second line / Third line"

    # Planner was called with (user_message, results, working_memory) — positional
    call = mock_planner.formulate_response.call_args
    args = call.args if call.args else (call.kwargs.get("user_message"),
                                         call.kwargs.get("results"),
                                         call.kwargs.get("working_memory"))
    assert "Write a haiku" in args[0]
    assert isinstance(args[1], list)  # results list


async def test_execute_task_passes_context_as_prior_tool_results():
    """Prior TaskOutputs become synthetic ToolResult entries."""
    agent = CrewAgent(role="writer", goal="write")
    t1 = CrewTask(description="research", expected_output="facts", agent=agent)
    t2 = CrewTask(description="write report", expected_output="text", agent=agent, context=[t1])

    from cognithor.crew.output import TaskOutput
    prior = [TaskOutput(task_id=t1.task_id, agent_role="writer", raw="FACTS_HERE")]

    mock_planner = MagicMock()
    mock_planner.formulate_response = AsyncMock(
        return_value=ResponseEnvelope(content="REPORT", directive=None),
    )
    mock_registry = MagicMock()
    mock_registry.get_tools_for_role.return_value = []

    await execute_task_async(
        t2, context=prior, inputs=None, registry=mock_registry, planner=mock_planner,
    )

    call = mock_planner.formulate_response.call_args
    args = call.args if call.args else (
        call.kwargs["user_message"], call.kwargs["results"], call.kwargs["working_memory"],
    )
    # Prior output appears as a ToolResult
    results = args[1]
    assert any("FACTS_HERE" in r.content for r in results)


async def test_execute_task_token_usage_from_cost_tracker():
    """Token usage is read from the Planner's CostTracker sidecar, not from the envelope.

    The real CostTracker.last_call() returns a CostRecord with
    `input_tokens`/`output_tokens` attributes; the Crew-Layer maps those to
    prompt_tokens/completion_tokens/total_tokens for TokenUsageDict.
    """
    agent = CrewAgent(role="writer", goal="write")
    task = CrewTask(description="x", expected_output="y", agent=agent)

    mock_planner = MagicMock()
    mock_planner.formulate_response = AsyncMock(
        return_value=ResponseEnvelope(content="ok", directive=None),
    )
    # CostRecord-shaped sidecar — matches the real
    # cognithor.models.CostRecord fields (input_tokens, output_tokens).
    from types import SimpleNamespace
    record = SimpleNamespace(input_tokens=42, output_tokens=7)
    tracker = MagicMock()
    tracker.last_call.return_value = record
    mock_planner._cost_tracker = tracker
    mock_registry = MagicMock()
    mock_registry.get_tools_for_role.return_value = []

    out = await execute_task_async(
        task, context=[], inputs=None, registry=mock_registry, planner=mock_planner,
    )
    assert out.token_usage == {
        "prompt_tokens": 42, "completion_tokens": 7, "total_tokens": 49,
    }
```

- [ ] **Step 3: Implement real `execute_task_async` in `compiler.py`**

Replace the stub with:

```python
from cognithor.crew.tool_resolver import resolve_tools
from cognithor.models import ToolResult, WorkingMemory


async def execute_task_async(
    task: CrewTask,
    *,
    context: list[TaskOutput],
    inputs: dict[str, Any] | None,
    registry: Any,
    planner: Any,
    trace_id: str | None = None,  # R3-NI3-partial / R4: kickoff-level correlation id
) -> TaskOutput:
    """Route one task through the Planner (which internally goes through
    Gatekeeper + Executor).

    Spec §1.6: the Crew-Layer must NOT bypass the Planner. Every task builds
    a proper WorkingMemory + ToolResult-list and calls
    Planner.formulate_response(user_message, results, working_memory).

    ``trace_id`` is plumbed from the kickoff (see Task 30) so every in-kickoff
    tool result / chat turn / audit event bucket under one audit session and
    concurrent kickoffs stay isolated. If omitted (pre-Task-30 call sites),
    a fresh UUID is minted.
    """
    import time
    import uuid

    # Resolve tools up-front so the error is raised before any LLM call
    agent_tools = resolve_tools(task.agent.tools, registry=registry)
    task_tools = resolve_tools(task.tools, registry=registry)
    all_tools = list({*agent_tools, *task_tools})  # currently informational only

    # Build the final user-message (description + inputs + expected_output)
    user_message = _build_user_message(task, inputs)

    # Synthesize prior-task outputs as ToolResult entries. Planner.formulate_response
    # treats `results` as the evidence it summarizes; this is exactly the channel
    # we need for cross-task context.
    prior_results: list[ToolResult] = [
        ToolResult(
            tool_name=f"crew_context__{prior.agent_role}",
            content=prior.raw,
            is_error=False,
        )
        for prior in context
    ]

    # WorkingMemory MUST carry a session_id — using the kickoff's trace_id
    # keeps every in-kickoff tool result / chat turn bucketed under one audit
    # session. Without it, the default `session_id=""` collapses all concurrent
    # kickoffs into the same audit bucket and taints cross-request isolation.
    # See NI3 (Round 3) and R3-NI3-partial (Round 4 — ``locals().get("trace_id")``
    # was always ``None`` because ``trace_id`` was never a local; now it's a
    # real kwarg).
    session_id = trace_id or uuid.uuid4().hex
    working_memory = WorkingMemory(session_id=session_id)

    t0 = time.perf_counter()
    envelope = await planner.formulate_response(
        user_message,
        prior_results,
        working_memory,
    )
    duration_ms = (time.perf_counter() - t0) * 1000.0

    raw = getattr(envelope, "content", "") or ""
    # Token usage via the Planner's cost tracker (if available). The tracker is
    # optional — fall back to zeros if not wired. Real CostTracker API scout
    # (confirmed 2026-04-24 against `src/cognithor/telemetry/cost_tracker.py`):
    #
    #   class CostTracker:
    #       def record_llm_call(self, model, input_tokens, output_tokens,
    #                           session_id="", agent_name="") -> CostRecord
    #       def get_session_cost(self, session_id: str) -> float
    #       def get_agent_costs(self, days=1) -> dict[str, float]
    #       # NO `last_call()` method exists today.
    #
    # Planner calls `self._cost_tracker.record_llm_call(...)` inside
    # `_record_cost()` at `planner.py:615` — the CostRecord is returned but
    # the Planner discards it. To expose the last call's token counts to the
    # Crew-Layer without plumbing the CostRecord through ResponseEnvelope, we
    # add a tiny additive helper (`last_call()`) to CostTracker in Step 5 —
    # a non-breaking read-only accessor backed by a single `self._last_record`
    # attribute set inside `record_llm_call()`. The duck-typed probe below
    # works correctly both with and without that helper: if absent, we return
    # None and upstream defaults to zeros.
    usage = _read_token_usage(planner) or TokenUsageDict(
        prompt_tokens=0, completion_tokens=0, total_tokens=0,
    )

    return TaskOutput(
        task_id=task.task_id,
        agent_role=task.agent.role,
        raw=raw,
        duration_ms=duration_ms,
        token_usage=usage,
    )


def _read_token_usage(planner: Any) -> TokenUsageDict | None:
    """Pull the last-call token count from the planner's cost tracker.

    Step 5 of this task adds a `last_call() -> CostRecord | None` method to
    `cognithor.telemetry.cost_tracker.CostTracker` (additive, non-breaking).
    This probe gracefully degrades when that method is missing:
    returns None and upstream defaults to zeros, so the Crew-Layer still
    functions against an older CostTracker build.
    """
    tracker = getattr(planner, "_cost_tracker", None)
    if tracker is None:
        return None
    last = getattr(tracker, "last_call", None)
    if not callable(last):
        return None
    try:
        record = last()
    except Exception:
        return None
    if record is None:
        return None
    # CostRecord is a Pydantic model with input_tokens / output_tokens fields.
    # Duck-type read to stay compatible with any future record-shape changes.
    input_tokens = int(getattr(record, "input_tokens", 0))
    output_tokens = int(getattr(record, "output_tokens", 0))
    return TokenUsageDict(
        prompt_tokens=input_tokens,
        completion_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )


def execute_task(
    task: CrewTask,
    *,
    context: list[TaskOutput],
    inputs: dict[str, Any] | None,
    registry: Any,
    planner: Any | None = None,
) -> TaskOutput:
    """Synchronous wrapper around ``execute_task_async``.

    Refuses to run from inside a running event loop — same guard as
    ``Crew.kickoff()`` — because ``asyncio.run`` cannot be called from an
    already-running loop and would otherwise raise a confusing
    ``RuntimeError: asyncio.run() cannot be called from a running event loop``.
    See NI2 in Round 3 review.
    """
    import asyncio
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass  # no running loop — safe to use asyncio.run
    else:
        raise RuntimeError(
            "execute_task() cannot be called from a running event loop. "
            "Use `await execute_task_async(...)` instead."
        )
    return asyncio.run(execute_task_async(
        task, context=context, inputs=inputs, registry=registry, planner=planner,
    ))


def _build_user_message(
    task: CrewTask,
    inputs: dict[str, Any] | None,
) -> str:
    """Render the Crew task as a single user message.

    System-level framing (role, goal, backstory) is owned by the Planner via
    its own SYSTEM_PROMPT — the Crew-Layer intentionally does NOT inject its
    own system prompt, to avoid duplicating Cognithor's identity framing.
    Agent role + backstory are included in the user-message as task framing.
    """
    parts: list[str] = []
    # Agent framing (lightweight — the Planner has its own system prompt)
    parts.append(f"[Crew role: {task.agent.role}] goal: {task.agent.goal}")
    if task.agent.backstory:
        parts.append(f"Background: {task.agent.backstory}")
    parts.append("")
    desc = task.description
    if inputs:
        for k, v in inputs.items():
            desc = desc.replace("{" + str(k) + "}", str(v))
    parts.append(desc)
    parts.append(f"\nExpected output: {task.expected_output}")
    return "\n".join(parts)
```

Update `compile_and_run_sync` + `compile_and_run_async` to accept and thread a `planner` argument. Update `Crew.kickoff_async()` to pull a Planner (explicit instance or factory — see Step 4).

- [ ] **Step 4: Wire Planner into `Crew.kickoff_async()`**

The Crew class gains a `planner` constructor kwarg for explicit injection (test / embedded callers) and falls back to `runtime.get_default_planner()` otherwise:

```python
# In crew.py — extend the Crew Pydantic model with a private planner field
class Crew(BaseModel):
    # ... existing fields ...
    # Note: planner is NOT a Pydantic field — it's assigned via __init__ override
    # because it's a live object, not a declarative config.
    _planner: Any = None

    def __init__(self, *, planner: Any = None, **kwargs) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "_planner", planner)

    async def kickoff_async(self, inputs: dict[str, Any] | None = None) -> CrewOutput:
        from cognithor.crew.compiler import compile_and_run_async
        from cognithor.crew.runtime import get_default_planner, get_default_tool_registry

        planner = self._planner or get_default_planner()
        return await compile_and_run_async(
            agents=self.agents,
            tasks=self.tasks,
            process=self.process,
            inputs=inputs,
            registry=get_default_tool_registry(),
            planner=planner,
        )

    def kickoff(self, inputs: dict[str, Any] | None = None) -> CrewOutput:
        import asyncio
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.kickoff_async(inputs))
        raise RuntimeError(
            "Crew.kickoff() called from within a running event loop. "
            "Use `await crew.kickoff_async(inputs)` instead."
        )
```

Extend `src/cognithor/crew/runtime.py` from Task 8 with the Planner factory.

**Async-safety:** Planner construction reads config, opens an Ollama HTTP client, and initializes the model router — all of which can take tens of milliseconds. Holding a `threading.Lock()` across that work blocks the asyncio event loop when `kickoff_async` is the first call. Build the candidate OUTSIDE the lock; the lock only guards the swap-in sentinel. This is the standard "double-checked-lock with pre-built candidate" pattern — safe because `Planner.__init__` is pure setup (no mutation of shared state) so two parallel builds are harmless, and only one survives the lock.

```python
# Appended to runtime.py
_planner_lock = threading.Lock()
_planner_singleton: Any = None


def get_default_planner() -> Any:
    """Return a process-wide default Planner instance.

    No auto-discovery: we always build a fresh Planner from config for
    standalone Crew scripts. Embedded callers (Gateway, tests) pass a live
    Planner to `Crew(planner=...)` instead of relying on this factory.

    Async-safe: the expensive construction happens OUTSIDE the threading.Lock,
    so this function never blocks an async event loop for meaningful time.
    The lock guards only the final sentinel swap.
    """
    global _planner_singleton
    # Fast path — no lock needed once the singleton exists. Plain attribute
    # read is atomic in CPython.
    if _planner_singleton is not None:
        return _planner_singleton

    # Build the candidate OUTSIDE the lock. If two threads / two coroutines
    # race here, both build a Planner — fine, they're pure constructors — and
    # the second one is GC'd after the sentinel check below.
    from cognithor.config import load_config
    from cognithor.core.model_router import ModelRouter, OllamaClient
    from cognithor.core.planner import Planner

    cfg = load_config()
    ollama = OllamaClient(cfg)
    router = ModelRouter(cfg, ollama)
    candidate = Planner(cfg, ollama, router)

    # Swap-in under lock; discard the candidate if another thread beat us.
    with _planner_lock:
        if _planner_singleton is None:
            _planner_singleton = candidate
        return _planner_singleton
```

- [ ] **Step 5: Add additive `last_call()` helper to the real CostTracker**

**Scouted (2026-04-24):** `class CostTracker` lives at `src/cognithor/telemetry/cost_tracker.py`. Exposed methods: `record_llm_call() -> CostRecord`, `get_session_cost()`, `get_daily_cost()`, `get_monthly_cost()`, `get_agent_costs()`, `check_budget()`, `check_agent_budget()`, `get_cost_report()`, `get_budget_info()`, `close()`. **No `last_call()` method.** `record_llm_call` returns the `CostRecord` but the Planner's `_record_cost()` discards the return value.

Apply a small additive patch — zero behavior change for existing callers, new accessor for the Crew-Layer:

```python
# src/cognithor/telemetry/cost_tracker.py
class CostTracker:
    def __init__(self, db_path: str, daily_budget: float = 0.0, monthly_budget: float = 0.0) -> None:
        # ... existing init ...
        self._last_record: CostRecord | None = None  # NEW: most recent record_llm_call

    def record_llm_call(self, model, input_tokens, output_tokens, session_id="", agent_name=""):
        # ... existing body builds `record` and inserts into SQLite ...
        self._last_record = record  # NEW: remember the last record returned
        return record

    def last_call(self) -> CostRecord | None:  # NEW
        """Return the most recent CostRecord from record_llm_call(), or None.

        Read-only accessor used by cognithor.crew to surface token usage per
        Crew-Task without plumbing the record through ResponseEnvelope.
        Not persisted — only the in-memory last-record is returned.
        """
        return self._last_record
```

**Test (add to `tests/test_telemetry/test_cost_tracker.py` — create if absent):**

```python
def test_last_call_returns_last_record(tmp_path):
    from cognithor.telemetry.cost_tracker import CostTracker
    tracker = CostTracker(db_path=str(tmp_path / "cost.db"))
    assert tracker.last_call() is None
    r1 = tracker.record_llm_call("ollama/qwen3:8b", 10, 20, session_id="s1")
    assert tracker.last_call() is r1
    r2 = tracker.record_llm_call("ollama/qwen3:8b", 30, 40, session_id="s2")
    assert tracker.last_call() is r2
```

**Commit:**

```bash
git add src/cognithor/telemetry/cost_tracker.py tests/test_telemetry/test_cost_tracker.py
git commit -m "feat(telemetry): CostTracker.last_call() accessor for Crew-Layer token usage"
```

Document in the PR 1 body: "Additive CostTracker change — adds a read-only `last_call()` accessor. Zero behavior change for existing callers; Planner's `_record_cost()` path untouched."

- [ ] **Step 6: Run the integration test + full test_crew**

```bash
python -m pytest tests/test_crew/test_pge_integration.py tests/test_crew/test_sequential_kickoff.py tests/test_crew/test_async_kickoff.py -v
```

Existing tests from Tasks 8-9 need to be updated to use `get_tools_for_role` on their registry mocks (not `list_tool_names`) and to supply an explicit `planner=` either via `Crew(planner=mock_planner)` or by patching `cognithor.crew.runtime.get_default_planner`.

- [ ] **Step 7: Ruff + commit**

```bash
git add src/cognithor/crew/compiler.py src/cognithor/crew/crew.py src/cognithor/crew/runtime.py tests/test_crew/test_pge_integration.py
git commit -m "feat(crew): real PGE-Trinity integration — execute_task routes through Planner.formulate_response"
```

---

### Task 12: Gatekeeper integration — every tool call classified

**Files:**
- Modify: `src/cognithor/crew/compiler.py` (wrap tool calls with Gatekeeper.classify())
- Create: `tests/test_crew/test_gatekeeper_integration.py`

The Planner already invokes the Gatekeeper internally when it plans a tool call (see `core/gatekeeper.py:53` — `classify()` returns a `RiskLevel`). The Crew-Layer does NOT bypass that path; it merely exposes it. This task adds a TEST that proves the path is intact: a Crew with a RED-listed tool must raise or prompt-for-approval, depending on `risk_ceiling`.

- [ ] **Step 1: Failing test**

```python
# tests/test_crew/test_gatekeeper_integration.py
from unittest.mock import AsyncMock, MagicMock
import pytest
from cognithor.crew import Crew, CrewAgent, CrewTask


async def test_gatekeeper_red_tool_blocks_execution():
    """When an agent lists a tool that Gatekeeper classifies as RED, the
    task must fail-closed unless explicit approval is configured."""
    agent = CrewAgent(role="deleter", goal="delete", tools=["delete_all"])
    task = CrewTask(description="x", expected_output="y", agent=agent)
    crew = Crew(agents=[agent], tasks=[task])

    from cognithor.crew.errors import CrewError
    mock_planner = MagicMock()
    # Simulate Planner raising when Gatekeeper denies the tool
    mock_planner.formulate_response = AsyncMock(
        side_effect=CrewError("Gatekeeper RED: 'delete_all' blocked")
    )
    mock_registry = MagicMock()
    fake_tool = MagicMock(); fake_tool.name = "delete_all"
    mock_registry.get_tools_for_role.return_value = [fake_tool]

    from cognithor.crew.compiler import compile_and_run_async
    from cognithor.crew.process import CrewProcess
    with pytest.raises(CrewError, match="Gatekeeper"):
        await compile_and_run_async(
            agents=[agent], tasks=[task],
            process=CrewProcess.SEQUENTIAL,
            inputs=None,
            registry=mock_registry,
            planner=mock_planner,
        )
```

- [ ] **Step 2: Run — expect this test to already pass**

The current implementation already propagates exceptions from the planner, so this test passes as a guardrail — it verifies the contract stays intact if someone later tries to add try/except that swallows Gatekeeper errors. The commit locks that behaviour in.

- [ ] **Step 3: Commit**

```bash
git add tests/test_crew/test_gatekeeper_integration.py
git commit -m "test(crew): Gatekeeper RED verdict propagates as CrewError"
```

---

### Task 13: Context-passing between tasks (task N consumes task N-1 output)

**Files:**
- Create: `tests/test_crew/test_context_passing.py`

The `context=[...]` field on CrewTask is already set in Task 5. The real behaviour check: when t2 declares `context=[t1]`, does the planner call for t2 receive t1's output text?

- [ ] **Step 1: Test**

```python
# tests/test_crew/test_context_passing.py
from unittest.mock import AsyncMock, MagicMock
import pytest
from cognithor.crew import Crew, CrewAgent, CrewTask
from cognithor.core.observer import ResponseEnvelope


async def test_task2_receives_task1_output():
    agent = CrewAgent(role="x", goal="y")
    t1 = CrewTask(description="phase 1", expected_output="res1", agent=agent)
    t2 = CrewTask(description="phase 2", expected_output="res2", agent=agent, context=[t1])

    captured_results: list = []
    captured_user_msgs: list = []

    async def capture(user_message, results, working_memory):
        captured_user_msgs.append(user_message)
        captured_results.append(list(results))
        n = len(captured_user_msgs)
        return ResponseEnvelope(
            content="PHASE1_RESULT" if n == 1 else "PHASE2_RESULT",
            directive=None,
        )

    mock_planner = MagicMock()
    mock_planner.formulate_response = AsyncMock(side_effect=capture)
    mock_registry = MagicMock()
    mock_registry.get_tools_for_role.return_value = []

    crew = Crew(agents=[agent], tasks=[t1, t2], planner=mock_planner)

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("cognithor.crew.runtime.get_default_tool_registry", lambda: mock_registry)
        result = await crew.kickoff_async()

    assert result.tasks_output[1].raw == "PHASE2_RESULT"
    # The 2nd call (for t2) must carry t1's output in `results`
    t2_results = captured_results[1]
    assert any("PHASE1_RESULT" in r.content for r in t2_results)
```

- [ ] **Step 2: Run — expect PASS** (the `execute_task_async` in Task 11 already threads prior outputs as `ToolResult` entries). If it fails, investigate the synthesis loop in `execute_task_async` and adjust.

- [ ] **Step 3: Commit**

```bash
git add tests/test_crew/test_context_passing.py
git commit -m "test(crew): context array threads prior task outputs into prompt"
```

---

### Task 14: Audit-chain integration — every kickoff emits a trace

**Files:**
- Modify: `src/cognithor/crew/compiler.py` (emit audit events)
- Create: `tests/test_crew/test_audit_chain.py`

- [ ] **Step 1: Scouted audit API (`src/cognithor/security/audit.py`)**

The real audit helper is `cognithor.security.audit.AuditTrail`:

```python
class AuditTrail:
    def __init__(self, log_dir: Path | None = None, *, log_path: Path | str | None = None,
                 hmac_key: bytes | None = None, ed25519_key: bytes | None = None) -> None: ...

    def record(self, entry: AuditEntry, *, mask: bool = True) -> str: ...
    def record_event(self, session_id: str, event_type: str,
                     details: dict[str, Any] | None = None) -> str: ...
    def verify_chain(self) -> tuple[bool, int, int]: ...
```

`record_event(session_id, event_type, details)` is the right entry point for free-form crew events — it writes JSONL with SHA-256 chain + optional HMAC.

There is **no** `cognithor.core.safe_call.append_audit` function. The Crew-Layer wraps `AuditTrail.record_event` in a thin module-local helper so tests can monkey-patch a single callable.

- [ ] **Step 2: Test**

```python
# tests/test_crew/test_audit_chain.py
from unittest.mock import patch, MagicMock, AsyncMock
import pytest
from cognithor.crew import Crew, CrewAgent, CrewTask
from cognithor.core.observer import ResponseEnvelope


async def test_kickoff_emits_audit_event_with_trace_id():
    agent = CrewAgent(role="x", goal="y")
    task = CrewTask(description="a", expected_output="b", agent=agent)

    events: list = []

    def spy(event_name, **fields):
        events.append((event_name, fields))

    mock_planner = MagicMock()
    mock_planner.formulate_response = AsyncMock(
        return_value=ResponseEnvelope(content="OK", directive=None),
    )
    mock_registry = MagicMock()
    mock_registry.get_tools_for_role.return_value = []

    crew = Crew(agents=[agent], tasks=[task], planner=mock_planner)

    with patch("cognithor.crew.compiler.append_audit", side_effect=spy):
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr("cognithor.crew.runtime.get_default_tool_registry", lambda: mock_registry)
            result = await crew.kickoff_async()

    # At least one audit event emitted with our trace_id
    kickoff_events = [e for e in events if "crew" in e[0]]
    assert kickoff_events
    assert any(fields.get("trace_id") == result.trace_id for _name, fields in kickoff_events)
```

- [ ] **Step 3: Emit audit events from the compiler**

Add near the top of `compiler.py`:

```python
# Audit helper — wraps cognithor.security.audit.AuditTrail.record_event()
# as a single module-local callable so tests can patch it cleanly.
_audit_trail: Any = None
_audit_lock = threading.Lock()


def _get_audit_trail() -> Any:
    """Lazy-build a process-wide AuditTrail under the Cognithor audit log path."""
    global _audit_trail
    with _audit_lock:
        if _audit_trail is not None:
            return _audit_trail
        try:
            from cognithor.config import load_config
            from cognithor.security.audit import AuditTrail
            cfg = load_config()
            log_dir = Path(cfg.cognithor_home) / "logs"
            _audit_trail = AuditTrail(log_dir=log_dir)
        except Exception:
            _audit_trail = None  # remains None — append_audit becomes a no-op
        return _audit_trail


def append_audit(event: str, **fields: Any) -> None:
    """Emit a Crew-Layer audit event via the Hashline-Guard chain.

    Falls back to a no-op when AuditTrail cannot be built (e.g. standalone
    test without ~/.cognithor/ present). Test code monkey-patches this
    callable directly rather than the AuditTrail inside it.
    """
    trail = _get_audit_trail()
    if trail is None:
        return
    session_id = fields.pop("trace_id", "crew")
    try:
        trail.record_event(session_id=session_id, event_type=event, details=fields)
    except Exception as exc:
        # Spec §11.5: audit failures must be SURFACED, not silently swallowed.
        # A debug-level log hides a broken Hashline-Guard chain — tamper
        # evidence becomes useless the moment writes start failing quietly.
        # We escalate to WARNING (+ full exc_info) AND tick a dedicated metric
        # so observability sees the break. We still don't re-raise: a broken
        # audit trail must not tear down the user's kickoff. See NI6.
        log.warning(
            "crew_audit_record_failed — Hashline-Guard chain may be incomplete",
            extra={"event": event, "session_id": session_id},
            exc_info=exc,
        )
        try:
            from cognithor.telemetry.metrics import MetricsProvider
            MetricsProvider.get_instance().counter(
                "cognithor_crew_audit_record_failures_total",
                1,
                labels={"reason": type(exc).__name__},
            )
        except (ImportError, AttributeError):
            # Metrics module optional in minimal installs — the log entry is
            # still present, so the failure isn't invisible.
            pass
```

Emit events at key lifecycle points inside `compile_and_run_async`:

```python
append_audit("crew_kickoff_started", trace_id=trace_id, n_tasks=len(ordered), process=process.value)
# ... inside loop ...
append_audit("crew_task_started", trace_id=trace_id, task_id=t.task_id, agent_role=t.agent.role)
# ... after completion ...
append_audit("crew_task_completed", trace_id=trace_id, task_id=t.task_id,
             duration_ms=out.duration_ms, tokens=out.token_usage.get("total_tokens", 0))
# ... at the end ...
append_audit("crew_kickoff_completed", trace_id=trace_id, n_tasks=len(outputs))
```

- [ ] **Step 4: Run + commit**

```bash
python -m pytest tests/test_crew/test_audit_chain.py -v
git add src/cognithor/crew/compiler.py tests/test_crew/test_audit_chain.py
git commit -m "feat(crew): emit Hashline-Guard audit events for crew lifecycle"
```

- [ ] **Step 5: Crew-Layer log/audit PII redaction (Spec §8.2, R4-I8)**

**Scouted (2026-04-24):** `src/cognithor/security/pii_redactor.py` exposes
`class PIIRedactor` with `.redact(text) -> (sanitized, matches)`. Patterns
cover emails, phone numbers, API keys, credit cards, SSNs, IBANs, and PEM
private-key blocks. Default-off (opt-in via `security.pii_redactor.enabled`).

Spec §8.2 requires Crew-Layer log/audit output to be routed through the same
PII-sanitization chain as the main runtime. Wire the redactor into
`append_audit` so no raw PII lands in the Hashline-Guard chain regardless of
whether the user enabled the outbound redactor for LLM calls (audit trails
are security-sensitive — we redact unconditionally for them):

```python
# Inside src/cognithor/crew/compiler.py, above append_audit:
from cognithor.security.pii_redactor import PIIRedactor

# Module-level singleton — the redactor is stateless, instantiating per-call
# would re-compile regex for every audit event.
_CREW_PII_REDACTOR = PIIRedactor()


def _scrub_audit_fields(fields: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``fields`` with string values passed through the
    PII redactor. Non-string values (ints, floats, bools, dicts) pass through
    untouched. Lists of strings are element-wise redacted; deeper nesting
    falls through as-is (audit-chain fields are flat by convention).
    """
    cleaned: dict[str, Any] = {}
    for key, value in fields.items():
        if isinstance(value, str):
            sanitized, _matches = _CREW_PII_REDACTOR.redact(value)
            cleaned[key] = sanitized
        elif isinstance(value, list) and value and all(isinstance(v, str) for v in value):
            cleaned[key] = [_CREW_PII_REDACTOR.redact(v)[0] for v in value]
        else:
            cleaned[key] = value
    return cleaned
```

Update `append_audit` to scrub before handing off to `record_event`:

```python
def append_audit(event: str, **fields: Any) -> None:
    trail = _get_audit_trail()
    if trail is None:
        return
    session_id = fields.pop("trace_id", "crew")
    scrubbed = _scrub_audit_fields(fields)  # R4-I8: PII redaction before persist
    try:
        trail.record_event(session_id=session_id, event_type=event, details=scrubbed)
    except Exception as exc:
        log.warning(
            "crew_audit_record_failed — Hashline-Guard chain may be incomplete",
            extra={"event": event, "session_id": session_id},
            exc_info=exc,
        )
        try:
            from cognithor.telemetry.metrics import MetricsProvider
            MetricsProvider.get_instance().counter(
                "cognithor_crew_audit_record_failures_total",
                1,
                labels={"reason": type(exc).__name__},
            )
        except (ImportError, AttributeError):
            pass
```

Regression test:

```python
# tests/test_crew/test_audit_chain.py — append this test
def test_audit_events_are_pii_scrubbed(tmp_path):
    """R4-I8: audit fields containing PII must be redacted before persisting."""
    from cognithor.crew.compiler import _scrub_audit_fields

    cleaned = _scrub_audit_fields({
        "task_id": "t1",
        "feedback": "Email user at test@example.com after the call",
        "duration_ms": 123.4,
    })
    assert "test@example.com" not in cleaned["feedback"]
    assert "[REDACTED:email]" in cleaned["feedback"]
    assert cleaned["task_id"] == "t1"       # non-PII strings pass through
    assert cleaned["duration_ms"] == 123.4  # non-string values pass through
```

Commit:

```bash
git add src/cognithor/crew/compiler.py tests/test_crew/test_audit_chain.py
git commit -m "feat(crew): route audit-chain fields through PII redactor (spec §8.2)"
```

---

### Task 15: Idempotent kickoff with Distributed-Lock

**Files:**
- Modify: `src/cognithor/crew/crew.py` (wrap kickoff_async in distributed lock)
- Create: `tests/test_crew/test_idempotent_kickoff.py`

Spec §1.6: "`kickoff()` ist idempotent re-aufrufbar (nutzt bestehende Distributed-Lock-Logik)". Wire it.

- [ ] **Step 1: Scouted DistributedLock API (`src/cognithor/core/distributed_lock.py`)**

The real API has a zero-arg constructor for concrete backends (`LocalLockBackend()`, `FileLockBackend(lock_dir=...)`, `RedisLockBackend(...)`), plus a `create_lock(config)` factory. Lock acquisition uses the `__call__(name, timeout)` pattern as an async context manager:

```python
# Real usage (from module docstring, lines 51-56):
lock = create_lock(config)
async with lock("session_123"):      # lock(name, timeout=10.0) returns _LockContext
    # critical section
    ...
```

The plan's earlier `DistributedLock(key, timeout_s=300)` is wrong on two counts: `DistributedLock` is an abstract base (never directly instantiated) and its `__init__` takes no args.

- [ ] **Step 2: Test**

```python
# tests/test_crew/test_idempotent_kickoff.py
from unittest.mock import AsyncMock, MagicMock
import pytest
from cognithor.crew import Crew, CrewAgent, CrewTask
from cognithor.core.observer import ResponseEnvelope


async def test_same_kickoff_id_returns_cached_output():
    """If the same kickoff_id is provided twice, the second call returns
    the cached CrewOutput without re-running tasks (deterministic replay).
    """
    agent = CrewAgent(role="x", goal="y")
    task = CrewTask(description="a", expected_output="b", agent=agent)

    mock_planner = MagicMock()
    call_count = {"n": 0}
    async def fake_resp(user_message, results, working_memory):
        call_count["n"] += 1
        return ResponseEnvelope(content=f"RUN-{call_count['n']}", directive=None)
    mock_planner.formulate_response = AsyncMock(side_effect=fake_resp)
    mock_registry = MagicMock()
    mock_registry.get_tools_for_role.return_value = []

    crew = Crew(agents=[agent], tasks=[task], planner=mock_planner)

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("cognithor.crew.runtime.get_default_tool_registry", lambda: mock_registry)
        out1 = await crew.kickoff_async(inputs={"_kickoff_id": "fixed-id-123"})
        out2 = await crew.kickoff_async(inputs={"_kickoff_id": "fixed-id-123"})

    assert out1.raw == out2.raw, "Same kickoff_id must return identical output"
    assert call_count["n"] == 1, "Planner must be called only once for same kickoff_id"


async def test_kickoff_id_removed_non_destructively():
    """Caller's inputs dict must not be mutated by the kickoff-id strip."""
    agent = CrewAgent(role="x", goal="y")
    task = CrewTask(description="a", expected_output="b", agent=agent)

    mock_planner = MagicMock()
    mock_planner.formulate_response = AsyncMock(
        return_value=ResponseEnvelope(content="ok", directive=None),
    )
    mock_registry = MagicMock()
    mock_registry.get_tools_for_role.return_value = []

    crew = Crew(agents=[agent], tasks=[task], planner=mock_planner)

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("cognithor.crew.runtime.get_default_tool_registry", lambda: mock_registry)
        inputs = {"_kickoff_id": "keep-me", "topic": "PKV"}
        await crew.kickoff_async(inputs=inputs)

    # The caller still sees their original dict intact
    assert inputs == {"_kickoff_id": "keep-me", "topic": "PKV"}
```

- [ ] **Step 3: Implement kickoff-caching in `crew.py`**

Uses the real `create_lock(config)` factory; splits `_kickoff_id` from the rest of `inputs` non-destructively. If the distributed-lock module is unavailable (e.g. stripped-down install, import error during dev), we still serialize concurrent same-id kickoffs in-process via an `asyncio.Lock()` fallback — NEVER silently bypass locking entirely.

```python
import asyncio
import logging
import threading
from collections import OrderedDict

log = logging.getLogger(__name__)

# Module-level bounded cache keyed by kickoff_id (best-effort, per-process).
# OrderedDict + LRU-style eviction caps memory growth in long-running
# processes. See NI7 in Round 3 review.
_KICKOFF_CACHE_MAX_SIZE = 128
_KICKOFF_CACHE: OrderedDict[str, CrewOutput] = OrderedDict()


def _cache_put(key: str, value: CrewOutput) -> None:
    """Insert or refresh a cache entry, evicting oldest when over capacity."""
    _KICKOFF_CACHE[key] = value
    _KICKOFF_CACHE.move_to_end(key)
    while len(_KICKOFF_CACHE) > _KICKOFF_CACHE_MAX_SIZE:
        _KICKOFF_CACHE.popitem(last=False)


def _cache_get(key: str) -> CrewOutput | None:
    """Return cached value (refreshing LRU position) or None."""
    if key in _KICKOFF_CACHE:
        _KICKOFF_CACHE.move_to_end(key)
        return _KICKOFF_CACHE[key]
    return None


# Process-wide distributed lock singleton. ``create_lock()`` with a
# LocalLockBackend builds a fresh ``dict[str, asyncio.Lock]`` per call —
# instantiating a new lock per kickoff_async() call would therefore NEVER
# serialize two concurrent same-id kickoffs inside one process (each call
# sees its own dict). We must reuse a single DistributedLock instance. See
# NC2 in Round 3 review.
_lock_singleton: "Any | None" = None
_lock_singleton_init = threading.Lock()


def _get_distributed_lock() -> "Any":
    """Return the process-wide DistributedLock, constructing it lazily once.

    Uses the double-checked-locking pattern with ``threading.Lock`` so multiple
    threads importing this module cannot produce racing singletons. The
    candidate ``create_lock(...)`` call happens OUTSIDE the threading.Lock
    to avoid holding that lock while loading config / opening files.
    """
    global _lock_singleton
    if _lock_singleton is not None:
        return _lock_singleton

    from cognithor.config import load_config
    from cognithor.core.distributed_lock import create_lock

    candidate = create_lock(load_config())  # built outside critical section
    with _lock_singleton_init:
        if _lock_singleton is None:
            _lock_singleton = candidate
    return _lock_singleton


# In-process fallback lock, lazily constructed. Used only when the
# distributed_lock module is unavailable. Cannot be created at import time
# because asyncio.Lock() binds to the current running loop.
_fallback_lock: asyncio.Lock | None = None


async def _get_fallback_lock() -> asyncio.Lock:
    """Lazily construct a module-level asyncio.Lock bound to the running loop."""
    global _fallback_lock
    if _fallback_lock is None:
        _fallback_lock = asyncio.Lock()
    return _fallback_lock


async def kickoff_async(self, inputs: dict[str, Any] | None = None) -> CrewOutput:
    # Non-destructive strip: dict-comprehension copy, leaving the caller's dict intact.
    kickoff_id: str | None = None
    if inputs:
        kickoff_id = inputs.get("_kickoff_id")
        inputs = {k: v for k, v in inputs.items() if k != "_kickoff_id"}

    if kickoff_id:
        cached = _cache_get(kickoff_id)
        if cached is not None:
            return cached

    from cognithor.crew.compiler import compile_and_run_async
    from cognithor.crew.runtime import get_default_planner, get_default_tool_registry

    planner = self._planner or get_default_planner()
    registry = get_default_tool_registry()

    async def _run_guarded() -> CrewOutput:
        """Run the compiler under whichever lock is available and cache."""
        # Double-check cache inside the lock to handle the race
        if kickoff_id:
            cached_inner = _cache_get(kickoff_id)
            if cached_inner is not None:
                return cached_inner
        result = await compile_and_run_async(
            agents=self.agents, tasks=self.tasks, process=self.process,
            inputs=inputs, registry=registry, planner=planner,
        )
        if kickoff_id:
            _cache_put(kickoff_id, result)
        return result

    if kickoff_id:
        # Preferred path: cross-process distributed lock (singleton).
        #   lock = _get_distributed_lock(); async with lock(name, timeout): ...
        try:
            lock = _get_distributed_lock()
        except ImportError:
            # Distributed-lock module missing (minimal install / dev setup).
            # Fall back to an in-process asyncio.Lock so concurrent kickoffs
            # within THIS process still serialize. Cross-process safety is
            # degraded but never silently dropped.
            log.warning(
                "cognithor.core.distributed_lock unavailable — falling back "
                "to in-process asyncio.Lock for crew kickoff serialization. "
                "Cross-process idempotency is NOT guaranteed in this config."
            )
            async with (await _get_fallback_lock()):
                return await _run_guarded()

        async with lock(f"crew:kickoff:{kickoff_id}", 300.0):
            return await _run_guarded()

    # No kickoff_id — plain unlocked execution.
    result = await compile_and_run_async(
        agents=self.agents, tasks=self.tasks, process=self.process,
        inputs=inputs, registry=registry, planner=planner,
    )
    return result
```

**Concurrency regression test — add to `tests/test_crew/test_idempotent_kickoff.py`:**

```python
async def test_concurrent_same_id_serializes_under_local_backend():
    """Two concurrent kickoffs with same _kickoff_id must serialize.

    Regression for NC2: constructing a NEW DistributedLock per kickoff_async
    call made the "double-check cache inside the lock" pattern useless under
    LocalLockBackend, because each call saw a fresh dict[str, asyncio.Lock].
    The singleton helper _get_distributed_lock() fixes this.
    """
    agent = CrewAgent(role="x", goal="y")
    task = CrewTask(description="a", expected_output="b", agent=agent)

    in_flight = {"n": 0, "max_seen": 0}

    async def fake_resp(user_message, results, working_memory):
        in_flight["n"] += 1
        in_flight["max_seen"] = max(in_flight["max_seen"], in_flight["n"])
        await asyncio.sleep(0.05)  # make concurrency observable
        in_flight["n"] -= 1
        return ResponseEnvelope(content="RUN", directive=None)

    mock_planner = MagicMock()
    mock_planner.formulate_response = AsyncMock(side_effect=fake_resp)
    mock_registry = MagicMock()
    mock_registry.get_tools_for_role.return_value = []

    crew = Crew(agents=[agent], tasks=[task], planner=mock_planner)

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("cognithor.crew.runtime.get_default_tool_registry", lambda: mock_registry)
        # Reset singleton so the test configures a fresh LocalLockBackend.
        mp.setattr("cognithor.crew.crew._lock_singleton", None, raising=False)
        out1, out2 = await asyncio.gather(
            crew.kickoff_async(inputs={"_kickoff_id": "serialize-me"}),
            crew.kickoff_async(inputs={"_kickoff_id": "serialize-me"}),
        )

    assert out1.raw == out2.raw
    # At no point were two runs in flight simultaneously (serialized).
    assert in_flight["max_seen"] == 1, (
        "Concurrent kickoffs with the same id must serialize via the "
        "singleton distributed lock, not run in parallel."
    )
```

- [ ] **Step 4: Run + commit**

```bash
python -m pytest tests/test_crew/test_idempotent_kickoff.py -v
git add src/cognithor/crew/crew.py tests/test_crew/test_idempotent_kickoff.py
git commit -m "feat(crew): idempotent kickoff via _kickoff_id + distributed lock"
```

---

### Task 16: YAML loader — `load_crew_from_yaml()`

**Files:**
- Create: `src/cognithor/crew/yaml_loader.py`
- Create: `tests/test_crew/test_yaml_loader.py`
- Create: `tests/test_crew/fixtures/sample_agents.yaml`
- Create: `tests/test_crew/fixtures/sample_tasks.yaml`

- [ ] **Step 1: Fixture YAML files**

```yaml
# tests/test_crew/fixtures/sample_agents.yaml
analyst:
  role: analyst
  goal: analyze PKV tariffs
  backstory: veteran broker with §34d certification
  tools: [web_search, pdf_reader]
  llm: ollama/qwen3:8b

writer:
  role: writer
  goal: write customer reports
  llm: ollama/qwen3:8b
```

```yaml
# tests/test_crew/fixtures/sample_tasks.yaml
research:
  description: Compare the top three PKV tariffs for a {age}-year-old
  expected_output: Tabular comparison with price, coverage, exclusions
  agent: analyst

report:
  description: Turn the analysis into a customer report
  expected_output: Markdown text
  agent: writer
  context: [research]
```

- [ ] **Step 2: Failing test**

```python
# tests/test_crew/test_yaml_loader.py
from pathlib import Path
import pytest
from cognithor.crew import Crew, CrewProcess
from cognithor.crew.yaml_loader import load_crew_from_yaml


class TestYamlLoader:
    def test_loads_two_agent_crew(self):
        fixtures = Path(__file__).parent / "fixtures"
        crew = load_crew_from_yaml(
            agents=fixtures / "sample_agents.yaml",
            tasks=fixtures / "sample_tasks.yaml",
            process=CrewProcess.SEQUENTIAL,
        )
        assert isinstance(crew, Crew)
        assert len(crew.agents) == 2
        assert len(crew.tasks) == 2
        assert crew.agents[0].role == "analyst"
        # Second task's context resolves to first task (by YAML key)
        assert crew.tasks[1].context[0].task_id == crew.tasks[0].task_id

    def test_missing_agent_reference_raises(self, tmp_path: Path):
        (tmp_path / "a.yaml").write_text("x: {role: x, goal: y}\n")
        (tmp_path / "t.yaml").write_text("t1: {description: d, expected_output: e, agent: unknown}\n")
        with pytest.raises(ValueError, match="unknown"):
            load_crew_from_yaml(agents=tmp_path / "a.yaml", tasks=tmp_path / "t.yaml")
```

- [ ] **Step 3: Implement `yaml_loader.py`**

```python
"""Load a Crew from YAML config files.

Accepts two files:
  agents.yaml — dict keyed by agent-alias, values are CrewAgent-kwargs dicts
  tasks.yaml  — dict keyed by task-alias, values are CrewTask-kwargs dicts
                (agent: <alias>, context: [<alias>...])
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from cognithor.crew.agent import CrewAgent
from cognithor.crew.crew import Crew
from cognithor.crew.process import CrewProcess
from cognithor.crew.task import CrewTask


def load_crew_from_yaml(
    *,
    agents: Path | str,
    tasks: Path | str,
    process: CrewProcess = CrewProcess.SEQUENTIAL,
    verbose: bool = False,
    planning: bool = False,
    manager_llm: str | None = None,
) -> Crew:
    agents_data: dict[str, Any] = yaml.safe_load(Path(agents).read_text(encoding="utf-8")) or {}
    tasks_data: dict[str, Any] = yaml.safe_load(Path(tasks).read_text(encoding="utf-8")) or {}

    # Build agents by alias
    agent_by_alias: dict[str, CrewAgent] = {
        alias: CrewAgent(**kwargs) for alias, kwargs in agents_data.items()
    }

    # Build tasks — requires two passes because `context` references other tasks
    # by alias, which must already be constructed. Pass 1: construct without context.
    task_by_alias: dict[str, CrewTask] = {}
    context_map: dict[str, list[str]] = {}
    for alias, kwargs in tasks_data.items():
        agent_alias = kwargs.pop("agent")
        if agent_alias not in agent_by_alias:
            # Bilingual error (resolves via cognithor.i18n.t with config.language).
            # Locale keys: crew.errors.unknown_agent
            from cognithor.i18n import t
            raise ValueError(
                t("crew.errors.unknown_agent",
                  task=alias, agent=agent_alias,
                  known=", ".join(agent_by_alias) or "(none)")
            )
        context_map[alias] = kwargs.pop("context", []) or []
        task_by_alias[alias] = CrewTask(
            agent=agent_by_alias[agent_alias], context=[], **kwargs
        )

    # Pass 2: resolve context references — Pydantic models are frozen, so use
    # `.model_copy(update=...)` for an immutable update.
    #
    # Why model_copy and not model_dump + rebuild: `model_dump()` cannot serialize
    # Callables. The `guardrail` field holds a Python callable (or StringGuardrail
    # instance); a round-trip through dump-then-init would silently drop it back
    # to None. `model_copy(update={...})` preserves every field by identity.
    for alias, refs in context_map.items():
        if not refs:
            continue
        ctx: list[CrewTask] = []
        for ref in refs:
            if ref not in task_by_alias:
                # R4-I1: localized error message via the i18n fallback chain.
                # Uses CrewCompilationError (not bare ValueError) so callers
                # can pattern-match on a crew-specific exception type.
                from cognithor.crew.errors import CrewCompilationError
                from cognithor.i18n import t
                raise CrewCompilationError(
                    t("crew.errors.unknown_task", task=alias, ref=ref)
                )
            ctx.append(task_by_alias[ref])
        existing = task_by_alias[alias]
        task_by_alias[alias] = existing.model_copy(update={"context": ctx})

    return Crew(
        agents=list(agent_by_alias.values()),
        tasks=list(task_by_alias.values()),
        process=process,
        verbose=verbose,
        planning=planning,
        manager_llm=manager_llm,
    )
```

- [ ] **Step 4: Run + commit**

```bash
python -m pytest tests/test_crew/test_yaml_loader.py -v
git add src/cognithor/crew/yaml_loader.py tests/test_crew/fixtures tests/test_crew/test_yaml_loader.py
git commit -m "feat(crew): YAML loader for agents.yaml + tasks.yaml"
```

---

### Task 17: Decorators — `@cognithor.crew.agent` / `@task` / `@crew`

**Files:**
- Create: `src/cognithor/crew/decorators.py`
- Create: `tests/test_crew/test_decorators.py`

- [ ] **Step 1: Test**

```python
# tests/test_crew/test_decorators.py
import pytest
from cognithor.crew import Crew, CrewAgent, CrewProcess, CrewTask
from cognithor.crew import decorators as crew_dec


def test_agent_decorator_binds_kwargs():
    class Host:
        @crew_dec.agent
        def analyst(self) -> CrewAgent:
            return CrewAgent(role="analyst", goal="x")

    host = Host()
    a = host.analyst()
    assert isinstance(a, CrewAgent)
    assert a.role == "analyst"


def test_task_decorator():
    class Host:
        @crew_dec.agent
        def writer(self) -> CrewAgent:
            return CrewAgent(role="writer", goal="w")

        @crew_dec.task
        def draft(self) -> CrewTask:
            return CrewTask(description="d", expected_output="e", agent=self.writer())

    host = Host()
    t = host.draft()
    assert isinstance(t, CrewTask)


def test_crew_decorator_assembles_from_declared_agents_and_tasks():
    class PKVCrew:
        @crew_dec.agent
        def analyst(self) -> CrewAgent:
            return CrewAgent(role="analyst", goal="analyze")

        @crew_dec.task
        def research(self) -> CrewTask:
            return CrewTask(description="r", expected_output="facts", agent=self.analyst())

        @crew_dec.crew
        def assemble(self) -> Crew:
            return Crew(agents=[self.analyst()], tasks=[self.research()])

    c = PKVCrew().assemble()
    assert isinstance(c, Crew)
    assert len(c.agents) == 1
```

- [ ] **Step 2: Implement decorators**

```python
"""Method decorators for building a Crew from a Python class.

Concept inspired by CrewAI's @agent/@task/@crew pattern — implementation
is Apache 2.0, no verbatim borrow.

Usage:
    class MyCrew:
        @agent
        def researcher(self) -> CrewAgent: ...

        @task
        def research(self) -> CrewTask: ...

        @crew
        def assemble(self) -> Crew: ...
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import TypeVar

T = TypeVar("T")


def agent(fn: Callable[..., T]) -> Callable[..., T]:
    """Mark a zero-arg method as a CrewAgent factory.

    Caches the result per instance so repeated calls return the same agent
    object — needed because Pydantic models are compared by identity in the
    CrewTask.context graph.
    """
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        attr = f"_crew_agent_cache__{fn.__name__}"
        if not hasattr(self, attr):
            setattr(self, attr, fn(self, *args, **kwargs))
        return getattr(self, attr)
    wrapper._crew_role = "agent"  # type: ignore[attr-defined]
    return wrapper


def task(fn: Callable[..., T]) -> Callable[..., T]:
    """Mark a zero-arg method as a CrewTask factory (same caching rules)."""
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        attr = f"_crew_task_cache__{fn.__name__}"
        if not hasattr(self, attr):
            setattr(self, attr, fn(self, *args, **kwargs))
        return getattr(self, attr)
    wrapper._crew_role = "task"  # type: ignore[attr-defined]
    return wrapper


def crew(fn: Callable[..., T]) -> Callable[..., T]:
    """Mark a method as the Crew assembly point."""
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        return fn(self, *args, **kwargs)
    wrapper._crew_role = "crew"  # type: ignore[attr-defined]
    return wrapper
```

- [ ] **Step 3: Run + commit**

```bash
python -m pytest tests/test_crew/test_decorators.py -v
git add src/cognithor/crew/decorators.py tests/test_crew/test_decorators.py
git commit -m "feat(crew): @agent/@task/@crew class-based decorators"
```

---

### Task 18: Error-message quality pass (missing tools, missing agents, invalid inputs) + trilingual localization

**Files:**
- Modify: `src/cognithor/crew/tool_resolver.py`, `src/cognithor/crew/yaml_loader.py`, `src/cognithor/crew/errors.py` (refine messages)
- Modify: `src/cognithor/i18n/locales/en.json`, `src/cognithor/i18n/locales/de.json`, `src/cognithor/i18n/locales/zh.json` (add Crew-Layer keys)
- Create: `tests/test_crew/test_error_messages.py`

> **R4-I9 / locale coverage:** scouting `src/cognithor/i18n/locales/` on
> 2026-04-24 found three installed packs: `en.json`, `de.json`, `zh.json`.
> No `ar.json`. Crew-Layer keys are added to all three; users on any other
> locale fall back to English via the `t()` fallback chain defined in
> `src/cognithor/i18n/__init__.py` (requested → English → raw key). This is
> acceptable degradation — documented in the Feature 1 CHANGELOG.

**Spec §8 (i18n):** Crew-Layer error paths must emit bilingual messages via `cognithor.i18n.t()` with the active `config.language` (defaults to "de"). This covers the three most-user-facing failure scenarios from spec §12:

1. Scenario 2 — YAML parse error (agents.yaml / tasks.yaml)
2. Scenario 3 — Ollama not running when kickoff starts
3. Scenario 4 — Guardrail failure after retries exhausted

**Locale keys to add** (both `en.json` and `de.json`):

```json
{
  "crew.errors.yaml_parse": "Failed to parse {file}: {error}",
  "crew.errors.ollama_offline": "Ollama server is not reachable at {url}. Start Ollama or set COGNITHOR_OLLAMA_BASE_URL.",
  "crew.errors.guardrail_failed": "Guardrail '{name}' rejected output from task '{task}' after {attempts} attempt(s): {reason}",
  "crew.errors.unknown_agent": "Task '{task}' references unknown agent '{agent}'. Known agents: {known}",
  "crew.errors.unknown_tool": "Agent references unknown tool '{tool}'. Known tools: {known}",
  "crew.errors.tool_suggestion": "Agent references unknown tool '{tool}'. Did you mean '{suggestion}'?",
  "crew.errors.unknown_task": "Task '{task}' references unknown task '{ref}'"
}
```

German equivalents in `de.json` (shortened for brevity; implementer ships all seven):

```json
{
  "crew.errors.yaml_parse": "Konnte {file} nicht parsen: {error}",
  "crew.errors.ollama_offline": "Ollama-Server nicht erreichbar unter {url}. Starte Ollama oder setze COGNITHOR_OLLAMA_BASE_URL.",
  "crew.errors.guardrail_failed": "Guardrail '{name}' hat Output von Task '{task}' nach {attempts} Versuch(en) abgelehnt: {reason}",
  "crew.errors.unknown_agent": "Task '{task}' referenziert unbekannten Agent '{agent}'. Bekannte Agents: {known}",
  "crew.errors.unknown_tool": "Agent referenziert unbekanntes Tool '{tool}'. Bekannte Tools: {known}",
  "crew.errors.tool_suggestion": "Agent referenziert unbekanntes Tool '{tool}'. Meintest du '{suggestion}'?",
  "crew.errors.unknown_task": "Task '{task}' referenziert unbekannten Task '{ref}'"
}
```

Chinese equivalents in `zh.json` (R4-I9 — the repo ships a `zh` locale pack; add
Crew-Layer keys so zh users don't fall straight to English for the Crew-Layer's
user-facing errors):

```json
{
  "crew.errors.yaml_parse": "无法解析 {file}:{error}",
  "crew.errors.ollama_offline": "无法连接到 Ollama 服务器 {url}。请启动 Ollama 或设置 COGNITHOR_OLLAMA_BASE_URL。",
  "crew.errors.guardrail_failed": "Guardrail '{name}' 在 {attempts} 次尝试后拒绝了任务 '{task}' 的输出:{reason}",
  "crew.errors.unknown_agent": "任务 '{task}' 引用了未知的 Agent '{agent}'。已知 Agents:{known}",
  "crew.errors.unknown_tool": "Agent 引用了未知的工具 '{tool}'。已知工具:{known}",
  "crew.errors.tool_suggestion": "Agent 引用了未知的工具 '{tool}'。您是否想使用 '{suggestion}'?",
  "crew.errors.unknown_task": "任务 '{task}' 引用了未知的任务 '{ref}'"
}
```

**Wire-in:**

1. `yaml_loader.load_crew_from_yaml()` — wrap `yaml.safe_load(...)` in try/except that raises `CrewCompilationError(t("crew.errors.yaml_parse", ...))`.
2. `yaml_loader.load_crew_from_yaml()` — unknown-task-reference check (Pass 2 context resolution) raises `CrewCompilationError(t("crew.errors.unknown_task", task=alias, ref=ref))`. See R4-I1.
3. `compiler.execute_task_async` — when `planner.formulate_response` raises a connection error to Ollama, re-raise as `CrewError(t("crew.errors.ollama_offline", ...))` with the base URL from config.
4. `GuardrailFailure.__str__` — emits the bilingual message via `t()`; falls back to English if the i18n module is not importable (dev-edit, standalone test).
5. `tool_resolver.resolve_tools()` — the existing `"Meintest du"` message becomes `t("crew.errors.tool_suggestion", ...)`.

**init_cmd respects `--lang`:** `cognithor init ... --lang=de|en` already forwards to `run_init(lang=...)`. When `lang` is set, set the i18n language for the duration of the command via `cognithor.i18n.set_locale(lang)` before rendering any error. If `--lang` is omitted, default to the global `config.language`. (R4-C3: the real public API is `set_locale` / `get_locale` / `get_available_locales` as defined in `src/cognithor/i18n/__init__.py`; the alternate names ``set_language`` / ``available_languages`` do not exist.)

- [ ] **Step 1: Test messaging contract**

```python
# tests/test_crew/test_error_messages.py
import pytest
from cognithor.crew.errors import ToolNotFoundError, CrewError


class TestErrorMessaging:
    def _registry_with(self, names):
        from unittest.mock import MagicMock
        registry = MagicMock()
        tools = []
        for n in names:
            m = MagicMock(); m.name = n
            tools.append(m)
        registry.get_tools_for_role.return_value = tools
        return registry

    def test_tool_not_found_mentions_name_and_did_you_mean(self):
        from cognithor.crew.tool_resolver import resolve_tools
        registry = self._registry_with(["web_search", "pdf_reader"])
        with pytest.raises(ToolNotFoundError) as exc:
            resolve_tools(["web_seach"], registry=registry)
        msg = str(exc.value)
        assert "web_seach" in msg
        assert "Meintest du 'web_search'?" in msg

    def test_tool_not_found_mentions_name_only_when_no_close_match(self):
        from cognithor.crew.tool_resolver import resolve_tools
        registry = self._registry_with(["completely_different"])
        with pytest.raises(ToolNotFoundError) as exc:
            resolve_tools(["totally_foreign"], registry=registry)
        assert "totally_foreign" in str(exc.value)
        assert "Meintest du" not in str(exc.value)

    def test_crew_error_is_base_class(self):
        assert issubclass(ToolNotFoundError, CrewError)


class TestYamlLoaderLocalizedErrors:
    """R4-I1: `unknown_task` YAML-load errors use the i18n pipeline."""

    def test_yaml_loader_unknown_task_raises_localized(self, tmp_path):
        from cognithor.crew.errors import CrewCompilationError
        from cognithor.crew.yaml_loader import load_crew_from_yaml

        agents_yaml = tmp_path / "agents.yaml"
        tasks_yaml = tmp_path / "tasks.yaml"
        agents_yaml.write_text(
            "a:\n  role: writer\n  goal: write\n",
            encoding="utf-8",
        )
        # Task `two` references non-existent task `missing`.
        tasks_yaml.write_text(
            "one:\n  description: first\n  expected_output: x\n  agent: a\n"
            "two:\n  description: second\n  expected_output: y\n  agent: a\n"
            "  context: [missing]\n",
            encoding="utf-8",
        )

        with pytest.raises(CrewCompilationError) as exc:
            load_crew_from_yaml(agents=agents_yaml, tasks=tasks_yaml)
        # Message text comes from i18n pack (default: EN). Both the referring
        # task alias and the missing ref must appear.
        assert "two" in str(exc.value)
        assert "missing" in str(exc.value)
```

- [ ] **Step 2: Run — expect pass from Task 7 already**

- [ ] **Step 3: Commit**

```bash
git add tests/test_crew/test_error_messages.py
git commit -m "test(crew): error-message quality contracts"
```

---

### Task 19: End-to-end PKV example from spec §1.4

**Files:**
- Create: `tests/test_crew/test_pkv_example.py`

- [ ] **Step 1: The test is the spec**

```python
# tests/test_crew/test_pkv_example.py
"""Spec §1.4 — end-to-end PKV example must be runnable with mocked Ollama."""
from unittest.mock import AsyncMock, MagicMock
import pytest
from cognithor.crew import Crew, CrewAgent, CrewProcess, CrewTask
from cognithor.core.observer import ResponseEnvelope


async def test_pkv_example_runs_end_to_end():
    analyst = CrewAgent(
        role="PKV-Tarif-Analyst",
        goal="Private Krankenversicherungstarife strukturiert vergleichen",
        backstory="Erfahrener Versicherungsmakler mit §34d-Zulassung, DSGVO-bewusst",
        tools=[],
        llm="ollama/qwen3:32b",
        memory=True,
    )
    writer = CrewAgent(
        role="Kunden-Report-Schreiber",
        goal="Analyst-Ergebnisse in eine kundenverständliche PDF überführen",
        backstory="Spezialist für kundentaugliche Finanzkommunikation",
        llm="ollama/qwen3:8b",
    )
    research = CrewTask(
        description="Vergleiche die drei Top-PKV-Tarife für einen 42-jährigen GGF mit 95k Jahreseinkommen.",
        expected_output="Tabellarische Gegenüberstellung mit Beitrag, Leistungen, Ausschlüssen.",
        agent=analyst,
    )
    report = CrewTask(
        description="Erstelle einen Kunden-Report basierend auf der Analyse.",
        expected_output="PDF-tauglicher Markdown-Text, 500-800 Wörter, keine Fachjargon-Überfrachtung.",
        agent=writer,
        context=[research],
    )

    # CostTracker shim — returns different CostRecord per call to mimic two-task run.
    # Real CostTracker.last_call() returns a CostRecord with input_tokens + output_tokens.
    from types import SimpleNamespace
    tracker = MagicMock()
    tracker.last_call = MagicMock(side_effect=[
        SimpleNamespace(input_tokens=500, output_tokens=100),
        SimpleNamespace(input_tokens=800, output_tokens=600),
    ])

    mock_planner = MagicMock()
    mock_planner._cost_tracker = tracker
    mock_planner.formulate_response = AsyncMock(side_effect=[
        ResponseEnvelope(
            content="| Tarif | Beitrag | Leistungen |\n|---|---|---|\n| A | 450€ | Stationär |",
            directive=None,
        ),
        ResponseEnvelope(
            content="# PKV-Empfehlung\nBasierend auf der Analyse empfehlen wir...",
            directive=None,
        ),
    ])
    mock_registry = MagicMock()
    mock_registry.get_tools_for_role.return_value = []

    crew = Crew(
        agents=[analyst, writer],
        tasks=[research, report],
        process=CrewProcess.SEQUENTIAL,
        verbose=True,
        planner=mock_planner,
    )

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("cognithor.crew.runtime.get_default_tool_registry", lambda: mock_registry)
        result = await crew.kickoff_async()

    assert "PKV-Empfehlung" in result.raw
    assert len(result.tasks_output) == 2
    assert result.trace_id
    assert result.token_usage["total_tokens"] == 2000
```

- [ ] **Step 2: Run + commit**

```bash
python -m pytest tests/test_crew/test_pkv_example.py -v
git add tests/test_crew/test_pkv_example.py
git commit -m "test(crew): spec §1.4 PKV example end-to-end"
```

---

### Task 20: Public `cognithor.crew` namespace pollution check + version bump + Feature-1 merge-prep

**Files:**
- Modify: `src/cognithor/__init__.py` (re-export `Crew`, `CrewAgent`, `CrewTask` at root for DX)
- Modify: `CHANGELOG.md` (new `[Unreleased]` section)
- Modify: `NOTICE` (add CrewAI attribution; create file if absent)
- Create: `tests/test_crew/test_public_api_stability.py`

- [ ] **Step 1: Re-exports + stability test**

```python
# tests/test_crew/test_public_api_stability.py
def test_top_level_reexports_match_subpackage():
    from cognithor import Crew as TopCrew
    from cognithor.crew import Crew as PkgCrew
    assert TopCrew is PkgCrew


def test_frozen_public_surface():
    """Guard against accidental public-API additions without a version bump."""
    from cognithor import crew as m
    public = {n for n in dir(m) if not n.startswith("_")}
    # Plus `decorators`, `errors`, `guardrails`, `compiler`, etc. — submodules
    required = {
        "Crew", "CrewAgent", "CrewTask", "CrewProcess",
        "CrewOutput", "TaskOutput", "TokenUsageDict",
        "LLMConfig",
        "GuardrailFailure", "ToolNotFoundError",
        "CrewError", "CrewCompilationError",
    }
    assert required.issubset(public), f"Missing exports: {required - public}"
```

- [ ] **Step 2: Update `src/cognithor/__init__.py`**

Add at an appropriate point (after existing imports, before `__all__`):

```python
# Re-export the Crew-Layer at the package root for DX.
# See docs/superpowers/specs/2026-04-23-cognithor-crew-v1-adoption.md
from cognithor.crew import (  # noqa: E402
    Crew,
    CrewAgent,
    CrewOutput,
    CrewProcess,
    CrewTask,
    LLMConfig,
    TaskOutput,
)
```

Extend `__all__` accordingly.

- [ ] **Step 3: CHANGELOG**

Add an `[Unreleased]` section at the top (the video-input PR's entries are under `[0.92.7]` which is already published):

```markdown
## [Unreleased]

### Added
- **`cognithor.crew` — Crew-Layer (Feature 1 of v1.0 adoption)** — high-level
  declarative Multi-Agent API on top of PGE-Trinity. `CrewAgent`, `CrewTask`,
  `Crew`, `CrewProcess` (SEQUENTIAL + HIERARCHICAL), plus async kickoff,
  YAML loader, and `@agent` / `@task` / `@crew` method decorators. Every
  execution routes through the existing Planner → Gatekeeper → Executor
  pipeline — no new LLM entry point, no bypass. Audit events emit via the
  Hashline-Guard chain. Spec at
  `docs/superpowers/specs/2026-04-23-cognithor-crew-v1-adoption.md`.

### Breaking Changes
None. The Crew-Layer is strictly additive — no existing public API changes.
```

- [ ] **Step 4: Create `NOTICE` at repo root**

Short-form attribution file that `pyproject.toml` declares in `license-files`. Create at the repo root (not inside the package):

```
Cognithor Crew-Layer
Copyright 2026 Alexander Söllner

This product includes software concepts inspired by CrewAI (https://github.com/crewAIInc/crewAI),
licensed under the MIT License. No CrewAI source code is included verbatim.
```

This is the short canonical form. If a longer attribution file already exists, keep the existing longer form but ensure it also carries this line under a "Third-party attributions" heading.

- [ ] **Step 5: Update `pyproject.toml` `[project]` to declare NOTICE**

Modify the `[project]` table so wheel + sdist distributions ship both `LICENSE` and `NOTICE`:

```toml
[project]
# ... existing keys ...
license = "Apache-2.0"
license-files = ["LICENSE", "NOTICE"]
```

This is the PEP 639 canonical form (hatchling ≥ 1.26 supports it). Without `license-files`, `NOTICE` never makes it into the wheel — breaking the MIT→Apache 2.0 attribution bridge required by spec §8.

- [ ] **Step 6: Coverage floor in `pyproject.toml`**

Add to `pyproject.toml` (PR 1 is the first place we touch it; subsequent PRs reuse the config):

```toml
[tool.coverage.report]
fail_under = 89
# Per-module gate for cognithor.crew uses --cov-fail-under=85 on pytest
# invocations in CI / per-PR closeout (Step A2).
show_missing = true
```

This makes the total-coverage floor explicit in config; per-module (85% on `cognithor.crew`) stays as a CLI flag because `coverage.report.fail_under` is a single global number.

- [ ] **Step 7: Verify NOTICE ships in the wheel**

```bash
python -m build
python -m zipfile -l dist/cognithor-*.whl | grep -E "(LICENSE|NOTICE)"
# Expected: both files appear under cognithor-0.93.0.dist-info/licenses/
```

- [ ] **Step 8: Run full test_crew/ + ruff + commit**

```bash
python -m pytest tests/test_crew/ -v 2>&1 | tail -10
python -m pytest --cov=cognithor.crew --cov-fail-under=85 tests/test_crew/
python -m ruff check src/cognithor/crew tests/test_crew
python -m ruff format --check src/cognithor/crew tests/test_crew
git add src/cognithor/__init__.py CHANGELOG.md NOTICE pyproject.toml tests/test_crew/test_public_api_stability.py
git commit -m "feat(crew): top-level re-exports + CHANGELOG + NOTICE (license-files) + coverage floor"
```

---

# FEATURE 4 — Task-Level Guardrails (Tasks 21-32)

Implements spec §4. Function-based and string-based guardrails, four built-in guardrails (`hallucination_check`, `word_count`, `no_pii`, `schema`), `chain()` combinator, retry-with-feedback logic, audit-chain integration.

---

### Task 21: `Guardrail` protocol + `GuardrailResult` dataclass

**Files:**
- Create: `src/cognithor/crew/guardrails/__init__.py`
- Create: `src/cognithor/crew/guardrails/base.py`
- Create: `tests/test_crew/test_guardrails/__init__.py`
- Create: `tests/test_crew/test_guardrails/test_base.py`

- [ ] **Step 1: Failing test**

```python
# tests/test_crew/test_guardrails/test_base.py
import pytest
from cognithor.crew.guardrails.base import Guardrail, GuardrailResult
from cognithor.crew.output import TaskOutput


class TestGuardrailResult:
    def test_pass_result(self):
        r = GuardrailResult(passed=True, feedback=None)
        assert r.passed
        assert r.feedback is None

    def test_fail_result(self):
        r = GuardrailResult(passed=False, feedback="too short")
        assert not r.passed
        assert r.feedback == "too short"

    def test_frozen(self):
        r = GuardrailResult(passed=True, feedback=None)
        with pytest.raises(Exception):
            r.passed = False  # type: ignore[misc]


class TestGuardrailProtocol:
    def test_callable_satisfies_protocol(self):
        def my_guard(output: TaskOutput) -> GuardrailResult:
            return GuardrailResult(passed=True, feedback=None)
        # Duck-typing check — the protocol is runtime-checkable
        assert callable(my_guard)
        result = my_guard(TaskOutput(task_id="t", agent_role="w", raw="x"))
        assert isinstance(result, GuardrailResult)
```

- [ ] **Step 2: Implement `base.py`**

```python
"""Guardrail protocol + result dataclass.

A Guardrail is a callable that takes a TaskOutput and returns a
GuardrailResult. Concrete implementations live in `function_guardrail.py`
(Python callable wrapper), `string_guardrail.py` (LLM-validated natural
language), and `builtin.py` (factory-produced presets).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from cognithor.crew.output import TaskOutput


class GuardrailResult(BaseModel):
    """Immutable verdict returned by every Guardrail."""

    model_config = ConfigDict(frozen=True)

    passed: bool
    feedback: str | None = None  # Required when passed is False
    pii_detected: bool = False   # Set by no_pii and related guardrails


@runtime_checkable
class Guardrail(Protocol):
    def __call__(self, output: TaskOutput) -> GuardrailResult: ...
```

Add `src/cognithor/crew/guardrails/__init__.py`:

```python
"""Cognithor Crew-Layer Guardrails."""

from __future__ import annotations

from cognithor.crew.guardrails.base import Guardrail, GuardrailResult

__all__ = ["Guardrail", "GuardrailResult"]
```

- [ ] **Step 3: Remove the PR 1 → PR 2 bridge-guard from `compiler.py`**

PR 1 (Task 8) added `_warn_if_guardrail_silently_ignored()` + its call sites as a foot-gun guard. Now that the guardrails module is available in the same release, that warning is noise — remove it cleanly. The real apply path lands in Task 29.

Delete from `src/cognithor/crew/compiler.py`:

```python
# DELETE: the _guardrails_available probe at module top
try:
    from cognithor.crew.guardrails import base as _guardrails_base  # noqa: F401
    _guardrails_available = True
except ImportError:
    _guardrails_available = False


# DELETE: the bridge-guard function
def _warn_if_guardrail_silently_ignored(task: CrewTask) -> None:
    ...
```

Delete the two `_warn_if_guardrail_silently_ignored(t)` call sites (one in `compile_and_run_sync`, one before the fan-out loop in `compile_and_run_async`). The real `_normalize_guardrail` path in Task 29 replaces them.

Add a regression test to lock in removal:

```python
# tests/test_crew/test_guardrails/test_no_silent_bridge.py
def test_no_bridge_guard_in_compiler():
    """Task 21: the PR 1 → PR 2 bridge guard is gone; real apply path used."""
    from cognithor.crew import compiler as m
    assert not hasattr(m, "_warn_if_guardrail_silently_ignored")
    assert not hasattr(m, "_guardrails_available")
```

- [ ] **Step 4: Run + commit**

```bash
touch tests/test_crew/test_guardrails/__init__.py
python -m pytest tests/test_crew/test_guardrails/test_base.py tests/test_crew/test_guardrails/test_no_silent_bridge.py -v
git add src/cognithor/crew/guardrails tests/test_crew/test_guardrails src/cognithor/crew/compiler.py
git commit -m "feat(crew): Guardrail protocol + GuardrailResult dataclass (+remove PR 1 bridge)"
```

---

### Task 22: Function-based guardrail wrapper

**Files:**
- Create: `src/cognithor/crew/guardrails/function_guardrail.py`
- Create: `tests/test_crew/test_guardrails/test_function.py`

- [ ] **Step 1: Failing test**

```python
# tests/test_crew/test_guardrails/test_function.py
import pytest
from cognithor.crew.guardrails.base import GuardrailResult
from cognithor.crew.guardrails.function_guardrail import FunctionGuardrail
from cognithor.crew.output import TaskOutput


def test_function_guardrail_passes():
    def min_len(out: TaskOutput) -> tuple[bool, str | TaskOutput]:
        return (True, out) if len(out.raw) >= 3 else (False, "too short")
    g = FunctionGuardrail(min_len)
    r = g(TaskOutput(task_id="t", agent_role="w", raw="hello"))
    assert isinstance(r, GuardrailResult)
    assert r.passed


def test_function_guardrail_fails_with_feedback():
    def min_len(out: TaskOutput) -> tuple[bool, str | TaskOutput]:
        return (False, "output ist kürzer als erwartet")
    g = FunctionGuardrail(min_len)
    r = g(TaskOutput(task_id="t", agent_role="w", raw="hi"))
    assert not r.passed
    assert r.feedback == "output ist kürzer als erwartet"


def test_function_guardrail_wraps_unexpected_exception_as_fail():
    def buggy(out: TaskOutput) -> tuple[bool, str | TaskOutput]:
        raise RuntimeError("unexpected")
    g = FunctionGuardrail(buggy)
    r = g(TaskOutput(task_id="t", agent_role="w", raw="x"))
    assert not r.passed
    assert "unexpected" in (r.feedback or "")
```

- [ ] **Step 2: Implement**

```python
"""Function-based guardrail — wraps a user callable into the protocol."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from cognithor.crew.guardrails.base import GuardrailResult
from cognithor.crew.output import TaskOutput


class FunctionGuardrail:
    """Adapter: user provides a callable with signature
        Callable[[TaskOutput], tuple[bool, str | TaskOutput]]
    and gets a Guardrail that catches exceptions + normalizes return shape.
    """

    def __init__(self, fn: Callable[[TaskOutput], tuple[bool, Any]]) -> None:
        self._fn = fn

    def __call__(self, output: TaskOutput) -> GuardrailResult:
        try:
            ok, payload = self._fn(output)
        except Exception as exc:
            return GuardrailResult(passed=False, feedback=f"Guardrail raised: {exc}")
        if ok:
            return GuardrailResult(passed=True, feedback=None)
        feedback = payload if isinstance(payload, str) else "validation failed"
        return GuardrailResult(passed=False, feedback=feedback)
```

- [ ] **Step 3: Run + commit**

```bash
python -m pytest tests/test_crew/test_guardrails/test_function.py -v
git add src/cognithor/crew/guardrails/function_guardrail.py tests/test_crew/test_guardrails/test_function.py
git commit -m "feat(crew): FunctionGuardrail adapter for user callables"
```

---

### Task 23: String-based guardrail (LLM-validated)

**Files:**
- Create: `src/cognithor/crew/guardrails/string_guardrail.py`
- Create: `tests/test_crew/test_guardrails/test_string.py`

- [ ] **Step 1: Scouted LLM-call path for validator calls**

`Planner` does **not** expose a generic `.chat()` method — the only public async entry point is `formulate_response(user_message, results, working_memory)`. That's too heavy for a binary pass/fail validator.

The right primitive is `cognithor.core.model_router.OllamaClient.chat(model, messages, ...)` which returns a dict-shaped Ollama response:

```python
async def chat(
    self,
    model: str,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    stream: bool = False,
    format_json: bool = False,
    options: dict[str, Any] | None = None,
    images: list[str] | None = None,
) -> dict[str, Any]: ...
```

Response shape: `{"message": {"content": "..."}, "prompt_eval_count": N, "eval_count": M, ...}`.

Spec §4.2 says the string guardrail "runs via the Gatekeeper" — architecturally the Gatekeeper already has access to the OllamaClient via the gateway. The Crew compiler already holds a Planner reference; the Planner internally has an `_ollama` attribute (`OllamaClient` or `UnifiedLLMClient`) that satisfies the `.chat(model, messages)` contract. So `StringGuardrail` accepts an LLM client duck-typed on `async def chat(model, messages)` — passing `planner._ollama` is how the compiler wires it.

- [ ] **Step 2: Failing test**

```python
# tests/test_crew/test_guardrails/test_string.py
from unittest.mock import AsyncMock, MagicMock
import pytest
from cognithor.crew.guardrails.string_guardrail import StringGuardrail
from cognithor.crew.output import TaskOutput


async def test_string_guardrail_passes_when_llm_says_yes():
    llm = MagicMock()
    # OllamaClient.chat returns a dict with nested message.content
    llm.chat = AsyncMock(return_value={
        "message": {"content": '{"passed": true, "feedback": null}'}
    })
    g = StringGuardrail("Output must be one sentence", llm_client=llm,
                       model="ollama/qwen3:8b")
    r = await g(TaskOutput(task_id="t", agent_role="w", raw="Hello."))
    assert r.passed


async def test_string_guardrail_fails_when_llm_says_no():
    llm = MagicMock()
    llm.chat = AsyncMock(return_value={
        "message": {"content": '{"passed": false, "feedback": "more than one sentence"}'}
    })
    g = StringGuardrail("one sentence", llm_client=llm, model="ollama/qwen3:8b")
    r = await g(TaskOutput(task_id="t", agent_role="w", raw="A. B."))
    assert not r.passed
    assert "more than one sentence" in (r.feedback or "")


async def test_string_guardrail_unparseable_llm_response_fails_safe():
    llm = MagicMock()
    llm.chat = AsyncMock(return_value={"message": {"content": "not json"}})
    g = StringGuardrail("x", llm_client=llm, model="ollama/qwen3:8b")
    r = await g(TaskOutput(task_id="t", agent_role="w", raw="y"))
    assert not r.passed
    assert "parse" in (r.feedback or "").lower()


async def test_string_guardrail_llm_unavailable_fails_safe():
    llm = MagicMock()
    llm.chat = AsyncMock(side_effect=ConnectionError("ollama down"))
    g = StringGuardrail("x", llm_client=llm, model="ollama/qwen3:8b")
    r = await g(TaskOutput(task_id="t", agent_role="w", raw="y"))
    assert not r.passed  # fail-safe: production can't silently skip validation
```

- [ ] **Step 3: Implement**

```python
"""String-based guardrail — LLM validates output against a natural-language rule.

The guardrail is **async** — it runs an LLM call via an OllamaClient-shaped
duck type (async `.chat(model, messages)` returning a dict with nested
`message.content`). The compiler awaits it inside `execute_task_async`.

Function-based guardrails (FunctionGuardrail) remain sync. The compiler
awaits only if the guardrail's `__call__` returns a coroutine.
"""

from __future__ import annotations

import inspect
import json
from typing import Any

from cognithor.crew.guardrails.base import GuardrailResult
from cognithor.crew.output import TaskOutput

_VALIDATOR_SYSTEM_PROMPT = (
    "You are a strict output validator. You will receive a RULE and an OUTPUT. "
    "Respond with a single JSON object: "
    '{"passed": boolean, "feedback": string_or_null}. '
    "If the output satisfies the rule, passed=true and feedback=null. "
    "If not, passed=false and feedback is a short German explanation."
)


class StringGuardrail:
    """LLM-validated guardrail. Offline-safe fallback: if the LLM is unavailable
    the result is `passed=False` with a clear feedback, so production can't
    skip validation silently.

    `llm_client` must expose an async `.chat(model, messages, ...)` method that
    returns an Ollama-shaped dict (`{"message": {"content": "..."}}`).
    `cognithor.core.model_router.OllamaClient` satisfies this contract directly;
    the compiler passes `planner._ollama` into this guardrail.
    """

    def __init__(
        self,
        rule: str,
        *,
        llm_client: Any,
        model: str,
    ) -> None:
        self._rule = rule
        self._llm = llm_client
        self._model = model

    async def __call__(self, output: TaskOutput) -> GuardrailResult:
        user_prompt = f"RULE: {self._rule}\n\nOUTPUT:\n{output.raw}"
        messages = [
            {"role": "system", "content": _VALIDATOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        raw = ""
        try:
            resp = await self._llm.chat(
                model=self._model,
                messages=messages,
                format_json=True,
                temperature=0.0,
            )
            raw = (resp.get("message", {}) or {}).get("content", "") or ""
        except Exception as exc:
            return GuardrailResult(
                passed=False,
                feedback=f"Validator-LLM nicht verfuegbar: {exc}",
            )

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return GuardrailResult(
                passed=False,
                feedback=f"Validator konnte LLM-Antwort nicht parsen: {raw[:100]}",
            )
        passed = bool(data.get("passed"))
        feedback = data.get("feedback") if not passed else None
        return GuardrailResult(passed=passed, feedback=feedback)
```

- [ ] **Step 3: Commit**

```bash
python -m pytest tests/test_crew/test_guardrails/test_string.py -v
git add src/cognithor/crew/guardrails/string_guardrail.py tests/test_crew/test_guardrails/test_string.py
git commit -m "feat(crew): StringGuardrail — LLM-validated natural-language rule"
```

---

### Task 24: Built-in guardrail `word_count`

**Files:**
- Create: `src/cognithor/crew/guardrails/builtin.py`
- Create: `tests/test_crew/test_guardrails/test_word_count.py`

- [ ] **Step 1: Test**

```python
# tests/test_crew/test_guardrails/test_word_count.py
import pytest
from cognithor.crew.guardrails.builtin import word_count
from cognithor.crew.output import TaskOutput


def _out(raw: str) -> TaskOutput:
    return TaskOutput(task_id="t", agent_role="w", raw=raw)


def test_word_count_min_pass():
    g = word_count(min_words=3)
    assert g(_out("one two three")).passed


def test_word_count_min_fail():
    g = word_count(min_words=5)
    r = g(_out("only three words"))
    assert not r.passed
    assert "5" in (r.feedback or "") or "mindestens" in (r.feedback or "").lower()


def test_word_count_max_pass():
    g = word_count(max_words=5)
    assert g(_out("one two")).passed


def test_word_count_max_fail():
    g = word_count(max_words=2)
    r = g(_out("one two three four"))
    assert not r.passed


def test_word_count_both_bounds():
    g = word_count(min_words=2, max_words=4)
    assert g(_out("a b c")).passed
    assert not g(_out("a")).passed
    assert not g(_out("a b c d e")).passed


def test_word_count_empty_string_fails_min():
    g = word_count(min_words=1)
    assert not g(_out("")).passed


def test_word_count_neither_bound_raises():
    with pytest.raises(ValueError):
        word_count()
```

- [ ] **Step 2: Implement (start of builtin.py — more factories added in later tasks)**

```python
"""Built-in Crew guardrail factories."""

from __future__ import annotations

from cognithor.crew.guardrails.base import GuardrailResult
from cognithor.crew.output import TaskOutput


def word_count(min_words: int | None = None, max_words: int | None = None):
    """Guardrail that checks output word count."""
    if min_words is None and max_words is None:
        raise ValueError("word_count requires at least min_words or max_words")

    def _guard(output: TaskOutput) -> GuardrailResult:
        count = len(output.raw.split())
        if min_words is not None and count < min_words:
            return GuardrailResult(
                passed=False,
                feedback=f"Output hat {count} Wörter, mindestens {min_words} erwartet.",
            )
        if max_words is not None and count > max_words:
            return GuardrailResult(
                passed=False,
                feedback=f"Output hat {count} Wörter, höchstens {max_words} erlaubt.",
            )
        return GuardrailResult(passed=True, feedback=None)

    return _guard
```

- [ ] **Step 3: Commit**

```bash
git add src/cognithor/crew/guardrails/builtin.py tests/test_crew/test_guardrails/test_word_count.py
git commit -m "feat(crew): word_count built-in guardrail"
```

---

### Task 25: Built-in guardrail `no_pii` (DE-focused)

**Files:**
- Modify: `src/cognithor/crew/guardrails/builtin.py` (add `no_pii`)
- Create: `tests/test_crew/test_guardrails/test_no_pii.py`

Spec §4.3: "blockt E-Mails, IBANs, Telefonnummern (DE-Format), Steuer-IDs".

- [ ] **Step 1: Test**

```python
# tests/test_crew/test_guardrails/test_no_pii.py
import pytest
from cognithor.crew.guardrails.builtin import no_pii
from cognithor.crew.output import TaskOutput


def _out(raw: str) -> TaskOutput:
    return TaskOutput(task_id="t", agent_role="w", raw=raw)


def test_clean_text_passes():
    g = no_pii()
    r = g(_out("Dies ist ein völlig harmloser Satz ohne persönliche Daten."))
    assert r.passed
    assert r.pii_detected is False


def test_email_detected():
    g = no_pii()
    r = g(_out("Kontakt: max.mustermann@example.com"))
    assert not r.passed
    assert r.pii_detected is True
    assert "email" in (r.feedback or "").lower() or "e-mail" in (r.feedback or "").lower()


def test_german_iban_detected():
    g = no_pii()
    r = g(_out("Konto: DE89 3704 0044 0532 0130 00"))
    assert not r.passed
    assert r.pii_detected is True


def test_german_phone_detected():
    g = no_pii()
    for ph in ["+49 30 12345678", "030 123 456 78", "0171-1234567", "0049 30 12345"]:
        r = g(_out(f"Telefon: {ph}"))
        assert not r.passed, f"Phone '{ph}' was not detected"


def test_german_steuer_id_11_digit_detected():
    g = no_pii()
    r = g(_out("Steuer-ID 12 345 678 901"))
    assert not r.passed


def test_multiple_pii_listed_in_feedback():
    g = no_pii()
    r = g(_out("Max: max@example.com, IBAN DE89 3704 0044 0532 0130 00"))
    assert not r.passed
    fb = (r.feedback or "").lower()
    assert "email" in fb or "e-mail" in fb
    assert "iban" in fb
```

- [ ] **Step 2: Implement**

Append to `builtin.py`:

```python
import re

# Regex patterns for common German PII
_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", re.IGNORECASE),
    "iban": re.compile(r"\bDE\d{2}(?:\s?\d{4}){4}\s?\d{2}\b"),
    "phone": re.compile(
        r"(?:\+49|0049|0)[\s.-]?\d{2,4}[\s.-]?\d{3,6}[\s.-]?\d{0,6}"
    ),
    "steuer_id": re.compile(r"\b\d{2}\s?\d{3}\s?\d{3}\s?\d{3}\b"),
}


def no_pii():
    """Guardrail that blocks outputs containing German PII.

    Detects email addresses, German IBANs, German phone numbers, and 11-digit
    Steuer-IDs. Emits a combined feedback listing every category found.
    """
    def _guard(output: TaskOutput) -> GuardrailResult:
        hits: list[str] = []
        for name, pat in _PATTERNS.items():
            if pat.search(output.raw):
                hits.append(name)
        if not hits:
            return GuardrailResult(passed=True, feedback=None, pii_detected=False)
        categories = ", ".join(hits)
        return GuardrailResult(
            passed=False,
            feedback=f"PII erkannt: {categories}. Bitte anonymisieren.",
            pii_detected=True,
        )

    return _guard
```

- [ ] **Step 3: Commit**

```bash
python -m pytest tests/test_crew/test_guardrails/test_no_pii.py -v
git add src/cognithor/crew/guardrails/builtin.py tests/test_crew/test_guardrails/test_no_pii.py
git commit -m "feat(crew): no_pii built-in guardrail (DE-focused)"
```

---

### Task 26: Built-in guardrail `schema` (Pydantic structured validation)

**Files:**
- Modify: `src/cognithor/crew/guardrails/builtin.py`
- Create: `tests/test_crew/test_guardrails/test_schema.py`

- [ ] **Step 1: Test**

```python
# tests/test_crew/test_guardrails/test_schema.py
from pydantic import BaseModel
from cognithor.crew.guardrails.builtin import schema
from cognithor.crew.output import TaskOutput


class Product(BaseModel):
    name: str
    price: float


def _out(raw: str) -> TaskOutput:
    return TaskOutput(task_id="t", agent_role="w", raw=raw)


def test_schema_passes_on_valid_json():
    g = schema(Product)
    r = g(_out('{"name": "Widget", "price": 9.99}'))
    assert r.passed


def test_schema_fails_on_missing_field():
    g = schema(Product)
    r = g(_out('{"name": "Widget"}'))
    assert not r.passed
    assert "price" in (r.feedback or "").lower()


def test_schema_fails_on_invalid_json():
    g = schema(Product)
    r = g(_out("not json"))
    assert not r.passed
    assert "json" in (r.feedback or "").lower()


def test_schema_fails_on_type_mismatch():
    g = schema(Product)
    r = g(_out('{"name": "x", "price": "not a number"}'))
    assert not r.passed
```

- [ ] **Step 2: Implement**

```python
from pydantic import BaseModel, ValidationError
import json as _json


def schema(model_cls: type[BaseModel]):
    """Guardrail that enforces a Pydantic schema on the output JSON."""
    def _guard(output: TaskOutput) -> GuardrailResult:
        try:
            data = _json.loads(output.raw)
        except _json.JSONDecodeError as exc:
            return GuardrailResult(
                passed=False, feedback=f"Output ist kein valides JSON: {exc}"
            )
        try:
            model_cls.model_validate(data)
        except ValidationError as exc:
            errs = "; ".join(
                f"{'/'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
            )
            return GuardrailResult(
                passed=False, feedback=f"Schema-Validierung fehlgeschlagen: {errs}"
            )
        return GuardrailResult(passed=True, feedback=None)
    return _guard
```

- [ ] **Step 3: Commit**

```bash
git add src/cognithor/crew/guardrails/builtin.py tests/test_crew/test_guardrails/test_schema.py
git commit -m "feat(crew): schema built-in guardrail with Pydantic validation"
```

---

### Task 27: Built-in guardrail `hallucination_check`

**Files:**
- Modify: `src/cognithor/crew/guardrails/builtin.py`
- Create: `tests/test_crew/test_guardrails/test_hallucination.py`

Spec §4.3: "vergleicht Output gegen Referenz-Kontext". Implementation: require that every factual claim (approximated by noun-phrases / numbers) appears somewhere in the reference text, with a configurable `min_overlap` ratio.

- [ ] **Step 1: Test**

```python
# tests/test_crew/test_guardrails/test_hallucination.py
from cognithor.crew.guardrails.builtin import hallucination_check
from cognithor.crew.output import TaskOutput


def _out(raw: str) -> TaskOutput:
    return TaskOutput(task_id="t", agent_role="w", raw=raw)


def test_passes_when_output_is_subset_of_reference():
    ref = "Der Tarif PrivatPlus kostet 450 Euro pro Monat und deckt stationäre Leistungen ab."
    g = hallucination_check(reference=ref)
    r = g(_out("PrivatPlus kostet 450 Euro."))
    assert r.passed


def test_fails_when_output_invents_a_number():
    ref = "Der Tarif kostet 450 Euro."
    g = hallucination_check(reference=ref)
    r = g(_out("Der Tarif kostet 99999 Euro."))
    assert not r.passed
    assert "99999" in (r.feedback or "")


def test_passes_when_exact_overlap_is_zero_but_min_is_zero():
    # Edge case: min_overlap=0 disables the check (useful as a test-only mode)
    g = hallucination_check(reference="x", min_overlap=0.0)
    r = g(_out("completely unrelated"))
    assert r.passed
```

- [ ] **Step 2: Implement**

```python
def hallucination_check(*, reference: str, min_overlap: float = 0.5):
    """Compare output tokens against a reference corpus. Fails when too few
    of the output's informative tokens appear in the reference (simple
    heuristic — not a substitute for retrieval grounding).
    """
    ref_tokens = {t.lower() for t in reference.split() if len(t) > 2}

    _number_re = re.compile(r"\b\d{3,}\b")  # 3+ digit numbers

    def _guard(output: TaskOutput) -> GuardrailResult:
        if min_overlap <= 0.0:
            return GuardrailResult(passed=True, feedback=None)

        out_tokens = [t.lower() for t in output.raw.split() if len(t) > 2]
        if not out_tokens:
            return GuardrailResult(passed=True, feedback=None)

        overlap = sum(1 for t in out_tokens if t in ref_tokens) / len(out_tokens)

        # Additionally fail when any 3+ digit number in the output is not in the reference
        invented = [n for n in _number_re.findall(output.raw) if n not in reference]
        if invented:
            return GuardrailResult(
                passed=False,
                feedback=f"Output enthält Zahlen ohne Referenz-Nachweis: {', '.join(invented)}",
            )
        if overlap < min_overlap:
            return GuardrailResult(
                passed=False,
                feedback=f"Output-Referenz-Überlappung {overlap:.0%} unter Schwelle {min_overlap:.0%}.",
            )
        return GuardrailResult(passed=True, feedback=None)
    return _guard
```

- [ ] **Step 3: Commit**

```bash
python -m pytest tests/test_crew/test_guardrails/test_hallucination.py -v
git add src/cognithor/crew/guardrails/builtin.py tests/test_crew/test_guardrails/test_hallucination.py
git commit -m "feat(crew): hallucination_check built-in guardrail (reference-overlap)"
```

---

### Task 28: `chain()` combinator + public guardrails exports

**Files:**
- Modify: `src/cognithor/crew/guardrails/builtin.py`
- Modify: `src/cognithor/crew/guardrails/__init__.py`
- Create: `tests/test_crew/test_guardrails/test_chain.py`

- [ ] **Step 1: Test**

```python
# tests/test_crew/test_guardrails/test_chain.py
#
# R4-C4: `chain()` returns an ASYNC callable (needed so StringGuardrail, whose
# __call__ is async, actually runs). All tests here await the chained result.
import pytest
from cognithor.crew.guardrails.builtin import chain, word_count, no_pii
from cognithor.crew.output import TaskOutput


def _out(raw: str) -> TaskOutput:
    return TaskOutput(task_id="t", agent_role="w", raw=raw)


async def test_chain_all_pass():
    g = chain(word_count(min_words=1), no_pii())
    result = await g(_out("Hallo Welt"))
    assert result.passed


async def test_chain_stops_on_first_failure():
    calls = []
    def tracker(label):
        def _g(out):
            calls.append(label)
            from cognithor.crew.guardrails.base import GuardrailResult
            return GuardrailResult(passed=(label != "B"), feedback=f"from-{label}")
        return _g

    g = chain(tracker("A"), tracker("B"), tracker("C"))
    r = await g(_out("x"))
    assert not r.passed
    assert r.feedback == "from-B"
    assert calls == ["A", "B"]  # C never runs


async def test_chain_pii_in_first_fails_even_if_second_would_pass():
    g = chain(no_pii(), word_count(min_words=1))
    r = await g(_out("Kontakt: x@example.com"))
    assert not r.passed
    assert r.pii_detected is True


async def test_chain_awaits_async_guardrails():
    """R4-C4 regression: async guards inside chain() MUST actually run.

    Previously chain() was synchronous — a coroutine returned from the first
    async guard was truthy, its `.passed` attribute-access failed silently,
    and the second guard never executed. This test forces an async guard to
    run and verifies the sync guard downstream is reached.
    """
    from cognithor.crew.guardrails.base import GuardrailResult

    call_count = {"async_g": 0, "sync_g": 0}

    async def async_g(_out):
        call_count["async_g"] += 1
        return GuardrailResult(passed=True, feedback=None)

    def sync_g(_out):
        call_count["sync_g"] += 1
        return GuardrailResult(passed=True, feedback=None)

    g = chain(async_g, sync_g)
    r = await g(_out("anything"))
    assert r.passed
    assert call_count == {"async_g": 1, "sync_g": 1}


async def test_chain_short_circuits_on_first_async_failure():
    """First (async) guard fails → second guard never called."""
    from cognithor.crew.guardrails.base import GuardrailResult

    second_calls = []

    async def failing_async(_out):
        return GuardrailResult(passed=False, feedback="blocked-async")

    def never_called(_out):
        second_calls.append(1)
        return GuardrailResult(passed=True, feedback=None)

    g = chain(failing_async, never_called)
    r = await g(_out("irrelevant"))
    assert not r.passed
    assert r.feedback == "blocked-async"
    assert second_calls == []  # short-circuit honored
```

- [ ] **Step 2: Implement `chain()` and wire all exports**

```python
import inspect

def chain(*guards):
    """Run guardrails in order; first failure short-circuits.

    R4-C4: this combinator MUST be async so ``StringGuardrail`` (whose
    ``__call__`` is ``async def``) actually runs. The previous synchronous
    version invoked ``g(output)`` and got a coroutine back — which is always
    truthy — so ``if not r.passed`` was evaluated against an un-awaited
    coroutine, and the second guardrail never ran. The ``versicherungs-vergleich``
    template's ``chain(no_pii(), StringGuardrail(...))`` required this fix.

    Returned ``GuardrailResult`` preserves the pii_detected flag from
    whichever guard signaled it, so the audit-chain record is complete.
    """
    async def _combined(output: TaskOutput) -> GuardrailResult:
        for g in guards:
            r = g(output)
            if inspect.iscoroutine(r):
                r = await r
            if not r.passed:
                return r
        return GuardrailResult(passed=True, feedback=None)
    return _combined
```

`chain()` now returns an async callable. The compiler's `_call_guardrail`
retry loop (Task 29) already detects coroutines via ``inspect.iscoroutine``,
so the async return path Just Works without further plumbing.

Update `__init__.py`:

```python
from cognithor.crew.errors import GuardrailFailure
from cognithor.crew.guardrails.base import Guardrail, GuardrailResult
from cognithor.crew.guardrails.builtin import (
    chain, hallucination_check, no_pii, schema, word_count,
)
from cognithor.crew.guardrails.function_guardrail import FunctionGuardrail
from cognithor.crew.guardrails.string_guardrail import StringGuardrail

__all__ = [
    "FunctionGuardrail",
    "Guardrail",
    "GuardrailFailure",  # re-exported from cognithor.crew.errors so users
                         # have one obvious import location
    "GuardrailResult",
    "StringGuardrail",
    "chain",
    "hallucination_check",
    "no_pii",
    "schema",
    "word_count",
]
```

- [ ] **Step 3: Commit**

```bash
python -m pytest tests/test_crew/test_guardrails/test_chain.py -v
git add src/cognithor/crew/guardrails
git commit -m "feat(crew): chain() combinator + public guardrails exports"
```

---

### Task 29: Guardrail execution in the compiler (retry + GuardrailFailure)

**Files:**
- Modify: `src/cognithor/crew/compiler.py`
- Create: `tests/test_crew/test_guardrails/test_compiler_integration.py`

Spec §4.2: "Nach `max_retries` (default 2) Abbruch mit `GuardrailFailure`-Exception".

- [ ] **Step 1: Test**

```python
# tests/test_crew/test_guardrails/test_compiler_integration.py
from unittest.mock import AsyncMock, MagicMock
import pytest
from cognithor.crew import Crew, CrewAgent, CrewTask
from cognithor.crew.errors import GuardrailFailure
from cognithor.crew.guardrails.base import GuardrailResult
from cognithor.crew.output import TaskOutput
from cognithor.core.observer import ResponseEnvelope


async def test_guardrail_failure_retries_then_raises():
    agent = CrewAgent(role="writer", goal="write")
    def fail_twice(_out):
        return GuardrailResult(passed=False, feedback="zu kurz")
    task = CrewTask(description="write", expected_output="long text",
                   agent=agent, guardrail=fail_twice, max_retries=2)

    call_count = {"n": 0}
    async def fake(user_message, results, working_memory):
        call_count["n"] += 1
        return ResponseEnvelope(content=f"attempt-{call_count['n']}", directive=None)
    mock_planner = MagicMock()
    mock_planner.formulate_response = AsyncMock(side_effect=fake)
    mock_registry = MagicMock()
    mock_registry.get_tools_for_role.return_value = []

    crew = Crew(agents=[agent], tasks=[task], planner=mock_planner)

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("cognithor.crew.runtime.get_default_tool_registry", lambda: mock_registry)
        with pytest.raises(GuardrailFailure) as exc_info:
            await crew.kickoff_async()

    # GuardrailFailure carries the real attempt count, not max_retries
    # Initial try + max_retries == 3 attempts total
    assert call_count["n"] == 3
    assert exc_info.value.attempts == 3
    assert "zu kurz" in exc_info.value.reason
    assert "after 3 attempt(s)" in str(exc_info.value)


async def test_guardrail_passes_after_retry():
    agent = CrewAgent(role="writer", goal="write")
    attempts = {"n": 0}
    def pass_on_second(_out):
        attempts["n"] += 1
        return GuardrailResult(passed=(attempts["n"] >= 2), feedback="try again")

    task = CrewTask(description="x", expected_output="y",
                   agent=agent, guardrail=pass_on_second, max_retries=2)

    mock_planner = MagicMock()
    mock_planner.formulate_response = AsyncMock(
        return_value=ResponseEnvelope(content="text", directive=None),
    )
    mock_registry = MagicMock()
    mock_registry.get_tools_for_role.return_value = []

    crew = Crew(agents=[agent], tasks=[task], planner=mock_planner)

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("cognithor.crew.runtime.get_default_tool_registry", lambda: mock_registry)
        result = await crew.kickoff_async()

    assert result.tasks_output[0].guardrail_verdict == "pass"
```

- [ ] **Step 2: Implement guardrail evaluation in `execute_task_async`**

Inside `execute_task_async` (from Task 11), after the Planner returns a response, add:

```python
import inspect
from cognithor.crew.errors import GuardrailFailure
from cognithor.crew.guardrails.base import GuardrailResult
from cognithor.crew.guardrails.function_guardrail import FunctionGuardrail
from cognithor.crew.guardrails.string_guardrail import StringGuardrail


def _normalize_guardrail(g: Any, *, ollama_client: Any, model: str) -> Any:
    """Normalize whatever the user stuck into `CrewTask.guardrail` into a callable.

    - None                -> None
    - str (rule text)     -> StringGuardrail(rule, llm_client=ollama_client, model=model)
    - already a Guardrail (has __call__ returning GuardrailResult) -> returned as-is
    - any other callable  -> wrapped in FunctionGuardrail for exception safety
    """
    if g is None:
        return None
    if isinstance(g, str):
        return StringGuardrail(g, llm_client=ollama_client, model=model)
    if _is_already_guardrail(g):
        return g
    if callable(g):
        return FunctionGuardrail(g)
    return g


def _is_already_guardrail(g: Any) -> bool:
    """Duck-type check: Guardrails (FunctionGuardrail, StringGuardrail, chain-wrapper)
    have a `__call__` AND either a `_rule` attribute (StringGuardrail) or
    a `_fn` attribute (FunctionGuardrail) or are builtin closures. Anything else
    the user passes is treated as a raw callable and wrapped.
    """
    return hasattr(g, "_rule") or hasattr(g, "_fn") or getattr(g, "_is_guardrail", False)


async def _call_guardrail(guardrail: Any, out: TaskOutput) -> GuardrailResult:
    """Invoke a guardrail — may be sync or async. Awaits if coroutine-returning."""
    result = guardrail(out)
    if inspect.iscoroutine(result):
        result = await result
    return result


# Inside execute_task_async, after the first `envelope = await planner.formulate_response(...)`:
# The string-guardrail path needs an OllamaClient. We pull it off the Planner;
# both `Planner` and `UnifiedLLMClient` expose a `.chat()` compatible shim via the
# `_ollama` attribute (see planner.py:509).
ollama_client = getattr(planner, "_ollama", None)
guardrail_model = task.agent.llm or "ollama/qwen3:8b"
guardrail = _normalize_guardrail(task.guardrail, ollama_client=ollama_client, model=guardrail_model)

attempts = 0
verdict = "skipped"
result: GuardrailResult | None = None
while True:
    out = TaskOutput(
        task_id=task.task_id, agent_role=task.agent.role, raw=raw,
        duration_ms=duration_ms, token_usage=usage,
    )
    if guardrail is None:
        verdict = "skipped"
        break
    result = await _call_guardrail(guardrail, out)
    if result.passed:
        verdict = "pass"
        break
    attempts += 1
    if attempts > task.max_retries:
        raise GuardrailFailure(
            task_id=task.task_id,
            guardrail_name=type(guardrail).__name__,
            attempts=attempts,
            reason=result.feedback or "(no feedback)",
        )
    # Retry: re-invoke Planner with a retry-nudge synthesized as an extra
    # ToolResult carrying the feedback. This keeps the Planner API stable.
    #
    # R4-I3: ``tool_name`` uses a namespaced ``crew:`` prefix so audit-log
    # scanners and the Gatekeeper's ``_classify_risk`` tool-name lookup never
    # confuse this synthetic retry-feedback blob with a real tool invocation.
    # (The Gatekeeper only inspects real ``PlannedAction`` objects — this
    # ToolResult never reaches it — but the namespaced prefix is defensive
    # hygiene for anyone grepping audit.jsonl by tool name.)
    retry_context = prior_results + [
        ToolResult(
            tool_name="crew:retry_feedback",
            content=f"Vorheriger Versuch wurde abgelehnt. Feedback: {result.feedback}. "
                    f"Bitte erneut versuchen und die Kritik einarbeiten.",
            is_error=False,
        )
    ]
    t0 = time.perf_counter()
    envelope = await planner.formulate_response(user_message, retry_context, working_memory)
    duration_ms = (time.perf_counter() - t0) * 1000.0
    raw = getattr(envelope, "content", "") or ""
    usage = _read_token_usage(planner) or TokenUsageDict(
        prompt_tokens=0, completion_tokens=0, total_tokens=0,
    )

# Attach verdict to the final output (Pydantic frozen; use model_copy)
return out.model_copy(update={"guardrail_verdict": verdict})
```

(Pydantic frozen models support `.model_copy(update=...)` — the final output gets the verdict attached.)

- [ ] **Step 3: Commit**

```bash
python -m pytest tests/test_crew/test_guardrails/test_compiler_integration.py -v
git add src/cognithor/crew/compiler.py tests/test_crew/test_guardrails/test_compiler_integration.py
git commit -m "feat(crew): guardrail execution with retry-with-feedback + GuardrailFailure"
```

---

### Task 30: Guardrail audit-chain integration

**Files:**
- Modify: `src/cognithor/crew/compiler.py` (emit guardrail events)
- Create: `tests/test_crew/test_guardrails/test_audit.py`

- [ ] **Step 1: Test**

```python
# tests/test_crew/test_guardrails/test_audit.py
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from cognithor.crew import Crew, CrewAgent, CrewTask
from cognithor.crew.guardrails.base import GuardrailResult


async def test_guardrail_pass_audited():
    agent = CrewAgent(role="writer", goal="write")
    task = CrewTask(description="x", expected_output="y", agent=agent,
                   guardrail=lambda o: GuardrailResult(passed=True, feedback=None))

    events: list = []
    def spy(name, **fields): events.append((name, fields))

    from cognithor.core.observer import ResponseEnvelope
    mock_planner = MagicMock()
    mock_planner.formulate_response = AsyncMock(
        return_value=ResponseEnvelope(content="ok", directive=None),
    )
    mock_registry = MagicMock()
    mock_registry.get_tools_for_role.return_value = []

    crew = Crew(agents=[agent], tasks=[task], planner=mock_planner)

    with patch("cognithor.crew.compiler.append_audit", side_effect=spy), \
         pytest.MonkeyPatch().context() as mp:
        mp.setattr("cognithor.crew.runtime.get_default_tool_registry", lambda: mock_registry)
        await crew.kickoff_async()

    guardrail_events = [e for e in events if "guardrail" in e[0]]
    assert guardrail_events
    assert any(fields.get("verdict") == "pass" for _name, fields in guardrail_events)
```

- [ ] **Step 2: Emit events inside the guardrail retry loop — plumb parent trace_id**

The compiler's `compile_and_run_async` already mints a single `trace_id = _uuid.uuid4().hex` per kickoff. That trace id must be plumbed DOWN into `execute_task_async` so guardrail audit events carry parent correlation — otherwise they orphan as `trace_id=None` and the audit-chain viewer cannot link guardrail verdicts back to their owning kickoff.

Add `trace_id` as a keyword arg to `execute_task_async`:

```python
async def execute_task_async(
    task: CrewTask,
    *,
    context: list[TaskOutput],
    inputs: dict[str, Any] | None,
    registry: Any,
    planner: Any,
    trace_id: str | None = None,  # NEW — plumbed down from compile_and_run_async
) -> TaskOutput:
    ...
```

Update both fan-out call sites in `compile_and_run_async` to forward `trace_id`:

```python
# Single task path
out = await execute_task_async(
    group[0], context=outputs, inputs=inputs,
    registry=registry, planner=planner, trace_id=trace_id,
)

# Parallel gather path
parallel_outs = await asyncio.gather(*[
    execute_task_async(
        t, context=outputs, inputs=inputs, registry=registry,
        planner=planner, trace_id=trace_id,
    )
    for t in group
])
```

Inside `execute_task_async`, after evaluating `result`, emit with the parent trace_id:

```python
append_audit(
    "crew_guardrail_check",
    trace_id=trace_id,  # parent correlation — links verdict back to the kickoff
    task_id=task.task_id,
    verdict="pass" if result.passed else "fail",
    retry_count=attempts,
    pii_detected=result.pii_detected,
    feedback=result.feedback,
)
```

Update the existing test `test_guardrail_pass_audited` to assert the guardrail audit entry carries the parent trace_id:

```python
# In tests/test_crew/test_guardrails/test_audit.py
result = await crew.kickoff_async()
guardrail_events = [e for e in events if "guardrail" in e[0]]
assert guardrail_events
for name, fields in guardrail_events:
    assert fields.get("trace_id") == result.trace_id, (
        f"Guardrail event '{name}' lost parent trace_id — "
        f"expected {result.trace_id}, got {fields.get('trace_id')}"
    )
```

- [ ] **Step 3: Commit**

```bash
git add src/cognithor/crew/compiler.py tests/test_crew/test_guardrails/test_audit.py
git commit -m "feat(crew): guardrail verdicts recorded in Hashline-Guard audit chain"
```

---

### Task 31: Feature-4 integration test (versicherungs-vergleich with no_pii + custom string guardrail)

**Files:**
- Create: `tests/test_crew/test_guardrails/test_versicherungs_integration.py`

Spec §4.5 AC 5: "Das `versicherungs-vergleich`-Template nutzt `no_pii()` UND einen custom String-Guardrail ('keine Tarif-Empfehlung, nur Vergleich')." **Both** guardrails must be wired — the integration test asserts chain(no_pii, StringGuardrail) is present and functional.

- [ ] **Step 1: Test — PII path + string-guardrail path**

```python
# tests/test_crew/test_guardrails/test_versicherungs_integration.py
from unittest.mock import AsyncMock, MagicMock
import pytest
from cognithor.crew import Crew, CrewAgent, CrewTask
from cognithor.crew.guardrails import StringGuardrail, chain, no_pii
from cognithor.crew.errors import GuardrailFailure
from cognithor.core.observer import ResponseEnvelope


def _mock_ollama_client(validator_verdict: dict) -> MagicMock:
    """Build an OllamaClient-shaped mock returning a JSON-wrapped verdict."""
    import json
    client = MagicMock()
    client.chat = AsyncMock(return_value={
        "message": {"content": json.dumps(validator_verdict)},
    })
    return client


async def test_versicherungs_crew_blocks_pii_output():
    agent = CrewAgent(role="analyst", goal="compare PKV tariffs",
                     llm="ollama/qwen3:8b")
    # Validator LLM (OllamaClient stand-in) passes every check — but no_pii runs first
    ollama = _mock_ollama_client({"passed": True, "feedback": None})

    neutral_rule = StringGuardrail(
        "Output darf keine Tarif-Empfehlung enthalten, nur neutralen Vergleich",
        llm_client=ollama,
        model="ollama/qwen3:8b",
    )
    task = CrewTask(
        description="Compare",
        expected_output="Tabular comparison",
        agent=agent,
        guardrail=chain(no_pii(), neutral_rule),
        max_retries=0,
    )

    mock_planner = MagicMock()
    mock_planner._ollama = ollama
    mock_planner.formulate_response = AsyncMock(
        return_value=ResponseEnvelope(
            content="Kontakt: sachbearbeiter@versicherer.de zur Beratung.",
            directive=None,
        )
    )
    mock_registry = MagicMock()
    mock_registry.get_tools_for_role.return_value = []

    crew = Crew(agents=[agent], tasks=[task], planner=mock_planner)

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("cognithor.crew.runtime.get_default_tool_registry", lambda: mock_registry)
        with pytest.raises(GuardrailFailure, match="PII erkannt"):
            await crew.kickoff_async()


async def test_versicherungs_crew_blocks_tarif_recommendation():
    """The string guardrail catches outputs that make recommendations (not just compare)."""
    agent = CrewAgent(role="analyst", goal="compare PKV tariffs",
                     llm="ollama/qwen3:8b")
    # Validator LLM says "no, this is a recommendation, not a comparison"
    ollama = _mock_ollama_client({
        "passed": False,
        "feedback": "Output enthält eine Empfehlung ('empfehle Tarif A'); nur Vergleich erlaubt.",
    })

    neutral_rule = StringGuardrail(
        "Output darf keine Tarif-Empfehlung enthalten, nur neutralen Vergleich",
        llm_client=ollama,
        model="ollama/qwen3:8b",
    )
    task = CrewTask(
        description="Compare",
        expected_output="Tabular comparison",
        agent=agent,
        guardrail=chain(no_pii(), neutral_rule),
        max_retries=0,
    )

    mock_planner = MagicMock()
    mock_planner._ollama = ollama
    # Clean output (no PII) that recommends a tariff — no_pii passes, string-guard fails
    mock_planner.formulate_response = AsyncMock(
        return_value=ResponseEnvelope(
            content="Ich empfehle Tarif A fuer Ihre Situation.",
            directive=None,
        )
    )
    mock_registry = MagicMock()
    mock_registry.get_tools_for_role.return_value = []

    crew = Crew(agents=[agent], tasks=[task], planner=mock_planner)

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("cognithor.crew.runtime.get_default_tool_registry", lambda: mock_registry)
        with pytest.raises(GuardrailFailure, match="Empfehlung"):
            await crew.kickoff_async()
```

- [ ] **Step 2: Commit**

```bash
python -m pytest tests/test_crew/test_guardrails/test_versicherungs_integration.py -v
git add tests/test_crew/test_guardrails/test_versicherungs_integration.py
git commit -m "test(crew): versicherungs-vergleich guardrail integration"
```

---

### Task 32: Feature-4 merge-prep (CHANGELOG + docstring sweep)

**Files:**
- Modify: `CHANGELOG.md`
- Modify: `src/cognithor/crew/guardrails/__init__.py` (docstring + usage example)

- [ ] **Step 1: Update CHANGELOG**

Inside the `[Unreleased]` section added in Task 20, append to the `### Added`:

```markdown
- **`cognithor.crew.guardrails` — Task-Level Guardrails (Feature 4)** — function-
  based + string-based validators, built-in `hallucination_check`, `word_count`,
  `no_pii` (DE-focused), `schema` (Pydantic), plus `chain()` combinator. Failures
  trigger retry-with-feedback up to `task.max_retries`, then raise
  `GuardrailFailure`. Every verdict is recorded in the Hashline-Guard audit chain
  with PII-detection flag.
```

The `### Breaking Changes\nNone.` block from Task 20 stays unchanged — Feature 4 is strictly additive.

- [ ] **Step 2: Guardrails `__init__.py` docstring with usage example**

Expand the module docstring:

```python
"""Cognithor Crew-Layer Guardrails.

Two flavors:
  * function-based — Python callable, pass Python to `CrewTask(guardrail=fn)`
  * string-based — natural language rule, evaluated by an LLM

Built-ins (factories):
  * word_count(min_words=..., max_words=...)
  * no_pii()
  * hallucination_check(reference=..., min_overlap=...)
  * schema(pydantic_model)
  * chain(*guardrails)

Example:
    from cognithor.crew import Crew, CrewAgent, CrewTask
    from cognithor.crew.guardrails import chain, no_pii, word_count

    task = CrewTask(
        description="Draft a customer email",
        expected_output="...",
        agent=writer,
        guardrail=chain(no_pii(), word_count(min_words=80, max_words=200)),
        max_retries=2,
    )
"""
```

- [ ] **Step 3: Run full Feature-4 test suite + ruff**

```bash
python -m pytest tests/test_crew/test_guardrails/ -v 2>&1 | tail -15
python -m ruff check src/cognithor/crew/guardrails tests/test_crew/test_guardrails
python -m ruff format --check src/cognithor/crew/guardrails tests/test_crew/test_guardrails
```

- [ ] **Step 4: Commit**

```bash
git add CHANGELOG.md src/cognithor/crew/guardrails/__init__.py
git commit -m "docs(crew): CHANGELOG + guardrails usage example"
```

---

# FEATURE 3 — `cognithor init` CLI + 5 Templates (Tasks 33-52)

Implements spec §3. CLI subcommand `cognithor init <name> --template <t>`, 5 first-party Jinja2 templates, `cognithor run` in-project runner, integration with existing skills scaffolder utilities.

---

### Task 33: Add Jinja2 to runtime deps

**Files:**
- Modify: `pyproject.toml` (add `jinja2>=3.1,<4`)

- [ ] **Step 1: Edit `pyproject.toml`** — locate the `dependencies = [` block:

```toml
dependencies = [
    # ... existing ...
    "python-dotenv>=1.0,<2",
    "jinja2>=3.1,<4",  # Crew-Layer template scaffolder (Feature 3)
]
```

- [ ] **Step 2: Verify install**

```bash
pip install -e .
python -c "import jinja2; print(jinja2.__version__)"
```

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "build: add jinja2>=3.1 runtime dep for Crew templates"
```

---

### Task 34: `cli.scaffolder` — Jinja2 render helper (shared)

**Files:**
- Create: `src/cognithor/crew/cli/__init__.py`
- Create: `src/cognithor/crew/cli/scaffolder.py`
- Create: `tests/test_crew/test_cli/__init__.py`
- Create: `tests/test_crew/test_cli/test_scaffolder.py`

- [ ] **Step 1: Test**

```python
# tests/test_crew/test_cli/test_scaffolder.py
from pathlib import Path
from cognithor.crew.cli.scaffolder import render_tree, sanitize_project_name


class TestSanitize:
    def test_spaces_to_underscore(self):
        assert sanitize_project_name("My Research Crew") == "my_research_crew"

    def test_hyphens_to_underscore(self):
        assert sanitize_project_name("my-crew") == "my_crew"

    def test_leading_digit_prefixed(self):
        assert sanitize_project_name("123abc") == "project_123abc"

    def test_empty_raises(self):
        import pytest
        with pytest.raises(ValueError):
            sanitize_project_name("")


class TestRenderTree:
    def test_renders_jinja_templates(self, tmp_path: Path):
        src = tmp_path / "src_templates"
        src.mkdir()
        (src / "hello.py.jinja").write_text("print('{{ project_name }}')")
        (src / "README.md.jinja").write_text("# {{ project_name | title }}")
        (src / "plain.txt").write_text("no substitution")  # non-.jinja copied as-is
        dest = tmp_path / "out"

        render_tree(src, dest, context={"project_name": "my_crew"})

        assert (dest / "hello.py").read_text() == "print('my_crew')"
        assert (dest / "README.md").read_text() == "# My_Crew"
        assert (dest / "plain.txt").read_text() == "no substitution"

    def test_refuses_non_empty_dest(self, tmp_path: Path):
        import pytest
        (tmp_path / "out").mkdir()
        (tmp_path / "out" / "existing.txt").write_text("already here")
        with pytest.raises(FileExistsError):
            render_tree(tmp_path / "src", tmp_path / "out", context={})
```

- [ ] **Step 2: Implement**

```python
# src/cognithor/crew/cli/scaffolder.py
"""Jinja2-based directory tree renderer for cognithor init templates.

Uses ``SandboxedEnvironment`` (not plain ``Environment``) so untrusted
template content cannot access Python internals via ``__class__`` /
``__mro__`` / ``__subclasses__`` tricks. HTML/XML autoescape is enabled for
future HTML template support. Path segments are validated per-render to
block traversal attacks like a filename literally containing
``{{ '../../etc/passwd' }}``. See NC3 in Round 3 review.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

from jinja2 import FileSystemLoader, StrictUndefined, select_autoescape
from jinja2.sandbox import SandboxedEnvironment


_PROJECT_NAME_CLEAN = re.compile(r"[^a-zA-Z0-9_]")

# Windows reserved device names that must never appear as file or project
# basenames — touching ``CON``, ``NUL``, ``COM1``–``COM9``, ``LPT1``–``LPT9``
# on NTFS returns to the console device and corrupts anything that tries to
# read them. See NI4 in Round 3 review.
_WIN_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}

# Path segments that must never appear AFTER rendering — blocks
# ``{{ '../../etc/passwd' }}`` in a template filename escaping dest_dir.
_FORBIDDEN_SEGMENTS = {"", ".", ".."}

# Language-specific template suffixes. ``README.md.jinja.de`` selects the DE
# variant when ``context["lang"] == "de"``; ``README.md.jinja.en`` is skipped.
# Without this handling, the plain ``.suffix == ".jinja"`` check would NOT match
# (because the real suffix is ``.de`` / ``.en``), so the file would be copied
# verbatim and both variants would ship side-by-side in the scaffold. See R4-C1.
_LANG_SUFFIXES = {".de", ".en", ".zh", ".ar"}


def sanitize_project_name(name: str) -> str:
    """Convert free-form name to a safe Python package identifier.

    Rejects empty strings and Windows reserved device names (``CON``, ``NUL``,
    ``COM1..9``, ``LPT1..9``, ``PRN``, ``AUX``) on ALL platforms — not just
    Windows — so projects scaffolded on Linux remain portable to Windows
    developers. See NI4 in Round 3 review.
    """
    if not name or not name.strip():
        raise ValueError("project name cannot be empty")
    cleaned = _PROJECT_NAME_CLEAN.sub("_", name.strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        raise ValueError(f"project name reduces to empty: {name!r}")
    if cleaned[0].isdigit():
        cleaned = f"project_{cleaned}"
    if cleaned.upper() in _WIN_RESERVED:
        raise ValueError(
            f"'{cleaned}' is a reserved Windows device name. "
            f"Please choose a different name (e.g. '{cleaned}_app')."
        )
    return cleaned


def _build_env(src_dir: Path) -> SandboxedEnvironment:
    """Construct a sandboxed Jinja2 environment with autoescape on."""
    return SandboxedEnvironment(
        loader=FileSystemLoader(str(src_dir)),
        undefined=StrictUndefined,
        autoescape=select_autoescape(["html", "xml"]),
        keep_trailing_newline=True,
        trim_blocks=False,
        lstrip_blocks=False,
    )


def _safe_join(dest_dir: Path, rendered_parts: list[str]) -> Path:
    """Validate every rendered segment, then join under dest_dir.

    Raises ``ValueError`` on any sign of path traversal — forbidden tokens
    (``.``, ``..``, empty), embedded separators (``/`` or ``\\``), or a
    rendered basename that lands outside ``dest_dir`` after ``resolve()``.
    """
    for seg in rendered_parts:
        if seg in _FORBIDDEN_SEGMENTS:
            raise ValueError(f"Forbidden path segment after render: {seg!r}")
        if "/" in seg or "\\" in seg or seg.startswith(".."):
            raise ValueError(f"Path traversal in rendered segment: {seg!r}")
    candidate = dest_dir.joinpath(*rendered_parts).resolve()
    try:
        candidate.relative_to(dest_dir.resolve())
    except ValueError as exc:
        raise ValueError(
            f"Rendered path {candidate} escapes dest_dir {dest_dir}"
        ) from exc
    return candidate


_LANG_FALLBACK = "en"


def _select_language_files(src_dir: Path, lang: str) -> set[Path]:
    """For each base path with language variants, pick exactly ONE to render.

    Fallback order per base (R5 fix for zh-no-README regression):
      1. Requested ``lang`` variant if it exists
      2. ``en`` variant as universal fallback
      3. First (alphabetically) available variant as last resort

    Files WITHOUT a language suffix are NOT in the returned set — the render
    loop always processes those unconditionally. The returned set only gates
    files that HAVE a language suffix.
    """
    variants: dict[Path, dict[str, Path]] = {}
    for src in src_dir.rglob("*"):
        if not src.is_file() or src.suffix not in _LANG_SUFFIXES:
            continue
        base = src.with_suffix("")  # path without .de/.en/.zh/.ar
        file_lang = src.suffix.lstrip(".")
        variants.setdefault(base, {})[file_lang] = src

    selected: set[Path] = set()
    for _base, by_lang in variants.items():
        if lang in by_lang:
            selected.add(by_lang[lang])
        elif _LANG_FALLBACK in by_lang:
            selected.add(by_lang[_LANG_FALLBACK])
        else:
            first = sorted(by_lang.keys())[0]
            selected.add(by_lang[first])
    return selected


def _resolve_language_variant(
    src_path: Path,
    lang: str,
    *,
    selected: set[Path] | None = None,
) -> Path | None:
    """Return ``src_path`` to keep, or ``None`` to skip.

    If ``selected`` is provided, files with a language suffix are kept only
    when they appear in the selection set (see :func:`_select_language_files`).
    If ``selected`` is ``None``, falls back to the Round-4 strict behaviour
    (exact-match or skip) — kept for tests that exercise the helper directly.
    """
    if src_path.suffix not in _LANG_SUFFIXES:
        return src_path
    if selected is not None:
        return src_path if src_path in selected else None
    file_lang = src_path.suffix.lstrip(".")
    return src_path if file_lang == lang else None


def _strip_template_suffixes(rel: Path, lang: str) -> Path:
    """Strip language suffix (if any) + ``.jinja`` suffix from the OUTPUT path.

    ``README.md.jinja.de`` (with ``lang='de'``) → ``README.md``.
    ``main.py.jinja`` → ``main.py``. ``plain.txt`` → ``plain.txt``.
    Only the trailing filename is rewritten; directory components are left alone.
    """
    parts = list(rel.parts)
    if not parts:
        return rel
    last = parts[-1]
    # Strip trailing language suffix first (.de/.en/.zh/.ar).
    for lang_suf in _LANG_SUFFIXES:
        if last.endswith(lang_suf):
            last = last[: -len(lang_suf)]
            break
    # Then strip .jinja if still present.
    if last.endswith(".jinja"):
        last = last[: -len(".jinja")]
    parts[-1] = last
    return Path(*parts)


def render_tree(src_dir: Path, dest_dir: Path, *, context: dict[str, Any]) -> None:
    """Render every file under src_dir into dest_dir, applying Jinja2 to .jinja files.

    Files ending in `.jinja` have that suffix stripped and their contents rendered.
    Path segments with `{{...}}` tags are also rendered — and validated through
    :func:`_safe_join` so a malicious template cannot escape ``dest_dir``.
    Non-.jinja files are copied verbatim.

    Language-variant handling (R4-C1): files ending in ``.de``/``.en``/``.zh``/
    ``.ar`` as their outermost suffix are filtered by ``context['lang']`` — the
    matching variant is kept, other variants are skipped. The output filename
    has both the language suffix and the ``.jinja`` suffix stripped (so
    ``README.md.jinja.de`` with ``lang='de'`` renders to ``README.md``).
    """
    src_dir = Path(src_dir)
    dest_dir = Path(dest_dir)
    if dest_dir.exists() and any(dest_dir.iterdir()):
        raise FileExistsError(f"dest exists and is not empty: {dest_dir}")

    env = _build_env(src_dir)
    lang = context.get("lang", "en")

    # Pre-scan: for every base filename with language variants, pick ONE
    # variant (requested lang → en fallback → first available). This closes
    # the Round-5 regression where ``lang='zh'`` with only .de/.en variants
    # produced a scaffold with no README at all.
    selected_variants = _select_language_files(src_dir, lang)

    for src_path in src_dir.rglob("*"):
        rel = src_path.relative_to(src_dir)

        # Filter out wrong-language variants BEFORE rendering anything.
        if src_path.is_file() and _resolve_language_variant(
            src_path, lang, selected=selected_variants
        ) is None:
            continue

        # Compute the output relative path by stripping template suffixes,
        # then render each remaining path segment through the sandbox and
        # validate against traversal.
        if src_path.is_file():
            out_rel = _strip_template_suffixes(rel, lang)
        else:
            out_rel = rel

        rendered_parts = [env.from_string(p).render(**context) for p in out_rel.parts]
        dest_path = _safe_join(dest_dir, rendered_parts)

        if src_path.is_dir():
            dest_path.mkdir(parents=True, exist_ok=True)
            continue

        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Determine whether contents need Jinja rendering. We inspect the
        # ORIGINAL filename: a file is a "template" if it contains ``.jinja``
        # anywhere in its suffix chain (e.g. ``README.md.jinja.de``,
        # ``main.py.jinja``). Pure ``.de``/``.en`` files without ``.jinja`` are
        # language-selected but copied verbatim.
        is_template = ".jinja" in src_path.name.split(".")[1:]
        if is_template:
            template = env.get_template(str(rel).replace("\\", "/"))
            dest_path.write_text(template.render(**context), encoding="utf-8")
        else:
            shutil.copy2(src_path, dest_path)
```

**Security regression tests — add to `tests/test_crew/test_cli/test_scaffolder.py`:**

```python
def test_scaffolder_blocks_path_traversal_in_filename(tmp_path):
    """A template filename containing ``{{ '../../etc/passwd' }}`` MUST raise.

    Regression for NC3: plain ``Environment`` rendered path segments with no
    validation, letting malicious templates write outside ``dest_dir``.
    The sandboxed environment plus ``_safe_join`` blocks this.
    """
    import pytest
    from cognithor.crew.cli.scaffolder import render_tree

    src = tmp_path / "tmpl"
    (src / "subdir").mkdir(parents=True)
    # A file whose NAME expands to "../../etc/passwd" at render time.
    traversal = src / "subdir" / "{{ payload }}.jinja"
    traversal.write_text("pwned", encoding="utf-8")

    dest = tmp_path / "out"
    with pytest.raises(ValueError, match="traversal|Forbidden"):
        render_tree(src, dest, context={"payload": "../../etc/passwd"})


def test_scaffolder_blocks_backslash_traversal_on_windows(tmp_path):
    """Backslash-based traversal payloads are also rejected."""
    import pytest
    from cognithor.crew.cli.scaffolder import render_tree

    src = tmp_path / "tmpl"
    src.mkdir()
    traversal = src / "{{ payload }}.jinja"
    traversal.write_text("pwned", encoding="utf-8")

    dest = tmp_path / "out"
    with pytest.raises(ValueError, match="traversal|Forbidden"):
        render_tree(src, dest, context={"payload": r"..\..\secrets"})


def test_sanitize_project_name_rejects_CON_on_all_platforms():
    """Windows reserved device names are rejected even when running on Linux.

    Scaffolded projects must be portable to Windows developers — naming a
    package ``con`` would make it unbuildable there. Regression for NI4.
    """
    import pytest
    from cognithor.crew.cli.scaffolder import sanitize_project_name

    for reserved in ("CON", "con", "nul", "COM1", "lpt9", "prn", "aux"):
        with pytest.raises(ValueError, match="reserved Windows device name"):
            sanitize_project_name(reserved)


def test_scaffolder_renders_language_specific_readme_de(tmp_path):
    """R4-C1: with ``lang='de'`` the scaffolder must render ``README.md.jinja.de``
    to ``README.md`` and NOT emit the ``.en`` variant.

    Before the R4-C1 fix, ``suffix == '.jinja'`` never matched for files whose
    real suffix is ``.de``/``.en``, so both variants landed verbatim in the
    scaffold and no ``README.md`` existed.
    """
    from cognithor.crew.cli.scaffolder import render_tree

    src = tmp_path / "tmpl"
    src.mkdir()
    (src / "README.md.jinja.de").write_text("# {{ project_name }} (DE)")
    (src / "README.md.jinja.en").write_text("# {{ project_name }} (EN)")
    dest = tmp_path / "out"

    render_tree(src, dest, context={"project_name": "demo", "lang": "de"})

    assert (dest / "README.md").read_text() == "# demo (DE)"
    assert not (dest / "README.md.jinja.en").exists()
    assert not (dest / "README.md.jinja.de").exists()


def test_scaffolder_renders_language_specific_readme_en(tmp_path):
    """Same contract as the DE test but with ``lang='en'``."""
    from cognithor.crew.cli.scaffolder import render_tree

    src = tmp_path / "tmpl"
    src.mkdir()
    (src / "README.md.jinja.de").write_text("# {{ project_name }} (DE)")
    (src / "README.md.jinja.en").write_text("# {{ project_name }} (EN)")
    dest = tmp_path / "out"

    render_tree(src, dest, context={"project_name": "demo", "lang": "en"})

    assert (dest / "README.md").read_text() == "# demo (EN)"
    assert not (dest / "README.md.jinja.en").exists()
    assert not (dest / "README.md.jinja.de").exists()


def test_scaffolder_falls_back_to_en_when_requested_lang_missing(tmp_path):
    """R5 regression: ``lang='zh'`` with only .de/.en variants must still
    produce a README.md by falling back to the EN variant — NOT drop the
    file entirely, which is what the Round-4 strict filter did.
    """
    from cognithor.crew.cli.scaffolder import render_tree

    src = tmp_path / "tmpl"
    src.mkdir()
    (src / "README.md.jinja.de").write_text("# {{ project_name }} (DE)")
    (src / "README.md.jinja.en").write_text("# {{ project_name }} (EN)")
    dest = tmp_path / "out"

    render_tree(src, dest, context={"project_name": "demo", "lang": "zh"})

    # Fallback to EN — never leave the scaffold without a README.
    assert (dest / "README.md").read_text() == "# demo (EN)"
    assert not (dest / "README.md.jinja.en").exists()
    assert not (dest / "README.md.jinja.de").exists()


def test_scaffolder_falls_back_to_first_sorted_when_no_en_variant(tmp_path):
    """Edge case: no requested lang AND no 'en' variant — pick the first
    alphabetically available variant deterministically (here: .de)."""
    from cognithor.crew.cli.scaffolder import render_tree

    src = tmp_path / "tmpl"
    src.mkdir()
    (src / "README.md.jinja.de").write_text("# {{ project_name }} (DE)")
    (src / "README.md.jinja.zh").write_text("# {{ project_name }} (ZH)")
    dest = tmp_path / "out"

    render_tree(src, dest, context={"project_name": "demo", "lang": "ar"})

    # .de sorts before .zh alphabetically.
    assert (dest / "README.md").read_text() == "# demo (DE)"
```

`src/cognithor/crew/cli/__init__.py` — keep empty for now.

- [ ] **Step 3: Commit**

```bash
python -m pytest tests/test_crew/test_cli/test_scaffolder.py -v
git add src/cognithor/crew/cli tests/test_crew/test_cli
git commit -m "feat(crew): scaffolder — sanitize + render Jinja2 template tree"
```

---

### Task 35: Template metadata discovery + `--list-templates`

**Files:**
- Create: `src/cognithor/crew/templates/__init__.py`
- Create: `src/cognithor/crew/cli/list_templates_cmd.py`
- Create: `tests/test_crew/test_cli/test_list_templates.py`

- [ ] **Step 1: Test**

```python
# tests/test_crew/test_cli/test_list_templates.py
from unittest.mock import patch, MagicMock
from pathlib import Path
import pytest
from cognithor.crew.cli.list_templates_cmd import list_templates, TemplateMeta


def test_discovers_template_from_template_yaml(tmp_path: Path):
    t_dir = tmp_path / "research"
    t_dir.mkdir()
    (t_dir / "template.yaml").write_text(
        "name: research\n"
        "description_de: Zwei-Agenten-Research-Crew\n"
        "description_en: Two-agent research crew\n"
        "required_models: ['ollama/qwen3:8b']\n"
        "tags: [demo, quickstart]\n"
    )
    with patch("cognithor.crew.cli.list_templates_cmd.TEMPLATES_ROOT", tmp_path):
        templates = list_templates()

    assert len(templates) == 1
    t = templates[0]
    assert isinstance(t, TemplateMeta)
    assert t.name == "research"
    assert t.description_de.startswith("Zwei")


def test_skips_dirs_without_template_yaml(tmp_path: Path):
    (tmp_path / "broken").mkdir()  # no template.yaml
    with patch("cognithor.crew.cli.list_templates_cmd.TEMPLATES_ROOT", tmp_path):
        templates = list_templates()
    assert templates == []
```

- [ ] **Step 2: Implement**

```python
# src/cognithor/crew/cli/list_templates_cmd.py
"""cognithor init --list-templates: discover + print template metadata."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


TEMPLATES_ROOT = Path(__file__).resolve().parent.parent / "templates"


class TemplateMeta(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    description_de: str
    description_en: str = ""
    required_models: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    # Explicit listing order for `cognithor init --list-templates`.
    # Without this, the list sorts alphabetically and the beginner-friendly
    # `research` template ends up BELOW `content` / `customer-support`. Lower
    # order numbers surface first; ties break on name (alphabetical).
    # Defaults to a large sentinel so legacy templates without `order` fall
    # to the bottom rather than competing with ordered ones.
    order: int = 999


def list_templates() -> list[TemplateMeta]:
    """Return metadata for every discoverable template, sorted by (order, name)."""
    if not TEMPLATES_ROOT.exists():
        return []
    out: list[TemplateMeta] = []
    for d in sorted(TEMPLATES_ROOT.iterdir()):
        meta_file = d / "template.yaml"
        if not meta_file.is_file():
            continue
        data = yaml.safe_load(meta_file.read_text(encoding="utf-8"))
        out.append(TemplateMeta(**data))
    # Explicit UX order: `order` ascending, ties break on name.
    out.sort(key=lambda t: (t.order, t.name))
    return out


def print_templates(*, lang: str = "de") -> int:
    """CLI handler — prints templates + descriptions. Returns exit code."""
    templates = list_templates()
    if not templates:
        print("Keine Templates gefunden." if lang == "de" else "No templates found.")
        return 1
    header = "Verfügbare Templates:" if lang == "de" else "Available templates:"
    print(header)
    for t in templates:
        desc = t.description_de if lang == "de" else (t.description_en or t.description_de)
        print(f"  - {t.name:25} {desc}")
    return 0
```

Create `src/cognithor/crew/templates/__init__.py` as empty placeholder.

- [ ] **Step 3: Commit**

```bash
python -m pytest tests/test_crew/test_cli/test_list_templates.py -v
git add src/cognithor/crew/templates src/cognithor/crew/cli/list_templates_cmd.py tests/test_crew/test_cli/test_list_templates.py
git commit -m "feat(crew): template metadata discovery + --list-templates"
```

---

### Task 36: `init_cmd` — core CLI handler (template selection + render)

**Files:**
- Create: `src/cognithor/crew/cli/init_cmd.py`
- Create: `tests/test_crew/test_cli/test_init.py`

- [ ] **Step 1: Test**

```python
# tests/test_crew/test_cli/test_init.py
from unittest.mock import patch
from pathlib import Path
import pytest
from cognithor.crew.cli.init_cmd import run_init, InitCommandError


@pytest.fixture
def mock_templates(tmp_path: Path, monkeypatch):
    """Plant a minimal mock template so the CLI has something to render.

    Mirrors the real first-party template layout (R4-C2): ``main.py.jinja``
    lives under ``src/{{ project_name }}/`` so the rendered
    ``src/<pkg>/main.py`` resolves the ``pyproject.toml`` script entry.
    """
    tpl_root = tmp_path / "templates"
    research = tpl_root / "research"
    research.mkdir(parents=True)
    (research / "template.yaml").write_text(
        "name: research\n"
        "description_de: Mock\n"
        "description_en: Mock\n"
    )
    (research / "README.md.jinja").write_text("# {{ project_name }}")
    src_dir = research / "src" / "{{ project_name }}"
    src_dir.mkdir(parents=True)
    (src_dir / "__init__.py").write_text("")
    (src_dir / "main.py.jinja").write_text("PROJECT = '{{ project_name }}'")

    monkeypatch.setattr("cognithor.crew.cli.list_templates_cmd.TEMPLATES_ROOT", tpl_root)
    monkeypatch.setattr("cognithor.crew.cli.init_cmd.TEMPLATES_ROOT", tpl_root)
    return tpl_root


def test_creates_project_from_template(tmp_path: Path, mock_templates):
    project_dir = tmp_path / "my_project"
    rc = run_init(
        name="My Project", template="research",
        directory=project_dir, lang="en",
    )
    assert rc == 0
    assert (project_dir / "README.md").read_text() == "# my_project"
    # R4-C2: main.py lives inside the package, not top-level.
    assert (project_dir / "src" / "my_project" / "main.py").read_text() == "PROJECT = 'my_project'"
    assert (project_dir / "src" / "my_project" / "__init__.py").exists()


def test_refuses_nonempty_directory(tmp_path: Path, mock_templates):
    project_dir = tmp_path / "existing"
    project_dir.mkdir()
    (project_dir / "file.txt").write_text("hello")
    with pytest.raises(InitCommandError):
        run_init(name="existing", template="research", directory=project_dir, lang="en")


def test_unknown_template_raises(tmp_path: Path, mock_templates):
    with pytest.raises(InitCommandError, match="unknown"):
        run_init(name="x", template="does_not_exist", directory=tmp_path / "x", lang="en")


def test_init_force_overwrites_existing_dir(tmp_path: Path, mock_templates, capsys):
    """R4-I5: `--force` removes an existing non-empty target before scaffolding."""
    project_dir = tmp_path / "my_project"
    project_dir.mkdir()
    (project_dir / "stale.txt").write_text("pre-existing junk")

    rc = run_init(
        name="My Project", template="research",
        directory=project_dir, lang="en", force=True,
    )
    assert rc == 0
    # Stale file is gone after the forced rebuild.
    assert not (project_dir / "stale.txt").exists()
    # Scaffolder output present.
    assert (project_dir / "README.md").read_text() == "# my_project"
    # Warning printed so the user sees what --force did.
    captured = capsys.readouterr()
    assert "--force" in captured.out
    assert "removing existing" in captured.out
```

- [ ] **Step 2: Implement**

```python
# src/cognithor/crew/cli/init_cmd.py
"""cognithor init <project_name> --template <template> — create a new Crew project."""

from __future__ import annotations

from pathlib import Path

from cognithor.crew.cli.list_templates_cmd import TEMPLATES_ROOT, list_templates
from cognithor.crew.cli.scaffolder import render_tree, sanitize_project_name


class InitCommandError(Exception):
    """Raised when the init subcommand cannot complete."""


def run_init(
    *,
    name: str,
    template: str,
    directory: Path | None = None,
    lang: str | None = None,
    force: bool = False,
) -> int:
    """Execute `cognithor init`. Returns shell exit code (0 on success).

    `lang` — if set (via `--lang=de|en`), overrides the i18n language for the
    duration of this command. When None, falls back to the global
    `config.language` (default "de").

    `force` — if True (via `--force`), overwrite an existing non-empty target
    directory by removing it first. Off by default; the scaffolder normally
    refuses to write into a non-empty directory. See R4-I5.
    """
    # Respect explicit --lang; otherwise keep the global config language.
    if lang is not None:
        try:
            # R4-C3: real public API is `set_locale`, not `set_language`
            # (see src/cognithor/i18n/__init__.py __all__).
            from cognithor.i18n import set_locale
            set_locale(lang)
        except ImportError:
            # i18n module unavailable in this build — proceed with English-only
            # error strings (acceptable degradation for a standalone test env).
            pass
    else:
        from cognithor.config import load_config
        lang = load_config().language

    project_name = sanitize_project_name(name)

    template_dir = TEMPLATES_ROOT / template
    if not template_dir.is_dir():
        known = ", ".join(t.name for t in list_templates()) or "none"
        raise InitCommandError(
            f"unknown template '{template}'. Known templates: {known}"
        )

    dest = directory if directory is not None else Path.cwd() / project_name
    dest = Path(dest)
    if dest.exists() and any(dest.iterdir()):
        if force:
            # R4-I5: --force overwrites by removing the existing tree.
            # Print a prominent warning so the user sees what just happened
            # (stdout, because init_cmd's normal output is informational).
            import shutil as _shutil
            print(f"WARNING: --force: removing existing non-empty directory {dest}")
            _shutil.rmtree(dest)
        else:
            raise InitCommandError(
                f"target directory is not empty: {dest} (pass --force to overwrite)"
            )

    context = {
        "project_name": project_name,
        "project_name_display": name,
        "lang": lang,
    }
    render_tree(template_dir, dest, context=context)

    # Success output: header + folder preview (top 2 levels, capped ~15 lines)
    # + next-command hint. Previously printed only "Projekt erstellt: <path>",
    # which left the user guessing what to do next. See NI8 in Round 3 review.
    msg_done = "Projekt erstellt" if lang == "de" else "Project created"
    print(f"{msg_done}: {dest}")
    print()
    for line in _render_folder_tree(dest, max_depth=2, max_lines=15):
        print(f"  {line}")
    print()
    try:
        from cognithor.i18n import t
        next_cmd = t("crew.init.next_command", dest=dest.name)
    except Exception:
        next_cmd = (
            f"Next: cd {dest.name} && pip install -e .[dev] && cognithor run"
            if lang != "de"
            else f"Nächste Schritte: cd {dest.name} && pip install -e .[dev] && cognithor run"
        )
    print(next_cmd)
    return 0


def _render_folder_tree(root: Path, *, max_depth: int = 2, max_lines: int = 15) -> list[str]:
    """Return up to ``max_lines`` string representations of the scaffolded tree.

    Breadth-first walk, capped at ``max_depth`` levels deep, sorted for
    deterministic output. Prefixes children with ``└── `` for readability.
    """
    lines: list[str] = [root.name + "/"]
    def walk(d: Path, depth: int, prefix: str) -> None:
        if depth > max_depth or len(lines) >= max_lines:
            return
        try:
            children = sorted(d.iterdir(), key=lambda p: (p.is_file(), p.name))
        except OSError:
            return
        for i, child in enumerate(children):
            if len(lines) >= max_lines:
                lines.append(prefix + "…")
                return
            is_last = i == len(children) - 1
            branch = "└── " if is_last else "├── "
            suffix = "/" if child.is_dir() else ""
            lines.append(prefix + branch + child.name + suffix)
            if child.is_dir():
                next_prefix = prefix + ("    " if is_last else "│   ")
                walk(child, depth + 1, next_prefix)
    walk(root, 1, "")
    return lines[:max_lines]
```

**Locale keys to add in `src/cognithor/i18n/locales/en.json` + `de.json`:**

```jsonc
// en.json
"crew.init.next_command": "Next: cd {dest} && pip install -e .[dev] && cognithor run"

// de.json
"crew.init.next_command": "Nächste Schritte: cd {dest} && pip install -e .[dev] && cognithor run"
```

Update `test_creates_project_from_template` to assert the tree preview + next-command line appear in captured stdout (use `capsys` fixture).

- [ ] **Step 3: Commit**

```bash
python -m pytest tests/test_crew/test_cli/test_init.py -v
git add src/cognithor/crew/cli/init_cmd.py tests/test_crew/test_cli/test_init.py
git commit -m "feat(crew): cognithor init — template-based project scaffolder"
```

---

### Task 37: Wire `init` + `run` into main Cognithor CLI

**Files:**
- Modify: `src/cognithor/__main__.py`
- Create: `src/cognithor/crew/cli/run_cmd.py`
- Create: `tests/test_crew/test_cli/test_cli_integration.py`

- [ ] **Step 1: Scout existing `__main__.py` for the argparse / click layout**

```bash
head -80 src/cognithor/__main__.py
```

Identify whether it uses argparse, click, or typer. The `cognithor init` and `cognithor run` subcommands must slot into the SAME CLI framework the codebase uses.

- [ ] **Step 2: Test (integration)**

```python
# tests/test_crew/test_cli/test_cli_integration.py
import subprocess
import sys
from pathlib import Path


def test_cognithor_init_from_cli(tmp_path: Path):
    """End-to-end: invoking `python -m cognithor init ... --template research`
    scaffolds a project from a real first-party template.

    This test is marked slow — only runs after Task 39 lands the research template.
    """
    import pytest
    pytest.skip("Runs after Task 39 lands the research template.")
```

- [ ] **Step 3: Add subcommand dispatch (pattern depends on existing CLI)**

Spec §3.2 requires the invocation shape `cognithor init --list-templates` (flag ON the `init` subcommand, not a separate subcommand). The `name` and `--template` args must be optional when `--list-templates` is set, and the list path short-circuits BEFORE validating the other args.

If existing `__main__` uses argparse subparsers:

```python
import argparse

def _validate_lang(value: str) -> str:
    """argparse type hook: accept any locale that i18n has a pack for.

    Originally the flag hardcoded ``choices=["de", "en"]`` — which silently
    broke ``--lang zh`` even though the i18n module ships a ``zh`` locale pack.
    See NI5 in Round 3 review. The available locales today are ``{en, de, zh}``
    (enumerate via ``get_available_locales()``); ``ar`` is NOT shipped, so
    passing ``--lang ar`` rightfully fails.

    R4-C3: uses the real ``get_available_locales`` API, not the fictitious
    ``available_languages``. See ``src/cognithor/i18n/__init__.py`` ``__all__``.
    """
    try:
        from cognithor.i18n import get_available_locales
        available = set(get_available_locales())  # today: {"en", "de", "zh"}
    except ImportError:
        # Minimal install without i18n — fall back to the hardcoded EN/DE pair.
        available = {"en", "de"}
    if value not in available:
        raise argparse.ArgumentTypeError(
            f"Unsupported language {value!r}. Available: {sorted(available)}"
        )
    return value


# Inside the existing argparse setup
init_parser = subparsers.add_parser("init", help="Scaffold a new Crew project")
init_parser.add_argument("name", nargs="?", help="Project name (required unless --list-templates)")
init_parser.add_argument("--template", help="Template name (required unless --list-templates)")
init_parser.add_argument("--dir", dest="directory", type=Path, default=None)
init_parser.add_argument(
    "--lang",
    type=_validate_lang,
    default=None,
    help="UI language (default: config.language). Accepts any i18n locale present in src/cognithor/i18n/locales/ (today: en, de, zh).",
)
init_parser.add_argument(
    "--list-templates",
    action="store_true",
    help="List available templates and exit",
)
init_parser.add_argument(
    "--force",
    action="store_true",
    help="Overwrite an existing non-empty target directory (removes it first). Off by default.",
)

# In the dispatch:
if args.command == "init":
    from cognithor.crew.cli.init_cmd import run_init
    from cognithor.crew.cli.list_templates_cmd import print_templates

    # Short-circuit: --list-templates runs before any other arg validation
    if args.list_templates:
        return print_templates(lang=args.lang)

    # Validate required args for the scaffold path
    if not args.name or not args.template:
        init_parser.error(
            "`init <name> --template <tmpl>` is required unless `--list-templates` is set"
        )

    try:
        return run_init(
            name=args.name,
            template=args.template,
            directory=args.directory,
            lang=args.lang,
            force=args.force,
        )
    except Exception as exc:
        print(f"init failed: {exc}", file=sys.stderr)
        return 1
```

Adapt to actual existing style. If `__main__` is click-based, use a `@click.option("--list-templates", is_flag=True)` on the `init` command that short-circuits similarly.

- [ ] **Step 4: Create `run_cmd.py` (used INSIDE scaffolded projects, not on the main CLI yet)**

```python
# src/cognithor/crew/cli/run_cmd.py
"""Scaffolded-project-internal 'cognithor run' — loads the Crew defined in
the generated project and calls kickoff()."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


def run_project_crew(project_dir: Path | None = None) -> int:
    project_dir = Path(project_dir) if project_dir else Path.cwd()
    src = project_dir / "src"
    if not src.is_dir():
        print("No src/ directory — not a scaffolded project?", file=sys.stderr)
        return 2
    # Find the single package dir under src/
    pkg_dirs = [p for p in src.iterdir() if p.is_dir() and (p / "__init__.py").exists()]
    if not pkg_dirs:
        print("No Python package under src/", file=sys.stderr)
        return 2
    pkg_name = pkg_dirs[0].name

    sys.path.insert(0, str(src))
    try:
        mod = importlib.import_module(f"{pkg_name}.main")
    except ModuleNotFoundError as exc:
        print(f"cannot import {pkg_name}.main: {exc}", file=sys.stderr)
        return 2

    # The scaffold convention: main.py exposes a `build_crew()` function
    if not hasattr(mod, "build_crew"):
        print(f"{pkg_name}.main does not define build_crew()", file=sys.stderr)
        return 2

    import asyncio
    crew = mod.build_crew()
    result = asyncio.run(crew.kickoff_async())
    print(result.raw)
    return 0
```

- [ ] **Step 5: Commit**

```bash
git add src/cognithor/__main__.py src/cognithor/crew/cli/run_cmd.py tests/test_crew/test_cli/test_cli_integration.py
git commit -m "feat(crew): wire cognithor init + run into main CLI"
```

---

### Task 38: Template package resources — ensure `templates/*` ships in wheel

**Files:**
- Modify: `pyproject.toml` (add `[tool.hatch.build.targets.wheel.shared-data]` or `include`)
- Modify: `MANIFEST.in` (if used)

Hatch includes `src/cognithor/**/*.py` by default but NOT `.jinja` / `.yaml` files. Without explicit inclusion, `cognithor init` fails on a fresh pip install because the template files aren't packaged.

- [ ] **Step 1: Test**

```python
# tests/test_crew/test_cli/test_package_resources.py
from pathlib import Path
import cognithor.crew.templates as _t


def test_templates_package_has_files():
    pkg_dir = Path(_t.__file__).parent
    # After install there must be at least one template/template.yaml
    yamls = list(pkg_dir.glob("*/template.yaml"))
    assert yamls, f"No template.yaml files shipped in package at {pkg_dir}"
```

- [ ] **Step 2: Modify `pyproject.toml`**

Add under `[tool.hatch.build.targets.wheel]`:

```toml
[tool.hatch.build.targets.wheel]
packages = ["src/cognithor"]

[tool.hatch.build.targets.wheel.force-include]
"src/cognithor/crew/templates" = "cognithor/crew/templates"
```

- [ ] **Step 3: Verify fresh install ships templates**

```bash
pip install -e . --force-reinstall
python -c "from cognithor.crew.cli.list_templates_cmd import list_templates; print([t.name for t in list_templates()])"
```

The list will be empty until Task 39 lands templates — but the packaging config must be in place first.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml tests/test_crew/test_cli/test_package_resources.py
git commit -m "build: include crew templates in wheel distribution"
```

---

### Task 39: `research` template (simplest — 2 agents, sequential)

**Files:**
- Create: `src/cognithor/crew/templates/research/template.yaml`
- Create: `src/cognithor/crew/templates/research/README.md.jinja.de`
- Create: `src/cognithor/crew/templates/research/README.md.jinja.en`
- Create: `src/cognithor/crew/templates/research/pyproject.toml.jinja`
- Create: `src/cognithor/crew/templates/research/.env.example`
- Create: `src/cognithor/crew/templates/research/src/{{ project_name }}/__init__.py`
- Create: `src/cognithor/crew/templates/research/src/{{ project_name }}/main.py.jinja` (R4-C2: lives under `src/` so `pyproject.toml`'s `[project.scripts]` entry `{project_name}.main:main` resolves and `cognithor run` can `importlib.import_module(f"{pkg}.main")`)
- Create: `src/cognithor/crew/templates/research/src/{{ project_name }}/crew.py.jinja`
- Create: `src/cognithor/crew/templates/research/config/agents.yaml.jinja`
- Create: `src/cognithor/crew/templates/research/config/tasks.yaml.jinja`
- Create: `src/cognithor/crew/templates/research/tests/test_crew.py.jinja`
- Create: `tests/test_crew/test_templates/test_research.py`

- [ ] **Step 1: `template.yaml`**

```yaml
name: research
description_de: Researcher + Reporter Zwei-Agenten-Crew mit sequenziellem Ablauf.
description_en: Two-agent researcher + reporter crew, sequential process.
order: 1  # beginner-friendly — surface first
required_models:
  - ollama/qwen3:8b
tags:
  - quickstart
  - beginner
  - sequential
```

- [ ] **Step 2: `src/{{ project_name }}/main.py.jinja`**

Lives at `src/{{ project_name }}/main.py.jinja` (NOT top-level). The
`pyproject.toml.jinja` script entry `{{ project_name }} = "{{ project_name }}.main:main"`
and `cognithor run`'s `importlib.import_module(f"{pkg}.main")` both resolve
against the installed package (`src/{{ project_name }}/main.py`); placing it at
the top-level would break both. See R4-C2.

```python
"""{{ project_name_display }} — entry point."""

from __future__ import annotations

import asyncio

from {{ project_name }}.crew import ResearchCrew


def build_crew():
    """Return a Crew instance. Used by `cognithor run` and main()."""
    return ResearchCrew().assemble()


def main() -> None:
    crew = build_crew()
    result = asyncio.run(crew.kickoff_async(inputs={"topic": "Beispielthema"}))
    print(result.raw)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: `src/{{ project_name }}/crew.py.jinja`**

```python
"""Crew definition for {{ project_name_display }}."""

from __future__ import annotations

from cognithor.crew import Crew, CrewAgent, CrewProcess, CrewTask
from cognithor.crew.decorators import agent, crew, task


class ResearchCrew:
    @agent
    def researcher(self) -> CrewAgent:
        return CrewAgent(
            role="Researcher",
            goal="Recherchiere das Thema '{topic}' und sammle Fakten",
            backstory="Erfahrener Research-Spezialist mit Fokus auf verlässliche Quellen.",
            tools=[],  # add MCP tool names here
            llm="ollama/qwen3:8b",
            memory=True,
        )

    @agent
    def reporter(self) -> CrewAgent:
        return CrewAgent(
            role="Reporter",
            goal="Schreibe einen strukturierten Report",
            backstory="Spezialist für kompakte, gut lesbare Zusammenfassungen.",
            llm="ollama/qwen3:8b",
        )

    @task
    def research(self) -> CrewTask:
        return CrewTask(
            description="Recherchiere: {topic}",
            expected_output="Bulletpoint-Liste der 5 wichtigsten Fakten.",
            agent=self.researcher(),
        )

    @task
    def report(self) -> CrewTask:
        return CrewTask(
            description="Schreibe basierend auf der Research einen Report.",
            expected_output="Markdown-Report, 300-500 Wörter.",
            agent=self.reporter(),
            context=[self.research()],
        )

    @crew
    def assemble(self) -> Crew:
        return Crew(
            agents=[self.researcher(), self.reporter()],
            tasks=[self.research(), self.report()],
            process=CrewProcess.SEQUENTIAL,
            verbose=True,
        )
```

- [ ] **Step 4: `tests/test_crew.py.jinja`**

```python
"""Smoke test for the {{ project_name_display }} Crew scaffold."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from {{ project_name }}.crew import ResearchCrew


async def test_crew_kickoff_with_mock_planner(monkeypatch):
    from cognithor.core.observer import ResponseEnvelope
    mock_planner = MagicMock()
    mock_planner.formulate_response = AsyncMock(
        return_value=ResponseEnvelope(content="MOCK_OUTPUT", directive=None),
    )
    mock_registry = MagicMock()
    mock_registry.get_tools_for_role.return_value = []

    monkeypatch.setattr("cognithor.crew.runtime.get_default_tool_registry", lambda: mock_registry)

    # The scaffolded ResearchCrew assembles with injected planner for testability
    crew = ResearchCrew(planner=mock_planner).assemble()
    result = await crew.kickoff_async(inputs={"topic": "test"})
    assert result.raw == "MOCK_OUTPUT"
    assert len(result.tasks_output) == 2
```

- [ ] **Step 5: `pyproject.toml.jinja`**

```toml
[project]
name = "{{ project_name }}"
version = "0.1.0"
description = "{{ project_name_display }} — scaffolded from the Cognithor research template"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "cognithor[all]>=0.93.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
]

[project.scripts]
{{ project_name }} = "{{ project_name }}.main:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/{{ project_name }}"]
```

- [ ] **Step 6: `.env.example`**

```
# Optional overrides for {{ project_name_display }}
COGNITHOR_OLLAMA_BASE_URL=http://localhost:11434
```

- [ ] **Step 7: `README.md.jinja.de`**

```markdown
# {{ project_name_display }}

Gescaffoldet aus dem Cognithor `research`-Template.

## Setup

```bash
cd {{ project_name }}
pip install -e ".[dev]"
```

## Run

```bash
cognithor run                    # nutzt build_crew() aus src/{{ project_name }}/main.py
# oder:
python -m {{ project_name }}.main
```

## Struktur

- `src/{{ project_name }}/crew.py` — Agents + Tasks + Crew-Assembly
- `config/agents.yaml` — alternative YAML-Definition der Agents
- `config/tasks.yaml` — alternative YAML-Definition der Tasks
- `tests/test_crew.py` — Smoke-Test mit Mock-Planner
```

- [ ] **Step 8: English version (README.md.jinja.en)** — analogous translation.

- [ ] **Step 9: `config/agents.yaml.jinja` + `config/tasks.yaml.jinja`** — mirror the `ResearchCrew` class in YAML form, referenced from the README for users who prefer config-driven crews.

- [ ] **Step 10: Integration test**

```python
# tests/test_crew/test_templates/test_research.py
from pathlib import Path
from cognithor.crew.cli.init_cmd import run_init


def test_research_template_renders_and_smoke_tests_pass(tmp_path: Path):
    project = tmp_path / "rc"
    rc = run_init(name="rc", template="research", directory=project, lang="de")
    assert rc == 0

    # Required artifacts exist — 8 user-editable files from the template tree
    assert (project / "pyproject.toml").exists()
    assert (project / "src" / "rc" / "crew.py").exists()
    assert (project / "src" / "rc" / "main.py").exists()
    assert (project / "tests" / "test_crew.py").exists()
    assert (project / "README.md").exists()
    assert (project / ".env.example").exists()

    # Plus the auto-injected .gitignore (scaffolder writes this regardless of
    # which template was picked; not part of the template tree itself).
    assert (project / ".gitignore").exists(), \
        "Scaffolder must auto-inject .gitignore to prevent accidental secret commits"
```

- [ ] **Step 11: Commit**

```bash
mkdir -p tests/test_crew/test_templates
touch tests/test_crew/test_templates/__init__.py
python -m pytest tests/test_crew/test_templates/test_research.py -v
git add src/cognithor/crew/templates/research tests/test_crew/test_templates
git commit -m "feat(crew): research template (researcher + reporter, sequential)"
```

---

### Task 40: `customer-support` template (3 agents, sequential, memory)

**Files:**
- Create: `src/cognithor/crew/templates/customer-support/*` (same layout as research)
- Create: `tests/test_crew/test_templates/test_customer_support.py`

Same structure as Task 39. Three agents: `intake`, `classifier`, `response_writer`. Task 2 (classifier) uses memory=True to access "prior customer interactions"-mocked tool. See spec §3.3.2.

- [ ] **Step 1-10: Mirror Task 39 structure, replacing content with the three-agent customer-support crew.** Save 200 lines here by reference — the IMPLEMENTER follows the pattern from Task 39 exactly, adjusting:
  - `template.yaml`: name: customer-support, 3 agents, sequential, **`order: 2`**
  - `crew.py.jinja`: IntakeCrew class with 3 @agent + 3 @task methods
  - Agents: `intake` (parses customer message), `classifier` (categorizes), `response_writer` (drafts reply)
  - Tasks: `parse`, `classify`, `draft_reply` — each feeds context to the next
  - `tests/test_crew.py.jinja`: kickoff with mock planner returning 3 mock responses

  **Ship all required template artifacts** (spec §3.4): scaffolding renders 8 user-editable files — `template.yaml` (metadata, not rendered), `README.md.jinja.de` + `.en`, `pyproject.toml.jinja`, `.env.example`, `src/{{ project_name }}/main.py.jinja` (see R4-C2 — must live inside the package so `pyproject.toml`'s `[project.scripts]` entry resolves), `src/{{ project_name }}/__init__.py`, `src/{{ project_name }}/crew.py.jinja`, `config/agents.yaml.jinja`, `config/tasks.yaml.jinja`, `tests/test_crew.py.jinja` — **plus 1 `.gitignore`** that every scaffolded project gets (auto-injected by the scaffolder, not part of the template tree). Test assertions count the 8 user-editable files; `.gitignore` is verified separately in Task 39.

- [ ] **Step 11: Integration test — three agents + all 8 user-editable files + .gitignore**

```python
from pathlib import Path
from cognithor.crew.cli.init_cmd import run_init


def test_customer_support_template_renders(tmp_path: Path):
    project = tmp_path / "cs"
    run_init(name="cs", template="customer-support", directory=project, lang="de")
    assert (project / "src" / "cs" / "crew.py").exists()
    # The scaffolded crew has three agents
    import ast
    tree = ast.parse((project / "src" / "cs" / "crew.py").read_text())
    agent_methods = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and any(
            isinstance(d, ast.Name) and d.id == "agent" for d in n.decorator_list
        )
    ]
    assert len(agent_methods) == 3


def test_customer_support_ships_all_required_files(tmp_path: Path):
    """Spec §3.4: every template ships 8 user-editable file groups + 1 .gitignore."""
    project = tmp_path / "cs"
    run_init(name="cs", template="customer-support", directory=project, lang="de")
    expected = [
        project / "pyproject.toml",
        project / ".env.example",
        project / "src" / "cs" / "main.py",  # R4-C2: main.py lives inside the package
        project / "src" / "cs" / "__init__.py",
        project / "src" / "cs" / "crew.py",
        project / "config" / "agents.yaml",
        project / "config" / "tasks.yaml",
        project / "tests" / "test_crew.py",
        project / "README.md",
    ]
    for f in expected:
        assert f.exists(), f"Missing: {f.relative_to(project)}"
```

- [ ] **Step 12: Commit**

```bash
git add src/cognithor/crew/templates/customer-support tests/test_crew/test_templates/test_customer_support.py
git commit -m "feat(crew): customer-support template (3-agent, sequential)"
```

---

### Task 41: `data-analyst` template (code-interpreter + viz)

**Files:**
- Create: `src/cognithor/crew/templates/data-analyst/*`
- Create: `tests/test_crew/test_templates/test_data_analyst.py`

Spec §3.3.3: Code-Interpreter-Agent (with `allow_code_execution=True` in sandboxed mode) + Visualization-Agent. Uses the existing sandbox module.

- [ ] **Step 1-10: Mirror Task 39/40 pattern, shipping 8 user-editable files + 1 auto-injected `.gitignore` per spec §3.4**
  - `template.yaml`: add **`order: 3`**
  - `analyst`: role="Analyst", runs data-summarization via code-exec tool
  - `visualizer`: role="Visualizer", produces matplotlib chart spec
  - Tasks: `analyze` (consumes CSV path from inputs), `visualize` (consumes analyst output)
  - **Critical:** the `analyst` agent's `tools` list includes the existing sandbox code-exec tool (e.g. `python_sandbox`). Scaffolded tests mock the registry.

- [ ] **Step 11: Integration test — assert all 8 user-editable files + .gitignore + analyst has code-exec tool**

```python
from pathlib import Path
from cognithor.crew.cli.init_cmd import run_init


def test_data_analyst_ships_all_required_files(tmp_path: Path):
    """Spec §3.4: every template ships 8 user-editable file groups + 1 .gitignore."""
    project = tmp_path / "da"
    run_init(name="da", template="data-analyst", directory=project, lang="de")
    expected = [
        project / "pyproject.toml",
        project / ".env.example",
        project / "src" / "da" / "main.py",  # R4-C2: main.py lives inside the package
        project / "src" / "da" / "__init__.py",
        project / "src" / "da" / "crew.py",
        project / "config" / "agents.yaml",
        project / "config" / "tasks.yaml",
        project / "tests" / "test_crew.py",
        project / "README.md",
    ]
    for f in expected:
        assert f.exists(), f"Missing: {f.relative_to(project)}"
```

- [ ] **Step 12: Commit**

```bash
git add src/cognithor/crew/templates/data-analyst tests/test_crew/test_templates/test_data_analyst.py
git commit -m "feat(crew): data-analyst template (code-interpreter + viz)"
```

---

### Task 42: `content` template (hierarchical with manager_llm)

**Files:**
- Create: `src/cognithor/crew/templates/content/*`
- Create: `tests/test_crew/test_templates/test_content.py`

Spec §3.3.4: Outline-Agent + Draft-Agent + Editor, hierarchical with `manager_llm="ollama/qwen3:32b"`.

- [ ] **Step 1-10: Mirror pattern, shipping 8 user-editable files + 1 auto-injected `.gitignore` per spec §3.4**
  - `template.yaml`: add **`order: 4`**
  - `Crew(process=CrewProcess.HIERARCHICAL, manager_llm="ollama/qwen3:32b", ...)`
  - Three agents: `outliner`, `drafter`, `editor`
  - Tasks: `outline`, `draft`, `edit` — hierarchical process chooses order dynamically
  - Smoke test verifies `crew.process == HIERARCHICAL` and `manager_llm` is set

- [ ] **Step 11: Integration test — all 8 user-editable files + .gitignore + hierarchical config**

```python
from pathlib import Path
from cognithor.crew.cli.init_cmd import run_init


def test_content_template_is_hierarchical_and_complete(tmp_path: Path):
    project = tmp_path / "content"
    run_init(name="content", template="content", directory=project, lang="de")
    expected = [
        project / "pyproject.toml",
        project / ".env.example",
        project / "src" / "content" / "main.py",  # R4-C2: main.py lives inside the package
        project / "src" / "content" / "__init__.py",
        project / "src" / "content" / "crew.py",
        project / "config" / "agents.yaml",
        project / "config" / "tasks.yaml",
        project / "tests" / "test_crew.py",
        project / "README.md",
    ]
    for f in expected:
        assert f.exists(), f"Missing: {f.relative_to(project)}"
    crew_src = (project / "src" / "content" / "crew.py").read_text()
    assert "HIERARCHICAL" in crew_src
    assert "manager_llm" in crew_src
```

- [ ] **Step 12: Commit**

```bash
git add src/cognithor/crew/templates/content tests/test_crew/test_templates/test_content.py
git commit -m "feat(crew): content template (3-agent, hierarchical)"
```

---

### Task 43: `versicherungs-vergleich` template (DACH-differentiator)

**Files:**
- Create: `src/cognithor/crew/templates/versicherungs-vergleich/*` (8 user-editable files + 1 .gitignore per spec §3.4, see below)
- Create: `tests/test_crew/test_templates/test_versicherungs_vergleich.py`

Spec §3.3.5: PKV/BU-Tarif-Vergleich. THREE agents: `Tarif-Researcher`, `Kunden-Profiler`, `Empfehlungs-Writer`. DSGVO-konform, **vollständig offline-fähig** (no external APIs). Includes explicit §34d-neutral guardrails.

**Per spec §3.4 every template ships exactly 8 user-editable files + 1 auto-injected `.gitignore`** (same contract as `research` in Task 39):
1. `template.yaml`
2. `README.md.jinja.de` and `README.md.jinja.en`
3. `pyproject.toml.jinja`
4. `.env.example`
5. `src/{{ project_name }}/main.py.jinja` (R4-C2: lives inside the package so `pyproject.toml` `[project.scripts]` entry resolves)
6. `src/{{ project_name }}/__init__.py`
7. `src/{{ project_name }}/crew.py.jinja`
8. `config/agents.yaml.jinja` and `config/tasks.yaml.jinja`
9. `tests/test_crew.py.jinja`

- [ ] **Step 1-10: Mirror pattern, with extra care for the spec's DSGVO requirements**
  - `template.yaml`: add **`order: 5`** (DACH-differentiator surfaces last in the quickstart listing since it targets a specialized domain)
  - `tools=[]` for all agents — NO external HTTP tools in default (spec §3.6)
  - Writer task has guardrail: **`chain(no_pii(), StringGuardrail(...))`** — spec §4.5 AC 5 requires BOTH guardrails. The rule text: `"Output darf keine Tarif-Empfehlung enthalten, nur neutralen Vergleich"`.
  - `required_models` in `template.yaml`: only Ollama — the template REFUSES to run against cloud models

**Crew.py.jinja snippet (shows both guardrails wired):**

```python
# In the scaffolded versicherungs-vergleich crew.py:
from cognithor.crew import Crew, CrewAgent, CrewTask
from cognithor.crew.guardrails import StringGuardrail, chain, no_pii

def build_crew(ollama_client):
    writer = CrewAgent(role="Empfehlungs-Writer", goal="...", tools=[], llm="ollama/qwen3:8b")
    # ... other agents ...

    neutral_rule = StringGuardrail(
        "Output darf keine Tarif-Empfehlung enthalten, nur neutralen Vergleich. "
        "§34d-konform: Information, nicht Beratung.",
        llm_client=ollama_client,
        model="ollama/qwen3:8b",
    )

    write_task = CrewTask(
        description="...",
        expected_output="...",
        agent=writer,
        guardrail=chain(no_pii(), neutral_rule),
        max_retries=2,
    )
    return Crew(agents=[...], tasks=[..., write_task])
```

- [ ] **Step 11: Integration tests — assert BOTH guardrails + all 8 user-editable files + .gitignore**

```python
from pathlib import Path
from cognithor.crew.cli.init_cmd import run_init


def test_versicherungs_template_is_offline_capable(tmp_path: Path):
    project = tmp_path / "pkv"
    run_init(name="pkv", template="versicherungs-vergleich", directory=project, lang="de")
    crew_file = (project / "src" / "pkv" / "crew.py").read_text()
    # No tools should be listed (offline-capable)
    assert 'tools=[]' in crew_file
    # Both guardrails wired (spec §4.5 AC 5)
    assert "no_pii" in crew_file
    assert "StringGuardrail" in crew_file or "string_guardrail" in crew_file
    assert "chain(" in crew_file


def test_versicherungs_template_ships_all_required_files(tmp_path: Path):
    """Spec §3.4: every template ships exactly 8 user-editable file groups + 1 .gitignore."""
    project = tmp_path / "pkv"
    run_init(name="pkv", template="versicherungs-vergleich", directory=project, lang="de")
    expected = [
        project / "pyproject.toml",
        project / ".env.example",
        project / "src" / "pkv" / "main.py",  # R4-C2: main.py lives inside the package
        project / "src" / "pkv" / "__init__.py",
        project / "src" / "pkv" / "crew.py",
        project / "config" / "agents.yaml",
        project / "config" / "tasks.yaml",
        project / "tests" / "test_crew.py",
        project / "README.md",
    ]
    for f in expected:
        assert f.exists(), f"Missing: {f.relative_to(project)}"
```

- [ ] **Step 12: Commit**

```bash
git add src/cognithor/crew/templates/versicherungs-vergleich tests/test_crew/test_templates/test_versicherungs_vergleich.py
git commit -m "feat(crew): versicherungs-vergleich template (DACH, offline-capable, no_pii + StringGuardrail)"
```

---

### Task 44: `cognithor init --list-templates` CLI integration test

**Files:**
- Modify: `tests/test_crew/test_cli/test_list_templates.py` (add real-template test)

- [ ] **Step 1: After Tasks 39-43, all 5 templates exist. Run:**

```bash
python -m cognithor init --list-templates
```

Expected output (ordered by each template's `order` field, set in Tasks 39-43 to 1-5):

```
Verfügbare Templates:
  - research                    Researcher + Reporter Zwei-Agenten-Crew ...
  - customer-support            ...
  - data-analyst                ...
  - content                     ...
  - versicherungs-vergleich     PKV/BU-Tarif-Vergleich ...
```

- [ ] **Step 2: Add assertion**

```python
def test_list_templates_cli_lists_all_five():
    from cognithor.crew.cli.list_templates_cmd import list_templates
    names = {t.name for t in list_templates()}
    assert names == {"research", "customer-support", "data-analyst", "content", "versicherungs-vergleich"}


def test_list_templates_respects_order_field():
    """Templates sort by `order` ascending, ties break on name. Tasks 39-43
    assign 1-5 in this specific sequence."""
    from cognithor.crew.cli.list_templates_cmd import list_templates
    names = [t.name for t in list_templates()]
    assert names == [
        "research",
        "customer-support",
        "data-analyst",
        "content",
        "versicherungs-vergleich",
    ]


def test_list_templates_via_cli_subprocess():
    """Full CLI invocation — must match spec §3.2 flag syntax."""
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "-m", "cognithor", "init", "--list-templates"],
        capture_output=True, text=True, check=True,
    )
    # Order must be preserved in the subprocess output (ordered listing).
    expected = ["research", "customer-support", "data-analyst", "content", "versicherungs-vergleich"]
    positions = [result.stdout.find(n) for n in expected]
    assert all(p >= 0 for p in positions), f"Template missing from CLI output: {result.stdout}"
    assert positions == sorted(positions), f"Templates listed out of order: {result.stdout}"
```

- [ ] **Step 3: Commit**

```bash
git add tests/test_crew/test_cli/test_list_templates.py
git commit -m "test(crew): CLI lists all 5 first-party templates"
```

---

### Task 45: CI workflow — scaffold every template in CI and run pytest inside

**Files:**
- Create: `.github/workflows/scaffold-templates.yml`

- [ ] **Step 1: Workflow**

```yaml
name: Scaffold Templates

on:
  push:
    branches: [main, "feat/**"]
    paths:
      - "src/cognithor/crew/templates/**"
      - "src/cognithor/crew/cli/**"
      - ".github/workflows/scaffold-templates.yml"
  pull_request:
    paths:
      - "src/cognithor/crew/templates/**"
      - "src/cognithor/crew/cli/**"
      - ".github/workflows/scaffold-templates.yml"

jobs:
  scaffold:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        template: [research, customer-support, data-analyst, content, versicherungs-vergleich]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install cognithor (editable)
        run: pip install -e ".[dev,mcp]"
      - name: Scaffold + smoke-test ${{ matrix.template }}
        run: |
          mkdir -p /tmp/scaffold_test
          cd /tmp/scaffold_test
          python -m cognithor init test_${{ matrix.template }} --template ${{ matrix.template }} --lang de
          cd test_${{ matrix.template }}
          pip install -e ".[dev]"
          pip install pytest pytest-asyncio
          python -m pytest tests/ -v
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/scaffold-templates.yml
git commit -m "ci: scaffold every template + run its smoke tests"
```

- [ ] **Step 3: Perf budget test (Spec §8.5, R4-I7)**

Spec §8.5 requires each template to scaffold in under 500ms. Add a parameterized
perf test to the test matrix so CI enforces the budget on every PR:

```python
# tests/test_crew/test_templates/test_scaffold_perf.py
"""Spec §8.5 / R4-I7: each template must scaffold in <500ms.

Run-time perf budget, NOT a pytest-benchmark statistical measurement.
Uses time.perf_counter() so the assertion is fast + deterministic enough
for CI. Templates themselves are static filesystem trees; the only variable
cost is Jinja rendering, which is dominated by the number of template files
(all five templates ship with the same 8 user-editable files).
"""

from pathlib import Path
import time

import pytest

from cognithor.crew.cli.init_cmd import run_init


_ALL_TEMPLATES = [
    "research",
    "customer-support",
    "data-analyst",
    "content",
    "versicherungs-vergleich",
]


@pytest.mark.parametrize("template_name", _ALL_TEMPLATES)
def test_template_generation_under_500ms(template_name: str, tmp_path: Path) -> None:
    """Spec §8.5: each template must scaffold in <500ms."""
    project = tmp_path / f"perf_{template_name.replace('-', '_')}"
    start = time.perf_counter()
    run_init(name=project.name, template=template_name, directory=project, lang="de")
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert elapsed_ms < 500, (
        f"{template_name} scaffolding took {elapsed_ms:.0f}ms (budget: 500ms). "
        f"Spec §8.5 budget violated — investigate render_tree or template bloat."
    )
```

Commit separately so the perf gate landing is visible in `git log`:

```bash
git add tests/test_crew/test_templates/test_scaffold_perf.py
git commit -m "test(crew): spec §8.5 perf gate — each template scaffolds in <500ms"
```

---

### Task 46: Feature-3 merge-prep — CHANGELOG + CLI help text

**Files:**
- Modify: `CHANGELOG.md`
- Modify: `src/cognithor/crew/cli/init_cmd.py` (ensure `--help` is bilingual)

- [ ] **Step 1: CHANGELOG append under `[Unreleased]` `### Added`:**

```markdown
- **`cognithor init` scaffolder + 5 first-party templates (Feature 3)** —
  `cognithor init <name> --template <name> [--dir PATH] [--lang de|en]`
  generates a runnable Crew project from Jinja2 templates. Templates: `research`,
  `customer-support`, `data-analyst`, `content`, `versicherungs-vergleich`
  (DACH-differentiator, fully offline-capable, §34d-neutral guardrails).
  `cognithor init --list-templates` prints the catalog with DE/EN descriptions.
  CI scaffolds every template on every PR.
```

`### Breaking Changes` stays `None.` — Feature 3 adds a new subcommand only.

- [ ] **Step 2: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs(crew): Feature-3 CHANGELOG entry"
```

---

### Task 47-52: Polish, release-prep tasks

- **Task 47:** Run full `tests/test_crew/` suite, fix any flakes, enforce ≥ 89% coverage on `cognithor.crew` module. Add missing edge-case tests. Commit as `test(crew): coverage polish pass`.
- **Task 48:** Ruff sweep across all new files. Commit as `style(crew): ruff + format sweep`.
- **Task 49:** Mypy --strict sweep across `src/cognithor/crew`. Fix any new errors. Commit as `type(crew): mypy --strict clean`.
- **Task 50:** Performance benchmark — CI-enforced per spec §8.5 (< 5% overhead vs direct Planner call).

  **Files:**
  - Create: `tests/test_crew/test_performance.py` — `@pytest.mark.benchmark`-decorated test
  - Modify: `.github/workflows/ci.yml` (or equivalent) — add a `pytest -m benchmark` step
  - (Optional) Create: `scripts/bench_crew_overhead.py` — one-shot local benchmark that reuses the same harness

  ```python
  # tests/test_crew/test_performance.py
  import asyncio
  import time
  from unittest.mock import AsyncMock, MagicMock
  import pytest
  from cognithor.crew import Crew, CrewAgent, CrewTask
  from cognithor.core.observer import ResponseEnvelope


  BUDGET_PERCENT = 5.0  # Spec §8.5 — Crew-Layer overhead must stay under 5%


  @pytest.mark.benchmark
  async def test_crew_kickoff_overhead_under_5_percent():
      """Measure Crew.kickoff_async() overhead vs a direct Planner.formulate_response()
      call with identical payload. Both should take ~the same time because the Crew
      compiler is a thin translation layer — spec §8.5 allows up to 5% overhead."""
      # Fixed-latency fake Planner so the measurement is deterministic
      async def fake_formulate(user_message, results, working_memory):
          await asyncio.sleep(0.020)  # 20 ms — simulated LLM
          return ResponseEnvelope(content="x", directive=None)

      mock_planner = MagicMock()
      mock_planner.formulate_response = AsyncMock(side_effect=fake_formulate)
      mock_registry = MagicMock()
      mock_registry.get_tools_for_role.return_value = []

      agent = CrewAgent(role="x", goal="y")
      task = CrewTask(description="z", expected_output="w", agent=agent)
      crew = Crew(agents=[agent], tasks=[task], planner=mock_planner)

      N = 50

      # Baseline: direct planner calls
      t0 = time.perf_counter()
      for _ in range(N):
          await mock_planner.formulate_response("z", [], None)
      baseline_ms = (time.perf_counter() - t0) * 1000 / N

      # Crew path
      with pytest.MonkeyPatch().context() as mp:
          mp.setattr("cognithor.crew.runtime.get_default_tool_registry", lambda: mock_registry)
          t0 = time.perf_counter()
          for _ in range(N):
              await crew.kickoff_async()
          crew_ms = (time.perf_counter() - t0) * 1000 / N

      overhead_percent = (crew_ms - baseline_ms) / baseline_ms * 100.0
      print(f"baseline={baseline_ms:.3f}ms crew={crew_ms:.3f}ms overhead={overhead_percent:.2f}%")
      assert overhead_percent < BUDGET_PERCENT, (
          f"Crew-Layer overhead {overhead_percent:.2f}% exceeds spec §8.5 budget "
          f"of {BUDGET_PERCENT}%"
      )
  ```

  Register the `benchmark` marker in `pyproject.toml`:
  ```toml
  [tool.pytest.ini_options]
  markers = ["benchmark: performance budget tests (spec §8.5)"]
  ```

  CI step:
  ```yaml
  - name: Performance benchmark (Crew-Layer overhead <5%)
    run: python -m pytest tests/test_crew/test_performance.py -m benchmark -v
  ```
- **Task 51:** `docs/superpowers/specs/...` — update spec status to "implemented" at the top.
- **Task 52:** Create README.md Highlights bullet for Crew-Layer + link to Feature-2 quickstart.

Each task: one commit with descriptive message.

---

# FEATURE 2 — Quickstart-Dokumentation (Tasks 53-66)

Implements spec §2. Seven documentation pages (each DE + EN), matching runnable examples under `examples/quickstart/`, plus a CI job that exercises every example against mock-Ollama.

---

### Task 53: `docs/quickstart/` scaffold + index

**Files:**
- Create: `docs/quickstart/README.md` (bilingual index)
- Create: `docs/quickstart/README.en.md`

- [ ] **Step 1: Index `README.md`**

```markdown
# Cognithor Quickstart (DE)

Von leerem Terminal zur ersten Crew in unter 10 Minuten.

| Schritt | Datei | Zeit |
|--------:|-------|------|
| 00 | [Installation](00-installation.md) | 3 min |
| 01 | [Erste Crew](01-first-crew.md) | 5 min |
| 02 | [Eigenes Tool](02-first-tool.md) | 5 min |
| 03 | [Erster Skill](03-first-skill.md) | 5 min |
| 04 | [Guardrails](04-guardrails.md) | 5 min |
| 05 | [Deployment](05-deployment.md) | 5 min |
| 06 | [Nächste Schritte](06-next-steps.md) | 2 min |

English: see [README.en.md](README.en.md).
```

`README.en.md` is a direct English translation.

- [ ] **Step 2: Commit**

```bash
mkdir -p docs/quickstart
git add docs/quickstart/README.md docs/quickstart/README.en.md
git commit -m "docs(quickstart): scaffold index (DE + EN)"
```

---

### Task 54: `00-installation.md` — 3 install paths

**Files:**
- Create: `docs/quickstart/00-installation.md`
- Create: `docs/quickstart/00-installation.en.md`

- [ ] **Step 1: Contents (DE version)**

```markdown
# 00 · Installation

> **Voraussetzungen:** Python 3.12+, Ollama 0.4.0+, cognithor>=0.93.0 (`pip install --upgrade cognithor`)

**Voraussetzung:** Python 3.12+, internet für die Erstinstallation, optional Docker.

## Option A — Windows One-Click-Installer

1. Lade den aktuellen `.exe`-Installer von https://github.com/Alex8791-cyber/cognithor/releases.
2. Starte `CognithorSetup-0.93.0.exe` (Administrator-Rechte nicht nötig).
3. Folge dem Wizard — Ollama + Python-Embedded werden mitinstalliert.
4. Nach Abschluss: `Cognithor.exe` auf dem Desktop doppelklicken.

**Verifikation:**

```powershell
cognithor --version
```

Erwartete Ausgabe: `Cognithor · Agent OS v0.93.0`

## Option B — `pip install` (Linux, macOS, Windows)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install cognithor[all]
```

**Verifikation:**

```bash
cognithor --version
curl http://localhost:8741/health  # nach `cognithor --no-cli &`
```

## Option C — Docker Compose

```bash
git clone https://github.com/Alex8791-cyber/cognithor.git
cd cognithor
docker compose up -d
```

**Verifikation:**

```bash
docker compose ps
curl http://localhost:8741/health
```

## Next

[01 · Erste Crew](01-first-crew.md)
```

- [ ] **Step 2: EN version** — direct translation. Must include the same prerequisites block at the top:

```markdown
# 00 · Installation

> **Prerequisites:** Python 3.12+, Ollama 0.4.0+, cognithor>=0.93.0 (`pip install --upgrade cognithor`)

...rest of the EN translation...
```

- [ ] **Step 3: Commit**

```bash
git add docs/quickstart/00-installation.md docs/quickstart/00-installation.en.md
git commit -m "docs(quickstart): installation page (DE + EN)"
```

- [ ] **Step 4: Recruit 2-3 external testers NOW for the Task 62 review slot**

Per NI11 in Round 3 review, external-reader recruitment starts in Week 4, not Week 6. Waiting until Task 62 (Week 6) has historically created release-day scrambles. Identify 2-3 developers who are (a) Python-literate, (b) comfortable installing Ollama, (c) new to Cognithor Crew-Layer, and (d) NOT the plan author. Secure a calendar slot for the review run targeting the week PR 4b opens. If no reader is available, escalate to the plan lead before the quickstart docs complete — a release-blocking gate you discover on release-day morning is a known anti-pattern.

Track candidates in `docs/quickstart/EXTERNAL_REVIEW_RESULTS.md` under a `## Recruited testers` section (names, contact, scheduled slot).

---

### Task 55: `01-first-crew.md` — PKV example walkthrough

**Files:**
- Create: `docs/quickstart/01-first-crew.md`
- Create: `docs/quickstart/01-first-crew.en.md`
- Create: `examples/quickstart/01_first_crew/main.py`
- Create: `examples/quickstart/01_first_crew/requirements.txt`
- Create: `examples/quickstart/01_first_crew/README.md`
- Create: `examples/quickstart/01_first_crew/test_example.py`

- [ ] **Step 1: `01-first-crew.md`**

Contents: Walk through the PKV example from spec §1.4 step-by-step. Points to the runnable file at `examples/quickstart/01_first_crew/main.py`. Screenshots of the output.

- [ ] **Step 2: `examples/quickstart/01_first_crew/main.py`**

Exact spec §1.4 code, with added `if __name__ == "__main__"` guard.

- [ ] **Step 3: `test_example.py` for CI**

```python
"""Smoke test — CI runs this to guarantee the quickstart example never breaks."""
from unittest.mock import AsyncMock, MagicMock
import pytest


async def test_pkv_example_runs_with_mock_planner(monkeypatch):
    # Import the example AS IF a user just installed cognithor
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from main import build_crew  # the example exports build_crew()

    from cognithor.core.observer import ResponseEnvelope
    mock_planner = MagicMock()
    mock_planner.formulate_response = AsyncMock(
        return_value=ResponseEnvelope(content="MOCK", directive=None),
    )
    mock_registry = MagicMock()
    mock_registry.get_tools_for_role.return_value = []

    monkeypatch.setattr("cognithor.crew.runtime.get_default_planner", lambda: mock_planner)
    monkeypatch.setattr("cognithor.crew.runtime.get_default_tool_registry", lambda: mock_registry)

    crew = build_crew()
    result = await crew.kickoff_async()
    assert result.raw
```

- [ ] **Step 4: Commit**

```bash
mkdir -p examples/quickstart/01_first_crew
git add docs/quickstart/01-first-crew.md docs/quickstart/01-first-crew.en.md examples/quickstart/01_first_crew
git commit -m "docs(quickstart): first-crew walkthrough + runnable example"
```

---

### Task 56: `02-first-tool.md` — register an `@tool` and use it in a Crew

**Files:**
- Create: `docs/quickstart/02-first-tool.md` + `.en.md`
- Create: `examples/quickstart/02_first_tool/*`

Pattern: existing `@tool` decorator from `src/cognithor/sdk/decorators.py`, wire into a Crew via the agent's `tools=[]` list.

- [ ] **Step 1: Commit page + example + test**

```bash
git add docs/quickstart/02-first-tool.md docs/quickstart/02-first-tool.en.md examples/quickstart/02_first_tool
git commit -m "docs(quickstart): first-tool walkthrough + example"
```

---

### Task 57: `03-first-skill.md` — Tool vs Skill distinction, using existing scaffolder

**Files:**
- Create: `docs/quickstart/03-first-skill.md` + `.en.md`
- Create: `examples/quickstart/03_first_skill/*`

- [ ] **Step 1: Commit page + example + test.**

---

### Task 58: `04-guardrails.md` — guardrail types + retry pattern

**Files:**
- Create: `docs/quickstart/04-guardrails.md` + `.en.md`
- Create: `examples/quickstart/04_guardrails/*`

Contents: Feature 4 overview, `word_count` + `no_pii` + `chain()` examples, retry behaviour, `GuardrailFailure` handling.

- [ ] **Step 1: Commit**

---

### Task 59: `05-deployment.md` — local / docker / systemd / --no-cli

**Files:**
- Create: `docs/quickstart/05-deployment.md` + `.en.md`

No code example file — this is pure operations guidance.

- [ ] **Step 1: Commit**

---

### Task 60: `06-next-steps.md` — cross-links to advanced docs

**Files:**
- Create: `docs/quickstart/06-next-steps.md` + `.en.md`

Links to Memory docs, Voice docs, Computer Use, MCP tool catalog. "After the quickstart" orientation.

- [ ] **Step 1: Commit**

---

### Task 60b: `07-troubleshooting.md` — common Quickstart stumbles

**Rationale:** Real first-time users hit well-known snags (Ollama not running, missing model pulls, port collisions, template-name typos) that derail the 10-minute onboarding promise. Shipping a bilingual FAQ page alongside the happy-path pages is a cheap way to keep the external-reader gate in Task 81 achievable. See NI9 in Round 3 review.

**Files:**
- Create: `docs/quickstart/07-troubleshooting.md`
- Create: `docs/quickstart/07-troubleshooting.en.md`

- [ ] **Step 1: Write DE content**

```markdown
# 07 · Fehler & Probleme

> Die häufigsten Stolperfallen beim Durchlauf der Cognithor-Quickstart.

## "Ollama is not running"

**Symptom:** `cognithor run` bricht sofort mit `ConnectionError: Ollama unreachable at 127.0.0.1:11434` ab.

**Lösung:**
- Linux/macOS: `ollama serve` im Hintergrund laufen lassen.
- Windows: Ollama-Dienst via Startmenü starten (es läuft als Hintergrund-Service).
- Prüfe: `curl http://127.0.0.1:11434/api/tags` liefert JSON.

## "Model pull failed" / "model qwen3:8b not found"

**Symptom:** Crew-Kickoff beklagt fehlendes Modell.

**Lösung:**
- `ollama pull qwen3:8b` (~5 GB, dauert je nach Bandbreite).
- Plattenplatz prüfen — Windows-Ollama legt Modelle standardmäßig in `%USERPROFILE%\.ollama\models` ab (kann >20 GB mit mehreren Modellen werden).

## Port 11434 / 8741 belegt

**Symptom:** `OSError: [Errno 48] Address already in use`.

**Lösung:**
- `netstat -ano | findstr :8741` (Windows) oder `lsof -i :8741` (Linux) zeigt den Nutzer.
- Alternative Ports: `COGNITHOR_API_PORT=8742 cognithor --no-cli`.

## `GuardrailFailure` beim Task-Output

**Symptom:** Crew liefert keine Antwort, stattdessen `GuardrailFailure: 'no_pii' rejected output after 3 attempt(s)`.

**Lösung:** Siehe [`docs/guardrails.md`](../guardrails.md) — meist hängt der Guardrail am Task-Output-Schema oder am Kontext fest.

## "Unknown tool 'xxx'"

**Symptom:** `ToolNotFoundError: Tool 'search_web' not registered`.

**Lösung:** `cognithor tools list` zeigt alle registrierten Tools. Fehlendes Tool kommt entweder aus einem fehlenden Pack oder aus nicht importiertem `@tool`-Modul.

## Template-Namen Kollision

**Symptom:** `FileExistsError: target directory is not empty: ./my_project`.

**Lösung:** `cognithor init my_project --template research --force` oder anderen Projekt-Namen wählen.
```

- [ ] **Step 2: Write EN content**

Same structure in English — not machine-translated, but written by hand so idioms + tone match the rest of the quickstart. Copy 1:1 structurally.

- [ ] **Step 3: Commit**

```bash
git add docs/quickstart/07-troubleshooting.md docs/quickstart/07-troubleshooting.en.md
git commit -m "docs(quickstart): 07 troubleshooting + FAQ (DE + EN)"
```

- [ ] **Step 4: Update `docs/quickstart/README.md` + `README.en.md` index**

Add the 8th row:

```markdown
| 07 | [Fehler & Probleme](07-troubleshooting.md) | 3 min |
```

(EN: `| 07 | [Troubleshooting](07-troubleshooting.en.md) | 3 min |`)

- [ ] **Step 5: Update the CHANGELOG note in Task 66**

The Feature 2 CHANGELOG bullet currently says "7 bilingual pages" — bump to "8 bilingual pages" when Task 66 runs.

---

### Task 61: CI workflow `quickstart-examples.yml`

**Files:**
- Create: `.github/workflows/quickstart-examples.yml`

- [ ] **Step 1: Workflow**

```yaml
name: Quickstart Examples

on:
  push:
    branches: [main, "feat/**"]
    paths:
      - "examples/quickstart/**"
      - "docs/quickstart/**"
      - "src/cognithor/crew/**"
  pull_request:
    paths:
      - "examples/quickstart/**"
      - "docs/quickstart/**"
      - "src/cognithor/crew/**"

jobs:
  run-examples:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        example:
          - 01_first_crew
          - 02_first_tool
          - 03_first_skill
          - 04_guardrails
          - 05_pkv_report
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install cognithor
        run: pip install -e ".[dev,mcp]"
      - name: Run example smoke test
        run: |
          cd examples/quickstart/${{ matrix.example }}
          pip install pytest pytest-asyncio
          python -m pytest test_example.py -v
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/quickstart-examples.yml
git commit -m "ci: exercise every quickstart example against mock Ollama"
```

---

### Task 62: External-reader usability test — checklist + actual external run

**Files:**
- Create: `docs/quickstart/EXTERNAL_REVIEW_CHECKLIST.md`
- Create: `docs/quickstart/EXTERNAL_REVIEW_RESULTS.md` (filled by the external reader)

Spec §2.4 AND §12 AC 4 require that an **external testreader** (not the author) completes the checklist successfully — this is a real usability gate, not just a template. The PR opening step (Task 81) is blocked until `EXTERNAL_REVIEW_RESULTS.md` exists with a passing entry.

- [ ] **Step 1: Create the checklist template**

```markdown
# External-Reader Usability Checklist

**Testreader requirements:**
- Must NOT be the plan author
- Must NOT have prior Cognithor Crew-Layer exposure
- Starts from a fresh clone + no prior `~/.cognithor/` state

## Timed milestones (start timer at `README.md`):
- [ ] Reach `docs/quickstart/00-installation.md` landing page
- [ ] Successfully install (`pip install cognithor==0.93.0.dev0` or `-e .`)
- [ ] Scaffold the first template (`cognithor init my_first_crew --template research`)
- [ ] Run the scaffolded crew successfully (`cognithor run`)
- [ ] First `crew.kickoff()` returns a CrewOutput

**Budget:** all five milestones complete within 30 minutes total, ≤1 small clarifying question back to author.

## Record
- Total elapsed time: ___ minutes
- Number of questions asked: ___
- Typos / confusions found: (bulleted list)
- Blocking bugs encountered: (bulleted list; empty list required for PASS or PASS_WITH_FOLLOWUPS)
- Follow-up issues filed (issue numbers, required if PASS_WITH_FOLLOWUPS): _______________
- **Verdict: PASS / PASS_WITH_FOLLOWUPS / FAIL**

**Verdict guide (per NI11 in Round 3 review):**
- **PASS:** happy path completed, ≤1 clarifying question, ≤30 min.
- **PASS_WITH_FOLLOWUPS:** happy path completed unassisted, ≥1 minor confusion logged as 0.93.1 issues, no blockers.
- **FAIL:** could not complete happy path unassisted, hit a blocker, or >30 min elapsed. Release blocked until fixed.
```

- [ ] **Step 2: Obtain a real external run**

The project lead identifies a non-author tester (e.g. a colleague, community member, or another developer), has them follow the checklist, and pastes their filled-in record into `EXTERNAL_REVIEW_RESULTS.md`. This is NOT optional — spec §12 AC 4 requires an actual passing entry.

If no external reader is available before the PR 4 opens, the plan-lead must find one or defer the v0.93.0 release until the test completes.

- [ ] **Step 3: Commit**

```bash
git add docs/quickstart/EXTERNAL_REVIEW_CHECKLIST.md docs/quickstart/EXTERNAL_REVIEW_RESULTS.md
git commit -m "docs(quickstart): external-reader usability checklist + results"
```

- [ ] **Step 4 (cross-reference — enforced at Task 81):**

Task 81 (PR 4 open) has an explicit acceptance step: verify `EXTERNAL_REVIEW_RESULTS.md` contains a line `Verdict: PASS` before running `gh pr create`. Without a PASS entry, the PR open is blocked.

---

### Task 63: `examples/quickstart/05_pkv_report/*` — spec §1.4 runnable example

**Files:**
- Create: `examples/quickstart/05_pkv_report/*`

- [ ] **Step 1: Commit**

---

### Task 64: Cross-link Highlights bullet in main README

**Files:**
- Modify: `README.md` (Highlights section)

- [ ] **Step 1: Add bullet after existing Crew-Layer bullet**

```markdown
- **Quickstart** — 8-page onboarding guide at [`docs/quickstart/`](docs/quickstart/README.md) (7 happy-path pages + a troubleshooting FAQ). From empty terminal to first Crew in <10 minutes.
```

- [ ] **Step 2: Commit**

---

### Task 65: `cognithor.ai` site-link update (documentation — actual site deploy is in Feature 7)

**Files:**
- Create: `docs/quickstart/SITE_LINK.md` — note for the site-deploy PR, telling the marketing-site repo to link `cognithor.ai/quickstart` to this doc tree.

- [ ] **Step 1: Commit**

---

### Task 66: Feature-2 CHANGELOG

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Append under `[Unreleased]` `### Added`**

```markdown
- **Quickstart docs (Feature 2)** — 8 bilingual (DE+EN) pages at `docs/quickstart/` (7 happy-path + 1 troubleshooting FAQ)
  covering installation, first-crew, first-tool, first-skill, guardrails,
  deployment, and next-steps. Every example runs in CI via
  `.github/workflows/quickstart-examples.yml`.
```

`### Breaking Changes` stays `None.` — Feature 2 is docs + examples only.

- [ ] **Step 2: Commit**

---

# FEATURE 7 — Integrations Katalog (Tasks 67-78)

Implements spec §7. Auto-generated catalog of MCP tools with Wahrheitspflicht (only list what exists in the repo), one DACH connector confirmed, website section on `cognithor.ai/integrations` (separate repo — this plan provides the JSON + verification script).

---

### Task 67: `generate_integrations_catalog.py` — scan MCP tools

**Files:**
- Create: `scripts/generate_integrations_catalog.py`
- Create: `docs/integrations/README.md`
- Create: `tests/test_integrations_catalog.py`

- [ ] **Step 1: Failing test**

```python
# tests/test_integrations_catalog.py
from pathlib import Path
import json
import subprocess
import sys


def test_generator_produces_valid_catalog(tmp_path: Path):
    out = tmp_path / "catalog.json"
    result = subprocess.run(
        [sys.executable, "scripts/generate_integrations_catalog.py", "--output", str(out)],
        capture_output=True, text=True, check=True,
    )
    assert out.exists()
    data = json.loads(out.read_text())
    assert "tools" in data
    assert isinstance(data["tools"], list)
    # Each entry has the required fields
    for entry in data["tools"]:
        assert "name" in entry
        assert "module" in entry
        assert "category" in entry
        assert "description" in entry


def test_catalog_only_includes_real_tools(tmp_path: Path):
    """Wahrheitspflicht: no entry is listed that doesn't exist in the repo."""
    out = tmp_path / "catalog.json"
    subprocess.run(
        [sys.executable, "scripts/generate_integrations_catalog.py", "--output", str(out)],
        check=True,
    )
    data = json.loads(out.read_text())
    import importlib
    for entry in data["tools"]:
        # Every listed module must import
        importlib.import_module(entry["module"])
```

- [ ] **Step 2: Implement generator**

```python
#!/usr/bin/env python3
"""Scan src/cognithor/ for MCP tool definitions and emit catalog.json.

Tool discovery:
  * Any module under src/cognithor/mcp/ containing a function decorated with
    @mcp_tool (or class registering to the tool_registry).
  * Skill modules that register MCP-compatible tools.

Output JSON shape:
  {
    "generated_at": "<iso8601>",
    "tool_count": N,
    "tools": [
      {"name": "...", "module": "cognithor.mcp.foo", "category": "...",
       "description": "...", "dach_specific": false}, ...
    ]
  }
"""

from __future__ import annotations

import argparse
import ast
import json
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
MCP_DIR = REPO_ROOT / "src" / "cognithor" / "mcp"

DACH_MARKERS = {"datev", "lexware", "sevdesk", "elster", "schufa"}


def extract_tools(py_file: Path) -> list[dict]:
    """Parse a Python file and return any @mcp_tool-decorated function metadata."""
    try:
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
    except SyntaxError:
        return []
    results: list[dict] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        # Scan decorators for mcp_tool / cognithor_tool
        for dec in node.decorator_list:
            dec_name = _decorator_name(dec)
            if dec_name in {"mcp_tool", "cognithor_tool", "tool"}:
                docstring = ast.get_docstring(node) or ""
                module = (
                    py_file.relative_to(REPO_ROOT / "src")
                    .with_suffix("")
                    .as_posix()
                    .replace("/", ".")
                )
                category = _infer_category(py_file, docstring)
                name_lower = node.name.lower()
                dach = any(marker in name_lower or marker in docstring.lower()
                          for marker in DACH_MARKERS)
                results.append({
                    "name": node.name,
                    "module": module,
                    "category": category,
                    "description": docstring.split("\n")[0][:200],
                    "dach_specific": dach,
                })
                break
    return results


def _decorator_name(dec: ast.expr) -> str:
    if isinstance(dec, ast.Name):
        return dec.id
    if isinstance(dec, ast.Call):
        return _decorator_name(dec.func)
    if isinstance(dec, ast.Attribute):
        return dec.attr
    return ""


def _infer_category(py_file: Path, docstring: str) -> str:
    parts = py_file.parts
    # File paths like .../mcp/filesystem/... -> category "filesystem"
    for marker in ("filesystem", "web", "shell", "memory", "vault", "browser",
                   "documents", "kanban", "identity", "reddit"):
        if marker in parts:
            return marker
    low = docstring.lower()
    if "http" in low or "url" in low or "web" in low:
        return "web"
    if "file" in low or "pdf" in low:
        return "filesystem"
    return "misc"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    tools: list[dict] = []
    for py in MCP_DIR.rglob("*.py"):
        tools.extend(extract_tools(py))

    # Deduplicate by (module, name)
    seen: set[tuple[str, str]] = set()
    deduped: list[dict] = []
    for t in tools:
        key = (t["module"], t["name"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(t)

    deduped.sort(key=lambda t: (t["category"], t["name"]))

    catalog = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tool_count": len(deduped),
        "tools": deduped,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(catalog, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"wrote {len(deduped)} tools to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Commit**

```bash
mkdir -p docs/integrations scripts
chmod +x scripts/generate_integrations_catalog.py
git add scripts/generate_integrations_catalog.py docs/integrations/README.md tests/test_integrations_catalog.py
git commit -m "feat(integrations): auto-generate catalog.json from MCP tool scan"
```

---

### Task 68: Generate + commit initial `docs/integrations/catalog.json`

**Files:**
- Create: `docs/integrations/catalog.json`

- [ ] **Step 1: Run the generator**

```bash
python scripts/generate_integrations_catalog.py --output docs/integrations/catalog.json
```

- [ ] **Step 2: Commit**

```bash
git add docs/integrations/catalog.json
git commit -m "feat(integrations): initial generated catalog.json"
```

---

### Task 69: CI workflow `integrations-catalog.yml`

**Files:**
- Create: `.github/workflows/integrations-catalog.yml`

- [ ] **Step 1: Workflow — fails if catalog drifts**

```yaml
name: Integrations Catalog Freshness

on:
  push:
    branches: [main, "feat/**"]
    paths:
      - "src/cognithor/mcp/**"
      - "scripts/generate_integrations_catalog.py"
  pull_request:
    paths:
      - "src/cognithor/mcp/**"
      - "scripts/generate_integrations_catalog.py"

jobs:
  check-drift:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install
        run: pip install -e .
      - name: Regenerate catalog
        run: |
          python scripts/generate_integrations_catalog.py --output /tmp/new_catalog.json
      - name: Diff against committed catalog
        run: |
          # Compare ignoring the generated_at timestamp
          python -c "
          import json
          committed = json.load(open('docs/integrations/catalog.json'))
          fresh = json.load(open('/tmp/new_catalog.json'))
          committed.pop('generated_at', None)
          fresh.pop('generated_at', None)
          if committed != fresh:
              print('::error::Integrations catalog is stale. Run scripts/generate_integrations_catalog.py and commit.')
              exit(1)
          print('Catalog is fresh.')
          "
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/integrations-catalog.yml
git commit -m "ci: fail on integrations-catalog drift"
```

---

### Task 70: DACH connector selection + implementation (sevDesk REST API)

Spec §3.3.5 + §7.2.2 requires at least ONE functional DACH connector. Recommendation: **sevDesk REST API** (German SaaS accounting tool) — smallest surface-area, public REST API, OAuth-free (API key).

**Files:**
- Create: `src/cognithor/mcp/sevdesk/__init__.py`
- Create: `src/cognithor/mcp/sevdesk/client.py`
- Create: `src/cognithor/mcp/sevdesk/tools.py`
- Create: `tests/test_mcp/test_sevdesk.py`

- [ ] **Step 1: Minimal API client + two MCP tools (`sevdesk_list_contacts`, `sevdesk_get_invoice`) with mock-based tests.**

(Detailed code skipped here — standard httpx-based REST client pattern with environment-variable API key. The implementer follows the existing MCP tool style from `src/cognithor/mcp/<other>/` modules.)

- [ ] **Step 2: Commit**

```bash
mkdir -p src/cognithor/mcp/sevdesk tests/test_mcp
git add src/cognithor/mcp/sevdesk tests/test_mcp/test_sevdesk.py
git commit -m "feat(mcp): sevDesk REST connector (DACH accounting)"
```

---

### Task 71: Regenerate catalog with sevDesk present

- [ ] **Step 1: Regenerate + verify sevDesk appears with `dach_specific: true`**

```bash
python scripts/generate_integrations_catalog.py --output docs/integrations/catalog.json
python -c "
import json
data = json.load(open('docs/integrations/catalog.json'))
sevdesk = [t for t in data['tools'] if 'sevdesk' in t['name'].lower()]
assert sevdesk, 'sevDesk tool missing'
assert all(t['dach_specific'] for t in sevdesk), 'sevDesk not marked DACH-specific'
print(f'OK — {len(sevdesk)} sevDesk tools catalogued')
"
```

- [ ] **Step 2: Commit**

```bash
git add docs/integrations/catalog.json
git commit -m "docs(integrations): catalog includes sevDesk DACH connector"
```

---

### Task 72: `docs/integrations/README.md` with category overview

**Files:**
- Modify: `docs/integrations/README.md`

- [ ] **Step 1: Content (DE)**

```markdown
# Integrations

Die Cognithor-Integrationen sind **MCP-Tools** — offenes Protokoll, self-hostable,
kein Vendor-Lock-In. Die Liste unten wird automatisch aus dem Repo generiert —
kein Vapourware.

**Catalog:** [catalog.json](catalog.json)
**Generator:** `scripts/generate_integrations_catalog.py`
**CI-Verifikation:** `.github/workflows/integrations-catalog.yml` (fails bei Drift)

## Kategorien

Siehe `catalog.json` für die vollständige Liste. Hauptkategorien:

- `filesystem` — Datei-Operationen
- `web` — HTTP / Web-Scraping / Search
- `documents` — PDF, DOCX, Excel
- `browser` — Playwright-basierte Browser-Automation
- `memory` — Zugriff auf das 6-Tier Cognitive Memory
- `identity` — Ed25519-Key-Management
- `shell` — Sandboxed Shell-Execution
- `sevdesk` — **DACH:** sevDesk-Buchhaltung (v1.0 Launch)

## MCP-Protokoll

Alle Integrations folgen dem Model Context Protocol. Eigene Integrations bauen:
siehe `docs/quickstart/02-first-tool.md`.
```

- [ ] **Step 2: Commit**

```bash
git add docs/integrations/README.md
git commit -m "docs(integrations): category overview + MCP-protocol link"
```

---

### Task 73: Site-link note for `cognithor.ai` marketing repo

**Files:**
- Create: `docs/integrations/SITE_INTEGRATION_NOTE.md`

Spec §7.2.1 requires `cognithor.ai/integrations`. That page is deployed from the separate `cognithor-site` Vercel repo. This note captures what the site-side PR needs to do:

```markdown
# Note for cognithor-site deployment

After v0.93.0 is released, the site repo needs to add a new page
`/integrations` that:

1. Fetches `docs/integrations/catalog.json` at build time from this repo
   (Octokit fetch at build time, analogous to the existing pack fetch).
2. Renders a grid of integration cards, grouped by category.
3. Highlights the `dach_specific: true` entries in a dedicated DACH section.
4. Links each card to `docs/quickstart/02-first-tool.md` as "build your own".

No additional API keys needed — the catalog.json is a public file in main.
```

- [ ] **Step 1: Commit**

```bash
git add docs/integrations/SITE_INTEGRATION_NOTE.md
git commit -m "docs(integrations): site-integration spec for cognithor-site repo"
```

---

### Task 74-78: Feature-7 polish + CHANGELOG + Highlights

- **Task 74:** Add README Highlights bullet:

```markdown
- **Integrations Catalog** — Auto-generated from `src/cognithor/mcp/` — see [`docs/integrations/catalog.json`](docs/integrations/catalog.json). DACH-specific: sevDesk REST connector (accounting).
```

- **Task 75:** CHANGELOG entry under `[Unreleased]` `### Added`:

```markdown
- **Integrations Catalog (Feature 7)** — `docs/integrations/catalog.json`
  auto-generated from MCP tool definitions by
  `scripts/generate_integrations_catalog.py`. CI fails on drift. Includes
  a new DACH-specific sevDesk REST connector (v1.0 Launch).
```

`### Breaking Changes` stays `None.` — sevDesk is a brand-new additive connector.

- **Task 76:** Full ruff sweep across Feature-7 files.
- **Task 77:** Add `Tool-of-the-month` idea to `docs/integrations/BACKLOG.md` for post-v1.0.
- **Task 78:** Commit final.

---

# MERGE-PREP — Per-PR Closeout Tasks (Restructured — 4-PR Split)

Tasks 79-82 below are restructured: each PR has its own mini-closeout (regression + push + open + CI + merge), and only PR 4 carries the version bump + release.

---

## Per-PR Closeout Template (applies to PRs 1, 2, 3)

After each Feature block completes, run this mini-closeout BEFORE opening the PR. The merge is the final step; the version bump is **not** in scope for PRs 1-3.

```bash
# Step A1: Full regression on the feature branch (total coverage floor)
python -m pytest tests/ -x -q --cov=src/cognithor --cov-report=term-missing 2>&1 | tail -30
# Expected: all pass; total ≥ 89% enforced by pyproject.toml [tool.coverage.report]

# Step A2: Per-module coverage gate — cognithor.crew must meet 85%
python -m pytest --cov=cognithor.crew --cov-report=term-missing \
                  --cov-fail-under=85 tests/test_crew/
# Non-zero exit here BLOCKS the PR — CI enforces the same invocation.

# Step B: Ruff (both check and format-check — see feedback memory)
python -m ruff check .
python -m ruff format --check .

# Step C: Mypy --strict on new code
python -m mypy --strict src/cognithor/crew

# Step D: Push + open PR (DO NOT chain with merge via &&)
git push -u origin <feature-branch-name>
gh pr create --title "<PR title>" --body "..."

# Step E: Wait CI green, then merge. NEVER chain merge + cleanup via &&
#        (per feedback memory — two branch-closure incidents this session).
gh pr merge <PR-number> --squash
# Cleanup runs in a SEPARATE command only after merge is confirmed.

# Step F: Verify main CI green BEFORE branching the next PR
#        If any workflow fails on main after merge, STOP — investigate and fix
#        before cutting the next feature branch off a broken main.
gh run list --branch main --limit 3 --json conclusion,workflowName
# Expected: every recent workflow shows "conclusion": "success".
```

**PR 1 (the first PR touching the Crew layer) owns the coverage-floor config.** Add to `pyproject.toml` in Task 20's Step-3 commit:

```toml
[tool.coverage.report]
fail_under = 89
# Individual per-module gates enforced via --cov-fail-under=85 on
# pytest --cov=cognithor.crew invocations in CI / per-PR closeout.
show_missing = true
```

---

### Task 79: PR 1 — Feature 1 (Crew-Layer Core) merge-prep

**Branch:** `feat/cognithor-crew-v1-f1`

- [ ] **Step 1:** Run per-PR closeout template Steps A-C on tasks 1-20.
- [ ] **Step 2:** Push:
  ```bash
  git push -u origin feat/cognithor-crew-v1-f1
  ```
- [ ] **Step 3:** Open PR:
  ```bash
  gh pr create \
    --title "feat(crew): Crew-Layer core (v1.0 adoption — Feature 1)" \
    --body "$(cat <<'EOF'
## Summary
- New `cognithor.crew` package: `CrewAgent`, `CrewTask`, `Crew`, `CrewProcess`, YAML loader, decorators
- Routes through existing PGE-Trinity (`Planner.formulate_response`) — no bypass
- Audit events via `cognithor.security.audit.AuditTrail.record_event`
- Idempotent kickoff via `create_lock()` from `cognithor.core.distributed_lock`
- CHANGELOG `[Unreleased]` block present; no version bump in this PR

Spec: `docs/superpowers/specs/2026-04-23-cognithor-crew-v1-adoption.md` §1

## Test plan
- [ ] `pytest tests/test_crew/ -v` all pass
- [ ] Coverage on `src/cognithor/crew/` ≥ 85%
- [ ] No public-API exports from existing modules changed
EOF
)"
  ```
- [ ] **Step 4:** Wait all CI jobs green (CI + existing pipelines). Merge. Run cleanup as a **separate** command.
- [ ] **Step 5: Verify `main` CI green BEFORE starting PR 2 branch**

  Wait for GitHub Actions `main`-branch CI to complete. If any workflow fails on main after PR 1 merge, STOP — investigate and fix before branching PR 2 off a broken main.

  ```bash
  gh run list --branch main --limit 3 --json conclusion,workflowName
  ```

  Expected: all recent workflows show `"conclusion": "success"`. If any show `"failure"`, open a hotfix PR, get it merged, and re-check before cutting `feat/cognithor-crew-v1-f4`.

---

### Task 79b: PR 2 — Feature 4 (Guardrails) merge-prep

**Branch:** `feat/cognithor-crew-v1-f4` (cut from `main` after PR 1 merges).

- [ ] **Step 1:** Create branch from updated `main`:
  ```bash
  git checkout main && git pull
  git checkout -b feat/cognithor-crew-v1-f4
  # Cherry-pick / rebase the 12 commits from tasks 21-32 onto this branch
  ```
- [ ] **Step 2:** Run per-PR closeout Steps A-C.
- [ ] **Step 3:** Push + open PR. Title: `feat(crew): Task-Level Guardrails (v1.0 — Feature 4)`.
- [ ] **Step 4:** Wait CI green. Merge. Cleanup separately.
- [ ] **Step 5: Verify `main` CI green BEFORE starting PR 3 branch**

  ```bash
  gh run list --branch main --limit 3 --json conclusion,workflowName
  ```

  Expected: all recent workflows show `"conclusion": "success"`. Halt here if any failure — never branch the next PR off a broken main.

---

### Task 79c: PR 3 — Feature 3 (CLI + Templates) merge-prep

**Branch:** `feat/cognithor-crew-v1-f3` (cut from `main` after PR 2 merges).

- [ ] **Step 1:** Branch from updated `main`, port the 20 commits from tasks 33-52.
- [ ] **Step 2:** Run per-PR closeout Steps A-C. **Additionally** run the scaffold-templates CI check locally:
  ```bash
  # Simulate .github/workflows/scaffold-templates.yml
  for t in research customer-support data-analyst content versicherungs-vergleich; do
      mkdir -p /tmp/t_"$t" && cd /tmp/t_"$t"
      python -m cognithor init test --template "$t" --lang de && \
      cd test && pip install -e ".[dev]" && python -m pytest tests/
  done
  ```
- [ ] **Step 3:** Push + open PR. Title: `feat(crew): init CLI + 5 first-party templates (v1.0 — Feature 3)`.
- [ ] **Step 4:** Wait CI green. Merge. Cleanup separately.
- [ ] **Step 5: Verify `main` CI green BEFORE starting PR 4 branch**

  ```bash
  gh run list --branch main --limit 3 --json conclusion,workflowName
  ```

  Expected: all recent workflows show `"conclusion": "success"`. Halt if any failure — PR 4 is the release PR, so a broken main here means v0.93.0 launches broken.

---

### Task 80a: PR 4a — Feature 7 (Integrations Catalog + sevDesk) merge-prep

**Branch:** `feat/cognithor-crew-v1-f7` (cut from `main` after PR 3 merges). Code-only PR — no version bump, no docs.

**Scope:** Tasks 67-78 only. The catalog generator, CI workflow, sevDesk connector, DACH category overview (`docs/integrations/README.md` + `catalog.json` — these are reference docs for the catalog itself, not user-facing quickstart).

- [ ] **Step 1:** Branch from updated `main`, port tasks 67-78 commits.
- [ ] **Step 2:** Run per-PR closeout Steps A-C.
- [ ] **Step 3:** Push + open PR. Title: `feat(crew): Integrations catalog + sevDesk connector (v1.0 — Feature 7)`. The PR body calls out that Feature 2 (docs + release) lands in the follow-up PR 4b.
- [ ] **Step 4:** Wait CI green. Merge. Cleanup separately.
- [ ] **Step 5: Verify `main` CI green BEFORE starting PR 4b branch**

  ```bash
  gh run list --branch main --limit 3 --json conclusion,workflowName
  ```

  Expected: all recent workflows show `"conclusion": "success"`.

---

### Task 80c: Draft cognithor-site PR for v0.93.0

**Rationale:** Task 82 Step 2 hard-gates the tag push on a merged cognithor-site PR with the new integrations page live. Until this task existed, the site-PR had no task, no owner, and no start trigger — turning the hard gate into a surprise. This task schedules the site-PR to run IN PARALLEL with the PR 4b review window (not sequentially after), so the site-PR is already merged by the time Task 82 Step 2 runs its curl check.

**Dependencies:** PR 4a merged (so `docs/integrations/catalog.json` exists on `main`). Runs in parallel with PR 4b review.

**Repo:** separate `cognithor-site` repo at `D:\Jarvis\cognithor-site\` (Vercel, Next.js). NOT in this plan's commit range.

**Files (in cognithor-site repo — relative to that repo's root):**
- Modify: `app/(main)/integrations/page.tsx` — fetch catalog.json at build time
- Modify: `app/(main)/changelog/page.mdx` — append v0.93.0 release notes
- Create: `content/releases/0.93.0.md` — short release summary linked from changelog
- Modify (if needed): `next.config.js` — whitelist new raw-content origin

- [ ] **Step 1: Create branch in cognithor-site repo**

```bash
cd D:\Jarvis\cognithor-site\
git checkout main && git pull
git checkout -b release/v0.93.0
```

- [ ] **Step 2: Update integrations page**

Fetch `catalog.json` at build time from
`https://raw.githubusercontent.com/Alex8791-cyber/cognithor/v0.93.0/docs/integrations/catalog.json`
(note: `v0.93.0` refers to the upcoming tag — at draft time, swap to a commit SHA on `main`, then bump to the tag once PR 4b merges). Render a category grid (5 categories: CRM, Productivity, Finance, DevOps, Messaging) with a dedicated DACH section for `dach_specific: true` entries.

Fallback: if the fetch fails at build time (offline CI), fall back to a committed `content/integrations/catalog.snapshot.json` last-known-good copy.

- [ ] **Step 3: Append changelog + release notes**

Use the GitHub release body drafted in Task 81b as source of truth. Render as MDX page. Add an entry to the changelog index page.

- [ ] **Step 4: Open PR**

```bash
gh pr create \
  --title "release: v0.93.0 integrations + changelog" \
  --body "Paired with cognithor v0.93.0 release. **BLOCKS tag push** in Task 82."
```

- [ ] **Step 5: Hand off to site owner for review**

Owner: @AlexanderSoellner. If unavailable during the PR 4b review window, delegate to the CI-only review path — if tests pass + Vercel preview deploy is green + screenshots look correct, the PR is mergeable without human re-review.

- [ ] **Step 6: Coordinate merge timing**

The site-PR MUST be merged and the Vercel deployment MUST be live BEFORE Task 82 Step 3 (`git tag v0.93.0`). If PR 4b merges first, PAUSE before tagging until the site-PR merges + `curl -fsSL https://cognithor.ai/integrations` returns 200.

---

### Task 80b: PR 4b — Feature 2 (Quickstart Docs) + version bump + release prep

**Branch:** `feat/cognithor-crew-v1-f2` (cut from `main` after PR 4a merges). This is the ONLY PR that bumps the version and triggers the release pipeline.

**Rationale for the split:** PR 4a is code; PR 4b is docs + release ceremony. Reviewer pools differ — docs benefit from preview builds + external-reader usability pass (spec §12 AC 4), code benefits from tight diff review. Keeping them separate reduces PR 4b's diff to docs + 5 version-bump lines.

**Files to modify in this PR (on top of tasks 53-66 commits):**
- `CHANGELOG.md`: `[Unreleased]` → `[0.93.0]` — consolidate all feature entries from PRs 1-4a under the dated 0.93.0 section
- `pyproject.toml`: `0.92.7` → `0.93.0`
- `src/cognithor/__init__.py`: `__version__ = "0.93.0"`
- `flutter_app/pubspec.yaml`: version
- `flutter_app/lib/providers/connection_provider.dart`: `kFrontendVersion`

The Crew-Layer is a MINOR bump (additive, no breaking changes — each feature's CHANGELOG already carries `### Breaking Changes: None.`). Date the `[0.93.0]` section with the merge day.

- [ ] **Step 1:** Branch from updated `main`, port tasks 53-66 commits.
- [ ] **Step 2:** Run per-PR closeout Steps A-C.
- [ ] **Step 3:** **Verify external-reader gate (Task 62):**
  ```bash
  grep -q "Verdict: PASS" docs/quickstart/EXTERNAL_REVIEW_RESULTS.md || {
    echo "BLOCKED: EXTERNAL_REVIEW_RESULTS.md has no PASS verdict — spec §12 AC 4"
    exit 1
  }
  ```
  This MUST pass before continuing. If no external reader has completed the checklist, find one.
- [ ] **Step 3b: Flutter CLI-command catalog check (R4-I4)**

  The Flutter Command Center may surface a CLI command catalog in-app. If it
  hardcodes the list (e.g. `const availableCommands = [...]`), the new `init`
  and `run` subcommands must be added; if it introspects `cognithor --help`
  dynamically, nothing is needed. Run:

  ```bash
  # Look for hardcoded CLI command catalogs in the Flutter sources.
  grep -rn -E "available_commands|availableCommands|cli_commands|cliCommands|command_list|commandList" flutter_app/lib/ || echo "no hardcoded catalog found"
  ```

  - If grep prints nothing: introspection path assumed — no action needed, but
    spin up a local Flutter build with a 0.93.0-dev cognithor and sanity-check
    that the help output flows through.
  - If grep prints hits: extend the catalog to include `init` and `run`, then
    commit the update together with the version bump below.
- [ ] **Step 4:** Apply the 5-file version bump. Commit:
  ```bash
  git add CHANGELOG.md pyproject.toml src/cognithor/__init__.py \
          flutter_app/pubspec.yaml flutter_app/lib/providers/connection_provider.dart
  git commit -m "chore(release): bump to 0.93.0 — Crew-Layer v1.0"
  ```
- [ ] **Step 4b: Release-notes date + NOTICE year verification (R4-C5)**

  Task 81b creates `docs/releases/v0.93.0.md` with a ``**Release date:** YYYY-MM-DD``
  placeholder — this placeholder MUST be replaced with the actual release date
  before PR 4b opens. The ``NOTICE`` file's copyright year should also match the
  current release year. Run these checks and fix before pushing:

  ```bash
  # Date placeholder must be replaced with a concrete ISO date.
  if grep -n "YYYY-MM-DD" docs/releases/v0.93.0.md ; then
      echo "ERROR: date placeholder YYYY-MM-DD not replaced in docs/releases/v0.93.0.md"
      echo "Fix: sed -i \"s/YYYY-MM-DD/$(date -u +%Y-%m-%d)/\" docs/releases/v0.93.0.md"
      exit 1
  fi

  # NOTICE copyright-year sanity check. Not a hard failure — the file may
  # legitimately span multiple years (e.g. 'Copyright 2025-2026') — but warn
  # when the current year is missing so a release slipping into 2027 catches
  # attention before it ships.
  release_year=$(date -u +%Y)
  if ! grep -qE "Copyright[^0-9]+${release_year}" NOTICE ; then
      echo "WARNING: NOTICE does not mention copyright year ${release_year} — review before tagging"
  fi

  # If v0.93.0.md or NOTICE changed in Step 4b, amend the version-bump commit:
  git add docs/releases/v0.93.0.md NOTICE 2>/dev/null || true
  git diff --cached --quiet || git commit --amend --no-edit
  ```
- [ ] **Step 5:** Push + open PR. Title: `docs(crew): Quickstart + v0.93.0 release (v1.0 — Feature 2)`. The PR body references the four earlier merged PRs and the spec §12 sign-off checklist.
- [ ] **Step 6:** Wait ALL CI jobs green (CI + scaffold-templates + quickstart-examples + integrations-catalog + Windows Installer + Mobile + Linux .deb + Flutter Web + Release Build + performance-benchmark).

---

### Task 81: PR 4b — PR open gate (external-reader pass required)

Before `gh pr create` runs in Task 80b Step 5, the PR-open gate (spec §12 AC 4) is enforced via the grep in Task 80b Step 3. If that grep fails, Task 80b aborts and we do not open the PR.

This task exists as an explicit acceptance step to make the external-reader dependency first-class rather than a footnote.

**Verdict model — 3 outcomes (see NI11 in Round 3 review):**

Spec §12 AC 4 asks for a "successful" external review. The original "PASS in ≤15 min, zero questions back to author" bar was unrealistically tight — realistic first-time users hit small snags (Ollama pull, Python version) that don't indicate a broken onboarding but do push them past 15 min. We split PASS into two actionable outcomes and keep FAIL strict:

- **PASS** — user completed happy path end-to-end with ≤1 small clarifying question, total elapsed time ≤30 min. Release goes ahead.
- **PASS_WITH_FOLLOWUPS** — user completed happy path unassisted, but logged ≥1 minor confusion worth fixing in 0.93.1. Acceptable for v0.93.0 release IF all follow-ups are filed as issues AND no issue is a blocker (i.e. doesn't prevent completion). Release goes ahead; follow-ups stack for 0.93.1.
- **FAIL** — user could NOT complete happy path without author help, OR hit a blocking bug, OR the snag took >30 min to work around. Release BLOCKED until fixed + re-tested.

**Acceptance (what Task 80b Step 3's grep looks for):**

- [ ] `docs/quickstart/EXTERNAL_REVIEW_RESULTS.md` contains `Verdict: PASS` OR `Verdict: PASS_WITH_FOLLOWUPS`
- [ ] At least one named external tester (not the plan author)
- [ ] Total elapsed time recorded, ≤30 min
- [ ] If `PASS_WITH_FOLLOWUPS`: every follow-up tracked as a 0.93.1 GitHub issue, with issue numbers pasted in the results file
- [ ] If `FAIL`: release blocked, fix + re-test required

Update the grep in Task 80b Step 3:

```bash
grep -qE "Verdict: (PASS|PASS_WITH_FOLLOWUPS)" docs/quickstart/EXTERNAL_REVIEW_RESULTS.md || {
  echo "BLOCKED: EXTERNAL_REVIEW_RESULTS.md has no passing verdict — spec §12 AC 4"
  exit 1
}
```

**Earlier recruitment — mini-task inside Task 54:**

To avoid a last-minute scramble, recruitment starts in Week 4, not Week 6. Add a mini-step at the end of Task 54 ("Recruit 2-3 external testers for the Week 6 review slot"): identify non-author developers who are Python-literate but new to Cognithor, and secure a calendar slot BEFORE PR 4b opens. If no reader is available, escalate to the plan lead before docs work completes — a release-blocking gate that you discover on the release-day morning is a known anti-pattern.

---

### Task 81b: Release announcement copy (GitHub release body + blog + social)

**Rationale:** Version 0.93.0 is the Crew-Layer debut — a marketing moment worth more than a one-line CHANGELOG entry. Shipping release notes + blog outline + 3 social posts in the same PR as the version bump avoids the "released to silence" pattern. See NI10 in Round 3 review.

**Files:**
- Create: `docs/releases/v0.93.0.md` — GitHub release body (used verbatim by `gh release create`)
- Create: `docs/releases/v0.93.0-announcement.md` — blog outline + 3 social posts (DE + EN)

- [ ] **Step 1: Write GitHub release body in `docs/releases/v0.93.0.md`**

Template (populate from the `[0.93.0]` CHANGELOG section — both Round 3 fixes and the feature list).

**Date placeholder (R4-C5):** the `YYYY-MM-DD` literal below MUST be replaced
with the concrete release date before PR 4b opens. Task 80b Step 4b runs a
`grep "YYYY-MM-DD"` gate that blocks PR creation until the substitution is made.
Use UTC date: `sed -i "s/YYYY-MM-DD/$(date -u +%Y-%m-%d)/" docs/releases/v0.93.0.md`.

```markdown
# Cognithor 0.93.0 — Crew-Layer, Guardrails, Templates

**Release date:** YYYY-MM-DD

**What's new:**
- **Crew-Layer v1.0:** declarative multi-agent crews on top of PGE-Trinity — `Crew(agents=[...], tasks=[...]).kickoff()`.
- **Guardrails:** function + string validators, 4 built-ins (`hallucination_check`, `word_count`, `no_pii`, `schema`), retryable verdicts with audit-chain events.
- **`cognithor init`:** Jinja2-based scaffolder + 5 first-party templates (`research`, `customer-support`, `data-analyst`, `content`, `versicherungs-vergleich`).
- **8-page Quickstart:** from empty terminal to first kickoff in <10 minutes (bilingual DE + EN, FAQ page included).
- **Integrations catalog:** DACH-aware, fully offline-capable `versicherungs-vergleich` template, first-party sevDesk connector.

**Hello, Crew:**

```python
from cognithor.crew import Crew, CrewAgent, CrewTask

analyst = CrewAgent(role="analyst", goal="compare PKV tariffs")
writer = CrewAgent(role="writer", goal="draft a customer report")
research = CrewTask(
    description="Compare the top three PKV tariffs for a 35-year-old",
    expected_output="Tabular comparison with price, coverage, exclusions",
    agent=analyst,
)
report = CrewTask(
    description="Turn the analysis into a customer report",
    expected_output="Markdown",
    agent=writer,
    context=[research],
)

out = Crew(agents=[analyst, writer], tasks=[research, report]).kickoff()
print(out.raw)
```

**Upgrade:**
```
pip install --upgrade cognithor==0.93.0
```

No breaking changes. Every existing `@agent` / `@tool` / `@skill` keeps working.

**Full changelog:** see [CHANGELOG.md](../CHANGELOG.md).

**Credits:** Thanks to reviewers, testers, and external readers — [see contributors](https://github.com/Alex8791-cyber/cognithor/graphs/contributors).
```

- [ ] **Step 2: Write blog outline in `docs/releases/v0.93.0-announcement.md`**

```markdown
# 0.93.0 — Crew-Layer lands (blog outline)

## Headers (DE + EN)

### DE
1. Problem: Agent-Orchestrierung ohne Framework-Lock-in
2. Wie's funktioniert: PGE-Trinity unter der Haube
3. Hello-World: Erste Crew in 10 Zeilen
4. Guardrails: 10-Zeilen-Validierung
5. Templates: In 30 Sekunden zum Start
6. Was kommt als Nächstes: Flows + Trace-UI in v1.x

### EN
1. Problem: agent orchestration without framework lock-in
2. How it works: PGE-Trinity under the hood
3. Hello-World: first crew in 10 lines
4. Guardrails: validation in 10 lines
5. Templates: go from zero to crew in 30 seconds
6. Next up: Flows + Trace-UI in v1.x

## 3 social posts (DE + EN each)

### Twitter/X (280 chars)
**EN:** "Cognithor 0.93.0 is out. Crew-Layer: declarative multi-agent teams on top of PGE-Trinity. 5 ready-to-use templates (including a DACH PKV-Vergleich). Guardrails + audit trail baked in. `pip install cognithor==0.93.0` — docs: cognithor.ai/quickstart"

**DE:** "Cognithor 0.93.0 ist da. Crew-Layer: deklarative Multi-Agent-Teams auf PGE-Trinity. 5 Templates (inkl. PKV-Vergleich für DACH). Guardrails + Audit-Trail. `pip install cognithor==0.93.0` — Doku: cognithor.ai/quickstart"

### LinkedIn (600 chars)
**EN:** "Shipped today: Cognithor 0.93.0. The big addition is the Crew-Layer — declarative multi-agent teams on top of our existing PGE-Trinity runtime. You write Python dataclasses for agents + tasks; Cognithor compiles them into governed PlanRequests with full Gatekeeper checks and Hashline-Guard audit trails. Five first-party templates ship with it, including a fully offline DACH PKV-Vergleich template (§34d-neutral). 8-page Quickstart gets you from empty terminal to first kickoff in under 10 minutes. No breaking changes. Docs: cognithor.ai/quickstart"

**DE:** "Heute veröffentlicht: Cognithor 0.93.0. Das Hauptmerkmal ist der Crew-Layer — deklarative Multi-Agent-Teams auf unserer bestehenden PGE-Trinity-Runtime. Agents und Tasks werden als Python-Dataclasses beschrieben; Cognithor kompiliert daraus geprüfte PlanRequests mit vollem Gatekeeper-Check und Hashline-Guard-Audit-Trail. Fünf First-Party-Templates sind dabei, inklusive eines komplett offline-fähigen PKV-Vergleichs-Templates für den DACH-Raum (§34d-neutral). Der 8-teilige Quickstart führt dich in unter 10 Minuten vom leeren Terminal zum ersten Kickoff. Keine Breaking Changes. Doku: cognithor.ai/quickstart"

### Discord community (400 chars + code snippet)
**EN:** "v0.93.0 is live — Crew-Layer, Guardrails, `cognithor init`. The one-liner you came here for:

```python
Crew(agents=[analyst, writer], tasks=[research, report]).kickoff()
```

Quickstart + migration notes in docs/quickstart/. Breaking changes: none. Feedback very welcome in #v0.93-feedback."

**DE:** "v0.93.0 ist live — Crew-Layer, Guardrails, `cognithor init`. Der Einzeiler, für den ihr gekommen seid:

```python
Crew(agents=[analyst, writer], tasks=[research, report]).kickoff()
```

Quickstart + Migrations-Notizen in docs/quickstart/. Breaking Changes: keine. Feedback gerne in #v0.93-feedback."
```

- [ ] **Step 3: Commit**

```bash
git add docs/releases/v0.93.0.md docs/releases/v0.93.0-announcement.md
git commit -m "docs(release): v0.93.0 release body + announcement outline + social copy"
```

- [ ] **Step 4: Use the GitHub release body verbatim**

Task 82 Step 4 runs `gh release edit v0.93.0 --notes-file docs/releases/v0.93.0.md` after `publish.yml` completes, replacing the auto-generated body with our curated text.

---

### Task 82: Post-merge release `v0.93.0`

(This task runs in a SEPARATE session/turn after PR 4b is green and merged — per the feedback memory "never chain merge + cleanup via &&". Cleanup runs after merge confirmation.)

**Ordering rationale:** The site-PR is a HARD GATE on tag creation. Published release notes link to `cognithor.ai/integrations` — if that page doesn't exist when `v0.93.0` ships to PyPI, every release-page visitor hits a 404. Therefore the site-PR is merged BEFORE we tag, not after. `publish.yml` fires on tag push, so once tag exists the train has left the station.

- [ ] **Step 1: Version metadata already bumped in PR 4b**

Verify the 5-file version bump from Task 80b is present on `main`:

```bash
git checkout main && git pull
grep -q 'version = "0.93.0"' pyproject.toml
grep -q '__version__ = "0.93.0"' src/cognithor/__init__.py
grep -q '## \[0.93.0\]' CHANGELOG.md
```

Expected: all three greps exit 0. If any fails, STOP — the version bump was missed in PR 4b and needs a hotfix PR before release.

- [ ] **Step 2: Verify cognithor-site PR (from Task 80c) is MERGED and live — BLOCKS Step 3**

The `cognithor.ai/integrations` page lives in the separate `cognithor-site` (Vercel) repo. The PR for it is drafted in **Task 80c** (runs in parallel with PR 4b review). By the time this step runs, that PR should already be merged. If not, this step blocks tag push until it is.

Expected content of the site-PR (as drafted in Task 80c):
1. Fetches `docs/integrations/catalog.json` from the upcoming `v0.93.0` tag at build time.
2. Renders a category-grouped integration grid with a dedicated DACH section for `dach_specific: true` entries.
3. Adds the `v0.93.0` release notes page linking back to the GitHub release.
4. Deploys to Vercel production.

**Wait until this site-PR is MERGED and the Vercel deployment is live before proceeding.** Verify:

```bash
curl -fsSL https://cognithor.ai/integrations >/dev/null && echo "live"
```

Expected: `live`. Log the site-PR number + merge timestamp in the release notes draft (kept local until Step 5).

**If site-PR is not yet merged, STOP — do not tag. Rationale: published release must link to live docs, not 404.**

This is a **hard gate** on spec §12 AC 7. Cross-repo coordination is unavoidable here; publish.yml fires on tag push and there is no undo. The parallelized schedule in Task 80c prevents this gate from being a surprise.

- [ ] **Step 3: Tag + push (only after Step 2 confirmed live)**

```bash
git checkout main && git pull  # pick up any last-minute fixes
git tag -a v0.93.0 -m "Cognithor v0.93.0 — Crew-Layer v1.0 (Features 1, 4, 3, 2, 7)"
git push origin v0.93.0
```

- [ ] **Step 4: `publish.yml` fires on tag push — wait for it to complete**

Monitor `gh run list --workflow=publish.yml --limit 1` until `"conclusion": "success"`. This uploads the sdist + wheel to PyPI and attaches binaries to the GitHub Release.

- [ ] **Step 5: Verify PyPI + GitHub release artifacts**

```bash
pip install --upgrade cognithor==0.93.0
cognithor --version        # must print 0.93.0
gh release view v0.93.0    # confirm 6 artifacts attached
```

Expected artifacts: Windows Installer, Launcher, Linux .deb, Android APK, iOS IPA, Flutter Web bundle.

**If any of Steps 3-5 fails, STOP immediately and consult `docs/releases/ROLLBACK.md` (created in Task 82b) before retrying. Do NOT re-run the tag push — a PyPI wheel, once published, can be yanked but never re-uploaded under the same version.**

---

### Task 82b: Release rollback runbook

**Rationale:** Tag-push → PyPI publish is one-way. If the Windows installer is broken, the PKV example fails on a real user's Ollama, or CI lied about success, there is currently zero documented recovery path. Every release process needs a rollback page. This task creates one and references it from Task 82's pre-publish checklist.

**Files:**
- Create: `docs/releases/ROLLBACK.md`
- Create: `scripts/release_rollback.sh`

- [ ] **Step 1: Create `docs/releases/ROLLBACK.md` decision matrix**

```markdown
# Release Rollback Runbook

Scope: what to do when a tagged release goes wrong. Covers `v0.93.0` and
every subsequent release. Consult BEFORE running `git tag` — not after.

## Decision matrix

| Severity | Symptom | Action |
|---|---|---|
| **CATASTROPHIC** | `pip install cognithor==X.Y.Z` completely broken (missing file, syntax error on import) | **Yank** PyPI release (below). Do NOT delete. |
| **HIGH** | Shipping bug found after release, `pip install` works but feature is broken or unsafe | **Hotfix** `X.Y.Z+1` (below). |
| **ARCHITECTURAL** | Regression so severe the new version should never have shipped | **Revert** to prior minor (below). |

## Yank procedure (CATASTROPHIC)

1. **NEVER** delete a PyPI release — it permanently reserves that version number, blocking any re-upload. Yank instead.
2. Sign in to https://pypi.org/manage/project/cognithor/release/X.Y.Z/ as a project maintainer.
3. Click "Options" → "Yank release". Provide reason (e.g. "Broken wheel, fixed in X.Y.Z+1").
4. PyPI keeps the file reachable for existing `==X.Y.Z` pins but omits it from `pip install cognithor` range resolution.
5. Announce in GitHub issue + Discord + mailing list. Link to the yank reason.

## Hotfix procedure (HIGH)

1. Branch `hotfix/X.Y.Z+1` off the broken tag:
   ```bash
   git checkout -b hotfix/0.93.1 v0.93.0
   ```
2. Commit the fix with a CHANGELOG entry:
   ```
   ## [0.93.1] — YYYY-MM-DD
   ### Fixed
   - <bug description>
   ```
3. Bump version in the 5 locations Task 80b enumerates:
   - `pyproject.toml`
   - `src/cognithor/__init__.py`
   - `CHANGELOG.md`
   - `flutter_app/pubspec.yaml`
   - `flutter_app/lib/providers/connection_provider.dart`
4. Re-run Task 82's full gate sequence:
   - cognithor-site PR for `0.93.1` merged
   - All CI jobs green
   - Artifacts verified on preview
5. Tag + push:
   ```bash
   git tag -a v0.93.1 -m "Cognithor v0.93.1 — hotfix"
   git push origin v0.93.1
   ```
6. `publish.yml` fires. Verify as in Task 82 Step 5.

## Revert procedure (ARCHITECTURAL)

1. Yank the bad PyPI release (as above).
2. Delete the tag locally + remote:
   ```bash
   git tag -d v0.93.0
   git push origin :refs/tags/v0.93.0
   ```
   (This does NOT remove the PyPI upload; yank is separate.)
3. File an incident postmortem in `docs/releases/INCIDENT-YYYY-MM-DD.md` with:
   - Timeline
   - Root cause
   - Why CI missed it
   - Prevention actions
4. Open a revert PR to `main` reverting the `0.93.0` version bump + feature merges.
5. Plan the re-release under a new version (e.g. `0.93.1` or `0.94.0`).

## Never

- **Never** `git push --force origin main` to erase the release commit.
- **Never** skip the incident postmortem — even for "obvious" fixes.
- **Never** re-upload the same version number to PyPI. Bump to the next patch.
```

- [ ] **Step 2: Create `scripts/release_rollback.sh` helper**

```bash
#!/usr/bin/env bash
# Release rollback helper — always run in DRY-RUN first.
# Usage: scripts/release_rollback.sh <version> [dry-run|delete-tag|yank|full]
set -euo pipefail

VERSION="${1:?Usage: release_rollback.sh <version> [dry-run|delete-tag|yank|full]}"
ACTION="${2:-dry-run}"

case "$ACTION" in
  dry-run)
    echo "DRY RUN for v$VERSION:"
    echo "  - would delete tag v$VERSION locally + remote"
    echo "  - would prompt for PyPI yank via https://pypi.org/manage/project/cognithor/release/$VERSION/"
    echo "  - would open revert PR draft to main"
    ;;
  delete-tag)
    git tag -d "v$VERSION" || true
    git push origin ":refs/tags/v$VERSION"
    echo "Tag v$VERSION deleted locally + on origin."
    ;;
  yank)
    echo "MANUAL STEP: yank v$VERSION via"
    echo "  https://pypi.org/manage/project/cognithor/release/$VERSION/"
    echo "Provide a reason. Never delete — only yank."
    ;;
  full)
    git tag -d "v$VERSION" || true
    git push origin ":refs/tags/v$VERSION"
    echo "Tag v$VERSION deleted locally + on origin."
    echo "MANUAL: yank PyPI release at https://pypi.org/manage/project/cognithor/release/$VERSION/"
    echo "MANUAL: open revert PR against main"
    ;;
  *)
    echo "Unknown action: $ACTION"
    echo "Actions: dry-run | delete-tag | yank | full"
    exit 1
    ;;
esac
```

- [ ] **Step 3: Commit**

```bash
chmod +x scripts/release_rollback.sh
git add docs/releases/ROLLBACK.md scripts/release_rollback.sh
git commit -m "docs(release): add v0.93.0 rollback runbook + helper script"
```

- [ ] **Step 4: Cross-reference from Task 82**

The final note under Task 82 Step 5 ("If any of Steps 3-5 fails, STOP … consult `docs/releases/ROLLBACK.md`") is already in place. Sanity-check that the reference exists.

---

# Cross-cutting Concerns

## Testing

- Every new module has ≥ 85% line coverage
- Every Feature ships with at least ONE integration test that exercises its public API end-to-end
- CI on every PR: full `pytest tests/` + scaffold-templates matrix + quickstart-examples matrix + integrations-catalog drift + existing CI pipelines

## Docs

- Every public class/function has a Google-style docstring
- Every new `__init__.py` exposes an `__all__` list
- Every new CLI subcommand has `--help` text in both DE and EN

## Lizenzhygiene

- `NOTICE` carries the CrewAI-inspiration line (added in Task 20)
- No `crewai` package imported anywhere
- No source-level copy from crewAIInc/crewAI — verified manually during PR review

## DSGVO

- `versicherungs-vergleich` template: `tools=[]` in default (no external calls)
- `no_pii()` guardrail active in all templates that emit user-facing text
- No new external HTTP endpoints added to the default code path

---

# Self-Review Checklist

After finishing Task 82, walk the spec one more time:

- [ ] Spec §1.6 acceptance — PKV example runs with Ollama → Task 19, 55
- [ ] Spec §1.6 — Each CrewTask produces an audit-chain trace block → Task 14
- [ ] Spec §1.6 — Gatekeeper checks every tool action → Task 12
- [ ] Spec §1.6 — Missing tool error with suggestion → Task 7, 18
- [ ] Spec §1.6 — `kickoff()` idempotent re-callable → Task 15
- [ ] Spec §1.6 — Tests under `tests/test_crew/` including sequential, hierarchical, missing-tool, context-passing, async, guardrail-retry → Tasks 8, 10, 18, 13, 9, 29
- [ ] Spec §4.5 — Function + string guardrails run → Tasks 22, 23
- [ ] Spec §4.5 — All four built-in guardrails tested → Tasks 24, 25, 26, 27
- [ ] Spec §4.5 — Guardrail-Events in audit-chain → Task 30
- [ ] Spec §4.5 — `docs/quickstart/04-guardrails.md` → Task 58
- [ ] Spec §3.6 — `cognithor init test_proj --template research` works → Task 39
- [ ] Spec §3.6 — `cognithor init --list-templates` shows 5 templates → Task 44
- [ ] Spec §3.6 — `versicherungs-vergleich` runs with Ollama only → Task 43
- [ ] Spec §3.6 — Scaffolded tests (`pytest`) pass → Task 45
- [ ] Spec §3.6 — CLI `--help` bilingual → Task 46
- [ ] Spec §3.6 — Existing skill-scaffolder still works → no-touch (verified by Task 79 full regression)
- [ ] Spec §2.4 — External-reader usability checklist → Task 62
- [ ] Spec §7.3 — Every listed integration has a doc link → Task 72
- [ ] Spec §7.3 — `scripts/generate_integrations_catalog.py` in CI → Task 69
- [ ] Spec §7.3 — One DACH connector functional + tested → Task 70, 71
- [ ] Spec §7.3 — No listing without repo-entsprechung → Task 67 (tests enforce)
- [ ] Spec §8 — All cross-cutting concerns addressed
- [ ] Spec §12 — All v1.0 sign-off criteria for Features 1-4, 7 met

Features 5 + 6 intentionally OUT of scope (v1.x).

---

# Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-04-24-cognithor-crew-v1.md`.**

Approach: **Subagent-Driven** — fresh subagent per task, two-stage review (spec compliance → code quality), same pattern as video-input PR #140.

After each task:
1. Dispatch spec-compliance reviewer
2. Dispatch code-quality reviewer
3. Mark task complete on checkboxes
4. Move to next task

After every Feature (1, 4, 3, 7, 2) is fully implemented + self-reviewed:
- Task 79 — PR 1 (Feature 1)
- Task 79b — PR 2 (Feature 4)
- Task 79c — PR 3 (Feature 3)
- Task 80a — PR 4a (Feature 7 — code only)
- Task 80b — PR 4b (Feature 2 — docs + version bump)
- Task 80c — cognithor-site PR (runs in parallel with PR 4b review; hard gate on Task 82)
- Task 81 — PR 4b open gate (external-reader PASS required)
- Task 81b — Release announcement copy (GitHub release body + social posts)
- Task 82 — post-merge release (site-PR must already be merged)
- Task 82b — release rollback runbook

Target: v0.93.0 released to PyPI + GitHub. All 6 release artifacts (Windows Installer, Launcher, Linux .deb, Android APK, iOS IPA, Flutter Web) auto-built + attached to the release.

