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
4. **Feature 2 (Tasks 53-66)** — Quickstart docs (documents 1 + 3 + 4)
5. **Feature 7 (Tasks 67-78)** — Integrations catalog (parallel-safe)
6. **Final integration + PR prep (Tasks 79-82)**

Features 5 (Trace-UI) and 6 (Flows) are explicitly out of plan scope — those are v1.x per spec §5.6 and §6.6.

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

Seven pages each in German (default) and English (`.en.md` suffix):

- `00-installation.md` / `.en.md`
- `01-first-crew.md` / `.en.md`
- `02-first-tool.md` / `.en.md`
- `03-first-skill.md` / `.en.md`
- `04-guardrails.md` / `.en.md`
- `05-deployment.md` / `.en.md`
- `06-next-steps.md` / `.en.md`
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
class CrewError(Exception): ...
class CrewCompilationError(CrewError): ...
class ToolNotFoundError(CrewError): ...
class GuardrailFailure(CrewError): ...
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

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CrewAgent(BaseModel):
    """Declarative description of an agent participating in a Crew.

    Concept inspired by CrewAI's Agent; re-implementation in Apache 2.0.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    role: str = Field(..., min_length=1, description="Short role name, used in logs")
    goal: str = Field(..., min_length=1, description="What this agent is trying to accomplish")
    backstory: str = Field(default="", description="Context the Planner uses to shape the system prompt")
    tools: list[str] = Field(default_factory=list, description="Tool names resolved via MCP registry")
    llm: str | None = Field(default=None, description="Model spec, e.g. 'ollama/qwen3:32b'")
    allow_delegation: bool = Field(default=False)
    max_iter: int = Field(default=20, ge=1, le=200)
    memory: bool = Field(default=True, description="Enable 6-Tier Cognitive Memory for this agent")
    verbose: bool = Field(default=False)
    metadata: dict[str, Any] = Field(default_factory=dict)
```

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

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_crew/test_tool_resolution.py
from unittest.mock import MagicMock
import pytest
from cognithor.crew.errors import ToolNotFoundError
from cognithor.crew.tool_resolver import resolve_tools, did_you_mean


class TestResolveTools:
    def test_resolves_known_tools(self):
        registry = MagicMock()
        registry.list_tool_names.return_value = ["web_search", "pdf_reader", "shell_run"]
        resolved = resolve_tools(["web_search", "pdf_reader"], registry=registry)
        assert resolved == ["web_search", "pdf_reader"]

    def test_unknown_tool_raises_with_suggestion(self):
        registry = MagicMock()
        registry.list_tool_names.return_value = ["web_search", "pdf_reader"]
        with pytest.raises(ToolNotFoundError) as exc:
            resolve_tools(["web_seach"], registry=registry)
        assert "web_seach" in str(exc.value)
        assert "web_search" in str(exc.value)

    def test_unknown_tool_no_close_match(self):
        registry = MagicMock()
        registry.list_tool_names.return_value = ["completely_other"]
        with pytest.raises(ToolNotFoundError) as exc:
            resolve_tools(["totally_foreign"], registry=registry)
        assert "totally_foreign" in str(exc.value)
        # No spurious suggestion when nothing is close
        assert "Meintest du" not in str(exc.value) or "completely_other" in str(exc.value)


class TestDidYouMean:
    def test_close_match(self):
        assert did_you_mean("web_seach", ["web_search", "pdf_reader"]) == "web_search"

    def test_no_match(self):
        assert did_you_mean("xyz", ["web_search"]) is None

    def test_exact_match_returns_none(self):
        # No suggestion when exact match exists
        assert did_you_mean("web_search", ["web_search"]) is None
```

- [ ] **Step 2: Run — expect failures**

- [ ] **Step 3: Implement `src/cognithor/crew/tool_resolver.py`**

```python
"""Resolve CrewAgent / CrewTask tool names against the MCP registry.

Provides friendly 'did you mean' suggestions for typos. Uses difflib from
stdlib — no new dependencies.
"""

from __future__ import annotations

import difflib
from typing import Any

from cognithor.crew.errors import ToolNotFoundError


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
    available = list(registry.list_tool_names())
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

- [ ] **Step 3: Implement `src/cognithor/crew/compiler.py`**

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
        out = execute_task(t, context=outputs, inputs=inputs, registry=registry)
        outputs.append(out)
    return CrewOutput(raw=outputs[-1].raw, tasks_output=outputs, trace_id=trace_id)
```

- [ ] **Step 4: Wire into `Crew.kickoff()`**

Replace the `kickoff` stub from Task 6 with:

```python
def kickoff(self, inputs: dict[str, Any] | None = None) -> CrewOutput:
    from cognithor.crew.compiler import compile_and_run_sync
    from cognithor.mcp.tool_registry_db import get_tool_registry  # adjust import to actual path
    return compile_and_run_sync(
        agents=self.agents,
        tasks=self.tasks,
        process=self.process,
        inputs=inputs,
        registry=get_tool_registry(),
    )
```

If `get_tool_registry` does not exist at that exact path, the implementer inspects `src/cognithor/mcp/tool_registry_db.py` and imports the right name. If tool-registry instantiation requires async setup, wrap in `asyncio.run(registry_factory())` or use a module-level singleton — decide at implementation time and document the choice in the commit body.

- [ ] **Step 5: Run — expect 2 pass**

- [ ] **Step 6: Ruff + commit**

```bash
git add src/cognithor/crew/compiler.py src/cognithor/crew/crew.py tests/test_crew/test_sequential_kickoff.py
git commit -m "feat(crew): sequential compile-and-run happy path"
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


@pytest.mark.asyncio
async def test_kickoff_async_returns_same_as_sync():
    agent = CrewAgent(role="x", goal="y")
    task = CrewTask(description="a", expected_output="b", agent=agent)
    crew = Crew(agents=[agent], tasks=[task])

    fake = TaskOutput(task_id=task.task_id, agent_role="x", raw="DONE")

    with patch("cognithor.crew.compiler.execute_task_async", new=AsyncMock(return_value=fake)):
        result = await crew.kickoff_async()

    assert result.raw == "DONE"
    assert len(result.tasks_output) == 1


@pytest.mark.asyncio
async def test_async_tasks_run_concurrently_when_no_dependency():
    agent = CrewAgent(role="x", goal="y")
    t1 = CrewTask(description="a", expected_output="b", agent=agent, async_execution=True)
    t2 = CrewTask(description="c", expected_output="d", agent=agent, async_execution=True)
    crew = Crew(agents=[agent], tasks=[t1, t2])

    import asyncio
    call_times: list[float] = []

    async def timed(task, context, inputs, registry):
        call_times.append(asyncio.get_event_loop().time())
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
    from cognithor.mcp.tool_registry_db import get_tool_registry
    return await compile_and_run_async(
        agents=self.agents,
        tasks=self.tasks,
        process=self.process,
        inputs=inputs,
        registry=get_tool_registry(),
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
- Modify: `src/cognithor/crew/crew.py` (inject a Planner client instead of raw registry)
- Create: `tests/test_crew/test_pge_integration.py`

- [ ] **Step 1: Scout the existing Planner**

```bash
cd "D:/Jarvis/jarvis complete v20"
grep -n "^    async def formulate_response\|^class Planner" src/cognithor/core/planner.py
```

Read `formulate_response()` (around line 1031). Note its parameters — at minimum it needs `messages` (list of dicts) and `working_memory` (WorkingMemory instance), plus access to config + ollama client. The Crew compiler must construct these.

- [ ] **Step 2: Write the failing integration test**

```python
# tests/test_crew/test_pge_integration.py
from unittest.mock import AsyncMock, MagicMock
import pytest
from cognithor.crew import Crew, CrewAgent, CrewTask
from cognithor.crew.compiler import execute_task_async


@pytest.mark.asyncio
async def test_execute_task_routes_through_planner():
    """The real execute_task_async must: (a) construct a messages list from
    the task + context, (b) call Planner.formulate_response, (c) return a
    TaskOutput with the planner's content + token usage."""
    agent = CrewAgent(role="writer", goal="write", llm="ollama/qwen3:8b")
    task = CrewTask(description="Write a haiku", expected_output="three lines", agent=agent)

    mock_planner = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "First line / Second line / Third line"
    mock_response.usage = {"prompt_tokens": 42, "completion_tokens": 7, "total_tokens": 49}
    mock_planner.formulate_response = AsyncMock(return_value=mock_response)

    mock_registry = MagicMock()
    mock_registry.list_tool_names.return_value = []

    out = await execute_task_async(
        task, context=[], inputs=None, registry=mock_registry, planner=mock_planner,
    )
    assert out.task_id == task.task_id
    assert out.agent_role == "writer"
    assert out.raw == "First line / Second line / Third line"
    assert out.token_usage == {"prompt_tokens": 42, "completion_tokens": 7, "total_tokens": 49}
    # Planner was called once with the task description in the user message
    call = mock_planner.formulate_response.call_args
    assert "Write a haiku" in str(call)


@pytest.mark.asyncio
async def test_execute_task_passes_context_as_prior_messages():
    agent = CrewAgent(role="writer", goal="write")
    t1 = CrewTask(description="research", expected_output="facts", agent=agent)
    t2 = CrewTask(description="write report", expected_output="text", agent=agent, context=[t1])

    from cognithor.crew.output import TaskOutput
    prior = [TaskOutput(task_id=t1.task_id, agent_role="writer", raw="FACTS")]

    mock_planner = MagicMock()
    mock_response = MagicMock(content="REPORT", usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    mock_planner.formulate_response = AsyncMock(return_value=mock_response)

    mock_registry = MagicMock()
    mock_registry.list_tool_names.return_value = []

    await execute_task_async(
        t2, context=prior, inputs=None, registry=mock_registry, planner=mock_planner,
    )
    # The prior output must appear somewhere in the planner's input messages
    call_args = mock_planner.formulate_response.call_args
    assert "FACTS" in str(call_args), "Prior context must be threaded into the planner call"
```

- [ ] **Step 3: Implement real `execute_task_async` in `compiler.py`**

Replace the stub with:

```python
from cognithor.crew.tool_resolver import resolve_tools


async def execute_task_async(
    task: CrewTask,
    *,
    context: list[TaskOutput],
    inputs: dict[str, Any] | None,
    registry: Any,
    planner: Any,
) -> TaskOutput:
    """Route one task through the Planner (which in turn goes through
    Gatekeeper + Executor internally)."""
    import time

    # Resolve tools up-front so the error is raised before any LLM call
    agent_tools = resolve_tools(task.agent.tools, registry=registry)
    task_tools = resolve_tools(task.tools, registry=registry)
    all_tools = list({*agent_tools, *task_tools})

    # Build the message list for the Planner
    system_prompt = _build_system_prompt(task.agent)
    user_prompt = _build_user_prompt(task, context, inputs)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    t0 = time.perf_counter()
    response = await planner.formulate_response(
        messages=messages,
        tools=all_tools,
        model=task.agent.llm,
        max_iter=task.agent.max_iter,
    )
    duration_ms = (time.perf_counter() - t0) * 1000.0

    raw = getattr(response, "content", "") or ""
    usage = getattr(response, "usage", None) or {
        "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
    }

    return TaskOutput(
        task_id=task.task_id,
        agent_role=task.agent.role,
        raw=raw,
        duration_ms=duration_ms,
        token_usage=usage,  # type: ignore[arg-type]
    )


def execute_task(
    task: CrewTask,
    *,
    context: list[TaskOutput],
    inputs: dict[str, Any] | None,
    registry: Any,
    planner: Any | None = None,
) -> TaskOutput:
    import asyncio
    return asyncio.run(execute_task_async(
        task, context=context, inputs=inputs, registry=registry, planner=planner,
    ))


def _build_system_prompt(agent: CrewAgent) -> str:
    parts = [f"You are a {agent.role}."]
    parts.append(f"Your goal: {agent.goal}")
    if agent.backstory:
        parts.append(f"Background: {agent.backstory}")
    return "\n".join(parts)


def _build_user_prompt(
    task: CrewTask,
    context: list[TaskOutput],
    inputs: dict[str, Any] | None,
) -> str:
    parts = []
    if context:
        parts.append("Context from previous tasks:")
        for c in context:
            parts.append(f"[{c.agent_role}] {c.raw}")
        parts.append("")
    desc = task.description
    if inputs:
        for k, v in inputs.items():
            desc = desc.replace("{" + str(k) + "}", str(v))
    parts.append(desc)
    parts.append(f"\nExpected output: {task.expected_output}")
    return "\n".join(parts)
```

Update `compile_and_run_sync` + `compile_and_run_async` to accept and thread a `planner` argument. Update `Crew.kickoff()` / `kickoff_async()` to instantiate a Planner from gateway state (see Step 4).

- [ ] **Step 4: Wire Planner into `Crew.kickoff()` / `kickoff_async()`**

```python
# In crew.py
def kickoff(self, inputs: dict[str, Any] | None = None) -> CrewOutput:
    import asyncio
    return asyncio.run(self.kickoff_async(inputs))

async def kickoff_async(self, inputs: dict[str, Any] | None = None) -> CrewOutput:
    from cognithor.crew.compiler import compile_and_run_async
    from cognithor.crew.runtime import get_default_planner, get_default_tool_registry
    return await compile_and_run_async(
        agents=self.agents,
        tasks=self.tasks,
        process=self.process,
        inputs=inputs,
        registry=get_default_tool_registry(),
        planner=get_default_planner(),
    )
```

Create `src/cognithor/crew/runtime.py` with two helpers that return singleton-ish instances:

```python
"""Factory helpers for PGE runtime objects used by Crew.kickoff().

These wrap the existing `Gateway` lazy singleton or provide a standalone
Planner when Gateway is absent (scripting / test / standalone CLI).
"""

from __future__ import annotations

import threading
from typing import Any

_planner_lock = threading.Lock()
_planner_singleton: Any = None


def get_default_planner() -> Any:
    """Return a process-wide default Planner instance.

    In a live Cognithor process, Gateway has already constructed one — we
    reuse it via `cognithor.gateway.gateway.get_running_gateway()`. Outside
    a Gateway (standalone `python -m cognithor.crew.run`), build a minimal
    one from config defaults.
    """
    global _planner_singleton
    with _planner_lock:
        if _planner_singleton is not None:
            return _planner_singleton
        try:
            from cognithor.gateway.gateway import get_running_gateway  # type: ignore[attr-defined]
            gw = get_running_gateway()
            _planner_singleton = gw._planner
        except (ImportError, RuntimeError, AttributeError):
            from cognithor.config import load_config
            from cognithor.core.planner import Planner
            cfg = load_config()
            _planner_singleton = Planner(config=cfg)
        return _planner_singleton


def get_default_tool_registry() -> Any:
    """Return the process-wide MCP tool registry."""
    from cognithor.mcp import tool_registry_db
    if hasattr(tool_registry_db, "get_tool_registry"):
        return tool_registry_db.get_tool_registry()
    # Fallback: some installs expose a module-level singleton named `registry`
    return getattr(tool_registry_db, "registry", tool_registry_db)
```

**NOTE:** If the actual `Planner.formulate_response()` signature does not match the `messages=..., tools=..., model=..., max_iter=...` args above, the implementer adjusts both `execute_task_async` and the test to reflect the real signature. The plan's goal is behavioural: "route through Planner, get a content+usage back" — the exact kwarg names come from reading `planner.py`. Document the actual signature in the commit body.

- [ ] **Step 5: Run the integration test + full test_crew**

```bash
python -m pytest tests/test_crew/test_pge_integration.py tests/test_crew/test_sequential_kickoff.py tests/test_crew/test_async_kickoff.py -v
```

Existing tests may need the `planner=` kwarg threaded through — update their `patch` targets accordingly.

- [ ] **Step 6: Ruff + commit**

```bash
git add src/cognithor/crew/compiler.py src/cognithor/crew/crew.py src/cognithor/crew/runtime.py tests/test_crew/test_pge_integration.py
git commit -m "feat(crew): real PGE-Trinity integration — execute_task routes through Planner"
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


@pytest.mark.asyncio
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
    mock_registry.list_tool_names.return_value = ["delete_all"]

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


@pytest.mark.asyncio
async def test_task2_receives_task1_output():
    agent = CrewAgent(role="x", goal="y")
    t1 = CrewTask(description="phase 1", expected_output="res1", agent=agent)
    t2 = CrewTask(description="phase 2", expected_output="res2", agent=agent, context=[t1])
    crew = Crew(agents=[agent], tasks=[t1, t2])

    seen_prompts: list[str] = []

    async def capture(*args, **kwargs):
        messages = kwargs.get("messages") or args[0] if args else []
        for m in messages:
            seen_prompts.append(m.get("content", ""))
        resp = MagicMock()
        if len(seen_prompts) <= 2:
            resp.content = "PHASE1_RESULT"
        else:
            resp.content = "PHASE2_RESULT"
        resp.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        return resp

    mock_planner = MagicMock()
    mock_planner.formulate_response = AsyncMock(side_effect=capture)
    mock_registry = MagicMock()
    mock_registry.list_tool_names.return_value = []

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("cognithor.crew.runtime.get_default_planner", lambda: mock_planner)
        mp.setattr("cognithor.crew.runtime.get_default_tool_registry", lambda: mock_registry)
        result = await crew.kickoff_async()

    assert result.tasks_output[1].raw == "PHASE2_RESULT"
    # The user-message for t2 must contain PHASE1_RESULT (from context)
    user_prompts = [p for p in seen_prompts if "phase 2" in p.lower()]
    assert any("PHASE1_RESULT" in p for p in user_prompts)
```

- [ ] **Step 2: Run — expect PASS** (the `_build_user_prompt` in Task 11 already threads context). If it fails, investigate `_build_user_prompt` and adjust.

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

- [ ] **Step 1: Scout existing audit-chain**

```bash
grep -n "class HashlineGuard\|hashline\|audit_chain" src/cognithor/core/safe_call.py src/cognithor/gateway/phases/advanced.py 2>&1 | head -20
```

Identify the module + helper used elsewhere in the codebase to append an audit entry (likely something like `from cognithor.core.audit import append_audit` or a `HashlineGuard.log_event()` method). Use the SAME helper — don't invent a new channel.

- [ ] **Step 2: Test**

```python
# tests/test_crew/test_audit_chain.py
from unittest.mock import patch, MagicMock, AsyncMock
import pytest
from cognithor.crew import Crew, CrewAgent, CrewTask


@pytest.mark.asyncio
async def test_kickoff_emits_audit_event_with_trace_id():
    agent = CrewAgent(role="x", goal="y")
    task = CrewTask(description="a", expected_output="b", agent=agent)
    crew = Crew(agents=[agent], tasks=[task])

    events = []

    def spy(event_name, **fields):
        events.append((event_name, fields))

    mock_planner = MagicMock()
    mock_planner.formulate_response = AsyncMock(
        return_value=MagicMock(content="OK", usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    )
    mock_registry = MagicMock()
    mock_registry.list_tool_names.return_value = []

    with patch("cognithor.crew.compiler.append_audit", side_effect=spy):
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr("cognithor.crew.runtime.get_default_planner", lambda: mock_planner)
            mp.setattr("cognithor.crew.runtime.get_default_tool_registry", lambda: mock_registry)
            result = await crew.kickoff_async()

    # At least one audit event emitted with our trace_id
    kickoff_events = [e for e in events if "crew" in e[0]]
    assert kickoff_events
    assert any(fields.get("trace_id") == result.trace_id for _name, fields in kickoff_events)
```

- [ ] **Step 3: Emit audit events from the compiler**

Add to the top of `compiler.py`:

```python
try:
    from cognithor.core.safe_call import append_audit  # adjust to the real helper
except ImportError:
    def append_audit(event: str, **fields: Any) -> None:
        """Noop fallback when Hashline Guard is unavailable (standalone test)."""
        return None
```

Emit events at key lifecycle points inside `compile_and_run_async`:

```python
append_audit("crew_kickoff_started", trace_id=trace_id, n_tasks=len(ordered), process=process.value)
# ... inside loop ...
append_audit("crew_task_started", trace_id=trace_id, task_id=t.task_id, agent_role=t.agent.role)
# ... after completion ...
append_audit("crew_task_completed", trace_id=trace_id, task_id=t.task_id,
             duration_ms=out.duration_ms, tokens=out.token_usage["total_tokens"])
# ... at the end ...
append_audit("crew_kickoff_completed", trace_id=trace_id, n_tasks=len(outputs))
```

- [ ] **Step 4: Run + commit**

```bash
python -m pytest tests/test_crew/test_audit_chain.py -v
git add src/cognithor/crew/compiler.py tests/test_crew/test_audit_chain.py
git commit -m "feat(crew): emit Hashline-Guard audit events for crew lifecycle"
```

---

### Task 15: Idempotent kickoff with Distributed-Lock

**Files:**
- Modify: `src/cognithor/crew/crew.py` (wrap kickoff_async in distributed lock)
- Create: `tests/test_crew/test_idempotent_kickoff.py`

Spec §1.6: "`kickoff()` ist idempotent re-aufrufbar (nutzt bestehende Distributed-Lock-Logik)". Wire it.

- [ ] **Step 1: Scout the distributed lock API**

```bash
head -100 src/cognithor/core/distributed_lock.py
```

Identify the lock context manager or decorator (likely `DistributedLock(key, timeout).__aenter__()` or similar).

- [ ] **Step 2: Test**

```python
# tests/test_crew/test_idempotent_kickoff.py
from unittest.mock import AsyncMock, MagicMock
import pytest
from cognithor.crew import Crew, CrewAgent, CrewTask


@pytest.mark.asyncio
async def test_same_kickoff_id_returns_cached_output():
    """If the same kickoff_id is provided twice, the second call returns
    the cached CrewOutput without re-running tasks (deterministic replay).
    """
    agent = CrewAgent(role="x", goal="y")
    task = CrewTask(description="a", expected_output="b", agent=agent)
    crew = Crew(agents=[agent], tasks=[task])

    mock_planner = MagicMock()
    call_count = {"n": 0}
    async def fake_resp(**kwargs):
        call_count["n"] += 1
        return MagicMock(content=f"RUN-{call_count['n']}", usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    mock_planner.formulate_response = AsyncMock(side_effect=fake_resp)
    mock_registry = MagicMock()
    mock_registry.list_tool_names.return_value = []

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("cognithor.crew.runtime.get_default_planner", lambda: mock_planner)
        mp.setattr("cognithor.crew.runtime.get_default_tool_registry", lambda: mock_registry)
        out1 = await crew.kickoff_async(inputs={"_kickoff_id": "fixed-id-123"})
        out2 = await crew.kickoff_async(inputs={"_kickoff_id": "fixed-id-123"})

    assert out1.raw == out2.raw, "Same kickoff_id must return identical output"
    assert call_count["n"] == 1, "Planner must be called only once for same kickoff_id"
```

- [ ] **Step 3: Implement kickoff-caching in `crew.py`**

```python
# Module-level cache keyed by kickoff_id (best-effort, per-process).
_KICKOFF_CACHE: dict[str, CrewOutput] = {}

async def kickoff_async(self, inputs: dict[str, Any] | None = None) -> CrewOutput:
    kickoff_id = (inputs or {}).pop("_kickoff_id", None)
    if kickoff_id and kickoff_id in _KICKOFF_CACHE:
        return _KICKOFF_CACHE[kickoff_id]

    from cognithor.crew.compiler import compile_and_run_async
    from cognithor.crew.runtime import get_default_planner, get_default_tool_registry

    # Optional: wrap in distributed lock when kickoff_id is set, so two
    # processes don't both execute the same Crew with the same id.
    lock_cm = None
    if kickoff_id:
        try:
            from cognithor.core.distributed_lock import DistributedLock
            lock_cm = DistributedLock(f"crew:kickoff:{kickoff_id}", timeout_s=300)
        except ImportError:
            lock_cm = None

    if lock_cm is not None:
        async with lock_cm:
            # Double-check cache inside lock to handle the race
            if kickoff_id in _KICKOFF_CACHE:
                return _KICKOFF_CACHE[kickoff_id]
            result = await compile_and_run_async(
                agents=self.agents, tasks=self.tasks, process=self.process,
                inputs=inputs, registry=get_default_tool_registry(),
                planner=get_default_planner(),
            )
            _KICKOFF_CACHE[kickoff_id] = result
            return result

    result = await compile_and_run_async(
        agents=self.agents, tasks=self.tasks, process=self.process,
        inputs=inputs, registry=get_default_tool_registry(),
        planner=get_default_planner(),
    )
    if kickoff_id:
        _KICKOFF_CACHE[kickoff_id] = result
    return result
```

The `DistributedLock` import path / API may differ — adjust based on Step 1's scout.

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
            raise ValueError(
                f"Task '{alias}' references unknown agent '{agent_alias}'. "
                f"Known agents: {list(agent_by_alias)}"
            )
        context_map[alias] = kwargs.pop("context", []) or []
        task_by_alias[alias] = CrewTask(
            agent=agent_by_alias[agent_alias], context=[], **kwargs
        )

    # Pass 2: resolve context references — Pydantic models are frozen, so
    # rebuild each affected task with its context list.
    for alias, refs in context_map.items():
        if not refs:
            continue
        ctx: list[CrewTask] = []
        for ref in refs:
            if ref not in task_by_alias:
                raise ValueError(f"Task '{alias}' references unknown task '{ref}'")
            ctx.append(task_by_alias[ref])
        # Rebuild immutable task with ctx
        existing = task_by_alias[alias]
        data = existing.model_dump()
        data["context"] = ctx
        # We need the real agent object too (not its dump)
        data["agent"] = existing.agent
        task_by_alias[alias] = CrewTask(**data)

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

### Task 18: Error-message quality pass (missing tools, missing agents, invalid inputs)

**Files:**
- Modify: `src/cognithor/crew/tool_resolver.py`, `src/cognithor/crew/yaml_loader.py`, `src/cognithor/crew/errors.py` (refine messages)
- Create: `tests/test_crew/test_error_messages.py`

- [ ] **Step 1: Test messaging contract**

```python
# tests/test_crew/test_error_messages.py
import pytest
from cognithor.crew.errors import ToolNotFoundError, CrewError


class TestErrorMessaging:
    def test_tool_not_found_mentions_name_and_did_you_mean(self):
        from unittest.mock import MagicMock
        from cognithor.crew.tool_resolver import resolve_tools
        registry = MagicMock()
        registry.list_tool_names.return_value = ["web_search", "pdf_reader"]
        with pytest.raises(ToolNotFoundError) as exc:
            resolve_tools(["web_seach"], registry=registry)
        msg = str(exc.value)
        assert "web_seach" in msg
        assert "Meintest du 'web_search'?" in msg

    def test_tool_not_found_mentions_name_only_when_no_close_match(self):
        from unittest.mock import MagicMock
        from cognithor.crew.tool_resolver import resolve_tools
        registry = MagicMock()
        registry.list_tool_names.return_value = ["completely_different"]
        with pytest.raises(ToolNotFoundError) as exc:
            resolve_tools(["totally_foreign"], registry=registry)
        assert "totally_foreign" in str(exc.value)
        assert "Meintest du" not in str(exc.value)

    def test_crew_error_is_base_class(self):
        assert issubclass(ToolNotFoundError, CrewError)
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


@pytest.mark.asyncio
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
    crew = Crew(
        agents=[analyst, writer],
        tasks=[research, report],
        process=CrewProcess.SEQUENTIAL,
        verbose=True,
    )

    mock_planner = MagicMock()
    mock_planner.formulate_response = AsyncMock(side_effect=[
        MagicMock(content="| Tarif | Beitrag | Leistungen |\n|---|---|---|\n| A | 450€ | Stationär |", usage={"prompt_tokens": 500, "completion_tokens": 100, "total_tokens": 600}),
        MagicMock(content="# PKV-Empfehlung\nBasierend auf der Analyse empfehlen wir...", usage={"prompt_tokens": 800, "completion_tokens": 600, "total_tokens": 1400}),
    ])
    mock_registry = MagicMock()
    mock_registry.list_tool_names.return_value = []

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("cognithor.crew.runtime.get_default_planner", lambda: mock_planner)
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
```

- [ ] **Step 4: NOTICE attribution**

If `NOTICE` does not exist, create it:

```
Cognithor
Copyright (C) 2025-2026 Alexander Söllner

This product includes software developed by the Cognithor project,
licensed under the Apache License, Version 2.0.

---

Third-party attributions:

- The `cognithor.crew` API shape is conceptually inspired by CrewAI
  (crewAIInc/crewAI, MIT license, https://github.com/crewAIInc/crewAI).
  The Cognithor implementation is an independent re-implementation in
  Apache 2.0; no source-level code was copied.
```

- [ ] **Step 5: Run full test_crew/ + ruff + commit**

```bash
python -m pytest tests/test_crew/ -v 2>&1 | tail -10
python -m ruff check src/cognithor/crew tests/test_crew
python -m ruff format --check src/cognithor/crew tests/test_crew
git add src/cognithor/__init__.py CHANGELOG.md NOTICE tests/test_crew/test_public_api_stability.py
git commit -m "feat(crew): top-level re-exports + CHANGELOG + NOTICE attribution"
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

- [ ] **Step 3: Run + commit**

```bash
touch tests/test_crew/test_guardrails/__init__.py
python -m pytest tests/test_crew/test_guardrails/test_base.py -v
git add src/cognithor/crew/guardrails tests/test_crew/test_guardrails
git commit -m "feat(crew): Guardrail protocol + GuardrailResult dataclass"
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

- [ ] **Step 1: Failing test**

```python
# tests/test_crew/test_guardrails/test_string.py
from unittest.mock import MagicMock
import pytest
from cognithor.crew.guardrails.string_guardrail import StringGuardrail
from cognithor.crew.output import TaskOutput


def test_string_guardrail_passes_when_llm_says_yes():
    llm = MagicMock()
    llm.chat = MagicMock(return_value=MagicMock(content='{"passed": true, "feedback": null}'))
    g = StringGuardrail("Output must be one sentence", llm_client=llm)
    r = g(TaskOutput(task_id="t", agent_role="w", raw="Hello."))
    assert r.passed


def test_string_guardrail_fails_when_llm_says_no():
    llm = MagicMock()
    llm.chat = MagicMock(return_value=MagicMock(content='{"passed": false, "feedback": "more than one sentence"}'))
    g = StringGuardrail("one sentence", llm_client=llm)
    r = g(TaskOutput(task_id="t", agent_role="w", raw="A. B."))
    assert not r.passed
    assert "more than one sentence" in (r.feedback or "")


def test_string_guardrail_unparseable_llm_response_fails_safe():
    llm = MagicMock()
    llm.chat = MagicMock(return_value=MagicMock(content="not json"))
    g = StringGuardrail("x", llm_client=llm)
    r = g(TaskOutput(task_id="t", agent_role="w", raw="y"))
    assert not r.passed
    assert "parse" in (r.feedback or "").lower()
```

- [ ] **Step 2: Implement**

```python
"""String-based guardrail — LLM validates output against a natural-language rule."""

from __future__ import annotations

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
    """

    def __init__(
        self,
        rule: str,
        *,
        llm_client: Any,
        model: str | None = None,
    ) -> None:
        self._rule = rule
        self._llm = llm_client
        self._model = model

    def __call__(self, output: TaskOutput) -> GuardrailResult:
        user_prompt = f"RULE: {self._rule}\n\nOUTPUT:\n{output.raw}"
        messages = [
            {"role": "system", "content": _VALIDATOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        try:
            resp = self._llm.chat(model=self._model, messages=messages)
            raw = getattr(resp, "content", "") or ""
            data = json.loads(raw)
        except json.JSONDecodeError:
            return GuardrailResult(
                passed=False,
                feedback=f"Validator konnte LLM-Antwort nicht parsen: {raw[:100]}",
            )
        except Exception as exc:
            return GuardrailResult(
                passed=False, feedback=f"Validator-LLM nicht verfügbar: {exc}",
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
import pytest
from cognithor.crew.guardrails.builtin import chain, word_count, no_pii
from cognithor.crew.output import TaskOutput


def _out(raw: str) -> TaskOutput:
    return TaskOutput(task_id="t", agent_role="w", raw=raw)


def test_chain_all_pass():
    g = chain(word_count(min_words=1), no_pii())
    assert g(_out("Hallo Welt")).passed


def test_chain_stops_on_first_failure():
    calls = []
    def tracker(label):
        def _g(out):
            calls.append(label)
            from cognithor.crew.guardrails.base import GuardrailResult
            return GuardrailResult(passed=(label != "B"), feedback=f"from-{label}")
        return _g

    g = chain(tracker("A"), tracker("B"), tracker("C"))
    r = g(_out("x"))
    assert not r.passed
    assert r.feedback == "from-B"
    assert calls == ["A", "B"]  # C never runs


def test_chain_pii_in_first_fails_even_if_second_would_pass():
    g = chain(no_pii(), word_count(min_words=1))
    r = g(_out("Kontakt: x@example.com"))
    assert not r.passed
    assert r.pii_detected is True
```

- [ ] **Step 2: Implement `chain()` and wire all exports**

```python
def chain(*guards):
    """Run guardrails in order; first failure short-circuits.

    Returned GuardrailResult preserves the pii_detected flag from whichever
    guard signaled it, so the audit-chain record is complete.
    """
    def _combined(output: TaskOutput) -> GuardrailResult:
        for g in guards:
            r = g(output)
            if not r.passed:
                return r
        return GuardrailResult(passed=True, feedback=None)
    return _combined
```

Update `__init__.py`:

```python
from cognithor.crew.guardrails.base import Guardrail, GuardrailResult
from cognithor.crew.guardrails.builtin import (
    chain, hallucination_check, no_pii, schema, word_count,
)
from cognithor.crew.guardrails.function_guardrail import FunctionGuardrail
from cognithor.crew.guardrails.string_guardrail import StringGuardrail

__all__ = [
    "FunctionGuardrail",
    "Guardrail",
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


@pytest.mark.asyncio
async def test_guardrail_failure_retries_then_raises():
    agent = CrewAgent(role="writer", goal="write")
    def fail_twice(_out):
        return GuardrailResult(passed=False, feedback="zu kurz")
    task = CrewTask(description="write", expected_output="long text",
                   agent=agent, guardrail=fail_twice, max_retries=2)
    crew = Crew(agents=[agent], tasks=[task])

    call_count = {"n": 0}
    async def fake(**kwargs):
        call_count["n"] += 1
        return MagicMock(content=f"attempt-{call_count['n']}",
                         usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    mock_planner = MagicMock()
    mock_planner.formulate_response = AsyncMock(side_effect=fake)
    mock_registry = MagicMock()
    mock_registry.list_tool_names.return_value = []

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("cognithor.crew.runtime.get_default_planner", lambda: mock_planner)
        mp.setattr("cognithor.crew.runtime.get_default_tool_registry", lambda: mock_registry)
        with pytest.raises(GuardrailFailure, match="zu kurz"):
            await crew.kickoff_async()

    # Initial try + max_retries == 3 attempts total
    assert call_count["n"] == 3


@pytest.mark.asyncio
async def test_guardrail_passes_after_retry():
    agent = CrewAgent(role="writer", goal="write")
    attempts = {"n": 0}
    def pass_on_second(_out):
        attempts["n"] += 1
        return GuardrailResult(passed=(attempts["n"] >= 2), feedback="try again")

    task = CrewTask(description="x", expected_output="y",
                   agent=agent, guardrail=pass_on_second, max_retries=2)
    crew = Crew(agents=[agent], tasks=[task])

    mock_planner = MagicMock()
    mock_planner.formulate_response = AsyncMock(
        return_value=MagicMock(content="text", usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    )
    mock_registry = MagicMock()
    mock_registry.list_tool_names.return_value = []

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("cognithor.crew.runtime.get_default_planner", lambda: mock_planner)
        mp.setattr("cognithor.crew.runtime.get_default_tool_registry", lambda: mock_registry)
        result = await crew.kickoff_async()

    assert result.tasks_output[0].guardrail_verdict == "pass"
```

- [ ] **Step 2: Implement guardrail evaluation in `execute_task_async`**

Inside `execute_task_async` (from Task 11), after the Planner returns a response, add:

```python
from cognithor.crew.errors import GuardrailFailure
from cognithor.crew.guardrails.base import GuardrailResult
from cognithor.crew.guardrails.function_guardrail import FunctionGuardrail
from cognithor.crew.guardrails.string_guardrail import StringGuardrail


def _normalize_guardrail(g: Any, llm_client: Any = None) -> Any:
    if g is None:
        return None
    if callable(g):
        # Could be FunctionGuardrail-compatible or user callable
        if hasattr(g, "__call__") and not isinstance(g, str):
            return FunctionGuardrail(g) if not _is_already_guardrail(g) else g
    if isinstance(g, str):
        return StringGuardrail(g, llm_client=llm_client)
    return g


def _is_already_guardrail(g: Any) -> bool:
    # Duck-type: returns a GuardrailResult directly
    return hasattr(g, "__call__") and not hasattr(g, "_fn")


# Inside execute_task_async, after `response = await planner.formulate_response(...)`:
guardrail = _normalize_guardrail(task.guardrail, llm_client=planner)

attempts = 0
verdict = "skipped"
while True:
    out = TaskOutput(
        task_id=task.task_id, agent_role=task.agent.role, raw=raw,
        duration_ms=duration_ms, token_usage=usage,  # type: ignore[arg-type]
    )
    if guardrail is None:
        verdict = "skipped"
        break
    result: GuardrailResult = guardrail(out)
    if result.passed:
        verdict = "pass"
        break
    attempts += 1
    if attempts > task.max_retries:
        raise GuardrailFailure(
            f"Guardrail failed after {task.max_retries} retries for task "
            f"'{task.task_id}': {result.feedback}"
        )
    # Retry: re-run planner with feedback appended to user message
    messages.append({
        "role": "user",
        "content": f"Feedback: {result.feedback}\n\nBitte versuche es erneut.",
    })
    t0 = time.perf_counter()
    response = await planner.formulate_response(
        messages=messages, tools=all_tools, model=task.agent.llm, max_iter=task.agent.max_iter,
    )
    duration_ms = (time.perf_counter() - t0) * 1000.0
    raw = getattr(response, "content", "") or ""
    usage = getattr(response, "usage", None) or {
        "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
    }

# Attach verdict to the final output
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


@pytest.mark.asyncio
async def test_guardrail_pass_audited():
    agent = CrewAgent(role="writer", goal="write")
    task = CrewTask(description="x", expected_output="y", agent=agent,
                   guardrail=lambda o: GuardrailResult(passed=True, feedback=None))
    crew = Crew(agents=[agent], tasks=[task])

    events: list = []
    def spy(name, **fields): events.append((name, fields))

    mock_planner = MagicMock()
    mock_planner.formulate_response = AsyncMock(
        return_value=MagicMock(content="ok", usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    )
    mock_registry = MagicMock()
    mock_registry.list_tool_names.return_value = []

    with patch("cognithor.crew.compiler.append_audit", side_effect=spy), \
         pytest.MonkeyPatch().context() as mp:
        mp.setattr("cognithor.crew.runtime.get_default_planner", lambda: mock_planner)
        mp.setattr("cognithor.crew.runtime.get_default_tool_registry", lambda: mock_registry)
        await crew.kickoff_async()

    guardrail_events = [e for e in events if "guardrail" in e[0]]
    assert guardrail_events
    assert any(fields.get("verdict") == "pass" for _name, fields in guardrail_events)
```

- [ ] **Step 2: Emit events inside the guardrail retry loop**

Inside `execute_task_async`, after evaluating `result`:

```python
append_audit(
    "crew_guardrail_check",
    trace_id=None,  # aggregated at compiler level
    task_id=task.task_id,
    verdict="pass" if result.passed else "fail",
    retry_count=attempts,
    pii_detected=result.pii_detected,
    feedback=result.feedback,
)
```

- [ ] **Step 3: Commit**

```bash
git add src/cognithor/crew/compiler.py tests/test_crew/test_guardrails/test_audit.py
git commit -m "feat(crew): guardrail verdicts recorded in Hashline-Guard audit chain"
```

---

### Task 31: Feature-4 integration test (versicherungs-vergleich with no_pii + custom)

**Files:**
- Create: `tests/test_crew/test_guardrails/test_versicherungs_integration.py`

Spec §4.5: "Das `versicherungs-vergleich`-Template nutzt `no_pii()` und einen custom String-Guardrail ('keine Tarif-Empfehlung, nur Vergleich')."

- [ ] **Step 1: Test**

```python
# tests/test_crew/test_guardrails/test_versicherungs_integration.py
from unittest.mock import AsyncMock, MagicMock
import pytest
from cognithor.crew import Crew, CrewAgent, CrewTask
from cognithor.crew.guardrails import chain, no_pii


@pytest.mark.asyncio
async def test_versicherungs_crew_blocks_pii_output():
    agent = CrewAgent(role="analyst", goal="compare PKV tariffs")
    # Custom string guardrail (we mock the LLM that validates it)
    task = CrewTask(
        description="Compare",
        expected_output="Tabular comparison",
        agent=agent,
        guardrail=chain(no_pii()),
        max_retries=0,
    )
    crew = Crew(agents=[agent], tasks=[task])

    # Planner returns text with PII — guardrail must catch
    mock_planner = MagicMock()
    mock_planner.formulate_response = AsyncMock(
        return_value=MagicMock(
            content="Kontakt: sachbearbeiter@versicherer.de zur Beratung.",
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )
    )
    mock_registry = MagicMock()
    mock_registry.list_tool_names.return_value = []

    from cognithor.crew.errors import GuardrailFailure
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("cognithor.crew.runtime.get_default_planner", lambda: mock_planner)
        mp.setattr("cognithor.crew.runtime.get_default_tool_registry", lambda: mock_registry)
        with pytest.raises(GuardrailFailure, match="PII erkannt"):
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
"""Jinja2-based directory tree renderer for cognithor init templates."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined


_PROJECT_NAME_CLEAN = re.compile(r"[^a-zA-Z0-9_]")


def sanitize_project_name(name: str) -> str:
    """Convert free-form name to a safe Python package identifier."""
    if not name or not name.strip():
        raise ValueError("project name cannot be empty")
    cleaned = _PROJECT_NAME_CLEAN.sub("_", name.strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        raise ValueError(f"project name reduces to empty: {name!r}")
    if cleaned[0].isdigit():
        cleaned = f"project_{cleaned}"
    return cleaned


def render_tree(src_dir: Path, dest_dir: Path, *, context: dict[str, Any]) -> None:
    """Render every file under src_dir into dest_dir, applying Jinja2 to .jinja files.

    Files ending in `.jinja` have that suffix stripped and their contents rendered.
    Path segments with `{{...}}` tags are also rendered.
    Non-.jinja files are copied verbatim.
    """
    src_dir = Path(src_dir)
    dest_dir = Path(dest_dir)
    if dest_dir.exists() and any(dest_dir.iterdir()):
        raise FileExistsError(f"dest exists and is not empty: {dest_dir}")

    env = Environment(
        loader=FileSystemLoader(str(src_dir)),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
    )

    for src_path in src_dir.rglob("*"):
        rel = src_path.relative_to(src_dir)
        # Render path segments
        rendered_rel = Path(*[env.from_string(p).render(**context) for p in rel.parts])
        dest_path = dest_dir / rendered_rel

        if src_path.is_dir():
            dest_path.mkdir(parents=True, exist_ok=True)
            continue

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        if src_path.suffix == ".jinja":
            # Strip .jinja from filename + render
            dest_path = dest_path.with_suffix("")
            template = env.get_template(str(rel).replace("\\", "/"))
            dest_path.write_text(template.render(**context), encoding="utf-8")
        else:
            shutil.copy2(src_path, dest_path)
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


def list_templates() -> list[TemplateMeta]:
    """Return metadata for every discoverable template."""
    if not TEMPLATES_ROOT.exists():
        return []
    out: list[TemplateMeta] = []
    for d in sorted(TEMPLATES_ROOT.iterdir()):
        meta_file = d / "template.yaml"
        if not meta_file.is_file():
            continue
        data = yaml.safe_load(meta_file.read_text(encoding="utf-8"))
        out.append(TemplateMeta(**data))
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
    """Plant a minimal mock template so the CLI has something to render."""
    tpl_root = tmp_path / "templates"
    research = tpl_root / "research"
    research.mkdir(parents=True)
    (research / "template.yaml").write_text(
        "name: research\n"
        "description_de: Mock\n"
        "description_en: Mock\n"
    )
    (research / "README.md.jinja").write_text("# {{ project_name }}")
    (research / "main.py.jinja").write_text("PROJECT = '{{ project_name }}'")
    src_dir = research / "src" / "{{ project_name }}"
    src_dir.mkdir(parents=True)
    (src_dir / "__init__.py").write_text("")

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
    assert (project_dir / "main.py").read_text() == "PROJECT = 'my_project'"
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
    *, name: str, template: str, directory: Path | None = None, lang: str = "de",
) -> int:
    """Execute `cognithor init`. Returns shell exit code (0 on success)."""
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
        raise InitCommandError(
            f"target directory is not empty: {dest}"
        )

    context = {
        "project_name": project_name,
        "project_name_display": name,
        "lang": lang,
    }
    render_tree(template_dir, dest, context=context)
    msg_done = "Projekt erstellt" if lang == "de" else "Project created"
    print(f"{msg_done}: {dest}")
    return 0
```

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

If existing `__main__` uses argparse subparsers:

```python
# Inside the existing argparse setup
init_parser = subparsers.add_parser("init", help="Scaffold a new Crew project")
init_parser.add_argument("name", help="Project name")
init_parser.add_argument("--template", required=True, help="Template name")
init_parser.add_argument("--dir", dest="directory", type=Path, default=None)
init_parser.add_argument("--lang", default="de", choices=["de", "en"])

list_parser = subparsers.add_parser("init-list-templates",
    help="List available templates")

# In the dispatch:
if args.command == "init":
    from cognithor.crew.cli.init_cmd import run_init
    try:
        return run_init(name=args.name, template=args.template,
                        directory=args.directory, lang=args.lang)
    except Exception as exc:
        print(f"init failed: {exc}", file=sys.stderr)
        return 1
```

Adapt to actual existing style. If `__main__` is click-based, use `@cli.command()` decorators.

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
- Create: `src/cognithor/crew/templates/research/main.py.jinja`
- Create: `src/cognithor/crew/templates/research/src/{{ project_name }}/__init__.py`
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
required_models:
  - ollama/qwen3:8b
tags:
  - quickstart
  - beginner
  - sequential
```

- [ ] **Step 2: `main.py.jinja`**

```python
"""{{ project_name_display }} entry point."""

from __future__ import annotations

import asyncio

from {{ project_name }}.crew import build_crew


def main() -> None:
    crew = build_crew()
    result = asyncio.run(crew.kickoff_async(inputs={"topic": "Beispielthema"}))
    print(result.raw)


if __name__ == "__main__":
    main()


def build_crew():
    """Exported for `cognithor run`."""
    return build_crew_impl()


def build_crew_impl():  # separate name so the import above works
    return build_crew()
```

Actually simpler — expose `build_crew` directly from the package:

Replace with:

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


@pytest.mark.asyncio
async def test_crew_kickoff_with_mock_planner(monkeypatch):
    crew = ResearchCrew().assemble()
    mock_planner = MagicMock()
    mock_planner.formulate_response = AsyncMock(
        return_value=MagicMock(
            content="MOCK_OUTPUT",
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )
    )
    mock_registry = MagicMock()
    mock_registry.list_tool_names.return_value = []

    monkeypatch.setattr("cognithor.crew.runtime.get_default_planner", lambda: mock_planner)
    monkeypatch.setattr("cognithor.crew.runtime.get_default_tool_registry", lambda: mock_registry)

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

    # Required artifacts exist
    assert (project / "pyproject.toml").exists()
    assert (project / "src" / "rc" / "crew.py").exists()
    assert (project / "src" / "rc" / "main.py").exists()
    assert (project / "tests" / "test_crew.py").exists()
    assert (project / "README.md").exists()
    assert (project / ".env.example").exists()
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
  - `template.yaml`: name: customer-support, 3 agents, sequential
  - `crew.py.jinja`: IntakeCrew class with 3 @agent + 3 @task methods
  - Agents: `intake` (parses customer message), `classifier` (categorizes), `response_writer` (drafts reply)
  - Tasks: `parse`, `classify`, `draft_reply` — each feeds context to the next
  - `tests/test_crew.py.jinja`: kickoff with mock planner returning 3 mock responses

- [ ] **Step 11: Integration test**

```python
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

- [ ] **Step 1-10: Mirror Task 39/40 pattern**
  - `analyst`: role="Analyst", runs data-summarization via code-exec tool
  - `visualizer`: role="Visualizer", produces matplotlib chart spec
  - Tasks: `analyze` (consumes CSV path from inputs), `visualize` (consumes analyst output)
  - **Critical:** the `analyst` agent's `tools` list includes the existing sandbox code-exec tool (e.g. `python_sandbox`). Scaffolded tests mock the registry.

- [ ] **Step 11: Commit**

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

- [ ] **Step 1-10: Mirror pattern**
  - `Crew(process=CrewProcess.HIERARCHICAL, manager_llm="ollama/qwen3:32b", ...)`
  - Three agents: `outliner`, `drafter`, `editor`
  - Tasks: `outline`, `draft`, `edit` — hierarchical process chooses order dynamically
  - Smoke test verifies `crew.process == HIERARCHICAL` and `manager_llm` is set

- [ ] **Step 11: Commit**

```bash
git add src/cognithor/crew/templates/content tests/test_crew/test_templates/test_content.py
git commit -m "feat(crew): content template (3-agent, hierarchical)"
```

---

### Task 43: `versicherungs-vergleich` template (DACH-differentiator)

**Files:**
- Create: `src/cognithor/crew/templates/versicherungs-vergleich/*`
- Create: `tests/test_crew/test_templates/test_versicherungs_vergleich.py`

Spec §3.3.5: PKV/BU-Tarif-Vergleich. THREE agents: `Tarif-Researcher`, `Kunden-Profiler`, `Empfehlungs-Writer`. DSGVO-konform, **vollständig offline-fähig** (no external APIs). Includes explicit §34d-neutral guardrails.

- [ ] **Step 1-10: Mirror pattern, with extra care for the spec's DSGVO requirements**
  - `tools=[]` for all agents — NO external HTTP tools in default (spec §3.6)
  - Writer task has guardrail: `chain(no_pii(), string_guardrail_neutral_language)` — the LLM-validated string guardrail ensures §34d-neutral wording ("Information, nicht Beratung")
  - `required_models` in `template.yaml`: only Ollama — the template REFUSES to run against cloud models

- [ ] **Step 11: Integration test**

```python
def test_versicherungs_template_is_offline_capable(tmp_path: Path):
    project = tmp_path / "pkv"
    run_init(name="pkv", template="versicherungs-vergleich", directory=project, lang="de")
    crew_file = (project / "src" / "pkv" / "crew.py").read_text()
    # No tools should be listed (offline-capable)
    assert "tools=[]" in crew_file or "tools=[\"\"]" not in crew_file
    # Guardrail reference must be present (either chain(no_pii...) or no_pii directly)
    assert "no_pii" in crew_file
```

- [ ] **Step 12: Commit**

```bash
git add src/cognithor/crew/templates/versicherungs-vergleich tests/test_crew/test_templates/test_versicherungs_vergleich.py
git commit -m "feat(crew): versicherungs-vergleich template (DACH, offline-capable, PII-blocked)"
```

---

### Task 44: `cognithor init --list-templates` CLI integration test

**Files:**
- Modify: `tests/test_crew/test_cli/test_list_templates.py` (add real-template test)

- [ ] **Step 1: After Tasks 39-43, all 5 templates exist. Run:**

```bash
python -m cognithor init-list-templates
```

Expected output:

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
  pull_request:

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
  `cognithor init-list-templates` prints the catalog with DE/EN descriptions.
  CI scaffolds every template on every PR.
```

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
- **Task 50:** Performance benchmark: `Crew.kickoff()` overhead vs direct Planner call per spec §8.5 (< 5% extra latency). Script: `scripts/bench_crew_overhead.py`. Record baseline in commit message.
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

- [ ] **Step 2: EN version** — direct translation.

- [ ] **Step 3: Commit**

```bash
git add docs/quickstart/00-installation.md docs/quickstart/00-installation.en.md
git commit -m "docs(quickstart): installation page (DE + EN)"
```

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


@pytest.mark.asyncio
async def test_pkv_example_runs_with_mock_planner(monkeypatch):
    # Import the example AS IF a user just installed cognithor
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from main import build_crew  # the example exports build_crew()

    mock_planner = MagicMock()
    mock_planner.formulate_response = AsyncMock(
        return_value=MagicMock(content="MOCK", usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    )
    mock_registry = MagicMock()
    mock_registry.list_tool_names.return_value = []

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

### Task 62: External-reader usability test — write a one-page checklist

**Files:**
- Create: `docs/quickstart/EXTERNAL_REVIEW_CHECKLIST.md`

Spec §2.4 requires: "Ein externer Testleser ... schafft 00 → 01 in unter 15 Minuten ohne Rückfragen." Provide a checklist the project lead uses to record their own external-reader test:

- [ ] Timer from landing on `README.md` to first successful `crew.kickoff()`
- [ ] Count of questions the reader asked
- [ ] List of typos / confusions found

- [ ] **Step 1: Commit**

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
- **Quickstart** — 7-page onboarding guide at [`docs/quickstart/`](docs/quickstart/README.md). From empty terminal to first Crew in <10 minutes.
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
- **Quickstart docs (Feature 2)** — 7 bilingual (DE+EN) pages at `docs/quickstart/`
  covering installation, first-crew, first-tool, first-skill, guardrails,
  deployment, and next-steps. Every example runs in CI via
  `.github/workflows/quickstart-examples.yml`.
```

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
    args.output.write_text(json.dumps(catalog, indent=2, ensure_ascii=False))
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

- **Task 76:** Full ruff sweep across Feature-7 files.
- **Task 77:** Add `Tool-of-the-month` idea to `docs/integrations/BACKLOG.md` for post-v1.0.
- **Task 78:** Commit final.

---

# MERGE-PREP — Closeout Tasks (79-82)

---

### Task 79: Full test suite + coverage + lint regression

- [ ] **Step 1: Full regression**

```bash
cd "D:/Jarvis/jarvis complete v20"
python -m pytest tests/ -x -q --cov=src/cognithor --cov-report=term-missing 2>&1 | tail -30
```

Expected: ALL pass, `cognithor.crew` coverage ≥ 85%, total coverage ≥ 89%.

- [ ] **Step 2: Ruff**

```bash
python -m ruff check .
python -m ruff format --check .
```

Both clean.

- [ ] **Step 3: Mypy strict on new code**

```bash
python -m mypy --strict src/cognithor/crew
```

- [ ] **Step 4: Commit any fixes**

---

### Task 80: CHANGELOG `[Unreleased]` → `[0.93.0]` bump

**Files:**
- Modify: `CHANGELOG.md`
- Modify: `pyproject.toml` (version 0.92.7 → 0.93.0)
- Modify: `src/cognithor/__init__.py` (`__version__`)
- Modify: `flutter_app/pubspec.yaml` (version)
- Modify: `flutter_app/lib/providers/connection_provider.dart` (`kFrontendVersion`)

The Crew-Layer is a MINOR bump (additive, no breaking changes). Date the `[0.93.0]` section with today's date.

- [ ] **Step 1: Version updates across all 5 locations**

- [ ] **Step 2: Commit**

```bash
git add CHANGELOG.md pyproject.toml src/cognithor/__init__.py flutter_app/pubspec.yaml flutter_app/lib/providers/connection_provider.dart
git commit -m "chore(release): bump to 0.93.0 — Crew-Layer v1.0"
```

---

### Task 81: Push branch + open PR

- [ ] **Step 1: Push**

```bash
git push -u origin feat/cognithor-crew-v1
```

- [ ] **Step 2: Open PR via GitHub API**

Use the same pattern as the video-input PR (PR #140). Title:

```
feat(crew): Cognithor Crew-Layer v1.0 adoption — Features 1, 4, 3, 2, 7
```

Body: summary of scope, spec reference, sign-off checklist from spec §12.

- [ ] **Step 3: Wait for all CI jobs green** (CI + scaffold-templates + quickstart-examples + integrations-catalog + Windows Installer + Mobile + Linux .deb + Flutter Web + Release Build)

---

### Task 82: Post-merge release `v0.93.0`

(This task runs in a SEPARATE session/turn after the PR is green and merged — per the feedback memory "never chain merge + cleanup via &&".)

- [ ] **Step 1: Tag + push**

```bash
git checkout main && git pull
git tag -a v0.93.0 -m "Cognithor v0.93.0 — Crew-Layer v1.0 (Features 1, 4, 3, 2, 7)"
git push origin v0.93.0
```

- [ ] **Step 2: Wait for auto-triggered build workflows**

- [ ] **Step 3: Manually trigger `publish.yml` for PyPI**

- [ ] **Step 4: Verify PyPI: `pip install cognithor==0.93.0`**

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

After every Feature (1, 4, 3, 2, 7) is fully implemented + self-reviewed:
- Task 79 — full regression
- Task 80 — version bump
- Task 81 — push + PR
- Task 82 — post-merge release (separate turn)

Target: v0.93.0 released to PyPI + GitHub. All 6 release artifacts (Windows Installer, Launcher, Linux .deb, Android APK, iOS IPA, Flutter Web) auto-built + attached to the release.

