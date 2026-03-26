# Agent Model & Temperature Wiring

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire agent-specific `preferred_model`, `temperature`, and new `top_p` field through the PGE loop so each agent actually uses its configured LLM settings at runtime.

**Architecture:** Add `top_p` to `AgentProfile`. Extract agent overrides in `_run_pge_loop` and pass them to `planner.plan()` / `planner.replan()` as optional kwargs. Planner applies overrides to model selection and LLM call parameters. Update all 5 agents with optimal values.

**Tech Stack:** Python 3.12+ (dataclasses, Pydantic), pytest

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `src/jarvis/core/agent_router.py:81` | Add `top_p` field to AgentProfile |
| Modify | `src/jarvis/core/planner.py:368-620` | Accept and apply agent overrides in plan/replan |
| Modify | `src/jarvis/gateway/gateway.py:1865-1979` | Extract overrides from route_decision, pass to planner |
| Modify | `src/jarvis/channels/config_routes.py` | Expose `top_p` in agent API |
| Modify | `scripts/setup_agents.py` | Update agents with optimal values |
| Create | `tests/unit/test_agent_overrides.py` | Tests for the entire override chain |

---

### Task 1: Add `top_p` to AgentProfile

**Files:**
- Modify: `src/jarvis/core/agent_router.py:81`

- [ ] **Step 1: Add `top_p` field after `temperature`**

In `src/jarvis/core/agent_router.py`, line 81, after `temperature: float | None = None`:

```python
    top_p: float | None = None  # None = Default des Modells
```

- [ ] **Step 2: Verify import works**

Run: `cd "D:\Jarvis\jarvis complete v20" && python -c "from jarvis.core.agent_router import AgentProfile; a = AgentProfile(name='test', top_p=0.85); print(a.top_p)"`
Expected: `0.85`

- [ ] **Step 3: Commit**

```bash
git add src/jarvis/core/agent_router.py
git commit -m "feat: add top_p field to AgentProfile"
```

---

### Task 2: Planner accepts agent overrides

**Files:**
- Modify: `src/jarvis/core/planner.py:368-620`
- Create: `tests/unit/test_agent_overrides.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_agent_overrides.py`:

```python
"""Tests for agent-specific model/temperature/top_p overrides in the Planner."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestPlannerAgentOverrides:
    """Planner.plan() and Planner.replan() must accept and apply agent overrides."""

    @pytest.fixture
    def mock_planner(self):
        """Create a minimal Planner with mocked dependencies."""
        from jarvis.config import JarvisConfig
        from jarvis.core.planner import Planner

        config = JarvisConfig()
        planner = Planner(config)
        # Mock LLM to avoid real Ollama calls
        planner._ollama = MagicMock()
        planner._ollama.chat = AsyncMock(return_value={
            "message": {"content": "Test response"},
        })
        # Mock circuit breaker to pass through
        planner._llm_circuit_breaker = MagicMock()
        planner._llm_circuit_breaker.call = AsyncMock(
            side_effect=lambda coro: coro
        )
        return planner

    @pytest.fixture
    def empty_wm(self):
        from jarvis.models import WorkingMemory
        return WorkingMemory()

    @pytest.mark.asyncio
    async def test_plan_uses_default_model_without_override(self, mock_planner, empty_wm):
        """Without overrides, plan() uses the default planner model."""
        await mock_planner.plan("hello", empty_wm, {})
        call_kwargs = mock_planner._ollama.chat.call_args
        # Default model should be from config (qwen3:32b)
        assert "qwen3:32b" in str(call_kwargs)

    @pytest.mark.asyncio
    async def test_plan_uses_agent_model_override(self, mock_planner, empty_wm):
        """With model_override, plan() uses the specified model."""
        await mock_planner.plan(
            "hello", empty_wm, {},
            model_override="qwen3-coder:30b",
        )
        call_kwargs = mock_planner._ollama.chat.call_args
        assert call_kwargs.kwargs.get("model") == "qwen3-coder:30b" or \
               call_kwargs.args[0] if call_kwargs.args else False

    @pytest.mark.asyncio
    async def test_plan_uses_agent_temperature_override(self, mock_planner, empty_wm):
        """With temperature_override, plan() uses the specified temperature."""
        await mock_planner.plan(
            "hello", empty_wm, {},
            temperature_override=0.2,
        )
        call_kwargs = mock_planner._ollama.chat.call_args
        assert call_kwargs.kwargs.get("temperature") == 0.2

    @pytest.mark.asyncio
    async def test_plan_uses_agent_top_p_override(self, mock_planner, empty_wm):
        """With top_p_override, plan() passes it in the LLM call."""
        await mock_planner.plan(
            "hello", empty_wm, {},
            top_p_override=0.85,
        )
        call_kwargs = mock_planner._ollama.chat.call_args
        assert call_kwargs.kwargs.get("top_p") == 0.85
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "D:\Jarvis\jarvis complete v20" && python -m pytest tests/unit/test_agent_overrides.py -v`
Expected: FAIL — `plan() got an unexpected keyword argument 'model_override'`

- [ ] **Step 3: Add override parameters to plan() and replan()**

In `src/jarvis/core/planner.py`, modify the `plan()` signature (line 368):

```python
    async def plan(
        self,
        user_message: str,
        working_memory: WorkingMemory,
        tool_schemas: dict[str, Any],
        *,
        model_override: str | None = None,
        temperature_override: float | None = None,
        top_p_override: float | None = None,
    ) -> ActionPlan:
```

Then in the body (around line 384), change model selection:

```python
        if model_override:
            model = model_override
        else:
            model = self._router.select_model("planning", "high")
        model_config = self._router.get_model_config(model)
```

And in the LLM call (around line 409), apply temperature/top_p overrides:

```python
                    temperature=temperature_override if temperature_override is not None else model_config.get("temperature", 0.7),
                    top_p=top_p_override if top_p_override is not None else model_config.get("top_p", 0.9),
```

Do the same for the retry block (~line 509) — apply the same overrides.

Apply the **identical changes** to `replan()` (line 525):

```python
    async def replan(
        self,
        original_goal: str,
        results: list[ToolResult],
        working_memory: WorkingMemory,
        tool_schemas: dict[str, Any],
        *,
        model_override: str | None = None,
        temperature_override: float | None = None,
        top_p_override: float | None = None,
    ) -> ActionPlan:
```

And its model selection (~line 548):

```python
        if model_override:
            model = model_override
        else:
            model = self._router.select_model("planning", "high")
        model_config = self._router.get_model_config(model)
```

And its LLM call (~line 567) and retry (~line 615): same temperature/top_p override pattern.

- [ ] **Step 4: Run tests**

Run: `cd "D:\Jarvis\jarvis complete v20" && python -m pytest tests/unit/test_agent_overrides.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/jarvis/core/planner.py tests/unit/test_agent_overrides.py
git commit -m "feat: planner accepts model_override, temperature_override, top_p_override"
```

---

### Task 3: Gateway passes agent overrides to Planner

**Files:**
- Modify: `src/jarvis/gateway/gateway.py:1865-1979`

- [ ] **Step 1: Extract overrides at the top of `_run_pge_loop`**

In `src/jarvis/gateway/gateway.py`, in the `_run_pge_loop` method (after line 1888), add:

```python
        # Agent-specific LLM overrides
        _agent_model = None
        _agent_temperature = None
        _agent_top_p = None
        if route_decision and route_decision.agent:
            _agent = route_decision.agent
            if _agent.preferred_model:
                _agent_model = _agent.preferred_model
            if _agent.temperature is not None:
                _agent_temperature = _agent.temperature
            if getattr(_agent, "top_p", None) is not None:
                _agent_top_p = _agent.top_p
```

- [ ] **Step 2: Pass overrides to plan() call (line ~1968)**

Change:
```python
                plan = await self._planner.plan(
                    user_message=msg.text,
                    working_memory=wm,
                    tool_schemas=tool_schemas,
                )
```

To:
```python
                plan = await self._planner.plan(
                    user_message=msg.text,
                    working_memory=wm,
                    tool_schemas=tool_schemas,
                    model_override=_agent_model,
                    temperature_override=_agent_temperature,
                    top_p_override=_agent_top_p,
                )
```

- [ ] **Step 3: Pass overrides to replan() call (line ~1974)**

Change:
```python
                plan = await self._planner.replan(
                    original_goal=msg.text,
                    results=all_results,
                    working_memory=wm,
                    tool_schemas=tool_schemas,
                )
```

To:
```python
                plan = await self._planner.replan(
                    original_goal=msg.text,
                    results=all_results,
                    working_memory=wm,
                    tool_schemas=tool_schemas,
                    model_override=_agent_model,
                    temperature_override=_agent_temperature,
                    top_p_override=_agent_top_p,
                )
```

- [ ] **Step 4: Verify syntax**

Run: `cd "D:\Jarvis\jarvis complete v20" && python -c "from jarvis.gateway.gateway import Gateway; print('OK')"`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add src/jarvis/gateway/gateway.py
git commit -m "feat: gateway passes agent model/temperature/top_p overrides to planner"
```

---

### Task 4: Expose top_p in agent API

**Files:**
- Modify: `src/jarvis/channels/config_routes.py`

- [ ] **Step 1: Find the agent serialization code**

Search for where agent fields are serialized (around line 254, the agent detail endpoint). Add `"top_p"` to the serialized fields wherever `"temperature"` appears.

Also find the agent creation/update handlers and ensure `top_p` is read from the request body and applied to the agent profile.

- [ ] **Step 2: Add top_p to agent response serialization**

Where you see `"temperature": getattr(agent_obj, "temperature", 0.7)`, add nearby:
```python
"top_p": getattr(agent_obj, "top_p", None),
```

- [ ] **Step 3: Add top_p to agent creation/update handling**

Where `temperature` is read from `body.get("temperature", ...)`, add:
```python
"top_p": body.get("top_p"),
```

- [ ] **Step 4: Verify endpoint**

Run: `cd "D:\Jarvis\jarvis complete v20" && python -c "from jarvis.channels.config_routes import create_config_routes; print('OK')"`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add src/jarvis/channels/config_routes.py
git commit -m "feat: expose top_p field in agent CRUD API"
```

---

### Task 5: Update agents with optimal parameters

**Files:**
- Modify: `scripts/setup_agents.py`

- [ ] **Step 1: Update all agent definitions with optimized values**

Update the AGENTS list in `scripts/setup_agents.py`:

**jarvis** (update via API):
- No model override (uses default planner)
- temperature: 0.7 (keep default — generalist needs flexibility)
- top_p: 0.9 (keep default)

**researcher**:
- preferred_model: "" (use default qwen3:32b — needs strong reasoning)
- temperature: 0.3 (precision over creativity for factual queries)
- top_p: 0.85 (slightly restricted for more focused outputs)

**coder**:
- preferred_model: "qwen3-coder:30b" (specialized code model)
- temperature: 0.2 (code must be deterministic)
- top_p: 0.8 (tight distribution — fewer hallucinated APIs)

**office**:
- preferred_model: "qwen3:8b" (fast model, simple tasks)
- temperature: 0.5 (emails need some natural variation)
- top_p: 0.9 (standard)

**operator**:
- preferred_model: "qwen3:8b" (fast model for direct commands)
- temperature: 0.2 (shell commands must be exact)
- top_p: 0.8 (tight — no creative command invention)

- [ ] **Step 2: Run the script**

Run: `cd "D:\Jarvis\jarvis complete v20" && python scripts/setup_agents.py`
Expected: All 5 agents updated with new values

- [ ] **Step 3: Commit**

```bash
git add scripts/setup_agents.py
git commit -m "feat: optimize agent temperature/top_p values for each role"
```

---

### Task 6: Run full test suite

- [ ] **Step 1: Run new tests**

Run: `cd "D:\Jarvis\jarvis complete v20" && python -m pytest tests/unit/test_agent_overrides.py -v`
Expected: All tests PASS

- [ ] **Step 2: Run existing tests**

Run: `cd "D:\Jarvis\jarvis complete v20" && python -m pytest tests/unit/test_tools_config.py -v`
Expected: All 10 tests PASS

- [ ] **Step 3: Verify no regressions in planner tests**

Run: `cd "D:\Jarvis\jarvis complete v20" && python -m pytest tests/ -k "planner" -v --timeout=60`
Expected: All PASS

- [ ] **Step 4: Commit any fixes**

```bash
git add -A
git commit -m "fix: test adjustments for agent override wiring"
```
