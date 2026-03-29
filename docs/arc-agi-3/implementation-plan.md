# ARC-AGI-3 Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cognithor als ARC-AGI-3 Agent lauffähig machen mit korrekter Codebase-Integration

**Architecture:** Neues `src/jarvis/arc/` Package (14 Dateien) mit Hybrid-Agent (algorithmisch + optional LLM + optional CNN). Nutzt bestehende Jarvis-Infrastruktur (Config, MCP, Audit). Phase 0 (SDK-Validierung) ZUERST.

**Tech Stack:** Python 3.12, arc-agi SDK, numpy, optional torch, Pydantic, pytest

---

## File Structure

**Create:**
- `src/jarvis/arc/__init__.py`
- `src/jarvis/arc/__main__.py`
- `src/jarvis/arc/validate_sdk.py`
- `src/jarvis/arc/error_handler.py`
- `src/jarvis/arc/adapter.py`
- `src/jarvis/arc/episode_memory.py`
- `src/jarvis/arc/visual_encoder.py`
- `src/jarvis/arc/goal_inference.py`
- `src/jarvis/arc/mechanics_model.py`
- `src/jarvis/arc/explorer.py`
- `src/jarvis/arc/agent.py`
- `src/jarvis/arc/audit.py`
- `src/jarvis/arc/swarm.py`
- `src/jarvis/arc/cnn_model.py`
- `src/jarvis/mcp/arc_tools.py`
- `tests/test_arc/__init__.py`
- `tests/test_arc/conftest.py`
- `tests/test_arc/test_episode_memory.py`
- `tests/test_arc/test_goal_inference.py`
- `tests/test_arc/test_explorer.py`
- `tests/test_arc/test_visual_encoder.py`
- `tests/test_arc/test_mechanics_model.py`
- `tests/test_arc/test_audit.py`
- `tests/test_arc/test_error_handler.py`
- `tests/test_arc/test_agent.py`

**Modify:**
- `src/jarvis/config.py` — add `ArcConfig` Pydantic model
- `pyproject.toml` — add `arc` and `arc-gpu` dependency groups
- `src/jarvis/__main__.py` — add `--arc` CLI flag

---

### Task 0: Phase 0 — SDK Installation & Validation

**Files:**
- Create: `src/jarvis/arc/__init__.py`
- Create: `src/jarvis/arc/validate_sdk.py`
- Modify: `pyproject.toml`

- [ ] **Step 0.1: Add arc dependency group to pyproject.toml**

In `pyproject.toml`, after the existing `blockchain` group, add:

```toml
arc = [
    "arc-agi>=0.3.0",
    "numpy>=2.0,<3",
]
arc-gpu = [
    "cognithor[arc]",
    "torch>=2.2.0",
]
```

- [ ] **Step 0.2: Create package init**

```python
# src/jarvis/arc/__init__.py
"""Cognithor ARC-AGI-3 Agent — Interactive Reasoning Benchmark Integration."""
```

- [ ] **Step 0.3: Install SDK**

```bash
pip install arc-agi
```

- [ ] **Step 0.4: Create validate_sdk.py**

Create `src/jarvis/arc/validate_sdk.py` with the full 10-step validation script from the corrected spec (Section 9). The script must:
- Import arc_agi and arcengine
- Inspect Arcade class methods
- Create environment "ls20"
- Inspect FrameDataRaw attributes and grid format
- Test GameState enum values
- Test GameAction enum values and is_simple()/is_complex()
- Check env.action_space
- Test env.step() return format
- Check scorecard format
- Save results to `arc_agi3_validation.json`

All output must use ASCII-safe characters (`[OK]`, `[FAIL]`, `[WARN]`).

- [ ] **Step 0.5: Run validation**

```bash
python -m jarvis.arc.validate_sdk
```

- [ ] **Step 0.6: Document results and adjust assumptions**

Read `arc_agi3_validation.json` and update the assumptions table. Any false assumptions must be noted for adaptation in later tasks.

- [ ] **Step 0.7: Commit**

```bash
git add src/jarvis/arc/__init__.py src/jarvis/arc/validate_sdk.py pyproject.toml
git commit -m "feat(arc): Phase 0 — SDK validation + arc dependency group"
```

---

### Task 1: Error Handler + Safe Frame Extract

**Files:**
- Create: `src/jarvis/arc/error_handler.py`
- Create: `tests/test_arc/__init__.py`
- Create: `tests/test_arc/conftest.py`
- Create: `tests/test_arc/test_error_handler.py`

- [ ] **Step 1.1: Write failing tests**

```python
# tests/test_arc/test_error_handler.py
import pytest
import numpy as np


class TestSafeFrameExtract:
    def test_none_returns_fallback(self):
        from jarvis.arc.error_handler import safe_frame_extract
        result = safe_frame_extract(None)
        assert result.shape == (64, 64, 3)
        assert np.all(result == 0)

    def test_valid_frame_attribute(self):
        from jarvis.arc.error_handler import safe_frame_extract
        obs = type("O", (), {"frame": np.ones((64, 64, 3), dtype=np.uint8)})()
        result = safe_frame_extract(obs)
        assert np.all(result == 1)

    def test_unknown_format_returns_fallback(self):
        from jarvis.arc.error_handler import safe_frame_extract
        obs = type("O", (), {"unknown_field": 42})()
        result = safe_frame_extract(obs)
        assert result.shape == (64, 64, 3)

    def test_flat_array_reshaped(self):
        from jarvis.arc.error_handler import safe_frame_extract
        flat = np.ones(64 * 64, dtype=np.uint8)
        obs = type("O", (), {"frame": flat})()
        result = safe_frame_extract(obs)
        assert result.shape == (64, 64)


class TestRetryOnError:
    def test_succeeds_on_first_try(self):
        from jarvis.arc.error_handler import retry_on_error

        @retry_on_error(max_retries=2, delay_seconds=0)
        def ok():
            return 42

        assert ok() == 42

    def test_retries_on_failure(self):
        from jarvis.arc.error_handler import retry_on_error

        call_count = 0

        @retry_on_error(max_retries=2, delay_seconds=0, exceptions=(ValueError,))
        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("fail")
            return "ok"

        assert flaky() == "ok"
        assert call_count == 3


class TestGameRunGuard:
    def test_suppresses_exceptions(self):
        from jarvis.arc.error_handler import GameRunGuard

        class FakeArcade:
            def make(self, game_id):
                return type("Env", (), {"reset": lambda s: None, "step": lambda s, a: None})()
            def get_scorecard(self):
                return None

        with GameRunGuard(FakeArcade(), "test") as guard:
            raise RuntimeError("boom")
        # Should not propagate
        assert len(guard.errors) == 1
```

- [ ] **Step 1.2: Run tests to verify they fail**

```bash
pytest tests/test_arc/test_error_handler.py -v
```

Expected: FAIL (module not found)

- [ ] **Step 1.3: Implement error_handler.py**

Create `src/jarvis/arc/error_handler.py` with:
- `ArcAgentError`, `FrameExtractionError`, `EnvironmentConnectionError` exceptions
- `retry_on_error` decorator (exponential backoff, configurable)
- `safe_frame_extract(obs, fallback_shape)` — tries frame/frame_data/grid/pixels/data/image attributes, always returns numpy array
- `GameRunGuard` context manager — suppresses exceptions, always gets scorecard

Use `from jarvis.utils.logging import get_logger` for logging.

- [ ] **Step 1.4: Run tests to verify they pass**

```bash
pytest tests/test_arc/test_error_handler.py -v
```

Expected: ALL PASS

- [ ] **Step 1.5: Commit**

```bash
git add src/jarvis/arc/error_handler.py tests/test_arc/
git commit -m "feat(arc): error handler with safe_frame_extract and GameRunGuard"
```

---

### Task 2: Episode Memory

**Files:**
- Create: `src/jarvis/arc/episode_memory.py`
- Create: `tests/test_arc/test_episode_memory.py`

- [ ] **Step 2.1: Write failing tests**

Tests from corrected spec Section 19: hash determinism, different grids differ, record_transition, action_effectiveness, unknown action = 0.5, novel state detection, max_transitions limit, clear_for_new_level preserves transitions, summary_for_llm returns string.

- [ ] **Step 2.2: Run tests — expect FAIL**
- [ ] **Step 2.3: Implement episode_memory.py**

StateTransition, Hypothesis, EpisodeMemory classes. MD5 hashing with cache, action_effect_map, 200K max transitions.

- [ ] **Step 2.4: Run tests — expect PASS**
- [ ] **Step 2.5: Commit**

```bash
git commit -m "feat(arc): episode memory with state hashing and action-effect tracking"
```

---

### Task 3: Visual State Encoder

**Files:**
- Create: `src/jarvis/arc/visual_encoder.py`
- Create: `tests/test_arc/test_visual_encoder.py`

- [ ] **Step 3.1: Write failing tests**

encode_for_llm returns string with "Farbverteilung", encode_compact returns bracketed string, RGB grid handling, diff analysis.

- [ ] **Step 3.2: Run tests — expect FAIL**
- [ ] **Step 3.3: Implement visual_encoder.py**

VisualStateEncoder with color_names dict, encode_for_llm (histogram + regions + diff), encode_compact, _rgb_to_index, _find_bounding_boxes.

- [ ] **Step 3.4: Run tests — expect PASS**
- [ ] **Step 3.5: Commit**

```bash
git commit -m "feat(arc): visual state encoder — grid to text for LLM context"
```

---

### Task 4: Goal Inference Module

**Files:**
- Create: `src/jarvis/arc/goal_inference.py`
- Create: `tests/test_arc/test_goal_inference.py`

- [ ] **Step 4.1: Write failing tests**

No wins → UNKNOWN goal, win transitions → REACH_STATE goal with action name, game over → AVOID goal, best_goal returns highest confidence.

- [ ] **Step 4.2: Run tests — expect FAIL**
- [ ] **Step 4.3: Implement goal_inference.py**

GoalType enum, InferredGoal dataclass, GoalInferenceModule with 4 analysis strategies.

- [ ] **Step 4.4: Run tests — expect PASS**
- [ ] **Step 4.5: Commit**

```bash
git commit -m "feat(arc): goal inference — autonomous win condition detection"
```

---

### Task 5: Mechanics Model

**Files:**
- Create: `src/jarvis/arc/mechanics_model.py`
- Create: `tests/test_arc/test_mechanics_model.py`

- [ ] **Step 5.1: Write failing tests**

Empty model returns empty, unknown action → UNKNOWN type, analyze_transitions creates mechanics, consistency EMA, reliable_mechanics filtering.

- [ ] **Step 5.2: Run tests — expect FAIL**
- [ ] **Step 5.3: Implement mechanics_model.py**

MechanicType enum, Mechanic dataclass, MechanicsModel with analyze_transitions, predict_action_effect, snapshot_level, EMA consistency.

- [ ] **Step 5.4: Run tests — expect PASS**
- [ ] **Step 5.5: Commit**

```bash
git commit -m "feat(arc): mechanics model — cross-level rule abstraction"
```

---

### Task 6: Hypothesis-Driven Explorer

**Files:**
- Create: `src/jarvis/arc/explorer.py`
- Create: `tests/test_arc/test_explorer.py`

- [ ] **Step 6.1: Write failing tests**

Initial phase is DISCOVERY, phase transition after 50 steps, discovery returns actions from queue, hypothesis prefers effective actions, exploitation replays win actions, _parse_action_str handles "ACTION6_32_15".

- [ ] **Step 6.2: Run tests — expect FAIL**
- [ ] **Step 6.3: Implement explorer.py**

ExplorationPhase enum, HypothesisDrivenExplorer with 3 phases, strategic grid sampling (13 positions), phase auto-transitions, action string parsing.

NOTE: `GameAction.is_simple()` / `is_complex()` methods may not exist — Phase 0 results determine this. Use defensive checks: `if hasattr(a, 'is_simple') and a.is_simple()`.

- [ ] **Step 6.4: Run tests — expect PASS**
- [ ] **Step 6.5: Commit**

```bash
git commit -m "feat(arc): hypothesis-driven explorer — 3-phase systematic exploration"
```

---

### Task 7: Environment Adapter

**Files:**
- Create: `src/jarvis/arc/adapter.py`

- [ ] **Step 7.1: Implement adapter.py**

ArcObservation dataclass, ArcEnvironmentAdapter with initialize(), act(), reset_level(), _process_frame(), _extract_grid() (uses safe_frame_extract from error_handler).

NOTE: _extract_grid() must be adapted based on Phase 0 validation results. Use safe_frame_extract as fallback.

- [ ] **Step 7.2: Commit**

```bash
git commit -m "feat(arc): environment adapter — ARC SDK bridge"
```

---

### Task 8: Audit Trail

**Files:**
- Create: `src/jarvis/arc/audit.py`
- Create: `tests/test_arc/test_audit.py`

- [ ] **Step 8.1: Write failing tests**

Hash chain integrity, export_jsonl format, game_start/game_end events.

- [ ] **Step 8.2: Run tests — expect FAIL**
- [ ] **Step 8.3: Implement audit.py**

ArcAuditEvent dataclass, ArcAuditTrail with SHA-256 hash chain, log_event, log_step, log_game_start, log_game_end, export_jsonl, verify_integrity.

- [ ] **Step 8.4: Run tests — expect PASS**
- [ ] **Step 8.5: Commit**

```bash
git commit -m "feat(arc): audit trail with SHA-256 hash chain"
```

---

### Task 9: Agent — Main Orchestration

**Files:**
- Create: `src/jarvis/arc/agent.py`
- Create: `tests/test_arc/test_agent.py`

- [ ] **Step 9.1: Write failing tests**

Agent initializes all modules, _action_to_str formats correctly, _on_level_complete resets explorer phase, step returns correct state strings.

- [ ] **Step 9.2: Run tests — expect FAIL**
- [ ] **Step 9.3: Implement agent.py**

CognithorArcAgent with run() game loop, _step(), _consult_llm_planner() (stub — TODO connects to Jarvis Planner), _on_level_complete().

- [ ] **Step 9.4: Run tests — expect PASS**
- [ ] **Step 9.5: Commit**

```bash
git commit -m "feat(arc): main agent orchestration with hybrid game loop"
```

---

### Task 10: Config Integration

**Files:**
- Modify: `src/jarvis/config.py`

- [ ] **Step 10.1: Add ArcConfig to config.py**

Add `ArcConfig(BaseModel)` with all fields from corrected spec Section 5. Add `arc: ArcConfig = ArcConfig()` to `JarvisConfig`.

- [ ] **Step 10.2: Run existing config tests**

```bash
pytest tests/test_core/test_config.py -v
```

Expected: ALL PASS (new field has default)

- [ ] **Step 10.3: Commit**

```bash
git commit -m "feat(arc): ArcConfig Pydantic model in config.py"
```

---

### Task 11: CLI Entry Point

**Files:**
- Create: `src/jarvis/arc/__main__.py`

- [ ] **Step 11.1: Implement __main__.py**

argparse with --game, --mode (single/benchmark/swarm), --no-llm, --cnn, --parallel, --verbose, --list-games. Uses `from jarvis.arc.agent import CognithorArcAgent`.

- [ ] **Step 11.2: Test manually**

```bash
python -m jarvis.arc --help
```

- [ ] **Step 11.3: Commit**

```bash
git commit -m "feat(arc): CLI entry point — python -m jarvis.arc"
```

---

### Task 12: MCP Tools

**Files:**
- Create: `src/jarvis/mcp/arc_tools.py`

- [ ] **Step 12.1: Implement arc_tools.py**

3 tool schemas (arc_play, arc_status, arc_replay) + async handlers. Register via `register_builtin_handler` pattern (same as other MCP tools).

- [ ] **Step 12.2: Commit**

```bash
git commit -m "feat(arc): MCP tools — arc_play, arc_status, arc_replay"
```

---

### Task 13: Swarm Mode

**Files:**
- Create: `src/jarvis/arc/swarm.py`

- [ ] **Step 13.1: Implement swarm.py**

ArcSwarmOrchestrator with run_swarm(), asyncio.Semaphore, run_in_executor for sync agent.run(). SwarmResult dataclass, get_aggregate_score(), get_summary().

- [ ] **Step 13.2: Commit**

```bash
git commit -m "feat(arc): swarm orchestrator for parallel game runs"
```

---

### Task 14: CNN Model (Optional)

**Files:**
- Create: `src/jarvis/arc/cnn_model.py`

- [ ] **Step 14.1: Implement cnn_model.py**

ActionPredictor (Conv2d stack → action head + coord head), OnlineTrainer with experience buffer, mini-batch training, grid-to-tensor one-hot encoding.

Guard all torch imports with try/except — module must not crash if torch is absent.

- [ ] **Step 14.2: Commit**

```bash
git commit -m "feat(arc): optional CNN action predictor for competition mode"
```

---

### Task 15: Ruff + Full Test Suite

- [ ] **Step 15.1: Run ruff format on all new files**

```bash
ruff format src/jarvis/arc/ tests/test_arc/ src/jarvis/mcp/arc_tools.py
```

- [ ] **Step 15.2: Run ruff check**

```bash
ruff check src/jarvis/arc/ tests/test_arc/ --select=F821,F811 --no-fix
```

- [ ] **Step 15.3: Run full test suite**

```bash
pytest tests/test_arc/ -v --tb=short
```

- [ ] **Step 15.4: Run existing tests to verify no regressions**

```bash
pytest tests/ -x -q --ignore=tests/test_channels/test_voice_ws_bridge.py
```

- [ ] **Step 15.5: Final commit**

```bash
git commit -m "feat(arc): ARC-AGI-3 integration complete — v0.67.0"
```

---

## Timeline

| Phase | Tasks | Zeitrahmen | Output |
|-------|-------|------------|--------|
| Phase 0 | Task 0 | Tag 1 | SDK validiert, Annahmen dokumentiert |
| Phase 1 | Tasks 1-5 | Woche 1 | Kernmodule mit Tests |
| Phase 2 | Tasks 6-9 | Woche 2 | Funktionierender Agent |
| Phase 3 | Tasks 10-13 | Woche 3 | Config, CLI, MCP, Swarm |
| Phase 4 | Tasks 14-15 | Woche 3-4 | CNN, Lint, Full Tests |

## Milestone Alignment

- 30. Juni 2026 (ARC Prize Milestone 1): Tasks 0-13, Score > 1%
- 30. September 2026 (ARC Prize Milestone 2): Task 14 + Tuning, Score > 5%
- 2. November 2026 (Final): Optimiert, Score > 10%
