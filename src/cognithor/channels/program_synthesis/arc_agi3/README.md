# `cognithor.channels.program_synthesis.arc_agi3`

Cognithor PSE — interactive ARC-AGI-3 game-agent integration.

ARC-AGI-3 is the *interactive* successor to ARC-AGI-1's static grids:
the agent sees one frame at a time, picks one of `RESET`/`ACTION1`..
`ACTION7` (with `ACTION6` taking `(x, y)` click coords), and the
upstream API delivers the next frame. Episodes cap at **80 actions**.

This package is the Cognithor side of the integration. It lands on top
of the official [`arcprize/ARC-AGI-3-Agents`](https://github.com/arcprize/ARC-AGI-3-Agents)
harness via the duck-typed Protocols in `protocol.py` — no `arcengine`
import required at module load.

## Layout

| Module | What it does |
|---|---|
| `protocol.py` | Local Protocols mirroring `arcengine.FrameData/GameAction/GameState` |
| `agent.py` | `CognithorPSEAgent` ABC + `RandomActionAgent` smoke-baseline |
| `frame_bridge.py` | Frame → Cognithor int8 grid (`ClampPolicy` for >9-colour palettes) |
| `action_decoder.py` | `ActionDecoder` ABC + `UniformActionDecoder` baseline |
| `episode_memory.py` | `EpisodeMemory` ring-buffer + `ChangeDetector` + `StuckDetector` |
| `dsl_action_decoder.py` | Heuristic least-tried action policy + state-keyed prioritisation |
| `dsl_agent.py` | `Sprint10DSLAgent` — full agent (wires bridge + memory + decoder + state graph + state-counter + audit + profile + frame-analyzer + fast-path + level-transition) |
| `llm_action_decoder.py` | LLM-driven decoder (`FrameContext`, `summarise_history`, `summarise_action_effects`, `render_grid`) |
| `llm_agent.py` | `LLMReasoningAgent` (subclass of `Sprint10DSLAgent`) + vLLM choice-fn factories (HTTP + InProcess) |
| `harness_shim.py` | `cognithor_agent_factory` — adapter to the official `agents.agent.Agent` ABC |
| `scorecard.py` | Parse the harness's per-game-id scorecard JSON |
| `state_graph.py` | `StateGraphNavigator` — BFS-replay-to-WIN |
| `state_action_counts.py` | `StateActionCounter` — Blind-Squirrel-style state-keyed action history |
| `game_prompts.py` | Per-game LLM prompt fragments (LS20 Locksmith rules etc.) |
| `audit.py` | `ArcAuditTrail` — append-only SHA-256 hash-chain event log |
| `game_profile.py` | `GameProfile` — persistent per-game mechanic + strategy metrics |
| `fast_grid_planner.py` | Pure-NumPy click-toggle solver (`plan_click_solution`) |
| `fast_path.py` | Glue: `detect_toggle_pair_from_memory` + `ClickPlanCache` |
| `frame_analyzer.py` | Per-action movement-signature tracker (`MovementInfo`) |
| `click_target_sampler.py` | Salience-ranked click coordinate generator for non-toggle games |

## Agents

### `RandomActionAgent`

Smoke baseline. Uniform-random sampler over `available_actions`. Use
this to verify the harness wiring; expect ≤ 0 levels solved.

### `Sprint10DSLAgent`

The Sprint-12 production heuristic agent. All Sprint-12 lifts opt-in
via constructor flags:

```python
agent = Sprint10DSLAgent(
    bridge=FrameBridge(),
    memory=EpisodeMemory(capacity=16),
    stuck_detector=StuckDetector(threshold=8),
    state_counter=StateActionCounter(),
    state_graph=StateGraphNavigator(),
    frame_analyzer=FrameAnalyzer(),                # PR-5/12
    audit_trail=ArcAuditTrail(game_id="ls20"),     # PR-7
    game_profile=GameProfile.load("ls20"),         # PR-6
    strategy_name="dsl_full",                      # PR-6
    fast_path_enabled=True,                        # PR-8
)

# At episode end:
agent.finalize_episode(score=2, won=True, levels_solved=2, budget_ratio=0.4)
agent.frame_analyzer  # for inspection
```

### `LLMReasoningAgent`

Subclass of `Sprint10DSLAgent` that swaps the `DSLActionDecoder` for
an LLM-driven `LLMActionDecoder`. Same persistence + analyzer plumbing
as the parent.

```python
from cognithor.channels.program_synthesis.arc_agi3 import (
    LLMReasoningAgent, build_inprocess_vllm_choice_fn
)

choice_fn = build_inprocess_vllm_choice_fn(
    model_name="sakamakismile/Qwen3.6-27B-NVFP4",
    max_model_len=32768,
    gpu_memory_utilization=0.92,
)
agent = LLMReasoningAgent(choice_fn=choice_fn, ...)
```

`build_vllm_choice_fn` (HTTP-based) is also available for cases where
vLLM runs out-of-process.

## Running inside the official harness

```python
from cognithor.channels.program_synthesis.arc_agi3 import (
    cognithor_agent_factory, Sprint10DSLAgent
)
from agents import AVAILABLE_AGENTS

cls = cognithor_agent_factory(
    delegate=Sprint10DSLAgent(fast_path_enabled=True),
    name_suffix="Sprint12Full",
    extra_tags=["sprint12", "fast_path"],
)
AVAILABLE_AGENTS[cls.__name__] = cls
# Now the harness can spawn this agent under that name.
```

The factory raises `ImportError` when `agents.agent` isn't on
sys.path — see the message for the install hint.

## Persistent artefacts

| Artefact | Path | Contents |
|---|---|---|
| Game profile | `~/.cognithor/arc/game_profiles/<game_id>.json` | win-rate per strategy, best score, total runs, click zones |
| Audit trail | `<custom-path>.jsonl` (via `ArcAuditTrail.export_jsonl`) | game_start + step + level_complete + game_end events with SHA-256 chain |

`GameProfile.load(game_id)` is forward-compatible with the legacy
`cognithor.arc/` disk format — existing profiles populated by the
old stack keep loading.

## Phase-A live validation

See `docs/superpowers/plans/2026-05-02-arc-agi-3-phase-a-runbook.md`
for the operator-facing runbook (RTX 5090 + WSL2 + vLLM NVFP4 setup,
driver script outline, pass criteria, fallback table).
