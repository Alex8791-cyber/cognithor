# ARC-AGI-3 Phase-A Live Validation Runbook

**Date:** 2026-05-02
**Hardware:** RTX 5090 + WSL2 Ubuntu 24.04 + vLLM 0.20.0 + NVFP4
**Stack:** Sprint-11 + Sprint-12 mega-merge complete

## Goal

Run an A/B comparison between three Cognithor PSE agents against three
ARC-AGI-3 games on the official `arcprize/ARC-AGI-3-Agents` harness.
Establish a current-state baseline, prove the Sprint-12 lifts move the
needle.

## Agents under test

| Agent | What it adds over baseline | New in Sprint-12 |
|---|---|---|
| `RandomActionAgent` | Random uniform sampler | None — baseline reference |
| `Sprint10DSLAgent` (default) | DSL search + EpisodeMemory + StateGraph + state-keyed counts | StateGraphNavigator (#290), state-keyed counts (#290), per-game prompts (#290) |
| `Sprint10DSLAgent(fast_path_enabled=True, frame_analyzer=..., audit_trail=..., game_profile=...)` | Click-toggle fast-path + per-action movement signatures + persistent audit + cross-episode profile | PR-4..PR-9 |
| `LLMReasoningAgent` (vLLM/qwen3.6:27b NVFP4) | LLM-driven action selection | Same persistence wiring as DSL agent |

## Games under test

| Game ID | Mechanic | Expected fast-path benefit |
|---|---|---|
| `bp35` | Click-target | `ClickTargetSampler` — small components first |
| `ft09` | Click-target with movement | `FrameAnalyzer` learns ACTION→direction; `ClickTargetSampler` samples |
| `lp85` | Toggle-cluster (LS20-like) | `plan_click_solution()` should hit immediately |

## Setup

### One-time

```bash
# 1. WSL2 Ubuntu 24.04 with vLLM 0.20.0 + CUDA 13.0 already in place
#    (~/vllm-env, /home/articall/hf_cache, sakamakismile/Qwen3.6-27B-NVFP4)

# 2. Clone the official harness
cd ~
git clone https://github.com/arcprize/ARC-AGI-3-Agents.git
cd ARC-AGI-3-Agents
uv sync

# 3. Install Cognithor (post-Sprint-12) into the harness venv
uv pip install -e /mnt/d/Jarvis/jarvis\ complete\ v20

# 4. Smoke-check imports
uv run python -c "from cognithor.channels.program_synthesis.arc_agi3 import (
    Sprint10DSLAgent, LLMReasoningAgent, RandomActionAgent,
    ArcAuditTrail, GameProfile, FrameAnalyzer, ClickTargetSampler,
    cognithor_agent_factory)
print('all imports green')"
```

### Per-run

```bash
# Activate vLLM in tmux pane 1
cd ~/vllm-env && source bin/activate
python -c "from vllm import LLM; LLM(model='sakamakismile/Qwen3.6-27B-NVFP4', \
    quantization='fp4', max_model_len=32768, gpu_memory_utilization=0.92, \
    max_num_seqs=64)"
# (warms the model in-process; for production, use the InProcess vLLM
# backend — see vllm_inprocess_backend.py)

# In tmux pane 2 — drive the agents
cd ~/ARC-AGI-3-Agents
uv run python sprint12_validation.py
```

## Driver script outline

`sprint12_validation.py`:

```python
from cognithor.channels.program_synthesis.arc_agi3 import (
    Sprint10DSLAgent, RandomActionAgent, LLMReasoningAgent,
    ArcAuditTrail, GameProfile, FrameAnalyzer,
    cognithor_agent_factory, build_inprocess_vllm_choice_fn,
)
from agents import AVAILABLE_AGENTS  # harness

GAMES = ["bp35", "ft09", "lp85"]
AGENT_CONFIGS = [
    ("random_baseline", lambda gid: RandomActionAgent()),
    ("dsl_baseline", lambda gid: Sprint10DSLAgent()),
    ("dsl_full",
        lambda gid: Sprint10DSLAgent(
            fast_path_enabled=True,
            audit_trail=ArcAuditTrail(game_id=gid),
            game_profile=GameProfile.load(gid) or GameProfile(...),
            strategy_name="dsl_full_v1",
        )),
    ("llm_full",
        lambda gid: LLMReasoningAgent(
            choice_fn=build_inprocess_vllm_choice_fn(...),
            audit_trail=ArcAuditTrail(game_id=gid),
            game_profile=GameProfile.load(gid) or GameProfile(...),
        )),
]

results = []
for game_id in GAMES:
    for label, factory in AGENT_CONFIGS:
        delegate = factory(game_id)
        cls = cognithor_agent_factory(delegate=delegate, name_suffix=label)
        AVAILABLE_AGENTS[cls.__name__] = cls
        # Drive the harness's Game.run with this agent + game_id ...
        # ... record (game_id, label, levels_completed, total_steps,
        #             final_score, audit_path)
        if hasattr(delegate, "finalize_episode"):
            delegate.finalize_episode(score=..., won=..., levels_solved=...)
        if hasattr(delegate, "_game_profile") and delegate._game_profile:
            delegate._game_profile.save()
```

## Metrics to record

For every (agent, game) cell:

- `levels_completed` — 0..win_levels per game
- `total_steps` — capped at MAX_ACTIONS=80 by the harness
- `final_score` — `levels_completed / win_levels`
- `wall_clock_s` — for vLLM-driven runs, dominated by model latency
- `fast_path_hits` (Sprint12 dsl_full only) — count `"fast-path"` in
  audit reasoning lines
- `audit_path` — written JSONL per run for replay

## Pass criteria (Sprint-12 success signal)

- `dsl_full` ≥ `dsl_baseline` on every game (no regression).
- `dsl_full` solves at least one game `dsl_baseline` did NOT.
- `llm_full` matches or beats `dsl_full` on `bp35` (LLM should grok
  click-target games quickest).
- Audit trails verify integrity (`trail.verify_integrity() == True`)
  on every saved run.
- GameProfiles saved + reloadable across re-runs.

## What to do if things break

| Symptom | First check | Fallback |
|---|---|---|
| `ImportError: agents.agent` | `cognithor_agent_factory` raised — install harness via `uv sync` | — |
| vLLM `_C` extension missing | wrong env (Windows vs WSL2) | activate `~/vllm-env`, NOT a Windows venv |
| OOM mid-game | reduce `gpu_memory_utilization` to 0.85, lower `max_model_len` to 16384 | switch to FP8 base, accept 28 GB VRAM use |
| `ValueError: shapes don't broadcast` in agent | level-transition shape change | merge / cherry-pick #294 / #296 / #297 fix |
| Audit trail integrity fails | concurrent writes? | add lock, or run agents sequentially |
| `pixels_changed` always 0 in audit | memory window order bug | inspect `EpisodeMemory.window(2)` ordering |

## Aftermath / artefacts

1. Commit raw run JSONL to a repo-private results folder
   (`cognithor_bench/results/2026-05-02-phase-a/`).
2. Roll the metrics into `MEMORY.md` — Sprint-12 ship log entry.
3. Snapshot the GameProfile JSON files to
   `~/.cognithor/arc/game_profiles/` before resetting.
4. Open Sprint-13 directive based on what's missing (e.g. "fast-path
   doesn't fire on bp35 because toggle pair isn't observed in the
   first 8 frames" → need bootstrap heuristic).

## Open questions for after the run

- Does the DSL search ever fire for non-toggle games once
  `ClickTargetSampler` is wired in (PR-10 follow-up)?
- Can the LLM agent benefit from being shown the
  `FrameAnalyzer.get_action_summary()` in its prompt?
- How often does the StateGraphNavigator's BFS-replay-to-WIN actually
  fire? (Compare against the simpler decoder.)
