# Sprint-11 — ARC-AGI-3 Game-Agent Integration: Complete

**Date:** 2026-05-02
**Owner-Direktive:** "sprint 11 go" (nach "erst sprint 10 abschliessen dann auf offizielle aktuelle arc agi 3 umbauen")
**Trajectory:** 6 PRs, all merged on top of `main` post-Sprint-10

## TL;DR

Sprint-11 ports Cognithor PSE from the static ARC-AGI-1 corpus to
the **interactive ARC-AGI-3 game** challenge — a fundamentally
different format that requires an Episode-Loop, State-Memory, and
per-frame Action-Decoding. Six waves shipped:

| Wave | PR | Inhalt | New tests |
|---|---|---|---|
| 1 | #282 | Foundation: protocol + ABC + RandomActionAgent | 13 |
| 2 | #283 | FrameBridge + ActionDecoder + UniformActionDecoder | 22 |
| 3 | #284 | EpisodeMemory + ChangeDetector + StuckDetector | 24 |
| 4 | #285 | Sprint10DSLAgent + DSLActionDecoder (least-tried policy) | 15 |
| 5 | #286 | LLMReasoningAgent + LLMActionDecoder over vLLM/qwen3.6:27b | 17 |
| 6 | (this) | arcengine harness shim + scorecard parser/aggregator | 14 |
| **Total** | 6 PRs | **12 source modules** | **105 tests** |

Full PSE suite: 1569 passed, 10 skipped — no regressions.
mypy --strict clean across all 12 `arc_agi3/` source files.

## Architecture in one paragraph

Cognithor's `CognithorPSEAgent` is an abstract base typed against
**local Protocol classes** (`FrameDataProtocol`, `GameActionProtocol`,
`GameStateProtocol`) that mirror the upstream `arcengine` API surface
as of ARC-AGI-3-Agents 0.9.3. Concrete agents subclass it: the
`RandomActionAgent` smoke baseline (Wave-1), the heuristic
`Sprint10DSLAgent` (Wave-4), and the LLM-driven `LLMReasoningAgent`
(Wave-5). All three inherit a stateful pipeline:
`FrameBridge` (Wave-2) converts the multi-layer `0..15`-palette
ARC-AGI-3 frame to a Cognithor `int8 [0..9]` grid; `EpisodeMemory`
(Wave-3) tracks `(grid, action, levels)` in a bounded ring-buffer;
`ChangeDetector` and `StuckDetector` reason about progress;
`ActionDecoder` subclasses pick the next move. The
`cognithor_agent_factory` (Wave-6) wraps any of these in a
harness-compatible class for `arcprize/ARC-AGI-3-Agents`; the
`scorecard` module (Wave-6) parses + aggregates the run results.

## What ARC-AGI-3 actually is (vs ARC-AGI-1)

| Aspect | ARC-AGI-1 (Sprint-9/10) | ARC-AGI-3 (Sprint-11) |
|---|---|---|
| Format | Static input/output mapping | Interactive game loop |
| Input | `examples: list[(in, out)]` | `frames: list[FrameData]` episode |
| Output | Program → grid | `GameAction` per frame |
| Scoring | Exact grid equality | `levels_completed / win_levels` |
| Search | Phase-1 enumeration | Game-tree (≤ 80 actions) |
| LLM role | Prior for synthesis | Action selection per frame |
| Repo | `fchollet/ARC-AGI` | `arcprize/ARC-AGI-3-Agents` |

## How to plug Cognithor into the official harness

```bash
# 1. Clone and bootstrap the harness
git clone https://github.com/arcprize/ARC-AGI-3-Agents.git
cd ARC-AGI-3-Agents
uv sync

# 2. Install Cognithor as an editable dependency
uv pip install -e /path/to/cognithor

# 3. Drop a registration shim into the harness:
```

```python
# agents/templates/cognithor_agent.py
from cognithor.channels.program_synthesis.arc_agi3 import (
    LLMReasoningAgent,
    Sprint10DSLAgent,
    RandomActionAgent,
    cognithor_agent_factory,
    build_vllm_choice_fn,
)
from cognithor.core.vllm_backend import VLLMBackend

# Random baseline for the lower bound:
CognithorRandom = cognithor_agent_factory(delegate=RandomActionAgent())

# Sprint-10 DSL heuristic (deterministic, no GPU needed):
CognithorDSL = cognithor_agent_factory(delegate=Sprint10DSLAgent())

# LLM-driven, requires running vLLM with qwen3.6:27b:
_backend = VLLMBackend(base_url="http://localhost:8000/v1")
_choice_fn = build_vllm_choice_fn(backend=_backend)
CognithorLLM = cognithor_agent_factory(
    delegate=LLMReasoningAgent(choice_fn=_choice_fn),
)
```

```bash
# 4. Register in agents/__init__.py's AVAILABLE_AGENTS dict, then:
uv run main.py --agent=cognithor_random --game=ls20
uv run main.py --agent=cognithor_dsl    --game=locksmith
uv run main.py --agent=cognithor_llm    --game=ls20
```

## Hardware-gated wiring (Wave-5)

The LLM agent's call adapter `build_vllm_choice_fn` is hardware-gated
on a running vLLM server with `sakamakismile/Qwen3.6-27B-NVFP4` loaded —
same constraint as Sprint-10 Track B. **Without one, the first call
raises and the decoder transparently falls back to Wave-4's
DSLActionDecoder policy.** Tests use deterministic stub `choice_fn`
callables to validate the wiring + fallback paths.

## What's NOT in Sprint-11 (out of autonomous-mode scope)

- A running vLLM server with qwen3.6:27b loaded
- A live API key for `https://three.arcprize.org/`
- End-to-end Score-Validation against actual ARC-AGI-3 games
- Sprint-10 DSL extension to the ARC-AGI-3 16-colour palette
  (currently clamped via `ClampPolicy.SATURATE`)
- Multi-frame program synthesis (the DSL agent picks actions
  heuristically, not by enumerating programs that explain frame
  transitions — that would be a Phase-4 sprint)

## Sprint-11 closure status

**All 6 waves merged on main.** The PSE channel is now ARC-AGI-3-ready
in the architectural sense: any Cognithor PSE agent can be plugged into
the official harness with three lines of glue code. Score-Lift on
real ARC-AGI-3 games is the operator-side experiment.

## Production-Wiring carry-forward

- **Wave-5 LLM-agent** needs a vLLM/qwen3.6:27b instance to validate
  end-to-end. Same hardware as Sprint-10 Track B.
- **Sprint-10 DSL primitives** (76 of them) are usable from inside
  the agents but currently only as part of the ActionDecoder's
  least-tried heuristic. A future sprint can add a "DSL-program-
  synthesis-from-frame-pair" decoder that runs Phase-1 search on
  the (previous_frame, current_frame) pair to predict the next
  frame, then picks the action that makes the predicted frame
  most likely.
- **Scorecard module** is ready to parse production scorecards but
  has no recorded baselines yet — Sprint-12+ should commit a few
  scorecards from `--agent=cognithor_random` and
  `--agent=cognithor_llm` runs as `.ci/arc_agi3_*.json` regression
  baselines.

## See also

- Sprint-11 spec: `docs/superpowers/specs/2026-05-02-sprint11-arc-agi-3-foundation.md`
- Sprint-10 closure: `project_pse_phase2_sprint10_complete.md` (memory)
- Channel docs: `docs/channels/program_synthesis/arc_agi3.md`
