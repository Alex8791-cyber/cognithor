# Cognithor PSE — ARC-AGI-3 Game-Agent Integration

**Sprint-11 (started 2026-05-02), Wave-1 foundation.**

ARC-AGI-3 is the interactive-game successor to the static input/output ARC-AGI-1 benchmark Cognithor's PSE channel covered through Sprint-10. ARC-AGI-3 puts the agent in a **game loop**: per frame, the agent reads a `FrameData` (multi-layer grid + state + available actions) and emits a `GameAction`, until the level is won or the action budget runs out.

This page documents the Cognithor side of the integration. The official harness lives at [`arcprize/ARC-AGI-3-Agents`](https://github.com/arcprize/ARC-AGI-3-Agents) (MIT).

## Architecture

```
Official ARC-AGI-3-Agents harness
            │
            ├─ Agent ABC (is_done, choose_action)
            │
            └─ Cognithor's CognithorPSEAgent subclass
                       │
                       ├─ FrameBridge      (Wave-2)  — FrameData → Cognithor Grid
                       ├─ Phase-1 search   (Sprint-10 DSL, 76 primitives)
                       ├─ ActionDecoder    (Wave-2)  — Programm → GameAction
                       └─ vLLM/qwen3.6:27b (Wave-4)  — Stage-1+2 reasoning per frame
```

## Wave-1 surface (this PR)

`src/cognithor/channels/program_synthesis/arc_agi3/`:

- `protocol.py` — `FrameDataProtocol`, `GameActionProtocol`, `GameStateProtocol`. Mirror the upstream `arcengine` API surface as of ARC-AGI-3-Agents 0.9.3 (post-2026-01-29 field rename: `score`→`levels_completed`, `win_score`→`win_levels`).
- `agent.py` — `CognithorPSEAgent` ABC + `RandomActionAgent` smoke baseline.

Cognithor's code is typed against the local `Protocol`s, not directly against `arcengine`. The package imports cleanly without `arc-agi`/`arcengine` installed; a Wave-5 thin adapter (`arcengine_adapter.py`) plugs in the live types when running against the official harness.

## How to run inside the official harness

Once you have a clone of `arcprize/ARC-AGI-3-Agents` set up per its quickstart:

```bash
# 1. Install Cognithor as an editable dep in the harness venv:
cd /path/to/ARC-AGI-3-Agents
uv pip install -e /path/to/cognithor

# 2. Drop a thin shim into agents/templates/cognithor_agent.py:
```

```python
# agents/templates/cognithor_agent.py
from typing import Any
from arcengine import GameAction, FrameData
from agents.agent import Agent  # the official harness ABC

from cognithor.channels.program_synthesis.arc_agi3 import RandomActionAgent


class CognithorRandom(Agent):
    """Wave-1 smoke baseline — uniform random over available actions."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._delegate = RandomActionAgent()

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return self._delegate.is_done(frames, latest_frame)

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        return self._delegate.choose_action(frames, latest_frame)
```

```bash
# 3. Register the agent in agents/__init__.py's AVAILABLE_AGENTS dict.

# 4. Run against a game:
uv run main.py --agent=cognithor_random --game=ls20
```

The `RandomActionAgent` is a lower-bound score — anything below it is broken, anything above it starts measuring real ability. Subsequent waves replace it with DSL-driven and LLM-driven agents.

## Subsequent waves

| Wave | PR | Adds |
|---|---|---|
| 1 (this) | foundation | Protocol + ABC + RandomActionAgent |
| 2 | bridge | `FrameBridge` (FrameData → Grid) + `ActionDecoder` (Programm → Action) |
| 3 | DSL search | `Sprint10DSLAgent` — Phase-1 search per frame, 76 Sprint-10 primitives |
| 4 | LLM reasoning | `LLMReasoningAgent` over vLLM/qwen3.6:27b — Stage-1+2 per frame |
| 5 | live validation | `arcengine_adapter` + scorecards against `https://three.arcprize.org/` |

## See also

- Sprint-11 spec: `docs/superpowers/specs/2026-05-02-sprint11-arc-agi-3-foundation.md`
- Sprint-10 closure: `project_pse_phase2_sprint10_complete.md` (memory)
- Sprint-9 reality-check: `docs/superpowers/reports/sprint9_real_arc_findings.md`
