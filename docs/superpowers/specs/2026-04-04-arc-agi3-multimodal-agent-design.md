# ARC-AGI-3: Multimodal Agent (Official Architecture) — Design Spec

**Date:** 2026-04-04
**Status:** Approved
**Based on:** Official ARC-AGI-3 reference implementation (arcprize/ARC-AGI-3-Agents/multimodal.py)

## Goal

Rebuild the ARC agent following the official multimodal architecture: send game frames as images to qwen3-vl:32b EVERY step with an analyze→plan→act loop and cumulative memory of discovered rules.

## Key Differences from Our Current Approach

| Aspect | Current Cognithor | Official Reference |
|--------|-------------------|-------------------|
| Vision calls | Every ~50 steps | **Every step** |
| Image scale | 8x (512x512) | **2x (128x128)** |
| Memory | None between calls | **Cumulative rules + action history** |
| Flow | Observe → Act | **Observe → Analyze previous result → Plan next → Act** |
| Max actions | 1000 | **40 per game** |
| Action knowledge | Actions are numbers | **ACTION1=UP, 2=DOWN, 3=LEFT, 4=RIGHT, 5=Interact, 6=Click(x,y)** |

## Architecture

```
For each step:
  1. ANALYZE: Send before-image + after-image + diff-image to LLM
     "What changed? Was my action successful? Update rules."
  2. PLAN: Send current image + memory to LLM
     "What should I do next? Which action and why?"
  3. ACT: Execute the chosen action
  4. UPDATE MEMORY: Add discovered rules and action result
```

## 1. Image Processing

Grid (64x64, values 0-15) → PNG (128x128, 2x upscale):

```python
PALETTE = [
    (255,255,255,255), (0,0,0,255), (0,116,217,255), (255,65,54,255),
    (46,204,64,255), (255,220,0,255), (170,170,170,255), (255,133,27,255),
    (127,219,255,255), (135,12,37,255), (240,18,190,255), (255,255,255,255),
    (200,200,100,255), (100,50,150,255), (0,200,200,255), (128,0,255,255),
]

def grid_to_image(grid, scale=2):
    """64x64 grid → 128x128 PNG."""

def image_diff(before, after, color=(255,0,0)):
    """Highlight changed pixels in red on black background."""
```

Scale 2x not 8x — smaller image = faster LLM inference, and 128x128 is enough for qwen3-vl:32b.

## 2. Action Mapping

```python
ACTION_DESCRIPTIONS = {
    "ACTION1": "Move UP",
    "ACTION2": "Move DOWN",
    "ACTION3": "Move LEFT",
    "ACTION4": "Move RIGHT",
    "ACTION5": "Interact / Undo",
    "ACTION6": "Click at position (x, y)",
}
```

The LLM chooses actions by name. The agent translates to GameAction enums.

## 3. Memory Prompt (Cumulative)

```
## Game Knowledge
- Inputs: [description of what I see]
- Goal: [what I think the goal is]
- Rules discovered:
  1. ACTION1 moves the character up
  2. Clicking on red squares toggles them
  3. ...
- Action history (last 10):
  Step 1: ACTION1 → character moved up 2 pixels
  Step 2: ACTION4 → character moved right, hit wall
  ...
```

Updated after every analyze phase. Carries forward between steps.

## 4. Analyze Phase

After each action, send 3 images to LLM:
- Before image (previous frame)
- After image (current frame)
- Diff image (changed pixels highlighted in red)

Prompt:
```
I performed {action_name} and expected {expected_outcome}.

[before_image] [after_image] [diff_image]

What actually changed? Was my action successful?
Update the rules if you learned something new.
Reply as JSON: {"observation": "...", "success": bool, "new_rule": "..." or null}
```

## 5. Plan Phase

Send current image + memory to LLM:

```
{memory_prompt}

[current_image]

Available actions: {available_action_descriptions}
What should I do next?

Reply as JSON: {"action": "ACTION1", "reasoning": "...", "expected_outcome": "..."}
```

## 6. Optimization for Local Model

qwen3-vl:32b is slower than GPT-4o-mini. Key adaptations:

- **Combine analyze+plan into ONE call** (not two separate calls) to halve LLM invocations
- **num_predict=500** to limit think-block length
- **128x128 images** (smaller than our previous 512x512)
- **40-action maximum** matches official spec (less time wasted on hopeless runs)
- **Skip analysis on first step** (no previous action to analyze)

Combined prompt:
```
{memory_prompt}

Previous action: {last_action} (expected: {expected})
[before_image] [after_image] [diff_image]

Current state:
[current_image]

1. What changed from my previous action?
2. What should I do next?

Reply as JSON: {
  "observation": "...",
  "new_rule": "..." or null,
  "next_action": "ACTION1",
  "reasoning": "...",
  "expected_outcome": "..."
}
```

One LLM call per step instead of two. ~15-30s per step × 40 steps = ~10-20 minutes per game.

## 7. Files

| File | Change |
|------|--------|
| `src/jarvis/arc/multimodal_agent.py` | **NEW** — Complete multimodal agent |
| `src/jarvis/arc/agent.py` | Add multimodal mode selection |
| `tests/test_arc/test_multimodal.py` | **NEW** — Tests |

## 8. Expected Performance

- 40 actions per game maximum
- ~15-30s per LLM call (qwen3-vl:32b local)
- ~10-20 minutes per game total
- The agent learns rules iteratively — first few actions are exploration, later actions are goal-directed
- ClusterSolver remains available as fast fallback for games where rule discovery identifies a toggle pattern
