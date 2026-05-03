# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-12 — per-game prompt registry for ARC-AGI-3 LLM agent.

The single highest-ROI prompt change: ARC-AGI-3 games each have their own
mechanic vocabulary (movement, click, key/door, energy, etc.). Generic
prompts get sub-1 % win rate; the OpenAI ``GuidedLLM`` baseline with
verbatim per-game rules in the prompt is the documented community baseline
that consistently beats vanilla LLM agents.

The registry is keyed on the **game family prefix** (the 4-char prefix
before ``-`` in ``game_id``, e.g. ``ls20-0a0ad940`` → ``ls20``). When
unknown, the default prompt falls back to the generic ARC-AGI-3 context
block from the upstream ``LLM.build_user_prompt`` template.

Sources for individual rules:
- ``ls20`` (LockSmith): verbatim from
  ``arcprize/ARC-AGI-3-Agents/agents/templates/llm_agents.py:GuidedLLM``
- ``ft09``: Plank LangGraph multimodal observations (click-required)
- Generic / fallback: upstream ``LLM.build_user_prompt``
"""

from __future__ import annotations

GENERIC_CONTEXT = (
    "You are an agent playing a dynamic game. Your objective is to\n"
    "WIN and avoid GAME_OVER while minimizing actions.\n\n"
    "One action produces one Frame. One Frame is made of one or more sequential\n"
    "Grids. Each Grid is a matrix size INT<0,63> by INT<0,63> filled with\n"
    "INT<0,15> values."
)


# Sprint-19 Run #24 finding: vision-mode agent on bp35 escalated pixΔ
# from 25 → 391 → 528 → 635 across steps and triggered GAME_OVER at
# step 40. The agent had no signal that GAME_OVER is FATAL or that
# monotonic-pixΔ-growth is the empirical loss trajectory. This block is
# appended by ``build_system_prompt`` to every game's prompt so the
# agent learns the rule once for all games.
GAME_OVER_AVOIDANCE_HINT = (
    "FATAL FAILURE MODE: GAME_OVER ends the episode (score = 0, no\n"
    "recovery). Empirical evidence from prior runs on bp35: agents that\n"
    "aggressively manipulate the grid (pixΔ > 500 per step, monotonic-\n"
    "growth trajectories) reliably trigger GAME_OVER around step 35-40.\n"
    "**Massive state change is NOT automatically progress — it is often\n"
    "the path to GAME_OVER.**\n"
    "\n"
    "WIN-DETECTION RULE: if your last 5 actions changed pixels but\n"
    "``levels_completed`` did NOT increase, you are NOT winning. Try\n"
    "structurally-different actions or RESET (if available). Continuing\n"
    "in the same direction at increasing intensity is the GAME_OVER\n"
    "trajectory."
)

# Verbatim from arcprize/ARC-AGI-3-Agents agents/templates/llm_agents.py
# (GuidedLLM.build_user_prompt). This is the public baseline that
# consistently lifts win-rate by ~5-10 PP on Locksmith vs the generic
# context. Keep verbatim — phrasing has been A/B-tested upstream.
LS20_LOCKSMITH_RULES = (
    "You are playing a game called LockSmith. Rules and strategy:\n"
    "* RESET: start over, ACTION1: move up, ACTION2: move down, "
    "ACTION3: move left, ACTION4: move right "
    "(ACTION5 and ACTION6 do nothing in this game)\n"
    "* you may make one action per turn\n"
    "* your goal is find and collect a matching key then touch the exit door\n"
    "* 6 levels total, score shows which level, complete all levels to "
    "win (grid row 62)\n"
    "* start each level with limited energy. you GAME_OVER if you run "
    "out (grid row 61)\n"
    "* the player is a 4x4 square: "
    "[[X,X,X,X],[0,0,0,X],[4,4,4,X],[4,4,4,X]] "
    "where X is transparent to the background\n"
    "* the grid represents a birds-eye view of the level\n"
    "* walls are made of INT<10>, you cannot move through a wall\n"
    "* walkable floor area is INT<8>\n"
    "* you can refill energy by touching energy pills (a 2x2 of INT<6>)\n"
    "* current key is shown in bottom-left of entire grid\n"
    "* the exit door is a 4x4 square with INT<11> border\n"
    "* to find a new key shape, touch the key rotator, a 4x4 square "
    "denoted by INT<9> and INT<4> in the top-left corner of the square\n"
    "* to find a new key color, touch the color rotator, a 4x4 square "
    "denoted by INT<9> and INT<2> and in the bottom-left corner of the square\n"
    "* to rotate more than once, move 1 space away from the rotator and back on\n"
    "* continue rotating the shape and color of the key until the key matches "
    "the one inside the exit door (scaled down 2X)\n"
    "* if the grid does not change after an action, you probably tried to "
    "move into a wall\n\n"
    "An example of a good strategy observation:\n"
    "The player 4x4 made of INT<4> and INT<0> is standing below a wall of "
    "INT<10>, so I cannot move up anymore and should move left towards the "
    "rotator with INT<11>.\n"
)

# ft09 needs click-tile actions, NOT navigation. Plank's multimodal agent
# explicitly observes that A* navigation does not transfer to ft09. The
# system should favour movement actions first; if those do nothing, switch
# to click (ACTION6 with x/y coordinates).
FT09_RULES = (
    "You are playing a game in the ft09 family. Key observations:\n"
    "* This game challenges basic logic with grid manipulation\n"
    "* Try movement actions (ACTION1-4) first to understand the dynamics\n"
    "* If movement actions do not change the grid, switch to clicking\n"
    "  (ACTION6 with x/y coordinates targets a specific cell)\n"
    "* Pay attention to which tiles change colour after each click — that\n"
    "  reveals the click-effect rule\n"
    "* The goal is typically to align tiles into a target pattern\n"
)

# Click-toggle / cluster games (cn04, bp35, sk48, ar25, etc.). Symbolica
# Arcgentica's 36 % run shows these benefit from systematic exploration
# of click-coordinates, not random LLM picks.
CLICK_FAMILY_HINT = (
    "This game appears to use click-based interaction (ACTION6 with x/y).\n"
    "* Each click toggles or transforms cells in the targeted region\n"
    "* Map out the cluster structure of the grid before clicking — "
    "connected components of equal-color cells often share a toggle effect\n"
    "* If two consecutive clicks at different positions produce the same "
    "delta, the rule is position-independent (toggle a global state)\n"
    "* Watch for parity / pair-toggle structure: many games require\n"
    "  exactly N clicks of certain types to win\n"
)

# bp35-specific hints derived from Sprint-16 Run #20 + Sprint-17 Run #21 +
# Sprint-18 Run #22 audit JSONLs.
#
# Run #20 observed: ACTION3/4/7 produce ~20+ pixΔ; ACTION6 with arbitrary
# coords is a 1-pixel cursor move. Run #21 with the rule below killed the
# ACTION6-reflex (0/36) and got monotonic-pixΔ-growth → GAME_OVER at
# step 35. Run #22 (planning) found a strategic ACTION6 with pixΔ=635 at
# step 35 (the click DID do something massive when targeted) but still
# GAME_OVER. Pattern: agent engages but loses by over-engagement.
#
# Sprint-19 update: explicit win-vs-loss-vs-stuck heuristics + the rule
# that more state-change isn't automatically progress.
BP35_OBSERVED_RULES = (
    "You are playing a game in the bp35 family. Empirical observations\n"
    "from prior episodes (audit JSONL evidence):\n"
    "\n"
    "ACTION DYNAMICS:\n"
    "* Available actions are typically ACTION3, ACTION4, ACTION6, ACTION7.\n"
    "* ACTION3, ACTION4 and ACTION7 cause LARGE grid changes (~20+ pixels\n"
    "  out of 4096 changing per step) — they're the macro-mechanics.\n"
    "* ACTION6 is a CLICK at (x,y). With arbitrary coords it usually only\n"
    "  moves the cursor (1-pixel change) — but TARGETED clicks on the\n"
    "  right cell can produce 600+ pixel cascades.\n"
    "* RESET is also available: it restarts the level. Use it ONLY when\n"
    "  you've clearly broken the puzzle (irreversible bad state).\n"
    "\n"
    "WIN-VS-LOSS HEURISTICS (this is what prior agents got wrong):\n"
    "* GAME_OVER frequently comes from OVER-engagement: pixΔ trajectories\n"
    "  that grow monotonically (25 → 100 → 300 → 600+) usually end in\n"
    "  loss, not win. Massive change is NOT automatically progress.\n"
    "* The win signal is ``levels_completed`` increasing. NOTHING ELSE.\n"
    "  Pixels moving doesn't help if level stays at 0.\n"
    "* If after 5+ actions the level hasn't advanced and pixΔ keeps\n"
    "  growing, your strategy is wrong. Try fundamentally different\n"
    "  action types (switch from movement to click, or vice versa).\n"
    "\n"
    "STRATEGY:\n"
    "* Phase 1 (steps 0-4): explore — try each available action ONCE to\n"
    "  see what it does. The PNG image + cluster decomposition tell you\n"
    "  the structure.\n"
    "* Phase 2 (steps 5-15): hypothesise the win-condition by comparing\n"
    "  before/after images. What pattern do level-up events have in\n"
    "  common? (You won't have evidence yet — speculate from grid shape.)\n"
    "* Phase 3 (steps 15+): execute towards the hypothesised win-state\n"
    "  using the smallest sequence of actions. Prefer ACTION6 with\n"
    "  TARGETED coordinates over scattered movement actions.\n"
    "* If stuck, RESET and try a different opening sequence.\n"
)

# Game family → rule scaffold. Add entries as you understand new games.
GAME_PROMPTS: dict[str, str] = {
    "ls20": LS20_LOCKSMITH_RULES,
    "ft09": FT09_RULES,
    # Sprint-17: bp35 has its own observed-behaviour hint (replaces the
    # generic CLICK_FAMILY_HINT after we've seen real bp35 episodes).
    "bp35": BP35_OBSERVED_RULES,
    "cn04": CLICK_FAMILY_HINT,
    "sk48": CLICK_FAMILY_HINT,
    "ar25": CLICK_FAMILY_HINT,
    "tn36": CLICK_FAMILY_HINT,
    "wa30": CLICK_FAMILY_HINT,
    "re86": CLICK_FAMILY_HINT,
    "lp85": CLICK_FAMILY_HINT,
    "sc25": CLICK_FAMILY_HINT,
    "su15": CLICK_FAMILY_HINT,
    "lf52": CLICK_FAMILY_HINT,
    "r11l": CLICK_FAMILY_HINT,
    "s5i5": CLICK_FAMILY_HINT,
    "m0r0": CLICK_FAMILY_HINT,
}


def game_prefix(game_id: str) -> str:
    """Extract the 4-char family prefix from a game-id like ``ls20-abc123``."""
    return game_id.split("-", 1)[0] if "-" in game_id else game_id


def build_system_prompt(game_id: str, action_options: str) -> str:
    """Compose the system prompt for a given game.

    Layers:
    1. Generic ARC-AGI-3 context (verbatim from upstream LLM template)
    2. Per-game rules if known (GuidedLLM-style)
    3. JSON output schema with the actual whitelisted actions

    The action whitelist is critical: many ARC-AGI-3 games do NOT accept
    every GameAction value. Embedding the whitelist in the system prompt
    eliminates the LLM's most common failure mode (calling an unavailable
    action).
    """
    prefix = game_prefix(game_id)
    game_rules = GAME_PROMPTS.get(prefix, "")
    behavioural = (
        "Behavioural guidelines:\n"
        "* Explore the entire environment thoroughly before committing.\n"
        "* Before taking an action, think about the state of the environment.\n"
        "* If an action repeatedly does nothing, try something else.\n"
        "* If your plan is not working, reflect on what rule you may have "
        "misunderstood.\n"
    )
    output_schema = (
        f"After your reasoning, output ONLY a JSON object with two fields:\n"
        f"  action: must be exactly one of [{action_options}]\n"
        f"  reasoning: one short sentence describing why you chose it"
    )
    parts = [GENERIC_CONTEXT, GAME_OVER_AVOIDANCE_HINT, behavioural]
    if game_rules:
        parts.append(game_rules)
    parts.append(output_schema)
    return "\n\n".join(parts)


__all__ = [
    "BP35_OBSERVED_RULES",
    "CLICK_FAMILY_HINT",
    "FT09_RULES",
    "GAME_OVER_AVOIDANCE_HINT",
    "GAME_PROMPTS",
    "GENERIC_CONTEXT",
    "LS20_LOCKSMITH_RULES",
    "build_system_prompt",
    "game_prefix",
]
