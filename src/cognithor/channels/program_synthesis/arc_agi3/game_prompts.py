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

# bp35-specific hints derived from Sprint-16 Run #20 observations on this
# very game-family. Empirical (40-step LLM trace + per-step pixΔ from the
# audit JSONL):
#
# * Available actions = [ACTION3, ACTION4, ACTION6, ACTION7]
# * ACTION3 / ACTION4 / ACTION7 produce LARGE state changes (pixΔ ≈ 19-25
#   per step on the 64×64 grid)
# * ACTION6 with arbitrary coords moves only the cursor pixel (pixΔ = 1)
# * Greedy LLMs default to ACTION6 every step → 100 % loop → score 0
#
# Strategy this hint pushes: try the non-click actions FIRST to understand
# the macro dynamics. Reserve ACTION6 for when you have a concrete cell
# you want to commit-click on (after a non-click action has revealed the
# board's structure).
BP35_OBSERVED_RULES = (
    "You are playing a game in the bp35 family. Observed behaviour from\n"
    "prior episodes:\n"
    "* Available actions are typically ACTION3, ACTION4, ACTION6, ACTION7.\n"
    "* ACTION3, ACTION4 and ACTION7 cause LARGE grid changes (~20+ pixels\n"
    "  out of 4096 changing per step) — they advance the game state.\n"
    "* ACTION6 is the click action (takes x/y coordinates). Without a\n"
    "  specific target cell it usually only moves the cursor (1-pixel\n"
    "  change) and does NOT advance the game.\n"
    "* Strategy: use ACTION3 / ACTION4 / ACTION7 first to discover the\n"
    "  game's transformation rules; reserve ACTION6 only for committing a\n"
    "  click on a cell you have a concrete reason to target.\n"
    "* The ``pixels_changed`` field in your action history shows which\n"
    "  actions actually moved the game forward — favour repeating actions\n"
    "  with large pixΔ over ACTION6 with pixΔ ≤ 1.\n"
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
    parts = [GENERIC_CONTEXT, behavioural]
    if game_rules:
        parts.append(game_rules)
    parts.append(output_schema)
    return "\n\n".join(parts)


__all__ = [
    "BP35_OBSERVED_RULES",
    "CLICK_FAMILY_HINT",
    "FT09_RULES",
    "GAME_PROMPTS",
    "GENERIC_CONTEXT",
    "LS20_LOCKSMITH_RULES",
    "build_system_prompt",
    "game_prefix",
]
