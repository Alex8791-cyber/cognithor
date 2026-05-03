"""ATL system prompt builder and JSON response parser."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# System prompt template (German, targeting Qwen3.5)
# ---------------------------------------------------------------------------

ATL_SYSTEM_PROMPT = """\
Du befindest dich im autonomen Denkmodus (ATL — Autonomous Thought Loop).
Dies ist Zyklus #{cycle_number}. Du laeuft kontinuierlich — jeder Zyklus
baut auf den vorherigen auf. Du bist NICHT in einer Initialisierungsphase,
es sei denn es ist tatsaechlich Zyklus #1.

{identity}

Aktueller Zeitpunkt: {now}

## Deine aktiven Ziele

{goals_formatted}

## Bisherige Aktivitaet (letzte Zyklen)

{recent_events}

## Vorhandenes Wissen zu den Zielen

{goal_knowledge}

## Deine Aufgabe

Baue auf deiner bisherigen Arbeit auf. Wiederhole NICHT was du bereits
recherchiert hast — suche nach NEUEN Aspekten und Luecken.

Fuer jedes Ziel: Was fehlt noch? Welche Teilaspekte sind unterbelichtet?
Schlage gezielte Recherchen vor die dein Wissen tatsaechlich erweitern.

Du darfst maximal {max_actions} Aktionen vorschlagen.

Antworte ausschliesslich mit einem JSON-Objekt im folgenden Format
(keine Erklaerungen ausserhalb des JSON):

{{
  "summary": "Was hast du ueberlegt und was ist der naechste Schritt?",
  "goal_evaluations": [
    {{"goal_id": "g_XXX", "progress_delta": 0.0, "note": "..."}}
  ],
  "proposed_actions": [
    {{"type": "research|notify|tool", "params": {{}}, "rationale": "..."}}
  ],
  "wants_to_notify": false,
  "notification": null,
  "priority": "low|medium|high"
}}
"""


# ---------------------------------------------------------------------------
# build_atl_prompt
# ---------------------------------------------------------------------------


def build_atl_prompt(
    identity: str,
    goals_formatted: str,
    recent_events: str,
    goal_knowledge: str,
    now: str,
    max_actions: int,
    cycle_number: int = 1,
) -> str:
    """Format the ATL system prompt with the given context."""
    return ATL_SYSTEM_PROMPT.format(
        identity=identity,
        goals_formatted=goals_formatted,
        recent_events=recent_events,
        goal_knowledge=goal_knowledge,
        now=now,
        max_actions=max_actions,
        cycle_number=cycle_number,
    )


# ---------------------------------------------------------------------------
# AutonomousThought dataclass
# ---------------------------------------------------------------------------


@dataclass
class AutonomousThought:
    """Parsed result of one ATL thinking cycle."""

    summary: str = ""
    goal_evaluations: list[dict[str, Any]] = field(default_factory=list)
    proposed_actions: list[dict[str, Any]] = field(default_factory=list)
    wants_to_notify: bool = False
    notification: str | None = None
    priority: str = "low"


# ---------------------------------------------------------------------------
# parse_atl_response
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_atl_response(raw: str) -> AutonomousThought:
    """Parse an LLM response into an *AutonomousThought*.

    Handles:
    - Qwen3 ``<think>...</think>`` tags
    - Markdown ````` json ```` code blocks
    - Bare JSON objects
    - Partial / missing keys (filled with defaults)
    - Complete parse failure (returns empty thought)
    """
    # 1. Strip think tags
    text = _THINK_RE.sub("", raw).strip()

    # 2. Strip markdown code blocks
    m = _CODE_BLOCK_RE.search(text)
    if m:
        text = m.group(1).strip()

    # 3. Try direct parse
    import contextlib

    data: dict[str, Any] | None = None
    with contextlib.suppress(json.JSONDecodeError, ValueError):
        data = json.loads(text)

    # 4. Fallback: extract first { ... } block
    if data is None:
        m2 = _JSON_OBJECT_RE.search(text)
        if m2:
            with contextlib.suppress(json.JSONDecodeError, ValueError):
                data = json.loads(m2.group(0))

    # 5. Build dataclass with defaults for missing keys
    if not isinstance(data, dict):
        return AutonomousThought()

    return AutonomousThought(
        summary=data.get("summary", ""),
        goal_evaluations=data.get("goal_evaluations", []),
        proposed_actions=data.get("proposed_actions", []),
        wants_to_notify=data.get("wants_to_notify", False),
        notification=data.get("notification"),
        priority=data.get("priority", "low"),
    )
