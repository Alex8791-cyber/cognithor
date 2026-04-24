"""First Skill — bundle a tool + persona + trigger keywords with @agent."""

from __future__ import annotations

import asyncio

from cognithor.sdk import agent, tool
from cognithor.sdk.decorators import get_registry


@tool(name="calculate_gcd", description="Berechne größten gemeinsamen Teiler")
async def calculate_gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


@agent(
    name="math_helper",
    description="Mathe-Skill — berechnet ggT und andere Zahlentheorie-Aufgaben",
    tools=["calculate_gcd"],
    system_prompt="Du bist ein präziser Mathematik-Assistent. Antworte knapp.",
    trigger_keywords=["ggT", "größter gemeinsamer Teiler", "GCD"],
    version="0.1.0",
)
class MathHelperAgent:
    """Skill persona — reagiert auf Mathe-Anfragen."""

    async def on_message(self, message: str) -> str:
        return f"Math-Helper bearbeitet: {message}"


async def main() -> None:
    # Direkter Tool-Call
    gcd = await calculate_gcd(48, 18)
    print(f"ggT(48, 18) = {gcd}")

    # Skill in der Registry finden
    registry = get_registry()
    defn = registry.get_agent("math_helper")
    assert defn is not None
    print(f"Skill: {defn.name}")
    print(f"  Tools: {defn.tools}")
    print(f"  Trigger: {defn.trigger_keywords}")
    print(f"  Persona: {defn.system_prompt!r}")

    # Skill-Instanz aufrufen
    skill = MathHelperAgent()
    response = await skill.on_message("Berechne ggT von 12 und 8")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
