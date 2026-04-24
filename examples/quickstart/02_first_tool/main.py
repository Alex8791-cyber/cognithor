"""First Tool — register a custom @tool via the Cognithor SDK."""

from __future__ import annotations

import asyncio

from cognithor.sdk import tool
from cognithor.sdk.decorators import get_registry


@tool(name="word_count", description="Zähle Wörter in einem Text")
async def word_count(text: str) -> int:
    return len(text.split())


async def main() -> None:
    # Direkt aufrufbar — Tools sind ganz normale Coroutinen.
    result = await word_count("Hallo Cognithor Welt")
    print(f"word_count: {result}")

    # Via SDK-Registry entdeckbar (für Discovery / Tool-Katalog).
    registry = get_registry()
    defn = registry.get_tool("word_count")
    assert defn is not None
    print(f"Registered: {defn.name} — {defn.description}")
    print(f"Input schema: {defn.input_schema}")


if __name__ == "__main__":
    asyncio.run(main())
