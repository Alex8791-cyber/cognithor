# 02 · First Tool

Tools are the verb primitives of an agent — "write file", "search web", "count words". On this page you register your first custom tool through the Cognithor Agent SDK.

**Prerequisites**
- Completed: [01 · First Crew](01-first-crew.en.md)

**Time:** 5 minutes
**End state:** A `@tool`-decorated Python callable is registered and discoverable in the SDK registry.

---

## 1. The `@tool` decorator

Cognithor ships an Agent SDK at `cognithor.sdk`. The `@tool` decorator registers a function with schema inference from the type signature:

```python
from cognithor.sdk import tool


@tool(name="word_count", description="Count words in a text")
async def word_count(text: str) -> int:
    return len(text.split())
```

That's it. No schema files, no JSON-Schema boilerplate — the signature (`text: str`, return type `int`) is automatically mapped to JSON Schema.

## 2. Find the tool in the SDK registry

```python
from cognithor.sdk.decorators import get_registry

registry = get_registry()
defn = registry.get_tool("word_count")
print(defn.name, defn.input_schema)
# → word_count {'type': 'object', 'properties': {'text': {'type': 'string'}}, 'required': ['text']}
```

## 3. Full example

Create `main.py`:

```python
"""First Tool — register a custom @tool via the Cognithor SDK."""

from __future__ import annotations

import asyncio

from cognithor.sdk import tool
from cognithor.sdk.decorators import get_registry


@tool(name="word_count", description="Count words in a text")
async def word_count(text: str) -> int:
    return len(text.split())


async def main() -> None:
    # Callable directly — tools are regular coroutines.
    result = await word_count("Hello Cognithor world")
    print(f"word_count: {result}")

    # Discoverable via the SDK registry (for discovery / tool catalog).
    registry = get_registry()
    defn = registry.get_tool("word_count")
    assert defn is not None
    print(f"Registered: {defn.name} — {defn.description}")
    print(f"Input schema: {defn.input_schema}")


if __name__ == "__main__":
    asyncio.run(main())
```

Run it:

```bash
python main.py
# → word_count: 3
# → Registered: word_count — Count words in a text
# → Input schema: {'type': 'object', 'properties': ..., 'required': ['text']}
```

## 4. Runnable copy in the repo

[`examples/quickstart/02_first_tool/`](../../examples/quickstart/02_first_tool/) contains the same script plus a `test_example.py`.

## 5. Using tools inside a Crew

`CrewAgent.tools` and `CrewTask.tools` reference **names from the MCP tool registry**, not directly from the SDK registry. In the current release MCP tools are synchronized into the Crew-side registry at gateway startup (`ToolRegistryDB.sync_from_mcp`). A full SDK-to-MCP-registry bridge for custom SDK tools is planned for a future minor release (see GitHub issue tracker).

**Usable today:** direct SDK tool calls (as shown above), registering custom MCP tools via the gateway endpoint, or referencing the existing 145+ MCP tools inside `CrewAgent.tools=["web_search", ...]`.

---

**Next:** [03 · First Skill](03-first-skill.en.md)
