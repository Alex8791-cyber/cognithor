# 02 · Eigenes Tool

Tools sind die Verb-Primitive eines Agenten — "schreibe Datei", "suche Web", "zähle Wörter". In dieser Seite registrierst du dein erstes eigenes Tool über das Cognithor-Agent-SDK.

**Voraussetzungen**
- Abgeschlossen: [01 · Erste Crew](01-first-crew.md)

**Zeitbedarf:** 5 Minuten
**Endzustand:** Ein `@tool`-dekoriertes Python-Callable ist registriert und in der SDK-Registry auffindbar.

---

## 1. Der `@tool`-Decorator

Cognithor hat ein Agent-SDK unter `cognithor.sdk`. Der `@tool`-Decorator registriert eine Funktion mit Schema-Inferenz aus der Typ-Signatur:

```python
from cognithor.sdk import tool


@tool(name="word_count", description="Zähle Wörter in einem Text")
async def word_count(text: str) -> int:
    return len(text.split())
```

Das war's. Keine Schema-Files, kein JSON-Schema-Geschreibe — die Signatur (`text: str`, Return-Type `int`) wird automatisch zu JSON-Schema.

## 2. Tool in der SDK-Registry finden

```python
from cognithor.sdk.decorators import get_registry

registry = get_registry()
defn = registry.get_tool("word_count")
print(defn.name, defn.input_schema)
# → word_count {'type': 'object', 'properties': {'text': {'type': 'string'}}, 'required': ['text']}
```

## 3. Vollständiges Beispiel

Leg `main.py` an:

```python
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
```

Ausführen:

```bash
python main.py
# → word_count: 3
# → Registered: word_count — Zähle Wörter in einem Text
# → Input schema: {'type': 'object', 'properties': ..., 'required': ['text']}
```

## 4. Lauffähige Version im Repo

[`examples/quickstart/02_first_tool/`](../../examples/quickstart/02_first_tool/) enthält das gleiche Skript plus `test_example.py`.

## 5. Tools in einer Crew verwenden

`CrewAgent.tools` und `CrewTask.tools` referenzieren **Namen aus der MCP-Tool-Registry**, nicht direkt aus der SDK-Registry. Im aktuellen Release werden MCP-Tools automatisch beim Gateway-Start in die Crew-seitige Registry synchronisiert (`ToolRegistryDB.sync_from_mcp`). Für eigene SDK-Tools ist die vollständige Bridge zur MCP-Registry in einem kommenden Minor-Release geplant (siehe GitHub-Issue-Tracker).

**Heute bereits nutzbar:** direkte SDK-Tool-Aufrufe (wie oben gezeigt), eigene MCP-Tools über den Gateway-Endpoint registrieren oder die bestehenden 145+ MCP-Tools in `CrewAgent.tools=["web_search", ...]` referenzieren.

---

**Next:** [03 · Erster Skill](03-first-skill.md)
