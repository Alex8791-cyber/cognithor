# 03 · Erster Skill

**Tool vs. Skill — der Unterschied in einem Satz:**

> Ein **Tool** ist ein einzelner Call ("zähle Wörter"). Ein **Skill** ist eine kuratierte Kombination aus System-Prompt, Trigger-Keywords und einer passenden Tool-Liste — ein wiederverwendbares "Rezept".

**Voraussetzungen**
- Abgeschlossen: [02 · Eigenes Tool](02-first-tool.md)

**Zeitbedarf:** 5 Minuten
**Endzustand:** Ein `@agent`-dekorierter Skill ist registriert, hat eine definierte Persona, eigene Tools und reagiert auf Keywords.

---

## 1. Der `@agent`-Decorator als Skill-Primitive

`cognithor.sdk` bietet `@agent` als Ein-Decorator-Skill-Definition:

```python
from cognithor.sdk import agent, tool


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
    async def on_message(self, message: str) -> str:
        return f"Math-Helper bearbeitet: {message}"
```

**Kernzutaten eines Skills:**

| Feld                | Zweck                                                          |
|---------------------|----------------------------------------------------------------|
| `name`              | Eindeutige ID, per `SDKRegistry` auffindbar                    |
| `tools`             | Welche Tools der Skill aufrufen darf                           |
| `system_prompt`     | Persona — wie der Skill denkt und formuliert                   |
| `trigger_keywords`  | Wann der Skill-Router diesen Skill aktiviert                   |
| `max_iterations`    | Sicherheits-Limit für die Tool-Use-Schleife (default: 5)       |

## 2. Skill entdecken

```python
from cognithor.sdk.decorators import get_registry

registry = get_registry()
defn = registry.get_agent("math_helper")
print(defn.name, defn.tools, defn.trigger_keywords)
# → math_helper ['calculate_gcd'] ['ggT', 'größter gemeinsamer Teiler', 'GCD']
```

## 3. Vollständiges Beispiel

Leg `main.py` an — siehe [`examples/quickstart/03_first_skill/main.py`](../../examples/quickstart/03_first_skill/main.py).

## 4. Skill-Scaffolder

Für schnellere Gerüste:

```python
from cognithor.sdk import scaffold_agent

src = scaffold_agent(
    name="my_skill",
    description="Mein Skill",
    keywords=["hallo", "welt"],
)
print(src)  # → vollständige Python-Datei mit @tool + @agent + @hook
```

Das Ergebnis ist eine direkt lauffähige Python-Datei, fertig zum Anpassen.

## 5. Was ist der Unterschied zur Community Skill Marketplace?

Die **Community Skill Marketplace** (`cognithor.skills.community.*`) ist die **hostete Distribution**: veröffentlichte, SHA-256-verifizierte Skills mit Publisher-Vertrauenslevels, Ratings und REST-API. Alles was du mit `@agent` baust, kannst du dort später publishen.

Die SDK-Ebene (`@agent`-Decorator) ist der **Entwickler-Einstieg**: lokal registrieren, lokal testen, dann optional publishen.

---

**Next:** [04 · Guardrails](04-guardrails.md)
