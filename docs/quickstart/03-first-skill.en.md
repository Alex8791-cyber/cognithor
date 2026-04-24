# 03 · First Skill

**Tool vs. Skill — the difference in one sentence:**

> A **tool** is a single call ("count words"). A **skill** is a curated bundle of a system prompt, trigger keywords, and a matching tool list — a reusable "recipe".

**Prerequisites**
- Completed: [02 · First Tool](02-first-tool.en.md)

**Time:** 5 minutes
**End state:** A `@agent`-decorated skill is registered with a defined persona, a tool list, and keyword triggers.

---

## 1. The `@agent` decorator as skill primitive

`cognithor.sdk` offers `@agent` as a one-decorator skill definition:

```python
from cognithor.sdk import agent, tool


@tool(name="calculate_gcd", description="Compute the greatest common divisor")
async def calculate_gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


@agent(
    name="math_helper",
    description="Math skill — computes GCD and other number-theory tasks",
    tools=["calculate_gcd"],
    system_prompt="You are a precise math assistant. Answer tersely.",
    trigger_keywords=["gcd", "greatest common divisor"],
    version="0.1.0",
)
class MathHelperAgent:
    async def on_message(self, message: str) -> str:
        return f"Math-Helper processing: {message}"
```

**Skill core ingredients:**

| Field               | Purpose                                                         |
|---------------------|-----------------------------------------------------------------|
| `name`              | Unique ID, discoverable via `SDKRegistry`                       |
| `tools`             | Which tools this skill is allowed to call                       |
| `system_prompt`     | Persona — how the skill thinks and speaks                       |
| `trigger_keywords`  | When the skill router activates this skill                      |
| `max_iterations`    | Safety cap on the tool-use loop (default: 5)                    |

## 2. Discover the skill

```python
from cognithor.sdk.decorators import get_registry

registry = get_registry()
defn = registry.get_agent("math_helper")
print(defn.name, defn.tools, defn.trigger_keywords)
# → math_helper ['calculate_gcd'] ['gcd', 'greatest common divisor']
```

## 3. Full example

Create `main.py` — see [`examples/quickstart/03_first_skill/main.py`](../../examples/quickstart/03_first_skill/main.py).

## 4. Skill scaffolder

For faster scaffolding:

```python
from cognithor.sdk import scaffold_agent

src = scaffold_agent(
    name="my_skill",
    description="My skill",
    keywords=["hello", "world"],
)
print(src)  # → a complete Python file with @tool + @agent + @hook
```

The result is a ready-to-run Python file, ready for customization.

## 5. What's the difference from the Community Skill Marketplace?

The **Community Skill Marketplace** (`cognithor.skills.community.*`) is the **hosted distribution layer**: published, SHA-256-verified skills with publisher trust levels, ratings, and REST API. Anything you build with `@agent` can later be published there.

The SDK layer (`@agent` decorator) is the **developer entry point**: register locally, test locally, optionally publish.

---

**Next:** [04 · Guardrails](04-guardrails.en.md)
