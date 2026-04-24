# 04 · Guardrails

Guardrails prüfen jeden Task-Output **bevor** er den nächsten Task oder den User erreicht. Schlägt ein Guardrail fehl, wird der Task mit dem Feedback des Guardrails automatisch erneut ausgeführt — bis zu `max_retries` Mal.

**Voraussetzungen**
- Abgeschlossen: [01 · Erste Crew](01-first-crew.md)

**Zeitbedarf:** 5 Minuten
**Endzustand:** Du weißt, wie du `no_pii()`, `word_count()` und `chain()` kombinierst, und wie Retry-Loops aussehen.

---

## 1. Built-in Guardrails

`cognithor.crew.guardrails.builtin` bringt vier Standard-Validatoren mit:

| Guardrail                              | Zweck                                                                    |
|----------------------------------------|--------------------------------------------------------------------------|
| `no_pii()`                             | Blockiert Emails, IBANs, DE-Telefonnummern, Steuer-IDs                   |
| `word_count(min_words=…, max_words=…)` | Erzwingt Längen-Budgets                                                  |
| `schema(MyPydanticModel)`              | Erzwingt JSON-Schema-Konformität                                         |
| `hallucination_check(reference=…)`     | Heuristische Prüfung gegen einen Referenz-Text                           |

Plus der Kombinator `chain(g1, g2, ...)` — führt Guards der Reihe nach aus, erste Fehlschlag kurzschließt.

## 2. Ein Guardrail anhängen

```python
from cognithor.crew import CrewAgent, CrewTask
from cognithor.crew.guardrails import chain, no_pii, word_count

reporter = CrewAgent(
    role="Reporter",
    goal="Schreibe einen knappen Report",
    llm="ollama/qwen3:8b",
)

report = CrewTask(
    description="Fasse die Research-Ergebnisse in ≤ 100 Wörtern zusammen.",
    expected_output="Markdown, maximal 100 Wörter, keine PII.",
    agent=reporter,
    guardrail=chain(no_pii(), word_count(max_words=100)),
    max_retries=2,
)
```

**Retry-Verhalten:**
- Guardrail fail → Task wird **inklusive des Guardrail-Feedbacks im Prompt** erneut ausgeführt.
- Nach `max_retries` (default 2) wiederholten Fehlern wird `GuardrailFailure` geworfen.

## 3. Eigene Guardrails

Ein Guardrail ist einfach eine Funktion `(TaskOutput) -> GuardrailResult`:

```python
from cognithor.crew.guardrails.base import GuardrailResult


def no_emojis(output) -> GuardrailResult:
    emoji_ranges = [(0x1F300, 0x1F9FF), (0x2600, 0x27BF)]
    for ch in output.raw:
        cp = ord(ch)
        if any(lo <= cp <= hi for lo, hi in emoji_ranges):
            return GuardrailResult(
                passed=False,
                feedback="Bitte ohne Emojis antworten.",
            )
    return GuardrailResult(passed=True, feedback=None)
```

Einhängen:

```python
report = CrewTask(
    ...,
    guardrail=chain(no_pii(), word_count(max_words=100), no_emojis),
)
```

## 4. Audit-Chain

Jeder Guardrail-Verdict landet in der **Hashline-Guard Audit-Chain** mit PII-Flag. Du findest den Hash-Trail später im Event-Bus oder unter `~/.cognithor/audit/`. Kein zusätzlicher Setup nötig.

## 5. Lauffähiges Beispiel

[`examples/quickstart/04_guardrails/`](../../examples/quickstart/04_guardrails/) zeigt:

1. Eine Task mit `chain(no_pii(), word_count(max_words=10))`, die bei PII oder zu langem Output per Feedback einen Retry auslöst.
2. Eine zweite Task, die nach dreimaligem Fehlschlag `GuardrailFailure` wirft — mit gemocktem Planner, damit kein Ollama gebraucht wird.

---

**Next:** [05 · Deployment](05-deployment.md)
