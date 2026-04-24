# 04 · Guardrails

Guardrails validate every task output **before** it reaches the next task or the user. If a guardrail fails, the task is automatically retried with the guardrail's feedback — up to `max_retries` times.

**Prerequisites**
- Completed: [01 · First Crew](01-first-crew.en.md)

**Time:** 5 minutes
**End state:** You know how to combine `no_pii()`, `word_count()`, and `chain()`, and you understand the retry loop.

---

## 1. Built-in guardrails

`cognithor.crew.guardrails.builtin` ships four standard validators:

| Guardrail                              | Purpose                                                                  |
|----------------------------------------|--------------------------------------------------------------------------|
| `no_pii()`                             | Blocks emails, German IBANs, German phone numbers, Steuer-IDs            |
| `word_count(min_words=…, max_words=…)` | Enforces length budgets                                                  |
| `schema(MyPydanticModel)`              | Enforces JSON Schema conformance                                         |
| `hallucination_check(reference=…)`     | Heuristic check against a reference corpus                               |

Plus the combinator `chain(g1, g2, ...)` — runs guards in order, first failure short-circuits.

## 2. Attach a guardrail

```python
from cognithor.crew import CrewAgent, CrewTask
from cognithor.crew.guardrails import chain, no_pii, word_count

reporter = CrewAgent(
    role="Reporter",
    goal="Write a tight report",
    llm="ollama/qwen3:8b",
)

report = CrewTask(
    description="Summarize the research in ≤ 100 words.",
    expected_output="Markdown, max 100 words, no PII.",
    agent=reporter,
    guardrail=chain(no_pii(), word_count(max_words=100)),
    max_retries=2,
)
```

**Retry behavior:**
- Guardrail fail → task is re-executed **with the guardrail feedback injected into the prompt**.
- After `max_retries` (default 2) consecutive failures, `GuardrailFailure` is raised.

## 3. Custom guardrails

A guardrail is just a function `(TaskOutput) -> GuardrailResult`:

```python
from cognithor.crew.guardrails.base import GuardrailResult


def no_emojis(output) -> GuardrailResult:
    emoji_ranges = [(0x1F300, 0x1F9FF), (0x2600, 0x27BF)]
    for ch in output.raw:
        cp = ord(ch)
        if any(lo <= cp <= hi for lo, hi in emoji_ranges):
            return GuardrailResult(
                passed=False,
                feedback="Please respond without emojis.",
            )
    return GuardrailResult(passed=True, feedback=None)
```

Wire it in:

```python
report = CrewTask(
    ...,
    guardrail=chain(no_pii(), word_count(max_words=100), no_emojis),
)
```

## 4. Audit chain

Every guardrail verdict is recorded in the **Hashline-Guard audit chain** with a PII flag. The hash trail lives in the event bus or under `~/.cognithor/audit/`. No extra setup required.

## 5. Runnable example

[`examples/quickstart/04_guardrails/`](../../examples/quickstart/04_guardrails/) demonstrates:

1. A task with `chain(no_pii(), word_count(max_words=10))` that triggers a retry-with-feedback when PII or too-long output appears.
2. A second task that raises `GuardrailFailure` after three consecutive failures — with a mocked planner so no Ollama is needed.

---

**Next:** [05 · Deployment](05-deployment.en.md)
