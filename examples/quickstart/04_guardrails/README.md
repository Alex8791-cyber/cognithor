# 04 · Guardrails — runnable example

Shows retry-with-feedback and `GuardrailFailure`. Walks through
[`docs/quickstart/04-guardrails.md`](../../../docs/quickstart/04-guardrails.md).

## Run it

```bash
pip install -r requirements.txt
python main.py
```

## Smoke-test it

```bash
pip install pytest pytest-asyncio
python -m pytest test_example.py -v
```

The smoke test covers both branches:
- First attempt fails `word_count(max_words=10)`, second attempt passes.
- All attempts fail → `GuardrailFailure` is raised after `max_retries + 1`
  tries.
