# 01 · First Crew — runnable example

The sequential researcher + reporter pattern from
[`docs/quickstart/01-first-crew.md`](../../../docs/quickstart/01-first-crew.md),
as a standalone mini-project.

## Run it

```bash
pip install -r requirements.txt
ollama pull qwen3:8b    # one-time
python main.py
```

## Smoke-test it (no Ollama needed)

```bash
pip install pytest pytest-asyncio
python -m pytest test_example.py -v
```

The smoke test monkeypatches the default planner + tool registry so it runs
in CI without any real LLM backend.
