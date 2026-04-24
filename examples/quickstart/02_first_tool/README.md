# 02 · First Tool — runnable example

Register a custom `@tool` via the Cognithor SDK — walks through
[`docs/quickstart/02-first-tool.md`](../../../docs/quickstart/02-first-tool.md).

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

No Ollama or network access required — the tool is pure Python.
