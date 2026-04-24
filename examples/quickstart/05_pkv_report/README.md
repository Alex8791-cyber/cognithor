# 05 · PKV-Report — runnable example

The complete PKV (Private Krankenversicherung) scenario from spec §1.4:
Analyst + Writer run sequentially, the writer produces a client-ready
Markdown report, the result is persisted via `output_file=`.

Same pattern as `01_first_crew` but with the DACH-specific PKV persona and
two-stage output aggregation (token totals across tasks, trace ID surfaced).

## Run it

```bash
pip install -r requirements.txt
ollama pull qwen3:8b
ollama pull qwen3:32b    # for the analyst
python main.py
# → writes output/pkv_report.md
```

## Smoke-test it

```bash
pip install pytest pytest-asyncio
python -m pytest test_example.py -v
```

Uses a mocked Planner — no Ollama required in CI. Verifies:
- Both tasks execute.
- Aggregate token count adds up across tasks.
- Final `result.raw` contains the writer's output.
- `trace_id` is populated.
