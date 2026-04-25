# cognithor-bench — Reproducible Multi-Agent Benchmark Scaffold

In-monorepo benchmark harness for Cognithor. Independent of `agbench`;
focuses on Cognithor-native scenarios with the option to compare against
AutoGen via the source-compat shim from `cognithor.compat.autogen`.

## Status

`v0.1.0` ships **scaffold + smoke-test only** with Cognithor v0.94.0.
GAIA / WebArena / AssistantBench scenario adapters are post-v0.94.0.

## Install

```bash
pip install -e .
# Optional: pull the AutoGen adapter dependency
pip install -e ".[autogen]"
```

## Usage

```bash
# Run the default smoke-test (3 trivial tasks)
cognithor-bench run src/cognithor_bench/scenarios/smoke_test.jsonl

# With repetition + sub-sampling
cognithor-bench run scenarios/foo.jsonl --repeat 5 --subsample 0.5

# Pick the AutoGen adapter (requires [autogen] extra)
cognithor-bench run scenarios/foo.jsonl --adapter autogen

# Pick a specific Ollama model
cognithor-bench run scenarios/foo.jsonl --model ollama/qwen3:8b

# Native execution (default) vs Docker isolation (opt-in)
cognithor-bench run scenarios/foo.jsonl --native     # default
cognithor-bench run scenarios/foo.jsonl --docker     # opt-in

# Aggregate a results directory into a Markdown table
cognithor-bench tabulate results/
```

## Scenario format (JSONL — one task per line)

```json
{"id": "smoke-001", "task": "Was ist 2+2?", "expected": "4", "timeout_sec": 30, "requires": ["no_network"]}
```

Fields:
- `id` — short identifier (used for result aggregation).
- `task` — natural-language prompt.
- `expected` — exact-match string OR substring (matched case-insensitively).
- `timeout_sec` — per-task timeout.
- `requires` — list of capability tags (`no_network`, `ollama`, `pdf-tools`, ...).

## Adding a new scenario file

Drop a JSONL file under `src/cognithor_bench/scenarios/`. Each line is a
discrete task. Run:

```bash
cognithor-bench run src/cognithor_bench/scenarios/my_new_set.jsonl
```

## License

Apache 2.0. See repo-root `LICENSE`.
