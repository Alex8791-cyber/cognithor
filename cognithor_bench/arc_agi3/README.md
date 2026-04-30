# `cognithor_bench/arc_agi3/` — ARC-AGI-3 Eval Set

This directory holds the **PSE-Phase-1 benchmark fixture set** —
not the eval-suite code (which lives under
`tests/test_channels/test_program_synthesis/eval/`), only the data the
harness consumes.

The harness is **skipped** until this directory contains a real
`manifest.json`. Filling it in is the spec **D5** acceptance work.

## Layout

```
cognithor_bench/arc_agi3/
├── README.md             # this file
├── manifest.json         # MISSING — committing it switches D5 ON
├── manifest.example.json # the schema, for reference
├── tasks/                # one *.json per task
│   ├── 0001_rotate.json
│   ├── 0002_recolor.json
│   └── ...
└── runs/                 # one sub-dir per benchmark run (gitignored)
    └── 2026-04-30T12:00:00Z/
        ├── pse_results.json
        ├── baseline_results.json
        └── summary.json
```

## Manifest schema (`manifest.json`)

```json
{
  "version": "1.2.0",
  "subsets": {
    "train": {
      "n": 100,
      "diversity": {"geom": 30, "color": 30, "object": 25, "mixed": 15},
      "task_files": ["tasks/0001_rotate.json", "..."]
    },
    "held_out": {
      "n": 30,
      "task_files": ["tasks/0101_xxx.json", "..."]
    }
  },
  "baseline": {
    "name": "v0.78 NumPy solver",
    "results_path": "baselines/baseline_v0.78.json"
  }
}
```

The schema is enforced by
`tests/test_channels/test_program_synthesis/eval/_loader.py:load_manifest`
— if a key is missing or a referenced task file doesn't exist, the
loader raises `EvalManifestError` with the exact line that broke.

## Task JSON shape

Each task file matches the same shape `cognithor pse run` already
accepts (see `cli/pse_cli.py`):

```json
{
  "examples": [
    {"input": [[0, 1], [1, 0]], "output": [[1, 0], [0, 1]]},
    ...
  ],
  "budget": {"max_depth": 4, "wall_clock_seconds": 30.0}
}
```

The harness reuses `_spec_from_payload` from the CLI module, so
adding a task means dropping a JSON file in `tasks/` and listing it
in the manifest — no harness change needed.

## How a run produces numbers

1. `make benchmark` (or the eval-suite test in `slow` mode) loads
   the manifest.
2. For each subset, every task is fed through both the PSE channel
   (`ProgramSynthesisChannel.synthesize`) and the baseline solver
   pre-recorded under `baseline_v0.78.json`.
3. The metrics aggregator writes
   `cognithor_bench/arc_agi3/runs/<timestamp>/summary.json` with the
   four spec §18.3 metrics (`Solved@30s`, `Solved@5s`,
   `Median-Time-Solved`, `FP-Rate`) and a hardware fingerprint.
4. The numbers are then promoted into
   `docs/channels/program_synthesis/benchmarks.md` *Latest run* table
   by hand (Phase 2 may automate the promotion).

## Why this is empty today

The 100-task curated train set + 30-task held-out set are the spec
**D5** deliverable. Phase 1 ships:

* the **fixture-set contract** (this README + the manifest schema);
* the **harness** (eval-suite with skip-if-no-manifest guard);
* the **metrics aggregator** (`_metrics.py`).

So that landing the actual JSON files and a `baseline_v0.78.json`
flips D5 on without any code change.
