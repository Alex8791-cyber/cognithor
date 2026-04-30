# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""ARC-AGI-3 evaluation harness for the PSE channel (spec §17.5).

This sub-package is the home for the **slow**, end-to-end PSE eval
suite. It is intentionally separated from the unit and microbench
trees because:

* the harness consumes external fixture data
  (``cognithor_bench/arc_agi3/manifest.json``) and runs nightly, not
  on every CI push;
* its result files (under ``cognithor_bench/arc_agi3/runs/``) are
  per-host artefacts, gitignored, and promoted into
  ``docs/channels/program_synthesis/benchmarks.md`` by hand.

Phase-1 lands the **scaffold + metrics aggregator** so D5 can be
closed by simply committing the 100-task train + 30-task held-out
JSON files plus a ``baseline_v0.78.json``.
"""
