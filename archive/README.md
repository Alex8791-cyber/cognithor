# `archive/`

Code that was once part of the live source tree but has been retired. Kept for historical reference and possible salvage of useful pieces.

**Archived modules are not imported by any production code, are not tested in CI, and are not shipped in the published `cognithor` PyPI wheel.**

If you need a piece of an archived module, copy the specific class or function into the appropriate live module — do not re-introduce parallel implementations.

## Contents

### `cognithor_internal_benchmark/` (archived 2026-04-28)

Moved from `src/cognithor/benchmark/` (`__init__.py`, `suite.py`).
Tests moved from `tests/test_benchmark/` to `cognithor_internal_benchmark_tests/`.

**Why archived:** Sophisticated in-process benchmark framework (~863 LOC, 7 classes including `BenchmarkRunner`, `RegressionDetector`, `BenchmarkReport`) that no production code called. The canonical benchmark home is the top-level `cognithor_bench/` package — own `pyproject.toml`, `cognithor-bench` console script, Cognithor + AutoGen adapters.

**Recommendation doc:** `docs/audits/2026-04-28-benchmark-archive-recommendation.md`.

If a piece of this framework turns out to be useful (e.g. `RegressionDetector`'s diff format, the markdown report shape), backport that one piece into `cognithor_bench/` rather than reviving the whole archive.
