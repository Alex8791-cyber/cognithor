# `src/cognithor/benchmark/` — archive recommendation — 2026-04-28

Closes the last open VERIFY-WIRING item from `docs/audits/2026-04-27-stale-module-triage.md`.

## TL;DR

**Archive `src/cognithor/benchmark/` (~863 LOC).** It's a sophisticated benchmark framework that no production code calls; the canonical benchmark home is the top-level `cognithor_bench/` package, which already has its own `pyproject.toml`, console-script `cognithor-bench`, and Cognithor + AutoGen adapters.

## Evidence

### `src/cognithor/benchmark/`

- 2 files, 863 LOC: `__init__.py` (4 lines), `suite.py` (859 LOC).
- Defines 7 classes — `BenchmarkTask`, `BenchmarkResult`, `BenchmarkScorer`, `BenchmarkRunner`, `BenchmarkReport`, `RegressionEntry`, `RegressionDetector` — and the `TaskCategory` / `TaskDifficulty` / `ResultStatus` enums.
- Architecture reference: §18.1.
- **Live import count outside its own package:** 0.
- **Test references:** 1 — `tests/test_benchmark/test_suite.py`.
- **Wired into:** nothing — not the CLI (`__main__.py`), not the gateway, not any channel, not a workflow.
- **Last commit:** 2026-04-10 (≥ 18 days stale).

### `cognithor_bench/` (top-level)

- Has its own `pyproject.toml` with version 0.1.0 and the CLI entry point `cognithor-bench = "cognithor_bench.cli:main"`.
- Lives outside the published `cognithor` PyPI package on purpose (per the dev-deps comment in the root `pyproject.toml`: "Local sub-packages are NOT listed here because direct file references are forbidden in PyPI metadata").
- Already has Cognithor + AutoGen adapters: `cognithor_bench/src/cognithor_bench/adapters/autogen_adapter.py` (lazy import per pyproject comment).
- Memory entry confirms: "in-monorepo benchmark scaffold with own pyproject.toml + console-script cognithor-bench."

## Why archive instead of rewire

1. **Duplication risk** — keeping both invites confusion about which is the canonical benchmark home. `cognithor_bench/` is the one with publishable artefacts and adapters; the internal one is a duplicated implementation.
2. **No callers** — 0 production imports in 18+ days means there's no migration cost on archival.
3. **The framework is genuinely fancy** — `RegressionDetector`, `BenchmarkReport` markdown + JSON, scoring across 7 categories. If we ever want any of this, we should backport the *useful pieces* into `cognithor_bench/`, not keep two parallel implementations.
4. **Test count is weak** — 1 test file. Tests can be archived alongside the source.

## Recommended action

```bash
# Move source to archive (keeps git history)
git mv src/cognithor/benchmark archive/cognithor_internal_benchmark
git mv tests/test_benchmark archive/cognithor_internal_benchmark_tests

# Or just delete if no historical value:
git rm -r src/cognithor/benchmark tests/test_benchmark
```

If a piece of the internal framework is actually useful (e.g. `RegressionDetector`'s diff format, the markdown report shape), copy that ONE piece into `cognithor_bench/` first — don't preserve the entire 863-LOC parallel impl just for one nice class.

## Counter-argument (for completeness)

If there's a use case for an *in-process* benchmark runner (i.e. one that runs INSIDE a live `cognithor` instance for self-evaluation, rather than as an external CLI), then keep `src/cognithor/benchmark/` — but actually wire it up to something (e.g. expose via `cognithor benchmark run` subcommand or via a gateway endpoint). Right now it's a dead module dressed as a feature.

## Decision required

This is a User-judgment call — pick one:

- [ ] **Archive** (`git mv` to `archive/`) — recommended.
- [ ] **Delete** (`git rm`) — also fine; git history preserves recoverability.
- [ ] **Keep + wire up** — only if there's a concrete in-process benchmark use case. Specify what consumer would call it.

Once decided, the relevant action is one PR; ~10 minutes either way. This doc doesn't decide for you — it surfaces the case clearly enough to decide quickly.
