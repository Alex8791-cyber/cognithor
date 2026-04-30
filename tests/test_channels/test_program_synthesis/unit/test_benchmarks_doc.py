# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Drift-gate for ``docs/channels/program_synthesis/benchmarks.md``.

Phase-1 spec §22 D9 requires a ``benchmarks.md`` to be *present* in the
docs tree (peer-reviewed, even if the numbers themselves are filled in
by the spec D5 eval harness later). This test pins three properties:

1. The file exists.
2. It documents the spec §18 metrics and the K1 success threshold.
3. The static catalog-metrics line up with the live registry — if a
   primitive lands or a constructor is added/removed, the numbers in
   the doc must be updated alongside the code.
"""

from __future__ import annotations

from pathlib import Path

from cognithor.channels.program_synthesis.dsl.lambdas import LAMBDA_CONSTRUCTORS
from cognithor.channels.program_synthesis.dsl.predicates import (
    PREDICATE_CONSTRUCTORS,
)
from cognithor.channels.program_synthesis.dsl.registry import REGISTRY

BENCH_DOC = (
    Path(__file__).resolve().parents[4]
    / "docs"
    / "channels"
    / "program_synthesis"
    / "benchmarks.md"
)


class TestBenchmarksDoc:
    def test_doc_exists(self) -> None:
        assert BENCH_DOC.is_file(), (
            "docs/channels/program_synthesis/benchmarks.md is missing — "
            "spec D9 requires it as part of Phase-1 docs."
        )

    def test_doc_quotes_k1_success_threshold(self) -> None:
        body = BENCH_DOC.read_text(encoding="utf-8")
        assert "Solved@30s" in body
        assert "Solved@5s" in body
        assert "FP-Rate" in body
        assert "Median-Time-Solved" in body
        assert "baseline_v0.78" in body or "v0.78" in body

    def test_doc_quotes_static_catalog_metrics(self) -> None:
        body = BENCH_DOC.read_text(encoding="utf-8")
        # Live counts have to appear in the doc — if a primitive lands
        # this test fails so the doc is updated alongside the catalog.
        assert f"**{len(REGISTRY)}**" in body, (
            f"benchmarks.md base-primitive count is stale; live registry has "
            f"{len(REGISTRY)} primitives but the doc does not quote it."
        )
        assert f"**{len(PREDICATE_CONSTRUCTORS)}**" in body, (
            f"benchmarks.md predicate-constructor count is stale; live count "
            f"is {len(PREDICATE_CONSTRUCTORS)}."
        )
        assert f"**{len(LAMBDA_CONSTRUCTORS)}**" in body, (
            f"benchmarks.md lambda-constructor count is stale; live count "
            f"is {len(LAMBDA_CONSTRUCTORS)}."
        )
        # Higher-order primitive count (spec §7.5 — anything taking a
        # closed-set parametric arg: Predicate, Lambda, AlignMode, SortKey).
        ho_arg_types = {"Predicate", "Lambda", "AlignMode", "SortKey"}
        ho_count = sum(
            1
            for s in REGISTRY.all_primitives()
            if any(t in ho_arg_types for t in s.signature.inputs)
        )
        assert f"**{ho_count}**" in body, (
            f"benchmarks.md higher-order-primitive count is stale; live count is {ho_count}."
        )
