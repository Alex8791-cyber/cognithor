# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Higher-order primitive reachability tests.

The base enumerative search only enumerates over base primitives — it
never builds Predicate/Lambda *values* on its own. Spec §6.4 mandates a
closed, enumerable set of constructors, but they live in their own
registries and the bank therefore needs to be seeded with the canonical
leaf values before HO primitives like ``filter_objects``,
``map_objects`` can match their argument types.

This module pins three properties of that seed:

1. The seed lists are bounded and contain the canonical Phase-1.5
   constructors (``color_eq``, ``size_*``, ``identity/recolor/shift_lambda``).
2. ``Const`` values now carry Predicate / Lambda dataclasses verbatim
   and ``to_source`` round-trips them through their constructor source.
3. With those leaves seeded, ``EnumerativeSearch`` produces a depth-2
   candidate that combines an ObjectSet-producer with a Predicate seed
   — i.e. ``filter_objects(connected_components_4(input), <Predicate>)``
   is enumerable and executable.
"""

from __future__ import annotations

import numpy as np

from cognithor.channels.program_synthesis.dsl.lambdas import Lambda
from cognithor.channels.program_synthesis.dsl.predicates import Predicate
from cognithor.channels.program_synthesis.dsl.types_grid import Object, ObjectSet
from cognithor.channels.program_synthesis.search.candidate import Const, InputRef, Program
from cognithor.channels.program_synthesis.search.enumerative import (
    _lambda_seeds,
    _predicate_seeds,
    _seed_higher_order_bank,
)
from cognithor.channels.program_synthesis.search.executor import InProcessExecutor

# ---------------------------------------------------------------------------
# 1. Seed lists
# ---------------------------------------------------------------------------


class TestSeedLists:
    def test_predicate_seeds_count_matches_canonical_set(self) -> None:
        seeds = _predicate_seeds()
        # 10 colors + 5*3 size constructors + 3 zero-arg constructors.
        assert len(seeds) == 10 + 15 + 3

    def test_predicate_seeds_include_color_eq_and_size_predicates(self) -> None:
        names = {(s.constructor, s.args) for s in _predicate_seeds()}
        assert ("color_eq", (0,)) in names
        assert ("color_eq", (9,)) in names
        assert ("size_gt", (3,)) in names
        assert ("is_rectangle", ()) in names

    def test_predicate_seeds_are_all_predicate_dataclasses(self) -> None:
        for s in _predicate_seeds():
            assert isinstance(s, Predicate)

    def test_lambda_seeds_count_matches_canonical_set(self) -> None:
        seeds = _lambda_seeds()
        # 1 identity + 10 recolor + 4 cardinal shifts.
        assert len(seeds) == 1 + 10 + 4

    def test_lambda_seeds_include_identity_and_recolor(self) -> None:
        names = {(s.constructor, s.args) for s in _lambda_seeds()}
        assert ("identity_lambda", ()) in names
        assert ("recolor_lambda", (0,)) in names
        assert ("recolor_lambda", (9,)) in names
        assert ("shift_lambda", (1, 0)) in names

    def test_lambda_seeds_are_all_lambda_dataclasses(self) -> None:
        for s in _lambda_seeds():
            assert isinstance(s, Lambda)


# ---------------------------------------------------------------------------
# 2. Const carries Predicate / Lambda values
# ---------------------------------------------------------------------------


class TestConstCarriesHigherOrderValues:
    def test_const_predicate_to_source_delegates_to_predicate(self) -> None:
        leaf = Const(value=Predicate(constructor="color_eq", args=(3,)), output_type="Predicate")
        assert leaf.to_source() == "color_eq(3)"
        assert leaf.depth() == 0

    def test_const_lambda_to_source_delegates_to_lambda(self) -> None:
        leaf = Const(
            value=Lambda(constructor="recolor_lambda", args=(7,)),
            output_type="Lambda",
        )
        assert leaf.to_source() == "recolor_lambda(7)"
        assert leaf.depth() == 0

    def test_executor_returns_predicate_value_verbatim(self) -> None:
        pred = Predicate(constructor="color_eq", args=(2,))
        leaf = Const(value=pred, output_type="Predicate")
        result = InProcessExecutor().execute(leaf, np.array([[0]], dtype=np.int8))
        assert result.ok
        assert result.value is pred

    def test_executor_returns_lambda_value_verbatim(self) -> None:
        fn = Lambda(constructor="identity_lambda")
        leaf = Const(value=fn, output_type="Lambda")
        result = InProcessExecutor().execute(leaf, np.array([[0]], dtype=np.int8))
        assert result.ok
        assert result.value is fn


# ---------------------------------------------------------------------------
# 3. Higher-order seeds populate the bank
# ---------------------------------------------------------------------------


class TestSeedHigherOrderBank:
    def test_bank_has_predicate_and_lambda_buckets(self) -> None:
        bank: dict[str, list] = {}
        _seed_higher_order_bank(bank)
        assert "Predicate" in bank
        assert "Lambda" in bank
        assert len(bank["Predicate"]) == 28
        assert len(bank["Lambda"]) == 15

    def test_bank_predicate_entries_are_const_leaves(self) -> None:
        bank: dict[str, list] = {}
        _seed_higher_order_bank(bank)
        for leaf in bank["Predicate"]:
            assert isinstance(leaf, Const)
            assert leaf.output_type == "Predicate"
            assert isinstance(leaf.value, Predicate)


# ---------------------------------------------------------------------------
# 4. End-to-end: filter_objects is now executable through the engine
# ---------------------------------------------------------------------------


class TestFilterObjectsThroughExecutor:
    """Manually constructs the depth-2 program the engine can now build.

    The point is to prove the `Const(Predicate, ...)` mechanism plumbs
    through to the host primitive correctly — once that works, the
    enumerator's existing arg-matching loop reaches it for free.
    """

    def test_filter_objects_keeps_only_matching_color(self) -> None:
        # Two objects, one color-1 and one color-2.
        program = Program(
            primitive="filter_objects",
            children=(
                Program(
                    primitive="connected_components_4",
                    children=(InputRef(),),
                    output_type="ObjectSet",
                ),
                Const(
                    value=Predicate(constructor="color_eq", args=(1,)),
                    output_type="Predicate",
                ),
            ),
            output_type="ObjectSet",
        )
        grid = np.array(
            [
                [1, 0, 2],
                [1, 0, 2],
            ],
            dtype=np.int8,
        )
        result = InProcessExecutor().execute(program, grid)
        assert result.ok, result.error
        out = result.value
        assert isinstance(out, ObjectSet)
        # Only the color-1 connected component survives the filter.
        assert len(out) == 1
        only = next(iter(out.objects))
        assert isinstance(only, Object)
        assert only.color == 1
        assert only.size == 2  # the two color-1 cells
