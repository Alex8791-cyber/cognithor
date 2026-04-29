# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""ObservationalEquivalencePruner tests (spec §9)."""

from __future__ import annotations

import numpy as np
import pytest

from cognithor.channels.program_synthesis.search.candidate import (
    Const,
    InputRef,
    Program,
)
from cognithor.channels.program_synthesis.search.equivalence import (
    ObservationalEquivalencePruner,
)


def _g(rows: list[list[int]]) -> np.ndarray:
    return np.array(rows, dtype=np.int8)


# ---------------------------------------------------------------------------
# Fingerprinting
# ---------------------------------------------------------------------------


class TestFingerprint:
    def test_deterministic(self) -> None:
        pruner = ObservationalEquivalencePruner()
        prog = Program("rotate90", (InputRef(),), "Grid")
        inputs = (_g([[1, 2], [3, 4]]),)
        a = pruner.fingerprint(prog, inputs)
        b = pruner.fingerprint(prog, inputs)
        assert a == b is not None

    def test_different_programs_different_outputs_different_fps(self) -> None:
        pruner = ObservationalEquivalencePruner()
        rot90 = Program("rotate90", (InputRef(),), "Grid")
        rot180 = Program("rotate180", (InputRef(),), "Grid")
        inputs = (_g([[1, 2], [3, 4]]),)
        # rot90 → [[3,1],[4,2]]; rot180 → [[4,3],[2,1]] — different
        # outputs must hash differently.
        assert pruner.fingerprint(rot90, inputs) != pruner.fingerprint(rot180, inputs)

    def test_same_outputs_same_fp_even_for_different_sources(self) -> None:
        # rotate180 and rotate90∘rotate90 are equivalent on every grid.
        pruner = ObservationalEquivalencePruner()
        rot180 = Program("rotate180", (InputRef(),), "Grid")
        rot_twice = Program("rotate90", (Program("rotate90", (InputRef(),), "Grid"),), "Grid")
        inputs = (_g([[1, 2], [3, 4]]), _g([[5, 6, 7], [8, 9, 0]]))
        assert pruner.fingerprint(rot180, inputs) == pruner.fingerprint(rot_twice, inputs)

    def test_fingerprint_changes_when_inputs_change(self) -> None:
        pruner = ObservationalEquivalencePruner()
        prog = Program("rotate90", (InputRef(),), "Grid")
        a = pruner.fingerprint(prog, (_g([[1, 2]]),))
        b = pruner.fingerprint(prog, (_g([[1, 3]]),))
        assert a != b

    def test_unreliable_program_returns_none(self) -> None:
        pruner = ObservationalEquivalencePruner(unreliable_threshold=0.5)
        # rotate90 expects a grid; pass strings → all crash.
        prog = Program("rotate90", (InputRef(),), "Grid")
        # 3 inputs all crash (> 50% threshold) → None.
        fp = pruner.fingerprint(prog, ("nope", "nada", "nogo"))
        assert fp is None

    def test_below_threshold_not_unreliable(self) -> None:
        pruner = ObservationalEquivalencePruner(unreliable_threshold=0.5)
        # 1 of 3 crashes: still below 50% threshold → fingerprint produced.
        prog = Program("rotate90", (InputRef(),), "Grid")
        fp = pruner.fingerprint(
            prog,
            (_g([[1, 2]]), _g([[3, 4]]), "broken"),
        )
        assert fp is not None

    def test_const_color_int_fingerprint(self) -> None:
        pruner = ObservationalEquivalencePruner()
        prog = Program(
            "recolor",
            (
                InputRef(),
                Const(value=1, output_type="Color"),
                Const(value=2, output_type="Color"),
            ),
            "Grid",
        )
        inputs = (_g([[1, 2]]),)
        # Same program/inputs → same fp; sanity check int-encoding works.
        a = pruner.fingerprint(prog, inputs)
        b = pruner.fingerprint(prog, inputs)
        assert a == b is not None

    def test_threshold_validation(self) -> None:
        with pytest.raises(ValueError, match="threshold"):
            ObservationalEquivalencePruner(unreliable_threshold=0.0)
        with pytest.raises(ValueError, match="threshold"):
            ObservationalEquivalencePruner(unreliable_threshold=1.5)


# ---------------------------------------------------------------------------
# Bookkeeping (is_duplicate / register / reset / admit)
# ---------------------------------------------------------------------------


class TestBookkeeping:
    def test_register_then_is_duplicate(self) -> None:
        pruner = ObservationalEquivalencePruner()
        pruner.register("abc", "Grid")
        assert pruner.is_duplicate("abc", "Grid")
        assert not pruner.is_duplicate("def", "Grid")

    def test_register_isolates_per_type_tag(self) -> None:
        pruner = ObservationalEquivalencePruner()
        pruner.register("abc", "Grid")
        # Same fingerprint under a different type tag is *not* a duplicate.
        assert not pruner.is_duplicate("abc", "Color")

    def test_register_idempotent(self) -> None:
        pruner = ObservationalEquivalencePruner()
        pruner.register("abc", "Grid")
        pruner.register("abc", "Grid")
        assert pruner.is_duplicate("abc", "Grid")

    def test_reset_clears_seen(self) -> None:
        pruner = ObservationalEquivalencePruner()
        pruner.register("abc", "Grid")
        pruner.reset()
        assert not pruner.is_duplicate("abc", "Grid")


class TestAdmit:
    def test_first_admit_returns_true(self) -> None:
        pruner = ObservationalEquivalencePruner()
        prog = Program("rotate90", (InputRef(),), "Grid")
        admitted = pruner.admit(prog, "Grid", (_g([[1, 2], [3, 4]]),))
        assert admitted is True

    def test_second_admit_with_equivalent_program_returns_false(self) -> None:
        pruner = ObservationalEquivalencePruner()
        # rotate180 admitted first, then rotate90∘rotate90 should be
        # rejected as observationally equivalent.
        rot180 = Program("rotate180", (InputRef(),), "Grid")
        rot_twice = Program("rotate90", (Program("rotate90", (InputRef(),), "Grid"),), "Grid")
        inputs = (_g([[1, 2], [3, 4]]),)
        assert pruner.admit(rot180, "Grid", inputs) is True
        assert pruner.admit(rot_twice, "Grid", inputs) is False

    def test_unreliable_programs_not_admitted(self) -> None:
        pruner = ObservationalEquivalencePruner(unreliable_threshold=0.5)
        prog = Program("rotate90", (InputRef(),), "Grid")
        # All inputs crash → None fingerprint → not admitted, not registered.
        assert pruner.admit(prog, "Grid", ("a", "b", "c")) is False
        # And a subsequent valid call should still admit a fresh program.
        good = Program("rotate90", (InputRef(),), "Grid")
        assert pruner.admit(good, "Grid", (_g([[1]]),)) is True

    def test_admit_isolates_per_type_tag(self) -> None:
        pruner = ObservationalEquivalencePruner()
        prog = Program("rotate90", (InputRef(),), "Grid")
        inputs = (_g([[1, 2]]),)
        # Same program under two different type tags should both succeed
        # (the pruner's bookkeeping is per-type).
        assert pruner.admit(prog, "Grid", inputs) is True
        assert pruner.admit(prog, "OtherType", inputs) is True
