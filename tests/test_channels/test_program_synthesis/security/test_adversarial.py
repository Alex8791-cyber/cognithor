# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Adversarial security tests — K4 hard gate (spec §3, §11.5, §22 D4).

Phase-1 enforcement layers (the only ones currently active):

* **Type system** — Program / InputRef / Const dataclasses are typed
  dataclasses; raw Python source never reaches the executor.
* **Registry whitelist** — InProcessExecutor only invokes functions
  registered in the PrimitiveRegistry. A primitive name that isn't in
  the registry surfaces as ``UnknownPrimitiveError`` with no execution.
* **Argument validation** — the per-primitive ``_check_grid`` /
  ``_check_color`` / ``_check_int`` validators reject malformed inputs
  *before* any numpy work. Exceptions are converted to
  ExecutionResult.error tags by the executor — they never propagate to
  the caller.

Phase-1 cases that **require the real subprocess sandbox** (setrlimit,
network namespaces, pickle-bomb defence) are scaffolded here and marked
``@pytest.mark.skip(reason="requires subprocess sandbox")``. A
follow-up PR lands the actual subprocess worker and flips them on.

Hard gate (K4): every test in this file must pass. CI green is the
release criterion (spec §22 D4).
"""

from __future__ import annotations

import numpy as np
import pytest

from cognithor.channels.program_synthesis.dsl.registry import PrimitiveRegistry
from cognithor.channels.program_synthesis.search.candidate import (
    Const,
    InputRef,
    Program,
)
from cognithor.channels.program_synthesis.search.executor import InProcessExecutor


def _g(rows: list[list[int]]) -> np.ndarray:
    return np.array(rows, dtype=np.int8)


# ---------------------------------------------------------------------------
# Layer 1: registry whitelist — unknown primitives are rejected.
# ---------------------------------------------------------------------------


class TestRegistryWhitelist:
    """The Executor only invokes registered primitives."""

    def test_unknown_primitive_blocked(self) -> None:
        # An empty registry can't execute any primitive.
        ex = InProcessExecutor(registry=PrimitiveRegistry())
        prog = Program("rotate90", (InputRef(),), "Grid")
        result = ex.execute(prog, _g([[1, 2]]))
        assert result.ok is False
        assert result.error == "UnknownPrimitiveError"

    def test_misspelled_primitive_blocked(self) -> None:
        ex = InProcessExecutor()
        # "rotat90" instead of "rotate90"
        prog = Program("rotat90", (InputRef(),), "Grid")
        result = ex.execute(prog, _g([[1, 2]]))
        assert result.ok is False
        assert result.error == "UnknownPrimitiveError"

    def test_double_underscore_name_blocked(self) -> None:
        # Names like "__class__" / "__bases__" should not match any
        # registered primitive.
        ex = InProcessExecutor()
        for name in ("__class__", "__import__", "__subclasses__"):
            prog = Program(name, (InputRef(),), "Grid")
            result = ex.execute(prog, _g([[1]]))
            assert result.ok is False
            assert result.error == "UnknownPrimitiveError"


# ---------------------------------------------------------------------------
# Layer 2: argument validators reject malformed inputs.
# ---------------------------------------------------------------------------


class TestArgumentValidators:
    """Per-primitive validators surface as TypeMismatchError, not silent
    misbehaviour."""

    def test_string_input_to_grid_primitive_caught(self) -> None:
        ex = InProcessExecutor()
        prog = Program("rotate90", (InputRef(),), "Grid")
        result = ex.execute(prog, "malicious")
        assert result.ok is False
        assert result.error == "TypeMismatchError"

    def test_oversized_color_caught(self) -> None:
        ex = InProcessExecutor()
        prog = Program(
            "recolor",
            (
                InputRef(),
                Const(value=99, output_type="Color"),  # out of 0..9 range
                Const(value=2, output_type="Color"),
            ),
            "Grid",
        )
        result = ex.execute(prog, _g([[1]]))
        assert result.ok is False
        assert result.error == "TypeMismatchError"

    def test_negative_pad_width_caught(self) -> None:
        ex = InProcessExecutor()
        prog = Program(
            "pad_with",
            (
                InputRef(),
                Const(value=0, output_type="Color"),
                Const(value=-5, output_type="Int"),
            ),
            "Grid",
        )
        result = ex.execute(prog, _g([[1]]))
        assert result.ok is False
        assert result.error == "TypeMismatchError"

    def test_wrong_dtype_caught(self) -> None:
        ex = InProcessExecutor()
        prog = Program("rotate90", (InputRef(),), "Grid")
        # int32 input should be rejected by _check_grid.
        bad = np.array([[1, 2]], dtype=np.int32)
        result = ex.execute(prog, bad)
        assert result.ok is False
        assert result.error == "TypeMismatchError"


# ---------------------------------------------------------------------------
# Layer 3: graceful failure — exceptions never propagate to the caller.
# ---------------------------------------------------------------------------


class TestGracefulFailure:
    """The executor catches every Exception, returns it as an error tag."""

    def test_division_by_zero_in_custom_primitive_isolated(self) -> None:
        from cognithor.channels.program_synthesis.dsl.registry import (
            PrimitiveSpec,
        )
        from cognithor.channels.program_synthesis.dsl.signatures import Signature

        reg = PrimitiveRegistry()
        reg.register(
            PrimitiveSpec(
                name="bomb",
                signature=Signature(inputs=("Grid",), output="Grid"),
                cost=1.0,
                fn=lambda g: 1 / 0,  # ZeroDivisionError on call
            )
        )
        ex = InProcessExecutor(registry=reg)
        result = ex.execute(Program("bomb", (InputRef(),), "Grid"), _g([[1]]))
        assert result.ok is False
        assert result.error == "ZeroDivisionError"

    def test_value_error_in_custom_primitive_isolated(self) -> None:
        from cognithor.channels.program_synthesis.dsl.registry import (
            PrimitiveSpec,
        )
        from cognithor.channels.program_synthesis.dsl.signatures import Signature

        def _raises(g):
            raise ValueError("bad input")

        reg = PrimitiveRegistry()
        reg.register(
            PrimitiveSpec(
                name="raiser",
                signature=Signature(inputs=("Grid",), output="Grid"),
                cost=1.0,
                fn=_raises,
            )
        )
        ex = InProcessExecutor(registry=reg)
        result = ex.execute(Program("raiser", (InputRef(),), "Grid"), _g([[1]]))
        assert result.ok is False
        assert result.error == "ValueError"

    def test_keyboard_interrupt_propagates(self) -> None:
        # Intentional non-coverage: KeyboardInterrupt SHOULD bypass the
        # generic Exception handler so Ctrl-C still works. This is a
        # safety property of the executor, not a security gate, but we
        # lock it down here because it intersects with sandbox design.
        from cognithor.channels.program_synthesis.dsl.registry import (
            PrimitiveSpec,
        )
        from cognithor.channels.program_synthesis.dsl.signatures import Signature

        def _ki(g):
            raise KeyboardInterrupt()

        reg = PrimitiveRegistry()
        reg.register(
            PrimitiveSpec(
                name="ki_raiser",
                signature=Signature(inputs=("Grid",), output="Grid"),
                cost=1.0,
                fn=_ki,
            )
        )
        ex = InProcessExecutor(registry=reg)
        with pytest.raises(KeyboardInterrupt):
            ex.execute(Program("ki_raiser", (InputRef(),), "Grid"), _g([[1]]))


# ---------------------------------------------------------------------------
# Layer 4: data integrity — Phase-1 primitives don't share state.
# ---------------------------------------------------------------------------


class TestDataIntegrity:
    """Pure-function contract: primitive output is independent of caller's
    input mutation."""

    def test_input_grid_not_mutated_by_rotate90(self) -> None:
        ex = InProcessExecutor()
        original = _g([[1, 2], [3, 4]])
        prog = Program("rotate90", (InputRef(),), "Grid")
        result = ex.execute(prog, original)
        assert result.ok
        result.value[0, 0] = 99
        # Source grid unchanged.
        assert original[0, 0] == 1

    def test_input_grid_not_mutated_by_recolor(self) -> None:
        ex = InProcessExecutor()
        original = _g([[1, 2], [1, 3]])
        prog = Program(
            "recolor",
            (
                InputRef(),
                Const(value=1, output_type="Color"),
                Const(value=9, output_type="Color"),
            ),
            "Grid",
        )
        result = ex.execute(prog, original)
        assert result.ok
        # Original still has the old colour.
        assert int(original[0, 0]) == 1


# ---------------------------------------------------------------------------
# Layer 5: cases requiring the real subprocess sandbox.
#
# These test the spec §11.5 list. They're scaffolded with the canonical
# adversarial payloads, but skipped until the subprocess + AST-whitelist
# + setrlimit machinery lands. Removing the skip is part of the K4
# hard-gate audit before release.
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="requires subprocess sandbox (Week 6+ PR)")
class TestSubprocessSandbox:
    """Scaffold for the full K4 audit. Each test maps to one of the
    20+ adversarial cases enumerated in spec §11.5."""

    def test_eval_payload_blocked(self) -> None:
        # eval("__import__('os').system('rm -rf /')")
        raise NotImplementedError

    def test_import_payload_blocked(self) -> None:
        # __import__('os').system(...)
        raise NotImplementedError

    def test_while_true_loop_killed_by_wall_clock(self) -> None:
        raise NotImplementedError

    def test_numpy_giant_alloc_killed_by_memory_limit(self) -> None:
        # numpy.zeros((10**9,))
        raise NotImplementedError

    def test_open_etc_passwd_blocked_by_no_files_limit(self) -> None:
        raise NotImplementedError

    def test_socket_connect_blocked(self) -> None:
        raise NotImplementedError

    def test_fork_bomb_blocked(self) -> None:
        raise NotImplementedError

    def test_recursion_stack_overflow_isolated(self) -> None:
        raise NotImplementedError

    def test_pickle_bomb_billion_laughs_blocked(self) -> None:
        raise NotImplementedError

    def test_dunder_class_reflection_blocked(self) -> None:
        raise NotImplementedError

    def test_dunder_subclasses_traversal_blocked(self) -> None:
        raise NotImplementedError

    def test_f_string_double_underscore_trick_blocked(self) -> None:
        raise NotImplementedError
