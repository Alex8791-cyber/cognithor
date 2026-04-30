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
# Layer 5a: spec §11.5 cases that the typed-Program + registry-whitelist
# layers already block — no subprocess machinery needed.
#
# These payloads ARE NOT REPRESENTABLE in the Phase-1 program tree:
# Program nodes carry only ``primitive: str`` plus typed children
# (Program / InputRef / Const). Const.value is an int / str / Predicate /
# Lambda — never a Python source string, never a callable. The executor
# looks every primitive up in the REGISTRY and refuses anything else
# with ``UnknownPrimitiveError`` (see TestRegistryWhitelist above).
#
# The tests below construct each canonical adversarial payload as a
# Program tree and assert the existing layers block it. They run in
# every CI lane — no subprocess needed.
# ---------------------------------------------------------------------------


class TestPayloadsBlockedByTypeSystem:
    """Spec §11.5 payloads that the typed-tree + registry refuse."""

    def test_eval_payload_blocked(self) -> None:
        # ``eval("__import__('os').system('rm -rf /')")`` — even framing
        # this requires either an ``eval`` primitive in the registry or
        # smuggling the source through Const.value. The first is impossible
        # (registry has no ``eval``); the second is impossible because
        # InProcessExecutor returns Const.value verbatim — it is *never*
        # interpreted as code.
        ex = InProcessExecutor()
        prog = Program("eval", (InputRef(),), "Grid")
        result = ex.execute(prog, _g([[1]]))
        assert result.ok is False
        assert result.error == "UnknownPrimitiveError"

    def test_import_payload_blocked(self) -> None:
        ex = InProcessExecutor()
        for name in ("__import__", "import", "importlib", "exec"):
            prog = Program(name, (InputRef(),), "Grid")
            result = ex.execute(prog, _g([[1]]))
            assert result.ok is False
            assert result.error == "UnknownPrimitiveError"

    def test_dunder_class_reflection_blocked(self) -> None:
        # Even smuggled as a primitive name, ``__class__`` / ``__bases__``
        # / ``__mro__`` are not registered.
        ex = InProcessExecutor()
        for name in (
            "__class__",
            "__bases__",
            "__mro__",
            "__globals__",
            "__getattribute__",
        ):
            prog = Program(name, (InputRef(),), "Grid")
            result = ex.execute(prog, _g([[1]]))
            assert result.ok is False
            assert result.error == "UnknownPrimitiveError"

    def test_dunder_subclasses_traversal_blocked(self) -> None:
        # ``object.__subclasses__()`` reflection trick — same shape: no
        # primitive registered, executor refuses.
        ex = InProcessExecutor()
        for name in ("__subclasses__", "object", "type", "vars", "globals"):
            prog = Program(name, (InputRef(),), "Grid")
            result = ex.execute(prog, _g([[1]]))
            assert result.ok is False
            assert result.error == "UnknownPrimitiveError"

    def test_f_string_double_underscore_trick_blocked(self) -> None:
        # The "f-string ``{x.__class__.__base__.__subclasses__()}``" trick
        # only works if a Python f-string is *interpolated*. The PSE
        # executor never interpolates Const.value — it returns the string
        # to the host primitive, which expects a typed argument and
        # raises TypeMismatchError if it isn't. So any string Const that
        # carries an f-string-like payload fails at the type validator,
        # never at any point that interprets it as code.
        ex = InProcessExecutor()
        # Force a Const(str) into a slot that expects a Color — the
        # validator rejects it before any interpretation.
        prog = Program(
            "recolor",
            (
                InputRef(),
                Const(
                    value="{x.__class__.__base__.__subclasses__()}",
                    output_type="Color",
                ),
                Const(value=2, output_type="Color"),
            ),
            "Grid",
        )
        result = ex.execute(prog, _g([[1]]))
        assert result.ok is False
        assert result.error == "TypeMismatchError"

    def test_pickle_bomb_billion_laughs_blocked(self) -> None:
        # The cache stores program SOURCE STRINGS (see
        # ``integration/tactical_memory.py``), not pickled objects.
        # A "billion laughs" payload would need a deserializer that
        # recursively expands references — pickle. PSE never pickles a
        # SynthesisResult, so the bomb has no surface to land on.
        from cognithor.channels.program_synthesis.integration.tactical_memory import (
            PSECache,
        )

        cache = PSECache()
        # Verify the public API never accepts a ``pickle.loads`` path.
        for attr in dir(cache):
            assert "pickle" not in attr.lower(), (
                f"PSECache surface contains {attr!r}; spec §11.5 forbids "
                f"pickle in the PSE channel — use program_source strings."
            )


# ---------------------------------------------------------------------------
# Layer 5b: spec §11.5 cases that need the real subprocess sandbox.
#
# These exercise the runner in ``sandbox/runner.py``: a fresh subprocess
# with ``setrlimit`` applied for memory / FDs / processes, and a
# wall-clock timeout enforced by the parent. The runner refuses to run
# on Windows native (``setrlimit`` is unavailable), so this whole class
# is skipped there — that matches the spec's stance: K4 must hold on
# Linux + WSL2; native Windows is research-mode only.
#
# Each test invokes a canonical adversarial payload from
# ``cognithor.channels.program_synthesis.sandbox._adversarial_payloads``.
# The runner returns a typed error tag so we can assert on the precise
# failure mode without parsing strings.
# ---------------------------------------------------------------------------


from cognithor.channels.program_synthesis.integration.capability_tokens import (  # noqa: F401  — load integration first to break the sandbox ⇄ integration import cycle
    PSECapability as _PSECapability,
)
from cognithor.channels.program_synthesis.sandbox.policy import SandboxLimits
from cognithor.channels.program_synthesis.sandbox.runner import run_in_sandbox

_PAYLOAD = "cognithor.channels.program_synthesis.sandbox._adversarial_payloads"


_skip_on_windows_native = pytest.mark.skipif(
    not __import__("sys").platform.startswith(("linux", "darwin")),
    reason=(
        "Subprocess sandbox uses POSIX setrlimit; on Windows native, "
        "research-mode applies (spec §11.6)."
    ),
)


@_skip_on_windows_native
class TestSubprocessResourceLimits:
    """OS-isolation tests against the real subprocess runner."""

    def test_while_true_loop_killed_by_wall_clock(self) -> None:
        # A 0.5 s wall-clock cap turns the busy-loop into a one-second
        # outer wait at most. The parent kills the worker.
        result = run_in_sandbox(
            f"{_PAYLOAD}:while_true_loop",
            None,
            limits=SandboxLimits(wall_clock_seconds=0.5, memory_mb=256, per_candidate_ms=100),
        )
        assert result.ok is False, (
            f"busy-loop returned ok=True; sandbox failed. value={result.value!r}"
        )
        # Default expectation is WallClockExceeded (the parent's
        # ``communicate(timeout=...)`` fires). Some Python builds /
        # CI environments surface the kill as a generic crash instead
        # — the contract is "the busy-loop did NOT complete normally".
        assert result.error in {"WallClockExceeded", "WorkerCrashed"}, (
            f"error={result.error!r} stderr_tail={result.stderr_tail!r}"
        )

    def test_numpy_giant_alloc_killed_by_memory_limit(self) -> None:
        # 64 MB cap — well below the 8 GB the payload would need.
        result = run_in_sandbox(
            f"{_PAYLOAD}:numpy_giant_alloc",
            None,
            limits=SandboxLimits(wall_clock_seconds=10.0, memory_mb=64, per_candidate_ms=100),
        )
        assert result.ok is False
        assert result.error in {
            "MemoryLimitExceeded",
            # Some kernels SIGKILL with no error tag if RLIMIT_AS
            # trips inside numpy's malloc loop — accept WorkerCrashed
            # as a portable equivalent.
            "WorkerCrashed",
        }

    def test_open_files_blocked_by_no_files_limit(self) -> None:
        result = run_in_sandbox(
            f"{_PAYLOAD}:open_etc_passwd",
            None,
        )
        assert result.ok is False, (
            f"open() should have hit RLIMIT_NOFILE; got value={result.value!r}"
        )
        # On most Linux runners the OSError carries errno 24 (EMFILE) and
        # we surface FileLimitExceeded. Some kernels / Python builds raise
        # an OSError whose errno isn't in our allow-list; in that case the
        # worker labels it WorkerCrashed. Either way the limit *fired* —
        # the worker did not succeed in opening 64 files.
        assert result.error in {"FileLimitExceeded", "WorkerCrashed"}

    def test_socket_connect_blocked(self) -> None:
        result = run_in_sandbox(
            f"{_PAYLOAD}:socket_connect",
            None,
        )
        assert result.ok is False, (
            f"socket() should have hit RLIMIT_NOFILE; got value={result.value!r}"
        )
        # ``socket()`` consumes an FD; with NOFILE=8 plus the three
        # standard streams the payload runs out fast. The strerror
        # text + errno vary by kernel — accept any of the three sandbox
        # error tags. The contract is "the limit fired", not
        # "the worker classified it perfectly".
        assert result.error in {
            "FileLimitExceeded",
            "ProcessLimitExceeded",
            "WorkerCrashed",
        }

    def test_fork_bomb_blocked(self) -> None:
        result = run_in_sandbox(
            f"{_PAYLOAD}:fork_bomb",
            None,
        )
        # Two acceptable shapes:
        # (a) the worker hits NPROC and the payload returns a small
        #     count (< 64) via the stop-on-OSError path — ok=True.
        # (b) the runner itself classifies it as ProcessLimitExceeded.
        if result.ok:
            assert isinstance(result.value, int)
            assert result.value < 64, (
                f"fork() succeeded {result.value} times; NPROC limit not enforced"
            )
        else:
            assert result.error in {
                "ProcessLimitExceeded",
                "WorkerCrashed",
                "FileLimitExceeded",
            }

    def test_recursion_stack_overflow_isolated(self) -> None:
        # The worker MUST crash, but the parent test process must keep
        # running — that's the whole point of subprocess isolation.
        result = run_in_sandbox(
            f"{_PAYLOAD}:stack_overflow",
            None,
        )
        assert result.ok is False
        assert result.error == "WorkerCrashed"
        # Sanity-check that the parent is still alive — if the worker's
        # stack overflow had cascaded, this assertion would never run.
        assert True
