# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Trace + replay tests — K9 / K10 hard gates (spec §3 + §22)."""

from __future__ import annotations

import numpy as np

from cognithor.channels.program_synthesis.search.candidate import (
    Const,
    InputRef,
    Program,
)
from cognithor.channels.program_synthesis.trace import (
    build_trace,
    format_trace,
    replay_program,
)


def _g(rows: list[list[int]]) -> np.ndarray:
    return np.array(rows, dtype=np.int8)


# ---------------------------------------------------------------------------
# build_trace — K9 hard gate (every solved program has a trace)
# ---------------------------------------------------------------------------


class TestBuildTrace:
    def test_trace_has_one_line_per_program_node(self) -> None:
        # rotate90(input) — single Program node → single trace line.
        prog = Program("rotate90", (InputRef(),), "Grid")
        trace = build_trace(prog, _g([[1, 2], [3, 4]]))
        assert len(trace.lines) == 1
        assert trace.lines[0].source == "rotate90(input)"

    def test_trace_lines_are_in_bottom_up_order(self) -> None:
        # mirror_horizontal(rotate90(input)) — inner step1 (rotate90)
        # must appear BEFORE step2 (mirror_horizontal).
        inner = Program("rotate90", (InputRef(),), "Grid")
        outer = Program("mirror_horizontal", (inner,), "Grid")
        trace = build_trace(outer, _g([[1, 2], [3, 4]]))
        assert len(trace.lines) == 2
        assert trace.lines[0].source == "rotate90(input)"
        assert trace.lines[1].source == "mirror_horizontal(step1)"

    def test_trace_summarises_grid_value(self) -> None:
        prog = Program("rotate90", (InputRef(),), "Grid")
        trace = build_trace(prog, _g([[1, 2, 3]]))
        # rotate90 of 1×3 → 3×1.
        assert "(3, 1)" in trace.lines[0].summary

    def test_trace_records_per_step_duration(self) -> None:
        prog = Program("rotate90", (InputRef(),), "Grid")
        trace = build_trace(prog, _g([[1, 2], [3, 4]]))
        assert trace.lines[0].duration_ms >= 0.0
        assert trace.total_duration_ms >= trace.lines[0].duration_ms

    def test_trace_carries_program_hash(self) -> None:
        prog = Program("rotate90", (InputRef(),), "Grid")
        trace = build_trace(prog, _g([[1, 2]]))
        assert trace.program_hash is not None
        assert trace.program_hash.startswith("sha256:")

    def test_trace_const_args_inlined_not_separate_step(self) -> None:
        # recolor(input, 1, 2) — Consts inline; only one Program node →
        # one trace line.
        prog = Program(
            "recolor",
            (
                InputRef(),
                Const(value=1, output_type="Color"),
                Const(value=2, output_type="Color"),
            ),
            "Grid",
        )
        trace = build_trace(prog, _g([[1, 2]]))
        assert len(trace.lines) == 1
        assert "1" in trace.lines[0].source
        assert "2" in trace.lines[0].source

    def test_trace_marks_failed_step_with_error_tag(self) -> None:
        # rotate90 expects an int8 grid; passing a string crashes.
        prog = Program("rotate90", (InputRef(),), "Grid")
        trace = build_trace(prog, "not a grid")
        assert not trace.all_ok
        assert "<error" in trace.lines[0].summary

    def test_trace_is_deterministic(self) -> None:
        prog = Program("rotate90", (InputRef(),), "Grid")
        a = build_trace(prog, _g([[1, 2], [3, 4]]))
        b = build_trace(prog, _g([[1, 2], [3, 4]]))
        # Source + summary + line count must match exactly; durations
        # are timing-dependent so we don't compare those.
        assert a.program_hash == b.program_hash
        assert a.final_value_summary == b.final_value_summary
        assert [(l.var, l.source, l.summary) for l in a.lines] == [
            (l.var, l.source, l.summary) for l in b.lines
        ]


# ---------------------------------------------------------------------------
# format_trace — human-readable pseudo-code
# ---------------------------------------------------------------------------


class TestFormatTrace:
    def test_includes_pse_header(self) -> None:
        prog = Program("rotate90", (InputRef(),), "Grid")
        trace = build_trace(prog, _g([[1, 2]]))
        out = format_trace(trace)
        assert "# PSE Solution Trace" in out
        assert "Program hash" in out

    def test_renders_step_assignments(self) -> None:
        inner = Program("rotate90", (InputRef(),), "Grid")
        outer = Program("mirror_horizontal", (inner,), "Grid")
        trace = build_trace(outer, _g([[1, 2], [3, 4]]))
        out = format_trace(trace)
        assert "Step 1: step1 = rotate90(input)" in out
        assert "Step 2: step2 = mirror_horizontal(step1)" in out

    def test_includes_optional_header_kv(self) -> None:
        prog = Program("rotate90", (InputRef(),), "Grid")
        trace = build_trace(prog, _g([[1, 2]]))
        out = format_trace(
            trace,
            header={"Search time": "0.4s, 12 candidates", "Spec hash": "sha256:abcd"},
        )
        assert "Search time" in out
        assert "0.4s" in out
        assert "Spec hash" in out

    def test_renders_final_summary(self) -> None:
        prog = Program("rotate90", (InputRef(),), "Grid")
        trace = build_trace(prog, _g([[1, 2, 3]]))
        out = format_trace(trace)
        assert "# Final" in out

    def test_zero_step_program_renders_inline(self) -> None:
        # InputRef alone has no Program nodes.
        trace = build_trace(InputRef(), _g([[1, 2]]))
        out = format_trace(trace)
        assert "result = input" in out


# ---------------------------------------------------------------------------
# replay_program — K10 hard gate (byte-identical re-execution + ≤ 100 ms)
# ---------------------------------------------------------------------------


class TestReplay:
    def test_identical_replay(self) -> None:
        prog = Program("rotate90", (InputRef(),), "Grid")
        inp = _g([[1, 2], [3, 4]])
        expected = _g([[3, 1], [4, 2]])
        result = replay_program(prog, inp, expected)
        assert result.identical
        assert result.duration_ms >= 0.0
        assert result.detail == "byte-identical"

    def test_mismatch_replay(self) -> None:
        prog = Program("rotate90", (InputRef(),), "Grid")
        result = replay_program(
            prog,
            _g([[1, 2], [3, 4]]),
            _g([[9, 9], [9, 9]]),  # wrong expected
        )
        assert not result.identical
        assert "mismatch" in result.detail

    def test_duration_under_100ms_for_simple_program(self) -> None:
        # K10 hard gate: replay duration P95 ≤ 100 ms. This single-call
        # check uses the strict 100ms cap as a smoke test (a real
        # benchmark over many programs would compute P95).
        prog = Program("rotate90", (InputRef(),), "Grid")
        result = replay_program(prog, _g([[1, 2], [3, 4]]), _g([[3, 1], [4, 2]]))
        assert result.identical
        assert result.duration_ms < 100.0

    def test_replay_failure_reported(self) -> None:
        # Program crashes (string input → TypeMismatchError).
        prog = Program("rotate90", (InputRef(),), "Grid")
        result = replay_program(prog, "not a grid", _g([[1, 2]]))
        assert not result.identical
        assert "failed" in result.detail
        assert "<error" in result.actual_summary

    def test_dtype_mismatch_counts_as_not_identical(self) -> None:
        prog = Program("rotate90", (InputRef(),), "Grid")
        # Same shape + values but different dtype → must NOT be identical.
        actual_input = _g([[1, 2], [3, 4]])
        expected_wrong_dtype = np.array([[3, 1], [4, 2]], dtype=np.int32)
        result = replay_program(prog, actual_input, expected_wrong_dtype)
        # actual is int8, expected is int32 → not identical by K10's strict rule.
        assert not result.identical


class TestReplayDeterminism:
    def test_two_replays_produce_identical_verdicts(self) -> None:
        prog = Program("rotate90", (InputRef(),), "Grid")
        inp = _g([[1, 2], [3, 4]])
        expected = _g([[3, 1], [4, 2]])
        a = replay_program(prog, inp, expected)
        b = replay_program(prog, inp, expected)
        assert a.identical == b.identical
        assert a.actual_summary == b.actual_summary
