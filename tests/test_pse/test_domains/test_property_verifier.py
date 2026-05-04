"""Tests for ``PropertyVerifier``."""

from __future__ import annotations

from cognithor.channels.program_synthesis.domains.property_verifier import (
    PropertyResult,
    PropertyVerifier,
)


def _passing(name: str = "ok") -> PropertyResult:
    return PropertyResult(name=name, ok=True)


def _failing(name: str = "bad", msg: str = "fail") -> PropertyResult:
    return PropertyResult(
        name=name,
        ok=False,
        error_message=msg,
        examples=({"x": 1},),
    )


class TestPropertyResult:
    def test_passed_property_mirrors_ok(self) -> None:
        assert _passing().passed is True
        assert _failing().passed is False

    def test_is_frozen(self) -> None:
        r = _passing()
        try:
            r.ok = False  # type: ignore[misc]
        except Exception:
            return
        # If no exception, the dataclass wasn't frozen — fail loudly.
        raise AssertionError("PropertyResult must be frozen")


class TestPropertyVerifier:
    def test_empty_returns_empty_tuple(self) -> None:
        v = PropertyVerifier()
        assert v.run() == ()
        assert v.all_pass() is True

    def test_runs_in_order(self) -> None:
        order: list[str] = []

        def t1() -> PropertyResult:
            order.append("t1")
            return _passing("t1")

        def t2() -> PropertyResult:
            order.append("t2")
            return _passing("t2")

        v = PropertyVerifier()
        v.add(t1)
        v.add(t2)
        results = v.run()
        assert order == ["t1", "t2"]
        assert [r.name for r in results] == ["t1", "t2"]
        assert all(r.ok for r in results)

    def test_fail_fast_default(self) -> None:
        order: list[str] = []

        def t1() -> PropertyResult:
            order.append("t1")
            return _failing("t1")

        def t2() -> PropertyResult:
            order.append("t2")
            return _passing("t2")

        v = PropertyVerifier()
        v.add(t1)
        v.add(t2)
        results = v.run()
        assert order == ["t1"]  # short-circuit
        assert len(results) == 1
        assert not results[0].ok

    def test_no_fail_fast_runs_all(self) -> None:
        v = PropertyVerifier(fail_fast=False)
        v.add(lambda: _failing("a"))
        v.add(lambda: _passing("b"))
        v.add(lambda: _failing("c"))
        results = v.run()
        assert len(results) == 3
        assert [r.ok for r in results] == [False, True, False]

    def test_exception_inside_test_becomes_failure(self) -> None:
        def boom() -> PropertyResult:
            raise RuntimeError("kaboom")

        v = PropertyVerifier()
        v.add(boom)
        results = v.run()
        assert len(results) == 1
        assert results[0].ok is False
        assert "RuntimeError" in results[0].error_message

    def test_all_pass_does_not_mutate_fail_fast(self) -> None:
        v = PropertyVerifier(fail_fast=True)
        v.add(lambda: _failing("a"))
        v.add(lambda: _passing("b"))

        # all_pass internally turns fail_fast off but must restore it.
        assert v.all_pass() is False
        assert v.fail_fast is True
        # Now confirm fail_fast still short-circuits the regular run.
        assert len(v.run()) == 1
