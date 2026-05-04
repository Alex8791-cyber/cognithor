"""Float-precision pipeline verifier (Sprint-26.3).

Same pipeline-spec shape as Datetime/JSON verifiers. Comparisons use
``nearly_equal`` semantics by default (relative + absolute eps) so a
program that's bit-exact-different but numerically equivalent passes.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from cognithor.channels.program_synthesis.domains.float_dsl.catalog import (
    FloatCatalog,
    build_float_catalog,
)

if TYPE_CHECKING:
    from collections.abc import Iterable


class FloatVerifierError(Exception):
    """Raised when a Float pipeline is malformed or fails verification."""


class FloatVerifier:
    """Run a Float pipeline spec against example inputs."""

    def __init__(
        self,
        catalog: FloatCatalog | None = None,
        *,
        eps: float = 1e-9,
    ) -> None:
        self._catalog = catalog or build_float_catalog()
        self._eps = eps

    def verify(
        self,
        program: Any,
        examples: Iterable[Mapping[str, Any]],
    ) -> bool:
        steps = self._coerce_steps(program)
        for index, example in enumerate(examples):
            actual = self._run(steps, example.get("input"))
            expected = example.get("output")
            if not self._values_match(actual, expected):
                msg = (
                    f"Example {index}: pipeline output {actual!r} "
                    f"!= expected {expected!r} (eps={self._eps})"
                )
                raise FloatVerifierError(msg)
        return True

    @staticmethod
    def _coerce_steps(program: Any) -> list[Mapping[str, Any]]:
        if isinstance(program, list):
            steps = program
        elif isinstance(program, Mapping):
            raw = program.get("program")
            if not isinstance(raw, list):
                msg = f"Float program 'program' field must be a list, got {type(raw).__name__}"
                raise FloatVerifierError(msg)
            steps = raw
        else:
            msg = f"Float program must be list or {{'program': list}}, got {type(program).__name__}"
            raise FloatVerifierError(msg)
        for step in steps:
            if not isinstance(step, Mapping):
                msg = f"Pipeline step must be a mapping, got {type(step).__name__}"
                raise FloatVerifierError(msg)
            if "primitive" not in step:
                msg = "Pipeline step missing 'primitive' key"
                raise FloatVerifierError(msg)
        return list(steps)

    def _run(self, steps: list[Mapping[str, Any]], input_value: Any) -> Any:
        cur: Any = input_value
        for step in steps:
            name = step["primitive"]
            try:
                primitive = self._catalog.get(name)
            except KeyError as exc:
                raise FloatVerifierError(str(exc)) from exc
            args = step.get("args", {})
            if not isinstance(args, Mapping):
                msg = f"Step {name!r}: 'args' must be a mapping"
                raise FloatVerifierError(msg)
            try:
                cur = primitive.fn(cur, **args)
            except (TypeError, ValueError) as exc:
                msg = f"Step {name!r}: {type(exc).__name__}: {exc}"
                raise FloatVerifierError(msg) from exc
        return cur

    def _values_match(self, actual: Any, expected: Any) -> bool:
        # Bool / non-numeric: strict equality.
        if isinstance(actual, bool) or isinstance(expected, bool):
            return bool(actual == expected)
        if not isinstance(actual, int | float) or not isinstance(expected, int | float):
            return bool(actual == expected)
        a = float(actual)
        b = float(expected)
        if math.isnan(a) and math.isnan(b):
            return True
        if math.isnan(a) or math.isnan(b):
            return False
        if math.isinf(a) or math.isinf(b):
            return a == b
        diff = abs(a - b)
        if diff <= self._eps:
            return True
        largest = max(abs(a), abs(b))
        return diff <= largest * self._eps
