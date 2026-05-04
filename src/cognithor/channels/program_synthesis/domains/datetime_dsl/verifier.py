"""Datetime pipeline verifier (Sprint-26.3).

A Datetime program is a pipeline-spec — same shape as the JSON
verifier in 26.2: ``[{primitive, args}, ...]``. The first step's
input is the example's ``input`` value; subsequent steps receive the
previous step's output as the first positional argument.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from cognithor.channels.program_synthesis.domains.datetime_dsl.catalog import (
    DatetimeCatalog,
    build_datetime_catalog,
)

if TYPE_CHECKING:
    from collections.abc import Iterable


class DatetimeVerifierError(Exception):
    """Raised when a Datetime pipeline is malformed or fails verification."""


class DatetimeVerifier:
    """Run a Datetime pipeline spec against example inputs."""

    def __init__(self, catalog: DatetimeCatalog | None = None) -> None:
        self._catalog = catalog or build_datetime_catalog()

    def verify(
        self,
        program: Any,
        examples: Iterable[Mapping[str, Any]],
    ) -> bool:
        steps = self._coerce_steps(program)
        for index, example in enumerate(examples):
            actual = self._run(steps, example.get("input"))
            expected = example.get("output")
            if actual != expected:
                msg = f"Example {index}: pipeline output {actual!r} != expected {expected!r}"
                raise DatetimeVerifierError(msg)
        return True

    @staticmethod
    def _coerce_steps(program: Any) -> list[Mapping[str, Any]]:
        if isinstance(program, list):
            steps = program
        elif isinstance(program, Mapping):
            raw = program.get("program")
            if not isinstance(raw, list):
                msg = f"Datetime program 'program' field must be a list, got {type(raw).__name__}"
                raise DatetimeVerifierError(msg)
            steps = raw
        else:
            msg = (
                "Datetime program must be list or {'program': list}, "
                f"got {type(program).__name__}"
            )
            raise DatetimeVerifierError(msg)
        for step in steps:
            if not isinstance(step, Mapping):
                msg = f"Pipeline step must be a mapping, got {type(step).__name__}"
                raise DatetimeVerifierError(msg)
            if "primitive" not in step:
                msg = "Pipeline step missing 'primitive' key"
                raise DatetimeVerifierError(msg)
        return list(steps)

    def _run(self, steps: list[Mapping[str, Any]], input_value: Any) -> Any:
        cur: Any = input_value
        for step in steps:
            name = step["primitive"]
            try:
                primitive = self._catalog.get(name)
            except KeyError as exc:
                raise DatetimeVerifierError(str(exc)) from exc
            args = step.get("args", {})
            if not isinstance(args, Mapping):
                msg = f"Step {name!r}: 'args' must be a mapping"
                raise DatetimeVerifierError(msg)
            try:
                cur = primitive.fn(cur, **args)
            except (TypeError, ValueError) as exc:
                msg = f"Step {name!r}: {type(exc).__name__}: {exc}"
                raise DatetimeVerifierError(msg) from exc
        return cur
