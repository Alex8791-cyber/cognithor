"""JSON pipeline verifier (Sprint-26.2).

A JSON program is a *pipeline spec* — a list of dicts where each entry
names a primitive plus its keyword arguments. The verifier interprets
the spec deterministically (no eval, no exec) and compares the
resulting output against each example's expected value.

Pipeline shape::

    [
        {"primitive": "field", "args": {"name": "user"}},
        {"primitive": "field", "args": {"name": "name"}},
        {"primitive": "lower_", "args": {}},
    ]

Invariant: pipeline executes left-to-right, each step receives the
previous step's output as the first positional argument.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from cognithor.channels.program_synthesis.domains.json_dsl.catalog import (
    JsonCatalog,
    build_json_catalog,
)


class JsonVerifierError(Exception):
    """Raised when a pipeline is malformed or fails verification."""


class JsonVerifier:
    """Run a JSON pipeline spec against example inputs."""

    def __init__(self, catalog: JsonCatalog | None = None) -> None:
        self._catalog = catalog or build_json_catalog()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify(
        self,
        program: Any,
        examples: Iterable[Mapping[str, Any]],
    ) -> bool:
        """Return True iff the pipeline reproduces every example.

        ``program`` is either a list of pipeline steps directly or
        ``{"program": [...]}`` (the shape the LLM-prior emits).
        """
        steps = self._coerce_steps(program)
        for index, example in enumerate(examples):
            actual = self._run(steps, example.get("input"))
            expected = example.get("output")
            if actual != expected:
                msg = f"Example {index}: pipeline output {actual!r} != expected {expected!r}"
                raise JsonVerifierError(msg)
        return True

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce_steps(program: Any) -> list[Mapping[str, Any]]:
        if isinstance(program, list):
            steps = program
        elif isinstance(program, Mapping):
            raw = program.get("program")
            if not isinstance(raw, list):
                msg = f"JSON program 'program' field must be a list, got {type(raw).__name__}"
                raise JsonVerifierError(msg)
            steps = raw
        else:
            msg = f"JSON program must be list or {{'program': list}}, got {type(program).__name__}"
            raise JsonVerifierError(msg)
        for step in steps:
            if not isinstance(step, Mapping):
                msg = f"Pipeline step must be a mapping, got {type(step).__name__}"
                raise JsonVerifierError(msg)
            if "primitive" not in step:
                msg = "Pipeline step missing 'primitive' key"
                raise JsonVerifierError(msg)
        return list(steps)

    def _run(self, steps: list[Mapping[str, Any]], input_value: Any) -> Any:
        cur: Any = input_value
        for step in steps:
            name = step["primitive"]
            try:
                primitive = self._catalog.get(name)
            except KeyError as exc:
                raise JsonVerifierError(str(exc)) from exc
            args = step.get("args", {})
            if not isinstance(args, Mapping):
                msg = f"Step {name!r}: 'args' must be a mapping"
                raise JsonVerifierError(msg)
            try:
                cur = primitive.fn(cur, **args)
            except TypeError as exc:
                msg = f"Step {name!r} TypeError: {exc}"
                raise JsonVerifierError(msg) from exc
        return cur
