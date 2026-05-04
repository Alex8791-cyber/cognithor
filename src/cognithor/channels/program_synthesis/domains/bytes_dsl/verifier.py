"""BinaryData pipeline verifier (Sprint-26.4).

Same pipeline-spec shape as Datetime/JSON/Float verifiers. Bytes
equality is exact; integer/string outputs use plain ``==``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from cognithor.channels.program_synthesis.domains.bytes_dsl.catalog import (
    BytesCatalog,
    build_bytes_catalog,
)

if TYPE_CHECKING:
    from collections.abc import Iterable


class BytesVerifierError(Exception):
    """Raised when a Bytes pipeline is malformed or fails verification."""


class BytesVerifier:
    """Run a Bytes pipeline spec against example inputs."""

    def __init__(self, catalog: BytesCatalog | None = None) -> None:
        self._catalog = catalog or build_bytes_catalog()

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
                raise BytesVerifierError(msg)
        return True

    @staticmethod
    def _coerce_steps(program: Any) -> list[Mapping[str, Any]]:
        if isinstance(program, list):
            steps = program
        elif isinstance(program, Mapping):
            raw = program.get("program")
            if not isinstance(raw, list):
                msg = f"Bytes program 'program' field must be a list, got {type(raw).__name__}"
                raise BytesVerifierError(msg)
            steps = raw
        else:
            msg = f"Bytes program must be list or {{'program': list}}, got {type(program).__name__}"
            raise BytesVerifierError(msg)
        for step in steps:
            if not isinstance(step, Mapping):
                msg = f"Pipeline step must be a mapping, got {type(step).__name__}"
                raise BytesVerifierError(msg)
            if "primitive" not in step:
                msg = "Pipeline step missing 'primitive' key"
                raise BytesVerifierError(msg)
        return list(steps)

    def _run(self, steps: list[Mapping[str, Any]], input_value: Any) -> Any:
        cur: Any = input_value
        for step in steps:
            name = step["primitive"]
            try:
                primitive = self._catalog.get(name)
            except KeyError as exc:
                raise BytesVerifierError(str(exc)) from exc
            args = step.get("args", {})
            if not isinstance(args, Mapping):
                msg = f"Step {name!r}: 'args' must be a mapping"
                raise BytesVerifierError(msg)
            try:
                cur = primitive.fn(cur, **args)
            except (TypeError, ValueError) as exc:
                msg = f"Step {name!r}: {type(exc).__name__}: {exc}"
                raise BytesVerifierError(msg) from exc
        return cur
