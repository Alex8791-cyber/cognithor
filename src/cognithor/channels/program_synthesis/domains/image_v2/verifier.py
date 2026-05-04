"""Image V2 pipeline verifier (Sprint-26.4).

Pipeline-spec interpreter. Equality is structural — input/output
grids are compared as tuple-of-tuples, ints stay ints.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from cognithor.channels.program_synthesis.domains.image_v2.catalog import (
    ImageV2Catalog,
    build_image_v2_catalog,
)

if TYPE_CHECKING:
    from collections.abc import Iterable


class ImageV2VerifierError(Exception):
    """Raised when an Image-V2 pipeline is malformed or fails."""


class ImageV2Verifier:
    def __init__(self, catalog: ImageV2Catalog | None = None) -> None:
        self._catalog = catalog or build_image_v2_catalog()

    def verify(
        self,
        program: Any,
        examples: Iterable[Mapping[str, Any]],
    ) -> bool:
        steps = self._coerce_steps(program)
        for index, example in enumerate(examples):
            actual = self._run(steps, example.get("input"))
            expected = example.get("output")
            if not self._grids_equal(actual, expected):
                msg = f"Example {index}: pipeline output {actual!r} != expected {expected!r}"
                raise ImageV2VerifierError(msg)
        return True

    @staticmethod
    def _coerce_steps(program: Any) -> list[Mapping[str, Any]]:
        if isinstance(program, list):
            steps = program
        elif isinstance(program, Mapping):
            raw = program.get("program")
            if not isinstance(raw, list):
                msg = f"Image program 'program' field must be a list, got {type(raw).__name__}"
                raise ImageV2VerifierError(msg)
            steps = raw
        else:
            msg = f"Image program must be list or {{'program': list}}, got {type(program).__name__}"
            raise ImageV2VerifierError(msg)
        for step in steps:
            if not isinstance(step, Mapping):
                msg = f"Pipeline step must be a mapping, got {type(step).__name__}"
                raise ImageV2VerifierError(msg)
            if "primitive" not in step:
                msg = "Pipeline step missing 'primitive' key"
                raise ImageV2VerifierError(msg)
        return list(steps)

    def _run(self, steps: list[Mapping[str, Any]], input_value: Any) -> Any:
        cur: Any = input_value
        for step in steps:
            name = step["primitive"]
            try:
                primitive = self._catalog.get(name)
            except KeyError as exc:
                raise ImageV2VerifierError(str(exc)) from exc
            args = step.get("args", {})
            if not isinstance(args, Mapping):
                msg = f"Step {name!r}: 'args' must be a mapping"
                raise ImageV2VerifierError(msg)
            try:
                cur = primitive.fn(cur, **args)
            except (TypeError, ValueError) as exc:
                msg = f"Step {name!r}: {type(exc).__name__}: {exc}"
                raise ImageV2VerifierError(msg) from exc
        return cur

    @staticmethod
    def _grids_equal(actual: Any, expected: Any) -> bool:
        """Compare grids structurally — treat list/tuple of int rows
        as equal regardless of which container the LLM emitted."""
        if actual is None or expected is None:
            return bool(actual == expected)
        if isinstance(actual, list | tuple) and isinstance(expected, list | tuple):
            try:
                a = tuple(tuple(r) for r in actual)
                e = tuple(tuple(r) for r in expected)
                return bool(a == e)
            except TypeError:
                return False
        return bool(actual == expected)
