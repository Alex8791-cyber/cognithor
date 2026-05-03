"""ARC-AGI-3 LLM code-generation solver.

Prompts an LLM to write a Python ``transform(grid)`` function, then
validates and executes that function inside a restricted sandbox.
"""

from __future__ import annotations

import ast
import re
import textwrap
from typing import Any

from cognithor.arc.classic.task_parser import ArcTask, Grid, Solution
from cognithor.utils.logging import get_logger

log = get_logger(__name__)

__all__ = ["LLMSolver"]

# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------

_BLOCKED_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bimport\b"),
    re.compile(r"\bexec\b"),
    re.compile(r"\beval\b"),
    re.compile(r"\bopen\b"),
    re.compile(r"\bsubprocess\b"),
    re.compile(r"\bos\b"),
    re.compile(r"\bsys\b"),
    re.compile(r"\b__import__\b"),
    re.compile(r"\bcompile\b"),
    re.compile(r"\bglobals\b"),
    re.compile(r"\blocals\b"),
    re.compile(r"\bvars\b"),
    re.compile(r"\bgetattr\b"),
    re.compile(r"\bsetattr\b"),
    re.compile(r"\bdelattr\b"),
    re.compile(r"\b__builtins__\b"),
    re.compile(r"\b__class__\b"),
    re.compile(r"\b__subclasses__\b"),
    re.compile(r"\bmro\b"),
]

# Only these names are exposed in the sandbox __builtins__.
_SAFE_BUILTINS: dict[str, Any] = {
    "len": len,
    "range": range,
    "zip": zip,
    "enumerate": enumerate,
    "min": min,
    "max": max,
    "sum": sum,
    "sorted": sorted,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "int": int,
    "str": str,
    "bool": bool,
    "abs": abs,
    "True": True,
    "False": False,
    "None": None,
    "any": any,
    "all": all,
    "map": map,
    "filter": filter,
}

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATE = textwrap.dedent(
    """\
    You are an expert at solving ARC-AGI-3 puzzles.

    Study the following input/output examples and write a Python function
    called `transform(grid)` that converts any input grid to the correct
    output grid.

    Rules:
    - The function must be named exactly `transform`.
    - It receives a `list[list[int]]` and must return a `list[list[int]]`.
    - Do NOT use any imports; rely only on built-ins: len, range, zip,
      enumerate, min, max, sum, sorted, list, dict, set, tuple, int, str,
      bool, abs, any, all, map, filter.
    - Keep the solution concise and correct.

    {examples}

    Respond ONLY with a Python code block:
    ```python
    def transform(grid):
        ...
    ```
    """
)


# ---------------------------------------------------------------------------
# LLMSolver
# ---------------------------------------------------------------------------


class LLMSolver:
    """Solve ARC tasks by asking an LLM to generate a ``transform`` function."""

    #: Model to use for code generation.
    MODEL: str = "qwen2.5-coder:7b"

    def __init__(self, model: str | None = None) -> None:
        self._model = model or self.MODEL

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def solve(self, task: ArcTask, max_attempts: int = 3) -> list[Solution]:
        """Try up to *max_attempts* times to generate a valid transform.

        Returns a (possibly empty) list of :class:`Solution` objects whose
        output is the result of running the validated transform on the test
        input.
        """
        solutions: list[Solution] = []
        prompt = self._format_task(task)

        for attempt in range(1, max_attempts + 1):
            log.debug("LLMSolver attempt %d/%d for task %s", attempt, max_attempts, task.task_id)
            try:
                response = await self._llm_call(prompt)
            except Exception as exc:
                log.warning("LLM call failed on attempt %d: %s", attempt, exc)
                continue

            code = self._extract_python(response)
            if not code:
                log.debug("No Python block found in LLM response (attempt %d)", attempt)
                continue

            if not self._is_safe(code):
                log.warning("Blocked unsafe code from LLM (attempt %d)", attempt)
                continue

            if not self._validates(code, task):
                log.debug("Generated code fails example validation (attempt %d)", attempt)
                continue

            # Run on the actual test input.
            output = self._execute_in_sandbox(code, task.test_input)
            if output is None:
                log.debug("Sandbox execution returned None for test input (attempt %d)", attempt)
                continue

            solutions.append(
                Solution(
                    output=output,
                    method="llm",
                    description=f"LLM-generated transform (attempt {attempt})",
                    complexity=attempt,
                    transform_fn=None,
                )
            )
            break  # Stop after first working solution.

        return solutions

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def _format_task(self, task: ArcTask) -> str:
        """Format the task as a human-readable prompt string."""
        lines: list[str] = []
        for i, (inp, out) in enumerate(task.examples, start=1):
            lines.append(f"Example {i}:")
            lines.append(f"  Input:  {inp!r}")
            lines.append(f"  Output: {out!r}")
        examples_text = "\n".join(lines)
        return _PROMPT_TEMPLATE.format(examples=examples_text)

    # ------------------------------------------------------------------
    # Code extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_python(response: str) -> str:
        """Return the first ```python ... ``` block, or empty string."""
        match = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    # ------------------------------------------------------------------
    # Safety
    # ------------------------------------------------------------------

    @staticmethod
    def _is_safe(code: str) -> bool:
        """Return True when none of the blocked patterns appear in *code*."""
        for pattern in _BLOCKED_PATTERNS:
            if pattern.search(code):
                return False
        # Additional AST-level check: reject any Import/ImportFrom nodes.
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return False
        return all(not isinstance(node, ast.Import | ast.ImportFrom) for node in ast.walk(tree))

    # ------------------------------------------------------------------
    # Sandbox execution
    # ------------------------------------------------------------------

    @staticmethod
    def _execute_in_sandbox(code: str, test_input: Grid) -> Grid | None:
        """Execute *code* in a restricted namespace and call ``transform``.

        Returns the resulting grid, or *None* on any error.
        """
        namespace: dict[str, Any] = {"__builtins__": _SAFE_BUILTINS}
        try:
            exec(code, namespace)  # sandbox by design
        except Exception as exc:
            log.debug("Sandbox exec error: %s", exc)
            return None

        transform = namespace.get("transform")
        if not callable(transform):
            log.debug("No callable 'transform' found after exec")
            return None

        try:
            result = transform(test_input)
        except Exception as exc:
            log.debug("transform() raised: %s", exc)
            return None

        # Basic type validation: must be a non-empty list of lists of ints.
        if not isinstance(result, list) or not result:
            return None
        if not all(isinstance(row, list) for row in result):
            return None
        return result

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validates(self, code: str, task: ArcTask) -> bool:
        """Return True iff the code produces correct output for ALL examples."""
        for inp, expected in task.examples:
            output = self._execute_in_sandbox(code, inp)
            if output != expected:
                return False
        return True

    # ------------------------------------------------------------------
    # LLM call (overridable in tests)
    # ------------------------------------------------------------------

    async def _llm_call(self, prompt: str) -> str:  # pragma: no cover
        """Send *prompt* to the configured LLM and return the raw text.

        Subclasses (or test mocks) can override this method.
        """
        try:
            from cognithor.config import get_config  # type: ignore[attr-defined]
            from cognithor.core.llm_backend import create_backend
        except ImportError as exc:
            raise RuntimeError("cognithor.core.llm_backend not available") from exc

        config = get_config()
        backend = create_backend(config)
        try:
            response = await backend.chat(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            return response.content
        finally:
            await backend.close()
