"""AST verifier — runs synthesised Python functions in a sandbox.

Sprint-26.3 deliverable. The AST domain doesn't compose primitives at
synthesis time the way SQL/JSON/Datetime do — it accepts a complete
Python function source string from the LLM-prior and validates it
against examples.

Validation steps:

1. Parse with ``ast.parse`` (mypy-strict-friendly, deterministic).
2. Reject obvious banned constructs (``import``, ``exec``, ``eval``,
   network names) at the AST level — *not* a security boundary, just
   a hint to the LLM that those aren't synthesisable here.
3. Run the function against every example via :func:`run_in_sandbox`.
4. Compare outputs.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING, Any

from cognithor.channels.program_synthesis.domains.ast_dsl.sandbox import (
    SandboxConfig,
    run_in_sandbox,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


_BANNED_NAMES: frozenset[str] = frozenset(
    {
        "exec",
        "eval",
        "compile",
        "__import__",
        "open",
        "input",
        "globals",
    }
)


_BANNED_MODULES: frozenset[str] = frozenset(
    {
        "os",
        "sys",
        "subprocess",
        "socket",
        "urllib",
        "http",
        "requests",
        "ftplib",
        "smtplib",
        "shutil",
    }
)


class AstVerifierError(Exception):
    """Raised when AST verification fails."""


class AstVerifier:
    """Verify a synthesised Python function against examples."""

    def __init__(self, *, config: SandboxConfig | None = None) -> None:
        self._config = config or SandboxConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify(
        self,
        program: Any,
        examples: Iterable[Mapping[str, Any]],
    ) -> bool:
        function_source, function_name = self._coerce_program(program)
        self._lint_ast(function_source, function_name)

        for index, example in enumerate(examples):
            args = tuple(example.get("args", ()))
            kwargs = dict(example.get("kwargs", {}))
            result = run_in_sandbox(
                function_source,
                function_name,
                args,
                kwargs,
                self._config,
            )
            if not result.ok:
                msg = f"Example {index}: sandbox {result.error_kind!r} — {result.error_message}"
                raise AstVerifierError(msg)
            expected = example.get("output")
            if result.value != expected:
                msg = (
                    f"Example {index}: function returned {result.value!r} != expected {expected!r}"
                )
                raise AstVerifierError(msg)
        return True

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce_program(program: Any) -> tuple[str, str]:
        """Return (function_source, function_name) for any accepted shape."""
        if isinstance(program, str):
            source = program
        elif isinstance(program, dict):
            value = program.get("function") or program.get("source") or ""
            if not isinstance(value, str):
                msg = f"AST program 'function' field must be a string, got {type(value).__name__}"
                raise AstVerifierError(msg)
            source = value
        else:
            msg = f"AST program must be str or {{'function': str}}, got {type(program).__name__}"
            raise AstVerifierError(msg)
        if not source.strip():
            msg = "empty AST function source"
            raise AstVerifierError(msg)

        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            raise AstVerifierError(f"AST parse error: {exc}") from exc

        # First top-level FunctionDef is the entry point.
        for node in tree.body:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                return source, node.name
        msg = "AST program contains no function definition"
        raise AstVerifierError(msg)

    @staticmethod
    def _lint_ast(source: str, function_name: str) -> None:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import | ast.ImportFrom):
                module_root = node.module if isinstance(node, ast.ImportFrom) else None
                if not module_root and isinstance(node, ast.Import):
                    module_root = node.names[0].name
                top = (module_root or "").split(".")[0]
                if top in _BANNED_MODULES:
                    msg = (
                        f"AST source imports banned module {top!r}; "
                        "synthesise pure-stdlib functions only "
                        "(no network, no I/O)"
                    )
                    raise AstVerifierError(msg)
            elif isinstance(node, ast.Name) and node.id in _BANNED_NAMES:
                msg = f"AST source uses banned name {node.id!r}"
                raise AstVerifierError(msg)
        # Defensive: confirm the function is still discoverable after
        # the lint pass (`function_name` came from `_coerce_program`).
        for top_node in tree.body:
            if (
                isinstance(top_node, ast.FunctionDef | ast.AsyncFunctionDef)
                and top_node.name == function_name
            ):
                return
        msg = f"AST source no longer defines {function_name!r}"
        raise AstVerifierError(msg)
