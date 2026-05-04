"""Tests for the AST/Code domain (Sprint-26.3)."""

from __future__ import annotations

import pytest

from cognithor.channels.program_synthesis.domains.ast_dsl import (
    AstDomain,
    AstVerifierError,
    SandboxConfig,
    register_ast_domain,
    run_in_sandbox,
)
from cognithor.channels.program_synthesis.domains.registry import DomainRegistry


class TestSandbox:
    def test_runs_simple_function(self) -> None:
        result = run_in_sandbox(
            "def add(a, b):\n    return a + b\n",
            "add",
            (2, 3),
        )
        assert result.ok
        assert result.value == 5

    def test_returns_complex_value(self) -> None:
        result = run_in_sandbox(
            "def make_dict(k, v):\n    return {k: v}\n",
            "make_dict",
            ("hello", 42),
        )
        assert result.ok
        assert result.value == {"hello": 42}

    def test_timeout(self) -> None:
        result = run_in_sandbox(
            "def loop():\n    while True:\n        pass\n",
            "loop",
            (),
            config=SandboxConfig(timeout_seconds=0.5),
        )
        assert not result.ok
        assert result.error_kind == "timeout"

    def test_function_exception_caught(self) -> None:
        result = run_in_sandbox(
            "def bad():\n    raise ValueError('boom')\n",
            "bad",
            (),
        )
        assert not result.ok
        assert result.error_kind == "exception"
        assert "ValueError" in result.error_message
        assert "boom" in result.error_message

    def test_sandbox_config_defaults(self) -> None:
        cfg = SandboxConfig()
        assert cfg.timeout_seconds == 2.0
        assert cfg.memory_mb == 128

    def test_kwargs_passed_through(self) -> None:
        result = run_in_sandbox(
            "def add(a, b=10):\n    return a + b\n",
            "add",
            (5,),
            kwargs={"b": 20},
        )
        assert result.ok
        assert result.value == 25


class TestAstVerifier:
    def test_verifies_passing(self) -> None:
        d = AstDomain()
        ok = d.verify(
            "def add(a, b):\n    return a + b\n",
            [
                {"args": [1, 2], "output": 3},
                {"args": [4, 5], "output": 9},
            ],
        )
        assert ok

    def test_verifies_dict_program(self) -> None:
        d = AstDomain()
        ok = d.verify(
            {"function": "def neg(x):\n    return -x\n"},
            [{"args": [5], "output": -5}],
        )
        assert ok

    def test_mismatch_raises(self) -> None:
        d = AstDomain()
        with pytest.raises(AstVerifierError, match="!= expected"):
            d.verify(
                "def add(a, b):\n    return a + b\n",
                [{"args": [1, 2], "output": 99}],
            )

    def test_runtime_exception_raises(self) -> None:
        d = AstDomain()
        with pytest.raises(AstVerifierError, match="exception"):
            d.verify(
                "def div(a, b):\n    return a / b\n",
                [{"args": [1, 0], "output": 0}],
            )

    def test_syntax_error_raises(self) -> None:
        d = AstDomain()
        with pytest.raises(AstVerifierError, match="parse error"):
            d.verify("def broken(:\n", [])

    def test_no_function_raises(self) -> None:
        d = AstDomain()
        with pytest.raises(AstVerifierError, match="no function"):
            d.verify("x = 1\n", [])

    def test_empty_program_rejected(self) -> None:
        d = AstDomain()
        with pytest.raises(AstVerifierError, match="empty"):
            d.verify("", [])

    def test_program_must_be_str_or_dict(self) -> None:
        d = AstDomain()
        with pytest.raises(AstVerifierError, match="must be"):
            d.verify(42, [])  # type: ignore[arg-type]

    def test_dict_program_non_str_function(self) -> None:
        d = AstDomain()
        with pytest.raises(AstVerifierError, match="must be a string"):
            d.verify({"function": 42}, [])

    def test_banned_module_rejected(self) -> None:
        d = AstDomain()
        with pytest.raises(AstVerifierError, match="banned module"):
            d.verify(
                "import os\ndef f():\n    return os.getpid()\n",
                [],
            )

    def test_banned_module_via_from_import(self) -> None:
        d = AstDomain()
        with pytest.raises(AstVerifierError, match="banned module"):
            d.verify(
                "from subprocess import run\ndef f():\n    return 1\n",
                [],
            )

    def test_banned_name_eval_rejected(self) -> None:
        d = AstDomain()
        with pytest.raises(AstVerifierError, match="banned name"):
            d.verify(
                "def f():\n    return eval('1+1')\n",
                [],
            )

    def test_safe_stdlib_function_passes(self) -> None:
        d = AstDomain()
        # math is NOT banned — synthesized functions can use stdlib.
        # We avoid imports here for simplicity, but `import math` is OK.
        ok = d.verify(
            "def square(x):\n    return x * x\n",
            [{"args": [4], "output": 16}],
        )
        assert ok


class TestAstDomain:
    def test_metadata(self) -> None:
        d = AstDomain()
        m = d.metadata
        assert m.name == "ast"
        assert m.benchmark_name == "humaneval-plus"
        assert m.benchmark_target == 0.45

    def test_register(self) -> None:
        reg = DomainRegistry()
        register_ast_domain(reg)
        assert isinstance(reg.get("ast"), AstDomain)

    def test_primitives_returns_null_catalog(self) -> None:
        d = AstDomain()
        cat = d.primitives()
        assert len(cat) == 0
        assert cat.names() == []
        assert "anything" not in cat
        with pytest.raises(KeyError):
            cat.get("anything")
