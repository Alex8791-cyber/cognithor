"""Verify cognithor_bench installs and exposes its public surface."""

import pytest

pytestmark = pytest.mark.xfail(
    reason="runner / cli / adapters arrive in Tasks 11-13",
    strict=False,
)


def test_package_imports() -> None:
    import cognithor_bench

    assert hasattr(cognithor_bench, "__version__")


def test_runner_module_imports() -> None:
    from cognithor_bench import runner

    assert hasattr(runner, "BenchRunner")


def test_cli_module_imports() -> None:
    from cognithor_bench import cli

    assert hasattr(cli, "main")


def test_cognithor_adapter_imports() -> None:
    from cognithor_bench.adapters import cognithor_adapter

    assert hasattr(cognithor_adapter, "CognithorAdapter")
