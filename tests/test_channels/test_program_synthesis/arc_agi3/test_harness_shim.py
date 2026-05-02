# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-11 Wave-6 — harness shim tests.

The arcengine harness isn't installed in this repo's CI environment,
so the canonical test is "factory raises a clear ImportError with
install instructions". The factory is invoked the same way from
inside a real ARC-AGI-3-Agents clone — only there the import
succeeds and the returned class actually subclasses the harness ABC.
"""

from __future__ import annotations

import pytest

from cognithor.channels.program_synthesis.arc_agi3.agent import RandomActionAgent
from cognithor.channels.program_synthesis.arc_agi3.harness_shim import (
    cognithor_agent_factory,
)
from cognithor.channels.program_synthesis.integration.capability_tokens import (  # noqa: F401
    PSECapability as _PSECapability,
)


class TestHarnessShim:
    def test_raises_import_error_without_harness(self) -> None:
        """The arcengine harness is not installed in CI; the factory
        must surface a clear actionable error rather than a cryptic
        ``ModuleNotFoundError``."""
        with pytest.raises(ImportError, match="ARC-AGI-3-Agents harness is not installed"):
            cognithor_agent_factory(delegate=RandomActionAgent())

    def test_error_message_includes_install_steps(self) -> None:
        """The hint walks the reader through the install path so
        operators don't need to read the source to fix it."""
        with pytest.raises(ImportError) as exc_info:
            cognithor_agent_factory(delegate=RandomActionAgent())
        msg = str(exc_info.value)
        assert "git clone" in msg
        assert "uv sync" in msg
        assert "uv pip install -e" in msg
