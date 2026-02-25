"""Tests für Test."""

import pytest
from .skill import TestSkill


class TestTestSkill:
    def test_execute(self) -> None:
        skill = TestSkill()
        # TODO: Test implementieren
        assert skill.NAME == "test"
