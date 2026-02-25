"""Tests für Test Skill."""

import pytest
from .skill import TestSkillSkill


class TestTestSkillSkill:
    def test_execute(self) -> None:
        skill = TestSkillSkill()
        # TODO: Test implementieren
        assert skill.NAME == "test_skill"
