"""Tests für Wetter Abfrage."""

import pytest
from .skill import WetterAbfrageSkill


class TestWetterAbfrageSkill:
    def test_execute(self) -> None:
        skill = WetterAbfrageSkill()
        # TODO: Test implementieren
        assert skill.NAME == "wetter_abfrage"
