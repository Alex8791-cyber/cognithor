import pytest

from cognithor.crew.process import CrewProcess


class TestCrewProcess:
    def test_has_sequential_and_hierarchical(self):
        assert CrewProcess.SEQUENTIAL.value == "sequential"
        assert CrewProcess.HIERARCHICAL.value == "hierarchical"

    def test_two_members_only(self):
        assert len(CrewProcess) == 2

    def test_from_string_roundtrip(self):
        assert CrewProcess("sequential") is CrewProcess.SEQUENTIAL
        assert CrewProcess("hierarchical") is CrewProcess.HIERARCHICAL

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            CrewProcess("parallel")

    def test_stringifies_for_logging(self):
        assert "SEQUENTIAL" in repr(CrewProcess.SEQUENTIAL)
