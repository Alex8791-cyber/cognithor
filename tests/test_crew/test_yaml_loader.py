"""Task 16 — YAML config loader for Crews.

Loads two YAML files (agents.yaml + tasks.yaml) into a Crew object. Tasks
reference agents and earlier tasks by YAML key; the loader resolves
references in two passes because Pydantic models are frozen.
"""

from pathlib import Path

import pytest

from cognithor.crew import Crew, CrewProcess
from cognithor.crew.yaml_loader import load_crew_from_yaml


class TestYamlLoader:
    def test_loads_two_agent_crew(self):
        fixtures = Path(__file__).parent / "fixtures"
        crew = load_crew_from_yaml(
            agents=fixtures / "sample_agents.yaml",
            tasks=fixtures / "sample_tasks.yaml",
            process=CrewProcess.SEQUENTIAL,
        )
        assert isinstance(crew, Crew)
        assert len(crew.agents) == 2
        assert len(crew.tasks) == 2
        assert crew.agents[0].role == "analyst"
        # Second task's context resolves to first task (by YAML key)
        assert crew.tasks[1].context[0].task_id == crew.tasks[0].task_id

    def test_missing_agent_reference_raises(self, tmp_path: Path):
        (tmp_path / "a.yaml").write_text("x: {role: x, goal: y}\n")
        (tmp_path / "t.yaml").write_text(
            "t1: {description: d, expected_output: e, agent: unknown}\n"
        )
        with pytest.raises(ValueError, match="unknown"):
            load_crew_from_yaml(agents=tmp_path / "a.yaml", tasks=tmp_path / "t.yaml")

    def test_yaml_loader_unknown_task_raises_localized(self, tmp_path: Path):
        """R4-I1: referencing an unknown task in context[] raises
        CrewCompilationError (not bare ValueError) with the referring task
        alias and the missing ref both surfaced.
        """
        from cognithor.crew.errors import CrewCompilationError

        (tmp_path / "a.yaml").write_text("x: {role: x, goal: y}\n")
        (tmp_path / "t.yaml").write_text(
            "two:\n  description: d\n  expected_output: e\n  agent: x\n  context: [missing]\n"
        )
        with pytest.raises(CrewCompilationError) as exc:
            load_crew_from_yaml(agents=tmp_path / "a.yaml", tasks=tmp_path / "t.yaml")
        msg = str(exc.value)
        assert "two" in msg
        assert "missing" in msg
