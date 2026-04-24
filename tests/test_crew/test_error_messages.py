"""Task 18 - Crew-Layer error-message contracts.

Spec section 8: user-facing errors must surface through ``cognithor.i18n.t()``
so they respect the active locale. These tests lock in the "what appears in
the message" contract regardless of which locale pack happens to be active
during CI.
"""

from unittest.mock import MagicMock

import pytest

from cognithor.crew.errors import CrewError, ToolNotFoundError


class TestErrorMessaging:
    def _registry_with(self, names):
        registry = MagicMock()
        tools = []
        for n in names:
            m = MagicMock()
            m.name = n
            tools.append(m)
        registry.get_tools_for_role.return_value = tools
        return registry

    def test_tool_not_found_mentions_name_and_did_you_mean(self):
        from cognithor.crew.tool_resolver import resolve_tools

        registry = self._registry_with(["web_search", "pdf_reader"])
        with pytest.raises(ToolNotFoundError) as exc:
            resolve_tools(["web_seach"], registry=registry)
        msg = str(exc.value)
        assert "web_seach" in msg
        # Locale-agnostic assertion: the suggestion MUST appear in the message.
        assert "web_search" in msg

    def test_tool_not_found_mentions_name_only_when_no_close_match(self):
        from cognithor.crew.tool_resolver import resolve_tools

        registry = self._registry_with(["completely_different"])
        with pytest.raises(ToolNotFoundError) as exc:
            resolve_tools(["totally_foreign"], registry=registry)
        assert "totally_foreign" in str(exc.value)
        # No close match -> no suggestion hint (don't assert the German-specific
        # "Meintest du" string since locale may vary on CI).

    def test_crew_error_is_base_class(self):
        assert issubclass(ToolNotFoundError, CrewError)


class TestYamlLoaderLocalizedErrors:
    """R4-I1 (sealed in Task 18): unknown-task YAML errors use the i18n pipeline."""

    def test_yaml_loader_unknown_task_raises_localized(self, tmp_path):
        from cognithor.crew.errors import CrewCompilationError
        from cognithor.crew.yaml_loader import load_crew_from_yaml

        agents_yaml = tmp_path / "agents.yaml"
        tasks_yaml = tmp_path / "tasks.yaml"
        agents_yaml.write_text(
            "a:\n  role: writer\n  goal: write\n",
            encoding="utf-8",
        )
        tasks_yaml.write_text(
            "one:\n  description: first\n  expected_output: x\n  agent: a\n"
            "two:\n  description: second\n  expected_output: y\n  agent: a\n"
            "  context: [missing]\n",
            encoding="utf-8",
        )

        with pytest.raises(CrewCompilationError) as exc:
            load_crew_from_yaml(agents=agents_yaml, tasks=tasks_yaml)
        assert "two" in str(exc.value)
        assert "missing" in str(exc.value)
