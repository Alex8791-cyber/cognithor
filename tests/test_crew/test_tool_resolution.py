from unittest.mock import MagicMock

import pytest

from cognithor.crew.errors import ToolNotFoundError
from cognithor.crew.tool_resolver import (
    available_tool_names,
    did_you_mean,
    resolve_tools,
)


class TestAvailableToolNames:
    def test_uses_get_tools_for_role_all(self):
        """available_tool_names() pulls every tool regardless of role."""
        registry = MagicMock()
        tool_a = MagicMock(name="tool_a")
        tool_a.name = "web_search"
        tool_b = MagicMock(name="tool_b")
        tool_b.name = "pdf_reader"
        registry.get_tools_for_role.return_value = [tool_a, tool_b]
        names = available_tool_names(registry)
        registry.get_tools_for_role.assert_called_once_with("all")
        assert names == ["web_search", "pdf_reader"]


class TestResolveTools:
    def _registry_with(self, names: list[str]) -> MagicMock:
        registry = MagicMock()
        tools = []
        for n in names:
            m = MagicMock()
            m.name = n
            tools.append(m)
        registry.get_tools_for_role.return_value = tools
        return registry

    def test_resolves_known_tools(self):
        registry = self._registry_with(["web_search", "pdf_reader", "shell_run"])
        resolved = resolve_tools(["web_search", "pdf_reader"], registry=registry)
        assert resolved == ["web_search", "pdf_reader"]

    def test_unknown_tool_raises_with_suggestion(self):
        registry = self._registry_with(["web_search", "pdf_reader"])
        with pytest.raises(ToolNotFoundError) as exc:
            resolve_tools(["web_seach"], registry=registry)
        assert "web_seach" in str(exc.value)
        assert "web_search" in str(exc.value)

    def test_unknown_tool_no_close_match(self):
        registry = self._registry_with(["completely_other"])
        with pytest.raises(ToolNotFoundError) as exc:
            resolve_tools(["totally_foreign"], registry=registry)
        assert "totally_foreign" in str(exc.value)
        assert "Meintest du" not in str(exc.value)


class TestDidYouMean:
    def test_close_match(self):
        assert did_you_mean("web_seach", ["web_search", "pdf_reader"]) == "web_search"

    def test_no_match(self):
        assert did_you_mean("xyz", ["web_search"]) is None

    def test_exact_match_returns_none(self):
        # No suggestion when exact match exists
        assert did_you_mean("web_search", ["web_search"]) is None
