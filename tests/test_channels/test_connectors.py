"""Tests: Enterprise-Konnektoren (Review-Punkt 7).

  - TeamsConnector: Nachrichten senden, Channels
  - JiraConnector: Issues CRUD, Transitions
  - ServiceNowConnector: Incidents, Knowledge-Base
  - ConnectorRegistry: Verwaltung
  - ScopeGuard: Least-Privilege, Rate-Limiting
"""

from __future__ import annotations

import pytest

from jarvis.channels.connectors import (
    TeamsConnector,
    JiraConnector,
    ServiceNowConnector,
    ConnectorRegistry,
    ScopeGuard,
    ScopePolicy,
    ConnectorScope,
    ConnectorConfig,
    ConnectorStatus,
)


class TestTeamsConnector:
    def test_send_message(self) -> None:
        teams = TeamsConnector()
        msg = teams.send_message("ch-1", "Hello Team!")
        assert msg["text"] == "Hello Team!"
        assert msg["channel_id"] == "ch-1"
        assert teams.sent_count == 1

    def test_send_with_card(self) -> None:
        teams = TeamsConnector()
        card = {"type": "AdaptiveCard", "body": [{"type": "TextBlock", "text": "Hi"}]}
        msg = teams.send_message("ch-1", "Card msg", card=card)
        assert msg["card"] is not None

    def test_register_and_list_channels(self) -> None:
        teams = TeamsConnector()
        teams.register_channel("ch-1", "General")
        teams.register_channel("ch-2", "Dev")
        channels = teams.list_channels("team-1")
        assert len(channels) == 2

    def test_connect(self) -> None:
        teams = TeamsConnector()
        assert teams.connect() is True
        assert teams.status == ConnectorStatus.CONNECTED

    def test_stats(self) -> None:
        teams = TeamsConnector()
        teams.send_message("ch-1", "test")
        stats = teams.stats()
        assert stats["total_requests"] == 1


class TestJiraConnector:
    def test_create_issue(self) -> None:
        jira = JiraConnector()
        issue = jira.create_issue("PROJ", "Fix Bug", description="Critical bug")
        assert issue["key"] == "PROJ-1"
        assert issue["status"] == "To Do"

    def test_get_issue(self) -> None:
        jira = JiraConnector()
        jira.create_issue("PROJ", "Task")
        issue = jira.get_issue("PROJ-1")
        assert issue is not None
        assert issue["summary"] == "Task"

    def test_get_nonexistent(self) -> None:
        jira = JiraConnector()
        assert jira.get_issue("NOPE-1") is None

    def test_transition_issue(self) -> None:
        jira = JiraConnector()
        jira.create_issue("PROJ", "Task")
        assert jira.transition_issue("PROJ-1", "In Progress") is True
        issue = jira.get_issue("PROJ-1")
        assert issue["status"] == "In Progress"

    def test_transition_nonexistent(self) -> None:
        jira = JiraConnector()
        assert jira.transition_issue("NOPE-1", "Done") is False

    def test_search_by_project(self) -> None:
        jira = JiraConnector()
        jira.create_issue("ALPHA", "Task A")
        jira.create_issue("BETA", "Task B")
        jira.create_issue("ALPHA", "Task C")
        results = jira.search_issues(project="ALPHA")
        assert len(results) == 2

    def test_issue_count(self) -> None:
        jira = JiraConnector()
        jira.create_issue("P", "A")
        jira.create_issue("P", "B")
        assert jira.issue_count == 2


class TestServiceNowConnector:
    def test_create_incident(self) -> None:
        snow = ServiceNowConnector()
        inc = snow.create_incident("Server down", urgency=1, impact=1)
        assert inc["number"] == "INC0000001"
        assert inc["state"] == "New"

    def test_get_incident(self) -> None:
        snow = ServiceNowConnector()
        snow.create_incident("Test")
        inc = snow.get_incident("INC0000001")
        assert inc is not None

    def test_update_incident(self) -> None:
        snow = ServiceNowConnector()
        snow.create_incident("Test")
        updated = snow.update_incident("INC0000001", state="In Progress")
        assert updated["state"] == "In Progress"

    def test_update_nonexistent(self) -> None:
        snow = ServiceNowConnector()
        assert snow.update_incident("INC9999999") is None

    def test_knowledge_base(self) -> None:
        snow = ServiceNowConnector()
        snow.add_knowledge_article("VPN Setup", "How to configure VPN…")
        snow.add_knowledge_article("Password Reset", "Steps to reset…")
        results = snow.search_knowledge("VPN")
        assert len(results) == 1
        assert results[0]["title"] == "VPN Setup"

    def test_incident_count(self) -> None:
        snow = ServiceNowConnector()
        snow.create_incident("A")
        snow.create_incident("B")
        assert snow.incident_count == 2


class TestScopeGuard:
    def test_allowed_scope(self) -> None:
        guard = ScopeGuard()
        guard.set_policy(ScopePolicy(
            agent_id="coder",
            connector_id="jira",
            allowed_scopes={ConnectorScope.JIRA_READ_ISSUES},
        ))
        assert guard.check("coder", "jira", ConnectorScope.JIRA_READ_ISSUES) is True

    def test_denied_scope(self) -> None:
        guard = ScopeGuard()
        guard.set_policy(ScopePolicy(
            agent_id="coder",
            connector_id="jira",
            allowed_scopes={ConnectorScope.JIRA_READ_ISSUES},
        ))
        assert guard.check("coder", "jira", ConnectorScope.JIRA_CREATE_ISSUES) is False
        assert guard.violation_count == 1

    def test_explicit_deny(self) -> None:
        guard = ScopeGuard()
        guard.set_policy(ScopePolicy(
            agent_id="coder",
            connector_id="jira",
            allowed_scopes={ConnectorScope.JIRA_READ_ISSUES, ConnectorScope.JIRA_CREATE_ISSUES},
            denied_scopes={ConnectorScope.JIRA_CREATE_ISSUES},
        ))
        assert guard.check("coder", "jira", ConnectorScope.JIRA_CREATE_ISSUES) is False

    def test_no_policy(self) -> None:
        guard = ScopeGuard()
        assert guard.check("unknown", "jira", ConnectorScope.JIRA_READ_ISSUES) is False
        assert guard.violations()[0]["type"] == "no_policy"

    def test_rate_limiting(self) -> None:
        guard = ScopeGuard()
        guard.set_policy(ScopePolicy(
            agent_id="coder",
            connector_id="jira",
            allowed_scopes={ConnectorScope.JIRA_READ_ISSUES},
            max_requests_per_minute=3,
        ))
        assert guard.check("coder", "jira", ConnectorScope.JIRA_READ_ISSUES) is True
        assert guard.check("coder", "jira", ConnectorScope.JIRA_READ_ISSUES) is True
        assert guard.check("coder", "jira", ConnectorScope.JIRA_READ_ISSUES) is True
        assert guard.check("coder", "jira", ConnectorScope.JIRA_READ_ISSUES) is False


class TestConnectorRegistry:
    def test_register_and_get(self) -> None:
        reg = ConnectorRegistry()
        reg.register(TeamsConnector())
        assert reg.get("teams-default") is not None

    def test_unregister(self) -> None:
        reg = ConnectorRegistry()
        reg.register(JiraConnector())
        assert reg.unregister("jira-default") is True
        assert reg.get("jira-default") is None

    def test_list_connectors(self) -> None:
        reg = ConnectorRegistry()
        reg.register(TeamsConnector())
        reg.register(JiraConnector())
        reg.register(ServiceNowConnector())
        assert reg.connector_count == 3
        connectors = reg.list_connectors()
        assert len(connectors) == 3

    def test_scope_enforcement(self) -> None:
        reg = ConnectorRegistry()
        reg.register(JiraConnector())
        reg.set_agent_policy(ScopePolicy(
            agent_id="coder", connector_id="jira-default",
            allowed_scopes={ConnectorScope.JIRA_READ_ISSUES},
        ))
        assert reg.check_access("coder", "jira-default", ConnectorScope.JIRA_READ_ISSUES) is True
        assert reg.check_access("coder", "jira-default", ConnectorScope.JIRA_CREATE_ISSUES) is False

    def test_stats(self) -> None:
        reg = ConnectorRegistry()
        reg.register(TeamsConnector())
        stats = reg.stats()
        assert stats["total_connectors"] == 1
        assert "scope_guard" in stats
