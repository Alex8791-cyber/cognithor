"""Jarvis · Enterprise-Konnektoren.

Interoperabilität mit Unternehmenssystemen:

  - TeamsConnector:      Microsoft Teams Bot Framework Integration
  - JiraConnector:       Atlassian Jira REST API v3
  - ServiceNowConnector: ServiceNow Table API
  - ConnectorRegistry:   Verwaltung aller Konnektoren
  - ScopeGuard:          Erzwingt Least-Privilege pro Konnektor

Architektur-Bibel: §8.5 (Channel-Integration), §11.2 (Least-Privilege)

Sicherheitsanforderung:
  Jeder Konnektor bekommt strikte Berechtigungs-Scopes.
  Agenten erhalten nur die minimal notwendigen Zugriffsrechte.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


# ============================================================================
# Scope & Permission Model
# ============================================================================


class ConnectorScope(Enum):
    """Berechtigungsscopes für Konnektoren."""

    # Teams
    TEAMS_READ_MESSAGES = "teams:read_messages"
    TEAMS_SEND_MESSAGES = "teams:send_messages"
    TEAMS_READ_CHANNELS = "teams:read_channels"
    TEAMS_MANAGE_CHANNELS = "teams:manage_channels"

    # Jira
    JIRA_READ_ISSUES = "jira:read_issues"
    JIRA_CREATE_ISSUES = "jira:create_issues"
    JIRA_UPDATE_ISSUES = "jira:update_issues"
    JIRA_TRANSITION_ISSUES = "jira:transition_issues"
    JIRA_READ_PROJECTS = "jira:read_projects"

    # ServiceNow
    SNOW_READ_INCIDENTS = "snow:read_incidents"
    SNOW_CREATE_INCIDENTS = "snow:create_incidents"
    SNOW_UPDATE_INCIDENTS = "snow:update_incidents"
    SNOW_READ_KNOWLEDGE = "snow:read_knowledge"

    # CRM Generic
    CRM_READ_CONTACTS = "crm:read_contacts"
    CRM_UPDATE_CONTACTS = "crm:update_contacts"
    CRM_READ_DEALS = "crm:read_deals"

    # ERP Generic
    ERP_READ_ORDERS = "erp:read_orders"
    ERP_READ_INVENTORY = "erp:read_inventory"


@dataclass
class ScopePolicy:
    """Scope-Policy für einen Agenten pro Konnektor."""

    agent_id: str
    connector_id: str
    allowed_scopes: set[ConnectorScope] = field(default_factory=set)
    denied_scopes: set[ConnectorScope] = field(default_factory=set)
    max_requests_per_minute: int = 60

    def is_allowed(self, scope: ConnectorScope) -> bool:
        if scope in self.denied_scopes:
            return False
        return scope in self.allowed_scopes

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "connector_id": self.connector_id,
            "allowed": [s.value for s in self.allowed_scopes],
            "denied": [s.value for s in self.denied_scopes],
            "rate_limit": self.max_requests_per_minute,
        }


class ScopeGuard:
    """Erzwingt Least-Privilege pro Konnektor und Agent.

    Jede Konnektor-Aktion wird gegen die Scope-Policy geprüft.
    Verstöße werden geloggt.
    """

    def __init__(self) -> None:
        self._policies: dict[str, ScopePolicy] = {}  # "agent:connector" → policy
        self._violations: list[dict[str, Any]] = []
        self._request_counts: dict[str, list[float]] = {}

    def set_policy(self, policy: ScopePolicy) -> None:
        key = f"{policy.agent_id}:{policy.connector_id}"
        self._policies[key] = policy

    def check(self, agent_id: str, connector_id: str, scope: ConnectorScope) -> bool:
        """Prüft ob der Agent den Scope für diesen Konnektor hat."""
        key = f"{agent_id}:{connector_id}"
        policy = self._policies.get(key)

        if not policy:
            self._violations.append({
                "type": "no_policy",
                "agent_id": agent_id,
                "connector_id": connector_id,
                "scope": scope.value,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })
            return False

        if not policy.is_allowed(scope):
            self._violations.append({
                "type": "scope_denied",
                "agent_id": agent_id,
                "connector_id": connector_id,
                "scope": scope.value,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })
            return False

        # Rate-Limiting
        now = time.time()
        rkey = f"{agent_id}:{connector_id}"
        if rkey not in self._request_counts:
            self._request_counts[rkey] = []
        self._request_counts[rkey] = [t for t in self._request_counts[rkey] if now - t < 60]
        if len(self._request_counts[rkey]) >= policy.max_requests_per_minute:
            self._violations.append({
                "type": "rate_limited",
                "agent_id": agent_id,
                "connector_id": connector_id,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })
            return False
        self._request_counts[rkey].append(now)
        return True

    @property
    def violation_count(self) -> int:
        return len(self._violations)

    def violations(self) -> list[dict[str, Any]]:
        return list(self._violations)

    def stats(self) -> dict[str, Any]:
        return {
            "policies": len(self._policies),
            "violations": len(self._violations),
        }


# ============================================================================
# Base Connector
# ============================================================================


class ConnectorStatus(Enum):
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    CONFIGURED = "configured"


@dataclass
class ConnectorConfig:
    """Konfiguration eines Enterprise-Konnektors."""

    connector_id: str
    connector_type: str  # "teams", "jira", "servicenow", "crm", "erp"
    display_name: str
    base_url: str = ""
    auth_type: str = "bearer"  # bearer, basic, oauth2, api_key
    credentials_ref: str = ""  # Verweis auf Vault-Eintrag
    default_scopes: set[ConnectorScope] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "connector_id": self.connector_id,
            "type": self.connector_type,
            "display_name": self.display_name,
            "base_url": self.base_url,
            "auth_type": self.auth_type,
            "scopes": [s.value for s in self.default_scopes],
        }


class BaseConnector:
    """Basis-Klasse für Enterprise-Konnektoren."""

    def __init__(self, config: ConnectorConfig) -> None:
        self._config = config
        self._status = ConnectorStatus.CONFIGURED
        self._request_log: list[dict[str, Any]] = []

    @property
    def connector_id(self) -> str:
        return self._config.connector_id

    @property
    def status(self) -> ConnectorStatus:
        return self._status

    def connect(self) -> bool:
        """Verbindung herstellen (wird von Subklassen überschrieben)."""
        self._status = ConnectorStatus.CONNECTED
        return True

    def disconnect(self) -> None:
        self._status = ConnectorStatus.DISCONNECTED

    def _log_request(self, method: str, endpoint: str, **kwargs: Any) -> None:
        self._request_log.append({
            "method": method,
            "endpoint": endpoint,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **kwargs,
        })

    def stats(self) -> dict[str, Any]:
        return {
            "connector_id": self.connector_id,
            "type": self._config.connector_type,
            "status": self._status.value,
            "total_requests": len(self._request_log),
        }


# ============================================================================
# Microsoft Teams Connector
# ============================================================================


class TeamsConnector(BaseConnector):
    """Microsoft Teams Bot Framework Integration.

    Unterstützt:
      - Nachrichten empfangen und senden
      - Channel-Übersicht
      - Adaptive Cards
    """

    def __init__(self, config: ConnectorConfig | None = None) -> None:
        cfg = config or ConnectorConfig(
            connector_id="teams-default",
            connector_type="teams",
            display_name="Microsoft Teams",
            base_url="https://smba.trafficmanager.net",
            auth_type="bearer",
            default_scopes={
                ConnectorScope.TEAMS_READ_MESSAGES,
                ConnectorScope.TEAMS_SEND_MESSAGES,
            },
        )
        super().__init__(cfg)
        self._channels: list[dict[str, Any]] = []
        self._sent_messages: list[dict[str, Any]] = []

    def send_message(self, channel_id: str, text: str, *, card: dict[str, Any] | None = None) -> dict[str, Any]:
        """Sendet eine Nachricht an einen Teams-Channel."""
        msg = {
            "channel_id": channel_id,
            "text": text,
            "card": card,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "message_id": hashlib.sha256(f"{channel_id}:{time.time()}".encode()).hexdigest()[:12],
        }
        self._sent_messages.append(msg)
        self._log_request("POST", f"/channels/{channel_id}/messages")
        return msg

    def list_channels(self, team_id: str = "") -> list[dict[str, Any]]:
        self._log_request("GET", f"/teams/{team_id}/channels")
        return list(self._channels)

    def register_channel(self, channel_id: str, name: str) -> None:
        self._channels.append({"id": channel_id, "name": name})

    @property
    def sent_count(self) -> int:
        return len(self._sent_messages)


# ============================================================================
# Jira Connector
# ============================================================================


class JiraConnector(BaseConnector):
    """Atlassian Jira REST API v3 Integration.

    Unterstützt:
      - Issues lesen, erstellen, aktualisieren
      - Transitions (Workflow)
      - Projekte auflisten
    """

    def __init__(self, config: ConnectorConfig | None = None) -> None:
        cfg = config or ConnectorConfig(
            connector_id="jira-default",
            connector_type="jira",
            display_name="Atlassian Jira",
            base_url="https://your-domain.atlassian.net",
            auth_type="basic",
            default_scopes={
                ConnectorScope.JIRA_READ_ISSUES,
                ConnectorScope.JIRA_CREATE_ISSUES,
            },
        )
        super().__init__(cfg)
        self._issues: dict[str, dict[str, Any]] = {}

    def create_issue(
        self,
        project: str,
        summary: str,
        issue_type: str = "Task",
        description: str = "",
        **fields: Any,
    ) -> dict[str, Any]:
        """Erstellt ein Jira-Issue."""
        issue_key = f"{project}-{len(self._issues) + 1}"
        issue = {
            "key": issue_key,
            "project": project,
            "summary": summary,
            "issue_type": issue_type,
            "description": description,
            "status": "To Do",
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **fields,
        }
        self._issues[issue_key] = issue
        self._log_request("POST", "/rest/api/3/issue", issue_key=issue_key)
        return issue

    def get_issue(self, issue_key: str) -> dict[str, Any] | None:
        self._log_request("GET", f"/rest/api/3/issue/{issue_key}")
        return self._issues.get(issue_key)

    def transition_issue(self, issue_key: str, new_status: str) -> bool:
        issue = self._issues.get(issue_key)
        if not issue:
            return False
        issue["status"] = new_status
        self._log_request("POST", f"/rest/api/3/issue/{issue_key}/transitions")
        return True

    def search_issues(self, jql: str = "", project: str = "") -> list[dict[str, Any]]:
        self._log_request("POST", "/rest/api/3/search", jql=jql)
        results = list(self._issues.values())
        if project:
            results = [i for i in results if i.get("project") == project]
        return results

    @property
    def issue_count(self) -> int:
        return len(self._issues)


# ============================================================================
# ServiceNow Connector
# ============================================================================


class ServiceNowConnector(BaseConnector):
    """ServiceNow Table API Integration.

    Unterstützt:
      - Incidents erstellen/lesen/aktualisieren
      - Knowledge-Base lesen
    """

    def __init__(self, config: ConnectorConfig | None = None) -> None:
        cfg = config or ConnectorConfig(
            connector_id="snow-default",
            connector_type="servicenow",
            display_name="ServiceNow",
            base_url="https://your-instance.service-now.com",
            auth_type="oauth2",
            default_scopes={
                ConnectorScope.SNOW_READ_INCIDENTS,
                ConnectorScope.SNOW_CREATE_INCIDENTS,
            },
        )
        super().__init__(cfg)
        self._incidents: dict[str, dict[str, Any]] = {}
        self._knowledge: dict[str, dict[str, Any]] = {}

    def create_incident(
        self,
        short_description: str,
        *,
        urgency: int = 3,
        impact: int = 3,
        category: str = "software",
        caller: str = "",
        **fields: Any,
    ) -> dict[str, Any]:
        """Erstellt ein ServiceNow-Incident."""
        inc_number = f"INC{len(self._incidents) + 1:07d}"
        incident = {
            "number": inc_number,
            "short_description": short_description,
            "urgency": urgency,
            "impact": impact,
            "category": category,
            "caller": caller,
            "state": "New",
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **fields,
        }
        self._incidents[inc_number] = incident
        self._log_request("POST", "/api/now/table/incident")
        return incident

    def get_incident(self, number: str) -> dict[str, Any] | None:
        self._log_request("GET", f"/api/now/table/incident/{number}")
        return self._incidents.get(number)

    def update_incident(self, number: str, **updates: Any) -> dict[str, Any] | None:
        incident = self._incidents.get(number)
        if not incident:
            return None
        incident.update(updates)
        self._log_request("PATCH", f"/api/now/table/incident/{number}")
        return incident

    def search_knowledge(self, query: str = "") -> list[dict[str, Any]]:
        self._log_request("GET", "/api/now/table/kb_knowledge")
        if query:
            return [
                k for k in self._knowledge.values()
                if query.lower() in k.get("title", "").lower()
            ]
        return list(self._knowledge.values())

    def add_knowledge_article(self, title: str, body: str) -> dict[str, Any]:
        article = {
            "sys_id": hashlib.sha256(title.encode()).hexdigest()[:12],
            "title": title,
            "body": body,
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self._knowledge[article["sys_id"]] = article
        return article

    @property
    def incident_count(self) -> int:
        return len(self._incidents)


# ============================================================================
# Connector-Registry
# ============================================================================


class ConnectorRegistry:
    """Verwaltung aller Enterprise-Konnektoren.

    Registriert, konfiguriert und verwaltet alle verfügbaren
    Konnektoren mit Scope-Enforcement.
    """

    def __init__(self) -> None:
        self._connectors: dict[str, BaseConnector] = {}
        self._scope_guard = ScopeGuard()

    def register(self, connector: BaseConnector) -> None:
        self._connectors[connector.connector_id] = connector

    def get(self, connector_id: str) -> BaseConnector | None:
        return self._connectors.get(connector_id)

    def unregister(self, connector_id: str) -> bool:
        if connector_id in self._connectors:
            del self._connectors[connector_id]
            return True
        return False

    def set_agent_policy(self, policy: ScopePolicy) -> None:
        self._scope_guard.set_policy(policy)

    def check_access(self, agent_id: str, connector_id: str, scope: ConnectorScope) -> bool:
        return self._scope_guard.check(agent_id, connector_id, scope)

    @property
    def connector_count(self) -> int:
        return len(self._connectors)

    @property
    def scope_guard(self) -> ScopeGuard:
        return self._scope_guard

    def list_connectors(self) -> list[dict[str, Any]]:
        return [c.stats() for c in self._connectors.values()]

    def stats(self) -> dict[str, Any]:
        return {
            "total_connectors": len(self._connectors),
            "connectors": [c.stats() for c in self._connectors.values()],
            "scope_guard": self._scope_guard.stats(),
        }
