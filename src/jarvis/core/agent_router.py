"""Multi-Agent Router: Spezialisierte Agenten mit Intent-basiertem Routing.

Architektur:
  User-Nachricht → AgentRouter.route() → bester Agent
                 → Agent.system_prompt + Tool-Filter
                 → Planner arbeitet im Agenten-Kontext

Agenten sind konfigurierte Persona-Profile mit:
  - Eigenem System-Prompt (Persönlichkeit + Expertise)
  - Tool-Whitelist (nur erlaubte Tools)
  - Skill-Zuordnung (bestimmte Skills gehören zu bestimmten Agenten)
  - Modell-Präferenz (z.B. starkes Modell für Coding-Agent)
  - Trigger-Patterns für automatisches Routing

Eingebaute Agenten:
  - jarvis (default): Generalist, kann alles
  - researcher: Web-Recherche, Zusammenfassungen
  - coder: Programmierung, Shell-Befehle, Dateien
  - organizer: Kalender, Todos, E-Mails, Briefings

Bibel-Referenz: §9.2 (Multi-Agent-Routing)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from jarvis.core.bindings import BindingEngine, MessageBinding, MessageContext
from jarvis.utils.logging import get_logger

if TYPE_CHECKING:
    from jarvis.audit import AuditLogger

log = get_logger(__name__)


# ============================================================================
# Datenmodelle
# ============================================================================


@dataclass
class AgentProfile:
    """Definition eines spezialisierten Agenten.

    Jeder Agent hat:
      - Eigenes Workspace-Verzeichnis (isoliert von anderen Agenten)
      - Eigene Sandbox-Konfiguration (Netzwerk, Memory-Limits)
      - Eigene Tool-Rechte (Whitelist/Blacklist)
      - Delegations-Fähigkeit (kann andere Agenten beauftragen)
    """

    name: str
    display_name: str = ""
    description: str = ""

    # Persona
    system_prompt: str = ""
    language: str = "de"  # Antwortsprache

    # Routing
    trigger_patterns: list[str] = field(default_factory=list)
    trigger_keywords: list[str] = field(default_factory=list)
    priority: int = 0  # Höher = bevorzugt bei Gleichstand

    # Tool-Zugriff
    allowed_tools: list[str] | None = None  # None = alle Tools erlaubt
    blocked_tools: list[str] = field(default_factory=list)

    # Modell-Präferenz
    preferred_model: str = ""  # Leer = Default des ModelRouters
    temperature: float | None = None  # Leer = Default

    # --- NEU: Workspace-Isolation ---
    workspace_subdir: str = ""  # Unterverzeichnis in ~/.jarvis/workspace/
    # Leer = eigenes Verzeichnis basierend auf Agent-Name
    # Isoliert Dateien, Outputs und temporäre Daten pro Agent
    shared_workspace: bool = False  # True = teilt Workspace mit Default-Agent

    # --- NEU: Per-Agent Sandbox ---
    sandbox_network: str = "allow"  # "allow" oder "block"
    sandbox_max_memory_mb: int = 512
    sandbox_max_processes: int = 64
    sandbox_timeout: int = 30

    # --- NEU: Delegation ---
    can_delegate_to: list[str] = field(default_factory=list)
    # Liste von Agent-Namen an die delegiert werden darf
    # Leer = kann nicht delegieren
    max_delegation_depth: int = 2  # Maximale Delegationstiefe

    # --- NEU: Per-Agent Credentials ---
    credential_scope: str = ""
    # Scope-Name für Credential-Isolation.
    # Leer = Zugriff nur auf globale Credentials.
    # Gesetzt = Zugriff auf agent-spezifische + globale Credentials.
    # Beispiel: "coder" → kann auf "coder/github:token" zugreifen
    credential_mappings: dict[str, str] = field(default_factory=dict)
    # Mapping: param_name → "service:key" für automatische Injection
    # Beispiel: {"api_key": "openai:api_key"}

    # Status
    enabled: bool = True

    @property
    def has_tool_restrictions(self) -> bool:
        return self.allowed_tools is not None or len(self.blocked_tools) > 0

    @property
    def effective_workspace_subdir(self) -> str:
        """Effektives Workspace-Unterverzeichnis.

        Wenn shared_workspace=True, wird "" zurückgegeben (gemeinsamer Workspace).
        Sonst workspace_subdir oder Agent-Name als Fallback.
        """
        if self.shared_workspace:
            return ""
        return self.workspace_subdir or self.name

    def resolve_workspace(self, base_workspace: Path) -> Path:
        """Löst das Agent-spezifische Workspace-Verzeichnis auf.

        Args:
            base_workspace: Basis-Workspace (z.B. ~/.jarvis/workspace/)

        Returns:
            Isoliertes Verzeichnis für diesen Agenten.
        """
        subdir = self.effective_workspace_subdir
        if not subdir:
            return base_workspace

        agent_workspace = base_workspace / "agents" / subdir
        agent_workspace.mkdir(parents=True, exist_ok=True)
        return agent_workspace

    def get_sandbox_config(self) -> dict[str, Any]:
        """Gibt Sandbox-Konfiguration für diesen Agenten zurück."""
        return {
            "network": self.sandbox_network,
            "max_memory_mb": self.sandbox_max_memory_mb,
            "max_processes": self.sandbox_max_processes,
            "timeout": self.sandbox_timeout,
        }

    def filter_tools(self, all_tools: dict[str, Any]) -> dict[str, Any]:
        """Filtert Tool-Schemas basierend auf Agent-Rechten.

        Args:
            all_tools: Alle verfügbaren Tool-Schemas.

        Returns:
            Gefilterte Tool-Schemas die dieser Agent nutzen darf.
        """
        if not self.has_tool_restrictions:
            return all_tools

        filtered = {}
        for name, schema in all_tools.items():
            if name in self.blocked_tools:
                continue
            if self.allowed_tools is not None and name not in self.allowed_tools:
                continue
            filtered[name] = schema

        return filtered


@dataclass
class RouteDecision:
    """Ergebnis des Agent-Routings."""

    agent: AgentProfile
    confidence: float  # 0.0-1.0
    reason: str = ""
    matched_patterns: list[str] = field(default_factory=list)


@dataclass
class DelegationRequest:
    """Anfrage eines Agenten, eine Teilaufgabe an einen anderen zu delegieren.

    Ermöglicht Agent-zu-Agent-Kommunikation:
      Jarvis: "Recherchiere die aktuellen BU-Tarife"
        → DelegationRequest(from=jarvis, to=researcher, task="BU-Tarife")
        → Researcher führt aus, gibt Ergebnis an Jarvis zurück
    """

    from_agent: str
    to_agent: str
    task: str
    depth: int = 1  # Aktuelle Delegationstiefe
    target_profile: AgentProfile | None = None
    result: str | None = None  # Wird nach Ausführung gesetzt
    success: bool | None = None


# ============================================================================
# Eingebaute Agenten
# ============================================================================


def _default_agents() -> list[AgentProfile]:
    """Erstellt nur den minimalen Default-Agenten.

    Alle spezialisierten Agenten werden vom Nutzer definiert
    (via ~/.jarvis/config/agents.yaml oder Onboarding).
    Jarvis ist ein universelles Agent-OS -- keine hardcodierten
    Branchen- oder Rollen-Agenten.
    """
    return [
        AgentProfile(
            name="jarvis",
            display_name="Jarvis",
            description="Universeller Assistent -- passt sich dynamisch an den Nutzer an.",
            system_prompt=(
                "Du bist Jarvis, ein persönlicher KI-Assistent. "
                "Du passt dich an die Sprache und Bedürfnisse des Nutzers an. "
                "Du hast Zugriff auf verschiedene Tools und wählst den besten Ansatz."
            ),
            priority=0,
            shared_workspace=True,
            enabled=True,
        ),
    ]


# ============================================================================
# Agent Router
# ============================================================================


class AgentRouter:
    """Routet Nachrichten zum passendsten spezialisierten Agenten.

    Usage:
        router = AgentRouter()
        router.initialize()  # Lädt eingebaute + konfigurierte Agenten

        decision = router.route("Recherchiere zum Thema KI-Sicherheit")
        # → RouteDecision(agent=researcher, confidence=0.85)

        # Tool-Schemas für den Agenten filtern:
        filtered_tools = decision.agent.filter_tools(all_tool_schemas)
    """

    def __init__(self, audit_logger: AuditLogger | None = None) -> None:
        self._agents: dict[str, AgentProfile] = {}
        self._default_agent: str = "jarvis"
        self._compiled_patterns: dict[str, list[re.Pattern]] = {}
        self._binding_engine: BindingEngine = BindingEngine()
        self._audit_logger = audit_logger

    @property
    def bindings(self) -> BindingEngine:
        """Zugriff auf die Binding-Engine für deterministische Routing-Regeln."""
        return self._binding_engine

    def initialize(self, custom_agents: list[AgentProfile] | None = None) -> None:
        """Initialisiert den Router mit eingebauten + optionalen Custom-Agenten.

        Args:
            custom_agents: Zusätzliche Agenten die die Defaults ergänzen/überschreiben.
        """
        # Eingebaute Agenten laden
        for agent in _default_agents():
            self._agents[agent.name] = agent

        # Custom-Agenten überschreiben/ergänzen
        if custom_agents:
            for agent in custom_agents:
                self._agents[agent.name] = agent

        # Regex-Patterns kompilieren
        self._compile_patterns()

        log.info(
            "agent_router_initialized",
            agents=list(self._agents.keys()),
            default=self._default_agent,
        )

    def _compile_patterns(self) -> None:
        """Kompiliert Trigger-Patterns für schnelles Matching."""
        self._compiled_patterns.clear()
        for name, agent in self._agents.items():
            patterns = []
            for pattern_str in agent.trigger_patterns:
                try:
                    patterns.append(re.compile(pattern_str, re.IGNORECASE))
                except re.error as exc:
                    log.warning(
                        "agent_pattern_compile_error",
                        agent=name,
                        pattern=pattern_str,
                        error=str(exc),
                    )
            self._compiled_patterns[name] = patterns

    # ========================================================================
    # Routing
    # ========================================================================

    def route(
        self,
        query: str,
        *,
        context: MessageContext | None = None,
    ) -> RouteDecision:
        """Routet eine User-Nachricht zum besten Agenten.

        Routing-Kaskade (deterministic → probabilistic):
          1. Bindings (deterministisch, First-Match-Wins)
          2. Regex-Pattern-Match: 0.9
          3. Exakter Keyword im Query: 0.7
          4. Teilwort-Match: 0.4
          5. Priority-Bonus: +0.05 * priority
          6. Default (Jarvis): 0.3

        Args:
            query: User-Nachricht.
            context: Optionaler MessageContext für Binding-Auswertung.
                     Wenn None, wird ein minimaler Kontext aus query erstellt.

        Returns:
            RouteDecision mit Agent und Confidence.
        """
        if not query.strip():
            return self._default_decision("Leere Nachricht")

        # --- Phase 1: Deterministische Bindings ---
        if self._binding_engine.binding_count > 0:
            ctx = context or MessageContext(text=query)
            match = self._binding_engine.evaluate(ctx)

            if match and match.matched:
                target = self._agents.get(match.target_agent)
                if target and target.enabled:
                    return RouteDecision(
                        agent=target,
                        confidence=1.0,
                        reason=f"Binding: {match.binding.name}",
                        matched_patterns=[f"binding:{match.binding.name}"],
                    )
                else:
                    log.warning(
                        "binding_target_not_found",
                        binding=match.binding.name,
                        target=match.target_agent,
                    )

        # --- Phase 2: Probabilistisches Keyword/Pattern-Matching ---
        query_lower = query.lower()
        query_words = set(re.findall(r"\w+", query_lower))
        scores: dict[str, tuple[float, list[str]]] = {}

        for name, agent in self._agents.items():
            if not agent.enabled:
                continue

            score = 0.0
            matched: list[str] = []

            # 1. Regex-Pattern-Matches (stärkster Indikator)
            for pattern in self._compiled_patterns.get(name, []):
                if pattern.search(query_lower):
                    score = max(score, 0.9)
                    matched.append(f"pattern:{pattern.pattern}")

            # 2. Keyword-Matches
            for kw in agent.trigger_keywords:
                kw_lower = kw.lower()
                if kw_lower in query_lower:
                    score = max(score, 0.7)
                    matched.append(f"keyword:{kw}")
                elif kw_lower in query_words:
                    score = max(score, 0.5)
                    matched.append(f"word:{kw}")

            # 3. Priority-Bonus
            score += agent.priority * 0.05

            # Clamp
            score = min(score, 1.0)

            scores[name] = (score, matched)

        # Bester Agent auswählen
        if not scores:
            return self._default_decision("Keine Agenten aktiv")

        best_name = max(scores, key=lambda n: scores[n][0])
        best_score, best_matched = scores[best_name]

        # Minimum-Confidence: Wenn kein Agent gut matcht, Fallback
        if best_score < 0.3:
            return self._default_decision("Kein Agent passt gut genug")

        agent = self._agents[best_name]

        decision = RouteDecision(
            agent=agent,
            confidence=best_score,
            reason=f"Best match: {agent.display_name or agent.name}",
            matched_patterns=best_matched,
        )

        log.info(
            "agent_routed",
            agent=agent.name,
            confidence=round(best_score, 2),
            matched=best_matched[:3],
        )

        return decision

    def _default_decision(self, reason: str) -> RouteDecision:
        """Erstellt eine Default-Routing-Entscheidung (Jarvis)."""
        default = self._agents.get(self._default_agent)
        if not default:
            # Absoluter Fallback
            default = AgentProfile(name="jarvis", display_name="Jarvis")

        return RouteDecision(
            agent=default,
            confidence=0.3,
            reason=reason,
        )

    # ========================================================================
    # Zugriff & Verwaltung
    # ========================================================================

    def get_agent(self, name: str) -> AgentProfile | None:
        """Gibt einen Agenten per Name zurück."""
        return self._agents.get(name)

    def list_agents(self) -> list[AgentProfile]:
        """Alle registrierten Agenten."""
        return list(self._agents.values())

    def list_enabled(self) -> list[AgentProfile]:
        """Nur aktive Agenten."""
        return [a for a in self._agents.values() if a.enabled]

    def add_agent(self, agent: AgentProfile) -> None:
        """Registriert einen neuen Agenten (oder überschreibt bestehenden)."""
        self._agents[agent.name] = agent
        self._compile_patterns()
        log.info("agent_added", name=agent.name)

    def auto_create_agent(
        self,
        name: str,
        description: str,
        *,
        trigger_keywords: list[str] | None = None,
        system_prompt: str = "",
        allowed_tools: list[str] | None = None,
        sandbox_network: str = "allow",
        can_delegate_to: list[str] | None = None,
        persist_path: Path | None = None,
    ) -> AgentProfile:
        """Erstellt dynamisch einen neuen Agenten zur Laufzeit.

        Jarvis kann diese Methode selbst aufrufen, wenn es erkennt dass
        ein Spezialist gebraucht wird. Der Agent wird sofort aktiv und
        optional in agents.yaml persistiert.

        Args:
            name: Eindeutiger Agent-Name (z.B. "tarif_berater").
            description: Kurzbeschreibung der Rolle.
            trigger_keywords: Keywords für automatisches Routing.
            system_prompt: System-Prompt für den Agenten.
            allowed_tools: Tool-Whitelist (None = alle).
            sandbox_network: "allow" oder "block".
            can_delegate_to: Liste von Agent-Namen für Delegation.
            persist_path: Wenn gesetzt, wird agents.yaml aktualisiert.

        Returns:
            Das erstellte AgentProfile.
        """
        agent = AgentProfile(
            name=name,
            display_name=description,
            description=description,
            system_prompt=system_prompt,
            trigger_keywords=trigger_keywords or [],
            allowed_tools=allowed_tools,
            sandbox_network=sandbox_network,
            can_delegate_to=can_delegate_to or [],
        )

        self.add_agent(agent)

        log.info(
            "agent_auto_created",
            name=name,
            description=description[:100],
            keywords=trigger_keywords,
        )

        # Optional persistieren
        if persist_path:
            self.save_agents_yaml(persist_path)

        return agent

    def save_agents_yaml(self, path: Path) -> None:
        """Speichert die aktuelle Agent-Konfiguration als YAML.

        Ermöglicht Persistenz von dynamisch erstellten Agenten.
        """
        import yaml

        agents_data = []
        for agent in self._agents.values():
            if agent.name == "jarvis":
                continue  # Default nicht speichern

            data: dict[str, Any] = {
                "name": agent.name,
                "display_name": agent.display_name,
                "description": agent.description,
            }

            if agent.system_prompt:
                data["system_prompt"] = agent.system_prompt
            if agent.trigger_keywords:
                data["trigger_keywords"] = agent.trigger_keywords
            if agent.trigger_patterns:
                data["trigger_patterns"] = agent.trigger_patterns
            if agent.allowed_tools is not None:
                data["allowed_tools"] = agent.allowed_tools
            if agent.blocked_tools:
                data["blocked_tools"] = agent.blocked_tools
            if agent.sandbox_network != "allow":
                data["sandbox_network"] = agent.sandbox_network
            if agent.sandbox_max_memory_mb != 512:
                data["sandbox_max_memory_mb"] = agent.sandbox_max_memory_mb
            if agent.sandbox_timeout != 30:
                data["sandbox_timeout"] = agent.sandbox_timeout
            if agent.can_delegate_to:
                data["can_delegate_to"] = agent.can_delegate_to
            if agent.workspace_subdir:
                data["workspace_subdir"] = agent.workspace_subdir
            if agent.shared_workspace:
                data["shared_workspace"] = True

            agents_data.append(data)

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.dump({"agents": agents_data}, default_flow_style=False, allow_unicode=True),
            encoding="utf-8",
        )
        log.info("agents_yaml_saved", path=str(path), count=len(agents_data))

    def remove_agent(self, name: str) -> bool:
        """Entfernt einen Agenten."""
        if name == self._default_agent:
            return False  # Default kann nicht entfernt werden
        if name in self._agents:
            del self._agents[name]
            self._compiled_patterns.pop(name, None)
            return True
        return False

    @classmethod
    def from_yaml(
        cls,
        config_path: str | Path,
        audit_logger: AuditLogger | None = None,
    ) -> "AgentRouter":
        """Lädt Agenten- und Binding-Konfiguration aus YAML-Datei(en).

        Erwartetes Format (agents.yaml):
            agents:
              - name: insurance_expert
                display_name: Versicherungs-Experte
                system_prompt: "Du bist ein Versicherungsexperte..."
                trigger_keywords: [versicherung, police, tarif]
                allowed_tools: [web_search, read_file]

        Bindings werden aus bindings.yaml im selben Verzeichnis geladen
        (falls vorhanden).
        """
        path = Path(config_path)
        router = cls(audit_logger=audit_logger)

        custom_agents = []
        if path.exists():
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
                for agent_data in data.get("agents", []):
                    custom_agents.append(AgentProfile(**agent_data))
            except Exception as exc:
                log.warning("agent_config_load_error", path=str(path), error=str(exc))

        router.initialize(custom_agents)

        # Bindings aus separater Datei laden (falls vorhanden)
        bindings_path = path.parent / "bindings.yaml"
        if bindings_path.exists():
            router._binding_engine = BindingEngine.from_yaml(bindings_path)
            log.info(
                "bindings_loaded_from_yaml",
                path=str(bindings_path),
                count=router._binding_engine.binding_count,
            )

        return router

    # ========================================================================
    # Agent-Delegation
    # ========================================================================

    def can_delegate(self, from_agent: str, to_agent: str) -> bool:
        """Prüft ob ein Agent an einen anderen delegieren darf."""
        source = self._agents.get(from_agent)
        target = self._agents.get(to_agent)

        if not source or not target:
            return False
        if not target.enabled:
            return False
        if to_agent not in source.can_delegate_to:
            return False

        return True

    def create_delegation(
        self,
        from_agent: str,
        to_agent: str,
        task: str,
        *,
        depth: int = 0,
    ) -> DelegationRequest | None:
        """Erstellt eine Delegationsanfrage.

        Args:
            from_agent: Name des delegierenden Agenten.
            to_agent: Name des Ziel-Agenten.
            task: Aufgabenbeschreibung.
            depth: Aktuelle Delegationstiefe (für Rekursionsschutz).

        Returns:
            DelegationRequest oder None wenn nicht erlaubt.
        """
        source = self._agents.get(from_agent)
        target = self._agents.get(to_agent)

        if not source or not target:
            log.warning("delegation_agents_not_found", from_=from_agent, to=to_agent)
            return None

        if not self.can_delegate(from_agent, to_agent):
            log.warning(
                "delegation_not_allowed",
                from_=from_agent,
                to=to_agent,
                allowed=source.can_delegate_to,
            )
            return None

        if depth >= source.max_delegation_depth:
            log.warning(
                "delegation_depth_exceeded",
                from_=from_agent,
                to=to_agent,
                depth=depth,
                max_depth=source.max_delegation_depth,
            )
            return None

        request = DelegationRequest(
            from_agent=from_agent,
            to_agent=to_agent,
            task=task,
            depth=depth + 1,
            target_profile=target,
        )

        # Audit: Delegation protokollieren
        if self._audit_logger:
            self._audit_logger.log_agent_delegation(from_agent, to_agent, task)

        log.info(
            "delegation_created",
            from_=from_agent,
            to=to_agent,
            task=task[:100],
            depth=depth + 1,
        )

        return request

    def get_delegation_targets(self, agent_name: str) -> list[AgentProfile]:
        """Gibt die Agenten zurück, an die delegiert werden darf."""
        source = self._agents.get(agent_name)
        if not source:
            return []

        targets = []
        for name in source.can_delegate_to:
            target = self._agents.get(name)
            if target and target.enabled:
                targets.append(target)
        return targets

    # ========================================================================
    # Workspace-Verwaltung
    # ========================================================================

    def resolve_agent_workspace(
        self,
        agent_name: str,
        base_workspace: Path,
    ) -> Path:
        """Löst das Workspace-Verzeichnis für einen Agenten auf.

        Args:
            agent_name: Name des Agenten.
            base_workspace: Basis-Workspace.

        Returns:
            Isoliertes Verzeichnis oder Basis bei shared_workspace.
        """
        agent = self._agents.get(agent_name)
        if not agent:
            return base_workspace

        return agent.resolve_workspace(base_workspace)

    def stats(self) -> dict[str, Any]:
        """Router-Statistiken."""
        return {
            "total_agents": len(self._agents),
            "enabled": len(self.list_enabled()),
            "default": self._default_agent,
            "bindings": self._binding_engine.stats(),
            "agents": {
                name: {
                    "display_name": a.display_name,
                    "keywords": len(a.trigger_keywords),
                    "patterns": len(a.trigger_patterns),
                    "has_tool_restrictions": a.has_tool_restrictions,
                }
                for name, a in self._agents.items()
            },
        }
