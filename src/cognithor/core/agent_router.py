"""Multi-Agent Router: Specialized agents with intent-based routing.

Architecture:
  User message -> AgentRouter.route() -> best agent
               -> Agent.system_prompt + tool filter
               -> Planner operates in agent context

Agents are configured persona profiles with:
  - Own system prompt (personality + expertise)
  - Tool whitelist (only allowed tools)
  - Skill assignment (certain skills belong to certain agents)
  - Model preference (e.g. strong model for coding agent)
  - Trigger patterns for automatic routing

Built-in agents:
  - jarvis (default): Generalist, can do everything
  - researcher: Web research, summaries
  - coder: Programming, shell commands, files
  - organizer: Calendar, todos, emails, briefings

Reference: §9.2 (Multi-Agent Routing)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml  # type: ignore[import-untyped]

from cognithor.core.bindings import BindingEngine, MessageContext
from cognithor.utils.logging import get_logger

if TYPE_CHECKING:
    from cognithor.audit import AuditLogger

log = get_logger(__name__)


# ============================================================================
# Data models
# ============================================================================


@dataclass
class AgentProfile:
    """Definition of a specialized agent.

    Each agent has:
      - Own workspace directory (isolated from other agents)
      - Own sandbox configuration (network, memory limits)
      - Own tool permissions (whitelist/blacklist)
      - Delegation capability (can assign tasks to other agents)
      - Role (orchestrator, worker, monitor)
    """

    name: str
    display_name: str = ""
    description: str = ""

    # Role (v0.36.0): orchestrator | worker | monitor
    role: str = "worker"  # Default "worker" for backward compat

    # Persona
    system_prompt: str = ""
    language: str = "de"  # Response language

    # Routing
    trigger_patterns: list[str] = field(default_factory=list)
    trigger_keywords: list[str] = field(default_factory=list)
    priority: int = 0  # Higher = preferred on tie

    # Tool access
    allowed_tools: list[str] | None = None  # None = all tools allowed
    blocked_tools: list[str] = field(default_factory=list)

    # Model preference
    preferred_model: str = ""  # Empty = ModelRouter default
    temperature: float | None = None  # Empty = default
    top_p: float | None = None  # None = model default

    # --- Workspace isolation ---
    workspace_subdir: str = ""  # Subdirectory in ~/.cognithor/workspace/
    # Empty = own directory based on agent name
    # Isolates files, outputs and temporary data per agent
    shared_workspace: bool = False  # True = shares workspace with default agent

    # --- Per-agent sandbox ---
    sandbox_network: str = "allow"  # "allow" or "block"
    sandbox_max_memory_mb: int = 512
    sandbox_max_processes: int = 64
    sandbox_timeout: int = 30

    # --- Delegation ---
    can_delegate_to: list[str] = field(default_factory=list)
    # List of agent names that can be delegated to
    # Empty = cannot delegate
    max_delegation_depth: int = 2  # Maximum delegation depth

    # --- Per-agent credentials ---
    credential_scope: str = ""
    # Scope name for credential isolation.
    # Empty = access only to global credentials.
    # Set = access to agent-specific + global credentials.
    # Example: "coder" -> can access "coder/github:token"
    credential_mappings: dict[str, str] = field(default_factory=dict)
    # Mapping: param_name -> "service:key" for automatic injection
    # Example: {"api_key": "openai:api_key"}

    # --- Identity Layer (Immortal Mind Protocol) ---
    identity_enabled: bool = True  # Does this agent have a personality?
    identity_id: str | None = None  # Custom identity ID (None = auto from agent name)

    # Status
    enabled: bool = True

    @property
    def has_tool_restrictions(self) -> bool:
        return self.allowed_tools is not None or len(self.blocked_tools) > 0

    @property
    def effective_workspace_subdir(self) -> str:
        """Effective workspace subdirectory.

        If shared_workspace=True, returns "" (shared workspace).
        Otherwise workspace_subdir or agent name as fallback.
        """
        if self.shared_workspace:
            return ""
        return self.workspace_subdir or self.name

    def resolve_workspace(self, base_workspace: Path) -> Path:
        """Resolve the agent-specific workspace directory.

        Args:
            base_workspace: Base workspace (e.g. ~/.cognithor/workspace/)

        Returns:
            Isolated directory for this agent.
        """
        subdir = self.effective_workspace_subdir
        if not subdir:
            return base_workspace

        agent_workspace = base_workspace / "agents" / subdir
        agent_workspace.mkdir(parents=True, exist_ok=True)
        return agent_workspace

    def get_sandbox_config(self) -> dict[str, Any]:
        """Return sandbox configuration for this agent."""
        return {
            "network": self.sandbox_network,
            "max_memory_mb": self.sandbox_max_memory_mb,
            "max_processes": self.sandbox_max_processes,
            "timeout": self.sandbox_timeout,
        }

    def filter_tools(self, all_tools: dict[str, Any]) -> dict[str, Any]:
        """Filter tool schemas based on agent permissions.

        Args:
            all_tools: All available tool schemas.

        Returns:
            Filtered tool schemas that this agent is allowed to use.
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
    """Result of agent routing."""

    agent: AgentProfile
    confidence: float  # 0.0-1.0
    reason: str = ""
    matched_patterns: list[str] = field(default_factory=list)


@dataclass
class DelegationRequest:
    """Request from an agent to delegate a subtask to another agent.

    Enables agent-to-agent communication:
      Jarvis: "Research current disability insurance rates"
        -> DelegationRequest(from=jarvis, to=researcher, task="rates")
        -> Researcher executes, returns result to Jarvis
    """

    from_agent: str
    to_agent: str
    task: str
    depth: int = 1  # Current delegation depth
    target_profile: AgentProfile | None = None
    result: str | None = None  # Set after execution
    success: bool | None = None


# ============================================================================
# Built-in agents
# ============================================================================


def _default_agents() -> list[AgentProfile]:
    """Create only the minimal default agent.

    All specialized agents are defined by the user
    (via ~/.cognithor/config/agents.yaml or onboarding).
    Jarvis is a universal agent OS -- no hardcoded
    industry or role agents.
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
    """Route messages to the best matching specialized agent.

    Usage:
        router = AgentRouter()
        router.initialize()  # Load built-in + configured agents

        decision = router.route("Research the topic of AI safety")
        # -> RouteDecision(agent=researcher, confidence=0.85)

        # Filter tool schemas for the agent:
        filtered_tools = decision.agent.filter_tools(all_tool_schemas)
    """

    def __init__(self, audit_logger: AuditLogger | None = None) -> None:
        self._agents: dict[str, AgentProfile] = {}
        self._default_agent: str = "jarvis"
        self._compiled_patterns: dict[str, list[re.Pattern[str]]] = {}
        self._binding_engine: BindingEngine = BindingEngine()
        self._audit_logger = audit_logger

    @property
    def bindings(self) -> BindingEngine:
        """Access the binding engine for deterministic routing rules."""
        return self._binding_engine

    def initialize(self, custom_agents: list[AgentProfile] | None = None) -> None:
        """Initialize the router with built-in + optional custom agents.

        Args:
            custom_agents: Additional agents that supplement/override defaults.
        """
        # Load built-in agents
        for agent in _default_agents():
            self._agents[agent.name] = agent

        # Custom agents override/supplement
        if custom_agents:
            for agent in custom_agents:
                self._agents[agent.name] = agent

        # Compile regex patterns
        self._compile_patterns()

        log.info(
            "agent_router_initialized",
            agents=list(self._agents.keys()),
            default=self._default_agent,
        )

    def _compile_patterns(self) -> None:
        """Compile trigger patterns for fast matching."""
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
        """Route a user message to the best agent.

        Routing cascade (deterministic -> probabilistic):
          1. Bindings (deterministic, first-match-wins)
          2. Regex pattern match: 0.9
          3. Exact keyword in query: 0.7
          4. Partial word match: 0.4
          5. Priority bonus: +0.05 * priority
          6. Default (Jarvis): 0.3

        Args:
            query: User message.
            context: Optional MessageContext for binding evaluation.
                     If None, a minimal context is created from query.

        Returns:
            RouteDecision with agent and confidence.
        """
        if not query.strip():
            return self._default_decision("Empty message")

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

            # 1. Regex pattern matches (strongest indicator)
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

        # Select best agent
        if not scores:
            return self._default_decision("No agents active")

        best_name = max(scores, key=lambda n: scores[n][0])
        best_score, best_matched = scores[best_name]

        # Minimum confidence: if no agent matches well, fallback
        if best_score < 0.3:
            return self._default_decision("No agent matches well enough")

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
        """Create a default routing decision (Jarvis)."""
        default = self._agents.get(self._default_agent)
        if not default:
            # Absolute fallback
            default = AgentProfile(name="jarvis", display_name="Jarvis")

        return RouteDecision(
            agent=default,
            confidence=0.3,
            reason=reason,
        )

    # ========================================================================
    # Access & management
    # ========================================================================

    def get_agent(self, name: str) -> AgentProfile | None:
        """Return an agent by name."""
        return self._agents.get(name)

    def list_agents(self) -> list[AgentProfile]:
        """All registered agents."""
        return list(self._agents.values())

    def list_enabled(self) -> list[AgentProfile]:
        """Only active agents."""
        return [a for a in self._agents.values() if a.enabled]

    def add_agent(self, agent: AgentProfile) -> None:
        """Register a new agent (or overwrite an existing one)."""
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
        """Dynamically create a new agent at runtime.

        Jarvis can call this method itself when it recognizes that
        a specialist is needed. The agent becomes active immediately and
        is optionally persisted in agents.yaml.

        Args:
            name: Unique agent name (e.g. "tarif_berater").
            description: Short description of the role.
            trigger_keywords: Keywords for automatic routing.
            system_prompt: System prompt for the agent.
            allowed_tools: Tool whitelist (None = all).
            sandbox_network: "allow" or "block".
            can_delegate_to: List of agent names for delegation.
            persist_path: If set, agents.yaml is updated.

        Returns:
            The created AgentProfile.
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

        # Optionally persist
        if persist_path:
            self.save_agents_yaml(persist_path)

        return agent

    def save_agents_yaml(self, path: Path) -> None:
        """Save the current agent configuration as YAML.

        Enables persistence of dynamically created agents.
        """
        import yaml

        agents_data = []
        for agent in self._agents.values():
            if agent.name == "jarvis":
                continue  # Don't save default

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

    def reload_from_yaml(self, yaml_path: Path) -> int:
        """Reload agents from YAML file without creating new instance.

        Clears current agents, re-initializes defaults, then loads
        custom agents from the given YAML file.

        Args:
            yaml_path: Path to agents.yaml file.

        Returns:
            Number of custom agents loaded from YAML.
        """
        if not yaml_path.exists():
            return 0

        custom_agents: list[AgentProfile] = []
        try:
            raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
            agents_list = raw.get("agents", [])
            for agent_data in agents_list:
                custom_agents.append(AgentProfile(**agent_data))
        except Exception as exc:
            log.warning(
                "reload_from_yaml_error",
                path=str(yaml_path),
                error=str(exc),
            )
            return 0

        # Clear current agents and re-initialize with defaults + custom
        self._agents.clear()
        self.initialize(custom_agents)

        log.info(
            "agents_reloaded_from_yaml",
            path=str(yaml_path),
            custom_count=len(custom_agents),
            total=len(self._agents),
        )
        return len(custom_agents)

    def remove_agent(self, name: str) -> bool:
        """Remove an agent."""
        if name == self._default_agent:
            return False  # Default cannot be removed
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
    ) -> AgentRouter:
        """Load agent and binding configuration from YAML file(s).

        Expected format (agents.yaml):
            agents:
              - name: insurance_expert
                display_name: Versicherungs-Experte
                system_prompt: "Du bist ein Versicherungsexperte..."
                trigger_keywords: [versicherung, police, tarif]
                allowed_tools: [web_search, read_file]

        Bindings are loaded from bindings.yaml in the same directory
        (if present).
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

        # Load bindings from separate file (if present)
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
    # Agent delegation
    # ========================================================================

    def can_delegate(self, from_agent: str, to_agent: str) -> bool:
        """Check whether an agent is allowed to delegate to another."""
        source = self._agents.get(from_agent)
        target = self._agents.get(to_agent)

        if not source or not target:
            return False
        if not target.enabled:
            return False
        return to_agent in source.can_delegate_to

    def create_delegation(
        self,
        from_agent: str,
        to_agent: str,
        task: str,
        *,
        depth: int = 0,
    ) -> DelegationRequest | None:
        """Create a delegation request.

        Args:
            from_agent: Name of the delegating agent.
            to_agent: Name of the target agent.
            task: Task description.
            depth: Current delegation depth (for recursion protection).

        Returns:
            DelegationRequest or None if not allowed.
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

        # Audit: log delegation
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
        """Return the agents that can be delegated to."""
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
    # Workspace management
    # ========================================================================

    def resolve_agent_workspace(
        self,
        agent_name: str,
        base_workspace: Path,
    ) -> Path:
        """Resolve the workspace directory for an agent.

        Args:
            agent_name: Name of the agent.
            base_workspace: Base workspace.

        Returns:
            Isolated directory or base for shared_workspace.
        """
        agent = self._agents.get(agent_name)
        if not agent:
            return base_workspace

        return agent.resolve_workspace(base_workspace)

    def stats(self) -> dict[str, Any]:
        """Router statistics."""
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
