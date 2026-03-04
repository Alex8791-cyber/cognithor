"""Multi-Agent Collaboration — structured cooperation patterns.

Builds on the Delegation Engine and Orchestrator to provide:
- Agent roles (Researcher, Analyst, Critic, Validator, Executor)
- Collaboration patterns: Debate, Voting, Critic Review, Pipeline
- Shared task board for concurrent agent state
- Consensus validation for final output quality

Architecture: §10.2 (Multi-Agent Collaboration)
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from jarvis.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Agent Roles
# ---------------------------------------------------------------------------


class AgentRole(StrEnum):
    """Predefined collaboration roles."""

    RESEARCHER = "researcher"
    ANALYST = "analyst"
    CRITIC = "critic"
    VALIDATOR = "validator"
    EXECUTOR = "executor"
    SYNTHESIZER = "synthesizer"
    MEDIATOR = "mediator"


@dataclass(frozen=True)
class RoleSpec:
    """Specification for an agent role in a collaboration."""

    role: AgentRole
    agent_name: str = ""
    system_prompt_suffix: str = ""
    required_capabilities: list[str] = field(default_factory=list)
    max_iterations: int = 5


# Role defaults with prompting guidance
ROLE_DEFAULTS: dict[AgentRole, str] = {
    AgentRole.RESEARCHER: "Recherchiere gründlich. Liefere Fakten, Quellen und Kontext.",
    AgentRole.ANALYST: "Analysiere die Daten. Identifiziere Muster, Trends und Zusammenhänge.",
    AgentRole.CRITIC: "Prüfe kritisch. Finde Fehler, Lücken und Schwächen im Argument.",
    AgentRole.VALIDATOR: "Validiere die Ergebnisse gegen die Anforderungen. Prüfe Korrektheit.",
    AgentRole.EXECUTOR: "Führe die Aufgabe aus. Produziere konkreten Output.",
    AgentRole.SYNTHESIZER: "Fasse alle Perspektiven zusammen. Erstelle eine kohärente Synthese.",
    AgentRole.MEDIATOR: "Vermittle zwischen Positionen. Finde Konsens und löse Konflikte.",
}


# ---------------------------------------------------------------------------
# Collaboration Patterns
# ---------------------------------------------------------------------------


class CollaborationPattern(StrEnum):
    """How agents collaborate."""

    DEBATE = "debate"  # Agents argue, best output wins
    VOTING = "voting"  # n agents rate, majority decides
    CRITIC_REVIEW = "critic_review"  # One produces, critic reviews
    PIPELINE = "pipeline"  # Sequential specialization
    PARALLEL = "parallel"  # All work independently, merge results


# ---------------------------------------------------------------------------
# Task Board (shared state)
# ---------------------------------------------------------------------------


@dataclass
class BoardEntry:
    """A single entry on the shared task board."""

    agent: str
    role: AgentRole
    content: str
    score: float = 0.0
    timestamp: float = field(default_factory=time.monotonic)
    metadata: dict[str, Any] = field(default_factory=dict)


class TaskBoard:
    """Shared state for collaborating agents.

    Thread-safe via asyncio.Lock (all collaborations run in one event loop).
    """

    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        self._entries: list[BoardEntry] = []
        self._votes: dict[str, dict[str, float]] = {}  # agent → {entry_agent: score}
        self._lock = asyncio.Lock()

    async def post(self, agent: str, role: AgentRole, content: str, **meta: Any) -> None:
        """Post a contribution to the board."""
        async with self._lock:
            self._entries.append(BoardEntry(
                agent=agent, role=role, content=content, metadata=meta,
            ))

    async def vote(self, voter: str, target_agent: str, score: float) -> None:
        """Cast a vote for an agent's contribution (0.0 - 1.0)."""
        async with self._lock:
            if voter not in self._votes:
                self._votes[voter] = {}
            self._votes[voter][target_agent] = max(0.0, min(1.0, score))

    def get_entries(self, role: AgentRole | None = None) -> list[BoardEntry]:
        """Get all board entries, optionally filtered by role."""
        if role is None:
            return list(self._entries)
        return [e for e in self._entries if e.role == role]

    def get_votes(self) -> dict[str, float]:
        """Get aggregated scores per agent (average of all votes)."""
        totals: dict[str, list[float]] = {}
        for voter_scores in self._votes.values():
            for agent, score in voter_scores.items():
                totals.setdefault(agent, []).append(score)
        return {
            agent: sum(scores) / len(scores)
            for agent, scores in totals.items()
            if scores
        }

    def get_winner(self) -> BoardEntry | None:
        """Get the entry with the highest aggregated score."""
        scores = self.get_votes()
        if not scores:
            return self._entries[0] if self._entries else None
        best_agent = max(scores, key=scores.get)  # type: ignore[arg-type]
        for entry in reversed(self._entries):
            if entry.agent == best_agent:
                entry.score = scores[best_agent]
                return entry
        return None

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    @property
    def voter_count(self) -> int:
        return len(self._votes)


# ---------------------------------------------------------------------------
# Collaboration Result
# ---------------------------------------------------------------------------


@dataclass
class CollaborationResult:
    """Result of a multi-agent collaboration."""

    collaboration_id: str = ""
    pattern: str = ""
    task: str = ""
    participants: list[str] = field(default_factory=list)
    final_output: str = ""
    winner: str = ""
    consensus_score: float = 0.0
    contributions: list[dict[str, Any]] = field(default_factory=list)
    duration_ms: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return len(self.errors) == 0 and bool(self.final_output)

    def to_dict(self) -> dict[str, Any]:
        return {
            "collaboration_id": self.collaboration_id,
            "pattern": self.pattern,
            "task": self.task,
            "participants": self.participants,
            "final_output": self.final_output[:500],
            "winner": self.winner,
            "consensus_score": round(self.consensus_score, 4),
            "contributions": len(self.contributions),
            "duration_ms": round(self.duration_ms, 1),
            "success": self.success,
            "errors": self.errors,
        }


# ---------------------------------------------------------------------------
# Collaboration Engine
# ---------------------------------------------------------------------------


class CollaborationEngine:
    """Orchestrate multi-agent collaboration patterns.

    Usage::

        engine = CollaborationEngine()
        result = await engine.debate(
            task="Analyse the market trend",
            participants=[
                RoleSpec(role=AgentRole.RESEARCHER, agent_name="researcher"),
                RoleSpec(role=AgentRole.ANALYST, agent_name="analyst"),
            ],
        )
    """

    def __init__(
        self,
        *,
        agent_runner: Any | None = None,
        max_rounds: int = 3,
        consensus_threshold: float = 0.6,
    ) -> None:
        self._runner = agent_runner  # Callable(task_str, agent_name) → str
        self._max_rounds = max_rounds
        self._consensus_threshold = consensus_threshold
        self._history: list[CollaborationResult] = []

    async def _invoke_agent(self, agent_name: str, task: str) -> str:
        """Invoke an agent and return its response."""
        if self._runner is None:
            return ""
        try:
            result = await self._runner(task, agent_name)
            if hasattr(result, "response"):
                return result.response
            return str(result)
        except Exception as exc:
            log.warning("agent_invoke_error", agent=agent_name, error=str(exc))
            return f"[Error: {exc}]"

    async def debate(
        self,
        task: str,
        participants: list[RoleSpec],
        *,
        rounds: int | None = None,
    ) -> CollaborationResult:
        """Debate pattern: agents argue, best output wins.

        Each agent produces a response, then all vote on quality.
        Runs for multiple rounds if configured.
        """
        collab_id = uuid.uuid4().hex[:12]
        start = time.monotonic()
        board = TaskBoard(collab_id)
        max_rounds = rounds or self._max_rounds

        result = CollaborationResult(
            collaboration_id=collab_id,
            pattern=CollaborationPattern.DEBATE,
            task=task,
            participants=[p.agent_name for p in participants],
        )

        if len(participants) < 2:
            result.errors.append("Debate requires at least 2 participants")
            result.duration_ms = (time.monotonic() - start) * 1000
            self._history.append(result)
            return result

        for round_num in range(max_rounds):
            # Each agent produces a response
            context = ""
            if round_num > 0:
                prev_entries = board.get_entries()
                context = "\n\nVorherige Beiträge:\n" + "\n".join(
                    f"- {e.agent} ({e.role}): {e.content[:200]}"
                    for e in prev_entries[-len(participants):]
                )

            tasks = []
            for p in participants:
                prompt = ROLE_DEFAULTS.get(p.role, "")
                full_task = f"{prompt}\n\nAufgabe: {task}{context}"
                tasks.append((p, full_task))

            responses = await asyncio.gather(
                *(self._invoke_agent(p.agent_name, t) for p, t in tasks),
                return_exceptions=True,
            )

            for p, resp in zip(participants, responses):
                if isinstance(resp, Exception):
                    result.errors.append(f"{p.agent_name}: {resp}")
                    continue
                await board.post(p.agent_name, p.role, str(resp))

        # Voting phase: each agent rates all other contributions
        entries = board.get_entries()
        for p in participants:
            for entry in entries:
                if entry.agent != p.agent_name:
                    # Simple heuristic: longer, more detailed answers score higher
                    score = min(1.0, len(entry.content) / 500)
                    await board.vote(p.agent_name, entry.agent, score)

        winner = board.get_winner()
        if winner:
            result.final_output = winner.content
            result.winner = winner.agent
            result.consensus_score = winner.score

        result.contributions = [
            {"agent": e.agent, "role": e.role.value, "content": e.content[:300]}
            for e in entries
        ]
        result.duration_ms = (time.monotonic() - start) * 1000
        self._history.append(result)
        return result

    async def vote(
        self,
        task: str,
        participants: list[RoleSpec],
        *,
        voter: RoleSpec | None = None,
    ) -> CollaborationResult:
        """Voting pattern: all participants produce, voter (or majority) decides."""
        collab_id = uuid.uuid4().hex[:12]
        start = time.monotonic()
        board = TaskBoard(collab_id)

        result = CollaborationResult(
            collaboration_id=collab_id,
            pattern=CollaborationPattern.VOTING,
            task=task,
            participants=[p.agent_name for p in participants],
        )

        # All produce
        responses = await asyncio.gather(
            *(self._invoke_agent(p.agent_name, task) for p in participants),
            return_exceptions=True,
        )

        for p, resp in zip(participants, responses):
            if isinstance(resp, Exception):
                result.errors.append(f"{p.agent_name}: {resp}")
                continue
            await board.post(p.agent_name, p.role, str(resp))

        # Voting: if explicit voter, they score; otherwise cross-vote
        if voter:
            entries = board.get_entries()
            for entry in entries:
                task_prompt = (
                    f"Bewerte die Qualität dieser Antwort (0.0-1.0):\n{entry.content[:500]}"
                )
                score_str = await self._invoke_agent(voter.agent_name, task_prompt)
                try:
                    score = float(score_str.strip().replace(",", "."))
                except (ValueError, AttributeError):
                    score = 0.5
                await board.vote(voter.agent_name, entry.agent, score)
        else:
            # Cross-vote (each rates others)
            for p in participants:
                for entry in board.get_entries():
                    if entry.agent != p.agent_name:
                        score = min(1.0, len(entry.content) / 500)
                        await board.vote(p.agent_name, entry.agent, score)

        winner = board.get_winner()
        if winner:
            result.final_output = winner.content
            result.winner = winner.agent
            result.consensus_score = winner.score

        result.contributions = [
            {"agent": e.agent, "role": e.role.value, "content": e.content[:300]}
            for e in board.get_entries()
        ]
        result.duration_ms = (time.monotonic() - start) * 1000
        self._history.append(result)
        return result

    async def critic_review(
        self,
        task: str,
        producer: RoleSpec,
        critic: RoleSpec,
        *,
        max_revisions: int = 2,
    ) -> CollaborationResult:
        """Critic review: producer creates, critic reviews, producer revises."""
        collab_id = uuid.uuid4().hex[:12]
        start = time.monotonic()
        board = TaskBoard(collab_id)

        result = CollaborationResult(
            collaboration_id=collab_id,
            pattern=CollaborationPattern.CRITIC_REVIEW,
            task=task,
            participants=[producer.agent_name, critic.agent_name],
        )

        current_output = await self._invoke_agent(producer.agent_name, task)
        await board.post(producer.agent_name, producer.role, current_output)

        for revision in range(max_revisions):
            # Critic reviews
            review_task = (
                f"{ROLE_DEFAULTS.get(AgentRole.CRITIC, '')}\n\n"
                f"Originalaufgabe: {task}\n\n"
                f"Zu prüfender Output:\n{current_output[:1000]}"
            )
            critique = await self._invoke_agent(critic.agent_name, review_task)
            await board.post(critic.agent_name, critic.role, critique)

            # Check if the critic approves
            if any(word in critique.lower() for word in ["gut", "korrekt", "akzeptiert", "approve", "ok"]):
                break

            # Producer revises
            revise_task = (
                f"Überarbeite deinen Output basierend auf dem Feedback:\n\n"
                f"Feedback: {critique[:500]}\n\n"
                f"Dein vorheriger Output: {current_output[:500]}"
            )
            current_output = await self._invoke_agent(producer.agent_name, revise_task)
            await board.post(producer.agent_name, producer.role, current_output)

        result.final_output = current_output
        result.winner = producer.agent_name
        result.consensus_score = 1.0 if board.entry_count > 2 else 0.5
        result.contributions = [
            {"agent": e.agent, "role": e.role.value, "content": e.content[:300]}
            for e in board.get_entries()
        ]
        result.duration_ms = (time.monotonic() - start) * 1000
        self._history.append(result)
        return result

    async def pipeline(
        self,
        task: str,
        stages: list[RoleSpec],
    ) -> CollaborationResult:
        """Pipeline pattern: sequential processing through specialized agents."""
        collab_id = uuid.uuid4().hex[:12]
        start = time.monotonic()
        board = TaskBoard(collab_id)

        result = CollaborationResult(
            collaboration_id=collab_id,
            pattern=CollaborationPattern.PIPELINE,
            task=task,
            participants=[s.agent_name for s in stages],
        )

        current_input = task
        for stage in stages:
            prompt = ROLE_DEFAULTS.get(stage.role, "")
            stage_task = f"{prompt}\n\nInput: {current_input}"
            output = await self._invoke_agent(stage.agent_name, stage_task)

            if output.startswith("[Error:"):
                result.errors.append(f"Pipeline stopped at {stage.agent_name}: {output}")
                break

            await board.post(stage.agent_name, stage.role, output)
            current_input = output

        result.final_output = current_input
        result.winner = stages[-1].agent_name if stages else ""
        result.consensus_score = 1.0 if not result.errors else 0.0
        result.contributions = [
            {"agent": e.agent, "role": e.role.value, "content": e.content[:300]}
            for e in board.get_entries()
        ]
        result.duration_ms = (time.monotonic() - start) * 1000
        self._history.append(result)
        return result

    async def parallel(
        self,
        task: str,
        participants: list[RoleSpec],
    ) -> CollaborationResult:
        """Parallel pattern: all work independently, synthesize results."""
        collab_id = uuid.uuid4().hex[:12]
        start = time.monotonic()
        board = TaskBoard(collab_id)

        result = CollaborationResult(
            collaboration_id=collab_id,
            pattern=CollaborationPattern.PARALLEL,
            task=task,
            participants=[p.agent_name for p in participants],
        )

        responses = await asyncio.gather(
            *(self._invoke_agent(p.agent_name, task) for p in participants),
            return_exceptions=True,
        )

        for p, resp in zip(participants, responses):
            if isinstance(resp, Exception):
                result.errors.append(f"{p.agent_name}: {resp}")
                continue
            await board.post(p.agent_name, p.role, str(resp))

        # Synthesize: combine all contributions
        entries = board.get_entries()
        if entries:
            combined = "\n\n---\n\n".join(
                f"**{e.agent}** ({e.role}):\n{e.content}" for e in entries
            )
            result.final_output = combined
            result.winner = entries[0].agent  # First contributor as "lead"
            result.consensus_score = 1.0

        result.contributions = [
            {"agent": e.agent, "role": e.role.value, "content": e.content[:300]}
            for e in entries
        ]
        result.duration_ms = (time.monotonic() - start) * 1000
        self._history.append(result)
        return result

    @property
    def history(self) -> list[CollaborationResult]:
        return list(self._history)

    def stats(self) -> dict[str, Any]:
        """Collaboration statistics."""
        by_pattern: dict[str, int] = {}
        for r in self._history:
            by_pattern[r.pattern] = by_pattern.get(r.pattern, 0) + 1
        return {
            "total_collaborations": len(self._history),
            "by_pattern": by_pattern,
            "success_count": sum(1 for r in self._history if r.success),
            "failure_count": sum(1 for r in self._history if not r.success),
            "max_rounds": self._max_rounds,
            "consensus_threshold": self._consensus_threshold,
        }
