"""Autonomous Task Orchestrator for Cognithor.

Handles complex, multi-step tasks that may require:
- Task decomposition into subtasks
- Tool discovery and skill creation
- Recurring execution via cron
- Self-evaluation and iterative refinement
- Learning from results for future tasks
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class AutonomousTask:
    """A complex task that Cognithor should solve autonomously."""

    task_id: str
    description: str
    subtasks: list[str] = field(default_factory=list)
    recurring: str = "none"  # "none", "hourly", "daily", "weekly"
    max_attempts: int = 3
    current_attempt: int = 0
    status: str = "pending"  # pending, running, completed, failed
    results: list[dict[str, Any]] = field(default_factory=list)
    quality_score: float = 0.0
    skills_created: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


class AutonomousOrchestrator:
    """Orchestrates complex autonomous task execution.

    Connects PGE loop, SkillGenerator, Reflector, and Cron to handle
    tasks that the system hasn't seen before. When Cognithor encounters
    a novel task, this orchestrator:

    1. Decomposes it into subtasks (via planner)
    2. Checks if required skills/tools exist
    3. Creates missing skills on-the-fly
    4. Executes subtasks with self-evaluation
    5. Sets up recurring execution if requested
    6. Learns from results for future tasks
    """

    # Minimum quality score to consider a task "well done"
    QUALITY_THRESHOLD = 0.7

    # Keywords that suggest the task should be recurring
    RECURRING_KEYWORDS = {
        "taeglich": "daily",
        "daily": "daily",
        "jeden tag": "daily",
        "every day": "daily",
        "woechentlich": "weekly",
        "weekly": "weekly",
        "jede woche": "weekly",
        "every week": "weekly",
        "stuendlich": "hourly",
        "hourly": "hourly",
        "jede stunde": "hourly",
        "every hour": "hourly",
        "regelmaessig": "daily",
        "continuously": "hourly",
        "keep checking": "hourly",
        "monitor": "hourly",
        "ueberwache": "hourly",
    }

    def __init__(
        self,
        gateway: Any = None,
        skill_generator: Any = None,
        reflector: Any = None,
    ) -> None:
        self._gateway = gateway
        self._skill_generator = skill_generator
        self._reflector = reflector
        self._active_tasks: dict[str, AutonomousTask] = {}

    def detect_recurring(self, user_message: str) -> str:
        """Detect if a task should be recurring based on keywords."""
        lower = user_message.lower()
        for keyword, frequency in self.RECURRING_KEYWORDS.items():
            if keyword in lower:
                return frequency
        return "none"

    def detect_complexity(self, user_message: str) -> str:
        """Classify task complexity: simple, moderate, complex."""
        lower = user_message.lower()
        # Complex: multiple distinct objectives or research + action
        complex_signals = [
            " und " in lower and len(lower) > 100,
            "recherchiere" in lower and ("erstell" in lower or "schreib" in lower),
            "setup" in lower or "installier" in lower,
            "monitor" in lower or "ueberwach" in lower,
            "analysier" in lower and "bericht" in lower,
            "vergleich" in lower and "zusammenfass" in lower,
        ]
        if sum(complex_signals) >= 2:
            return "complex"
        if sum(complex_signals) >= 1 or len(lower) > 150:
            return "moderate"
        return "simple"

    def should_orchestrate(self, user_message: str) -> bool:
        """Determine if a message needs autonomous orchestration.

        Returns True for complex or recurring tasks that benefit
        from decomposition and self-evaluation.
        """
        complexity = self.detect_complexity(user_message)
        recurring = self.detect_recurring(user_message)
        return complexity in ("complex", "moderate") or recurring != "none"

    def create_task(self, user_message: str, session_id: str) -> AutonomousTask:
        """Create an autonomous task from a user message."""
        task_id = f"auto_{int(time.time())}_{session_id[:8]}"
        recurring = self.detect_recurring(user_message)

        task = AutonomousTask(
            task_id=task_id,
            description=user_message,
            recurring=recurring,
        )
        self._active_tasks[task_id] = task
        log.info(
            "autonomous_task_created: task_id=%s recurring=%s complexity=%s",
            task_id,
            recurring,
            self.detect_complexity(user_message),
        )
        return task

    def get_orchestration_prompt(self, task: AutonomousTask) -> str:
        """Generate enhanced system prompt additions for autonomous execution.

        This prompt is injected into the planner to guide autonomous behavior:
        - Decompose complex tasks
        - Self-evaluate after each step
        - Create skills for reusable patterns
        - Set up cron for recurring tasks
        """
        parts = [
            "## Autonome Ausfuehrung",
            "",
            "Diese Aufgabe erfordert autonomes Handeln. Befolge diese Strategie:",
            "",
            "1. **Zerlege** die Aufgabe in konkrete Teilschritte",
            "2. **Pruefe** nach jedem Schritt: War das Ergebnis gut genug?",
            "3. **Verbessere** wenn noetig: Suche mehr Quellen, probiere andere Tools",
            "4. **Lerne** daraus: Wenn du einen neuen Workflow entdeckst, erstelle einen Skill mit create_skill",
        ]

        if task.recurring != "none":
            freq_map = {
                "hourly": "stuendlich",
                "daily": "taeglich",
                "weekly": "woechentlich",
            }
            freq_de = freq_map.get(task.recurring, task.recurring)
            parts.extend([
                "",
                f"5. **Wiederkehrend**: Diese Aufgabe soll {freq_de} ausgefuehrt werden.",
                f"   Erstelle am Ende einen Reminder mit set_reminder(repeat='{task.recurring}').",
                "   Der Reminder-Text soll die Aufgabe beschreiben, damit sie automatisch wiederholt wird.",
            ])

        if task.current_attempt > 0:
            parts.extend([
                "",
                f"HINWEIS: Dies ist Versuch {task.current_attempt + 1} von {task.max_attempts}.",
                "Vorherige Versuche waren nicht gut genug. Probiere einen anderen Ansatz.",
            ])

        return "\n".join(parts)

    def evaluate_result(
        self, task: AutonomousTask, response: str, tool_results: list[Any]
    ) -> float:
        """Evaluate the quality of task execution.

        Returns a score from 0.0 to 1.0:
        - 1.0: Task fully completed with good results
        - 0.7+: Acceptable quality
        - 0.5: Partial completion
        - 0.0: Complete failure
        """
        score = 0.5  # Base score

        # Positive signals
        if tool_results:
            successful = sum(1 for r in tool_results if getattr(r, "success", True))
            total = len(tool_results)
            if total > 0:
                score += 0.3 * (successful / total)

        if len(response) > 100:
            score += 0.1  # Substantive response

        if any(
            getattr(r, "tool_name", "")
            in ("search_and_read", "deep_research", "verified_web_lookup")
            for r in tool_results
        ):
            score += 0.1  # Used research tools

        # Negative signals
        if not tool_results:
            score -= 0.2  # No tools used for a complex task

        error_results = sum(1 for r in tool_results if getattr(r, "is_error", False))
        if error_results > 0:
            score -= 0.1 * min(error_results, 3)

        return max(0.0, min(1.0, score))

    def get_active_tasks(self) -> list[dict[str, Any]]:
        """List all active autonomous tasks."""
        return [
            {
                "task_id": t.task_id,
                "description": t.description[:100],
                "status": t.status,
                "recurring": t.recurring,
                "quality_score": t.quality_score,
                "attempt": t.current_attempt,
                "skills_created": t.skills_created,
            }
            for t in self._active_tasks.values()
        ]
