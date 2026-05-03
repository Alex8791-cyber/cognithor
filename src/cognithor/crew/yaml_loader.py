"""Load a Crew from YAML config files.

Accepts two files:
  agents.yaml — dict keyed by agent-alias, values are CrewAgent-kwargs dicts
  tasks.yaml  — dict keyed by task-alias, values are CrewTask-kwargs dicts
                (``agent: <alias>``, ``context: [<alias>, ...]``)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from cognithor.crew.agent import CrewAgent
from cognithor.crew.crew import Crew
from cognithor.crew.errors import CrewCompilationError
from cognithor.crew.process import CrewProcess
from cognithor.crew.task import CrewTask


def _localized(key: str, *, fallback: str, **kwargs: Any) -> str:
    """Return the translated message for *key* or *fallback* if untranslated.

    ``cognithor.i18n.t`` returns the raw key string when no locale pack
    covers it — in that case ``format_map(kwargs)`` is skipped, so the
    template placeholders are never substituted and the message is
    useless. Detect that exact pass-through and substitute our English
    fallback (which already has the placeholders interpolated by the
    caller's f-string) so errors stay readable until Task 18 ships the
    real keys.
    """
    from cognithor.i18n import t

    message = t(key, **kwargs)
    if message == key:
        return fallback
    return message


def load_crew_from_yaml(
    *,
    agents: Path | str,
    tasks: Path | str,
    process: CrewProcess = CrewProcess.SEQUENTIAL,
    verbose: bool = False,
    planning: bool = False,
    manager_llm: str | None = None,
) -> Crew:
    """Load a Crew from paired ``agents.yaml`` + ``tasks.yaml`` files.

    Task aliases with ``context: [other_alias, ...]`` are resolved in a
    second pass via :meth:`pydantic.BaseModel.model_copy` — model_dump
    cannot serialize callable ``guardrail`` fields, so a dump-then-reinit
    round-trip would silently drop them. ``model_copy(update=...)`` keeps
    every untouched field by identity.
    """
    agents_data: dict[str, Any] = yaml.safe_load(Path(agents).read_text(encoding="utf-8")) or {}
    tasks_data: dict[str, Any] = yaml.safe_load(Path(tasks).read_text(encoding="utf-8")) or {}

    # Pass 1a: build agents by alias
    agent_by_alias: dict[str, CrewAgent] = {
        alias: CrewAgent(**kwargs) for alias, kwargs in agents_data.items()
    }

    # Pass 1b: build tasks without context (context refs need all tasks first)
    task_by_alias: dict[str, CrewTask] = {}
    context_map: dict[str, list[str]] = {}
    for alias, kwargs in tasks_data.items():
        kwargs = dict(kwargs)  # don't mutate user data
        agent_alias = kwargs.pop("agent")
        if agent_alias not in agent_by_alias:
            raise ValueError(
                _localized(
                    "crew.errors.unknown_agent",
                    fallback=(
                        f"Task '{alias}' references unknown agent "
                        f"'{agent_alias}'. Known agents: "
                        f"{', '.join(agent_by_alias) or '(none)'}."
                    ),
                    task=alias,
                    agent=agent_alias,
                    known=", ".join(agent_by_alias) or "(none)",
                )
            )
        context_map[alias] = kwargs.pop("context", []) or []
        task_by_alias[alias] = CrewTask(agent=agent_by_alias[agent_alias], context=[], **kwargs)

    # Pass 2: resolve context refs; rebuild frozen tasks via model_copy
    for alias, refs in context_map.items():
        if not refs:
            continue
        ctx: list[CrewTask] = []
        for ref in refs:
            if ref not in task_by_alias:
                raise CrewCompilationError(
                    _localized(
                        "crew.errors.unknown_task",
                        fallback=(
                            f"Task '{alias}' references unknown task "
                            f"'{ref}' in its context. Known tasks: "
                            f"{', '.join(task_by_alias) or '(none)'}."
                        ),
                        task=alias,
                        ref=ref,
                        known=", ".join(task_by_alias) or "(none)",
                    )
                )
            ctx.append(task_by_alias[ref])
        existing = task_by_alias[alias]
        task_by_alias[alias] = existing.model_copy(update={"context": ctx})

    return Crew(
        agents=list(agent_by_alias.values()),
        tasks=list(task_by_alias.values()),
        process=process,
        verbose=verbose,
        planning=planning,
        manager_llm=manager_llm,
    )
