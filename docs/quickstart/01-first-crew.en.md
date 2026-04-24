# 01 · First Crew

Your first multi-agent workflow: researcher + reporter running sequentially.

**Prerequisites**
- Completed: [00 · Installation](00-installation.en.md)
- `ollama list` shows `qwen3:8b`

**Time:** 5 minutes
**End state:** You have run a `main.py` locally and seen a two-stage Markdown report as output.

---

## 1. Create the project

```bash
mkdir my_first_crew && cd my_first_crew
```

Create `main.py` with the following content (from spec §1.4 — the PKV example, simplified to smart-home research):

```python
"""First Crew — sequential researcher+reporter pattern."""

from __future__ import annotations

import asyncio

from cognithor.crew import Crew, CrewAgent, CrewProcess, CrewTask


def build_crew() -> Crew:
    researcher = CrewAgent(
        role="Researcher",
        goal="Research facts on the given topic",
        llm="ollama/qwen3:8b",
    )
    reporter = CrewAgent(
        role="Reporter",
        goal="Write a structured report",
        llm="ollama/qwen3:8b",
    )
    research = CrewTask(
        description="Research: Home-automation trends in 2026",
        expected_output="Bullet points of the 5 most important trends",
        agent=researcher,
    )
    report = CrewTask(
        description="Produce a report based on the research",
        expected_output="Markdown report, 300 words",
        agent=reporter,
        context=[research],
    )
    return Crew(
        agents=[researcher, reporter],
        tasks=[research, report],
        process=CrewProcess.SEQUENTIAL,
    )


def main() -> None:
    crew = build_crew()
    result = asyncio.run(crew.kickoff_async())
    print(result.raw)


if __name__ == "__main__":
    main()
```

## 2. Run it

```bash
python main.py
```

The first run can take 30–60 s while Ollama warms up the model. Subsequent runs are much faster.

Expected output (truncated):

```
# Home-Automation Trends 2026

- Matter/Thread adoption on >60% of new devices
- Local AI inference (edge LLMs) for voice control
- ...
```

## 3. What happened?

1. **`CrewAgent`** declares only the role — no prompts, no code.
2. **`CrewTask`** declares the work, plus dependencies via `context=[research]`.
3. **`Crew(process=SEQUENTIAL)`** compiles both tasks into a DAG and runs them in order.
4. Every task internally goes through the **PGE-Trinity** (Planner → Gatekeeper → Executor), exactly like any other Cognithor action. No new security surface.
5. The final `result.raw` is the output of the last task. All intermediate outputs are in `result.tasks_output`.

## 4. Runnable copy in the repo

The same example lives as a standalone mini-project at [`examples/quickstart/01_first_crew/`](../../examples/quickstart/01_first_crew/). You'll find:

- `main.py` — identical script
- `requirements.txt` — just `cognithor>=0.93.0`
- `test_example.py` — a smoke test with mocked planner, runs in CI

## 5. Variations

- **More agents:** add any number of `CrewAgent` instances to `agents=`.
- **Hierarchical:** set `process=CrewProcess.HIERARCHICAL` + `manager=CrewAgent(...)` to let a manager agent delegate tasks.
- **YAML config:** instead of Python, use `agents.yaml` + `tasks.yaml` — see `cognithor.crew.yaml_loader`.

---

**Next:** [02 · First Tool](02-first-tool.en.md)
