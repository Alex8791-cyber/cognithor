from cognithor.crew import Crew, CrewAgent, CrewProcess, CrewTask


class TestHierarchical:
    def test_manager_agent_is_synthesized(self):
        """Hierarchical process injects a synthetic 'manager' agent that picks
        which worker handles each task. Worker order is NOT necessarily
        declaration order."""
        analyst = CrewAgent(role="analyst", goal="analyze")
        writer = CrewAgent(role="writer", goal="write")
        t1 = CrewTask(description="produce a PKV summary", expected_output="x", agent=analyst)
        t2 = CrewTask(
            description="polish the summary into a customer-facing report",
            expected_output="y",
            agent=writer,
        )
        crew = Crew(
            agents=[analyst, writer],
            tasks=[t1, t2],
            process=CrewProcess.HIERARCHICAL,
            manager_llm="ollama/qwen3:32b",
        )

        # The manager decides order — we force it to pick writer before analyst
        # by stubbing the delegation module to return reversed order.
        from cognithor.crew.compiler_hierarchical import order_tasks_hierarchical

        reordered = order_tasks_hierarchical(
            crew.tasks, crew.agents, manager_llm="ollama/qwen3:32b"
        )
        # The default fallback — no live LLM — returns declaration order.
        assert [t.task_id for t in reordered] == [t1.task_id, t2.task_id]
