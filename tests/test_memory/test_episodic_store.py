"""Tests fuer EpisodicStore und EpisodicSummarizer."""

import pytest
from jarvis.memory.episodic_store import EpisodicStore
from jarvis.memory.episodic_summarizer import EpisodicSummarizer
from jarvis.models import EpisodicEntry


class TestEpisodicStore:
    def setup_method(self):
        self.store = EpisodicStore()  # in-memory

    def teardown_method(self):
        self.store.close()

    def test_store_episode(self):
        eid = self.store.store_episode(
            session_id="s1",
            topic="File Operations",
            content="Read and modified config.yaml",
            outcome="Success",
            tool_sequence=["read_file", "write_file"],
            success_score=0.9,
            tags=["file", "config"],
        )
        assert eid
        assert self.store.get_episode_count() == 1

    def test_get_session_episodes(self):
        self.store.store_episode("s1", "Topic A", "Content A")
        self.store.store_episode("s1", "Topic B", "Content B")
        self.store.store_episode("s2", "Topic C", "Content C")

        episodes = self.store.get_session_episodes("s1")
        assert len(episodes) == 2
        assert all(e.session_id == "s1" for e in episodes)

    def test_search_episodes_fts(self):
        self.store.store_episode("s1", "Python Development", "Wrote a REST API with Flask")
        self.store.store_episode("s2", "Database Setup", "Configured PostgreSQL")

        results = self.store.search_episodes("Python Flask")
        assert len(results) >= 1
        assert any("Python" in e.topic for e in results)

    def test_search_with_min_score(self):
        self.store.store_episode("s1", "Good Task", "Success", success_score=0.9)
        self.store.store_episode("s2", "Bad Task", "Failed task execution", success_score=0.2)

        results = self.store.search_episodes("Task", min_score=0.5)
        assert all(e.success_score >= 0.5 for e in results)

    def test_get_similar_episodes(self):
        self.store.store_episode(
            "s1", "API Test", "Tested API",
            tool_sequence=["read_file", "exec_command", "write_file"],
            success_score=0.9,
        )
        self.store.store_episode(
            "s2", "Another Test", "Another test",
            tool_sequence=["read_file", "exec_command"],
            success_score=0.8,
        )
        self.store.store_episode(
            "s3", "Unrelated", "Unrelated",
            tool_sequence=["search_memory"],
            success_score=0.7,
        )

        similar = self.store.get_similar_episodes(["read_file", "exec_command"], limit=2)
        assert len(similar) >= 1
        # The most similar should be s1 or s2
        assert similar[0].session_id in ("s1", "s2")

    def test_store_and_get_summary(self):
        sid = self.store.store_summary(
            period="day",
            start_date="2025-01-01",
            end_date="2025-01-02",
            summary="Productive day",
            key_learnings=["Use caching"],
        )
        assert sid

        summaries = self.store.get_summaries(period="day")
        assert len(summaries) == 1
        assert summaries[0]["summary"] == "Productive day"
        assert "Use caching" in summaries[0]["key_learnings"]

    def test_search_invalid_query(self):
        """FTS5 invalid syntax should return empty, not crash."""
        results = self.store.search_episodes("AND OR NOT")
        assert results == []

    def test_empty_store(self):
        assert self.store.get_episode_count() == 0
        assert self.store.get_session_episodes("nonexistent") == []
        assert self.store.get_similar_episodes([]) == []


class TestEpisodicSummarizer:
    def setup_method(self):
        self.store = EpisodicStore()
        self.summarizer = EpisodicSummarizer(self.store, llm=None)

    def teardown_method(self):
        self.store.close()

    @pytest.mark.asyncio
    async def test_summarize_day_no_episodes(self):
        from datetime import date
        result = await self.summarizer.summarize_day(date(2025, 1, 1))
        assert "Keine Episoden" in result

    @pytest.mark.asyncio
    async def test_summarize_week_no_summaries(self):
        from datetime import date
        result = await self.summarizer.summarize_week(date(2025, 1, 1))
        assert "Keine Zusammenfassungen" in result
