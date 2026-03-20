"""Integration tests for v0.48.0 features."""

import pytest
from pathlib import Path


# Test Reflexion Memory
class TestReflexionIntegration:
    def test_record_lookup_prevention_cycle(self, tmp_path):
        from jarvis.learning.reflexion import ReflexionMemory

        mem = ReflexionMemory(data_dir=tmp_path)
        # Record -> lookup -> prevention rule -> adopt
        entry = mem.record_error(
            "web_search",
            "timeout",
            "Connection timed out after 30s",
            "rate_limited",
            "Add 2s delay between calls",
            "search weather",
        )
        found = mem.get_solution("web_search", "timeout", "Connection timed out after 30s")
        assert found is not None
        assert found.prevention_rule == "Add 2s delay between calls"
        rules = mem.get_prevention_rules("web_search")
        assert len(rules) == 1
        mem.adopt_rule(entry.error_signature)
        assert (
            mem.get_solution("web_search", "timeout", "Connection timed out after 30s").status
            == "adopted"
        )


# Test Confidence Checker
class TestConfidenceIntegration:
    def test_confidence_with_reflexion(self, tmp_path):
        from jarvis.core.confidence import ConfidenceChecker
        from jarvis.learning.reflexion import ReflexionMemory

        mem = ReflexionMemory(data_dir=tmp_path)
        # Record recurring error
        for _ in range(5):
            mem.record_error("web_search", "timeout", "timeout", "slow", "add delay")
        checker = ConfidenceChecker(reflexion_memory=mem)
        result = checker.assess("search for weather", "web_search")
        # Past mistakes should lower score
        assert result.mistake_score < 1.0


# Test Token Budget
class TestTokenBudgetIntegration:
    def test_complexity_detection(self):
        from jarvis.core.token_budget import TokenBudgetManager

        assert TokenBudgetManager.detect_complexity("fix typo") == "simple"
        assert (
            TokenBudgetManager.detect_complexity(
                "research and analyze the complete architecture of"
                " distributed systems with examples"
            )
            == "research"
        )

    def test_channel_multiplier(self):
        from jarvis.core.token_budget import TokenBudgetManager

        tg = TokenBudgetManager(complexity="medium", channel="telegram")
        web = TokenBudgetManager(complexity="medium", channel="webui")
        assert tg.total < web.total


# Test Channel Flags
class TestChannelFlagsIntegration:
    def test_telegram_compact(self):
        from jarvis.core.channel_flags import get_channel_flags

        flags = get_channel_flags("telegram")
        assert flags.token_efficient is True
        assert flags.compact_output is True
        assert flags.max_response_length == 4000

    def test_voice_short(self):
        from jarvis.core.channel_flags import get_channel_flags

        flags = get_channel_flags("voice")
        assert flags.max_response_length == 500
        assert flags.allow_markdown is False


# Test Session Store Extensions
class TestSessionStoreIntegration:
    def test_session_lifecycle(self, tmp_path):
        from jarvis.gateway.session_store import SessionStore
        from jarvis.models import SessionContext

        store = SessionStore(str(tmp_path / "sessions.db"))
        # Create session
        session = SessionContext(
            session_id="test123",
            channel="webui",
            user_id="user1",
            agent_name="jarvis",
        )
        store.save_session(session)
        # List sessions
        sessions = store.list_sessions_for_channel("webui", "user1")
        assert len(sessions) >= 1
        # Update title
        store.update_session_title("test123", "My Test Chat")
        sessions = store.list_sessions_for_channel("webui", "user1")
        assert any(s["title"] == "My Test Chat" for s in sessions)
        # Delete
        store.delete_session("test123")
        sessions = store.list_sessions_for_channel("webui", "user1")
        assert not any(s["session_id"] == "test123" for s in sessions)


# Test Response Validator
class TestResponseValidatorIntegration:
    def test_assumption_detection(self):
        from jarvis.core.response_validator import ResponseValidator

        v = ResponseValidator()
        result = v.validate("This probably works and should be fine", "fix the bug")
        # penalized for "probably" and "should"
        assert result.assumption_score < 0.8

    def test_good_response(self):
        from jarvis.core.response_validator import ResponseValidator

        v = ResponseValidator()
        result = v.validate(
            "The file was updated successfully. The test passes.",
            "update the file",
        )
        assert result.score >= 0.5
