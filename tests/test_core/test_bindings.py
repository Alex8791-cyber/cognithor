"""Tests für das deterministische Bindings-System.

Testet alle Aspekte:
  - MessageContext-Erstellung und Command-Parsing
  - Einzelne Binding-Bedingungen (Channel, User, Command, Regex, Metadata, Zeit)
  - AND-Verknüpfung mehrerer Bedingungen
  - Negation (NOT-Logik)
  - BindingEngine: Prioritäts-Sortierung, First-Match-Wins, Fallback
  - YAML-Persistenz (Laden und Speichern)
  - Factory-Funktionen
  - Integration mit AgentRouter (Bindings vor Keywords)
  - TimeWindow mit Über-Mitternacht-Logik
  - Edge Cases und Fehlerbehandlung
"""

from __future__ import annotations

from datetime import datetime, time
from pathlib import Path

import pytest

from jarvis.core.bindings import (
    BindingEngine,
    BindingMatchResult,
    MessageBinding,
    MessageContext,
    TimeWindow,
    Weekday,
    channel_binding,
    command_binding,
    regex_binding,
    schedule_binding,
    user_binding,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def engine() -> BindingEngine:
    """Leere BindingEngine."""
    return BindingEngine()


@pytest.fixture
def telegram_ctx() -> MessageContext:
    """Standard-Telegram-Kontext."""
    return MessageContext(
        text="Hallo Jarvis, was gibt's Neues?",
        channel="telegram",
        user_id="alex",
    )


@pytest.fixture
def cli_ctx() -> MessageContext:
    """Standard-CLI-Kontext."""
    return MessageContext(
        text="/code Schreibe einen Python-Parser",
        channel="cli",
        user_id="developer",
    )


# ============================================================================
# MessageContext
# ============================================================================


class TestMessageContext:
    """MessageContext-Erstellung und Command-Parsing."""

    def test_basic_creation(self) -> None:
        ctx = MessageContext(text="Hallo Welt", channel="telegram", user_id="alex")
        assert ctx.text == "Hallo Welt"
        assert ctx.channel == "telegram"
        assert ctx.user_id == "alex"

    def test_command_extraction(self) -> None:
        ctx = MessageContext(text="/code Schreibe Python")
        assert ctx.command == "/code"
        assert ctx.text_without_command == "Schreibe Python"

    def test_command_case_insensitive(self) -> None:
        ctx = MessageContext(text="/CODE Hallo")
        assert ctx.command == "/code"

    def test_no_command(self) -> None:
        ctx = MessageContext(text="Normaler Text ohne Command")
        assert ctx.command == ""
        assert ctx.text_without_command == "Normaler Text ohne Command"

    def test_command_only(self) -> None:
        ctx = MessageContext(text="/hilfe")
        assert ctx.command == "/hilfe"
        assert ctx.text_without_command == ""

    def test_empty_text(self) -> None:
        ctx = MessageContext(text="")
        assert ctx.command == ""
        assert ctx.text_without_command == ""

    def test_from_incoming_message(self) -> None:
        """Simuliert IncomingMessage-Konvertierung."""
        class FakeMsg:
            text = "Test"
            channel = "api"
            user_id = "bot"
            metadata = {"key": "value"}
            timestamp = None

        ctx = MessageContext.from_incoming(FakeMsg())
        assert ctx.text == "Test"
        assert ctx.channel == "api"
        assert ctx.user_id == "bot"
        assert ctx.metadata == {"key": "value"}

    def test_metadata_default(self) -> None:
        ctx = MessageContext(text="Test")
        assert ctx.metadata == {}


# ============================================================================
# Einzelne Binding-Bedingungen
# ============================================================================


class TestChannelCondition:
    """Channel-basierte Bindings."""

    def test_matching_channel(self) -> None:
        binding = MessageBinding(
            name="tg",
            target_agent="organizer",
            channels=["telegram"],
        )
        ctx = MessageContext(text="Test", channel="telegram")
        assert binding.evaluate(ctx) == BindingMatchResult.MATCH

    def test_non_matching_channel(self) -> None:
        binding = MessageBinding(
            name="tg",
            target_agent="organizer",
            channels=["telegram"],
        )
        ctx = MessageContext(text="Test", channel="cli")
        assert binding.evaluate(ctx) == BindingMatchResult.NO_MATCH

    def test_multiple_channels(self) -> None:
        binding = MessageBinding(
            name="web",
            target_agent="support",
            channels=["webui", "api"],
        )
        ctx_webui = MessageContext(text="Test", channel="webui")
        ctx_api = MessageContext(text="Test", channel="api")
        ctx_cli = MessageContext(text="Test", channel="cli")

        assert binding.evaluate(ctx_webui) == BindingMatchResult.MATCH
        assert binding.evaluate(ctx_api) == BindingMatchResult.MATCH
        assert binding.evaluate(ctx_cli) == BindingMatchResult.NO_MATCH

    def test_no_channel_filter(self) -> None:
        binding = MessageBinding(name="all", target_agent="jarvis")
        ctx = MessageContext(text="Test", channel="anything")
        assert binding.evaluate(ctx) == BindingMatchResult.MATCH


class TestUserCondition:
    """User-basierte Bindings."""

    def test_matching_user(self) -> None:
        binding = MessageBinding(
            name="vip",
            target_agent="premium",
            user_ids=["alex", "boss"],
        )
        ctx = MessageContext(text="Test", user_id="alex")
        assert binding.evaluate(ctx) == BindingMatchResult.MATCH

    def test_non_matching_user(self) -> None:
        binding = MessageBinding(
            name="vip",
            target_agent="premium",
            user_ids=["alex"],
        )
        ctx = MessageContext(text="Test", user_id="stranger")
        assert binding.evaluate(ctx) == BindingMatchResult.NO_MATCH

    def test_no_user_filter(self) -> None:
        binding = MessageBinding(name="all", target_agent="jarvis")
        ctx = MessageContext(text="Test", user_id="anyone")
        assert binding.evaluate(ctx) == BindingMatchResult.MATCH


class TestCommandCondition:
    """Command-prefix-basierte Bindings."""

    def test_matching_command(self) -> None:
        binding = MessageBinding(
            name="code_cmd",
            target_agent="coder",
            command_prefixes=["/code", "/shell"],
        )
        ctx = MessageContext(text="/code Schreibe einen Parser")
        assert binding.evaluate(ctx) == BindingMatchResult.MATCH

    def test_second_command(self) -> None:
        binding = MessageBinding(
            name="code_cmd",
            target_agent="coder",
            command_prefixes=["/code", "/shell"],
        )
        ctx = MessageContext(text="/shell ls -la")
        assert binding.evaluate(ctx) == BindingMatchResult.MATCH

    def test_non_matching_command(self) -> None:
        binding = MessageBinding(
            name="code_cmd",
            target_agent="coder",
            command_prefixes=["/code"],
        )
        ctx = MessageContext(text="/hilfe Was ist Jarvis?")
        assert binding.evaluate(ctx) == BindingMatchResult.NO_MATCH

    def test_no_command_in_text(self) -> None:
        binding = MessageBinding(
            name="code_cmd",
            target_agent="coder",
            command_prefixes=["/code"],
        )
        ctx = MessageContext(text="Normaler Text")
        assert binding.evaluate(ctx) == BindingMatchResult.NO_MATCH


class TestRegexCondition:
    """Regex-pattern-basierte Bindings."""

    def test_simple_pattern(self) -> None:
        binding = MessageBinding(
            name="insurance",
            target_agent="tarif_berater",
            message_patterns=[r"bu[- ]?tarif", r"berufsunfähigkeit"],
        )
        ctx = MessageContext(text="Was kostet ein BU-Tarif?")
        assert binding.evaluate(ctx) == BindingMatchResult.MATCH

    def test_case_insensitive(self) -> None:
        binding = MessageBinding(
            name="insurance",
            target_agent="tarif_berater",
            message_patterns=[r"versicherung"],
        )
        ctx = MessageContext(text="Meine VERSICHERUNG kündigen")
        assert binding.evaluate(ctx) == BindingMatchResult.MATCH

    def test_no_match(self) -> None:
        binding = MessageBinding(
            name="insurance",
            target_agent="tarif_berater",
            message_patterns=[r"versicherung"],
        )
        ctx = MessageContext(text="Wie wird das Wetter?")
        assert binding.evaluate(ctx) == BindingMatchResult.NO_MATCH

    def test_complex_regex(self) -> None:
        binding = MessageBinding(
            name="email",
            target_agent="mail_bot",
            message_patterns=[r"mail\s+(an|to|für)\s+\w+"],
        )
        ctx = MessageContext(text="Schick eine Mail an Max")
        assert binding.evaluate(ctx) == BindingMatchResult.MATCH

    def test_invalid_regex_graceful(self) -> None:
        """Ungültige Regex → kein Match, kein Crash."""
        binding = MessageBinding(
            name="broken",
            target_agent="test",
            message_patterns=["[invalid"],
        )
        ctx = MessageContext(text="Test")
        assert binding.evaluate(ctx) == BindingMatchResult.NO_MATCH


class TestMetadataCondition:
    """Metadata-basierte Bindings."""

    def test_matching_metadata(self) -> None:
        binding = MessageBinding(
            name="cron",
            target_agent="organizer",
            metadata_conditions={"cron_job": "morning_briefing"},
        )
        ctx = MessageContext(
            text="Test",
            metadata={"cron_job": "morning_briefing"},
        )
        assert binding.evaluate(ctx) == BindingMatchResult.MATCH

    def test_missing_key(self) -> None:
        binding = MessageBinding(
            name="cron",
            target_agent="organizer",
            metadata_conditions={"cron_job": "test"},
        )
        ctx = MessageContext(text="Test", metadata={})
        assert binding.evaluate(ctx) == BindingMatchResult.NO_MATCH

    def test_wrong_value(self) -> None:
        binding = MessageBinding(
            name="cron",
            target_agent="organizer",
            metadata_conditions={"cron_job": "morning"},
        )
        ctx = MessageContext(
            text="Test",
            metadata={"cron_job": "evening"},
        )
        assert binding.evaluate(ctx) == BindingMatchResult.NO_MATCH

    def test_multiple_conditions(self) -> None:
        binding = MessageBinding(
            name="specific",
            target_agent="special",
            metadata_conditions={"source": "api", "version": "2"},
        )
        ctx_match = MessageContext(
            text="Test",
            metadata={"source": "api", "version": "2"},
        )
        ctx_partial = MessageContext(
            text="Test",
            metadata={"source": "api"},
        )

        assert binding.evaluate(ctx_match) == BindingMatchResult.MATCH
        assert binding.evaluate(ctx_partial) == BindingMatchResult.NO_MATCH


# ============================================================================
# TimeWindow
# ============================================================================


class TestTimeWindow:
    """Zeitfenster-Bedingungen."""

    def test_within_business_hours(self) -> None:
        tw = TimeWindow(
            start_time=time(8, 0),
            end_time=time(18, 0),
        )
        # Dienstag 10:30
        now = datetime(2026, 2, 24, 10, 30)
        assert tw.matches(now) is True

    def test_outside_business_hours(self) -> None:
        tw = TimeWindow(
            start_time=time(8, 0),
            end_time=time(18, 0),
        )
        # Dienstag 20:00
        now = datetime(2026, 2, 24, 20, 0)
        assert tw.matches(now) is False

    def test_weekday_filter(self) -> None:
        tw = TimeWindow(
            weekdays=[Weekday.MONDAY, Weekday.TUESDAY, Weekday.WEDNESDAY,
                      Weekday.THURSDAY, Weekday.FRIDAY],
        )
        # Montag (24.02.2026 = Dienstag)
        tuesday = datetime(2026, 2, 24, 12, 0)
        saturday = datetime(2026, 2, 28, 12, 0)

        assert tw.matches(tuesday) is True
        assert tw.matches(saturday) is False

    def test_overnight_window(self) -> None:
        """22:00 - 06:00 (über Mitternacht)."""
        tw = TimeWindow(
            start_time=time(22, 0),
            end_time=time(6, 0),
        )
        late_night = datetime(2026, 2, 24, 23, 30)
        early_morning = datetime(2026, 2, 25, 3, 0)
        afternoon = datetime(2026, 2, 24, 14, 0)

        assert tw.matches(late_night) is True
        assert tw.matches(early_morning) is True
        assert tw.matches(afternoon) is False

    def test_no_restrictions(self) -> None:
        tw = TimeWindow()
        now = datetime(2026, 2, 24, 12, 0)
        assert tw.matches(now) is True

    def test_combined_time_and_weekday(self) -> None:
        tw = TimeWindow(
            start_time=time(9, 0),
            end_time=time(17, 0),
            weekdays=[Weekday.MONDAY, Weekday.WEDNESDAY, Weekday.FRIDAY],
        )
        # Mittwoch 12:00 → Match
        wednesday_noon = datetime(2026, 2, 25, 12, 0)
        # Dienstag 12:00 → No Match (falscher Tag)
        tuesday_noon = datetime(2026, 2, 24, 12, 0)
        # Mittwoch 20:00 → No Match (falsche Zeit)
        wednesday_evening = datetime(2026, 2, 25, 20, 0)

        assert tw.matches(wednesday_noon) is True
        assert tw.matches(tuesday_noon) is False
        assert tw.matches(wednesday_evening) is False


class TestTimeWindowInBinding:
    """TimeWindow als Binding-Bedingung."""

    def test_binding_with_time_window(self) -> None:
        binding = MessageBinding(
            name="business_hours",
            target_agent="support",
            time_windows=[TimeWindow(
                start_time=time(8, 0),
                end_time=time(18, 0),
                weekdays=[Weekday.MONDAY, Weekday.TUESDAY, Weekday.WEDNESDAY,
                          Weekday.THURSDAY, Weekday.FRIDAY],
            )],
        )
        # Dienstag 10:00
        ctx_match = MessageContext(
            text="Hilfe",
            timestamp=datetime(2026, 2, 24, 10, 0),
        )
        # Samstag 10:00
        ctx_weekend = MessageContext(
            text="Hilfe",
            timestamp=datetime(2026, 2, 28, 10, 0),
        )

        assert binding.evaluate(ctx_match) == BindingMatchResult.MATCH
        assert binding.evaluate(ctx_weekend) == BindingMatchResult.NO_MATCH

    def test_multiple_time_windows(self) -> None:
        """Mehrere Zeitfenster: eins muss matchen."""
        binding = MessageBinding(
            name="split_schedule",
            target_agent="agent",
            time_windows=[
                TimeWindow(start_time=time(8, 0), end_time=time(12, 0)),
                TimeWindow(start_time=time(14, 0), end_time=time(18, 0)),
            ],
        )
        morning = MessageContext(text="X", timestamp=datetime(2026, 2, 24, 10, 0))
        lunch = MessageContext(text="X", timestamp=datetime(2026, 2, 24, 13, 0))
        afternoon = MessageContext(text="X", timestamp=datetime(2026, 2, 24, 15, 0))

        assert binding.evaluate(morning) == BindingMatchResult.MATCH
        assert binding.evaluate(lunch) == BindingMatchResult.NO_MATCH
        assert binding.evaluate(afternoon) == BindingMatchResult.MATCH


# ============================================================================
# AND-Verknüpfung und Negation
# ============================================================================


class TestCombinedConditions:
    """Mehrere Bedingungen AND-verknüpft."""

    def test_channel_plus_user(self) -> None:
        binding = MessageBinding(
            name="vip_telegram",
            target_agent="premium",
            channels=["telegram"],
            user_ids=["boss"],
        )
        # Beides matcht
        ctx_match = MessageContext(text="X", channel="telegram", user_id="boss")
        # Nur Channel matcht
        ctx_wrong_user = MessageContext(text="X", channel="telegram", user_id="intern")
        # Nur User matcht
        ctx_wrong_channel = MessageContext(text="X", channel="cli", user_id="boss")

        assert binding.evaluate(ctx_match) == BindingMatchResult.MATCH
        assert binding.evaluate(ctx_wrong_user) == BindingMatchResult.NO_MATCH
        assert binding.evaluate(ctx_wrong_channel) == BindingMatchResult.NO_MATCH

    def test_command_plus_channel(self) -> None:
        binding = MessageBinding(
            name="tg_code",
            target_agent="coder",
            channels=["telegram"],
            command_prefixes=["/code"],
        )
        ctx_match = MessageContext(text="/code Test", channel="telegram")
        ctx_wrong_channel = MessageContext(text="/code Test", channel="cli")
        ctx_no_cmd = MessageContext(text="Code schreiben", channel="telegram")

        assert binding.evaluate(ctx_match) == BindingMatchResult.MATCH
        assert binding.evaluate(ctx_wrong_channel) == BindingMatchResult.NO_MATCH
        assert binding.evaluate(ctx_no_cmd) == BindingMatchResult.NO_MATCH

    def test_triple_condition(self) -> None:
        binding = MessageBinding(
            name="strict",
            target_agent="special",
            channels=["api"],
            user_ids=["service_account"],
            message_patterns=[r"^URGENT:"],
        )
        ctx = MessageContext(
            text="URGENT: Server down",
            channel="api",
            user_id="service_account",
        )
        assert binding.evaluate(ctx) == BindingMatchResult.MATCH


class TestNegation:
    """NOT-Logik für Bindings."""

    def test_negate_match_becomes_no_match(self) -> None:
        binding = MessageBinding(
            name="not_telegram",
            target_agent="default",
            channels=["telegram"],
            negate=True,
        )
        # Telegram → eigentlich Match, negiert → NO_MATCH
        ctx_tg = MessageContext(text="X", channel="telegram")
        assert binding.evaluate(ctx_tg) == BindingMatchResult.NO_MATCH

    def test_negate_no_match_becomes_match(self) -> None:
        binding = MessageBinding(
            name="not_telegram",
            target_agent="default",
            channels=["telegram"],
            negate=True,
        )
        # CLI → eigentlich NO_MATCH, negiert → MATCH
        ctx_cli = MessageContext(text="X", channel="cli")
        assert binding.evaluate(ctx_cli) == BindingMatchResult.MATCH


class TestDisabledBinding:
    """Deaktivierte Bindings."""

    def test_disabled_returns_disabled(self) -> None:
        binding = MessageBinding(
            name="off",
            target_agent="agent",
            enabled=False,
        )
        ctx = MessageContext(text="Test")
        assert binding.evaluate(ctx) == BindingMatchResult.DISABLED


# ============================================================================
# BindingEngine
# ============================================================================


class TestBindingEngine:
    """Engine-Level Tests: Verwaltung und Auswertung."""

    def test_empty_engine(self, engine: BindingEngine) -> None:
        ctx = MessageContext(text="Test")
        assert engine.evaluate(ctx) is None
        assert engine.binding_count == 0

    def test_add_and_evaluate(self, engine: BindingEngine) -> None:
        engine.add_binding(MessageBinding(
            name="catch_all",
            target_agent="jarvis",
        ))
        ctx = MessageContext(text="Test")
        match = engine.evaluate(ctx)
        assert match is not None
        assert match.target_agent == "jarvis"

    def test_priority_ordering(self, engine: BindingEngine) -> None:
        engine.add_bindings([
            MessageBinding(name="low", target_agent="low_agent", priority=50),
            MessageBinding(name="high", target_agent="high_agent", priority=200),
            MessageBinding(name="mid", target_agent="mid_agent", priority=100),
        ])
        # Alle matchen (keine Bedingungen), aber high hat höchste Prio
        ctx = MessageContext(text="Test")
        match = engine.evaluate(ctx)
        assert match is not None
        assert match.target_agent == "high_agent"

    def test_same_priority_alphabetical(self, engine: BindingEngine) -> None:
        engine.add_bindings([
            MessageBinding(name="beta", target_agent="agent_b", priority=100),
            MessageBinding(name="alpha", target_agent="agent_a", priority=100),
        ])
        ctx = MessageContext(text="Test")
        match = engine.evaluate(ctx)
        assert match is not None
        assert match.target_agent == "agent_a"  # Alpha vor Beta

    def test_first_match_wins(self, engine: BindingEngine) -> None:
        engine.add_bindings([
            MessageBinding(
                name="specific",
                target_agent="coder",
                priority=200,
                command_prefixes=["/code"],
            ),
            MessageBinding(
                name="general",
                target_agent="jarvis",
                priority=100,
            ),
        ])
        ctx_code = MessageContext(text="/code Test")
        ctx_normal = MessageContext(text="Normal")

        match_code = engine.evaluate(ctx_code)
        match_normal = engine.evaluate(ctx_normal)

        assert match_code.target_agent == "coder"
        assert match_normal.target_agent == "jarvis"

    def test_remove_binding(self, engine: BindingEngine) -> None:
        engine.add_binding(MessageBinding(name="test", target_agent="x"))
        assert engine.binding_count == 1
        assert engine.remove_binding("test") is True
        assert engine.binding_count == 0

    def test_remove_nonexistent(self, engine: BindingEngine) -> None:
        assert engine.remove_binding("ghost") is False

    def test_enable_disable(self, engine: BindingEngine) -> None:
        engine.add_binding(MessageBinding(name="toggleable", target_agent="x"))
        assert engine.active_count == 1

        engine.disable_binding("toggleable")
        assert engine.active_count == 0

        ctx = MessageContext(text="Test")
        assert engine.evaluate(ctx) is None

        engine.enable_binding("toggleable")
        assert engine.active_count == 1
        assert engine.evaluate(ctx) is not None

    def test_clear(self, engine: BindingEngine) -> None:
        engine.add_bindings([
            MessageBinding(name="a", target_agent="x"),
            MessageBinding(name="b", target_agent="y"),
        ])
        assert engine.binding_count == 2
        engine.clear()
        assert engine.binding_count == 0

    def test_list_bindings_sorted(self, engine: BindingEngine) -> None:
        engine.add_bindings([
            MessageBinding(name="low", target_agent="x", priority=50),
            MessageBinding(name="high", target_agent="y", priority=200),
        ])
        bindings = engine.list_bindings()
        assert bindings[0].name == "high"
        assert bindings[1].name == "low"

    def test_get_binding(self, engine: BindingEngine) -> None:
        engine.add_binding(MessageBinding(name="findme", target_agent="x"))
        assert engine.get_binding("findme") is not None
        assert engine.get_binding("ghost") is None


class TestEvaluateAll:
    """evaluate_all() für Debugging."""

    def test_returns_all_results(self, engine: BindingEngine) -> None:
        engine.add_bindings([
            MessageBinding(name="a", target_agent="x", channels=["telegram"]),
            MessageBinding(name="b", target_agent="y"),
        ])
        ctx = MessageContext(text="Test", channel="cli")
        results = engine.evaluate_all(ctx)

        assert len(results) == 2
        assert results[0].result == BindingMatchResult.NO_MATCH  # Channel mismatch
        assert results[1].result == BindingMatchResult.MATCH


class TestStats:
    """Engine-Statistiken."""

    def test_stats(self, engine: BindingEngine) -> None:
        engine.add_bindings([
            MessageBinding(name="a", target_agent="coder"),
            MessageBinding(name="b", target_agent="coder"),
            MessageBinding(name="c", target_agent="organizer", enabled=False),
        ])
        stats = engine.stats()
        assert stats["total_bindings"] == 3
        assert stats["active_bindings"] == 2
        assert stats["agent_distribution"]["coder"] == 2
        assert stats["agent_distribution"]["organizer"] == 1


# ============================================================================
# YAML Persistenz
# ============================================================================


class TestYAMLPersistence:
    """Laden und Speichern von Bindings als YAML."""

    def test_save_and_load(self, engine: BindingEngine, tmp_path: Path) -> None:
        engine.add_bindings([
            MessageBinding(
                name="telegram_code",
                target_agent="coder",
                priority=200,
                channels=["telegram"],
                command_prefixes=["/code", "/shell"],
            ),
            MessageBinding(
                name="vip",
                target_agent="premium",
                priority=150,
                user_ids=["boss"],
            ),
        ])
        yaml_path = tmp_path / "bindings.yaml"
        engine.save_yaml(yaml_path)

        assert yaml_path.exists()

        # Neu laden
        loaded = BindingEngine.from_yaml(yaml_path)
        assert loaded.binding_count == 2

        # Funktional identisch
        ctx = MessageContext(text="/code Test", channel="telegram")
        match = loaded.evaluate(ctx)
        assert match is not None
        assert match.target_agent == "coder"

    def test_save_with_time_windows(self, engine: BindingEngine, tmp_path: Path) -> None:
        engine.add_binding(MessageBinding(
            name="business",
            target_agent="support",
            time_windows=[TimeWindow(
                start_time=time(8, 0),
                end_time=time(18, 0),
                weekdays=[Weekday.MONDAY, Weekday.FRIDAY],
            )],
        ))
        yaml_path = tmp_path / "bindings.yaml"
        engine.save_yaml(yaml_path)

        loaded = BindingEngine.from_yaml(yaml_path)
        assert loaded.binding_count == 1

        binding = loaded.list_bindings()[0]
        assert binding.time_windows is not None
        assert len(binding.time_windows) == 1
        tw = binding.time_windows[0]
        assert tw.start_time == time(8, 0)
        assert tw.end_time == time(18, 0)
        assert len(tw.weekdays) == 2

    def test_load_nonexistent_file(self, tmp_path: Path) -> None:
        loaded = BindingEngine.from_yaml(tmp_path / "ghost.yaml")
        assert loaded.binding_count == 0

    def test_load_invalid_yaml(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "bad.yaml"
        bad_file.write_text("{{invalid yaml", encoding="utf-8")
        loaded = BindingEngine.from_yaml(bad_file)
        assert loaded.binding_count == 0

    def test_load_empty_yaml(self, tmp_path: Path) -> None:
        empty_file = tmp_path / "empty.yaml"
        empty_file.write_text("", encoding="utf-8")
        loaded = BindingEngine.from_yaml(empty_file)
        assert loaded.binding_count == 0

    def test_save_with_metadata_conditions(self, engine: BindingEngine, tmp_path: Path) -> None:
        engine.add_binding(MessageBinding(
            name="api_v2",
            target_agent="v2_handler",
            metadata_conditions={"api_version": "2", "auth": "token"},
        ))
        yaml_path = tmp_path / "bindings.yaml"
        engine.save_yaml(yaml_path)

        loaded = BindingEngine.from_yaml(yaml_path)
        binding = loaded.list_bindings()[0]
        assert binding.metadata_conditions == {"api_version": "2", "auth": "token"}


# ============================================================================
# Factory-Funktionen
# ============================================================================


class TestFactoryFunctions:
    """Convenience-Funktionen für häufige Binding-Patterns."""

    def test_channel_binding(self) -> None:
        b = channel_binding("tg", "organizer", ["telegram"])
        ctx = MessageContext(text="Test", channel="telegram")
        assert b.evaluate(ctx) == BindingMatchResult.MATCH
        assert b.priority == 100

    def test_command_binding(self) -> None:
        b = command_binding("code_cmd", "coder", ["code", "shell"])
        ctx = MessageContext(text="/code Test")
        assert b.evaluate(ctx) == BindingMatchResult.MATCH
        assert b.priority == 200

    def test_command_binding_normalizes(self) -> None:
        b = command_binding("x", "y", ["code"])  # Ohne /
        assert b.command_prefixes == ["/code"]

    def test_user_binding(self) -> None:
        b = user_binding("vip", "premium", ["alex"])
        ctx = MessageContext(text="Test", user_id="alex")
        assert b.evaluate(ctx) == BindingMatchResult.MATCH

    def test_regex_binding(self) -> None:
        b = regex_binding("ins", "tarif_berater", [r"versicherung|police"])
        ctx = MessageContext(text="Meine Police kündigen")
        assert b.evaluate(ctx) == BindingMatchResult.MATCH

    def test_schedule_binding(self) -> None:
        b = schedule_binding(
            "biz",
            "support",
            start="09:00",
            end="17:00",
            weekdays=["mo", "di", "mi", "do", "fr"],
        )
        # Dienstag 12:00
        ctx = MessageContext(text="X", timestamp=datetime(2026, 2, 24, 12, 0))
        assert b.evaluate(ctx) == BindingMatchResult.MATCH


# ============================================================================
# Integration mit AgentRouter
# ============================================================================


class TestAgentRouterIntegration:
    """Bindings werden VOR Keyword-Routing ausgewertet."""

    def test_binding_overrides_keywords(self) -> None:
        from jarvis.core.agent_router import AgentProfile, AgentRouter

        router = AgentRouter()
        router.initialize(custom_agents=[
            AgentProfile(
                name="keyword_agent",
                trigger_keywords=["versicherung"],
            ),
            AgentProfile(
                name="binding_agent",
            ),
        ])

        # Binding mit höchster Priorität
        router.bindings.add_binding(MessageBinding(
            name="force_binding",
            target_agent="binding_agent",
            message_patterns=[r"versicherung"],
            priority=200,
        ))

        # "versicherung" matcht sowohl Keyword als auch Binding
        # → Binding gewinnt (confidence 1.0)
        decision = router.route("Meine Versicherung kündigen")
        assert decision.agent.name == "binding_agent"
        assert decision.confidence == 1.0
        assert "binding:force_binding" in decision.matched_patterns

    def test_no_binding_falls_to_keywords(self) -> None:
        from jarvis.core.agent_router import AgentProfile, AgentRouter

        router = AgentRouter()
        router.initialize(custom_agents=[
            AgentProfile(
                name="keyword_agent",
                trigger_keywords=["versicherung"],
            ),
        ])
        # Keine Bindings → Keyword-Routing greift
        decision = router.route("Meine Versicherung kündigen")
        assert decision.agent.name == "keyword_agent"

    def test_binding_with_message_context(self) -> None:
        from jarvis.core.agent_router import AgentProfile, AgentRouter

        router = AgentRouter()
        router.initialize(custom_agents=[
            AgentProfile(name="telegram_bot"),
        ])
        router.bindings.add_binding(channel_binding(
            "tg_route",
            "telegram_bot",
            ["telegram"],
        ))

        ctx = MessageContext(text="Hallo", channel="telegram")
        decision = router.route("Hallo", context=ctx)
        assert decision.agent.name == "telegram_bot"

    def test_binding_target_not_found_falls_through(self) -> None:
        from jarvis.core.agent_router import AgentRouter

        router = AgentRouter()
        router.initialize()

        # Binding zeigt auf nicht-existierenden Agenten
        router.bindings.add_binding(MessageBinding(
            name="ghost",
            target_agent="nonexistent_agent",
        ))

        # Sollte graceful auf Default fallen
        decision = router.route("Test")
        assert decision.agent.name == "jarvis"

    def test_router_stats_include_bindings(self) -> None:
        from jarvis.core.agent_router import AgentRouter

        router = AgentRouter()
        router.initialize()
        router.bindings.add_binding(MessageBinding(
            name="test",
            target_agent="jarvis",
        ))

        stats = router.stats()
        assert "bindings" in stats
        assert stats["bindings"]["total_bindings"] == 1

    def test_from_yaml_loads_bindings(self, tmp_path: Path) -> None:
        from jarvis.core.agent_router import AgentProfile, AgentRouter

        import yaml

        # agents.yaml
        agents_path = tmp_path / "agents.yaml"
        agents_path.write_text(yaml.dump({
            "agents": [{"name": "coder", "trigger_keywords": ["code"]}],
        }), encoding="utf-8")

        # bindings.yaml im selben Verzeichnis
        bindings_path = tmp_path / "bindings.yaml"
        bindings_path.write_text(yaml.dump({
            "bindings": [{
                "name": "force_coder",
                "target_agent": "coder",
                "command_prefixes": ["/code"],
                "priority": 200,
            }],
        }), encoding="utf-8")

        router = AgentRouter.from_yaml(agents_path)
        assert router.bindings.binding_count == 1

        ctx = MessageContext(text="/code Hallo")
        decision = router.route("/code Hallo", context=ctx)
        assert decision.agent.name == "coder"
        assert decision.confidence == 1.0


# ============================================================================
# Realwelt-Szenarien
# ============================================================================


class TestRealWorldScenarios:
    """Praxisnahe Konfigurationen."""

    def test_insurance_broker_setup(self, engine: BindingEngine) -> None:
        """Alexander's Versicherungsmakler-Setup."""
        engine.add_bindings([
            # Slash-Commands
            command_binding("cmd_tarif", "tarif_berater", ["/tarif", "/bu", "/versicherung"]),
            command_binding("cmd_code", "coder", ["/code", "/shell", "/dev"]),
            command_binding("cmd_orga", "organizer", ["/kalender", "/todo", "/brief"]),

            # Regex für Versicherungsthemen
            regex_binding(
                "insurance_topics",
                "tarif_berater",
                [r"bu[- ]?tarif", r"berufsunfähigkeit", r"police\s+\d+",
                 r"versicherung(s|en)?", r"beitrag(s|shöhe)?"],
                priority=170,
            ),

            # API-Anfragen an Coder
            MessageBinding(
                name="api_requests",
                target_agent="coder",
                channels=["api"],
                priority=150,
            ),

            # Telegram-Organizer für Geschäftszeiten
            MessageBinding(
                name="business_hours_telegram",
                target_agent="organizer",
                channels=["telegram"],
                priority=50,
                time_windows=[TimeWindow(
                    start_time=time(7, 0),
                    end_time=time(19, 0),
                    weekdays=[Weekday.MONDAY, Weekday.TUESDAY, Weekday.WEDNESDAY,
                              Weekday.THURSDAY, Weekday.FRIDAY],
                )],
            ),
        ])

        # /tarif → tarif_berater (Command-Binding, Prio 200)
        ctx1 = MessageContext(text="/tarif BU-Vergleich", channel="telegram")
        assert engine.evaluate(ctx1).target_agent == "tarif_berater"

        # "BU-Tarif vergleichen" → tarif_berater (Regex-Binding, Prio 170)
        ctx2 = MessageContext(text="Vergleiche die BU-Tarife", channel="telegram")
        assert engine.evaluate(ctx2).target_agent == "tarif_berater"

        # API-Request → Coder (Channel-Binding, Prio 150)
        ctx3 = MessageContext(text="Deploy the new feature", channel="api")
        assert engine.evaluate(ctx3).target_agent == "coder"

        # Telegram während Geschäftszeiten ohne Command → Organizer
        ctx4 = MessageContext(
            text="Was steht heute an?",
            channel="telegram",
            timestamp=datetime(2026, 2, 24, 10, 0),  # Dienstag 10:00
        )
        assert engine.evaluate(ctx4).target_agent == "organizer"

        # Telegram am Wochenende → kein Match (kein Catch-All)
        ctx5 = MessageContext(
            text="Was steht heute an?",
            channel="telegram",
            timestamp=datetime(2026, 2, 28, 10, 0),  # Samstag
        )
        assert engine.evaluate(ctx5) is None  # Fallback auf Keyword-Routing

    def test_multi_tenant_setup(self, engine: BindingEngine) -> None:
        """Multi-User mit verschiedenen Agenten."""
        engine.add_bindings([
            user_binding("alex_premium", "premium_assistant", ["alex"]),
            user_binding("team_support", "team_agent", ["member1", "member2"]),
            MessageBinding(name="default", target_agent="jarvis", priority=10),
        ])

        ctx_alex = MessageContext(text="X", user_id="alex")
        ctx_team = MessageContext(text="X", user_id="member1")
        ctx_stranger = MessageContext(text="X", user_id="unknown")

        assert engine.evaluate(ctx_alex).target_agent == "premium_assistant"
        assert engine.evaluate(ctx_team).target_agent == "team_agent"
        assert engine.evaluate(ctx_stranger).target_agent == "jarvis"

    def test_deterministic_always_same_result(self, engine: BindingEngine) -> None:
        """Gleiches Input → immer gleiches Output."""
        engine.add_bindings([
            channel_binding("a", "agent_a", ["telegram"], priority=100),
            regex_binding("b", "agent_b", [r"urgent"], priority=150),
        ])

        ctx = MessageContext(text="urgent request", channel="telegram")

        # 100 Mal auswerten → immer agent_b (höhere Prio)
        results = [engine.evaluate(ctx).target_agent for _ in range(100)]
        assert all(r == "agent_b" for r in results)
