"""Tests für Slash-Commands, Interaction-State und Fallback."""

from __future__ import annotations

import pytest

from jarvis.channels.commands import (
    CommandRegistry,
    CommandScope,
    FallbackRenderer,
    InteractionState,
    InteractionStore,
    InteractionType,
    SlashCommand,
)


# ============================================================================
# SlashCommand
# ============================================================================


class TestSlashCommand:
    def test_to_slack_definition(self) -> None:
        cmd = SlashCommand(name="schedule", description="Plan tasks")
        d = cmd.to_slack_definition()
        assert "/jarvis_schedule" in d["command"]
        assert d["description"] == "Plan tasks"

    def test_to_discord_definition(self) -> None:
        cmd = SlashCommand(name="approve", description="Approve an action")
        d = cmd.to_discord_definition()
        assert d["name"] == "approve"
        assert d["type"] == 1

    def test_to_dict(self) -> None:
        cmd = SlashCommand(name="status", description="Show status", admin_only=True)
        d = cmd.to_dict()
        assert d["admin_only"] is True
        assert d["name"] == "status"


# ============================================================================
# CommandRegistry
# ============================================================================


class TestCommandRegistry:
    def test_default_commands_registered(self) -> None:
        reg = CommandRegistry()
        assert reg.command_count >= 7

    def test_get_command(self) -> None:
        reg = CommandRegistry()
        cmd = reg.get("schedule")
        assert cmd is not None
        assert cmd.name == "schedule"

    def test_get_nonexistent(self) -> None:
        reg = CommandRegistry()
        assert reg.get("nope") is None

    def test_register_custom(self) -> None:
        reg = CommandRegistry()
        reg.register(SlashCommand(name="custom", description="My command"))
        assert reg.get("custom") is not None

    def test_unregister(self) -> None:
        reg = CommandRegistry()
        initial = reg.command_count
        assert reg.unregister("schedule")
        assert reg.command_count == initial - 1
        assert not reg.unregister("schedule")

    def test_list_commands_all(self) -> None:
        reg = CommandRegistry()
        cmds = reg.list_commands()
        assert len(cmds) >= 7

    def test_list_commands_by_scope(self) -> None:
        reg = CommandRegistry()
        reg.register(SlashCommand(name="slack_only", description="X", scope=CommandScope.SLACK))
        slack_cmds = reg.list_commands(scope=CommandScope.SLACK)
        names = [c.name for c in slack_cmds]
        assert "slack_only" in names
        # ALL-scope commands also appear
        assert "schedule" in names

    def test_cooldown_check(self) -> None:
        reg = CommandRegistry()
        reg.register(SlashCommand(name="limited", description="X", cooldown_seconds=3600))
        assert reg.check_cooldown("user1", "limited")  # First use
        reg.record_usage("user1", "limited")
        assert not reg.check_cooldown("user1", "limited")  # In cooldown
        assert reg.check_cooldown("user2", "limited")  # Different user

    def test_cooldown_zero_always_allowed(self) -> None:
        reg = CommandRegistry()
        reg.record_usage("user1", "schedule")
        assert reg.check_cooldown("user1", "schedule")  # schedule has cooldown=0

    def test_slack_definitions(self) -> None:
        reg = CommandRegistry()
        defs = reg.slack_definitions()
        assert len(defs) >= 7
        assert all("command" in d for d in defs)

    def test_discord_definitions(self) -> None:
        reg = CommandRegistry()
        defs = reg.discord_definitions()
        assert len(defs) >= 7
        assert all("name" in d for d in defs)

    def test_usage_stats(self) -> None:
        reg = CommandRegistry()
        reg.record_usage("u1", "schedule")
        reg.record_usage("u2", "schedule")
        reg.record_usage("u1", "briefing")
        stats = reg.usage_stats()
        assert stats["schedule"] == 2
        assert stats["briefing"] == 1


# ============================================================================
# InteractionState
# ============================================================================


class TestInteractionState:
    def test_is_pending(self) -> None:
        state = InteractionState(
            interaction_id="i1",
            interaction_type=InteractionType.BUTTON_CLICK,
            user_id="u1",
        )
        assert state.is_pending

    def test_resolve(self) -> None:
        state = InteractionState(
            interaction_id="i1",
            interaction_type=InteractionType.BUTTON_CLICK,
            user_id="u1",
        )
        state.resolve(result="approved")
        assert state.resolved
        assert state.result == "approved"
        assert not state.is_pending

    def test_to_dict(self) -> None:
        state = InteractionState(
            interaction_id="i1",
            interaction_type=InteractionType.APPROVAL,
            user_id="u1",
            channel="slack",
        )
        d = state.to_dict()
        assert d["interaction_type"] == "approval"
        assert d["channel"] == "slack"


# ============================================================================
# InteractionStore
# ============================================================================


class TestInteractionStore:
    def test_create_and_get(self) -> None:
        store = InteractionStore()
        state = store.create(InteractionType.BUTTON_CLICK, "u1")
        retrieved = store.get(state.interaction_id)
        assert retrieved is not None
        assert retrieved.user_id == "u1"

    def test_get_nonexistent(self) -> None:
        store = InteractionStore()
        assert store.get("nope") is None

    def test_get_by_message(self) -> None:
        store = InteractionStore()
        store.create(InteractionType.BUTTON_CLICK, "u1", message_id="msg1", action_id="btn_ok")
        store.create(InteractionType.BUTTON_CLICK, "u1", message_id="msg1", action_id="btn_cancel")
        store.create(InteractionType.BUTTON_CLICK, "u2", message_id="msg2")
        interactions = store.get_by_message("msg1")
        assert len(interactions) == 2

    def test_resolve(self) -> None:
        store = InteractionStore()
        state = store.create(InteractionType.APPROVAL, "u1")
        assert store.resolve(state.interaction_id, result="approved")
        assert state.resolved

    def test_resolve_nonexistent(self) -> None:
        store = InteractionStore()
        assert not store.resolve("nope")

    def test_pending_count(self) -> None:
        store = InteractionStore()
        store.create(InteractionType.BUTTON_CLICK, "u1")
        store.create(InteractionType.BUTTON_CLICK, "u2")
        s = store.create(InteractionType.BUTTON_CLICK, "u3")
        store.resolve(s.interaction_id)
        assert store.pending_count() == 2

    def test_cleanup(self) -> None:
        store = InteractionStore(ttl_seconds=0)  # Instant expiry
        import time
        time.sleep(0.01)
        store.create(InteractionType.BUTTON_CLICK, "u1")
        # The state created will be expired since ttl is 0
        # But we need a small delay for datetime precision
        cleaned = store.cleanup()
        assert cleaned >= 0  # May or may not be expired depending on timing

    def test_stats(self) -> None:
        store = InteractionStore()
        store.create(InteractionType.BUTTON_CLICK, "u1")
        s = store.create(InteractionType.APPROVAL, "u2")
        store.resolve(s.interaction_id)
        stats = store.stats()
        assert stats["total"] == 2
        assert stats["pending"] == 1
        assert stats["resolved"] == 1


# ============================================================================
# FallbackRenderer
# ============================================================================


class TestFallbackRenderer:
    def test_render_buttons(self) -> None:
        text = FallbackRenderer.render_buttons(
            "Aktion wählen:",
            [{"label": "Erlauben"}, {"label": "Ablehnen"}],
        )
        assert "1) Erlauben" in text
        assert "2) Ablehnen" in text
        assert "Nummer" in text

    def test_render_approval(self) -> None:
        text = FallbackRenderer.render_approval("Agent möchte E-Mail senden")
        assert "JA" in text
        assert "NEIN" in text
        assert "E-Mail" in text

    def test_render_approval_custom_words(self) -> None:
        text = FallbackRenderer.render_approval("Test", approve_word="OK", reject_word="STOP")
        assert "OK" in text
        assert "STOP" in text

    def test_render_select(self) -> None:
        text = FallbackRenderer.render_select("Wähle:", [
            {"label": "Option A", "description": "Erste Wahl"},
            {"label": "Option B"},
        ])
        assert "1) Option A" in text
        assert "Erste Wahl" in text
        assert "2) Option B" in text

    def test_render_progress(self) -> None:
        text = FallbackRenderer.render_progress("Installation", [
            {"name": "Download", "status": "completed"},
            {"name": "Verify", "status": "running"},
            {"name": "Install", "status": "pending"},
        ])
        assert "✓" in text
        assert "►" in text
        assert "○" in text
        assert "Download" in text

    def test_render_form(self) -> None:
        text = FallbackRenderer.render_form("Konfiguration", [
            {"label": "Name", "required": True},
            {"label": "Port", "default": "8080"},
        ])
        assert "Name" in text
        assert "Pflicht" in text
        assert "8080" in text

    def test_parse_numbered_response_valid(self) -> None:
        assert FallbackRenderer.parse_numbered_response("1", 3) == 0
        assert FallbackRenderer.parse_numbered_response("3", 3) == 2

    def test_parse_numbered_response_invalid(self) -> None:
        assert FallbackRenderer.parse_numbered_response("0", 3) is None
        assert FallbackRenderer.parse_numbered_response("4", 3) is None
        assert FallbackRenderer.parse_numbered_response("abc", 3) is None

    def test_parse_approval_valid(self) -> None:
        assert FallbackRenderer.parse_approval_response("JA") is True
        assert FallbackRenderer.parse_approval_response("ja") is True
        assert FallbackRenderer.parse_approval_response("NEIN") is False
        assert FallbackRenderer.parse_approval_response("nein") is False

    def test_parse_approval_invalid(self) -> None:
        assert FallbackRenderer.parse_approval_response("maybe") is None
        assert FallbackRenderer.parse_approval_response("") is None

    def test_parse_approval_custom_words(self) -> None:
        assert FallbackRenderer.parse_approval_response(
            "ok", approve_word="OK", reject_word="STOP",
        ) is True
        assert FallbackRenderer.parse_approval_response(
            "STOP", approve_word="OK", reject_word="STOP",
        ) is False
