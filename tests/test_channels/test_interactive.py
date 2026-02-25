"""Tests für interaktive UI-Komponenten.

Testet: SlackMessageBuilder, DiscordMessageBuilder,
FormField, ProgressTracker, AdaptiveCard.
"""

from __future__ import annotations

import pytest

from jarvis.channels.interactive import (
    AdaptiveCard,
    ButtonStyle,
    DiscordColor,
    DiscordMessageBuilder,
    FallbackRenderer,
    FieldType,
    FormField,
    InteractionStateStore,
    ModalHandler,
    ModalSubmission,
    ProgressStep,
    ProgressTracker,
    SignatureVerifier,
    SlackMessageBuilder,
    SlashCommand,
    SlashCommandRegistry,
)


# ============================================================================
# SlackMessageBuilder
# ============================================================================


class TestSlackMessageBuilder:
    def test_empty_build(self) -> None:
        msg = SlackMessageBuilder().build()
        assert msg["blocks"] == []

    def test_text_and_section(self) -> None:
        msg = (
            SlackMessageBuilder()
            .text("Fallback")
            .section("*Hallo Welt*")
            .build()
        )
        assert msg["text"] == "Fallback"
        assert msg["blocks"][0]["type"] == "section"
        assert msg["blocks"][0]["text"]["type"] == "mrkdwn"

    def test_fields(self) -> None:
        msg = (
            SlackMessageBuilder()
            .fields([("Status", "✅ Online"), ("Version", "1.0")])
            .build()
        )
        assert len(msg["blocks"][0]["fields"]) == 2

    def test_header(self) -> None:
        msg = SlackMessageBuilder().header("Dashboard").build()
        assert msg["blocks"][0]["type"] == "header"

    def test_divider(self) -> None:
        msg = SlackMessageBuilder().divider().build()
        assert msg["blocks"][0]["type"] == "divider"

    def test_context(self) -> None:
        msg = SlackMessageBuilder().context(["Info 1", "Info 2"]).build()
        assert msg["blocks"][0]["type"] == "context"
        assert len(msg["blocks"][0]["elements"]) == 2

    def test_single_button(self) -> None:
        msg = (
            SlackMessageBuilder()
            .button("Klick", "btn_1", style=ButtonStyle.PRIMARY)
            .build()
        )
        actions = msg["blocks"][0]
        assert actions["type"] == "actions"
        assert actions["elements"][0]["style"] == "primary"
        assert actions["elements"][0]["action_id"] == "btn_1"

    def test_multiple_buttons_same_row(self) -> None:
        msg = (
            SlackMessageBuilder()
            .button("A", "a")
            .button("B", "b")
            .build()
        )
        assert len(msg["blocks"]) == 1  # Gleiche Actions-Row
        assert len(msg["blocks"][0]["elements"]) == 2

    def test_button_with_url(self) -> None:
        msg = (
            SlackMessageBuilder()
            .button("Öffnen", "open", url="https://example.com")
            .build()
        )
        assert msg["blocks"][0]["elements"][0]["url"] == "https://example.com"

    def test_image(self) -> None:
        msg = (
            SlackMessageBuilder()
            .image("https://img.example.com/x.png", "Bild", "Titel")
            .build()
        )
        assert msg["blocks"][0]["type"] == "image"

    def test_progress_bar(self) -> None:
        msg = SlackMessageBuilder().progress_bar(70, "Upload").build()
        text = msg["blocks"][0]["text"]["text"]
        assert "70%" in text
        assert "█" in text

    def test_build_modal(self) -> None:
        fields = [
            FormField(name="email", label="E-Mail", field_type=FieldType.EMAIL, required=True),
            FormField(name="plan", label="Tarif", field_type=FieldType.SELECT, options=[
                {"text": "Basis", "value": "basis"},
                {"text": "Premium", "value": "premium"},
            ]),
        ]
        modal = (
            SlackMessageBuilder()
            .section("Bitte ausfüllen:")
            .build_modal("Kontakt", "contact_form", form_fields=fields)
        )
        assert modal["type"] == "modal"
        assert modal["callback_id"] == "contact_form"
        assert len(modal["blocks"]) == 3  # 1 section + 2 fields

    def test_block_count(self) -> None:
        builder = SlackMessageBuilder().header("X").section("Y").divider()
        assert builder.block_count == 3

    def test_complex_message(self) -> None:
        msg = (
            SlackMessageBuilder()
            .text("Dashboard-Update")
            .header("📊 Jarvis Status")
            .fields([
                ("Agent", "jarvis (main)"),
                ("Uptime", "4h 23m"),
                ("Sessions", "12 aktiv"),
                ("Token", "45.2k / 100k"),
            ])
            .divider()
            .section("Letzte Aktion: *BU-Vergleich* für Kunde Müller")
            .button("Details", "show_details", style=ButtonStyle.PRIMARY)
            .button("Abbrechen", "cancel", style=ButtonStyle.DANGER)
            .context(["Aktualisiert vor 30 Sekunden"])
            .build()
        )
        assert len(msg["blocks"]) == 6  # header, fields, divider, section, actions, context


# ============================================================================
# DiscordMessageBuilder
# ============================================================================


class TestDiscordMessageBuilder:
    def test_content(self) -> None:
        msg = DiscordMessageBuilder().content("Hallo").build()
        assert msg["content"] == "Hallo"

    def test_embed(self) -> None:
        msg = (
            DiscordMessageBuilder()
            .embed("Titel", "Beschreibung", DiscordColor.SUCCESS)
            .build()
        )
        assert len(msg["embeds"]) == 1
        assert msg["embeds"][0]["title"] == "Titel"
        assert msg["embeds"][0]["color"] == 0x2ECC71

    def test_embed_fields(self) -> None:
        msg = (
            DiscordMessageBuilder()
            .embed("Status")
            .embed_field("Agent", "jarvis")
            .embed_field("Uptime", "4h", inline=False)
            .build()
        )
        fields = msg["embeds"][0]["fields"]
        assert len(fields) == 2
        assert fields[0]["inline"] is True
        assert fields[1]["inline"] is False

    def test_embed_footer(self) -> None:
        msg = (
            DiscordMessageBuilder()
            .embed("X")
            .embed_footer("Powered by Jarvis")
            .build()
        )
        assert msg["embeds"][0]["footer"]["text"] == "Powered by Jarvis"

    def test_embed_author(self) -> None:
        msg = (
            DiscordMessageBuilder()
            .embed("X")
            .embed_author("Jarvis Bot", icon_url="https://example.com/icon.png")
            .build()
        )
        assert msg["embeds"][0]["author"]["name"] == "Jarvis Bot"

    def test_embed_thumbnail(self) -> None:
        msg = (
            DiscordMessageBuilder()
            .embed("X")
            .embed_thumbnail("https://example.com/thumb.png")
            .build()
        )
        assert "thumbnail" in msg["embeds"][0]

    def test_multiple_embeds(self) -> None:
        msg = (
            DiscordMessageBuilder()
            .embed("Embed 1")
            .embed("Embed 2")
            .build()
        )
        assert len(msg["embeds"]) == 2

    def test_button(self) -> None:
        msg = (
            DiscordMessageBuilder()
            .button("Klick", "btn_1", ButtonStyle.PRIMARY)
            .build()
        )
        assert len(msg["components"]) == 1
        assert msg["components"][0]["type"] == 1  # ACTION_ROW
        btn = msg["components"][0]["components"][0]
        assert btn["type"] == 2  # BUTTON
        assert btn["label"] == "Klick"

    def test_buttons_grouped(self) -> None:
        msg = (
            DiscordMessageBuilder()
            .button("A", "a")
            .button("B", "b")
            .build()
        )
        assert len(msg["components"]) == 1  # Eine Action-Row
        assert len(msg["components"][0]["components"]) == 2

    def test_select_menu(self) -> None:
        msg = (
            DiscordMessageBuilder()
            .select_menu("select_tarif", "Tarif wählen", [
                {"text": "Basis", "value": "basis"},
                {"text": "Premium", "value": "premium"},
            ])
            .build()
        )
        assert msg["components"][0]["components"][0]["type"] == 3  # STRING_SELECT

    def test_progress_bar(self) -> None:
        msg = (
            DiscordMessageBuilder()
            .embed("Status")
            .progress_bar(60, "Fortschritt")
            .build()
        )
        desc = msg["embeds"][0]["description"]
        assert "60%" in desc
        assert "▓" in desc

    def test_build_modal(self) -> None:
        fields = [
            FormField(name="name", label="Name", required=True),
            FormField(name="plan", label="Tarif", field_type=FieldType.SELECT, options=[
                {"text": "Basis", "value": "basis"},
            ]),
        ]
        modal = DiscordMessageBuilder().build_modal("Formular", "form_1", fields)
        assert modal["title"] == "Formular"
        assert modal["custom_id"] == "form_1"
        assert len(modal["components"]) == 2

    def test_embed_count(self) -> None:
        builder = DiscordMessageBuilder().embed("A").embed("B")
        assert builder.embed_count == 2

    def test_button_with_url(self) -> None:
        msg = (
            DiscordMessageBuilder()
            .button("Link", "x", url="https://example.com")
            .build()
        )
        btn = msg["components"][0]["components"][0]
        assert btn["style"] == 5  # LINK style
        assert btn["url"] == "https://example.com"


# ============================================================================
# FormField
# ============================================================================


class TestFormField:
    def test_text_to_slack(self) -> None:
        f = FormField(name="email", label="E-Mail", field_type=FieldType.TEXT)
        block = f.to_slack_block()
        assert block["type"] == "input"
        assert block["element"]["type"] == "plain_text_input"

    def test_select_to_slack(self) -> None:
        f = FormField(
            name="plan", label="Plan", field_type=FieldType.SELECT,
            options=[{"text": "A", "value": "a"}],
        )
        block = f.to_slack_block()
        assert block["element"]["type"] == "static_select"

    def test_multiselect_to_slack(self) -> None:
        f = FormField(
            name="tags", label="Tags", field_type=FieldType.MULTI_SELECT,
            options=[{"text": "A", "value": "a"}, {"text": "B", "value": "b"}],
        )
        block = f.to_slack_block()
        assert block["element"]["type"] == "multi_static_select"

    def test_date_to_slack(self) -> None:
        f = FormField(name="date", label="Datum", field_type=FieldType.DATE)
        block = f.to_slack_block()
        assert block["element"]["type"] == "datepicker"

    def test_checkbox_to_slack(self) -> None:
        f = FormField(
            name="agree", label="Zustimmung", field_type=FieldType.CHECKBOX,
            options=[{"text": "Ja", "value": "yes"}],
        )
        block = f.to_slack_block()
        assert block["element"]["type"] == "checkboxes"

    def test_select_to_discord(self) -> None:
        f = FormField(
            name="plan", label="Plan", field_type=FieldType.SELECT,
            options=[{"text": "A", "value": "a"}],
        )
        comp = f.to_discord_component()
        assert comp["type"] == 3  # SELECT_MENU

    def test_text_to_discord(self) -> None:
        f = FormField(name="name", label="Name", field_type=FieldType.TEXT)
        comp = f.to_discord_component()
        assert comp["type"] == 4  # TEXT_INPUT

    def test_required_field(self) -> None:
        f = FormField(name="x", label="X", required=True)
        assert f.to_slack_block()["optional"] is False


# ============================================================================
# ProgressTracker
# ============================================================================


class TestProgressTracker:
    def test_create(self) -> None:
        pt = ProgressTracker("Import", ["Laden", "Parsen", "Speichern"])
        assert len(pt.steps) == 3
        assert pt.percent_complete == 0
        assert not pt.is_complete

    def test_step_lifecycle(self) -> None:
        pt = ProgressTracker("Job", ["Schritt 1", "Schritt 2"])
        step = pt.start_step()
        assert step is not None
        assert step.status == "running"

        pt.complete_step(message="Fertig")
        assert step.status == "completed"
        assert step.duration_ms >= 0
        assert pt.percent_complete == 50

    def test_fail_step(self) -> None:
        pt = ProgressTracker("Job", ["A", "B"])
        pt.start_step()
        pt.fail_step(error="Timeout")
        assert pt.steps[0].status == "failed"
        assert pt.has_failures

    def test_skip_step(self) -> None:
        pt = ProgressTracker("Job", ["A", "B", "C"])
        pt.skip_step(1)
        assert pt.steps[1].status == "skipped"
        assert pt.percent_complete == 33  # 1/3

    def test_complete_all(self) -> None:
        pt = ProgressTracker("Job", ["A", "B"])
        pt.start_step()
        pt.complete_step()
        pt.start_step()
        pt.complete_step()
        assert pt.is_complete
        assert pt.percent_complete == 100

    def test_to_slack_blocks(self) -> None:
        pt = ProgressTracker("Import", ["Laden", "Parsen"])
        pt.start_step()
        pt.complete_step()
        blocks = pt.to_slack_blocks()
        assert len(blocks) >= 3  # header + steps + progress bar

    def test_to_discord_embed(self) -> None:
        pt = ProgressTracker("Import", ["Laden", "Parsen"])
        pt.start_step()
        result = pt.to_discord_embed()
        assert "embeds" in result
        assert result["embeds"][0]["title"] == "Import"

    def test_status_emoji(self) -> None:
        step = ProgressStep(name="Test")
        assert step.status_emoji == "⏳"
        step.status = "running"
        assert step.status_emoji == "🔄"
        step.status = "completed"
        assert step.status_emoji == "✅"
        step.status = "failed"
        assert step.status_emoji == "❌"


# ============================================================================
# AdaptiveCard
# ============================================================================


class TestAdaptiveCard:
    def test_basic_card(self) -> None:
        card = AdaptiveCard("Status", "System läuft")
        card.add_field("Uptime", "4h 23m")
        card.add_field("Sessions", "12")
        card.add_button("Details", "show_details", ButtonStyle.PRIMARY)
        card.set_footer("Aktualisiert vor 30s")

        assert card.field_count == 2
        assert card.action_count == 1

    def test_to_slack(self) -> None:
        card = (
            AdaptiveCard("Alarm", "Agent down!")
            .add_field("Agent", "coder")
            .add_button("Neustart", "restart", ButtonStyle.DANGER)
            .set_color("error")
        )
        result = card.to_slack()
        assert "blocks" in result
        assert len(result["blocks"]) >= 3  # header, body, fields+button

    def test_to_discord(self) -> None:
        card = (
            AdaptiveCard("Status", "Alles OK")
            .add_field("Version", "1.0")
            .add_button("Details", "details")
            .set_color("success")
        )
        result = card.to_discord()
        assert "embeds" in result
        assert result["embeds"][0]["color"] == DiscordColor.SUCCESS.value

    def test_cross_platform_consistency(self) -> None:
        card = (
            AdaptiveCard("Test", "Body")
            .add_field("A", "1")
            .add_button("OK", "ok")
        )
        slack = card.to_slack()
        discord = card.to_discord()
        # Beide haben Content
        assert slack["blocks"]
        assert discord["embeds"]
        # Beide haben Buttons
        assert discord["components"]


# ============================================================================
# Slash-Commands
# ============================================================================


class TestSlashCommandRegistry:
    def test_register_command(self) -> None:
        reg = SlashCommandRegistry()
        cmd = reg.register("/jarvis", "Main command")
        assert cmd.name == "/jarvis"
        assert reg.command_count == 1

    def test_register_with_handler(self) -> None:
        reg = SlashCommandRegistry()
        def handler(payload: dict) -> dict:
            return {"text": "OK"}
        reg.register("/schedule", "Schedule task", handler)
        result = reg.dispatch("/schedule", {"text": "daily briefing"})
        assert result["text"] == "OK"

    def test_dispatch_unknown(self) -> None:
        reg = SlashCommandRegistry()
        result = reg.dispatch("/unknown", {})
        assert "error" in result

    def test_list_commands(self) -> None:
        reg = SlashCommandRegistry()
        reg.register("/approve", "Approve request")
        reg.register("/delegate", "Delegate task")
        assert len(reg.list_commands()) == 2

    def test_to_slack_manifest(self) -> None:
        reg = SlashCommandRegistry()
        reg.register("/jarvis", "Main")
        manifest = reg.to_slack_manifest()
        assert len(manifest) == 1
        assert manifest[0]["command"] == "/jarvis"

    def test_to_discord_commands(self) -> None:
        reg = SlashCommandRegistry()
        reg.register("/approve", "Approve")
        cmds = reg.to_discord_commands()
        assert len(cmds) == 1
        assert cmds[0]["name"] == "approve"
        assert cmds[0]["type"] == 1

    def test_handler_error_returns_error(self) -> None:
        reg = SlashCommandRegistry()
        def bad_handler(p: dict) -> dict:
            raise ValueError("broken")
        reg.register("/bad", "Bad", bad_handler)
        result = reg.dispatch("/bad", {})
        assert "error" in result

    def test_slash_command_to_dict(self) -> None:
        cmd = SlashCommand(
            name="/test", description="Test cmd",
            options=[{"name": "arg", "type": 3}],
        )
        d = cmd.to_discord()
        assert d["name"] == "test"
        assert len(d["options"]) == 1


# ============================================================================
# Modal Handler
# ============================================================================


class TestModalHandler:
    def test_register_and_handle(self) -> None:
        mh = ModalHandler()
        def config_handler(sub: ModalSubmission) -> dict:
            return {"ok": True, "values": sub.values}
        mh.register("config_modal", config_handler)
        sub = ModalSubmission(
            callback_id="config_modal", user_id="u1",
            channel="slack", values={"name": "test"},
        )
        result = mh.handle(sub)
        assert result["ok"] is True

    def test_handle_unknown_callback(self) -> None:
        mh = ModalHandler()
        sub = ModalSubmission(callback_id="unknown", user_id="u1", channel="slack")
        result = mh.handle(sub)
        assert "error" in result

    def test_has_handler(self) -> None:
        mh = ModalHandler()
        mh.register("test", lambda s: {})
        assert mh.has_handler("test")
        assert not mh.has_handler("nope")

    def test_handler_count(self) -> None:
        mh = ModalHandler()
        mh.register("a", lambda s: {})
        mh.register("b", lambda s: {})
        assert mh.handler_count == 2

    def test_handler_error(self) -> None:
        mh = ModalHandler()
        mh.register("bad", lambda s: 1/0)
        sub = ModalSubmission(callback_id="bad", user_id="u1", channel="slack")
        result = mh.handle(sub)
        assert "error" in result


# ============================================================================
# Signature Verification
# ============================================================================


class TestSignatureVerifier:
    def test_no_secret_returns_false(self) -> None:
        v = SignatureVerifier()
        assert v.verify_slack(b"body", "ts", "sig") is False
        assert v.verify_discord(b"body", "ts", "sig") is False

    def test_has_secret_flags(self) -> None:
        v = SignatureVerifier(slack_signing_secret="secret123")
        assert v.has_slack_secret is True
        assert v.has_discord_key is False

    def test_slack_signature_valid(self) -> None:
        import hashlib
        import hmac
        secret = "test_secret_key"
        body = b'{"text":"hello"}'
        timestamp = "1234567890"
        basestring = f"v0:{timestamp}:{body.decode('utf-8')}"
        expected = "v0=" + hmac.new(
            secret.encode(), basestring.encode(), hashlib.sha256,
        ).hexdigest()
        v = SignatureVerifier(slack_signing_secret=secret)
        assert v.verify_slack(body, timestamp, expected) is True

    def test_slack_signature_invalid(self) -> None:
        v = SignatureVerifier(slack_signing_secret="secret")
        assert v.verify_slack(b"body", "ts", "v0=wrong") is False


# ============================================================================
# Interaction State Management
# ============================================================================


class TestInteractionStateStore:
    def test_create_and_get(self) -> None:
        store = InteractionStateStore()
        state = store.create("int_1", "u1", "approval", {"task_id": "t1"})
        assert state.interaction_id == "int_1"
        retrieved = store.get("int_1")
        assert retrieved is not None
        assert retrieved.context["task_id"] == "t1"

    def test_get_nonexistent(self) -> None:
        store = InteractionStateStore()
        assert store.get("nope") is None

    def test_complete(self) -> None:
        store = InteractionStateStore()
        store.create("int_1", "u1", "approval")
        assert store.complete("int_1") is True
        state = store.get("int_1")
        assert state is not None
        assert state.completed is True

    def test_complete_nonexistent(self) -> None:
        store = InteractionStateStore()
        assert store.complete("nope") is False

    def test_expired_returns_none(self) -> None:
        store = InteractionStateStore(ttl=0)  # Sofort abgelaufen
        store.create("int_1", "u1", "test")
        import time
        time.sleep(0.01)
        assert store.get("int_1") is None

    def test_cleanup_expired(self) -> None:
        store = InteractionStateStore(ttl=0)
        store.create("int_1", "u1", "test")
        store.create("int_2", "u2", "test")
        import time
        time.sleep(0.01)
        removed = store.cleanup_expired()
        assert removed == 2
        assert store.state_count == 0

    def test_state_count(self) -> None:
        store = InteractionStateStore()
        store.create("a", "u1", "t")
        store.create("b", "u2", "t")
        assert store.state_count == 2

    def test_response_url_stored(self) -> None:
        store = InteractionStateStore()
        state = store.create(
            "int_1", "u1", "approval",
            response_url="https://hooks.slack.com/actions/T/B/123",
        )
        assert state.response_url.startswith("https://")


# ============================================================================
# Fallback-Renderer
# ============================================================================


class TestFallbackRenderer:
    def test_render_card(self) -> None:
        card = (
            AdaptiveCard("Genehmigung", "Bitte prüfen Sie:")
            .add_field("Task", "CRM-Update")
            .add_field("Agent", "coder")
            .add_button("Erlauben", "approve")
            .add_button("Ablehnen", "reject")
            .set_footer("Ablauf in 24h")
        )
        text = FallbackRenderer.render_card(card)
        assert "Genehmigung" in text
        assert "CRM-Update" in text
        assert "Erlauben" in text
        assert "Ablehnen" in text
        assert "Ablauf in 24h" in text

    def test_render_progress(self) -> None:
        tracker = ProgressTracker("Deployment", ["Build", "Test", "Deploy"])
        tracker.start_step(0)
        tracker.complete_step(0)
        text = FallbackRenderer.render_progress(tracker)
        assert "Deployment" in text
        assert "Build" in text
        assert "✓" in text

    def test_render_buttons(self) -> None:
        buttons = [
            {"label": "Ja", "action_id": "yes"},
            {"label": "Nein", "action_id": "no"},
        ]
        text = FallbackRenderer.render_buttons(buttons)
        assert "[1] Ja" in text
        assert "[2] Nein" in text

    def test_render_form(self) -> None:
        fields = [
            FormField(
                name="name", label="Name",
                field_type=FieldType.TEXT,
                required=True, placeholder="z.B. coder",
            ),
            FormField(
                name="desc", label="Beschreibung",
                field_type=FieldType.TEXT,
                required=False,
            ),
        ]
        text = FallbackRenderer.render_form(fields)
        assert "Name *" in text
        assert "z.B. coder" in text
        assert "Beschreibung" in text
