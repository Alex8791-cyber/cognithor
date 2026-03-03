"""Tests für alle 24 Audit-Fixes (C-01 bis H-28).

Verifiziert jede einzelne Korrektur aus dem Codebase-Audit:
  - Config Validation (C-13, C-14, C-15)
  - Security Hygiene (C-08, C-09, C-11, C-12, H-19)
  - Compliance Phase (C-01, C-02, C-04)
  - A2A Protocol (C-16, C-17)
  - Gateway Issues (H-05, H-07)
  - Dead Code Wiring (H-09, H-12, H-14)
  - Stubs/Simulations (H-17, H-22, H-28)
  - Channel Approvals (H-01, H-02)
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jarvis.config import JarvisConfig

_SRC_ROOT = Path(__file__).resolve().parent.parent / "src" / "jarvis"


# ============================================================================
# C-13: anthropic_max_tokens Bounds
# ============================================================================


class TestC13_MaxTokensBounds:
    """anthropic_max_tokens muss zwischen 1 und 1_000_000 liegen."""

    def test_default_value_valid(self, tmp_path: Path) -> None:
        config = JarvisConfig(jarvis_home=tmp_path)
        assert config.anthropic_max_tokens == 4096

    def test_valid_value(self, tmp_path: Path) -> None:
        config = JarvisConfig(jarvis_home=tmp_path, anthropic_max_tokens=8192)
        assert config.anthropic_max_tokens == 8192

    def test_zero_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(Exception):
            JarvisConfig(jarvis_home=tmp_path, anthropic_max_tokens=0)

    def test_negative_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(Exception):
            JarvisConfig(jarvis_home=tmp_path, anthropic_max_tokens=-1)

    def test_too_large_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(Exception):
            JarvisConfig(jarvis_home=tmp_path, anthropic_max_tokens=2_000_000)

    def test_boundary_min(self, tmp_path: Path) -> None:
        config = JarvisConfig(jarvis_home=tmp_path, anthropic_max_tokens=1)
        assert config.anthropic_max_tokens == 1

    def test_boundary_max(self, tmp_path: Path) -> None:
        config = JarvisConfig(jarvis_home=tmp_path, anthropic_max_tokens=1_000_000)
        assert config.anthropic_max_tokens == 1_000_000


# ============================================================================
# C-14: redis_url Pattern Validation
# ============================================================================


class TestC14_RedisUrlPattern:
    """redis_url muss mit redis:// oder rediss:// beginnen."""

    def test_default_valid(self, tmp_path: Path) -> None:
        config = JarvisConfig(jarvis_home=tmp_path)
        assert config.redis_url.startswith("redis://")

    def test_redis_url(self, tmp_path: Path) -> None:
        config = JarvisConfig(jarvis_home=tmp_path, redis_url="redis://myhost:6379/1")
        assert config.redis_url == "redis://myhost:6379/1"

    def test_rediss_url(self, tmp_path: Path) -> None:
        config = JarvisConfig(jarvis_home=tmp_path, redis_url="rediss://secure:6380/0")
        assert config.redis_url == "rediss://secure:6380/0"

    def test_http_url_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(Exception):
            JarvisConfig(jarvis_home=tmp_path, redis_url="http://wrong:6379")

    def test_empty_string_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(Exception):
            JarvisConfig(jarvis_home=tmp_path, redis_url="")


# ============================================================================
# C-15: API Key Length Validation
# ============================================================================


class TestC15_ApiKeyLength:
    """API Keys muessen mind. 8 Zeichen haben wenn gesetzt."""

    def test_empty_string_accepted(self, tmp_path: Path) -> None:
        config = JarvisConfig(jarvis_home=tmp_path, openai_api_key="")
        assert config.openai_api_key == ""

    def test_short_key_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(Exception, match="zu kurz"):
            JarvisConfig(jarvis_home=tmp_path, openai_api_key="short")

    def test_7_char_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(Exception, match="zu kurz"):
            JarvisConfig(jarvis_home=tmp_path, anthropic_api_key="1234567")

    def test_8_char_accepted(self, tmp_path: Path) -> None:
        config = JarvisConfig(jarvis_home=tmp_path, openai_api_key="12345678")
        assert config.openai_api_key == "12345678"

    def test_mask_placeholder_accepted(self, tmp_path: Path) -> None:
        """'***' Masken-Platzhalter darf durchgehen (UI-Roundtrip)."""
        config = JarvisConfig(jarvis_home=tmp_path, openai_api_key="***")
        assert config.openai_api_key == "***"

    def test_all_key_fields_validated(self, tmp_path: Path) -> None:
        """Alle 14 API-Key-Felder werden validiert."""
        key_fields = [
            "openai_api_key", "anthropic_api_key", "gemini_api_key",
            "groq_api_key", "deepseek_api_key", "mistral_api_key",
            "together_api_key", "openrouter_api_key", "xai_api_key",
            "cerebras_api_key", "github_api_key", "bedrock_api_key",
            "huggingface_api_key", "moonshot_api_key",
        ]
        for field_name in key_fields:
            with pytest.raises(Exception, match="zu kurz"):
                JarvisConfig(jarvis_home=tmp_path, **{field_name: "abc"})


# ============================================================================
# C-08: Token Store Base64 Fallback Warning (per-call)
# ============================================================================


class TestC08_TokenStoreFallbackWarning:
    """SecureTokenStore muss pro store()-Aufruf warnen wenn kein Fernet."""

    def test_store_warns_on_base64_fallback(self) -> None:
        from jarvis.security.token_store import SecureTokenStore

        store = SecureTokenStore()
        # Force no-Fernet mode
        store._fernet = None

        with patch("jarvis.security.token_store.logger") as mock_log:
            store.store("test_token", "my-secret-value")
            mock_log.warning.assert_called()
            call_args = str(mock_log.warning.call_args)
            assert "base64" in call_args.lower() or "insecure" in call_args.lower()


# ============================================================================
# C-09: Audit Trail Raises on Write Failure
# ============================================================================


class TestC09_AuditRaisesOnWriteFailure:
    """AuditTrail.record() muss bei Schreibfehler OSError werfen."""

    def test_record_raises_on_write_error(self, tmp_path: Path) -> None:
        from jarvis.security.audit import AuditTrail

        trail = AuditTrail(log_dir=tmp_path / "audit")
        # Set log path to a non-existent deeply nested directory
        trail._log_path = tmp_path / "nonexistent" / "deep" / "audit.jsonl"

        from jarvis.models import AuditEntry, GateStatus, RiskLevel

        entry = AuditEntry(
            session_id="test",
            action_tool="test_tool",
            action_params_hash="abc123",
            decision_status=GateStatus.ALLOW,
            decision_reason="test",
            risk_level=RiskLevel.GREEN,
        )
        with pytest.raises(OSError):
            trail.record(entry)

    def test_record_event_raises_on_write_error(self, tmp_path: Path) -> None:
        from jarvis.security.audit import AuditTrail

        trail = AuditTrail(log_dir=tmp_path / "audit")
        trail._log_path = tmp_path / "nonexistent" / "deep" / "audit.jsonl"

        with pytest.raises(OSError):
            trail.record_event(
                session_id="test",
                event_type="test_event",
                details={"key": "value"},
            )


# ============================================================================
# C-11: Credentials chmod Logs Warning
# ============================================================================


class TestC11_CredentialChmodWarning:
    """chmod-Fehler duerfen nicht verschluckt werden, sondern geloggt."""

    def test_set_file_permissions_logs_on_failure(self, tmp_path: Path) -> None:
        from jarvis.security.credentials import CredentialStore

        fake_path = tmp_path / "nonexistent_file.enc"

        with patch("jarvis.security.credentials.log") as mock_log:
            CredentialStore._set_file_permissions(fake_path)
            mock_log.warning.assert_called_once()
            call_args = str(mock_log.warning.call_args)
            assert "credential_chmod_failed" in call_args


# ============================================================================
# C-12: AgentVault Narrowed Exception
# ============================================================================


class TestC12_VaultNarrowedException:
    """_decrypt() faengt nur InvalidToken, nicht alle Exceptions."""

    def test_decrypt_invalid_fernet_raises_valueerror(self) -> None:
        """Invalid Fernet token -> InvalidToken -> ValueError (legacy fallback)."""
        from jarvis.security.agent_vault import AgentVault

        vault = AgentVault("test-agent")
        with pytest.raises(ValueError, match="Decryption failed"):
            vault._decrypt("not-valid-fernet-token")

    def test_decrypt_propagates_other_errors(self) -> None:
        from jarvis.security.agent_vault import AgentVault

        vault = AgentVault("test-agent")
        with pytest.raises((TypeError, AttributeError)):
            vault._decrypt(12345)  # type: ignore[arg-type]


# ============================================================================
# H-19: Revoked Secrets Removed from Vault
# ============================================================================


class TestH19_RevokedSecretsRemoved:
    """revoke() muss Secrets tatsaechlich aus dem dict entfernen."""

    def test_revoke_removes_secret(self) -> None:
        from jarvis.security.agent_vault import AgentVault

        vault = AgentVault("test-agent")
        secret = vault.store("my_api_key", "secret-value-12345")

        assert vault.secret_count == 1
        result = vault.revoke(secret.secret_id)
        assert result is True
        assert vault.secret_count == 0
        assert vault.retrieve(secret.secret_id) is None

    def test_revoke_nonexistent_returns_false(self) -> None:
        from jarvis.security.agent_vault import AgentVault

        vault = AgentVault("test-agent")
        assert vault.revoke("SEC-NOTEXIST-0001") is False


# ============================================================================
# C-01 / C-02: Compliance Phase Init
# ============================================================================


class TestC01C02_CompliancePhaseInit:
    """init_compliance() wird aufgerufen und loggt verfuegbare Komponenten."""

    @pytest.mark.asyncio()
    async def test_init_compliance_with_no_components(self) -> None:
        from jarvis.gateway.phases.compliance import init_compliance

        result = await init_compliance(config=None)
        assert isinstance(result, dict)

    @pytest.mark.asyncio()
    async def test_init_compliance_with_components(self) -> None:
        from jarvis.gateway.phases.compliance import init_compliance

        result = await init_compliance(
            config=None,
            compliance_framework=MagicMock(),
            decision_log=MagicMock(),
        )
        assert isinstance(result, dict)


# ============================================================================
# C-04: Governance Exports
# ============================================================================


class TestC04_GovernanceExports:
    """governance/__init__.py exportiert GovernanceAgent und PolicyPatcher."""

    def test_governance_exports(self) -> None:
        import jarvis.governance as gov

        assert hasattr(gov, "GovernanceAgent")
        assert hasattr(gov, "PolicyPatcher")
        assert "GovernanceAgent" in gov.__all__
        assert "PolicyPatcher" in gov.__all__

    def test_governance_classes_importable(self) -> None:
        from jarvis.governance import GovernanceAgent, PolicyPatcher

        assert GovernanceAgent is not None
        assert PolicyPatcher is not None


# ============================================================================
# C-16: A2A POST /a2a Endpoint in Fallback Server
# ============================================================================


class TestC16_A2AFallbackEndpoint:
    """Der minimalistische Fallback-Server muss POST /a2a routen."""

    def test_post_a2a_route_in_source(self) -> None:
        """Verify the route exists in source code."""
        source = (_SRC_ROOT / "a2a" / "http_handler.py").read_text(encoding="utf-8")
        assert 'POST' in source and '/a2a' in source
        assert 'handle_a2a_request' in source


# ============================================================================
# C-17: A2A Task Done Callback
# ============================================================================


class TestC17_A2ATaskDoneCallback:
    """_make_task_done_callback erstellt einen Closure fuer Task-Fehler."""

    def test_callback_factory_exists(self) -> None:
        from jarvis.a2a.server import A2AServer

        assert hasattr(A2AServer, "_make_task_done_callback")

    def test_callback_handles_exception(self) -> None:
        from jarvis.a2a.server import A2AServer

        server = A2AServer.__new__(A2AServer)
        server._tasks_failed = 0

        mock_task = MagicMock()
        mock_task.id = "test-task-1"
        mock_task.is_active = True
        mock_task.transition = MagicMock()

        callback = server._make_task_done_callback(mock_task)

        mock_asyncio_task = MagicMock()
        mock_asyncio_task.cancelled.return_value = False
        mock_asyncio_task.exception.return_value = RuntimeError("boom")

        callback(mock_asyncio_task)

        mock_task.transition.assert_called_once()
        assert server._tasks_failed == 1

    def test_callback_ignores_cancelled(self) -> None:
        from jarvis.a2a.server import A2AServer

        server = A2AServer.__new__(A2AServer)
        server._tasks_failed = 0

        mock_task = MagicMock()
        callback = server._make_task_done_callback(mock_task)

        mock_asyncio_task = MagicMock()
        mock_asyncio_task.cancelled.return_value = True

        callback(mock_asyncio_task)
        assert server._tasks_failed == 0


# ============================================================================
# H-05: Gateway Silent Except Blocks Removed
# ============================================================================


class TestH05_GatewaySilentExceptsRemoved:
    """except Exception: pass wurde durch log.debug() ersetzt."""

    def test_gateway_has_no_silent_pass_blocks(self) -> None:
        content = (_SRC_ROOT / "gateway" / "gateway.py").read_text(encoding="utf-8")
        silent_blocks = re.findall(
            r"except\s+Exception\s*:.*?\n\s+pass\s*\n",
            content,
        )
        assert len(silent_blocks) == 0, (
            f"Found {len(silent_blocks)} silent 'except Exception: pass' blocks"
        )


# ============================================================================
# H-07: Magic Strings Extracted to Constants
# ============================================================================


class TestH07_MagicStringsExtracted:
    """Presearch Magic Strings sind als Konstanten definiert."""

    def test_constants_defined(self) -> None:
        from jarvis.gateway import gateway

        assert hasattr(gateway, "_PRESEARCH_NO_RESULTS")
        assert hasattr(gateway, "_PRESEARCH_NO_ENGINE")
        assert isinstance(gateway._PRESEARCH_NO_RESULTS, str)
        assert isinstance(gateway._PRESEARCH_NO_ENGINE, str)


# ============================================================================
# H-09: Orchestrator Wired in Agents Phase
# ============================================================================


class TestH09_OrchestratorWired:
    """Die agents-Phase enthält Orchestrator-Initialisierung."""

    def test_orchestrator_code_in_agents_phase(self) -> None:
        source = (_SRC_ROOT / "gateway" / "phases" / "agents.py").read_text(encoding="utf-8")
        assert "Orchestrator" in source
        assert "orchestrator" in source


# ============================================================================
# H-12: add_failure_pattern Wired
# ============================================================================


class TestH12_FailurePatternWired:
    """add_failure_pattern wird in gateway.py aufgerufen."""

    def test_failure_pattern_in_gateway(self) -> None:
        content = (_SRC_ROOT / "gateway" / "gateway.py").read_text(encoding="utf-8")
        assert "add_failure_pattern" in content


# ============================================================================
# H-14: Dead _chunk_hash_map_version Removed
# ============================================================================


class TestH14_DeadAttributeRemoved:
    """_chunk_hash_map_version wurde aus memory/search.py entfernt."""

    def test_no_chunk_hash_map_version(self) -> None:
        content = (_SRC_ROOT / "memory" / "search.py").read_text(encoding="utf-8")
        assert "_chunk_hash_map_version" not in content


# ============================================================================
# H-17: WebhookNotifier Real HTTP
# ============================================================================


class TestH17_WebhookNotifierRealHTTP:
    """WebhookNotifier nutzt echtes httpx mit HMAC."""

    def test_webhook_notifier_uses_httpx(self) -> None:
        """Verify httpx is used in notify()."""
        source = (_SRC_ROOT / "security" / "hardening.py").read_text(encoding="utf-8")
        assert "httpx" in source
        assert "hmac" in source

    def test_webhook_notifier_sends_with_signature(self) -> None:
        from jarvis.security.hardening import WebhookConfig, WebhookNotifier

        notifier = WebhookNotifier()
        webhook = WebhookConfig(
            url="https://example.com/hook",
            events=["test_event"],
            secret="my-hmac-secret",
        )
        notifier.add_webhook(webhook)
        assert notifier.webhook_count == 1

        # Mock httpx for the synchronous notify() — httpx is imported locally
        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post = MagicMock(return_value=mock_response)

        import httpx as real_httpx
        with patch.object(real_httpx, "Client", return_value=mock_client):
            sent = notifier.notify("test_event", {"key": "value"})
            assert sent == 1
            mock_client.post.assert_called_once()
            # Verify HMAC header was sent
            call_kwargs = mock_client.post.call_args
            headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers", {})
            assert "X-Signature-SHA256" in headers


# ============================================================================
# H-22: SkillUpdater Version-Tracking Warning
# ============================================================================


class TestH22_SkillUpdaterWarning:
    """install_update() loggt eine Warnung ueber Version-Tracking."""

    def test_install_logs_version_tracking_warning(self) -> None:
        from jarvis.skills.updater import SkillUpdater

        updater = SkillUpdater()
        updater.register_installed("test-skill", "1.0.0")
        updater.check_update("test-skill", "1.1.0")

        with patch("jarvis.skills.updater.log") as mock_log:
            result = updater.install_update("test-skill")

            assert result.success is True
            mock_log.warning.assert_called()
            call_args = str(mock_log.warning.call_args)
            assert "version_tracking" in call_args.lower() or "tracking" in call_args.lower()


# ============================================================================
# H-28: MCP HTTP Transport
# ============================================================================


class TestH28_MCPHttpTransport:
    """JarvisMCPClient hat _connect_http_server() Methode."""

    def test_http_transport_method_exists(self) -> None:
        from jarvis.mcp.client import JarvisMCPClient

        assert hasattr(JarvisMCPClient, "_connect_http_server")

    def test_http_transport_in_source(self) -> None:
        source = (_SRC_ROOT / "mcp" / "client.py").read_text(encoding="utf-8")
        assert "_connect_http_server" in source
        assert "sse_client" in source


# ============================================================================
# H-01: IRC Approval Flow
# ============================================================================


class TestH01_IRCApproval:
    """IRC-Channel hat funktionale textbasierte Approval."""

    def test_irc_has_approval_futures(self) -> None:
        from jarvis.channels.irc import IRCChannel

        ch = IRCChannel(server="irc.example.com", nick="TestBot")
        assert hasattr(ch, "_approval_futures")
        assert hasattr(ch, "_approval_lock")
        assert isinstance(ch._approval_futures, dict)

    @pytest.mark.asyncio()
    async def test_irc_approval_timeout(self) -> None:
        from jarvis.channels.irc import IRCChannel

        ch = IRCChannel(server="irc.example.com", nick="TestBot", channels=["#test"])
        ch._writer = MagicMock()
        ch._writer.write = MagicMock()
        ch._writer.drain = AsyncMock()

        mock_action = MagicMock()
        mock_action.tool_name = "test_tool"

        with patch.object(ch, "_send_message", new=AsyncMock()):
            with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError):
                result = await ch.request_approval("sess-1", mock_action, "test reason")
                assert result is False

    def test_irc_ja_nein_in_source(self) -> None:
        """IRC message handler fängt ja/nein ab."""
        source = (_SRC_ROOT / "channels" / "irc.py").read_text(encoding="utf-8")
        assert '"ja"' in source or "'ja'" in source
        assert "approval_futures" in source


# ============================================================================
# H-02: Twitch Approval Flow
# ============================================================================


class TestH02_TwitchApproval:
    """Twitch-Channel hat funktionale textbasierte Approval."""

    def test_twitch_has_approval_futures(self) -> None:
        from jarvis.channels.twitch import TwitchChannel

        ch = TwitchChannel(token="oauth:test123456", channel="testchannel")
        assert hasattr(ch, "_approval_futures")
        assert hasattr(ch, "_approval_lock")
        assert isinstance(ch._approval_futures, dict)

    @pytest.mark.asyncio()
    async def test_twitch_approval_timeout(self) -> None:
        from jarvis.channels.twitch import TwitchChannel

        ch = TwitchChannel(token="oauth:test123456", channel="testchannel")
        ch._writer = MagicMock()
        ch._writer.write = MagicMock()
        ch._writer.drain = AsyncMock()

        mock_action = MagicMock()
        mock_action.tool_name = "test_tool"

        with patch.object(ch, "_send_chat", new=AsyncMock()):
            with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError):
                result = await ch.request_approval("sess-2", mock_action, "test reason")
                assert result is False

    def test_twitch_ja_nein_in_source(self) -> None:
        """Twitch message handler fängt ja/nein ab."""
        source = (_SRC_ROOT / "channels" / "twitch.py").read_text(encoding="utf-8")
        assert '"ja"' in source or "'ja'" in source
        assert "approval_futures" in source
