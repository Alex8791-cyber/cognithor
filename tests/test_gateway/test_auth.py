"""Tests für Auth-Gateway."""

from __future__ import annotations

import pytest

from jarvis.gateway.auth import (
    AgentSession,
    AuthGateway,
    AuthMethod,
    GatewayToken,
)


class TestGatewayToken:
    def test_valid_token(self) -> None:
        gw = AuthGateway()
        raw, token = gw.create_token("user1", "coder")
        assert token.is_valid
        assert not token.is_expired
        assert not token.revoked

    def test_to_dict(self) -> None:
        gw = AuthGateway()
        _, token = gw.create_token("user1", "coder")
        d = token.to_dict()
        assert d["user_id"] == "user1"
        assert d["agent_id"] == "coder"
        assert d["is_valid"] is True


class TestAuthGateway:
    def test_create_and_validate_token(self) -> None:
        gw = AuthGateway()
        raw, token = gw.create_token("user1", "coder")
        validated = gw.validate_token(raw)
        assert validated is not None
        assert validated.token_id == token.token_id

    def test_invalid_token_rejected(self) -> None:
        gw = AuthGateway()
        assert gw.validate_token("invalid_token_xyz") is None

    def test_revoke_token(self) -> None:
        gw = AuthGateway()
        raw, token = gw.create_token("user1", "coder")
        assert gw.revoke_token(token.token_id)
        assert gw.validate_token(raw) is None

    def test_revoke_nonexistent(self) -> None:
        gw = AuthGateway()
        assert not gw.revoke_token("nonexistent")

    def test_revoke_all_for_user(self) -> None:
        gw = AuthGateway()
        gw.create_token("alex", "coder")
        gw.create_token("alex", "researcher")
        gw.create_token("bob", "coder")
        count = gw.revoke_all_for_user("alex")
        assert count == 2

    def test_expired_token(self) -> None:
        gw = AuthGateway(token_ttl=0)  # 0 = no expiry
        raw, _ = gw.create_token("u", "a", ttl_seconds=0)
        assert gw.validate_token(raw) is not None  # 0 TTL = no expiry

    def test_use_count_tracked(self) -> None:
        gw = AuthGateway()
        raw, token = gw.create_token("user1", "coder")
        gw.validate_token(raw)
        gw.validate_token(raw)
        gw.validate_token(raw)
        assert token.use_count == 3

    def test_create_session(self) -> None:
        gw = AuthGateway()
        _, token = gw.create_token("user1", "coder")
        session = gw.create_session("user1", "coder", token.token_id)
        assert session.active
        assert session.user_id == "user1"
        assert session.agent_id == "coder"

    def test_get_session(self) -> None:
        gw = AuthGateway()
        _, token = gw.create_token("user1", "coder")
        gw.create_session("user1", "coder", token.token_id)
        session = gw.get_session("user1", "coder")
        assert session is not None

    def test_get_session_nonexistent(self) -> None:
        gw = AuthGateway()
        assert gw.get_session("nope", "nope") is None

    def test_end_session(self) -> None:
        gw = AuthGateway()
        _, token = gw.create_token("u", "a")
        session = gw.create_session("u", "a", token.token_id)
        assert gw.end_session(session.session_key)
        assert not session.active

    def test_end_session_nonexistent(self) -> None:
        gw = AuthGateway()
        assert not gw.end_session("fake:key:123")

    def test_user_sessions(self) -> None:
        gw = AuthGateway()
        _, t1 = gw.create_token("alex", "coder")
        _, t2 = gw.create_token("alex", "researcher")
        gw.create_session("alex", "coder", t1.token_id)
        gw.create_session("alex", "researcher", t2.token_id)
        assert len(gw.user_sessions("alex")) == 2

    def test_active_sessions(self) -> None:
        gw = AuthGateway()
        _, t1 = gw.create_token("alex", "coder")
        _, t2 = gw.create_token("alex", "researcher")
        s1 = gw.create_session("alex", "coder", t1.token_id)
        gw.create_session("alex", "researcher", t2.token_id)
        gw.end_session(s1.session_key)
        assert len(gw.active_sessions("alex")) == 1

    def test_sso_login(self) -> None:
        gw = AuthGateway()
        result = gw.login("alex", ["coder", "researcher"])
        assert len(result) == 2
        assert "coder" in result
        assert "researcher" in result
        raw, session = result["coder"]
        assert isinstance(raw, str)
        assert session.active

    def test_logout(self) -> None:
        gw = AuthGateway()
        gw.login("alex", ["coder", "researcher"])
        revoked = gw.logout("alex")
        assert revoked == 2
        assert len(gw.active_sessions("alex")) == 0

    def test_check_scope_empty_denies_all(self) -> None:
        gw = AuthGateway()
        _, token = gw.create_token("u", "a", scopes=[])
        assert not gw.check_scope(token, "anything")  # deny-by-default

    def test_check_scope_wildcard(self) -> None:
        gw = AuthGateway()
        _, token = gw.create_token("u", "a", scopes=["*"])
        assert gw.check_scope(token, "config:write")

    def test_check_scope_specific(self) -> None:
        gw = AuthGateway()
        _, token = gw.create_token("u", "a", scopes=["read", "execute"])
        assert gw.check_scope(token, "read")
        assert not gw.check_scope(token, "write")

    def test_stats(self) -> None:
        gw = AuthGateway()
        gw.login("alex", ["coder"])
        s = gw.stats()
        assert s["total_tokens"] == 1
        assert s["active_tokens"] == 1
        assert s["total_sessions"] == 1
        assert s["unique_users"] == 1

    def test_audit_log(self) -> None:
        gw = AuthGateway()
        gw.login("alex", ["coder"])
        assert len(gw.audit_log) >= 2  # token_created + session_created + sso_login

    def test_session_touch(self) -> None:
        gw = AuthGateway()
        _, token = gw.create_token("u", "a")
        session = gw.create_session("u", "a", token.token_id)
        old = session.last_activity
        session.touch()
        assert session.last_activity >= old
