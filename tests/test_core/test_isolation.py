"""Tests für Agent-Isolation.

Testet: WorkspaceGuard, AgentResourceQuota, RateLimiter,
UserAgentScope, MultiUserIsolation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from jarvis.core.isolation import (
    AgentResourceQuota,
    MultiUserIsolation,
    RateLimiter,
    UserAgentScope,
    WorkspaceGuard,
    WorkspacePolicy,
)


# ============================================================================
# WorkspaceGuard
# ============================================================================


class TestWorkspaceGuard:
    def test_register_agent(self, tmp_path: Path) -> None:
        guard = WorkspaceGuard(tmp_path)
        policy = guard.register_agent("coder")
        assert policy.agent_id == "coder"
        assert (tmp_path / "coder").is_dir()
        assert "coder" in guard.registered_agents

    def test_own_workspace_always_allowed(self, tmp_path: Path) -> None:
        guard = WorkspaceGuard(tmp_path)
        guard.register_agent("agent_a")
        test_file = tmp_path / "agent_a" / "test.py"
        assert guard.check_access("agent_a", test_file, "read")
        assert guard.check_access("agent_a", test_file, "write")

    def test_cross_agent_access_blocked(self, tmp_path: Path) -> None:
        guard = WorkspaceGuard(tmp_path)
        guard.register_agent("agent_a")
        guard.register_agent("agent_b")

        target = tmp_path / "agent_b" / "secret.txt"
        assert not guard.check_access("agent_a", target, "read")
        assert not guard.check_access("agent_a", target, "write")

    def test_shared_workspace_read(self, tmp_path: Path) -> None:
        guard = WorkspaceGuard(tmp_path)
        shared = tmp_path / "_shared"
        shared.mkdir(parents=True, exist_ok=True)

        guard.register_agent("reader", allow_read_shared=True)
        guard.register_agent("no_access")

        shared_file = shared / "data.csv"
        assert guard.check_access("reader", shared_file, "read")
        assert not guard.check_access("reader", shared_file, "write")
        assert not guard.check_access("no_access", shared_file, "read")

    def test_shared_workspace_write(self, tmp_path: Path) -> None:
        guard = WorkspaceGuard(tmp_path)
        shared = tmp_path / "_shared"
        shared.mkdir(parents=True, exist_ok=True)

        guard.register_agent("writer", allow_write_shared=True)
        assert guard.check_access("writer", shared / "out.txt", "write")

    def test_external_whitelist(self, tmp_path: Path) -> None:
        guard = WorkspaceGuard(tmp_path)
        ext = tmp_path / "external_data"
        ext.mkdir(parents=True, exist_ok=True)

        guard.register_agent("trusted", allowed_external_paths=[str(ext)])
        assert guard.check_access("trusted", ext / "file.txt", "read")

    def test_outside_workspace_blocked(self, tmp_path: Path) -> None:
        guard = WorkspaceGuard(tmp_path)
        guard.register_agent("agent_a")
        assert not guard.check_access("agent_a", Path("/etc/passwd"), "read")

    def test_unregistered_agent_blocked(self, tmp_path: Path) -> None:
        guard = WorkspaceGuard(tmp_path)
        assert not guard.check_access("unknown", tmp_path / "test", "read")

    def test_path_traversal_blocked(self, tmp_path: Path) -> None:
        guard = WorkspaceGuard(tmp_path)
        guard.register_agent("agent_a")

        result = guard.resolve_path("agent_a", "../../etc/passwd")
        assert result is None

    def test_resolve_path_within_workspace(self, tmp_path: Path) -> None:
        guard = WorkspaceGuard(tmp_path)
        guard.register_agent("agent_a")

        result = guard.resolve_path("agent_a", "subdir/file.txt")
        assert result is not None
        assert str(result).startswith(str(tmp_path / "agent_a"))

    def test_violations_recorded(self, tmp_path: Path) -> None:
        guard = WorkspaceGuard(tmp_path)
        guard.register_agent("agent_a")
        guard.register_agent("agent_b")

        guard.check_access("agent_a", tmp_path / "agent_b" / "secret", "read")
        guard.check_access("agent_a", Path("/etc/hosts"), "read")

        all_violations = guard.violations()
        assert len(all_violations) == 2

        a_violations = guard.violations("agent_a")
        assert len(a_violations) == 2

    def test_get_workspace(self, tmp_path: Path) -> None:
        guard = WorkspaceGuard(tmp_path)
        guard.register_agent("test")
        assert guard.get_workspace("test") == tmp_path / "test"
        assert guard.get_workspace("nonexistent") is None


# ============================================================================
# AgentResourceQuota
# ============================================================================


class TestAgentResourceQuota:
    def test_check_token_budget(self) -> None:
        quota = AgentResourceQuota(agent_id="coder", daily_token_budget=1000)
        assert quota.check_token_budget(500)
        assert quota.check_token_budget(1000)
        assert not quota.check_token_budget(1001)

    def test_consume_tokens(self) -> None:
        quota = AgentResourceQuota(agent_id="coder", daily_token_budget=100)
        assert quota.consume_tokens(50)
        assert quota.tokens_remaining == 50
        assert quota.consume_tokens(50)
        assert not quota.consume_tokens(1)

    def test_reset_daily(self) -> None:
        quota = AgentResourceQuota(agent_id="coder", daily_token_budget=100)
        quota.consume_tokens(80)
        assert quota.tokens_remaining == 20
        quota.reset_daily()
        assert quota.tokens_remaining == 100

    def test_budget_utilization(self) -> None:
        quota = AgentResourceQuota(agent_id="coder", daily_token_budget=200)
        quota.consume_tokens(50)
        assert quota.budget_utilization_percent == 25.0

    def test_tool_allowed(self) -> None:
        quota = AgentResourceQuota(
            agent_id="restricted",
            blocked_tools=["shell.run", "fs.delete"],
        )
        assert quota.is_tool_allowed("web.search")
        assert not quota.is_tool_allowed("shell.run")
        assert not quota.is_tool_allowed("fs.delete")

    def test_to_dict(self) -> None:
        quota = AgentResourceQuota(agent_id="test", daily_token_budget=5000)
        d = quota.to_dict()
        assert d["agent_id"] == "test"
        assert d["daily_token_budget"] == 5000
        assert "tokens_remaining" in d


# ============================================================================
# RateLimiter
# ============================================================================


class TestRateLimiter:
    def test_allows_within_limit(self) -> None:
        rl = RateLimiter()
        for _ in range(5):
            assert rl.check_and_consume("agent_a", max_per_minute=10)

    def test_blocks_over_limit(self) -> None:
        rl = RateLimiter()
        for _ in range(5):
            rl.check_and_consume("agent_a", max_per_minute=5)
        assert not rl.check_and_consume("agent_a", max_per_minute=5)

    def test_separate_agents(self) -> None:
        rl = RateLimiter()
        for _ in range(3):
            rl.check_and_consume("a", max_per_minute=3)
        assert not rl.check_and_consume("a", max_per_minute=3)
        assert rl.check_and_consume("b", max_per_minute=3)  # Anderer Agent

    def test_current_rate(self) -> None:
        rl = RateLimiter()
        rl.check_and_consume("x", max_per_minute=100)
        rl.check_and_consume("x", max_per_minute=100)
        assert rl.current_rate("x") == 2
        assert rl.current_rate("y") == 0


# ============================================================================
# UserAgentScope
# ============================================================================


class TestUserAgentScope:
    def test_scope_key(self) -> None:
        scope = UserAgentScope(user_id="alice", agent_id="coder")
        assert scope.scope_key == "alice:coder"

    def test_credential_namespace(self) -> None:
        scope = UserAgentScope(user_id="alice", agent_id="coder")
        assert scope.effective_credential_namespace == "alice:coder"

    def test_custom_credential_namespace(self) -> None:
        scope = UserAgentScope(
            user_id="alice", agent_id="coder",
            credential_namespace="custom_ns",
        )
        assert scope.effective_credential_namespace == "custom_ns"


# ============================================================================
# MultiUserIsolation
# ============================================================================


class TestMultiUserIsolation:
    def test_create_scope(self) -> None:
        iso = MultiUserIsolation()
        scope = iso.get_or_create_scope("alice", "jarvis")
        assert scope.user_id == "alice"
        assert scope.agent_id == "jarvis"

    def test_scope_cached(self) -> None:
        iso = MultiUserIsolation()
        s1 = iso.get_or_create_scope("alice", "jarvis")
        s2 = iso.get_or_create_scope("alice", "jarvis")
        assert s1 is s2

    def test_register_and_validate_session(self) -> None:
        iso = MultiUserIsolation()
        iso.register_session("alice", "jarvis", "session_1")

        assert iso.validate_session_access("alice", "jarvis", "session_1")
        assert not iso.validate_session_access("alice", "jarvis", "session_2")
        assert not iso.validate_session_access("bob", "jarvis", "session_1")

    def test_cross_user_session_blocked(self) -> None:
        iso = MultiUserIsolation()
        iso.register_session("alice", "jarvis", "alice_s1")
        iso.register_session("bob", "jarvis", "bob_s1")

        assert not iso.validate_session_access("bob", "jarvis", "alice_s1")
        assert not iso.validate_session_access("alice", "jarvis", "bob_s1")

    def test_revoke_session(self) -> None:
        iso = MultiUserIsolation()
        iso.register_session("alice", "jarvis", "s1")
        assert iso.revoke_session("alice", "jarvis", "s1")
        assert not iso.validate_session_access("alice", "jarvis", "s1")

    def test_revoke_nonexistent(self) -> None:
        iso = MultiUserIsolation()
        assert not iso.revoke_session("nobody", "jarvis", "s1")

    def test_credential_namespace_isolation(self) -> None:
        iso = MultiUserIsolation()
        ns_a = iso.get_credential_namespace("alice", "jarvis")
        ns_b = iso.get_credential_namespace("bob", "jarvis")
        assert ns_a != ns_b
        assert "alice" in ns_a
        assert "bob" in ns_b

    def test_delegation_within_user(self) -> None:
        iso = MultiUserIsolation()
        iso.get_or_create_scope("alice", "jarvis")
        assert iso.can_delegate("alice", "jarvis", "coder")

    def test_delegation_unknown_user_blocked(self) -> None:
        iso = MultiUserIsolation()
        assert not iso.can_delegate("unknown", "jarvis", "coder")

    def test_user_scopes(self) -> None:
        iso = MultiUserIsolation()
        iso.get_or_create_scope("alice", "jarvis")
        iso.get_or_create_scope("alice", "coder")
        iso.get_or_create_scope("bob", "jarvis")

        alice_scopes = iso.user_scopes("alice")
        assert len(alice_scopes) == 2

    def test_agent_scopes(self) -> None:
        iso = MultiUserIsolation()
        iso.get_or_create_scope("alice", "jarvis")
        iso.get_or_create_scope("bob", "jarvis")

        jarvis_scopes = iso.agent_scopes("jarvis")
        assert len(jarvis_scopes) == 2

    def test_quota_management(self) -> None:
        iso = MultiUserIsolation()
        quota = AgentResourceQuota(agent_id="coder", daily_token_budget=1000)
        iso.set_quota("coder", quota)

        assert iso.get_quota("coder") is not None
        assert iso.consume_tokens("coder", 500)
        assert iso.consume_tokens("coder", 500)
        assert not iso.consume_tokens("coder", 1)

    def test_no_quota_allows_all(self) -> None:
        iso = MultiUserIsolation()
        assert iso.consume_tokens("unconfigured", 999999)
        assert iso.check_rate_limit("unconfigured")

    def test_rate_limit(self) -> None:
        iso = MultiUserIsolation()
        quota = AgentResourceQuota(agent_id="fast", max_requests_per_minute=2)
        iso.set_quota("fast", quota)

        assert iso.check_rate_limit("fast")
        assert iso.check_rate_limit("fast")
        assert not iso.check_rate_limit("fast")

    def test_stats(self) -> None:
        iso = MultiUserIsolation()
        iso.register_session("alice", "jarvis", "s1")
        iso.register_session("bob", "coder", "s2")
        iso.set_quota("jarvis", AgentResourceQuota(agent_id="jarvis"))

        s = iso.stats()
        assert s["total_scopes"] == 2
        assert s["unique_users"] == 2
        assert s["unique_agents"] == 2
        assert s["active_sessions"] == 2
        assert s["quotas_configured"] == 1
