"""Tests: Multi-Agent-Separation unter konkurrierenden Sessions.

Prüft, dass bei gleichzeitigem Zugriff mehrerer Agenten:
  - Workspace-Zugriffe korrekt blockiert werden
  - Credentials nicht zwischen Agenten lecken
  - Heartbeat-Tasks unabhängig laufen
  - Auth-Tokens pro Agent getrennt sind
  - Resource-Quotas pro Agent greifen
  - RBAC verschiedene User korrekt trennt
  - Multi-User-Isolation skaliert

Bibel-Referenz: §8 (Agent-Separation), §14 (Security)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


# ============================================================================
# Workspace-Guard bei konkurrierenden Zugriffen
# ============================================================================


class TestConcurrentWorkspaceGuard:
    """Prüft, dass WorkspaceGuard bei parallelen Agenten korrekt blockiert."""

    def test_agents_cannot_access_each_others_workspace(self, tmp_path: Path) -> None:
        from jarvis.core.isolation import WorkspaceGuard

        guard = WorkspaceGuard(tmp_path)
        guard.register_agent("agent_a")
        guard.register_agent("agent_b")

        # Agent A arbeitet in seinem Workspace
        assert guard.check_access("agent_a", tmp_path / "agent_a" / "file.txt")

        # Agent A versucht auf Agent B's Workspace zuzugreifen
        assert not guard.check_access("agent_a", tmp_path / "agent_b" / "file.txt")

        # Agent B arbeitet in seinem Workspace
        assert guard.check_access("agent_b", tmp_path / "agent_b" / "data.json")

        # Agent B versucht auf Agent A zuzugreifen
        assert not guard.check_access("agent_b", tmp_path / "agent_a" / "secret.key")

    def test_violations_tracked_per_agent(self, tmp_path: Path) -> None:
        from jarvis.core.isolation import WorkspaceGuard

        guard = WorkspaceGuard(tmp_path)
        guard.register_agent("bad_agent_1")
        guard.register_agent("bad_agent_2")
        guard.register_agent("agent_x")
        guard.register_agent("agent_y")
        guard.register_agent("agent_z")

        guard.check_access("bad_agent_1", tmp_path / "agent_x" / "f.txt")
        guard.check_access("bad_agent_2", tmp_path / "agent_y" / "f.txt")
        guard.check_access("bad_agent_1", tmp_path / "agent_z" / "f.txt")

        agent1_violations = guard.violations("bad_agent_1")
        agent2_violations = guard.violations("bad_agent_2")
        assert len(agent1_violations) == 2
        assert len(agent2_violations) == 1

    def test_many_agents_simultaneous(self, tmp_path: Path) -> None:
        """50 Agenten greifen gleichzeitig zu — jeder nur auf seinen Workspace."""
        from jarvis.core.isolation import WorkspaceGuard

        guard = WorkspaceGuard(tmp_path)
        for i in range(50):
            guard.register_agent(f"agent_{i}")

        for i in range(50):
            agent_id = f"agent_{i}"
            assert guard.check_access(agent_id, tmp_path / agent_id / "work.txt")
            other = f"agent_{(i+1) % 50}"
            assert not guard.check_access(agent_id, tmp_path / other / "secret.txt")


# ============================================================================
# Credential-Isolation
# ============================================================================


class TestConcurrentCredentialIsolation:
    """Prüft, dass Credentials pro Agent isoliert sind."""

    def test_agents_have_separate_credentials(self, tmp_path: Path) -> None:
        from jarvis.security.credentials import CredentialStore

        store = CredentialStore(store_path=tmp_path / "creds", passphrase="test-secret")

        store.store("openai", "api_key", "sk-agent-a-secret", agent_id="agent_a")
        store.store("openai", "api_key", "sk-agent-b-secret", agent_id="agent_b")

        assert store.retrieve("openai", "api_key", agent_id="agent_a") == "sk-agent-a-secret"
        assert store.retrieve("openai", "api_key", agent_id="agent_b") == "sk-agent-b-secret"

    def test_credential_listing_per_agent(self, tmp_path: Path) -> None:
        from jarvis.security.credentials import CredentialStore

        store = CredentialStore(store_path=tmp_path / "creds2", passphrase="test-secret")
        store.store("svc1", "key1", "val1", agent_id="agent_x")
        store.store("svc2", "key2", "val2", agent_id="agent_x")
        store.store("svc3", "key3", "val3", agent_id="agent_y")

        x_entries = store.list_entries(agent_id="agent_x")
        y_entries = store.list_entries(agent_id="agent_y")
        assert len(x_entries) == 2
        assert len(y_entries) == 1


# ============================================================================
# Auth-Gateway: Parallele SSO-Sessions
# ============================================================================


class TestConcurrentAuthSessions:
    """Prüft Token- und Session-Isolation bei parallelem Login."""

    def test_sso_creates_independent_tokens(self) -> None:
        from jarvis.gateway.auth import AuthGateway

        gw = AuthGateway()
        result = gw.login("alex", ["coder", "researcher", "assistant"])
        assert len(result) == 3
        tokens = [result[a][0] for a in result]
        assert len(set(tokens)) == 3

    def test_token_revocation_per_agent(self) -> None:
        from jarvis.gateway.auth import AuthGateway

        gw = AuthGateway()
        result = gw.login("alex", ["coder", "researcher"])

        coder_raw, _ = result["coder"]
        researcher_raw, _ = result["researcher"]

        coder_token = gw.validate_token(coder_raw)
        gw.revoke_token(coder_token.token_id)

        assert gw.validate_token(coder_raw) is None
        assert gw.validate_token(researcher_raw) is not None

    def test_parallel_users_isolated(self) -> None:
        from jarvis.gateway.auth import AuthGateway

        gw = AuthGateway()
        alex_result = gw.login("alex", ["coder"])
        bob_result = gw.login("bob", ["coder"])

        alex_token = alex_result["coder"][0]
        bob_token = bob_result["coder"][0]
        assert alex_token != bob_token

        gw.logout("alex")
        assert gw.validate_token(alex_token) is None
        assert gw.validate_token(bob_token) is not None

    def test_scope_isolation(self) -> None:
        from jarvis.gateway.auth import AuthGateway

        gw = AuthGateway()
        _, coder_token = gw.create_token("alex", "coder", scopes=["execute", "shell"])
        _, reader_token = gw.create_token("alex", "reader", scopes=["read"])

        assert gw.check_scope(coder_token, "shell")
        assert not gw.check_scope(reader_token, "shell")
        assert gw.check_scope(reader_token, "read")

    def test_many_users_many_agents(self) -> None:
        """20 User × 5 Agents = 100 separate Sessions."""
        from jarvis.gateway.auth import AuthGateway

        gw = AuthGateway()
        agents = ["coder", "researcher", "assistant", "writer", "analyst"]
        all_tokens = set()

        for i in range(20):
            result = gw.login(f"user_{i}", agents)
            for agent_id in agents:
                all_tokens.add(result[agent_id][0])

        assert len(all_tokens) == 100
        stats = gw.stats()
        assert stats["total_tokens"] == 100
        assert stats["unique_users"] == 20


# ============================================================================
# Heartbeat: Getrennte Task-Ausführung
# ============================================================================


class TestConcurrentHeartbeat:
    def test_tasks_isolated_between_agents(self) -> None:
        from jarvis.core.agent_heartbeat import (
            AgentHeartbeatConfig,
            AgentHeartbeatScheduler,
            AgentTask,
        )

        sched = AgentHeartbeatScheduler()
        sched.configure_agent(AgentHeartbeatConfig(agent_id="coder", interval_minutes=15))
        sched.configure_agent(AgentHeartbeatConfig(agent_id="researcher", interval_minutes=60))

        sched.add_task(AgentTask(task_id="build", agent_id="coder", name="Build"))
        sched.add_task(AgentTask(task_id="test", agent_id="coder", name="Test"))
        sched.add_task(AgentTask(task_id="search", agent_id="researcher", name="Search"))

        assert len(sched.agent_tasks("coder")) == 2
        assert len(sched.agent_tasks("researcher")) == 1

        run = sched.start_task("coder", "build")
        sched.complete_task(run, success=True)

        assert sched.agent_summary("coder")["total_runs"] == 1
        assert sched.agent_summary("researcher")["total_runs"] == 0

    def test_failure_in_one_agent_doesnt_affect_other(self) -> None:
        from jarvis.core.agent_heartbeat import (
            AgentHeartbeatScheduler,
            AgentTask,
            TaskStatus,
        )

        sched = AgentHeartbeatScheduler()
        sched.add_task(AgentTask(task_id="t1", agent_id="a", name="Task A"))
        sched.add_task(AgentTask(task_id="t2", agent_id="b", name="Task B"))

        run_a = sched.start_task("a", "t1")
        sched.complete_task(run_a, success=False, error="crash")

        run_b = sched.start_task("b", "t2")
        sched.complete_task(run_b, success=True)

        assert sched.get_task("a", "t1").last_status == TaskStatus.FAILED
        assert sched.get_task("b", "t2").last_status == TaskStatus.COMPLETED

    def test_global_dashboard_aggregates_correctly(self) -> None:
        from jarvis.core.agent_heartbeat import (
            AgentHeartbeatConfig,
            AgentHeartbeatScheduler,
            AgentTask,
        )

        sched = AgentHeartbeatScheduler()
        for i in range(5):
            agent_id = f"agent_{i}"
            sched.configure_agent(AgentHeartbeatConfig(agent_id=agent_id))
            sched.add_task(AgentTask(task_id=f"task_{i}", agent_id=agent_id, name=f"Task {i}"))
            run = sched.start_task(agent_id, f"task_{i}")
            sched.complete_task(run, success=(i % 2 == 0))

        dashboard = sched.global_dashboard()
        assert dashboard["agent_count"] == 5
        assert dashboard["total_tasks"] == 5
        assert dashboard["total_runs"] == 5
        assert dashboard["total_fails"] == 2


# ============================================================================
# Multi-User-Isolation: Scoping
# ============================================================================


class TestConcurrentMultiUserIsolation:
    def test_different_users_same_agent_isolated(self) -> None:
        from jarvis.core.isolation import MultiUserIsolation

        iso = MultiUserIsolation()
        scope_alex = iso.get_or_create_scope("alex", "coder")
        scope_bob = iso.get_or_create_scope("bob", "coder")

        assert scope_alex.user_id == "alex"
        assert scope_bob.user_id == "bob"
        assert scope_alex.scope_key != scope_bob.scope_key

    def test_same_user_different_agents_isolated(self) -> None:
        from jarvis.core.isolation import MultiUserIsolation

        iso = MultiUserIsolation()
        scope_coder = iso.get_or_create_scope("alex", "coder")
        scope_researcher = iso.get_or_create_scope("alex", "researcher")

        assert scope_coder.agent_id == "coder"
        assert scope_researcher.agent_id == "researcher"
        assert scope_coder.scope_key != scope_researcher.scope_key

    def test_scope_reuse(self) -> None:
        from jarvis.core.isolation import MultiUserIsolation

        iso = MultiUserIsolation()
        scope1 = iso.get_or_create_scope("alex", "coder")
        scope2 = iso.get_or_create_scope("alex", "coder")
        assert scope1.scope_key == scope2.scope_key

    def test_many_concurrent_scopes(self) -> None:
        """100 User × 3 Agents = 300 isolierte Scopes."""
        from jarvis.core.isolation import MultiUserIsolation

        iso = MultiUserIsolation()
        scope_keys = set()

        for u in range(100):
            for agent in ["coder", "researcher", "assistant"]:
                scope = iso.get_or_create_scope(f"user_{u}", agent)
                scope_keys.add(scope.scope_key)

        assert len(scope_keys) == 300

    def test_stats_count_correct(self) -> None:
        from jarvis.core.isolation import MultiUserIsolation

        iso = MultiUserIsolation()
        for i in range(10):
            iso.get_or_create_scope(f"user_{i}", "coder")

        stats = iso.stats()
        assert stats["total_scopes"] == 10
        assert stats["unique_users"] == 10
        assert stats["unique_agents"] == 1


# ============================================================================
# Resource-Quotas: Konkurrenz
# ============================================================================


class TestConcurrentResourceQuotas:
    def test_quotas_independent_per_agent(self) -> None:
        from jarvis.core.isolation import AgentResourceQuota

        quota_a = AgentResourceQuota(agent_id="agent_a", daily_token_budget=1000)
        quota_b = AgentResourceQuota(agent_id="agent_b", daily_token_budget=500)

        quota_a.consume_tokens(800)
        assert quota_b.tokens_used_today == 0
        assert quota_a.tokens_used_today == 800

    def test_token_budget_enforcement(self) -> None:
        from jarvis.core.isolation import AgentResourceQuota

        quota = AgentResourceQuota(agent_id="limited", daily_token_budget=100)
        assert quota.check_token_budget(50)
        quota.consume_tokens(80)
        assert not quota.check_token_budget(50)
        assert quota.check_token_budget(20)

    def test_rate_limiter_per_agent(self) -> None:
        from jarvis.core.isolation import RateLimiter

        limiter = RateLimiter()

        for _ in range(5):
            assert limiter.check_and_consume("agent_a", max_per_minute=5)

        assert not limiter.check_and_consume("agent_a", max_per_minute=5)
        assert limiter.check_and_consume("agent_b", max_per_minute=5)


# ============================================================================
# RBAC: Konkurrenz
# ============================================================================


class TestConcurrentRBAC:
    def test_admin_and_viewer_coexist(self) -> None:
        from jarvis.gateway.wizards import RBACManager, UserRole

        rbac = RBACManager()
        rbac.add_user("admin", "Admin Alex", UserRole.ADMIN)
        rbac.add_user("viewer", "Viewer Bob", UserRole.VIEWER)

        assert rbac.check_permission("admin", "config", "write")
        assert not rbac.check_permission("viewer", "config", "write")
        assert rbac.check_permission("viewer", "config", "read")

    def test_agent_scope_filtering(self) -> None:
        from jarvis.gateway.wizards import DashboardUser, UserRole

        user = DashboardUser(
            user_id="bob", display_name="Bob", role=UserRole.USER,
            agent_scope=["coder", "assistant"],
        )
        assert user.can_access_agent("coder")
        assert user.can_access_agent("assistant")
        assert not user.can_access_agent("researcher")

    def test_role_upgrade_immediate(self) -> None:
        from jarvis.gateway.wizards import RBACManager, UserRole

        rbac = RBACManager()
        rbac.add_user("bob", "Bob", UserRole.VIEWER)
        assert not rbac.check_permission("bob", "agents", "write")
        rbac.update_role("bob", UserRole.ADMIN)
        assert rbac.check_permission("bob", "agents", "write")

    def test_multiple_admins_independent(self) -> None:
        from jarvis.gateway.wizards import RBACManager, UserRole

        rbac = RBACManager()
        rbac.add_user("a1", "Admin 1", UserRole.ADMIN)
        rbac.add_user("a2", "Admin 2", UserRole.ADMIN)
        rbac.add_user("v1", "Viewer 1", UserRole.VIEWER)

        rbac.remove_user("a1")
        assert rbac.check_permission("a2", "config", "write")
        assert rbac.get_user("a1") is None
