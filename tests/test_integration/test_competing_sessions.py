"""Tests für konkurrierende Agent-Sessions.

Prüft die Isolation unter Last:
  - WorkspaceGuard blockiert Cross-Agent-Zugriffe
  - Rate-Limiter und Quotas pro Agent
  - Auth-Gateway isoliert Tokens und Sessions korrekt
  - Per-Agent-Heartbeat-Tasks laufen isoliert
  - Interaction-States gehören zum richtigen Agent
  - RBAC unter konkurrierenden Zugriffen
"""

from __future__ import annotations

from pathlib import Path

import pytest


# ============================================================================
# 1. MultiUserIsolation: Scoped Sessions
# ============================================================================


class TestConcurrentSessionIsolation:
    """Mehrere Agents dürfen nicht auf fremde Scopes zugreifen."""

    def test_multi_user_isolation_scopes(self) -> None:
        from jarvis.core.isolation import MultiUserIsolation

        iso = MultiUserIsolation()
        scope_a = iso.get_or_create_scope("alex", "coder")
        scope_b = iso.get_or_create_scope("alex", "researcher")
        scope_c = iso.get_or_create_scope("bob", "coder")

        assert scope_a.agent_id == "coder"
        assert scope_b.agent_id == "researcher"
        assert scope_a.scope_key != scope_b.scope_key
        assert scope_a.scope_key != scope_c.scope_key

    def test_10_agents_isolated(self) -> None:
        from jarvis.core.isolation import MultiUserIsolation

        iso = MultiUserIsolation()
        agents = [f"agent_{i}" for i in range(10)]
        scopes = {a: iso.get_or_create_scope("user1", a) for a in agents}

        keys = [s.scope_key for s in scopes.values()]
        assert len(set(keys)) == 10

    def test_same_user_different_agents_no_data_leak(self) -> None:
        from jarvis.core.isolation import MultiUserIsolation

        iso = MultiUserIsolation()
        scope_coder = iso.get_or_create_scope("alex", "coder")
        scope_research = iso.get_or_create_scope("alex", "researcher")

        scope_coder.session_ids.append("session_code_1")
        scope_research.session_ids.append("session_research_1")

        assert "session_code_1" not in scope_research.session_ids
        assert "session_research_1" not in scope_coder.session_ids


# ============================================================================
# 2. Workspace-Guard: Cross-Agent-Zugriffe
# ============================================================================


class TestConcurrentWorkspaceGuard:
    def test_registered_agent_allowed_in_own_workspace(self, tmp_path: Path) -> None:
        from jarvis.core.isolation import WorkspaceGuard

        guard = WorkspaceGuard(tmp_path)
        guard.register_agent("coder")

        assert guard.check_access("coder", tmp_path / "coder" / "file.py")

    def test_agent_blocked_from_other_workspace(self, tmp_path: Path) -> None:
        from jarvis.core.isolation import WorkspaceGuard

        guard = WorkspaceGuard(tmp_path)
        guard.register_agent("coder")
        guard.register_agent("researcher")

        blocked = guard.check_access("coder", tmp_path / "researcher" / "file.py")
        assert not blocked

    def test_violations_logged(self, tmp_path: Path) -> None:
        from jarvis.core.isolation import WorkspaceGuard

        guard = WorkspaceGuard(tmp_path)
        guard.register_agent("coder")
        guard.register_agent("researcher")
        guard.check_access("coder", tmp_path / "researcher" / "secrets.json")

        violations = guard.violations()
        assert len(violations) >= 1
        assert violations[0]["agent_id"] == "coder"

    def test_multiple_agents_separate_workspaces(self, tmp_path: Path) -> None:
        from jarvis.core.isolation import WorkspaceGuard

        guard = WorkspaceGuard(tmp_path)
        agents = ["agent_a", "agent_b", "agent_c", "agent_d", "agent_e"]
        for a in agents:
            guard.register_agent(a)

        for agent in agents:
            assert guard.check_access(agent, tmp_path / agent / "data.db")
            for other in agents:
                if other != agent:
                    assert not guard.check_access(agent, tmp_path / other / "data.db")

    def test_unregistered_agent_blocked(self, tmp_path: Path) -> None:
        from jarvis.core.isolation import WorkspaceGuard

        guard = WorkspaceGuard(tmp_path)
        assert not guard.check_access("unknown", tmp_path / "unknown" / "file.py")


# ============================================================================
# 3. Rate-Limiter & Quotas pro Agent
# ============================================================================


class TestConcurrentQuotas:
    def test_independent_rate_limits(self) -> None:
        from jarvis.core.isolation import RateLimiter

        limiter = RateLimiter()
        # Agent A: 3 Anfragen mit Limit 3/min
        for _ in range(3):
            assert limiter.check_and_consume("agent_a", max_per_minute=3)
        # Agent A: Limit erreicht
        assert not limiter.check_and_consume("agent_a", max_per_minute=3)
        # Agent B: Eigenes Limit, noch frei
        assert limiter.check_and_consume("agent_b", max_per_minute=3)

    def test_quota_per_agent(self) -> None:
        from jarvis.core.isolation import AgentResourceQuota

        quota_a = AgentResourceQuota(agent_id="coder", daily_token_budget=1000)
        quota_b = AgentResourceQuota(agent_id="researcher", daily_token_budget=1000)

        quota_a.consume_tokens(800)
        quota_b.consume_tokens(200)

        assert not quota_a.check_token_budget(300)  # 800 + 300 > 1000
        assert quota_b.check_token_budget(300)        # 200 + 300 <= 1000

    def test_quota_utilization(self) -> None:
        from jarvis.core.isolation import AgentResourceQuota

        quota = AgentResourceQuota(agent_id="test", daily_token_budget=1000)
        quota.consume_tokens(500)
        assert quota.budget_utilization_percent == 50.0
        assert quota.tokens_remaining == 500


# ============================================================================
# 4. Auth-Gateway: Token- und Session-Isolation
# ============================================================================


class TestConcurrentAuthGateway:
    def test_sso_creates_per_agent_tokens(self) -> None:
        from jarvis.gateway.auth import AuthGateway

        gw = AuthGateway()
        result = gw.login("alex", ["coder", "researcher", "assistant"])

        assert len(result) == 3
        tokens = {aid: gw.validate_token(raw) for aid, (raw, _) in result.items()}
        agent_ids = {t.agent_id for t in tokens.values() if t}
        assert agent_ids == {"coder", "researcher", "assistant"}

    def test_token_from_agent_a_cannot_act_as_agent_b(self) -> None:
        from jarvis.gateway.auth import AuthGateway

        gw = AuthGateway()
        result = gw.login("alex", ["coder", "researcher"])

        raw_coder, _ = result["coder"]
        token = gw.validate_token(raw_coder)
        assert token.agent_id == "coder"
        assert token.agent_id != "researcher"

    def test_revoke_one_agent_keeps_others(self) -> None:
        from jarvis.gateway.auth import AuthGateway

        gw = AuthGateway()
        result = gw.login("alex", ["coder", "researcher"])

        raw_coder, _ = result["coder"]
        raw_researcher, _ = result["researcher"]

        token = gw.validate_token(raw_coder)
        gw.revoke_token(token.token_id)

        assert gw.validate_token(raw_coder) is None
        assert gw.validate_token(raw_researcher) is not None

    def test_concurrent_sessions_per_user(self) -> None:
        from jarvis.gateway.auth import AuthGateway

        gw = AuthGateway()
        gw.login("alex", ["coder", "researcher", "assistant"])

        sessions = gw.active_sessions("alex")
        assert len(sessions) == 3
        agent_ids = {s.agent_id for s in sessions}
        assert agent_ids == {"coder", "researcher", "assistant"}

    def test_logout_ends_all_agent_sessions(self) -> None:
        from jarvis.gateway.auth import AuthGateway

        gw = AuthGateway()
        gw.login("alex", ["coder", "researcher"])
        assert len(gw.active_sessions("alex")) == 2

        gw.logout("alex")
        assert len(gw.active_sessions("alex")) == 0

    def test_scoped_tokens_respect_boundaries(self) -> None:
        from jarvis.gateway.auth import AuthGateway

        gw = AuthGateway()
        _, token = gw.create_token("alex", "coder", scopes=["read", "execute"])
        assert gw.check_scope(token, "read")
        assert gw.check_scope(token, "execute")
        assert not gw.check_scope(token, "admin")


# ============================================================================
# 5. Per-Agent-Heartbeat-Isolation
# ============================================================================


class TestConcurrentHeartbeat:
    def test_tasks_isolated_per_agent(self) -> None:
        from jarvis.core.agent_heartbeat import AgentHeartbeatScheduler, AgentTask

        sched = AgentHeartbeatScheduler()
        sched.add_task(AgentTask(task_id="build", agent_id="coder", name="Build"))
        sched.add_task(AgentTask(task_id="search", agent_id="researcher", name="Search"))
        sched.add_task(AgentTask(task_id="test", agent_id="coder", name="Test"))

        assert len(sched.agent_tasks("coder")) == 2
        assert len(sched.agent_tasks("researcher")) == 1

        coder_ids = {t.task_id for t in sched.agent_tasks("coder")}
        assert "search" not in coder_ids

    def test_task_execution_scoped(self) -> None:
        from jarvis.core.agent_heartbeat import AgentHeartbeatScheduler, AgentTask

        sched = AgentHeartbeatScheduler()
        sched.add_task(AgentTask(task_id="t1", agent_id="coder", name="Build"))
        sched.add_task(AgentTask(task_id="t2", agent_id="researcher", name="Search"))

        run = sched.start_task("coder", "t1")
        sched.complete_task(run, success=True)

        assert sched.get_task("coder", "t1").run_count == 1
        assert sched.get_task("researcher", "t2").run_count == 0

    def test_dashboard_shows_all_agents(self) -> None:
        from jarvis.core.agent_heartbeat import (
            AgentHeartbeatConfig,
            AgentHeartbeatScheduler,
            AgentTask,
        )

        sched = AgentHeartbeatScheduler()
        for agent in ["coder", "researcher", "assistant"]:
            sched.configure_agent(AgentHeartbeatConfig(agent_id=agent))
            sched.add_task(AgentTask(
                task_id=f"{agent}_task", agent_id=agent, name=f"{agent} Task",
            ))

        dash = sched.global_dashboard()
        assert dash["agent_count"] == 3
        assert dash["total_tasks"] == 3
        for summary in dash["agents"]:
            assert summary["task_count"] == 1


# ============================================================================
# 6. Interaction-State per Agent
# ============================================================================


class TestConcurrentInteractionState:
    def test_interactions_scoped_to_agent(self) -> None:
        from jarvis.channels.commands import InteractionStore, InteractionType

        store = InteractionStore()
        s1 = store.create(InteractionType.BUTTON_CLICK, "u1", agent_id="coder")
        s2 = store.create(InteractionType.APPROVAL, "u1", agent_id="researcher")

        assert s1.agent_id == "coder"
        assert s2.agent_id == "researcher"
        assert s1.interaction_id != s2.interaction_id

    def test_resolve_only_own_interaction(self) -> None:
        from jarvis.channels.commands import InteractionStore, InteractionType

        store = InteractionStore()
        s_coder = store.create(InteractionType.BUTTON_CLICK, "u1", agent_id="coder")
        s_researcher = store.create(InteractionType.BUTTON_CLICK, "u1", agent_id="researcher")

        store.resolve(s_coder.interaction_id, result="approved")

        assert s_coder.resolved
        assert not s_researcher.resolved

    def test_many_agents_pending_interactions(self) -> None:
        from jarvis.channels.commands import InteractionStore, InteractionType

        store = InteractionStore()
        for i in range(10):
            for j in range(5):
                store.create(InteractionType.BUTTON_CLICK, f"user_{i}", agent_id=f"agent_{i}")

        assert store.pending_count() == 50


# ============================================================================
# 7. RBAC unter konkurrierenden Zugriffen
# ============================================================================


class TestConcurrentRBAC:
    def test_user_role_restricts_agent_access(self) -> None:
        from jarvis.gateway.wizards import DashboardUser, UserRole

        admin = DashboardUser(user_id="admin", display_name="Admin", role=UserRole.ADMIN)
        user = DashboardUser(
            user_id="alex", display_name="Alex", role=UserRole.USER,
            agent_scope=["coder"],
        )

        assert admin.can_access_agent("coder")
        assert admin.can_access_agent("researcher")
        assert user.can_access_agent("coder")
        assert not user.can_access_agent("researcher")

    def test_multiple_users_different_permissions(self) -> None:
        from jarvis.gateway.wizards import RBACManager, UserRole

        rbac = RBACManager()
        rbac.add_user("admin", "Admin", UserRole.ADMIN)
        rbac.add_user("operator", "Ops", UserRole.OPERATOR)
        rbac.add_user("viewer", "Viewer", UserRole.VIEWER)

        assert rbac.check_permission("admin", "config", "write")
        assert not rbac.check_permission("operator", "config", "write")
        assert rbac.check_permission("operator", "agents", "execute")
        assert not rbac.check_permission("viewer", "agents", "write")
        assert rbac.check_permission("viewer", "config", "read")


# ============================================================================
# 8. End-to-End: Vollständiger Isolations-Durchlauf
# ============================================================================


class TestEndToEndIsolation:
    def test_full_isolation_scenario(self, tmp_path: Path) -> None:
        """Kompletter Durchlauf: 3 Agents, vollständig isoliert."""
        from jarvis.core.agent_heartbeat import (
            AgentHeartbeatConfig,
            AgentHeartbeatScheduler,
            AgentTask,
        )
        from jarvis.core.isolation import MultiUserIsolation, WorkspaceGuard
        from jarvis.gateway.auth import AuthGateway

        auth = AuthGateway()
        isolation = MultiUserIsolation()
        guard = WorkspaceGuard(tmp_path)
        heartbeat = AgentHeartbeatScheduler()

        agents = ["coder", "researcher", "assistant"]

        # Alle Agents registrieren
        for a in agents:
            guard.register_agent(a)

        # SSO-Login
        login_result = auth.login("alex", agents)
        assert len(login_result) == 3

        for agent_id in agents:
            raw_token, session = login_result[agent_id]

            # Isolation-Scope
            scope = isolation.get_or_create_scope("alex", agent_id)
            assert scope.agent_id == agent_id

            # Workspace: eigenes erlaubt
            assert guard.check_access(agent_id, tmp_path / agent_id / "work.py")
            # Workspace: fremdes blockiert
            for other in agents:
                if other != agent_id:
                    assert not guard.check_access(agent_id, tmp_path / other / "work.py")

            # Token validieren
            token = auth.validate_token(raw_token)
            assert token is not None
            assert token.agent_id == agent_id

            # Heartbeat-Task
            heartbeat.configure_agent(AgentHeartbeatConfig(agent_id=agent_id))
            heartbeat.add_task(AgentTask(
                task_id=f"{agent_id}_main", agent_id=agent_id, name=f"{agent_id} Task",
            ))

        # Verify
        assert heartbeat.global_dashboard()["agent_count"] == 3
        assert len(auth.active_sessions("alex")) == 3
        assert isolation.stats()["total_scopes"] >= 3
        # 6 Violations: jeder Agent x 2 fremde Workspaces
        assert len(guard.violations()) == 6

    def test_two_users_five_agents_isolation(self) -> None:
        """2 Users mit je 5 Agents -> vollständig isoliert."""
        from jarvis.core.isolation import MultiUserIsolation
        from jarvis.gateway.auth import AuthGateway

        auth = AuthGateway()
        isolation = MultiUserIsolation()

        for user in ["alex", "bob"]:
            agents = [f"{user}_agent_{i}" for i in range(5)]
            auth.login(user, agents)
            for agent_id in agents:
                isolation.get_or_create_scope(user, agent_id)

        all_sessions = auth.active_sessions("alex") + auth.active_sessions("bob")
        assert len(all_sessions) == 10

        stats = isolation.stats()
        assert stats["total_scopes"] >= 10
