"""Tests für Konfigurations-Assistenten und RBAC.

Testet: HeartbeatWizard, BindingWizard, AgentWizard,
RBAC (UserRole, Permission, RBACManager), WizardRegistry.
"""

from __future__ import annotations

import pytest

from jarvis.gateway.wizards import (
    AgentWizard,
    BaseWizard,
    BindingWizard,
    DashboardUser,
    HeartbeatWizard,
    Permission,
    RBACManager,
    UserRole,
    WizardRegistry,
    WizardResult,
    WizardStep,
    WizardStepType,
    WizardTemplate,
)


# ============================================================================
# WizardStep
# ============================================================================


class TestWizardStep:
    def test_validate_required_missing(self) -> None:
        step = WizardStep(
            step_id="s1", title="Name", description="",
            field_type=WizardStepType.TEXT, field_name="name",
        )
        ok, msg = step.validate(None)
        assert not ok
        assert "erforderlich" in msg

    def test_validate_required_present(self) -> None:
        step = WizardStep(
            step_id="s1", title="Name", description="",
            field_type=WizardStepType.TEXT, field_name="name",
        )
        ok, _ = step.validate("test")
        assert ok

    def test_validate_number_invalid(self) -> None:
        step = WizardStep(
            step_id="s1", title="Count", description="",
            field_type=WizardStepType.NUMBER, field_name="count",
        )
        ok, msg = step.validate("abc")
        assert not ok
        assert "Zahl" in msg

    def test_validate_number_valid(self) -> None:
        step = WizardStep(
            step_id="s1", title="Count", description="",
            field_type=WizardStepType.NUMBER, field_name="count",
        )
        ok, _ = step.validate(42)
        assert ok

    def test_validate_select_invalid(self) -> None:
        step = WizardStep(
            step_id="s1", title="Channel", description="",
            field_type=WizardStepType.SELECT, field_name="ch",
            options=[{"value": "cli"}, {"value": "slack"}],
        )
        ok, _ = step.validate("telegram")
        assert not ok

    def test_validate_select_valid(self) -> None:
        step = WizardStep(
            step_id="s1", title="Channel", description="",
            field_type=WizardStepType.SELECT, field_name="ch",
            options=[{"value": "cli"}, {"value": "slack"}],
        )
        ok, _ = step.validate("slack")
        assert ok

    def test_validate_boolean_invalid(self) -> None:
        step = WizardStep(
            step_id="s1", title="Enabled", description="",
            field_type=WizardStepType.BOOLEAN, field_name="en",
        )
        ok, _ = step.validate("yes")
        assert not ok

    def test_validate_boolean_valid(self) -> None:
        step = WizardStep(
            step_id="s1", title="Enabled", description="",
            field_type=WizardStepType.BOOLEAN, field_name="en",
        )
        ok, _ = step.validate(True)
        assert ok

    def test_optional_none_ok(self) -> None:
        step = WizardStep(
            step_id="s1", title="Opt", description="",
            field_type=WizardStepType.TEXT, field_name="opt",
            required=False,
        )
        ok, _ = step.validate(None)
        assert ok

    def test_to_dict(self) -> None:
        step = WizardStep(
            step_id="s1", title="Name", description="desc",
            field_type=WizardStepType.TEXT, field_name="name",
            tooltip="Hilfe",
        )
        d = step.to_dict()
        assert d["step_id"] == "s1"
        assert d["tooltip"] == "Hilfe"
        assert d["field_type"] == "text"


# ============================================================================
# HeartbeatWizard
# ============================================================================


class TestHeartbeatWizard:
    def test_has_steps(self) -> None:
        wiz = HeartbeatWizard()
        assert wiz.step_count >= 4

    def test_has_templates(self) -> None:
        wiz = HeartbeatWizard()
        assert len(wiz.templates) >= 3

    def test_template_daily_briefing(self) -> None:
        wiz = HeartbeatWizard()
        t = wiz.get_template("daily_briefing")
        assert t is not None
        assert t.preset_values["channel"] == "slack"

    def test_apply_template(self) -> None:
        wiz = HeartbeatWizard()
        values = wiz.apply_template("daily_briefing")
        assert values["enabled"] is True
        assert values["channel"] == "slack"

    def test_generate_config(self) -> None:
        wiz = HeartbeatWizard()
        result = wiz.generate_config({
            "enabled": True,
            "interval_minutes": 30,
            "channel": "cli",
        })
        assert result.valid
        assert result.config_patch["heartbeat"]["interval_minutes"] == 30

    def test_generate_config_warning_short_interval(self) -> None:
        wiz = HeartbeatWizard()
        result = wiz.generate_config({
            "enabled": True,
            "interval_minutes": 2,
            "channel": "cli",
        })
        assert any("Intervall" in w or "Performance" in w for w in result.warnings)

    def test_validate_step(self) -> None:
        wiz = HeartbeatWizard()
        ok, _ = wiz.validate_step("hb_enabled", True)
        assert ok
        ok, _ = wiz.validate_step("hb_enabled", "not_bool")
        assert not ok

    def test_wizard_type(self) -> None:
        wiz = HeartbeatWizard()
        assert wiz.wizard_type == "heartbeat"

    def test_to_dict(self) -> None:
        wiz = HeartbeatWizard()
        d = wiz.to_dict()
        assert d["wizard_type"] == "heartbeat"
        assert len(d["steps"]) >= 4
        assert len(d["templates"]) >= 3


# ============================================================================
# BindingWizard
# ============================================================================


class TestBindingWizard:
    def test_has_steps(self) -> None:
        wiz = BindingWizard()
        assert wiz.step_count >= 5

    def test_has_templates(self) -> None:
        wiz = BindingWizard()
        assert len(wiz.templates) >= 3

    def test_template_slash_code(self) -> None:
        wiz = BindingWizard()
        t = wiz.get_template("slash_code")
        assert t is not None
        assert t.preset_values["target_agent"] == "coder"

    def test_generate_config(self) -> None:
        wiz = BindingWizard()
        result = wiz.generate_config({
            "name": "test_binding",
            "target_agent": "coder",
            "command_prefix": "/code",
        })
        assert result.config_patch["bindings"][0]["name"] == "test_binding"
        assert result.config_patch["bindings"][0]["target_agent"] == "coder"

    def test_generate_config_with_regex(self) -> None:
        wiz = BindingWizard()
        result = wiz.generate_config({
            "name": "bugs",
            "target_agent": "coder",
            "regex_pattern": "(?i)bug|fehler",
        })
        assert result.config_patch["bindings"][0]["regex_pattern"] == "(?i)bug|fehler"

    def test_wizard_type(self) -> None:
        assert BindingWizard().wizard_type == "binding"


# ============================================================================
# AgentWizard
# ============================================================================


class TestAgentWizard:
    def test_has_steps(self) -> None:
        wiz = AgentWizard()
        assert wiz.step_count >= 5

    def test_has_templates(self) -> None:
        wiz = AgentWizard()
        assert len(wiz.templates) >= 3

    def test_template_coder(self) -> None:
        wiz = AgentWizard()
        t = wiz.get_template("coder")
        assert t is not None
        assert "code" in t.preset_values.get("preferred_model", "").lower() or \
               t.preset_values.get("name") == "coder"

    def test_generate_config_standard_sandbox(self) -> None:
        wiz = AgentWizard()
        result = wiz.generate_config({
            "name": "test_agent",
            "system_prompt": "Test prompt",
            "sandbox_profile": "standard",
        })
        agent = result.config_patch["agents"][0]
        assert agent["sandbox_network"] == "allow"
        assert agent["sandbox_max_memory_mb"] == 512

    def test_generate_config_minimal_sandbox(self) -> None:
        wiz = AgentWizard()
        result = wiz.generate_config({
            "name": "safe_agent",
            "system_prompt": "Restricted",
            "sandbox_profile": "minimal",
        })
        agent = result.config_patch["agents"][0]
        assert agent["sandbox_network"] == "block"
        assert agent["sandbox_max_memory_mb"] == 256

    def test_generate_config_full_sandbox(self) -> None:
        wiz = AgentWizard()
        result = wiz.generate_config({
            "name": "power_agent",
            "system_prompt": "Full power",
            "sandbox_profile": "full",
        })
        agent = result.config_patch["agents"][0]
        assert agent["sandbox_max_memory_mb"] == 8192

    def test_template_family_assistant(self) -> None:
        wiz = AgentWizard()
        t = wiz.get_template("family")
        assert t is not None
        assert "family" in t.preset_values.get("name", "").lower() or \
               "Familie" in t.description

    def test_wizard_type(self) -> None:
        assert AgentWizard().wizard_type == "agent"


# ============================================================================
# RBAC
# ============================================================================


class TestPermission:
    def test_key(self) -> None:
        p = Permission("config", "write")
        assert p.key == "config:write"


class TestDashboardUser:
    def test_owner_has_all_permissions(self) -> None:
        user = DashboardUser(user_id="u1", display_name="Admin", role=UserRole.OWNER)
        assert user.has_permission("config", "write")
        assert user.has_permission("users", "delete")
        assert user.has_permission("credentials", "write")

    def test_viewer_read_only(self) -> None:
        user = DashboardUser(user_id="u2", display_name="Viewer", role=UserRole.VIEWER)
        assert user.has_permission("config", "read")
        assert not user.has_permission("config", "write")
        assert not user.has_permission("agents", "write")

    def test_user_can_execute_agents(self) -> None:
        user = DashboardUser(user_id="u3", display_name="User", role=UserRole.USER)
        assert user.has_permission("agents", "execute")
        assert not user.has_permission("config", "write")

    def test_operator_monitoring_access(self) -> None:
        user = DashboardUser(user_id="u4", display_name="Ops", role=UserRole.OPERATOR)
        assert user.has_permission("monitoring", "read")
        assert user.has_permission("audit", "read")
        assert not user.has_permission("config", "write")

    def test_agent_scope_admin_sees_all(self) -> None:
        user = DashboardUser(user_id="u5", display_name="Admin", role=UserRole.ADMIN)
        assert user.can_access_agent("any_agent")

    def test_agent_scope_user_restricted(self) -> None:
        user = DashboardUser(
            user_id="u6", display_name="User", role=UserRole.USER,
            agent_scope=["coder", "researcher"],
        )
        assert user.can_access_agent("coder")
        assert not user.can_access_agent("admin_agent")

    def test_agent_scope_empty_sees_all(self) -> None:
        user = DashboardUser(user_id="u7", display_name="User", role=UserRole.USER)
        assert user.can_access_agent("anything")

    def test_to_dict(self) -> None:
        user = DashboardUser(
            user_id="u8", display_name="Alex", role=UserRole.ADMIN, email="a@b.de",
        )
        d = user.to_dict()
        assert d["role"] == "admin"
        assert d["email"] == "a@b.de"
        assert len(d["permissions"]) > 0


class TestRBACManager:
    def test_add_and_get_user(self) -> None:
        rbac = RBACManager()
        rbac.add_user("u1", "Alex", UserRole.ADMIN)
        user = rbac.get_user("u1")
        assert user is not None
        assert user.role == UserRole.ADMIN

    def test_remove_user(self) -> None:
        rbac = RBACManager()
        rbac.add_user("u1", "Alex", UserRole.ADMIN)
        assert rbac.remove_user("u1")
        assert rbac.get_user("u1") is None

    def test_remove_nonexistent(self) -> None:
        rbac = RBACManager()
        assert not rbac.remove_user("nope")

    def test_update_role(self) -> None:
        rbac = RBACManager()
        rbac.add_user("u1", "Alex", UserRole.USER)
        rbac.update_role("u1", UserRole.ADMIN)
        assert rbac.get_user("u1").role == UserRole.ADMIN

    def test_update_role_nonexistent(self) -> None:
        rbac = RBACManager()
        assert not rbac.update_role("nope", UserRole.ADMIN)

    def test_check_permission(self) -> None:
        rbac = RBACManager()
        rbac.add_user("admin", "Admin", UserRole.ADMIN)
        rbac.add_user("viewer", "Viewer", UserRole.VIEWER)
        assert rbac.check_permission("admin", "config", "write")
        assert not rbac.check_permission("viewer", "config", "write")

    def test_check_permission_unknown_user(self) -> None:
        rbac = RBACManager()
        assert not rbac.check_permission("ghost", "config", "read")

    def test_list_users_all(self) -> None:
        rbac = RBACManager()
        rbac.add_user("u1", "A", UserRole.ADMIN)
        rbac.add_user("u2", "B", UserRole.USER)
        assert len(rbac.list_users()) == 2

    def test_list_users_by_role(self) -> None:
        rbac = RBACManager()
        rbac.add_user("u1", "A", UserRole.ADMIN)
        rbac.add_user("u2", "B", UserRole.USER)
        rbac.add_user("u3", "C", UserRole.ADMIN)
        assert len(rbac.list_users(role=UserRole.ADMIN)) == 2

    def test_user_count(self) -> None:
        rbac = RBACManager()
        rbac.add_user("u1", "A", UserRole.ADMIN)
        assert rbac.user_count == 1

    def test_roles_summary(self) -> None:
        rbac = RBACManager()
        rbac.add_user("u1", "A", UserRole.ADMIN)
        rbac.add_user("u2", "B", UserRole.ADMIN)
        rbac.add_user("u3", "C", UserRole.USER)
        summary = rbac.roles_summary()
        assert summary["admin"] == 2
        assert summary["user"] == 1


# ============================================================================
# WizardRegistry
# ============================================================================


class TestWizardRegistry:
    def test_default_wizards_registered(self) -> None:
        reg = WizardRegistry()
        assert reg.wizard_count == 3

    def test_get_heartbeat(self) -> None:
        reg = WizardRegistry()
        wiz = reg.get("heartbeat")
        assert wiz is not None
        assert wiz.wizard_type == "heartbeat"

    def test_get_binding(self) -> None:
        reg = WizardRegistry()
        assert reg.get("binding") is not None

    def test_get_agent(self) -> None:
        reg = WizardRegistry()
        assert reg.get("agent") is not None

    def test_get_nonexistent(self) -> None:
        reg = WizardRegistry()
        assert reg.get("nope") is None

    def test_list_wizards(self) -> None:
        reg = WizardRegistry()
        wizards = reg.list_wizards()
        assert len(wizards) == 3
        types = [w["wizard_type"] for w in wizards]
        assert "heartbeat" in types
        assert "binding" in types
        assert "agent" in types

    def test_run_wizard(self) -> None:
        reg = WizardRegistry()
        result = reg.run_wizard("heartbeat", {
            "enabled": True,
            "interval_minutes": 30,
            "channel": "cli",
        })
        assert result is not None
        assert result.valid

    def test_run_wizard_nonexistent(self) -> None:
        reg = WizardRegistry()
        assert reg.run_wizard("nope", {}) is None

    def test_register_custom_wizard(self) -> None:
        reg = WizardRegistry()

        class CustomWizard(BaseWizard):
            wizard_type = "custom"

        reg.register(CustomWizard())
        assert reg.wizard_count == 4
        assert reg.get("custom") is not None
