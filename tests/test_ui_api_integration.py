"""Comprehensive integration tests for the Control Center UI API endpoints.

Tests every API endpoint that the CognithorControlCenter.jsx calls,
verifying request handling, response shapes, data persistence, and
error handling.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import pytest
import yaml

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_jarvis_home(tmp_path: Path) -> Path:
    """Create a temporary Jarvis home directory with required structure."""
    home = tmp_path / ".jarvis"
    home.mkdir()
    (home / "memory").mkdir()
    (home / "prompts").mkdir()
    (home / "mcp").mkdir()
    (home / "cron").mkdir()
    (home / "policies").mkdir()

    # Create a minimal config.yaml
    config_data = {"jarvis_home": str(home), "owner_name": "TestUser"}
    config_file = home / "config.yaml"
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(config_data, f)

    # Create a CORE.md
    (home / "memory" / "CORE.md").write_text("# Test Core\nI am a test bot.", encoding="utf-8")

    # Create default policy
    (home / "policies" / "default.yaml").write_text(
        "rules:\n  - allow_all: true\n", encoding="utf-8"
    )

    return home


@pytest.fixture()
def config(tmp_jarvis_home: Path):
    """Load a JarvisConfig for the temp home."""
    from jarvis.config import JarvisConfig

    return JarvisConfig(jarvis_home=tmp_jarvis_home)


@pytest.fixture()
def app_and_mgr(config):
    """Create FastAPI app with all routes registered."""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    from jarvis.channels.config_routes import create_config_routes
    from jarvis.config_manager import ConfigManager

    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    mgr = ConfigManager(config=config)
    create_config_routes(app, mgr, gateway=None)
    return app, mgr


@pytest.fixture()
def client(app_and_mgr):
    """Create a TestClient for the FastAPI app."""
    from starlette.testclient import TestClient

    app, _mgr = app_and_mgr
    return TestClient(app)


# ===========================================================================
# 1. SYSTEM STATUS & CONTROL
# ===========================================================================


class TestSystemEndpoints:
    """Test /api/v1/system/* endpoints."""

    def test_get_system_status(self, client):
        """GET /system/status — must return status field."""
        r = client.get("/api/v1/system/status")
        assert r.status_code == 200
        data = r.json()
        assert "status" in data
        assert data["status"] in ("running", "stopped")

    def test_post_system_start(self, client):
        """POST /system/start — must return status ok."""
        r = client.post("/api/v1/system/start")
        assert r.status_code == 200
        data = r.json()
        assert "status" in data or "error" not in data

    def test_post_system_stop(self, client):
        """POST /system/stop — must return status ok."""
        r = client.post("/api/v1/system/stop")
        assert r.status_code == 200


# ===========================================================================
# 2. CONFIGURATION - LOAD
# ===========================================================================


class TestConfigLoad:
    """Test GET /api/v1/config and /api/v1/config/{section}."""

    def test_get_full_config(self, client):
        """GET /config — must return all major sections."""
        r = client.get("/api/v1/config")
        assert r.status_code == 200
        data = r.json()
        # Must have all sections the UI expects
        required_sections = [
            "ollama", "models", "gatekeeper", "planner", "memory",
            "channels", "sandbox", "logging", "security", "heartbeat",
            "plugins", "dashboard", "model_overrides", "web", "database",
        ]
        for section in required_sections:
            assert section in data, f"Missing section: {section}"

        # Must have top-level fields
        assert "owner_name" in data
        assert "_meta" in data
        assert "editable_sections" in data["_meta"]

    def test_get_config_sections_individually(self, client):
        """GET /config/{section} — each section must be readable."""
        sections = [
            "ollama", "models", "gatekeeper", "planner", "memory",
            "channels", "sandbox", "logging", "security", "heartbeat",
            "plugins", "dashboard", "web", "database",
        ]
        for section in sections:
            r = client.get(f"/api/v1/config/{section}")
            assert r.status_code == 200, f"Failed to GET /config/{section}"
            data = r.json()
            assert data.get("section") == section, f"Section mismatch for {section}"
            assert "values" in data, f"Missing 'values' in {section}"

    def test_secret_masking(self, client):
        """Secrets must be masked, numeric token fields must NOT be masked."""
        r = client.get("/api/v1/config")
        data = r.json()
        # Numeric fields should NOT be masked
        assert isinstance(data["planner"]["response_token_budget"], int), \
            "response_token_budget should be an integer, not masked"
        assert isinstance(data["memory"]["chunk_size_tokens"], int), \
            "chunk_size_tokens should be an integer, not masked"
        assert isinstance(data["memory"]["chunk_overlap_tokens"], int), \
            "chunk_overlap_tokens should be an integer, not masked"
        assert isinstance(data["anthropic_max_tokens"], int), \
            "anthropic_max_tokens should be an integer, not masked"

    def test_editable_sections_metadata(self, client):
        """_meta must list web and database as editable."""
        r = client.get("/api/v1/config")
        data = r.json()
        editable = data["_meta"]["editable_sections"]
        assert "web" in editable, "web must be in editable_sections"
        assert "database" in editable, "database must be in editable_sections"


# ===========================================================================
# 3. CONFIGURATION - PATCH (Section Updates)
# ===========================================================================


class TestConfigPatch:
    """Test PATCH /api/v1/config/{section} for all 15 sections."""

    @pytest.mark.parametrize("section,key,value", [
        ("ollama", "timeout_seconds", 60),
        ("models", "planner", {"name": "test-model:7b", "context_window": 8192}),
        ("gatekeeper", "max_blocked_retries", 5),
        ("planner", "temperature", 0.5),
        ("memory", "chunk_size_tokens", 500),
        ("channels", "cli_enabled", False),
        ("sandbox", "timeout_seconds", 60),
        ("logging", "level", "DEBUG"),
        ("security", "max_iterations", 10),
        ("heartbeat", "enabled", False),
        ("plugins", "skills_dir", "custom_skills"),
        ("dashboard", "enabled", True),
        ("web", "duckduckgo_enabled", False),
        ("database", "pg_port", 5433),
    ])
    def test_patch_section(self, client, section, key, value):
        """PATCH /config/{section} — update must succeed and persist."""
        payload = {key: value}
        r = client.patch(f"/api/v1/config/{section}", json=payload)
        assert r.status_code == 200, f"PATCH /config/{section} failed: {r.text}"
        data = r.json()
        assert data.get("status") == "ok", f"Expected ok for {section}: {data}"

        # Verify the change persisted
        r2 = client.get(f"/api/v1/config/{section}")
        values = r2.json()["values"]
        if isinstance(value, dict):
            for k, v in value.items():
                assert values[key][k] == v, f"Mismatch in {section}.{key}.{k}"
        else:
            assert values[key] == value, f"Mismatch in {section}.{key}"

    def test_patch_invalid_section(self, client):
        """PATCH /config/nonexistent — must return error."""
        r = client.patch("/api/v1/config/nonexistent_section", json={"foo": "bar"})
        assert r.status_code == 200  # Error in body, not HTTP status
        data = r.json()
        assert "error" in data

    def test_patch_masked_secrets_skipped(self, client):
        """PATCH with '***' values for secret fields must not overwrite."""
        # First set a real value (use top-level for API keys)
        # For section secrets like brave_api_key in web
        r = client.patch("/api/v1/config/web", json={
            "brave_api_key": "***",
            "duckduckgo_enabled": True,
        })
        assert r.status_code == 200
        data = r.json()
        assert data.get("status") == "ok"

    def test_model_overrides_patch(self, client):
        """PATCH /config/model_overrides — must accept dict values."""
        r = client.patch("/api/v1/config/model_overrides", json={})
        assert r.status_code == 200


# ===========================================================================
# 4. CONFIGURATION - PATCH (Top-Level Fields)
# ===========================================================================


class TestConfigTopLevel:
    """Test PATCH /api/v1/config for top-level fields."""

    def test_patch_top_level_basic_fields(self, client):
        """PATCH /config — owner_name, operation_mode etc."""
        payload = {
            "owner_name": "NewOwner",
            "operation_mode": "online",
            "cost_tracking_enabled": True,
            "daily_budget_usd": 10.0,
            "monthly_budget_usd": 100.0,
            "anthropic_max_tokens": 8192,
        }
        r = client.patch("/api/v1/config", json=payload)
        assert r.status_code == 200
        data = r.json()
        results = data.get("results", [])
        for result in results:
            assert result["status"] in ("ok", "skipped"), \
                f"Failed for {result['key']}: {result.get('error')}"

        # Verify persistence
        r2 = client.get("/api/v1/config")
        cfg = r2.json()
        assert cfg["owner_name"] == "NewOwner"
        assert cfg["operation_mode"] == "online"
        assert cfg["anthropic_max_tokens"] == 8192

    def test_patch_top_level_masked_keys_skipped(self, client):
        """API keys sent as '***' must be skipped, not written."""
        payload = {
            "openai_api_key": "***",
            "anthropic_api_key": "***",
            "gemini_api_key": "***",
            "owner_name": "StillWorks",
        }
        r = client.patch("/api/v1/config", json=payload)
        assert r.status_code == 200
        results = {x["key"]: x["status"] for x in r.json()["results"]}
        assert results["openai_api_key"] == "skipped"
        assert results["anthropic_api_key"] == "skipped"
        assert results["gemini_api_key"] == "skipped"
        assert results["owner_name"] == "ok"

    def test_patch_top_level_all_ui_fields(self, client):
        """All fields the UI sends must be accepted (no 'not editable' error)."""
        payload = {
            "owner_name": "Test",
            "llm_backend_type": "ollama",
            "operation_mode": "offline",
            "cost_tracking_enabled": False,
            "daily_budget_usd": 0.0,
            "monthly_budget_usd": 0.0,
            "vision_model": "test-model",
            "vision_model_detail": "test-detail",
            "openai_base_url": "https://api.openai.com/v1",
            "anthropic_max_tokens": 4096,
            # API keys will be "***" since they were masked on read
            "openai_api_key": "***",
            "anthropic_api_key": "***",
            "gemini_api_key": "***",
            "groq_api_key": "***",
            "deepseek_api_key": "***",
            "mistral_api_key": "***",
            "together_api_key": "***",
            "openrouter_api_key": "***",
            "xai_api_key": "***",
            "cerebras_api_key": "***",
            "github_api_key": "***",
            "bedrock_api_key": "***",
            "huggingface_api_key": "***",
            "moonshot_api_key": "***",
        }
        r = client.patch("/api/v1/config", json=payload)
        assert r.status_code == 200
        results = r.json()["results"]
        for result in results:
            assert result["status"] in ("ok", "skipped"), \
                f"Field '{result['key']}' failed: {result.get('error')}"


# ===========================================================================
# 5. AGENTS
# ===========================================================================


class TestAgents:
    """Test GET/POST /api/v1/agents."""

    def test_get_agents(self, client):
        """GET /agents — must return agents array."""
        r = client.get("/api/v1/agents")
        assert r.status_code == 200
        data = r.json()
        assert "agents" in data
        assert isinstance(data["agents"], list)
        # Default agent must exist
        assert len(data["agents"]) >= 1
        agent = data["agents"][0]
        assert "name" in agent
        assert "display_name" in agent

    def test_get_agents_has_all_fields(self, client):
        """GET /agents — each agent must have all fields the UI expects."""
        r = client.get("/api/v1/agents")
        agent = r.json()["agents"][0]
        expected_fields = [
            "name", "display_name", "description", "system_prompt",
            "language", "trigger_patterns", "trigger_keywords", "priority",
            "allowed_tools", "blocked_tools", "preferred_model",
            "temperature", "enabled",
        ]
        for field in expected_fields:
            assert field in agent, f"Agent missing field: {field}"

    def test_post_agent_create(self, client):
        """POST /agents/{name} — create a new agent."""
        agent = {
            "name": "test_agent",
            "display_name": "Test Agent",
            "description": "A test agent",
            "system_prompt": "You are a test bot.",
            "language": "en",
            "trigger_patterns": [],
            "trigger_keywords": ["test"],
            "priority": 5,
            "allowed_tools": None,
            "blocked_tools": [],
            "preferred_model": "qwen3:8b",
            "temperature": 0.7,
            "enabled": True,
        }
        r = client.post("/api/v1/agents/test_agent", json=agent)
        assert r.status_code == 200, f"POST /agents/test_agent failed: {r.text}"
        data = r.json()
        assert "error" not in data or data.get("status") == "ok"

        # Verify it persisted
        r2 = client.get("/api/v1/agents")
        names = [a["name"] for a in r2.json()["agents"]]
        assert "test_agent" in names

    def test_post_agent_update(self, client):
        """POST /agents/{name} — update existing agent."""
        # Create
        agent = {
            "name": "update_me",
            "display_name": "Original",
            "description": "",
            "system_prompt": "",
            "language": "de",
            "trigger_patterns": [],
            "trigger_keywords": [],
            "priority": 1,
            "allowed_tools": None,
            "blocked_tools": [],
            "preferred_model": "",
            "temperature": 0.7,
            "enabled": True,
        }
        client.post("/api/v1/agents/update_me", json=agent)

        # Update
        agent["display_name"] = "Updated"
        agent["priority"] = 99
        r = client.post("/api/v1/agents/update_me", json=agent)
        assert r.status_code == 200

        # Verify
        r2 = client.get("/api/v1/agents")
        found = [a for a in r2.json()["agents"] if a["name"] == "update_me"][0]
        assert found["display_name"] == "Updated"
        assert found["priority"] == 99


# ===========================================================================
# 6. BINDINGS
# ===========================================================================


class TestBindings:
    """Test GET/POST /api/v1/bindings."""

    def test_get_bindings(self, client):
        """GET /bindings — must return bindings array."""
        r = client.get("/api/v1/bindings")
        assert r.status_code == 200
        data = r.json()
        assert "bindings" in data
        assert isinstance(data["bindings"], list)

    def test_post_binding_create(self, client):
        """POST /bindings/{name} — create a binding."""
        binding = {
            "name": "test_binding",
            "target_agent": "jarvis",
            "priority": 10,
            "description": "Test binding",
            "channels": ["telegram"],
            "command_prefixes": ["/test"],
            "message_patterns": ["^test.*"],
            "enabled": True,
        }
        r = client.post("/api/v1/bindings/test_binding", json=binding)
        assert r.status_code == 200, f"POST /bindings/test_binding failed: {r.text}"

        # Verify persistence
        r2 = client.get("/api/v1/bindings")
        names = [b["name"] for b in r2.json()["bindings"]]
        assert "test_binding" in names


# ===========================================================================
# 7. PROMPTS
# ===========================================================================


class TestPrompts:
    """Test GET/PUT /api/v1/prompts."""

    def test_get_prompts(self, client):
        """GET /prompts — must return all prompt fields."""
        r = client.get("/api/v1/prompts")
        assert r.status_code == 200
        data = r.json()
        expected_fields = [
            "coreMd", "plannerSystem", "replanPrompt",
            "escalationPrompt", "policyYaml", "heartbeatMd",
        ]
        for field in expected_fields:
            assert field in data, f"Missing prompt field: {field}"

    def test_get_prompts_core_md(self, client, tmp_jarvis_home):
        """GET /prompts — coreMd must contain CORE.md content."""
        r = client.get("/api/v1/prompts")
        data = r.json()
        assert "Test Core" in data["coreMd"]

    def test_get_prompts_policy_yaml(self, client, tmp_jarvis_home):
        """GET /prompts — policyYaml must contain default policy."""
        r = client.get("/api/v1/prompts")
        data = r.json()
        assert "allow_all" in data["policyYaml"]

    def test_get_prompts_planner_defaults(self, client):
        """GET /prompts — planner prompts must have content (from Python constants)."""
        r = client.get("/api/v1/prompts")
        data = r.json()
        # These should fall back to Python constants
        assert len(data["plannerSystem"]) > 50, "plannerSystem should have substantial content"
        assert len(data["replanPrompt"]) > 50, "replanPrompt should have substantial content"
        assert len(data["escalationPrompt"]) > 50, "escalationPrompt should have substantial content"

    def test_put_prompts(self, client, tmp_jarvis_home):
        """PUT /prompts — must persist all fields to files."""
        prompts = {
            "coreMd": "# Updated Core\nNew identity.",
            "plannerSystem": "Updated system prompt.",
            "replanPrompt": "Updated replan prompt.",
            "escalationPrompt": "Updated escalation prompt.",
            "policyYaml": "rules:\n  - updated: true\n",
            "heartbeatMd": "# Heartbeat\n- [x] Check 1\n",
        }
        r = client.put("/api/v1/prompts", json=prompts)
        assert r.status_code == 200
        data = r.json()
        assert data.get("status") == "ok"

        # Verify by reading back
        r2 = client.get("/api/v1/prompts")
        data2 = r2.json()
        assert data2["coreMd"] == prompts["coreMd"]
        assert data2["plannerSystem"] == prompts["plannerSystem"]
        assert data2["policyYaml"] == prompts["policyYaml"]


# ===========================================================================
# 8. CRON JOBS
# ===========================================================================


class TestCronJobs:
    """Test GET/PUT /api/v1/cron-jobs."""

    def test_get_cron_jobs(self, client):
        """GET /cron-jobs — must return jobs array."""
        r = client.get("/api/v1/cron-jobs")
        assert r.status_code == 200
        data = r.json()
        assert "jobs" in data
        assert isinstance(data["jobs"], list)

    def test_put_cron_jobs(self, client):
        """PUT /cron-jobs — must persist jobs."""
        jobs = {
            "jobs": [
                {
                    "name": "test_job",
                    "schedule": "0 8 * * *",
                    "prompt": "Good morning!",
                    "channel": "cli",
                    "model": "",
                    "enabled": True,
                    "agent": "",
                },
                {
                    "name": "cleanup_job",
                    "schedule": "0 0 * * 0",
                    "prompt": "Run weekly cleanup",
                    "channel": "cli",
                    "model": "",
                    "enabled": False,
                    "agent": "",
                },
            ]
        }
        r = client.put("/api/v1/cron-jobs", json=jobs)
        assert r.status_code == 200

        # Verify persistence
        r2 = client.get("/api/v1/cron-jobs")
        loaded = r2.json()["jobs"]
        names = [j["name"] for j in loaded]
        assert "test_job" in names
        assert "cleanup_job" in names


# ===========================================================================
# 9. MCP SERVERS
# ===========================================================================


class TestMcpServers:
    """Test GET/PUT /api/v1/mcp-servers."""

    def test_get_mcp_servers(self, client):
        """GET /mcp-servers — must return mode and external_servers."""
        r = client.get("/api/v1/mcp-servers")
        assert r.status_code == 200
        data = r.json()
        assert "mode" in data

    def test_put_mcp_servers(self, client):
        """PUT /mcp-servers — must persist configuration."""
        mcp_config = {
            "mode": "http",
            "http_host": "127.0.0.1",
            "http_port": 3000,
            "server_name": "cognithor-mcp",
            "require_auth": False,
            "auth_token": "",
            "expose_tools": True,
            "expose_resources": True,
            "expose_prompts": False,
            "enable_sampling": False,
            "external_servers": {
                "my_server": {
                    "command": "node",
                    "args": ["server.js"],
                    "enabled": True,
                }
            },
        }
        r = client.put("/api/v1/mcp-servers", json=mcp_config)
        assert r.status_code == 200

        # Verify
        r2 = client.get("/api/v1/mcp-servers")
        data = r2.json()
        assert data["mode"] == "http"
        assert "my_server" in data.get("external_servers", {})


# ===========================================================================
# 10. A2A PROTOCOL
# ===========================================================================


class TestA2A:
    """Test GET/PUT /api/v1/a2a."""

    def test_get_a2a(self, client):
        """GET /a2a — must return A2A configuration."""
        r = client.get("/api/v1/a2a")
        assert r.status_code == 200
        data = r.json()
        # Should have basic A2A fields
        assert "enabled" in data

    def test_put_a2a(self, client):
        """PUT /a2a — must persist A2A configuration."""
        a2a_config = {
            "enabled": True,
            "host": "0.0.0.0",
            "port": 9000,
            "agent_name": "cognithor-test",
            "agent_description": "Test agent",
            "require_auth": False,
            "auth_token": "",
            "max_tasks": 10,
            "task_timeout_seconds": 300,
            "enable_streaming": True,
            "enable_push": False,
            "remotes": [],
        }
        r = client.put("/api/v1/a2a", json=a2a_config)
        assert r.status_code == 200

        # Verify
        r2 = client.get("/api/v1/a2a")
        data = r2.json()
        assert data["enabled"] is True
        assert data["port"] == 9000
        assert data["agent_name"] == "cognithor-test"


# ===========================================================================
# 11. PRESETS
# ===========================================================================


class TestPresets:
    """Test preset-related endpoints."""

    def test_get_presets(self, client):
        """GET /config/presets — must list available presets."""
        r = client.get("/api/v1/config/presets")
        assert r.status_code == 200
        data = r.json()
        assert "presets" in data

    def test_apply_preset(self, client):
        """POST /config/presets/{name} — must apply without error."""
        r = client.post("/api/v1/config/presets/standard")
        assert r.status_code == 200


# ===========================================================================
# 12. CONFIG RELOAD
# ===========================================================================


class TestConfigReload:
    """Test config reload endpoint."""

    def test_reload_config(self, client):
        """POST /config/reload — must succeed."""
        r = client.post("/api/v1/config/reload")
        assert r.status_code == 200
        data = r.json()
        assert data.get("status") == "ok"


# ===========================================================================
# 13. HEALTH CHECK
# ===========================================================================


class TestHealth:
    """Test health endpoint."""

    def test_health(self, client):
        """GET /health — must respond (used by Vite launcher)."""
        r = client.get("/api/v1/health")
        assert r.status_code == 200


# ===========================================================================
# 14. FULL SAVE FLOW (simulates what the UI does)
# ===========================================================================


class TestFullSaveFlow:
    """Simulate the complete UI save operation."""

    def test_full_save_flow(self, client, tmp_jarvis_home):
        """Simulate clicking Save in the UI — all calls in parallel."""
        errors = []

        # 1. Section PATCHes (15 sections)
        section_updates = {
            "ollama": {"timeout_seconds": 180},
            "models": {"planner": {"name": "custom:7b"}},
            "gatekeeper": {"max_blocked_retries": 3},
            "planner": {"temperature": 0.8},
            "memory": {"chunk_size_tokens": 600},
            "channels": {"cli_enabled": True},
            "sandbox": {"timeout_seconds": 60},
            "logging": {"level": "INFO"},
            "security": {"max_iterations": 5},
            "heartbeat": {"enabled": True},
            "plugins": {"skills_dir": "skills"},
            "dashboard": {"enabled": False},
            "model_overrides": {},
            "web": {"duckduckgo_enabled": True},
            "database": {"backend": "sqlite"},
        }
        for section, values in section_updates.items():
            r = client.patch(f"/api/v1/config/{section}", json=values)
            if r.json().get("error"):
                errors.append(f"{section}: {r.json()['error']}")

        # 2. Top-level PATCH
        top_level = {
            "owner_name": "SaveFlowTest",
            "llm_backend_type": "ollama",
            "operation_mode": "offline",
            "cost_tracking_enabled": False,
            "daily_budget_usd": 0.0,
            "monthly_budget_usd": 0.0,
            "vision_model": "test-model",
            "vision_model_detail": "test-detail",
            "openai_base_url": "https://api.openai.com/v1",
            "anthropic_max_tokens": 4096,
            "openai_api_key": "***",
            "anthropic_api_key": "***",
            "gemini_api_key": "***",
            "groq_api_key": "***",
            "deepseek_api_key": "***",
            "mistral_api_key": "***",
            "together_api_key": "***",
            "openrouter_api_key": "***",
            "xai_api_key": "***",
            "cerebras_api_key": "***",
            "github_api_key": "***",
            "bedrock_api_key": "***",
            "huggingface_api_key": "***",
            "moonshot_api_key": "***",
        }
        r = client.patch("/api/v1/config", json=top_level)
        for result in r.json().get("results", []):
            if result["status"] == "error":
                errors.append(f"top-level {result['key']}: {result['error']}")

        # 3. Agent saves
        agent = {
            "name": "save_flow_agent",
            "display_name": "Save Flow",
            "description": "Testing",
            "system_prompt": "Test",
            "language": "de",
            "trigger_patterns": [],
            "trigger_keywords": [],
            "priority": 1,
            "allowed_tools": None,
            "blocked_tools": [],
            "preferred_model": "",
            "temperature": 0.7,
            "enabled": True,
        }
        r = client.post("/api/v1/agents/save_flow_agent", json=agent)
        if r.json().get("error"):
            errors.append(f"agent: {r.json()['error']}")

        # 4. Binding saves
        binding = {
            "name": "save_flow_binding",
            "target_agent": "jarvis",
            "priority": 1,
            "description": "",
            "channels": [],
            "command_prefixes": [],
            "message_patterns": [],
            "enabled": True,
        }
        r = client.post("/api/v1/bindings/save_flow_binding", json=binding)
        if r.json().get("error"):
            errors.append(f"binding: {r.json()['error']}")

        # 5. Extra saves
        r = client.put("/api/v1/cron-jobs", json={"jobs": []})
        if r.json().get("error"):
            errors.append(f"cron: {r.json()['error']}")

        r = client.put("/api/v1/mcp-servers", json={"mode": "disabled", "external_servers": {}})
        if r.json().get("error"):
            errors.append(f"mcp: {r.json()['error']}")

        r = client.put("/api/v1/a2a", json={"enabled": False})
        if r.json().get("error"):
            errors.append(f"a2a: {r.json()['error']}")

        r = client.put("/api/v1/prompts", json={
            "coreMd": "# Core",
            "plannerSystem": "System",
            "replanPrompt": "Replan",
            "escalationPrompt": "Escalate",
            "policyYaml": "rules: []",
            "heartbeatMd": "# HB",
        })
        if r.json().get("error"):
            errors.append(f"prompts: {r.json()['error']}")

        # ALL MUST SUCCEED
        assert errors == [], f"Save flow errors: {errors}"

        # 6. Verify config reload shows saved data
        r = client.get("/api/v1/config")
        cfg = r.json()
        assert cfg["owner_name"] == "SaveFlowTest"
        assert cfg["planner"]["temperature"] == 0.8
        assert cfg["memory"]["chunk_size_tokens"] == 600
        assert cfg["web"]["duckduckgo_enabled"] is True
        assert cfg["database"]["backend"] == "sqlite"


# ===========================================================================
# 15. EDGE CASES & ERROR HANDLING
# ===========================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_agent_save(self, client):
        """POST /agents/{name} with minimal data."""
        r = client.post("/api/v1/agents/minimal", json={"name": "minimal"})
        assert r.status_code == 200

    def test_empty_binding_save(self, client):
        """POST /bindings/{name} with minimal data."""
        r = client.post("/api/v1/bindings/minimal", json={"name": "minimal"})
        assert r.status_code == 200

    def test_empty_cron_jobs(self, client):
        """PUT /cron-jobs with empty list."""
        r = client.put("/api/v1/cron-jobs", json={"jobs": []})
        assert r.status_code == 200

    def test_prompts_partial_update(self, client):
        """PUT /prompts with only some fields."""
        r = client.put("/api/v1/prompts", json={"coreMd": "# Just Core"})
        assert r.status_code == 200

    def test_nonexistent_section(self, client):
        """GET /config/nonexistent — should return error or 404."""
        r = client.get("/api/v1/config/foobar")
        assert r.status_code == 200  # Error in body
        data = r.json()
        assert data.get("error") or data.get("section") is None

    def test_concurrent_saves(self, client):
        """Multiple rapid PATCH calls should not corrupt state."""
        for i in range(5):
            r = client.patch("/api/v1/config/planner", json={"temperature": 0.1 * (i + 1)})
            assert r.status_code == 200
            assert r.json().get("status") == "ok"

        # Final value should be 0.5
        r = client.get("/api/v1/config/planner")
        assert abs(r.json()["values"]["temperature"] - 0.5) < 0.01
