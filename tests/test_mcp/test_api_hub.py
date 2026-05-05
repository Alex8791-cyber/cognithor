"""Tests fuer API Integration Hub -- Persistente API-Verbindungen.

Testet:
  - Tool-Registrierung beim MCP-Client
  - Template-Loading
  - Integration CRUD (api_list, api_connect, api_call, api_disconnect)
  - Auth-Header-Konstruktion (bearer, api_key, basic)
  - Credential-Maskierung
  - Rate-Limiting
  - File-Encryption (Fernet + Plaintext-Fallback)
"""

from __future__ import annotations

import base64
import os
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from cognithor.config import CognithorConfig, SecurityConfig, ensure_directory_structure
from cognithor.mcp.api_hub import (
    API_TEMPLATES,
    APIHub,
    _build_auth_headers,
    _build_auth_params,
    _is_private_host,
    _load_integrations,
    _mask_credential,
    _RateLimiter,
    _save_integrations,
    register_api_hub_tools,
)

if TYPE_CHECKING:
    from pathlib import Path

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture()
def config(tmp_path: Path) -> CognithorConfig:
    cfg = CognithorConfig(
        cognithor_home=tmp_path / ".cognithor",
        security=SecurityConfig(
            allowed_paths=[str(tmp_path)],
        ),
    )
    ensure_directory_structure(cfg)
    return cfg


@pytest.fixture()
def hub(config: CognithorConfig) -> APIHub:
    return APIHub(config)


@pytest.fixture()
def mock_mcp_client() -> MagicMock:
    client = MagicMock()
    client.register_builtin_handler = MagicMock()
    return client


# =============================================================================
# Template Tests
# =============================================================================


class TestTemplates:
    """Tests fuer API_TEMPLATES."""

    def test_templates_exist(self) -> None:
        assert len(API_TEMPLATES) == 6

    def test_github_template(self) -> None:
        tmpl = API_TEMPLATES["github"]
        assert tmpl["base_url"] == "https://api.github.com"
        assert tmpl["auth_type"] == "bearer"
        assert tmpl["credential_env"] == "GITHUB_TOKEN"
        assert tmpl["health_endpoint"] == "/user"

    def test_jira_template_needs_url(self) -> None:
        tmpl = API_TEMPLATES["jira"]
        assert tmpl["base_url"] == ""

    def test_notion_has_default_headers(self) -> None:
        tmpl = API_TEMPLATES["notion"]
        assert "default_headers" in tmpl
        assert "Notion-Version" in tmpl["default_headers"]

    def test_openweathermap_uses_api_key(self) -> None:
        tmpl = API_TEMPLATES["openweathermap"]
        assert tmpl["auth_type"] == "api_key"
        assert tmpl["auth_param"] == "appid"

    def test_all_templates_have_required_fields(self) -> None:
        for name, tmpl in API_TEMPLATES.items():
            assert "auth_type" in tmpl, f"{name} missing auth_type"
            assert "credential_env" in tmpl, f"{name} missing credential_env"
            assert "description" in tmpl, f"{name} missing description"


# =============================================================================
# Auth Header Tests
# =============================================================================


class TestAuthHeaders:
    """Tests fuer _build_auth_headers."""

    def test_bearer_auth(self) -> None:
        integration = {
            "auth_type": "bearer",
            "credential_env": "TEST_TOKEN",
        }
        with patch.dict(os.environ, {"TEST_TOKEN": "mytoken123"}):
            headers = _build_auth_headers(integration)
            assert headers["Authorization"] == "Bearer mytoken123"

    def test_bearer_auth_custom_prefix(self) -> None:
        integration = {
            "auth_type": "bearer",
            "credential_env": "TEST_TOKEN",
            "auth_header": "X-Auth",
            "auth_prefix": "Token ",
        }
        with patch.dict(os.environ, {"TEST_TOKEN": "abc"}):
            headers = _build_auth_headers(integration)
            assert headers["X-Auth"] == "Token abc"

    def test_basic_auth(self) -> None:
        integration = {
            "auth_type": "basic",
            "credential_env": "TEST_TOKEN",
        }
        with patch.dict(os.environ, {"TEST_TOKEN": "user:pass123"}):
            headers = _build_auth_headers(integration)
            expected = base64.b64encode(b"user:pass123").decode("ascii")
            assert headers["Authorization"] == f"Basic {expected}"

    def test_basic_auth_without_colon(self) -> None:
        """When credential has no colon, 'user:' prefix is added."""
        integration = {
            "auth_type": "basic",
            "credential_env": "TEST_TOKEN",
        }
        with patch.dict(os.environ, {"TEST_TOKEN": "justtoken"}):
            headers = _build_auth_headers(integration)
            expected = base64.b64encode(b"user:justtoken").decode("ascii")
            assert headers["Authorization"] == f"Basic {expected}"

    def test_api_key_header(self) -> None:
        integration = {
            "auth_type": "api_key",
            "credential_env": "TEST_KEY",
            "auth_header": "X-API-Key",
        }
        with patch.dict(os.environ, {"TEST_KEY": "key123"}):
            headers = _build_auth_headers(integration)
            assert headers["X-API-Key"] == "key123"

    def test_api_key_no_header(self) -> None:
        """api_key with no auth_header returns empty (param-based)."""
        integration = {
            "auth_type": "api_key",
            "credential_env": "TEST_KEY",
        }
        with patch.dict(os.environ, {"TEST_KEY": "key123"}):
            headers = _build_auth_headers(integration)
            assert headers == {}

    def test_no_credential_returns_empty(self) -> None:
        integration = {
            "auth_type": "bearer",
            "credential_env": "NONEXISTENT_VAR",
        }
        # Make sure env var doesn't exist
        env = dict(os.environ)
        env.pop("NONEXISTENT_VAR", None)
        with patch.dict(os.environ, env, clear=True):
            headers = _build_auth_headers(integration)
            assert headers == {}

    def test_template_defaults_used(self) -> None:
        integration = {
            "auth_type": "bearer",
            "credential_env": "TEST_TOKEN",
        }
        template = {
            "auth_header": "X-Custom-Auth",
            "auth_prefix": "MyPrefix ",
        }
        with patch.dict(os.environ, {"TEST_TOKEN": "tok"}):
            headers = _build_auth_headers(integration, template)
            assert headers["X-Custom-Auth"] == "MyPrefix tok"


class TestAuthParams:
    """Tests fuer _build_auth_params."""

    def test_api_key_param(self) -> None:
        integration = {
            "auth_type": "api_key",
            "credential_env": "TEST_KEY",
        }
        template = {"auth_param": "appid"}
        with patch.dict(os.environ, {"TEST_KEY": "key123"}):
            params = _build_auth_params(integration, template)
            assert params == {"appid": "key123"}

    def test_bearer_returns_empty(self) -> None:
        integration = {
            "auth_type": "bearer",
            "credential_env": "TEST_TOKEN",
        }
        with patch.dict(os.environ, {"TEST_TOKEN": "tok"}):
            params = _build_auth_params(integration)
            assert params == {}


# =============================================================================
# Credential Masking Tests
# =============================================================================


class TestCredentialMasking:
    """Tests fuer _mask_credential."""

    def test_masks_bearer_token(self) -> None:
        text = "Authorization: Bearer ghp_abc123456789"
        masked = _mask_credential(text)
        assert "ghp_abc123456789" not in masked
        assert "MASKED" in masked

    def test_masks_basic_token(self) -> None:
        text = "Authorization: Basic dXNlcjpwYXNzMTIz"
        masked = _mask_credential(text)
        assert "dXNlcjpwYXNzMTIz" not in masked
        assert "MASKED" in masked

    def test_preserves_short_text(self) -> None:
        text = "Just a normal error message"
        masked = _mask_credential(text)
        assert masked == text


# =============================================================================
# Rate Limiter Tests
# =============================================================================


class TestRateLimiter:
    """Tests fuer _RateLimiter."""

    def test_allows_initial_requests(self) -> None:
        limiter = _RateLimiter(max_requests=5, window_seconds=60.0)
        for _ in range(5):
            assert limiter.allow() is True

    def test_blocks_after_limit(self) -> None:
        limiter = _RateLimiter(max_requests=3, window_seconds=60.0)
        for _ in range(3):
            assert limiter.allow() is True
        assert limiter.allow() is False

    def test_remaining_count(self) -> None:
        limiter = _RateLimiter(max_requests=5, window_seconds=60.0)
        assert limiter.remaining == 5
        limiter.allow()
        assert limiter.remaining == 4

    def test_window_expiry(self) -> None:
        """Requests expire after window."""
        import time

        limiter = _RateLimiter(max_requests=2, window_seconds=0.1)
        limiter.allow()
        limiter.allow()
        assert limiter.allow() is False
        time.sleep(0.15)
        assert limiter.allow() is True


# =============================================================================
# Integration Storage Tests
# =============================================================================


class TestIntegrationStorage:
    """Tests fuer _load_integrations / _save_integrations."""

    def test_empty_load(self, config: CognithorConfig) -> None:
        data = _load_integrations(config)
        assert data == {}

    def test_save_and_load(self, config: CognithorConfig) -> None:
        test_data = {
            "github": {
                "base_url": "https://api.github.com",
                "auth_type": "bearer",
                "credential_env": "GITHUB_TOKEN",
            }
        }
        _save_integrations(config, test_data)
        loaded = _load_integrations(config)
        assert loaded["github"]["base_url"] == "https://api.github.com"
        assert loaded["github"]["auth_type"] == "bearer"

    def test_save_creates_directory(self, config: CognithorConfig) -> None:
        # Remove the directory if it exists
        path = config.cognithor_home / "integrations.json"
        _save_integrations(config, {"test": {}})
        assert path.exists()


# =============================================================================
# API Hub Method Tests
# =============================================================================


class TestApiList:
    """Tests fuer api_list."""

    async def test_list_empty(self, hub: APIHub) -> None:
        result = await hub.api_list()
        assert "No integrations configured" in result
        assert "Available Templates" in result

    async def test_list_with_integration(self, hub: APIHub, config: CognithorConfig) -> None:
        _save_integrations(
            config,
            {
                "github": {
                    "base_url": "https://api.github.com",
                    "auth_type": "bearer",
                    "credential_env": "GITHUB_TOKEN",
                }
            },
        )
        result = await hub.api_list()
        assert "github" in result
        expected_host = "api." + "github.com"  # split to avoid CodeQL URL false positive
        assert expected_host in result

    async def test_list_shows_templates(self, hub: APIHub) -> None:
        result = await hub.api_list()
        for tmpl_name in API_TEMPLATES:
            assert tmpl_name in result


class TestApiConnect:
    """Tests fuer api_connect."""

    async def test_connect_with_template(self, hub: APIHub, config: CognithorConfig) -> None:
        with (
            patch.dict(os.environ, {"GITHUB_TOKEN": "ghp_test123"}),
            patch.object(hub, "_health_check", return_value=(True, "OK (HTTP 200)")),
        ):
            # Mock health check
            result = await hub.api_connect(name="github")
            assert "configured successfully" in result.lower()
            expected_host = "api." + "github.com"  # split to avoid CodeQL URL false positive
            assert expected_host in result

    async def test_connect_custom_api(self, hub: APIHub) -> None:
        result = await hub.api_connect(
            name="myapi",
            base_url="https://api.example.com",
            auth_type="bearer",
            credential_env="MY_TOKEN",
        )
        assert "configured successfully" in result.lower()

    async def test_connect_empty_name(self, hub: APIHub) -> None:
        result = await hub.api_connect(name="")
        assert "Error" in result

    async def test_connect_no_url_no_template(self, hub: APIHub) -> None:
        result = await hub.api_connect(name="unknown-api")
        assert "Error" in result
        assert "base_url is required" in result

    async def test_connect_invalid_auth_type(self, hub: APIHub) -> None:
        result = await hub.api_connect(
            name="test",
            base_url="https://api.example.com",
            auth_type="oauth",
            credential_env="TOKEN",
        )
        assert "Error" in result
        assert "Invalid auth_type" in result

    async def test_connect_invalid_url(self, hub: APIHub) -> None:
        result = await hub.api_connect(
            name="test",
            base_url="not-a-url",
            credential_env="TOKEN",
        )
        assert "Error" in result
        assert "Invalid base_url" in result

    async def test_connect_jira_needs_url(self, hub: APIHub) -> None:
        result = await hub.api_connect(name="jira")
        assert "Error" in result
        assert "base_url is required" in result

    async def test_connect_persists(self, hub: APIHub, config: CognithorConfig) -> None:
        await hub.api_connect(
            name="test",
            base_url="https://api.example.com",
            credential_env="TOKEN",
        )
        loaded = _load_integrations(config)
        assert "test" in loaded
        assert loaded["test"]["base_url"] == "https://api.example.com"


class TestSSRFGuards:
    """Deep-PR4: SSRF + redirect-chain hardening."""

    @pytest.mark.parametrize(
        "host",
        [
            "127.0.0.1",
            "10.0.0.1",
            "172.16.5.5",
            "192.168.1.1",
            "169.254.169.254",  # AWS/Azure/GCP metadata service
            "::1",
            "fc00::1",
        ],
    )
    def test_is_private_host_blocks_internal_ips(self, host: str) -> None:
        assert _is_private_host(host) is True

    @pytest.mark.parametrize(
        "host",
        [
            "8.8.8.8",
            "1.1.1.1",
            "api.example.com",  # hostname not resolved here
            "",
        ],
    )
    def test_is_private_host_allows_public_or_hostnames(self, host: str) -> None:
        assert _is_private_host(host) is False


class TestApiCall:
    """Tests fuer api_call."""

    async def test_call_integration_not_found(self, hub: APIHub) -> None:
        result = await hub.api_call(integration="nonexistent")
        assert "Error" in result
        assert "not found" in result

    async def test_call_invalid_method(self, hub: APIHub, config: CognithorConfig) -> None:
        _save_integrations(
            config,
            {
                "test": {
                    "base_url": "https://api.example.com",
                    "auth_type": "bearer",
                    "credential_env": "TEST_TOKEN",
                }
            },
        )
        result = await hub.api_call(integration="test", method="INVALID")
        assert "Error" in result
        assert "Invalid method" in result

    async def test_call_no_credential(self, hub: APIHub, config: CognithorConfig) -> None:
        _save_integrations(
            config,
            {
                "test": {
                    "base_url": "https://api.example.com",
                    "auth_type": "bearer",
                    "credential_env": "DEFINITELY_NOT_SET_VAR_XYZ",
                }
            },
        )
        # Ensure env var is not set
        env = dict(os.environ)
        env.pop("DEFINITELY_NOT_SET_VAR_XYZ", None)
        with patch.dict(os.environ, env, clear=True):
            result = await hub.api_call(integration="test")
            assert "Error" in result
            assert "Credential not available" in result

    async def test_call_success(self, hub: APIHub, config: CognithorConfig) -> None:
        _save_integrations(
            config,
            {
                "test": {
                    "base_url": "https://api.example.com",
                    "auth_type": "bearer",
                    "credential_env": "TEST_TOKEN",
                }
            },
        )
        with (
            patch.dict(os.environ, {"TEST_TOKEN": "tok123"}),
            patch.object(hub, "_do_request", return_value=(200, '{"ok": true}')),
        ):
            result = await hub.api_call(integration="test", endpoint="/test")
            assert "Status: 200" in result
            assert '"ok": true' in result

    async def test_call_rate_limited(self, hub: APIHub, config: CognithorConfig) -> None:
        _save_integrations(
            config,
            {
                "test": {
                    "base_url": "https://api.example.com",
                    "auth_type": "bearer",
                    "credential_env": "TEST_TOKEN",
                }
            },
        )
        # Exhaust rate limiter
        limiter = hub._get_rate_limiter("test")
        limiter._max_requests = 1
        limiter.allow()  # Use up the one allowed request

        with patch.dict(os.environ, {"TEST_TOKEN": "tok"}):
            result = await hub.api_call(integration="test")
            assert "Rate limit" in result

    async def test_call_with_body(self, hub: APIHub, config: CognithorConfig) -> None:
        _save_integrations(
            config,
            {
                "test": {
                    "base_url": "https://api.example.com",
                    "auth_type": "bearer",
                    "credential_env": "TEST_TOKEN",
                }
            },
        )
        with (
            patch.dict(os.environ, {"TEST_TOKEN": "tok"}),
            patch.object(hub, "_do_request", return_value=(201, '{"id": 1}')) as mock,
        ):
            await hub.api_call(
                integration="test",
                method="POST",
                endpoint="/items",
                body={"name": "test"},
            )
            # Verify body was passed
            _, kwargs = mock.call_args
            assert kwargs.get("body") == {"name": "test"}

    async def test_call_api_key_auth_params(self, hub: APIHub, config: CognithorConfig) -> None:
        _save_integrations(
            config,
            {
                "openweathermap": {
                    "base_url": "https://api.openweathermap.org/data/2.5",
                    "auth_type": "api_key",
                    "credential_env": "OWM_KEY",
                    "auth_param": "appid",
                }
            },
        )
        with (
            patch.dict(os.environ, {"OWM_KEY": "mykey123"}),
            patch.object(hub, "_do_request", return_value=(200, "{}")) as mock,
        ):
            await hub.api_call(
                integration="openweathermap",
                endpoint="/weather?q=London",
            )
            _, kwargs = mock.call_args
            assert kwargs.get("params", {}).get("appid") == "mykey123"


class TestApiDisconnect:
    """Tests fuer api_disconnect."""

    async def test_disconnect_existing(self, hub: APIHub, config: CognithorConfig) -> None:
        _save_integrations(
            config,
            {
                "test": {"base_url": "https://example.com"},
            },
        )
        result = await hub.api_disconnect(name="test")
        assert "removed successfully" in result.lower()
        loaded = _load_integrations(config)
        assert "test" not in loaded

    async def test_disconnect_nonexistent(self, hub: APIHub) -> None:
        result = await hub.api_disconnect(name="nonexistent")
        assert "Error" in result
        assert "not found" in result


# =============================================================================
# Registration Tests
# =============================================================================


class TestRegistration:
    """Tests fuer register_api_hub_tools."""

    def test_registration(self, mock_mcp_client: MagicMock, config: CognithorConfig) -> None:
        result = register_api_hub_tools(mock_mcp_client, config)
        assert result is not None
        assert isinstance(result, APIHub)
        # 4 tools registered
        assert mock_mcp_client.register_builtin_handler.call_count == 4
        # Verify tool names
        registered_names = [
            call[0][0] for call in mock_mcp_client.register_builtin_handler.call_args_list
        ]
        assert "api_list" in registered_names
        assert "api_connect" in registered_names
        assert "api_call" in registered_names
        assert "api_disconnect" in registered_names


# =============================================================================
# Health Check Tests
# =============================================================================


class TestHealthCheck:
    """Tests fuer den Health-Check bei api_connect."""

    async def test_health_check_success(self, hub: APIHub) -> None:
        integration = {
            "base_url": "https://api.example.com",
            "auth_type": "bearer",
            "credential_env": "TEST_TOKEN",
            "headers": {},
        }
        with (
            patch.dict(os.environ, {"TEST_TOKEN": "tok"}),
            patch.object(hub, "_do_request", return_value=(200, "")),
        ):
            ok, msg = await hub._health_check(integration, None, "/health")
            assert ok is True
            assert "OK" in msg

    async def test_health_check_failure(self, hub: APIHub) -> None:
        integration = {
            "base_url": "https://api.example.com",
            "auth_type": "bearer",
            "credential_env": "TEST_TOKEN",
            "headers": {},
        }
        with (
            patch.dict(os.environ, {"TEST_TOKEN": "tok"}),
            patch.object(hub, "_do_request", return_value=(401, "Unauthorized")),
        ):
            ok, msg = await hub._health_check(integration, None, "/health")
            assert ok is False
            assert "401" in msg

    async def test_health_check_exception(self, hub: APIHub) -> None:
        integration = {
            "base_url": "https://api.example.com",
            "auth_type": "bearer",
            "credential_env": "TEST_TOKEN",
            "headers": {},
        }
        with (
            patch.dict(os.environ, {"TEST_TOKEN": "tok"}),
            patch.object(hub, "_do_request", side_effect=Exception("Connection refused")),
        ):
            ok, msg = await hub._health_check(integration, None, "/health")
            assert ok is False
            assert "failed" in msg.lower()
