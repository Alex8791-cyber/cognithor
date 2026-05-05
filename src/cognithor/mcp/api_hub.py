"""API Integration Hub for Cognithor -- Persistent API connections.

Enables the agent to configure and call external APIs:
  - api_list: List configured integrations
  - api_connect: Set up an API integration
  - api_call: Make authenticated API calls
  - api_disconnect: Remove an integration

Credentials are NEVER stored -- only the environment variable name.
Configuration resides in ~/.cognithor/integrations.json (optionally Fernet-encrypted).

Reference: §5.3 (MCP Tools)
"""

from __future__ import annotations

import base64
import contextlib
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from cognithor.utils.logging import get_logger

if TYPE_CHECKING:
    from cognithor.config import CognithorConfig

log = get_logger(__name__)

__all__ = [
    "API_TEMPLATES",
    "APIHub",
    "APIHubError",
    "register_api_hub_tools",
]

# ── Constants ──────────────────────────────────────────────────────────────

_MAX_RESPONSE_CHARS = 50_000
_DEFAULT_TIMEOUT = 30  # seconds per request
_DEFAULT_RATE_LIMIT = 60  # requests per minute
_ALLOWED_METHODS = frozenset({"GET", "POST", "PUT", "DELETE", "PATCH", "HEAD"})
# httpx default is 20 redirects. We tighten to 3 since legitimate APIs
# rarely chain more than one or two redirects.
_MAX_REDIRECTS = 3


def _is_private_host(host: str) -> bool:
    """Return True if *host* resolves to a loopback/private/link-local IP.

    SSRF-defense helper: a redirect chain that passes through a private
    address (incl. AWS/Azure/GCP metadata service 169.254.169.254 or
    Ollama/vLLM at 127.0.0.1) leaks internal data even when the operator-
    configured base_url is public. We refuse to surface response bodies
    from such chains.
    """
    import ipaddress

    if not host:
        return False
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        # Hostname, not an IP. We don't resolve it here — full DNS-rebind
        # protection requires socket-level interception. The httpx caller
        # passes user-configured base_urls (operator trust), so a
        # hostname like "metadata.internal" passes; but a literal IP
        # redirect target is the practical attack vector.
        return False
    return bool(
        addr.is_loopback
        or addr.is_private
        or addr.is_link_local
        or addr.is_multicast
        or addr.is_unspecified
        or addr.is_reserved
    )


# Fernet encryption (optional)
_fernet: Any = None
_fernet_key_path: Path | None = None

try:
    from cryptography.fernet import Fernet as _FernetClass

    _HAS_FERNET = True
except ImportError:
    _FernetClass = None  # type: ignore[assignment, misc]
    _HAS_FERNET = False


# ── Pre-built Templates ───────────────────────────────────────────────────

API_TEMPLATES: dict[str, dict[str, Any]] = {
    "github": {
        "base_url": "https://api.github.com",
        "auth_type": "bearer",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
        "credential_env": "GITHUB_TOKEN",
        "health_endpoint": "/user",
        "description": "GitHub API v3",
    },
    "jira": {
        "base_url": "",  # user must provide
        "auth_type": "basic",
        "credential_env": "JIRA_TOKEN",
        "health_endpoint": "/rest/api/2/myself",
        "description": "Jira REST API",
    },
    "notion": {
        "base_url": "https://api.notion.com/v1",
        "auth_type": "bearer",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
        "credential_env": "NOTION_TOKEN",
        "default_headers": {"Notion-Version": "2022-06-28"},
        "health_endpoint": "/users/me",
        "description": "Notion API",
    },
    "todoist": {
        "base_url": "https://api.todoist.com/rest/v2",
        "auth_type": "bearer",
        "credential_env": "TODOIST_TOKEN",
        "health_endpoint": "/projects",
        "description": "Todoist REST API",
    },
    "home_assistant": {
        "base_url": "",  # user must provide (e.g. http://homeassistant.local:8123/api)
        "auth_type": "bearer",
        "auth_prefix": "Bearer ",
        "credential_env": "HA_TOKEN",
        "health_endpoint": "/",
        "description": "Home Assistant API",
    },
    "openweathermap": {
        "base_url": "https://api.openweathermap.org/data/2.5",
        "auth_type": "api_key",
        "auth_param": "appid",  # query parameter
        "credential_env": "OPENWEATHER_API_KEY",
        "health_endpoint": "/weather?q=London",
        "description": "OpenWeatherMap API",
    },
}


class APIHubError(Exception):
    """Error during API hub operations."""


class _RateLimiter:
    """Simple Token-Bucket Rate-Limiter per Integration."""

    def __init__(
        self,
        max_requests: int = _DEFAULT_RATE_LIMIT,
        window_seconds: float = 60.0,
    ) -> None:
        self._max_requests = max_requests
        self._window = window_seconds
        self._timestamps: list[float] = []

    def allow(self) -> bool:
        """Prueft ob ein Request erlaubt ist und registriert ihn."""
        now = time.monotonic()
        cutoff = now - self._window
        self._timestamps = [ts for ts in self._timestamps if ts > cutoff]
        if len(self._timestamps) >= self._max_requests:
            return False
        self._timestamps.append(now)
        return True

    @property
    def remaining(self) -> int:
        """Verbleibende Requests im aktuellen Fenster."""
        now = time.monotonic()
        cutoff = now - self._window
        active = [ts for ts in self._timestamps if ts > cutoff]
        return max(0, self._max_requests - len(active))


def _get_integrations_path(config: Any) -> Path:
    """Gibt den Pfad zur integrations.json zurueck."""
    cognithor_home = getattr(config, "cognithor_home", None)
    if cognithor_home:
        return Path(cognithor_home) / "integrations.json"
    return Path.home() / ".cognithor" / "integrations.json"


def _get_fernet_key_path(config: Any) -> Path:
    """Gibt den Pfad zum Fernet-Key zurueck."""
    cognithor_home = getattr(config, "cognithor_home", None)
    if cognithor_home:
        return Path(cognithor_home) / ".integrations.key"
    return Path.home() / ".cognithor" / ".integrations.key"


def _get_or_create_fernet(config: Any) -> Any | None:
    """Erstellt oder laedt einen Fernet-Key fuer Verschluesselung.

    Returns:
        Fernet-Instanz oder None wenn cryptography nicht verfuegbar.
    """
    if not _HAS_FERNET or _FernetClass is None:
        return None

    key_path = _get_fernet_key_path(config)
    try:
        if key_path.exists():
            key = key_path.read_bytes().strip()
        else:
            key = _FernetClass.generate_key()
            key_path.parent.mkdir(parents=True, exist_ok=True)
            key_path.write_bytes(key)
            # Restrict file access to owner only
            with contextlib.suppress(OSError):
                key_path.chmod(0o600)
            if sys.platform == "win32":
                try:
                    username = os.environ.get("USERNAME", "")
                    if username:
                        subprocess.run(
                            [
                                "icacls",
                                str(key_path),
                                "/inheritance:r",
                                "/grant:r",
                                f"{username}:(R,W)",
                            ],
                            capture_output=True,
                            timeout=10,
                        )
                except Exception:
                    pass  # Best-effort ACL restriction
        return _FernetClass(key)
    except Exception as exc:
        log.warning("fernet_key_error", error=str(exc))
        return None


def _load_integrations(config: Any) -> dict[str, Any]:
    """Laedt Integrationen aus der JSON-Datei (ggf. entschluesselt)."""
    path = _get_integrations_path(config)
    if not path.exists():
        return {}

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        log.error("integrations_read_failed", error=str(exc))
        return {}

    # Versuche Fernet-Entschluesselung
    fernet = _get_or_create_fernet(config)
    if fernet is not None:
        try:
            decrypted = fernet.decrypt(raw.encode("utf-8"))
            return cast("dict[str, Any]", json.loads(decrypted))

        except Exception:
            pass  # Fallthrough zu Plaintext

    # Plaintext-Fallback
    try:
        return cast("dict[str, Any]", json.loads(raw))

    except json.JSONDecodeError as exc:
        log.error("integrations_json_invalid", error=str(exc))
        return {}


def _save_integrations(config: Any, data: dict[str, Any]) -> None:
    """Speichert Integrationen (ggf. verschluesselt)."""
    path = _get_integrations_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)

    json_str = json.dumps(data, indent=2, ensure_ascii=False)

    fernet = _get_or_create_fernet(config)
    if fernet is not None:
        try:
            encrypted = fernet.encrypt(json_str.encode("utf-8"))
            path.write_bytes(encrypted)
            log.debug("integrations_saved_encrypted")
            return
        except Exception as exc:
            log.warning("fernet_encrypt_failed_plaintext_fallback", error=str(exc))

    # Plaintext-Fallback mit Warnung
    log.warning("integrations_saved_plaintext_no_encryption")
    path.write_text(json_str, encoding="utf-8")


def _resolve_credential(integration: dict[str, Any]) -> str | None:
    """Liest das Credential aus der angegebenen Umgebungsvariable.

    Returns:
        Credential-String oder None.
    """
    env_var = integration.get("credential_env", "")
    if not env_var:
        return None
    return os.environ.get(env_var)


def _build_auth_headers(
    integration: dict[str, Any],
    template: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Baut Authentifizierungs-Header basierend auf auth_type.

    Returns:
        Dict mit Auth-Headern.
    """
    credential = _resolve_credential(integration)
    if not credential:
        return {}

    auth_type = integration.get("auth_type", "bearer")
    headers: dict[str, str] = {}

    if auth_type == "bearer":
        auth_header = integration.get("auth_header") or (template or {}).get(
            "auth_header", "Authorization"
        )
        auth_prefix = integration.get("auth_prefix") or (template or {}).get(
            "auth_prefix", "Bearer "
        )
        headers[auth_header] = f"{auth_prefix}{credential}"

    elif auth_type == "basic":
        # Credential format: "user:password" or just token
        if ":" not in credential:
            # Assume it's email:token (common for Jira)
            credential = f"user:{credential}"
        encoded = base64.b64encode(credential.encode("utf-8")).decode("ascii")
        headers["Authorization"] = f"Basic {encoded}"

    elif auth_type == "api_key":
        auth_header = integration.get("auth_header") or (template or {}).get("auth_header", "")
        if auth_header:
            headers[auth_header] = credential
        # If auth_param is set, it's a query parameter -- handled separately

    return headers


def _build_auth_params(
    integration: dict[str, Any],
    template: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Baut Query-Parameter fuer API-Key-Auth.

    Returns:
        Dict mit Auth-Query-Parametern.
    """
    auth_type = integration.get("auth_type", "bearer")
    if auth_type != "api_key":
        return {}

    credential = _resolve_credential(integration)
    if not credential:
        return {}

    auth_param = integration.get("auth_param") or (template or {}).get("auth_param", "")
    if auth_param:
        return {auth_param: credential}
    return {}


def _truncate(text: str, max_chars: int = _MAX_RESPONSE_CHARS) -> str:
    """Kuerzt Text auf max_chars mit Hinweis."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n\n[... response truncated at {max_chars} chars]"


def _mask_credential(text: str) -> str:
    """Maskiert moegliche Credentials in Fehlermeldungen."""
    # Mask anything that looks like a token/key (long alphanumeric strings)
    import re

    return re.sub(
        r'(?:Bearer |Basic |token[=:]\s*|key[=:]\s*)["\']?([a-zA-Z0-9_-]{8,})',
        lambda m: m.group(0)[:10] + "***MASKED***",
        text,
    )


class APIHub:
    """API Integration Hub -- Persistente Verbindungen zu externen APIs. [B§5.3]

    Verwaltet API-Konfigurationen und fuehrt authentifizierte Requests
    durch. Credentials werden NIE gespeichert, nur Env-Var-Namen.
    """

    def __init__(self, config: CognithorConfig) -> None:
        self._config = config
        self._rate_limiters: dict[str, _RateLimiter] = {}
        log.info("api_hub_init")

    def _get_rate_limiter(self, name: str) -> _RateLimiter:
        """Holt oder erstellt einen Rate-Limiter fuer eine Integration."""
        if name not in self._rate_limiters:
            self._rate_limiters[name] = _RateLimiter()
        return self._rate_limiters[name]

    async def api_list(self) -> str:
        """Listet konfigurierte API-Integrationen auf.

        Returns:
            Formatierte Liste von Integrationen und verfuegbaren Templates.
        """
        integrations = _load_integrations(self._config)

        lines: list[str] = []

        # Konfigurierte Integrationen
        if integrations:
            lines.append("=== Configured Integrations ===\n")
            for name, cfg in sorted(integrations.items()):
                credential = _resolve_credential(cfg)
                status = "connected" if credential else "disconnected (env var not set)"
                base_url = cfg.get("base_url", "")
                auth_type = cfg.get("auth_type", "unknown")
                env_var = cfg.get("credential_env", "")
                lines.append(f"  {name}:")
                lines.append(f"    URL:       {base_url}")
                lines.append(f"    Auth:      {auth_type}")
                lines.append(f"    Env var:   {env_var}")
                lines.append(f"    Status:    {status}")
                lines.append("")
        else:
            lines.append("No integrations configured yet.\n")

        # Verfuegbare Templates
        unconfigured = [t for t in API_TEMPLATES if t not in integrations]
        if unconfigured:
            lines.append("=== Available Templates ===\n")
            for tmpl_name in sorted(unconfigured):
                tmpl = API_TEMPLATES[tmpl_name]
                desc = tmpl.get("description", "")
                env_var = tmpl.get("credential_env", "")
                base_url = tmpl.get("base_url", "(user must provide)")
                lines.append(f"  {tmpl_name}: {desc}")
                lines.append(f"    Default URL: {base_url or '(user must provide)'}")
                lines.append(f"    Env var:     {env_var}")
                lines.append("")

        return "\n".join(lines)

    async def api_connect(
        self,
        *,
        name: str,
        base_url: str | None = None,
        auth_type: str | None = None,
        credential_env: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> str:
        """Konfiguriert eine API-Integration.

        Args:
            name: Integrations-Name (z.B. "github").
            base_url: API-Basis-URL (optional, nutzt Template-Default).
            auth_type: Authentifizierungs-Typ (bearer/api_key/basic).
            credential_env: Name der Umgebungsvariable mit dem Token/Key.
            headers: Zusaetzliche HTTP-Header.

        Returns:
            Erfolgs-/Fehlermeldung mit Integrations-Details.
        """
        name = name.strip().lower()
        if not name:
            return "Error: Integration name is required."

        # Template laden falls vorhanden
        template = API_TEMPLATES.get(name, {})

        # Werte mit Template-Defaults zusammenfuehren
        resolved_url = base_url or template.get("base_url", "")
        if not resolved_url:
            return f"Error: base_url is required for '{name}'. No default template URL available."

        resolved_auth = auth_type or template.get("auth_type", "bearer")
        if resolved_auth not in ("bearer", "api_key", "basic"):
            return f"Error: Invalid auth_type '{resolved_auth}'. Allowed: bearer, api_key, basic."

        resolved_env = credential_env or template.get("credential_env", "")
        if not resolved_env:
            return (
                "Error: credential_env is required. Specify the environment "
                "variable name containing the API key/token."
            )

        # URL validieren
        resolved_url = resolved_url.rstrip("/")
        if not resolved_url.startswith(("http://", "https://")):
            return f"Error: Invalid base_url '{resolved_url}'. Must start with http:// or https://."

        # Merge headers
        merged_headers: dict[str, str] = {}
        if template.get("default_headers"):
            merged_headers.update(template["default_headers"])
        if headers:
            merged_headers.update(headers)

        # Integration-Konfiguration erstellen
        integration: dict[str, Any] = {
            "base_url": resolved_url,
            "auth_type": resolved_auth,
            "credential_env": resolved_env,
            "headers": merged_headers,
            "created_at": datetime.now(UTC).isoformat(),
        }

        # Template-spezifische Felder uebernehmen
        for field in ("auth_header", "auth_prefix", "auth_param"):
            if field in template:
                integration[field] = template[field]

        # Health-Check (optional)
        health_ok = False
        health_msg = ""
        health_endpoint = template.get("health_endpoint", "")
        credential = os.environ.get(resolved_env)

        if health_endpoint and credential:
            try:
                health_ok, health_msg = await self._health_check(
                    integration,
                    template,
                    health_endpoint,
                )
            except Exception as exc:
                health_msg = f"Health check failed: {_mask_credential(str(exc))}"
        elif not credential:
            health_msg = (
                f"Warning: Environment variable '{resolved_env}' is not set. "
                "Set it before making API calls."
            )

        # Speichern
        integrations = _load_integrations(self._config)
        integrations[name] = integration
        _save_integrations(self._config, integrations)

        # Ergebnis
        lines = [
            f"Integration '{name}' configured successfully.",
            f"  URL:      {resolved_url}",
            f"  Auth:     {resolved_auth}",
            f"  Env var:  {resolved_env}",
        ]
        if merged_headers:
            lines.append(f"  Headers:  {', '.join(merged_headers.keys())}")
        if health_msg:
            lines.append(f"  Health:   {health_msg}")
        elif health_ok:
            lines.append("  Health:   OK")

        if not _HAS_FERNET:
            lines.append(
                "\n  Warning: cryptography package not installed. Config stored as plaintext."
            )
            lines.append("  Install: pip install cryptography")

        return "\n".join(lines)

    async def api_call(
        self,
        *,
        integration: str,
        method: str = "GET",
        endpoint: str = "",
        body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> str:
        """Fuehrt einen authentifizierten API-Call aus.

        Args:
            integration: Name der Integration.
            method: HTTP-Methode (GET/POST/PUT/DELETE/PATCH/HEAD).
            endpoint: Pfad, wird an base_url angehaengt.
            body: Request-Body (als JSON).
            headers: Zusaetzliche Header.

        Returns:
            Status-Code + Response-Body.
        """
        name = integration.strip().lower()
        integrations = _load_integrations(self._config)
        cfg = integrations.get(name)
        if cfg is None:
            return (
                f"Error: Integration '{name}' not found. "
                "Use api_list to see available integrations."
            )

        # Method validieren
        method = method.upper()
        if method not in _ALLOWED_METHODS:
            return (
                f"Error: Invalid method '{method}'. Allowed: {', '.join(sorted(_ALLOWED_METHODS))}."
            )

        # Rate-Limiting
        limiter = self._get_rate_limiter(name)
        if not limiter.allow():
            return (
                f"Error: Rate limit exceeded for '{name}'. "
                f"{limiter.remaining} requests remaining. "
                "Wait before retrying."
            )

        # URL zusammenbauen
        base_url = cfg.get("base_url", "").rstrip("/")
        if endpoint and not endpoint.startswith("/"):
            endpoint = "/" + endpoint
        url = base_url + endpoint

        # Template fuer Auth-Konfiguration
        template = API_TEMPLATES.get(name)

        # Auth-Header
        auth_headers = _build_auth_headers(cfg, template)
        if not auth_headers and cfg.get("auth_type") != "api_key":
            env_var = cfg.get("credential_env", "")
            return (
                "Error: Credential not available. "
                f"Set environment variable '{env_var}' with your API token."
            )

        # Auth-Query-Parameter (fuer api_key Auth)
        auth_params = _build_auth_params(cfg, template)

        # Alle Header zusammenfuehren
        all_headers: dict[str, str] = {}
        if cfg.get("headers"):
            all_headers.update(cfg["headers"])
        all_headers.update(auth_headers)
        if headers:
            all_headers.update(headers)

        # Content-Type fuer POST/PUT/PATCH mit Body
        if body and method in ("POST", "PUT", "PATCH"):
            all_headers.setdefault("Content-Type", "application/json")

        log.info(
            "api_call_start",
            integration=name,
            method=method,
            endpoint=endpoint,
            # NEVER log credentials
        )

        # HTTP-Request ausfuehren
        try:
            status_code, response_text = await self._do_request(
                method=method,
                url=url,
                headers=all_headers,
                params=auth_params,
                body=body,
            )
        except Exception as exc:
            return f"Error: Request failed: {_mask_credential(str(exc))}"

        # Antwort formatieren
        lines = [f"Status: {status_code}"]
        if response_text:
            # JSON huebsch formatieren wenn moeglich
            try:
                parsed = json.loads(response_text)
                formatted = json.dumps(parsed, indent=2, ensure_ascii=False)
                lines.append(f"\n{_truncate(formatted)}")
            except (json.JSONDecodeError, ValueError):
                lines.append(f"\n{_truncate(response_text)}")
        else:
            lines.append("\n(empty response)")

        return "\n".join(lines)

    async def api_disconnect(
        self,
        *,
        name: str,
    ) -> str:
        """Entfernt eine API-Integration.

        Args:
            name: Integrations-Name.

        Returns:
            Bestaetigung.
        """
        name = name.strip().lower()
        integrations = _load_integrations(self._config)

        if name not in integrations:
            return f"Error: Integration '{name}' not found."

        del integrations[name]
        _save_integrations(self._config, integrations)

        # Rate-Limiter entfernen
        self._rate_limiters.pop(name, None)

        return f"Integration '{name}' removed successfully."

    async def _health_check(
        self,
        integration: dict[str, Any],
        template: dict[str, Any] | None,
        endpoint: str,
    ) -> tuple[bool, str]:
        """Fuehrt einen Health-Check gegen die API durch.

        Returns:
            Tuple (success, message).
        """
        base_url = integration.get("base_url", "").rstrip("/")
        if not endpoint.startswith("/"):
            endpoint = "/" + endpoint
        url = base_url + endpoint

        auth_headers = _build_auth_headers(integration, template)
        all_headers: dict[str, str] = {}
        if integration.get("headers"):
            all_headers.update(integration["headers"])
        all_headers.update(auth_headers)

        auth_params = _build_auth_params(integration, template)

        try:
            status_code, _ = await self._do_request(
                method="GET",
                url=url,
                headers=all_headers,
                params=auth_params,
                timeout=10,
            )
            if 200 <= status_code < 400:
                return True, f"OK (HTTP {status_code})"
            return False, f"Health check returned HTTP {status_code}"
        except Exception as exc:
            return False, f"Health check failed: {_mask_credential(str(exc))}"

    async def _do_request(
        self,
        *,
        method: str,
        url: str,
        headers: dict[str, str] | None = None,
        params: dict[str, str] | None = None,
        body: dict[str, Any] | None = None,
        timeout: int = _DEFAULT_TIMEOUT,
    ) -> tuple[int, str]:
        """Fuehrt den HTTP-Request via httpx aus.

        Falls httpx nicht verfuegbar, Fallback auf urllib.

        Returns:
            Tuple (status_code, response_text).
        """
        # Versuche httpx (bevorzugt)
        try:
            return await self._do_request_httpx(
                method=method,
                url=url,
                headers=headers,
                params=params,
                body=body,
                timeout=timeout,
            )
        except ImportError:
            pass

        # Fallback: urllib (synchron, in Executor)
        import asyncio

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._do_request_urllib(
                method=method,
                url=url,
                headers=headers or {},
                params=params or {},
                body=body,
                timeout=timeout,
            ),
        )

    async def _do_request_httpx(
        self,
        *,
        method: str,
        url: str,
        headers: dict[str, str] | None = None,
        params: dict[str, str] | None = None,
        body: dict[str, Any] | None = None,
        timeout: int = _DEFAULT_TIMEOUT,
    ) -> tuple[int, str]:
        """HTTP-Request via httpx (async)."""
        import httpx

        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            max_redirects=_MAX_REDIRECTS,
        ) as client:
            kwargs: dict[str, Any] = {
                "method": method,
                "url": url,
                "headers": headers,
            }
            if params:
                kwargs["params"] = params
            if body and method in ("POST", "PUT", "PATCH"):
                kwargs["json"] = body

            response = await client.request(**kwargs)
            # SSRF defense: refuse to return a response body that came
            # from a redirect chain visiting a private/loopback IP.
            # The request was already made — but never propagating the
            # body upward stops the planner / LLM from seeing internal
            # data such as cloud-metadata creds or local Ollama state.
            for prior in (*response.history, response):
                host = prior.url.host
                if host and _is_private_host(host):
                    log.warning(
                        "api_hub_redirect_blocked",
                        target_host=host,
                        chain_length=len(response.history),
                    )
                    return 502, "Redirect chain reached a private/loopback host; refused."
            return response.status_code, response.text

    @staticmethod
    def _do_request_urllib(
        *,
        method: str,
        url: str,
        headers: dict[str, str],
        params: dict[str, str],
        body: dict[str, Any] | None = None,
        timeout: int = _DEFAULT_TIMEOUT,
    ) -> tuple[int, str]:
        """HTTP-Request via urllib (synchron, Fallback)."""
        import urllib.parse
        import urllib.request

        if params:
            sep = "&" if "?" in url else "?"
            url = url + sep + urllib.parse.urlencode(params)

        data = None
        if body and method in ("POST", "PUT", "PATCH"):
            data = json.dumps(body).encode("utf-8")

        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status, resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read().decode("utf-8", errors="replace")
        except urllib.error.URLError as exc:
            raise APIHubError(f"Connection failed: {exc.reason}") from exc


def register_api_hub_tools(
    mcp_client: Any,
    config: Any,
) -> APIHub:
    """Registriert API-Hub-Tools beim MCP-Client.

    Returns:
        APIHub-Instanz.
    """
    hub = APIHub(config)

    # api_list — GREEN
    mcp_client.register_builtin_handler(
        "api_list",
        hub.api_list,
        description=(
            "List all configured API integrations and available templates. "
            "Shows connection status, auth type, and base URL."
        ),
        input_schema={
            "type": "object",
            "properties": {},
            "required": [],
        },
    )

    # api_connect — YELLOW
    mcp_client.register_builtin_handler(
        "api_connect",
        hub.api_connect,
        description=(
            "Configure a new API integration. Supports pre-built templates for "
            "GitHub, Jira, Notion, Todoist, Home Assistant, and OpenWeatherMap. "
            "Credentials are never stored -- only the environment variable name."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Integration name (e.g. 'github', 'jira', 'my-api')",
                },
                "base_url": {
                    "type": "string",
                    "description": "API base URL (uses template default if available)",
                },
                "auth_type": {
                    "type": "string",
                    "enum": ["bearer", "api_key", "basic"],
                    "description": "Authentication type (default: bearer)",
                },
                "credential_env": {
                    "type": "string",
                    "description": "Environment variable name containing the token/key",
                },
                "headers": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                    "description": "Additional HTTP headers",
                },
            },
            "required": ["name"],
        },
    )

    # api_call — YELLOW
    mcp_client.register_builtin_handler(
        "api_call",
        hub.api_call,
        description=(
            "Make an authenticated API call to a configured integration. "
            "Supports GET, POST, PUT, DELETE, PATCH, HEAD methods."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "integration": {
                    "type": "string",
                    "description": "Name of the configured integration",
                },
                "method": {
                    "type": "string",
                    "enum": ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD"],
                    "description": "HTTP method (default: GET)",
                    "default": "GET",
                },
                "endpoint": {
                    "type": "string",
                    "description": "API endpoint path (appended to base_url)",
                },
                "body": {
                    "type": "object",
                    "description": "Request body (sent as JSON for POST/PUT/PATCH)",
                },
                "headers": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                    "description": "Additional request headers",
                },
            },
            "required": ["integration"],
        },
    )

    # api_disconnect — YELLOW
    mcp_client.register_builtin_handler(
        "api_disconnect",
        hub.api_disconnect,
        description=(
            "Remove an API integration configuration. "
            "This does not revoke any API keys -- only removes the local config."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the integration to remove",
                },
            },
            "required": ["name"],
        },
    )

    log.info(
        "api_hub_tools_registered",
        tools=["api_list", "api_connect", "api_call", "api_disconnect"],
        templates=list(API_TEMPLATES.keys()),
    )
    return hub
