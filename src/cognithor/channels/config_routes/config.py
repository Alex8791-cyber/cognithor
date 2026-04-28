"""Cognithor · Config read / write routes.

Sub-Modul des `config_routes`-Pakets (siehe
`docs/superpowers/plans/2026-04-29-config-routes-split.md`). Enthaelt
`_register_config_routes()` — registriert Health, Config-CRUD, Locales,
Translation, Presets, Revisions, Network-Endpoints, Devices und das
Config-Section-Routing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

try:
    from starlette.requests import Request
except ImportError:
    Request = Any  # type: ignore[assignment,misc]

try:
    from fastapi import HTTPException
except ImportError:
    try:
        from starlette.exceptions import HTTPException  # type: ignore[assignment]
    except ImportError:
        HTTPException = Exception  # type: ignore[assignment,misc]

from cognithor.utils.logging import get_logger

if TYPE_CHECKING:
    from cognithor.config_manager import ConfigManager

log = get_logger(__name__)


# ======================================================================
# Config read / write routes
# ======================================================================


def _register_config_routes(
    app: Any,
    deps: list[Any],
    config_manager: ConfigManager,
    gateway: Any = None,
) -> None:
    """Config CRUD, presets, reload."""

    @app.get("/api/v1/health")
    async def health_check() -> dict[str, Any]:
        """Health check endpoint used by the Vite launcher."""
        return {"status": "ok"}

    @app.get("/api/v1/config", dependencies=deps)
    async def get_config() -> dict[str, Any]:
        """Gibt die gesamte Konfiguration zurueck (ohne Secrets)."""
        data = config_manager.read()
        data["_meta"] = {
            "editable_sections": config_manager.editable_sections(),
            "editable_top_level": config_manager.editable_top_level_fields(),
        }
        return data

    @app.patch("/api/v1/config", dependencies=deps)
    async def update_config_top_level(updates: dict[str, Any]) -> dict[str, Any]:
        """Aktualisiert Top-Level-Felder."""
        from cognithor.config_manager import _is_secret_field

        results: list[dict[str, Any]] = []
        for key, value in updates.items():
            # Skip masked secret values — the UI sends "***" for untouched secrets.
            # Real changes (new value or "") are passed through and persisted.
            if value == "***" and _is_secret_field(key):
                results.append({"key": key, "status": "skipped"})
                continue
            try:
                config_manager.update_top_level(key, value)
                results.append({"key": key, "status": "ok"})
            except ValueError as exc:
                log.error("config_update_key_failed", key=key, error=str(exc))
                results.append({"key": key, "status": "error", "error": "Ungueltige Konfiguration"})
        config_manager.save()
        # Trigger live-reload of runtime components.
        # Language changes must also reload the planner's cached system prompts
        # (issue #136) — otherwise the planner keeps the German preset it
        # loaded at startup and the LLM keeps answering in German.
        if gateway is not None and hasattr(gateway, "reload_components"):
            gateway.reload_components(
                config=True,
                prompts="language" in updates,
            )
        # Sync backend i18n locale when language changes (#33)
        if "language" in updates:
            try:
                from cognithor.i18n import set_locale

                set_locale(updates["language"])
            except Exception:
                log.debug("config_locale_sync_failed", exc_info=True)
        return {"results": results}

    @app.post("/api/v1/config/reload", dependencies=deps)
    async def reload_config() -> dict[str, Any]:
        """Laedt die Konfiguration neu aus der Datei."""
        config_manager.reload()
        if gateway is not None and hasattr(gateway, "reload_components"):
            gateway.reload_components(prompts=True, policies=True, core_memory=True, config=True)
        return {"status": "ok", "message": "Konfiguration und Komponenten neu geladen"}

    @app.post("/api/v1/config/factory-reset", dependencies=deps)
    async def factory_reset_config() -> dict[str, Any]:
        """Reset configuration to defaults."""
        from cognithor.config import CognithorConfig

        try:
            config_manager._config = CognithorConfig()
            config_manager.save()
            if gateway is not None and hasattr(gateway, "reload_components"):
                gateway.reload_components(config=True)
            return {"status": "ok", "message": "Configuration reset to defaults"}
        except Exception as exc:
            return {"error": str(exc)}

    # -- Locales (available i18n language packs) --

    @app.get("/api/v1/locales", dependencies=deps)
    async def list_locales() -> dict[str, Any]:
        """Returns available i18n locales and the currently active one."""
        from cognithor.i18n import get_available_locales, get_locale
        from cognithor.i18n.prompt_presets import available_preset_locales

        locales = get_available_locales()
        return {
            "locales": locales,
            "active": get_locale(),
            "preset_locales": available_preset_locales(),
        }

    @app.post("/api/v1/translate-prompts", dependencies=deps)
    async def translate_prompts(request: Request) -> dict[str, Any]:
        """Translate system prompts to a target language.

        Supports two methods:
          - ``"preset"`` — Use curated prompt presets (instant, no LLM needed).
          - ``"ollama"`` — Use local Ollama LLM to translate on-the-fly.

        Request body::

            {
              "target_locale": "en",
              "method": "preset" | "ollama",
              "prompts": {                     // only for method=ollama
                "plannerSystem": "...",
                "replanPrompt": "...",
                "escalationPrompt": "..."
              }
            }

        Returns::

            {
              "translations": {
                "plannerSystem": "...",
                "replanPrompt": "...",
                "escalationPrompt": "..."
              },
              "method": "preset" | "ollama"
            }
        """
        try:
            body = await request.json()
        except Exception:
            return {"error": "Invalid JSON body", "status": 400}

        target_locale = body.get("target_locale", "").strip().lower()
        method = body.get("method", "preset").strip().lower()

        if not target_locale:
            return {"error": "target_locale is required", "status": 400}
        if method not in ("preset", "ollama"):
            return {"error": "method must be 'preset' or 'ollama'", "status": 400}

        # ── Method: preset ──────────────────────────────────────────
        if method == "preset":
            from cognithor.i18n.prompt_presets import get_preset

            preset = get_preset(target_locale)
            if preset is None:
                from cognithor.i18n.prompt_presets import available_preset_locales

                return {
                    "error": f"No preset available for '{target_locale}'",
                    "available": available_preset_locales(),
                    "status": 404,
                }
            return {"translations": preset, "method": "preset"}

        # ── Method: ollama ──────────────────────────────────────────
        prompts = body.get("prompts", {})
        if not prompts:
            return {"error": "prompts dict is required for method=ollama", "status": 400}

        cfg = config_manager.config
        ollama_url = cfg.ollama.base_url.rstrip("/")
        model = cfg.models.planner.name
        timeout = cfg.ollama.timeout_seconds

        import httpx

        translations: dict[str, str] = {}
        errors: list[str] = []

        async with httpx.AsyncClient(timeout=timeout) as client:
            for key, text in prompts.items():
                if key not in ("plannerSystem", "replanPrompt", "escalationPrompt"):
                    continue
                if not text or not text.strip():
                    continue

                system_msg = (
                    f"You are a professional translator. Translate the following system "
                    f"prompt to {target_locale}. Preserve ALL template variables exactly "
                    f"as they are (e.g., {{tools_section}}, {{owner_name}}, "
                    f"{{results_section}}, {{original_goal}}, {{tool}}, {{reason}}). "
                    f"Preserve all markdown formatting, code blocks, and JSON structures. "
                    f"Output ONLY the translated text, nothing else."
                )

                try:
                    resp = await client.post(
                        f"{ollama_url}/api/chat",
                        json={
                            "model": model,
                            "messages": [
                                {"role": "system", "content": system_msg},
                                {"role": "user", "content": text},
                            ],
                            "stream": False,
                        },
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    translated = data.get("message", {}).get("content", "").strip()

                    # Validate that template variables survived translation
                    import re

                    original_vars = set(re.findall(r"\{[\w_]+\}", text))
                    translated_vars = set(re.findall(r"\{[\w_]+\}", translated))
                    missing = original_vars - translated_vars
                    if missing:
                        errors.append(f"{key}: template variables lost in translation: {missing}")
                        # Still include the translation but flag it
                        translations[key] = translated
                    else:
                        translations[key] = translated
                except httpx.TimeoutException:
                    errors.append(f"{key}: Ollama request timed out")
                except httpx.HTTPStatusError as exc:
                    errors.append(f"{key}: Ollama HTTP {exc.response.status_code}")
                except Exception as exc:
                    errors.append(f"{key}: {exc!s}")

        result: dict[str, Any] = {"translations": translations, "method": "ollama"}
        if errors:
            result["warnings"] = errors
        return result

    # -- Presets (BEFORE {section} routes to avoid path parameter conflict) --

    @app.get("/api/v1/config/presets", dependencies=deps)
    async def list_presets() -> dict[str, Any]:
        """Listet verfuegbare Konfigurations-Presets."""
        return {
            "presets": [
                {
                    "name": "minimal",
                    "description": "Minimale Konfiguration (CLI-only, kleine Modelle)",
                    "sections": {
                        "channels": {
                            "cli_enabled": True,
                            "telegram_enabled": False,
                            "webui_enabled": False,
                        },
                        "heartbeat": {"enabled": False},
                        "dashboard": {"enabled": False},
                    },
                },
                {
                    "name": "standard",
                    "description": "Standard-Setup (CLI + WebUI, Heartbeat, Dashboard)",
                    "sections": {
                        "channels": {"cli_enabled": True, "webui_enabled": True},
                        "heartbeat": {"enabled": True, "interval_minutes": 30},
                        "dashboard": {"enabled": True},
                    },
                },
                {
                    "name": "full",
                    "description": "Vollausbau (alle Channels, Heartbeat, Dashboard, Plugins)",
                    "sections": {
                        "channels": {
                            "cli_enabled": True,
                            "webui_enabled": True,
                            "telegram_enabled": True,
                            "slack_enabled": True,
                            "discord_enabled": True,
                        },
                        "heartbeat": {"enabled": True, "interval_minutes": 15},
                        "dashboard": {"enabled": True},
                        "plugins": {"auto_update": True},
                    },
                },
            ],
        }

    @app.post("/api/v1/config/presets/{preset_name}", dependencies=deps)
    async def apply_preset(preset_name: str) -> dict[str, Any]:
        """Wendet ein Konfigurations-Preset an."""
        presets = (await list_presets())["presets"]
        preset = next((p for p in presets if p["name"] == preset_name), None)
        if not preset:
            return {"error": f"Preset '{preset_name}' nicht gefunden", "status": 404}
        results = []
        for section, values in preset["sections"].items():
            try:
                config_manager.update_section(section, values)
                results.append({"section": section, "status": "ok"})
            except ValueError as exc:
                log.warning("preset_section_update_failed", section=section, error=str(exc))
                results.append(
                    {"section": section, "status": "error", "error": "Ungueltige Konfiguration"}
                )
        config_manager.save()
        return {"preset": preset_name, "results": results}

    # -- Config Versioning / Rollback ------------------------------------------

    @app.get("/api/v1/config/revisions", dependencies=deps)
    async def list_config_revisions() -> dict[str, Any]:
        """List all saved config revisions (newest first)."""
        from cognithor.core.config_versioning import list_revisions

        return {"revisions": list_revisions()}

    @app.post("/api/v1/config/rollback/{revision_id}", dependencies=deps)
    async def rollback_config(revision_id: str) -> dict[str, Any]:
        """Roll back the config to a previous revision.

        Saves the current config as a new revision first, then applies
        the historic config and persists it.
        """
        from cognithor.core.config_versioning import rollback_to, save_config_revision

        # Save current state before rollback
        try:
            current = config_manager.read(include_secrets=True)
            save_config_revision(current, reason=f"pre-rollback to {revision_id}")
        except Exception:
            log.warning("config_pre_rollback_save_failed", exc_info=True)

        try:
            historic_config = rollback_to(revision_id)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail=f"Revision '{revision_id}' not found",
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        # Apply the historic config via Pydantic validation
        from pydantic import ValidationError

        from cognithor.config import CognithorConfig

        try:
            new_cfg = CognithorConfig(**historic_config)
        except ValidationError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Historic config fails validation: {exc}",
            ) from exc

        config_manager._config = new_cfg
        config_manager.save()

        if gateway is not None and hasattr(gateway, "reload_components"):
            gateway.reload_components(config=True)

        return {"status": "ok", "revision_id": revision_id, "message": "Rollback applied"}

    # -- Network Endpoints -------------------------------------------------------

    @app.get("/api/v1/network/interfaces", dependencies=deps)
    async def list_network_interfaces() -> dict[str, Any]:
        """Detected network interfaces with enable/disable status."""
        try:
            from cognithor.core.network_endpoints import NetworkEndpointManager

            mgr = NetworkEndpointManager()
            return {"interfaces": mgr.get_detected_interfaces()}
        except Exception as exc:
            return {"interfaces": [], "error": str(exc)}

    @app.put("/api/v1/network/endpoints", dependencies=deps)
    async def update_network_endpoints(request: Request) -> dict[str, Any]:
        """Update which network interfaces are enabled for API binding."""
        body = await request.json()
        try:
            from cognithor.core.network_endpoints import NetworkEndpointManager

            mgr = NetworkEndpointManager()
            if "enabled_ips" in body:
                mgr.set_enabled_ips(body["enabled_ips"])
            if "auto_detect" in body:
                mgr.set_auto_detect(body["auto_detect"])
            return {
                "status": "ok",
                "bind_host": mgr.get_bind_host(),
                "active_ips": mgr.get_active_ips(),
                "message": "Restart required for bind changes to take effect",
            }
        except Exception as exc:
            return {"error": str(exc)}

    # -- Device Pairing (Mobile) ------------------------------------------------

    @app.get("/api/v1/devices", dependencies=deps)
    async def list_paired_devices() -> dict[str, Any]:
        """List all paired devices."""
        try:
            from cognithor.security.device_pairing import DevicePairingManager

            _secret = getattr(gateway, "_internal_api_token", "") if gateway else ""
            mgr = DevicePairingManager(master_secret=_secret or "fallback")
            return {"devices": mgr.list_devices()}
        except Exception as exc:
            return {"devices": [], "error": str(exc)}

    @app.post("/api/v1/devices/pair", dependencies=deps)
    async def pair_device(request: Request) -> dict[str, Any]:
        """Create a new pairing token for a mobile device."""
        body = await request.json()
        device_name = body.get("name", "Unknown Device")
        try:
            from cognithor.security.device_pairing import DevicePairingManager

            _secret = getattr(gateway, "_internal_api_token", "") if gateway else ""
            mgr = DevicePairingManager(master_secret=_secret or "fallback")
            pt = mgr.create_pairing_token(device_name)
            _port = request.url.port or 8741
            try:
                from cognithor.utils.network import get_reachable_url

                _reach_url = get_reachable_url("0.0.0.0", _port)
                # Extract host from URL
                from urllib.parse import urlparse

                _host = urlparse(_reach_url).hostname or "127.0.0.1"
            except ImportError:
                _host = request.client.host if request.client else "127.0.0.1"
            qr = mgr.qr_payload(pt, _host, _port)
            return {
                "device_id": pt.device_id,
                "token": pt.token,
                "expires_at": pt.expires_at,
                "qr_payload": qr,
            }
        except Exception as exc:
            return {"error": str(exc)}

    @app.delete("/api/v1/devices/{device_id}", dependencies=deps)
    async def revoke_device(device_id: str) -> dict[str, Any]:
        """Revoke a paired device."""
        try:
            from cognithor.security.device_pairing import DevicePairingManager

            _secret = getattr(gateway, "_internal_api_token", "") if gateway else ""
            mgr = DevicePairingManager(master_secret=_secret or "fallback")
            if mgr.revoke_device(device_id):
                return {"status": "revoked", "device_id": device_id}
            return {"error": "Device not found", "status": 404}
        except Exception as exc:
            return {"error": str(exc)}

    # -- Config Section CRUD (AFTER presets to avoid {section} capturing "presets") --

    @app.get("/api/v1/config/{section}", dependencies=deps)
    async def get_config_section(section: str) -> dict[str, Any]:
        """Gibt eine einzelne Konfigurations-Sektion zurueck."""
        result = config_manager.read_section(section)
        if result is None:
            return {"error": f"Sektion '{section}' nicht gefunden", "status": 404}
        return {"section": section, "values": result}

    @app.patch("/api/v1/config/{section}", dependencies=deps)
    async def update_config_section(section: str, values: dict[str, Any]) -> dict[str, Any]:
        """Aktualisiert eine Konfigurations-Sektion."""
        from cognithor.config_manager import _is_secret_field

        def _deep_clean_secrets(
            data: dict[str, Any],
            existing: dict[str, Any] | None = None,
            *,
            _depth: int = 0,
        ) -> dict[str, Any]:
            """Recursively strip masked ('***') and empty secret values."""
            if _depth > 5:
                return data
            out: dict[str, Any] = {}
            for k, v in data.items():
                if isinstance(v, dict):
                    ex_sub = existing.get(k) if isinstance(existing, dict) else None
                    cleaned_sub = _deep_clean_secrets(
                        v,
                        ex_sub if isinstance(ex_sub, dict) else None,
                        _depth=_depth + 1,
                    )
                    if cleaned_sub:  # only include non-empty dicts
                        out[k] = cleaned_sub
                elif _is_secret_field(k):
                    if v == "***":
                        continue  # skip masked placeholders
                    if (v == "" or v is None) and existing:
                        ex_val = existing.get(k, "") if isinstance(existing, dict) else ""
                        if ex_val and ex_val != "":
                            continue  # protect existing non-empty secret
                    out[k] = v
                else:
                    out[k] = v
            return out

        # Get existing section values (raw, unmasked) for protection comparison
        raw_cfg = config_manager.config.model_dump(mode="json")
        existing_section = (
            raw_cfg.get(section, {}) if isinstance(raw_cfg.get(section), dict) else {}
        )
        cleaned = _deep_clean_secrets(values, existing_section)
        try:
            config_manager.update_section(section, cleaned)
            config_manager.save()
            # Trigger live-reload of runtime components (executor, web tools, model router)
            if gateway is not None and hasattr(gateway, "reload_components"):
                gateway.reload_components(config=True)

            # Validate model availability when saving 'models' section
            model_warnings: list[str] = []
            if section == "models":
                router = getattr(gateway, "_model_router", None) if gateway else None
                if router and router._available_models:
                    available = router._available_models
                    cfg_models = config_manager.config.models
                    for role, model_cfg in [
                        ("planner", cfg_models.planner),
                        ("executor", cfg_models.executor),
                        ("coder", cfg_models.coder),
                        ("embedding", cfg_models.embedding),
                    ]:
                        if model_cfg.name and model_cfg.name not in available:
                            model_warnings.append(
                                f"Modell '{model_cfg.name}' ({role}) nicht in Ollama gefunden. "
                                f"Bitte installieren: ollama pull {model_cfg.name}"
                            )

            result: dict[str, Any] = {
                "status": "ok",
                "section": section,
                "updated_keys": list(cleaned.keys()),
            }
            if model_warnings:
                result["warnings"] = model_warnings
            return result
        except ValueError as exc:
            log.warning("config_section_update_failed", section=section, error=str(exc))
            return {"error": "Ungueltige Konfiguration", "status": 400}
