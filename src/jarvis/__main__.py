"""
Cognithor · Agent OS -- Entry Point.

Usage: cognithor
       cognithor --config /path/to/config.yaml
       cognithor --version
       python -m jarvis
"""

from __future__ import annotations

import argparse
import contextlib
from pathlib import Path
import os
import sys
from typing import Any

from jarvis import __version__

# WebSocket/FastAPI types must be at module level so that
# `from __future__ import annotations` (PEP 563) can resolve
# string-ified type hints via get_type_hints().
try:
    from starlette.websockets import WebSocket, WebSocketDisconnect
except ImportError:  # pragma: no cover
    WebSocket = None  # type: ignore[assignment,misc]
    WebSocketDisconnect = None  # type: ignore[assignment,misc]


def parse_args() -> argparse.Namespace:
    """Kommandozeilen-Argumente parsen."""
    parser = argparse.ArgumentParser(
        prog="cognithor",
        description="Cognithor · Agent OS -- Local-first autonomous agent operating system",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"Cognithor v{__version__}",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Pfad zur config.yaml (Default: ~/.jarvis/config.yaml)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Log-Level überschreiben",
    )
    parser.add_argument(
        "--init-only",
        action="store_true",
        help="Nur Verzeichnisstruktur erstellen, nicht starten",
    )
    parser.add_argument(
        "--no-cli",
        action="store_true",
        help="CLI-Channel nicht starten (Headless-Betrieb für Control Center)",
    )
    parser.add_argument(
        "--api-port",
        type=int,
        default=8741,
        help="Port für die Control Center API (Default: 8741)",
    )
    parser.add_argument(
        "--api-host",
        type=str,
        default=None,
        help="Host für die Control Center API (Default: JARVIS_API_HOST env oder 127.0.0.1)",
    )
    parser.add_argument(
        "--lite",
        action="store_true",
        help="Lite-Modus: qwen3:8b als Planner und Executor (6 GB statt 26 GB VRAM)",
    )
    return parser.parse_args()


def _check_python_version() -> None:
    """Sicherstellen, dass Python >= 3.12 läuft."""
    if sys.version_info < (3, 12):
        sys.exit(
            f"Cognithor benötigt Python >= 3.12, "
            f"aktuell: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}\n"
            f"Bitte installiere eine neuere Python-Version: https://www.python.org/downloads/"
        )


def main() -> None:
    """Haupteintrittspunkt für Jarvis."""
    _check_python_version()

    # Windows: stdout/stderr auf UTF-8 umstellen, damit Umlaute in cmd.exe
    # (ohne chcp 65001) nicht zu Encoding-Crashes fuehren.
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                try:
                    stream.reconfigure(encoding="utf-8", errors="replace")
                except Exception:
                    pass

    args = parse_args()

    # 0. .env-Datei laden (Secrets aus ~/.jarvis/.env oder Projekt-Root)
    try:
        from dotenv import load_dotenv

        # Zuerst Projekt-.env, dann User-.env (User überschreibt)
        load_dotenv(Path(".env"), override=False)
        load_dotenv(Path.home() / ".jarvis" / ".env", override=True)
    except ImportError:
        pass  # python-dotenv optional

    # 1. Konfiguration laden
    from jarvis.config import ensure_directory_structure, load_config

    config = load_config(args.config)

    # 1.5 Lite-Modus: kleinere Modelle fuer niedrigen VRAM-Verbrauch
    if args.lite:
        config.models.planner.name = "qwen3:8b"
        config.models.coder.name = "qwen2.5-coder:7b"

    # 2. Verzeichnisstruktur sicherstellen
    created = ensure_directory_structure(config)

    # 3. Logging initialisieren
    from jarvis.utils.logging import setup_logging

    log_level = args.log_level or config.logging.level
    setup_logging(
        level=log_level,
        log_dir=config.logs_dir,
        json_logs=config.logging.json_logs,
        console=config.logging.console,
    )

    from jarvis.utils.logging import get_logger

    log = get_logger("jarvis")

    # 3.5 Startup-Check: Fehlende Abhängigkeiten automatisch laden
    from jarvis.core.startup_check import StartupChecker

    checker = StartupChecker(config)
    report = checker.check_and_fix_all()
    if report.fixes_applied:
        log.info("startup_auto_fixes", fixes=report.fixes_applied, warnings=report.warnings)
    if report.errors:
        log.warning("startup_check_errors", errors=report.errors)

    # 4. Startup-Info
    log.info(
        "jarvis_starting",
        version=__version__,
        home=str(config.jarvis_home),
        log_level=log_level,
    )

    if created:
        for path in created:
            log.info("created_path", path=path)

    if args.init_only:
        log.info("init_complete", paths_created=len(created))
        log.info(
            "init_summary",
            version=__version__,
            home=str(config.jarvis_home),
            config_file=str(config.config_file),
            paths_created=len(created),
        )
        return

    # 5. System-Check -- startup banner (intentional CLI output)
    _api_host = args.api_host or os.environ.get("JARVIS_API_HOST", "127.0.0.1")
    _print_banner(config, api_host=_api_host, api_port=args.api_port, lite=args.lite)

    # Phase 0 Checkpoint: Setup OK
    log.info(
        "setup_ok",
        backend=getattr(config, "llm_backend_type", "ollama"),
        planner_model=config.models.planner.name,
        executor_model=config.models.executor.name,
    )

    # Phase 1: Gateway + CLI starten
    import asyncio

    async def run() -> None:
        """Startet den Gateway und CLI-Channel als asynchrone Hauptschleife."""
        from jarvis.channels.cli import CliChannel
        from jarvis.gateway.gateway import Gateway

        gateway = Gateway(config)
        api_server = None

        try:
            # Alle Subsysteme initialisieren
            await gateway.initialize()

            # LLM-Erreichbarkeit pruefen und prominent warnen
            _llm = getattr(gateway, "_llm", None)
            if _llm and not await _llm.is_available():
                _backend = getattr(_llm, "backend_type", "ollama")
                print()
                print("!" * 60)
                print("  WARNUNG: Sprachmodell nicht erreichbar!")
                print("!" * 60)
                if _backend == "ollama":
                    _ollama_url = config.ollama.base_url
                    print(f"  Ollama antwortet nicht unter {_ollama_url}")
                    print()
                    print("  Bitte starte Ollama:")
                    print("    ollama serve")
                    print()
                    print("  Falls noch keine Modelle installiert sind:")
                    print(f"    ollama pull {config.models.planner.name}")
                    print(f"    ollama pull {config.models.executor.name}")
                elif _backend == "lmstudio":
                    print(f"  LM Studio ist nicht erreichbar unter {config.lmstudio_base_url}")
                    print("  Bitte starte LM Studio und lade ein Modell.")
                else:
                    print(f"  LLM-Backend '{_backend}' ist nicht erreichbar.")
                    print("  Bitte pruefe deine API-Keys und Netzwerkverbindung.")
                print()
                print("  Jarvis startet trotzdem, aber Anfragen werden fehlschlagen")
                print("  bis das Sprachmodell erreichbar ist.")
                print("!" * 60)
                print()

            # SessionStore-Referenz für Channel-Persistenz
            _session_store = getattr(gateway, "_session_store", None)

            # SSL-Config für TLS-fähige Channels
            _ssl_cert = config.security.ssl_certfile
            _ssl_key = config.security.ssl_keyfile

            # Control Center API-Server starten (immer, Port 8741)
            try:
                from fastapi import FastAPI
                from fastapi.middleware.cors import CORSMiddleware
                import uvicorn
                from jarvis.channels.config_routes import create_config_routes
                from jarvis.config_manager import ConfigManager

                api_host = args.api_host or os.environ.get("JARVIS_API_HOST", "127.0.0.1")

                # CORS: Wenn API-Token gesetzt, Origins einschränken
                api_token = os.environ.get("JARVIS_API_TOKEN")
                if api_token:
                    cors_raw = os.environ.get("JARVIS_API_CORS_ORIGINS", "")
                    cors_origins = [o.strip() for o in cors_raw.split(",") if o.strip()] if cors_raw else []
                else:
                    cors_origins = ["*"]

                # allow_credentials nur wenn Origins explizit eingeschränkt
                _allow_creds = cors_origins != ["*"]

                api_app = FastAPI(title="Cognithor Control Center API")
                api_app.add_middleware(
                    CORSMiddleware,
                    allow_origins=cors_origins,
                    allow_credentials=_allow_creds,
                    allow_methods=["*"],
                    allow_headers=["*"],
                )

                # ── Rate Limiting Middleware ──────────────────────────────
                import time as _time
                from collections import defaultdict as _defaultdict
                from starlette.middleware.base import BaseHTTPMiddleware
                from starlette.responses import JSONResponse as _JSONResponse

                _rate_limit = int(os.environ.get("JARVIS_API_RATE_LIMIT", "60"))
                _rate_window = 60.0  # seconds
                _rate_exempt = {"/api/v1/health"}
                _rate_hits: dict[str, list[float]] = _defaultdict(list)

                class _RateLimitMiddleware(BaseHTTPMiddleware):
                    async def dispatch(self, request, call_next):
                        if request.url.path in _rate_exempt:
                            return await call_next(request)
                        client = request.client.host if request.client else "unknown"
                        now = _time.monotonic()
                        hits = _rate_hits[client]
                        # Purge expired entries
                        cutoff = now - _rate_window
                        _rate_hits[client] = hits = [t for t in hits if t > cutoff]
                        if len(hits) >= _rate_limit:
                            return _JSONResponse(
                                {"error": "Too many requests", "retry_after_seconds": int(_rate_window)},
                                status_code=429,
                            )
                        hits.append(now)
                        return await call_next(request)

                api_app.add_middleware(_RateLimitMiddleware)

                # Health-Endpoint
                _api_start = _time.monotonic()

                @api_app.get("/api/v1/health")
                async def _cc_health() -> dict[str, Any]:
                    return {
                        "status": "ok",
                        "version": __version__,
                        "uptime_seconds": _time.monotonic() - _api_start,
                    }

                config_mgr = ConfigManager(config=config)
                create_config_routes(api_app, config_mgr, gateway=gateway)

                # Skill Marketplace API Router einbinden
                if getattr(config, "marketplace", None) and config.marketplace.enabled:
                    try:
                        from jarvis.skills.api import router as skills_router
                        if skills_router is not None:
                            api_app.include_router(skills_router)
                            log.info("skills_marketplace_api_registered")
                    except Exception as _skills_exc:
                        log.warning("skills_marketplace_api_failed", error=str(_skills_exc))

                # Community Marketplace API Router einbinden
                _cm_cfg = getattr(config, "community_marketplace", None)
                if _cm_cfg and getattr(_cm_cfg, "enabled", False):
                    try:
                        from jarvis.skills.api import community_router
                        if community_router is not None:
                            api_app.include_router(community_router)
                            log.info("community_marketplace_api_registered")
                        else:
                            log.warning("community_marketplace_router_none", hint="FastAPI nicht installiert?")
                    except Exception as _cm_exc:
                        log.warning("community_marketplace_api_failed", error=str(_cm_exc))

                # ── WebSocket Chat-Endpoint ──────────────────────────────
                import json as _json
                _ws_connections: dict[str, WebSocket] = {}

                @api_app.websocket("/ws/{session_id}")
                async def _cc_ws(websocket: WebSocket, session_id: str) -> None:
                    # ── Token-based authentication ────────────────────────
                    required_token = os.environ.get("JARVIS_API_TOKEN")
                    if required_token:
                        import hmac as _hmac
                        client_token = websocket.query_params.get("token") or ""
                        if not _hmac.compare_digest(client_token, required_token):
                            await websocket.close(code=4001, reason="Unauthorized")
                            log.warning(
                                "cc_ws_auth_rejected",
                                session_id=session_id,
                                reason="missing_or_invalid_token",
                            )
                            return

                    await websocket.accept()

                    # ── Session collision: close existing connection ──────
                    existing = _ws_connections.get(session_id)
                    if existing is not None:
                        log.info("cc_ws_closing_stale", session_id=session_id)
                        try:
                            await existing.close(code=4002, reason="Session replaced")
                        except Exception:
                            pass  # already closed / broken

                    _ws_connections[session_id] = websocket
                    log.info("cc_ws_connected", session_id=session_id)
                    try:
                        while True:
                            raw = await websocket.receive_text()
                            try:
                                msg = _json.loads(raw)
                            except _json.JSONDecodeError:
                                await websocket.send_json({"type": "error", "error": "Ungültiges JSON"})
                                continue

                            msg_type = msg.get("type", "")

                            if msg_type == "ping":
                                await websocket.send_json({"type": "pong"})
                                continue

                            if msg_type in ("user_message", "message"):
                                text = (msg.get("text") or "").strip()
                                metadata = msg.get("metadata", {})
                                if not text:
                                    await websocket.send_json({"type": "error", "error": "Leere Nachricht"})
                                    continue

                                # ── Audio transcription ────────────────────
                                audio_b64 = metadata.get("audio_base64")
                                if not audio_b64 and metadata.get("file_type", "").startswith("audio/"):
                                    audio_b64 = metadata.get("file_base64")
                                if audio_b64:
                                    import base64 as _b64
                                    import tempfile as _tmpfile
                                    audio_type = metadata.get("audio_type") or metadata.get("file_type") or "audio/webm"
                                    ext = {"audio/webm": ".webm", "audio/ogg": ".ogg", "audio/wav": ".wav", "audio/mp3": ".mp3", "audio/mpeg": ".mp3", "audio/m4a": ".m4a", "audio/flac": ".flac"}.get(audio_type, ".webm")
                                    tmp_path = None
                                    try:
                                        raw_audio = _b64.b64decode(audio_b64)
                                        with _tmpfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                                            tmp.write(raw_audio)
                                            tmp_path = tmp.name
                                        from jarvis.mcp.media import MediaPipeline
                                        _media = MediaPipeline()
                                        result = await _media.transcribe_audio(tmp_path, language="de")
                                        if result.success and result.text and result.text.strip():
                                            text = result.text.strip()
                                            log.info("ws_audio_transcribed", text=text[:80])
                                            # Remove raw audio from metadata so gateway
                                            # doesn't see the placeholder context
                                            metadata.pop("audio_base64", None)
                                            metadata.pop("file_base64", None)
                                            metadata["transcribed_from"] = "audio"
                                            # Tell frontend the real text so it can
                                            # update the user bubble
                                            await websocket.send_json({
                                                "type": "transcription",
                                                "text": text,
                                                "session_id": session_id,
                                            })
                                        else:
                                            log.warning("ws_audio_transcription_failed", error=getattr(result, "error", ""))
                                            await websocket.send_json({"type": "error", "error": "Audiodatei konnte nicht transkribiert werden."})
                                            continue
                                    except Exception as _audio_exc:
                                        log.error("ws_audio_transcription_error", error=str(_audio_exc))
                                        await websocket.send_json({"type": "error", "error": "Fehler bei der Audio-Transkription."})
                                        continue
                                    finally:
                                        if tmp_path:
                                            try:
                                                import os as _os
                                                _os.unlink(tmp_path)
                                            except Exception:
                                                pass

                                from jarvis.models import IncomingMessage
                                incoming = IncomingMessage(
                                    text=text,
                                    channel="webui",
                                    session_id=session_id,
                                    user_id="web_user",
                                    metadata=metadata,
                                )
                                try:
                                    response = await gateway.handle_message(incoming)
                                    await websocket.send_json({
                                        "type": "assistant_message",
                                        "text": response.text,
                                        "session_id": session_id,
                                    })
                                    await websocket.send_json({
                                        "type": "stream_end",
                                        "session_id": session_id,
                                    })
                                except Exception as _ws_exc:
                                    log.error("cc_ws_handler_error", error=str(_ws_exc))
                                    await websocket.send_json({
                                        "type": "error",
                                        "error": "Verarbeitungsfehler aufgetreten.",
                                    })
                                continue

                            await websocket.send_json({"type": "error", "error": f"Unbekannter Typ: {msg_type}"})
                    except WebSocketDisconnect:
                        log.info("cc_ws_disconnected", session_id=session_id)
                    except Exception as _ws_exc:
                        log.error("cc_ws_error", error=str(_ws_exc), session_id=session_id)
                    finally:
                        _ws_connections.pop(session_id, None)

                log.info("cc_websocket_endpoint_registered")

                # ── TTS-Endpoint (Piper) ─────────────────────────────────
                _voice_cfg = getattr(getattr(config, "channels", None), "voice_config", None)
                _default_piper_voice = getattr(_voice_cfg, "piper_voice", "de_DE-thorsten_emotional-medium") if _voice_cfg else "de_DE-thorsten_emotional-medium"
                _default_length_scale = getattr(_voice_cfg, "piper_length_scale", 1.0) if _voice_cfg else 1.0

                @api_app.post("/api/v1/tts")
                async def _cc_tts(body: dict[str, Any]) -> Any:
                    """Text-to-Speech via Piper TTS."""
                    from fastapi.responses import Response
                    text = (body.get("text") or "").strip()
                    if not text:
                        return {"error": "Kein Text angegeben"}

                    voice = body.get("voice", _default_piper_voice)
                    length_scale = body.get("length_scale", _default_length_scale)
                    try:
                        wav_bytes = await _run_piper_tts(text, voice, length_scale)
                        return Response(content=wav_bytes, media_type="audio/wav")
                    except FileNotFoundError:
                        return {"error": "Piper TTS nicht installiert. Bitte: pip install piper-tts"}
                    except Exception as _tts_exc:
                        log.error("tts_error", error=str(_tts_exc))
                        return {"error": f"TTS-Fehler: {_tts_exc}"}

                @api_app.get("/api/v1/tts/voices")
                async def _cc_tts_voices() -> dict[str, Any]:
                    """Listet verfuegbare Piper-Stimmen und die aktuell konfigurierte."""
                    voices_dir = Path(config.jarvis_home) / "voices"
                    installed: list[str] = []
                    if voices_dir.exists():
                        installed = [f.stem for f in voices_dir.glob("*.onnx")]
                    return {
                        "current": _default_piper_voice,
                        "installed": installed,
                        "available": [
                            {"id": "de_DE-pavoque-low", "name": "Pavoque (Maennlich, Bariton)", "quality": "low"},
                            {"id": "de_DE-karlsson-low", "name": "Karlsson (Maennlich)", "quality": "low"},
                            {"id": "de_DE-thorsten-high", "name": "Thorsten (Maennlich)", "quality": "high"},
                            {"id": "de_DE-thorsten-medium", "name": "Thorsten (Maennlich)", "quality": "medium"},
                            {"id": "de_DE-thorsten_emotional-medium", "name": "Thorsten Emotional", "quality": "medium"},
                            {"id": "de_DE-kerstin-low", "name": "Kerstin (Weiblich)", "quality": "low"},
                            {"id": "de_DE-ramona-low", "name": "Ramona (Weiblich)", "quality": "low"},
                            {"id": "de_DE-eva_k-x_low", "name": "Eva K (Weiblich)", "quality": "x_low"},
                        ],
                    }

                async def _run_piper_tts(text: str, voice: str, length_scale: float = 1.0) -> bytes:
                    """Generiert WAV-Audio via Piper TTS."""
                    import tempfile

                    # Voice-Modell-Pfad ermitteln
                    voices_dir = Path(config.jarvis_home) / "voices"
                    voices_dir.mkdir(exist_ok=True)
                    model_path = voices_dir / f"{voice}.onnx"

                    # Auto-Download wenn nicht vorhanden
                    if not model_path.exists():
                        await _download_piper_voice(voice, voices_dir)

                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        tmp_path = tmp.name

                    try:
                        # Write text to a temp file with explicit UTF-8 encoding
                        # to avoid Windows cp1252 stdin encoding issues with umlauts
                        import tempfile as _tts_tmpfile
                        with _tts_tmpfile.NamedTemporaryFile(
                            mode="w", suffix=".txt", delete=False, encoding="utf-8"
                        ) as txt_tmp:
                            txt_tmp.write(text)
                            txt_input_path = txt_tmp.name

                        cmd = [
                            sys.executable, "-m", "piper",
                            "--model", str(model_path),
                            "--output_file", tmp_path,
                            "--length-scale", str(length_scale),
                        ]
                        # Multi-speaker models (e.g. thorsten_emotional) need --speaker
                        model_json = model_path.with_suffix(".onnx.json")
                        if model_json.exists():
                            try:
                                import json as _mj
                                _model_cfg = _mj.loads(model_json.read_text(encoding="utf-8"))
                                _speaker_map = _model_cfg.get("speaker_id_map", {})
                                if _model_cfg.get("num_speakers", 1) > 1 and _speaker_map:
                                    # Prefer "neutral", fallback to first speaker
                                    _spk = "neutral" if "neutral" in _speaker_map else next(iter(_speaker_map))
                                    cmd.extend(["--speaker", str(_speaker_map[_spk])])
                            except Exception:
                                pass
                        # Read the UTF-8 text file as bytes for stdin
                        with open(txt_input_path, "rb") as _tts_in:
                            _tts_input_bytes = _tts_in.read()
                        proc = await asyncio.create_subprocess_exec(
                            *cmd,
                            stdin=asyncio.subprocess.PIPE,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                        )
                        stdout, stderr = await proc.communicate(input=_tts_input_bytes)

                        if proc.returncode != 0:
                            raise RuntimeError(f"Piper fehlgeschlagen: {stderr.decode()[:200]}")

                        with open(tmp_path, "rb") as f:
                            return f.read()
                    finally:
                        with contextlib.suppress(OSError):
                            os.unlink(tmp_path)
                        with contextlib.suppress(OSError, NameError):
                            os.unlink(txt_input_path)

                async def _download_piper_voice(voice: str, dest: Path) -> None:
                    """Lädt ein Piper-Voicemodell von HuggingFace herunter."""
                    import urllib.request

                    parts = voice.split("-")  # de_DE-pavoque-low
                    lang = parts[0]  # de_DE
                    name = parts[1]  # pavoque
                    quality = parts[2] if len(parts) > 2 else "low"
                    lang_short = lang.split("_")[0]  # de

                    base = f"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/{lang_short}/{lang}/{name}/{quality}"
                    onnx_url = f"{base}/{voice}.onnx?download=true"
                    json_url = f"{base}/{voice}.onnx.json?download=true"

                    log.info("downloading_piper_voice", voice=voice, url=onnx_url)

                    def _dl() -> None:
                        urllib.request.urlretrieve(onnx_url, str(dest / f"{voice}.onnx"))
                        urllib.request.urlretrieve(json_url, str(dest / f"{voice}.onnx.json"))

                    await asyncio.get_running_loop().run_in_executor(None, _dl)
                    log.info("piper_voice_downloaded", voice=voice)

                log.info("cc_tts_endpoint_registered")

                # Mount pre-built React UI at / (catch-all, MUSS als letztes)
                _ui_dist = Path(__file__).resolve().parent.parent.parent / "ui" / "dist"
                if _ui_dist.is_dir() and (_ui_dist / "index.html").exists():
                    from fastapi.staticfiles import StaticFiles
                    api_app.mount("/", StaticFiles(directory=str(_ui_dist), html=True), name="ui")
                    log.info("prebuilt_ui_mounted", path=str(_ui_dist))

                # TLS-Durchreichung
                uvi_kwargs: dict[str, Any] = {
                    "app": api_app,
                    "host": api_host,
                    "port": args.api_port,
                    "log_level": "warning",
                }
                if _ssl_cert and _ssl_key:
                    uvi_kwargs["ssl_certfile"] = _ssl_cert
                    uvi_kwargs["ssl_keyfile"] = _ssl_key

                uvi_config = uvicorn.Config(**uvi_kwargs)
                api_server = uvicorn.Server(uvi_config)
                asyncio.create_task(api_server.serve())
                log.info("control_center_api_started", host=api_host, port=args.api_port, tls=bool(_ssl_cert))
            except ImportError:
                log.warning("control_center_api_requires_fastapi_uvicorn")
            except Exception as exc:
                log.warning("control_center_api_failed", error=str(exc))

            # CLI-Channel registrieren und starten
            if config.channels.cli_enabled and not args.no_cli:
                cli = CliChannel(version=__version__)
                gateway.register_channel(cli)

            # Telegram-Channel (auto-detect: token in env → start)
            telegram_token = os.environ.get("JARVIS_TELEGRAM_TOKEN")
            if telegram_token:
                from jarvis.channels.telegram import TelegramChannel

                allowed = [int(u) for u in os.environ.get("JARVIS_TELEGRAM_ALLOWED_USERS", "").split(",") if u]
                _tg_use_webhook = config.channels.telegram_use_webhook
                _tg_webhook_url = config.channels.telegram_webhook_url
                _tg_webhook_port = config.channels.telegram_webhook_port
                _tg_webhook_host = config.channels.telegram_webhook_host
                gateway.register_channel(TelegramChannel(
                    token=telegram_token,
                    allowed_users=allowed,
                    session_store=_session_store,
                    use_webhook=_tg_use_webhook,
                    webhook_url=_tg_webhook_url,
                    webhook_port=_tg_webhook_port,
                    webhook_host=_tg_webhook_host,
                    ssl_certfile=_ssl_cert,
                    ssl_keyfile=_ssl_key,
                ))
                _tg_mode = "webhook" if (_tg_use_webhook and _tg_webhook_url) else "polling"
                log.info("telegram_channel_registered", allowed_users=len(allowed), mode=_tg_mode)

            # Slack-Channel (auto-detect: token + channel in env → start)
            slack_token = os.environ.get("JARVIS_SLACK_TOKEN")
            if slack_token:
                from jarvis.channels.slack import SlackChannel

                slack_app_token = os.environ.get("JARVIS_SLACK_APP_TOKEN", "")
                default_channel = config.channels.slack_default_channel or os.environ.get(
                    "JARVIS_SLACK_CHANNEL", ""
                )
                if default_channel:
                    gateway.register_channel(
                        SlackChannel(
                            token=slack_token,
                            app_token=slack_app_token,
                            default_channel=default_channel,
                        )
                    )
                else:
                    log.warning("slack_token_found_but_no_channel")

            # Discord-Channel (auto-detect: token + channel_id → start)
            discord_token = os.environ.get("JARVIS_DISCORD_TOKEN")
            if discord_token:
                from jarvis.channels.discord import DiscordChannel

                channel_id = config.channels.discord_channel_id or os.environ.get("JARVIS_DISCORD_CHANNEL_ID")
                try:
                    channel_id_int = int(channel_id) if channel_id else 0
                except Exception:
                    channel_id_int = 0
                if channel_id_int:
                    gateway.register_channel(
                        DiscordChannel(
                            token=discord_token,
                            channel_id=channel_id_int,
                            session_store=_session_store,
                        )
                    )
                else:
                    log.warning("discord_token_found_but_no_channel_id")

            # WhatsApp-Channel (auto-detect: token + phone_number_id → start)
            wa_token = os.environ.get("JARVIS_WHATSAPP_TOKEN")
            if wa_token:
                from jarvis.channels.whatsapp import WhatsAppChannel

                phone_number_id = (
                    config.channels.whatsapp_phone_number_id
                    or os.environ.get("JARVIS_WHATSAPP_PHONE_NUMBER_ID", "")
                )
                verify_token = (
                    config.channels.whatsapp_verify_token
                    or os.environ.get("JARVIS_WHATSAPP_VERIFY_TOKEN", "")
                )
                allowed = config.channels.whatsapp_allowed_numbers
                if phone_number_id:
                    gateway.register_channel(
                        WhatsAppChannel(
                            api_token=wa_token,
                            phone_number_id=phone_number_id,
                            verify_token=verify_token,
                            webhook_port=config.channels.whatsapp_webhook_port,
                            allowed_numbers=allowed,
                            ssl_certfile=_ssl_cert,
                            ssl_keyfile=_ssl_key,
                            session_store=_session_store,
                        )
                    )
                else:
                    log.warning("whatsapp_token_found_but_no_phone_number_id")

            # Signal-Channel (auto-detect: token + default_user → start)
            signal_token = os.environ.get("JARVIS_SIGNAL_TOKEN")
            if signal_token:
                from jarvis.channels.signal import SignalChannel

                default_user = config.channels.signal_default_user or os.environ.get(
                    "JARVIS_SIGNAL_DEFAULT_USER", ""
                )
                if default_user:
                    gateway.register_channel(
                        SignalChannel(token=signal_token, default_user=default_user)
                    )
                else:
                    log.warning("signal_token_found_but_no_default_user")

            # Matrix-Channel (auto-detect: token + homeserver + user_id → start)
            matrix_token = os.environ.get("JARVIS_MATRIX_TOKEN")
            if matrix_token:
                from jarvis.channels.matrix import MatrixChannel

                homeserver = (
                    os.environ.get("JARVIS_MATRIX_HOMESERVER")
                    or config.channels.matrix_homeserver
                )
                user_id = (
                    os.environ.get("JARVIS_MATRIX_USER_ID") or config.channels.matrix_user_id
                )
                if homeserver and user_id:
                    gateway.register_channel(
                        MatrixChannel(
                            access_token=matrix_token, homeserver=homeserver, user_id=user_id
                        )
                    )
                else:
                    log.warning("matrix_token_found_but_no_homeserver_or_user_id")

            # Teams-Channel (auto-detect: app_id + app_password → start)
            teams_app_id = os.environ.get("JARVIS_TEAMS_APP_ID", "")
            teams_app_pw = os.environ.get("JARVIS_TEAMS_TOKEN") or os.environ.get("JARVIS_TEAMS_APP_PASSWORD", "")
            if teams_app_id or teams_app_pw:
                from jarvis.channels.teams import TeamsChannel

                teams_host = os.environ.get("JARVIS_TEAMS_WEBHOOK_HOST", "127.0.0.1")
                teams_port = int(os.environ.get("JARVIS_TEAMS_WEBHOOK_PORT", "3978"))
                gateway.register_channel(
                    TeamsChannel(
                        app_id=teams_app_id,
                        app_password=teams_app_pw,
                        webhook_host=teams_host,
                        webhook_port=teams_port,
                        ssl_certfile=_ssl_cert,
                        ssl_keyfile=_ssl_key,
                        session_store=_session_store,
                    )
                )

            # iMessage-Channel
            if getattr(config.channels, "imessage_enabled", False):
                from jarvis.channels.imessage import IMessageChannel

                device_id = config.channels.imessage_device_id or os.environ.get(
                    "JARVIS_IMESSAGE_DEVICE_ID"
                )
                # iMessage hat keine Token; device_id ist optional
                gateway.register_channel(IMessageChannel(device_id=device_id))

            # Start Dashboard falls aktiviert
            if config.dashboard.enabled:
                try:
                    from jarvis.dashboard import Dashboard

                    dashboard = Dashboard(config, gateway)
                    await dashboard.start()
                except Exception:
                    log.warning("dashboard_failed_to_start")

            log.info("jarvis_ready", channels=list(gateway._channels.keys()))
            await gateway.start()

            # Headless-Modus: wenn keine interaktiven Channels laufen,
            # halte den Prozess am Leben für die API
            if args.no_cli and api_server and not api_server.should_exit:
                log.info("jarvis_headless_mode", port=args.api_port)
                while not api_server.should_exit:
                    await asyncio.sleep(1)

        except KeyboardInterrupt:
            log.info("jarvis_interrupted")
        finally:
            if api_server:
                api_server.should_exit = True
            await gateway.shutdown()
            log.info("jarvis_stopped")

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        log.info("jarvis_shutdown_by_user")


def _print_banner(
    config: Any,
    api_host: str = "127.0.0.1",
    api_port: int = 8741,
    lite: bool = False,
) -> None:
    """Print the startup banner to the console.

    This is intentional CLI output so we use print() rather than the
    logger.  Keeping it in a dedicated function makes the main flow
    cleaner and easier to test.
    """
    backend = getattr(config, "llm_backend_type", "ollama")
    scheme = "https" if config.security.ssl_certfile else "http"
    lite_tag = " [LITE]" if lite else ""
    print(f"\n{'=' * 60}")
    print(f"  COGNITHOR · Agent OS v{__version__}{lite_tag}")
    print(f"  Home:   {config.jarvis_home}")
    print(f"  API:    {scheme}://{api_host}:{api_port}")
    if backend == "ollama":
        print(f"  Ollama: {config.ollama.base_url}")
    elif backend == "lmstudio":
        print(f"  LM Studio: {config.lmstudio_base_url}")
    else:
        print(f"  Backend: {backend}")
    print(f"  Planner: {config.models.planner.name}")
    print(f"  Executor: {config.models.executor.name}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
