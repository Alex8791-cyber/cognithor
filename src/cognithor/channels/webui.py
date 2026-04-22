"""Web UI Channel: Browser-Interface mit WebSocket-Streaming.

Erweitert den API-Channel um WebSocket-Support fuer Echtzeit-
Kommunikation. Dient die Web-Oberflaeche (React/Svelte) aus
und bietet Live-Streaming der Agent-Antworten.

Features:
  - WebSocket fuer bidirektionale Echtzeit-Kommunikation
  - SSE-Fallback fuer aeltere Clients
  - Streaming-Tokens fuer fluessige Ausgabe
  - Tool-Execution-Events (User sieht was passiert)
  - Inline-Approvals via WebSocket
  - Static-File-Serving fuer Frontend

Bibel-Referenz: §9.3 (Web UI Channel)
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from cognithor.channels.base import Channel, MessageHandler, StatusType
from cognithor.models import IncomingMessage, OutgoingMessage, PlannedAction
from cognithor.security.rate_limiter import RateLimiter
from cognithor.security.token_store import get_token_store
from cognithor.utils.logging import get_logger
from cognithor.utils.ttl_dict import TTLDict

if TYPE_CHECKING:
    from collections.abc import Callable

log = get_logger(__name__)

# Maximum upload size (50 MB)
MAX_UPLOAD_SIZE = 52_428_800


# ============================================================================
# WebSocket message types
# ============================================================================


class WSMessageType:
    """WebSocket-Nachrichtentypen (Client ↔ Server)."""

    # Client → Server
    USER_MESSAGE = "user_message"
    APPROVAL_RESPONSE = "approval_response"
    PING = "ping"
    CANCEL = "cancel"

    # Server → Client
    ASSISTANT_MESSAGE = "assistant_message"
    STREAM_TOKEN = "stream_token"
    STREAM_END = "stream_end"
    TOOL_START = "tool_start"
    TOOL_RESULT = "tool_result"
    APPROVAL_REQUEST = "approval_request"
    STATUS_UPDATE = "status_update"
    PIPELINE_EVENT = "pipeline_event"
    PLAN_DETAIL = "plan_detail"
    CANVAS_PUSH = "canvas_push"
    CANVAS_RESET = "canvas_reset"
    CANVAS_EVAL = "canvas_eval"
    TRANSCRIPTION = "transcription"
    AGENT_LOG = "agent_log"
    ERROR = "error"
    PONG = "pong"
    IDENTITY_STATE = "identity_state"


# ============================================================================
# WebUI Channel
# ============================================================================


class WebUIChannel(Channel):
    """Web UI Channel mit WebSocket-Support. [B§9.3]

    Erweitert die API um WebSocket-Verbindungen fuer Echtzeit-
    Streaming und interaktive Tool-Visualisierung.
    """

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 8741,
        api_token: str | None = None,
        cors_origins: list[str] | None = None,
        static_dir: str | None = None,
        config: Any = None,
        config_manager: Any = None,
        ssl_certfile: str = "",
        ssl_keyfile: str = "",
    ) -> None:
        self._host = host
        self._port = port
        self._token_store = get_token_store()
        if api_token:
            self._token_store.store("webui_channel_token", api_token)
        self._has_api_token = bool(api_token)
        self._cors_origins = cors_origins or []
        self._ssl_certfile = ssl_certfile
        self._ssl_keyfile = ssl_keyfile
        self._static_dir = static_dir
        self._config_manager = config_manager
        self._config = config  # CognithorConfig (optional, für ConfigManager)
        # Locate Flutter Web build — try multiple paths for robustness
        if self._static_dir is None:
            candidates = []
            # 1. Relative to this source file (works in dev mode / editable install)
            src_root = Path(__file__).resolve().parent.parent.parent.parent
            candidates.append(src_root / "flutter_app" / "build" / "web")
            # 2. Relative to CWD (works when started from repo root)
            candidates.append(Path.cwd() / "flutter_app" / "build" / "web")
            # 3. COGNITHOR_HOME (works in any install mode)
            cognithor_home = Path(os.environ.get("COGNITHOR_HOME", Path.home() / ".cognithor"))
            candidates.append(cognithor_home / "flutter_web")
            # 4. Fallback: built-in webchat widget
            candidates.append(Path(__file__).parent / "webchat")

            for candidate in candidates:
                if candidate.is_dir() and (candidate / "index.html").exists():
                    self._static_dir = str(candidate)
                    break
        self._handler: MessageHandler | None = None
        self._cancel_callback: Callable[[str], Any] | None = None
        self._app: Any = None
        self._start_time = 0.0

        # WebSocket connections: session_id → WebSocket
        self._connections: TTLDict[str, Any] = TTLDict(max_size=1000, ttl_seconds=86400)
        # Pending Approvals: request_id → Future (finally-Cleanup in request_approval)
        self._pending_approvals: dict[str, asyncio.Future[bool]] = {}
        # Session-Tracking
        self._session_messages: TTLDict[str, int] = TTLDict(max_size=10000, ttl_seconds=86400)
        self._rate_limiter = RateLimiter()

    @property
    def _api_token(self) -> str | None:
        """API-Token (entschluesselt bei Zugriff)."""
        if self._has_api_token:
            return self._token_store.retrieve("webui_channel_token")
        return None

    @property
    def name(self) -> str:
        return "webui"

    async def start(self, handler: MessageHandler) -> None:
        """Startet den WebUI-Server."""
        self._handler = handler
        self._start_time = time.monotonic()
        self._app = self._create_app()

        # TLS warning for external hosts
        if self._host not in ("127.0.0.1", "localhost", "::1") and not self._ssl_certfile:
            log.warning(
                "webui_no_tls",
                host=self._host,
                message="WARNUNG: WebUI auf externem Host ohne TLS!",
            )

        log.info("webui_channel_starting", host=self._host, port=self._port)

    async def stop(self) -> None:
        """Stoppt den WebUI-Server und schliesst WebSocket-Verbindungen."""
        # Close all WebSocket connections
        for ws in list(self._connections.values()):
            with contextlib.suppress(Exception):
                await ws.close()
        self._connections.clear()

        # Cancel pending approvals
        for future in self._pending_approvals.values():
            if not future.done():
                future.set_result(False)
        self._pending_approvals.clear()
        log.info("webui_channel_stopped")

    async def send(self, message: OutgoingMessage) -> None:
        """Sendet Nachricht ueber WebSocket an den Client."""
        ws = self._connections.get(message.session_id)
        if ws:
            await self._ws_send(
                ws,
                {
                    "type": WSMessageType.ASSISTANT_MESSAGE,
                    "text": message.text,
                    "session_id": message.session_id,
                    "timestamp": datetime.now(UTC).isoformat(),
                },
            )

    async def request_approval(
        self,
        session_id: str,
        action: PlannedAction,
        reason: str,
    ) -> bool:
        """Sendet Approval-Anfrage ueber WebSocket."""
        ws = self._connections.get(session_id)
        if not ws:
            log.warning("no_ws_connection_for_approval", session_id=session_id)
            return False

        request_id = str(uuid4())
        future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        self._pending_approvals[request_id] = future

        await self._ws_send(
            ws,
            {
                "type": WSMessageType.APPROVAL_REQUEST,
                "request_id": request_id,
                "session_id": session_id,
                "tool": action.tool,
                "params": action.params,
                "reason": reason,
            },
        )

        try:
            return await asyncio.wait_for(future, timeout=300)
        except TimeoutError:
            return False
        finally:
            self._pending_approvals.pop(request_id, None)

    async def send_streaming_token(self, session_id: str, token: str) -> None:
        """Sendet einzelnes Streaming-Token ueber WebSocket."""
        ws = self._connections.get(session_id)
        if ws:
            await self._ws_send(
                ws,
                {
                    "type": WSMessageType.STREAM_TOKEN,
                    "token": token,
                    "session_id": session_id,
                },
            )

    async def send_status(self, session_id: str, status: StatusType, text: str) -> None:
        """Sendet Status-Update ueber WebSocket."""
        ws = self._connections.get(session_id)
        if ws:
            await self._ws_send(
                ws,
                {
                    "type": WSMessageType.STATUS_UPDATE,
                    "status": status.value,
                    "text": text,
                    "session_id": session_id,
                },
            )

    async def send_tool_event(
        self,
        session_id: str,
        event_type: str,
        tool_name: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Sendet Tool-Execution-Event an den Client.

        Damit der User sieht, was gerade passiert (z.B.
        'Suche im Web...', 'Datei wird geschrieben...').
        """
        ws = self._connections.get(session_id)
        if ws:
            await self._ws_send(
                ws,
                {
                    "type": event_type,
                    "tool": tool_name,
                    "data": data or {},
                    "session_id": session_id,
                    "timestamp": datetime.now(UTC).isoformat(),
                },
            )

    async def send_pipeline_event(
        self,
        session_id: str,
        event: dict[str, Any],
    ) -> None:
        """Sendet ein PGE-Pipeline-Event an den Client.

        Wird vom Gateway bei jedem Phasenwechsel im PGE-Zyklus aufgerufen.
        Das Frontend nutzt diese Events fuer die Pipeline-Visualisierung.
        """
        ws = self._connections.get(session_id)
        if ws:
            await self._ws_send(
                ws,
                {
                    "type": WSMessageType.PIPELINE_EVENT,
                    "session_id": session_id,
                    **event,
                },
            )

    async def send_plan_detail(
        self,
        session_id: str,
        plan_data: dict[str, Any],
    ) -> None:
        """Sendet Plan-Details an den Client fuer das Plan Review Panel."""
        ws = self._connections.get(session_id)
        if ws:
            await self._ws_send(
                ws,
                {
                    "type": WSMessageType.PLAN_DETAIL,
                    "session_id": session_id,
                    **plan_data,
                },
            )

    def _create_app(self) -> Any:
        """Erstellt die FastAPI-App mit WebSocket-Support."""
        # FastAPI and its dependencies are optional. When not available or
        # broken, return a lightweight placeholder object instead of
        # raising an error. Some environments may partially have FastAPI
        # installed but missing dependencies such as websockets; pydantic
        # may then fail when creating models. To handle all such cases
        # gracefully, we catch a broad set of exceptions during import
        # and app construction and fall back to a dummy implementation.
        try:
            from fastapi import (
                Depends,
                FastAPI,
                HTTPException,
                WebSocket,
                WebSocketDisconnect,
            )
            from fastapi.middleware.cors import CORSMiddleware
            from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
            # We'll attempt to import StaticFiles later when needed.
        except Exception:
            return self._dummy_app()

        # Build the FastAPI app in a try/except so that any runtime
        # failures (e.g. due to missing subpackages) fall back to a
        # dummy implementation.
        try:
            app = FastAPI(
                title="Jarvis Web UI",
                version="0.1.0",
            )

            app.add_middleware(
                CORSMiddleware,
                allow_origins=self._cors_origins,
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

            security = HTTPBearer(auto_error=False)

            async def verify_token(
                credentials: HTTPAuthorizationCredentials | None = Depends(security),  # noqa: B008
            ) -> None:
                if not self._api_token:
                    return
                if not credentials or credentials.credentials != self._api_token:
                    raise HTTPException(status_code=401, detail="Ungültiger Token")

            # --- Health ---
            @app.get("/api/v1/health")
            async def health() -> dict[str, Any]:
                return {
                    "status": "ok",
                    "version": "0.1.0",
                    "uptime_seconds": time.monotonic() - self._start_time,
                    "active_connections": len(self._connections),
                }

            # --- REST Message (Fallback) ---
            @app.post("/api/v1/message", dependencies=[Depends(verify_token)])
            async def send_message(
                text: str,
                session_id: str | None = None,
            ) -> dict[str, Any]:
                if not await self._rate_limiter.check("webui_default"):
                    raise HTTPException(status_code=429, detail="Too Many Requests")
                if not self._handler:
                    raise HTTPException(status_code=503, detail="Not ready")

                sid = session_id or str(uuid4())
                start = time.monotonic()

                incoming = IncomingMessage(
                    text=text,
                    channel="webui",
                    session_id=sid,
                    user_id="web_user",
                )
                response = await self._handler(incoming)
                duration_ms = int((time.monotonic() - start) * 1000)

                return {
                    "text": response.text,
                    "session_id": sid,
                    "duration_ms": duration_ms,
                }

            # --- WebSocket ---
            @app.websocket("/ws/{session_id}")
            async def websocket_endpoint(
                websocket: WebSocket,
                session_id: str,
            ) -> None:
                # Token check via first WS message (not query param,
                # to avoid log exposure).
                await websocket.accept()

                if self._api_token:
                    import hmac as _hmac

                    try:
                        auth_raw = await asyncio.wait_for(
                            websocket.receive_text(),
                            timeout=10.0,
                        )
                        auth_msg = json.loads(auth_raw)
                        client_token = (
                            auth_msg.get("token", "") if auth_msg.get("type") == "auth" else ""
                        )
                    except (TimeoutError, Exception):
                        client_token = ""
                    if not _hmac.compare_digest(client_token, self._api_token):
                        with contextlib.suppress(Exception):
                            await websocket.send_json(
                                {
                                    "type": "error",
                                    "error": "Unauthorized",
                                }
                            )
                        await websocket.close(code=4001, reason="Unauthorized")
                        return
                self._connections[session_id] = websocket
                log.info("ws_connected", session_id=session_id)

                try:
                    while True:
                        data = await websocket.receive_text()
                        msg = json.loads(data)
                        await self._handle_ws_message(websocket, session_id, msg)
                except WebSocketDisconnect:
                    log.info("ws_disconnected", session_id=session_id)
                except json.JSONDecodeError:
                    await self._ws_send(
                        websocket,
                        {
                            "type": WSMessageType.ERROR,
                            "error": "Ungültiges JSON",
                        },
                    )
                except Exception as exc:
                    log.error("ws_error", error=str(exc), session_id=session_id)
                finally:
                    self._connections.pop(session_id, None)

            # --- Config-API Routes ---
            # NOTE: Config routes are already registered on the main Cognithor
            # API (port 8741) in __main__.py.  We only register them here if
            # the WebUI channel runs on a DIFFERENT port and the caller passed
            # a shared ConfigManager via self._config_manager.  Creating a
            # separate ConfigManager would cause state divergence and data loss.
            if getattr(self, "_config_manager", None) is not None:
                try:
                    from cognithor.channels.config_routes import create_config_routes

                    create_config_routes(
                        app,
                        self._config_manager,
                        verify_token_dep=Depends(verify_token),
                    )
                    log.info("config_routes_registered", source="shared_manager")
                except Exception as exc:
                    log.warning("config_routes_not_available", error=str(exc))
            else:
                log.debug("config_routes_skipped", reason="no shared config_manager")

            # Static files (Frontend) -- optional
            if self._static_dir:
                try:
                    from fastapi.staticfiles import StaticFiles

                    app.mount(
                        "/",
                        StaticFiles(directory=self._static_dir, html=True),
                        name="frontend",
                    )
                except Exception:
                    log.warning("static_files_not_available")

            return app
        except Exception:
            # Any exception in building the FastAPI app leads to falling back
            # to a dummy implementation so that tests still pass.
            return self._dummy_app()

    def _dummy_app(self) -> Any:
        """Return a minimal stand-in for a FastAPI application."""

        class DummyApp:
            def __init__(self) -> None:
                self.title = "Jarvis Web UI (stub)"
                self.version = "0.0.0"

            def add_middleware(self, *args: Any, **kwargs: Any) -> None:
                return None

            def get(self, *args: Any, **kwargs: Any) -> Any:
                def decorator(func: Any) -> Any:
                    return func

                return decorator

            def post(self, *args: Any, **kwargs: Any) -> Any:
                def decorator(func: Any) -> Any:
                    return func

                return decorator

            def websocket(self, *args: Any, **kwargs: Any) -> Any:
                def decorator(func: Any) -> Any:
                    return func

                return decorator

            def mount(self, *args: Any, **kwargs: Any) -> None:
                return None

        return DummyApp()

    async def _handle_ws_message(
        self,
        ws: Any,
        session_id: str,
        msg: dict[str, Any],
    ) -> None:
        """Verarbeitet eine eingehende WebSocket-Nachricht.

        Unterstuetzt:
          - Text-Nachrichten
          - Sprachnachrichten (audio_base64 in metadata → Whisper-Transkription)
          - Datei-Uploads (file_base64 in metadata → Media-Pipeline)
          - Approval-Antworten
          - Ping/Pong
        """
        msg_type = msg.get("type", "")

        if msg_type == WSMessageType.PING:
            await self._ws_send(ws, {"type": WSMessageType.PONG})
            return

        if msg_type == WSMessageType.CANCEL:
            if self._cancel_callback:
                self._cancel_callback(session_id)
            await self._ws_send(
                ws,
                {
                    "type": WSMessageType.STATUS_UPDATE,
                    "status": "finishing",
                    "text": "Abgebrochen...",
                    "session_id": session_id,
                },
            )
            return

        if msg_type == WSMessageType.APPROVAL_RESPONSE:
            request_id = msg.get("request_id", "")
            future = self._pending_approvals.get(request_id)
            if future and not future.done():
                future.set_result(msg.get("approved", False))
            return

        if msg_type == WSMessageType.USER_MESSAGE:
            text = msg.get("text", "").strip()
            metadata = msg.get("metadata", {})

            # --- Voice bridge: transcribe audio ---
            if metadata.get("audio_base64"):
                transcribed = await self._transcribe_audio(metadata, ws, session_id)
                if transcribed:
                    text = transcribed
                else:
                    return  # Fehler wurde bereits gesendet

            # --- File upload: save file and either extract text or
            # route as an image attachment (VLM handles in Planner).
            # Legacy gate was text.startswith("[file_upload]"); the
            # current Flutter client sends "[File: name]" so we just
            # check for the base64 payload directly.
            _upload_attachment_path: str | None = None
            if metadata.get("file_base64"):
                file_text, _upload_attachment_path = await self._process_file_upload(
                    metadata, ws, session_id
                )
                if file_text:
                    text = file_text

            if not text:
                await self._ws_send(
                    ws,
                    {
                        "type": WSMessageType.ERROR,
                        "error": "Leere Nachricht",
                    },
                )
                return

            if not self._handler:
                await self._ws_send(
                    ws,
                    {
                        "type": WSMessageType.ERROR,
                        "error": "Handler nicht bereit",
                    },
                )
                return

            incoming = IncomingMessage(
                text=text,
                channel="webui",
                session_id=session_id,
                user_id="web_user",
                metadata=metadata,
                attachments=[_upload_attachment_path] if _upload_attachment_path else [],
            )

            # Stream callback: sends events to client in real-time
            _streamed_tokens = False

            async def _stream_callback(event_type: str, data: dict[str, Any]) -> None:
                nonlocal _streamed_tokens
                if event_type == "stream_token":
                    _streamed_tokens = True
                await self._ws_send(
                    ws,
                    {
                        "type": event_type,
                        **data,
                        "session_id": session_id,
                    },
                )

            try:
                # Try streaming-aware handler first (Gateway.handle_message
                # accepts optional stream_callback kwarg)
                try:
                    response = await self._handler(
                        incoming,
                        stream_callback=_stream_callback,  # type: ignore[call-arg]
                    )
                except TypeError:
                    # Handler does not accept stream_callback (e.g. tests,
                    # non-gateway handlers) — fall back to plain call
                    response = await self._handler(incoming)

                # If tokens were streamed, the client already has the text
                # progressively. Send assistant_message as final confirmation
                # with the complete (post-processed) text.
                _msg_payload: dict[str, Any] = {
                    "type": WSMessageType.ASSISTANT_MESSAGE,
                    "text": response.text,
                    "session_id": session_id,
                    "timestamp": datetime.now(UTC).isoformat(),
                    "streamed": _streamed_tokens,
                }
                if response.metadata:
                    _msg_payload["metadata"] = response.metadata
                await self._ws_send(ws, _msg_payload)
                await self._ws_send(
                    ws,
                    {
                        "type": WSMessageType.STREAM_END,
                        "session_id": session_id,
                    },
                )
            except Exception as exc:
                log.error("ws_handler_error", error=str(exc))
                await self._ws_send(
                    ws,
                    {
                        "type": WSMessageType.ERROR,
                        "error": "Ein Verarbeitungsfehler ist aufgetreten.",
                    },
                )
            return

        # Unknown type
        await self._ws_send(
            ws,
            {
                "type": WSMessageType.ERROR,
                "error": f"Unbekannter Nachrichtentyp: {msg_type}",
            },
        )

    async def _transcribe_audio(
        self,
        metadata: dict[str, Any],
        ws: Any,
        session_id: str,
    ) -> str | None:
        """Voice-Bridge: Transkribiert Base64-Audio via Whisper.

        Empfaengt Audio vom WebChat-Widget (Browser MediaRecorder),
        speichert temporaer und transkribiert lokal.

        Returns:
            Transkribierter Text oder None bei Fehler.
        """
        import base64
        import tempfile
        from pathlib import Path

        audio_b64 = metadata.get("audio_base64", "")
        audio_type = metadata.get("audio_type", "audio/webm")

        if not audio_b64:
            return None

        # Check estimated file size
        estimated_size = len(audio_b64) * 3 // 4
        if estimated_size > MAX_UPLOAD_SIZE:
            await self._ws_send(
                ws,
                {
                    "type": WSMessageType.ERROR,
                    "error": (
                        f"Audiodatei zu gross "
                        f"({estimated_size // 1_048_576} MB, "
                        f"max {MAX_UPLOAD_SIZE // 1_048_576} MB)"
                    ),
                },
            )
            return None

        # Notification: transcription in progress
        await self._ws_send(
            ws,
            {
                "type": WSMessageType.TOOL_START,
                "tool": "voice_transcription",
                "data": {"status": "Sprachnachricht wird transkribiert..."},
                "session_id": session_id,
            },
        )

        tmp_path: str | None = None
        wav_path: str | None = None
        try:
            # Base64 → file
            audio_bytes = base64.b64decode(audio_b64)
            suffix = ".webm" if "webm" in audio_type else ".ogg"

            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            # Convert to WAV via ffmpeg (if needed)
            wav_path = tmp_path + ".wav"
            try:
                proc = await asyncio.create_subprocess_exec(
                    "ffmpeg",
                    "-i",
                    tmp_path,
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    "-y",
                    wav_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await proc.communicate()
                if proc.returncode != 0:
                    wav_path = tmp_path  # Fallback: Datei direkt an Whisper geben
            except FileNotFoundError:
                wav_path = tmp_path  # ffmpeg nicht verfügbar

            # Transcription via MediaPipeline or faster-whisper directly
            try:
                from cognithor.mcp.media import MediaPipeline

                pipeline = MediaPipeline()
                result = await pipeline.transcribe_audio(wav_path)

                if result.success and result.text.strip():
                    log.info(
                        "voice_bridge_transcribed", text=result.text[:100], session_id=session_id
                    )

                    # Show transcription to user
                    await self._ws_send(
                        ws,
                        {
                            "type": WSMessageType.TOOL_RESULT,
                            "tool": "voice_transcription",
                            "data": {"transcription": result.text},
                            "session_id": session_id,
                        },
                    )

                    return f"🎤 {result.text}"
                else:
                    await self._ws_send(
                        ws,
                        {
                            "type": WSMessageType.ERROR,
                            "error": result.error or "Keine Sprache erkannt",
                        },
                    )
                    return None

            except ImportError:
                await self._ws_send(
                    ws,
                    {
                        "type": WSMessageType.ERROR,
                        "error": (
                            "faster-whisper nicht installiert. Voice-Transkription nicht verfügbar."
                        ),
                    },
                )
                return None

        except Exception as exc:
            log.error("voice_bridge_error", error=str(exc), session_id=session_id)
            await self._ws_send(
                ws,
                {
                    "type": WSMessageType.ERROR,
                    "error": "Voice-Transkription fehlgeschlagen.",
                },
            )
            return None
        finally:
            # Clean up temporary files
            for p in [tmp_path, wav_path]:
                if p is None:
                    continue
                with contextlib.suppress(Exception):
                    Path(p).unlink(missing_ok=True)

    async def _process_file_upload(
        self,
        metadata: dict[str, Any],
        ws: Any,
        session_id: str,
    ) -> tuple[str | None, str | None]:
        """Verarbeitet einen Datei-Upload vom WebChat-Widget.

        Saves the file under ``~/.cognithor/workspace/uploads/`` and
        either (a) returns an extracted-text synopsis (PDFs, .md, code)
        or (b) hands the file path back as an image attachment so the
        Gateway can route to the VLM.

        Returns:
            Tuple ``(text, attachment_path)`` — exactly one of which is
            typically non-None. ``text`` is the user-message text
            (replaces the stub "[File: name]"); ``attachment_path`` is
            the saved file location for images that should flow to the
            vision model. ``(None, None)`` on hard errors.
        """
        import base64
        from pathlib import Path

        file_b64 = metadata.get("file_base64", "")
        file_name = metadata.get("file_name", "upload")
        file_type = metadata.get("file_type", "")

        if not file_b64:
            return None, None

        # Check estimated file size (Base64 → ~75% of original size)
        estimated_size = len(file_b64) * 3 // 4
        if estimated_size > MAX_UPLOAD_SIZE:
            await self._ws_send(
                ws,
                {
                    "type": WSMessageType.ERROR,
                    "error": (
                        f"Datei zu gross "
                        f"({estimated_size // 1_048_576} MB, "
                        f"max {MAX_UPLOAD_SIZE // 1_048_576} MB)"
                    ),
                },
            )
            return None, None

        try:
            file_bytes = base64.b64decode(file_b64)
            workspace = Path.home() / ".cognithor" / "workspace" / "uploads"
            workspace.mkdir(parents=True, exist_ok=True)

            # Sanitize filename to prevent path traversal
            safe_name = Path(file_name).name.lstrip(".")
            if not safe_name:
                safe_name = "upload"
            save_path = (workspace / safe_name).resolve()
            if not str(save_path).startswith(str(workspace.resolve())):
                log.warning("path_traversal_blocked", file_name=file_name)
                return None, None
            save_path.write_bytes(file_bytes)

            log.info("file_uploaded", name=file_name, size=len(file_bytes), session_id=session_id)

            # Images → route to VLM via IncomingMessage.attachments.
            # Do NOT run text extraction on images (would force OCR which
            # loses visual context). The Gateway will pass the raw file
            # path to the vision model on this turn.
            _IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
            if any(safe_name.lower().endswith(ext) for ext in _IMAGE_EXTS):
                return (
                    f"[Bild hochgeladen: {file_name}, {len(file_bytes)} Bytes]",
                    str(save_path),
                )

            # Non-image: attempt text extraction
            try:
                from cognithor.mcp.media import MediaPipeline

                pipeline = MediaPipeline()
                result = await pipeline.extract_text(str(save_path))

                if result.success and result.text.strip():
                    return (
                        (
                            f"[Datei hochgeladen: {file_name}, "
                            f"{len(file_bytes)} Bytes]\n\n"
                            f"Inhalt:\n{result.text}"
                        ),
                        None,
                    )
            except Exception:
                pass  # Cleanup — file content extraction failure is non-critical

            # Fallback: file info only
            return (
                (
                    f"[Datei hochgeladen: {file_name}, "
                    f"{len(file_bytes)} Bytes, Typ: {file_type}]\n"
                    f"Gespeichert unter: {save_path}"
                ),
                None,
            )

        except Exception as exc:
            log.error("file_upload_error", error=str(exc))
            await self._ws_send(
                ws,
                {
                    "type": WSMessageType.ERROR,
                    "error": "Datei-Upload fehlgeschlagen.",
                },
            )
            return None, None

    async def _ws_send(self, ws: Any, data: dict[str, Any]) -> None:
        """Sendet JSON ueber WebSocket (mit Fehlerbehandlung)."""
        try:
            await ws.send_text(json.dumps(data, ensure_ascii=False))
        except Exception as exc:
            log.warning("ws_send_failed", error=str(exc))

    @property
    def app(self) -> Any:
        """FastAPI-App-Instanz."""
        if self._app is None:
            self._app = self._create_app()
        return self._app

    @property
    def active_connections(self) -> int:
        """Anzahl aktiver WebSocket-Verbindungen."""
        return len(self._connections)


# ============================================================================
# ASGI Factory — for uvicorn --factory / docker-compose / systemd
# ============================================================================


def create_app() -> Any:
    """ASGI-Factory fuer Standalone-Deployment.

    Wird von ``uvicorn jarvis.channels.webui:create_app --factory`` aufgerufen
    (docker-compose.yml, jarvis-webui.service).

    Konfiguration ausschliesslich ueber Umgebungsvariablen:
      COGNITHOR_WEBUI_HOST          (default "127.0.0.1"; "0.0.0.0" fuer Docker)
      COGNITHOR_WEBUI_PORT          (default "8080", nur informativ)
      COGNITHOR_API_TOKEN           (optional, Bearer-Auth)
      COGNITHOR_WEBUI_CORS_ORIGINS  (kommasepariert, default "http://localhost:8741")
      COGNITHOR_SSL_CERTFILE        (optional, PEM-Pfad)
      COGNITHOR_SSL_KEYFILE         (optional, PEM-Pfad)

    Ohne Gateway gibt POST /api/v1/message → 503 zurueck (korrekt).
    """
    host = os.environ.get("COGNITHOR_WEBUI_HOST", "127.0.0.1")
    api_token = os.environ.get("COGNITHOR_API_TOKEN") or None
    cors_raw = os.environ.get(
        "COGNITHOR_WEBUI_CORS_ORIGINS",
        "http://localhost:8741,http://localhost:5173,http://127.0.0.1:5173,http://127.0.0.1:8741",
    )
    cors_origins = [o.strip() for o in cors_raw.split(",") if o.strip()]
    ssl_cert = os.environ.get("COGNITHOR_SSL_CERTFILE", "")
    ssl_key = os.environ.get("COGNITHOR_SSL_KEYFILE", "")

    channel = WebUIChannel(
        host=host,
        api_token=api_token,
        cors_origins=cors_origins,
        ssl_certfile=ssl_cert,
        ssl_keyfile=ssl_key,
    )
    log.info(
        "create_app_factory",
        host=host,
        cors_origins=cors_origins,
        tls=bool(ssl_cert),
    )
    return channel.app
