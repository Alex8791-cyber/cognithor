"""REST-API Channel: HTTP-Endpunkte für externe Integration.

Bietet eine vollständige REST-API für programmatischen Zugriff
auf Jarvis. Bearer-Token-Authentifizierung, JSON-basiert,
SSE-Streaming für Echtzeit-Antworten.

Endpunkte:
  POST /api/v1/message     → Nachricht senden
  GET  /api/v1/message/stream → SSE-Stream
  GET  /api/v1/sessions    → Sessions auflisten
  GET  /api/v1/health      → Health-Check
  GET  /api/v1/tools       → Verfügbare Tools

Bibel-Referenz: §9.3 (API Channel), §11.1 (Authentication)
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from jarvis.channels.base import Channel, MessageHandler
from jarvis.models import IncomingMessage, OutgoingMessage, PlannedAction
from jarvis.security.rate_limiter import RateLimiter
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

# ============================================================================
# API-Datenmodelle (Pydantic für FastAPI)
# ============================================================================


try:
    from pydantic import BaseModel, Field
except ImportError:
    BaseModel = object  # type: ignore[assignment, misc]
    Field = lambda **kw: None  # type: ignore[assignment]  # noqa: E731


class MessageRequest(BaseModel):
    """Eingehende Nachricht über die API."""

    text: str = Field(..., min_length=1, max_length=100_000)
    session_id: str | None = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MessageResponse(BaseModel):
    """API-Antwort auf eine Nachricht."""

    text: str
    session_id: str
    timestamp: str
    duration_ms: int
    tools_used: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Health-Check-Antwort."""

    status: str = "ok"
    version: str = "0.1.0"
    uptime_seconds: float = 0.0
    active_sessions: int = 0


class SessionInfo(BaseModel):
    """Informationen über eine Session."""

    session_id: str
    created_at: str
    message_count: int = 0
    last_activity: str = ""


class ApprovalRequest(BaseModel):
    """Approval-Anfrage für ORANGE-Aktionen."""

    session_id: str
    action_tool: str
    reason: str
    request_id: str


class ApprovalResponse(BaseModel):
    """Approval-Antwort vom Client."""

    request_id: str
    approved: bool


# ============================================================================
# API Channel
# ============================================================================


class APIChannel(Channel):
    """REST-API Channel via FastAPI. [B§9.3]

    Stellt HTTP-Endpunkte bereit für programmatischen Zugriff.
    Authentifizierung via Bearer-Token.
    """

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 8741,
        api_token: str | None = None,
        cors_origins: list[str] | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._api_token = api_token
        self._cors_origins = cors_origins or []
        self._handler: MessageHandler | None = None
        self._app: Any = None
        self._server: Any = None
        self._start_time = 0.0
        self._sessions: dict[str, SessionInfo] = {}
        self._pending_approvals: dict[str, asyncio.Future[bool]] = {}
        self._serve_task: asyncio.Task[None] | None = None
        self._rate_limiter = RateLimiter()

    @property
    def name(self) -> str:
        return "api"

    async def start(self, handler: MessageHandler) -> None:
        """Startet den FastAPI-Server."""
        self._handler = handler
        self._start_time = time.monotonic()
        self._app = self._create_app()
        log.info("api_channel_starting", host=self._host, port=self._port)

    async def stop(self) -> None:
        """Stoppt den API-Server."""
        if self._serve_task and not self._serve_task.done():
            self._serve_task.cancel()
        # Cancel pending approvals
        for future in self._pending_approvals.values():
            if not future.done():
                future.set_result(False)
        self._pending_approvals.clear()
        log.info("api_channel_stopped")

    async def send(self, message: OutgoingMessage) -> None:
        """API-Channel sendet nicht aktiv -- Antworten gehen via Response."""
        pass

    async def request_approval(
        self,
        session_id: str,
        action: PlannedAction,
        reason: str,
    ) -> bool:
        """Stellt Approval-Anfrage in Queue, wartet auf Client-Antwort."""
        request_id = str(uuid4())
        future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        self._pending_approvals[request_id] = future

        log.info(
            "approval_requested_via_api",
            session_id=session_id,
            tool=action.tool,
            request_id=request_id,
        )

        try:
            # Warte max 5 Minuten auf Approval
            result = await asyncio.wait_for(future, timeout=300)
            return result
        except TimeoutError:
            log.warning("approval_timeout", request_id=request_id)
            return False
        finally:
            self._pending_approvals.pop(request_id, None)

    async def send_streaming_token(self, session_id: str, token: str) -> None:
        """Streaming-Tokens werden über SSE gesendet (in der Route)."""
        pass

    def _create_app(self) -> Any:
        """Erstellt die FastAPI-Applikation mit allen Routen."""
        try:
            from fastapi import Depends, FastAPI, HTTPException, Request
            from fastapi.middleware.cors import CORSMiddleware
            from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
            # Request muss im Modul-Namespace verfügbar sein, damit FastAPI
            # die String-Annotation (PEP 563) via get_type_hints() auflösen kann
            globals()["Request"] = Request
        except ImportError as exc:
            log.error("fastapi_not_installed", error=str(exc))
            raise ImportError(
                "FastAPI ist für den API-Channel erforderlich: pip install fastapi uvicorn"
            ) from exc

        app = FastAPI(
            title="Jarvis Agent OS API",
            version="0.1.0",
            description="REST-API für Jarvis Agent OS",
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
                raise HTTPException(status_code=503, detail="API-Token nicht konfiguriert")
            if not credentials or credentials.credentials != self._api_token:
                raise HTTPException(status_code=401, detail="Ungültiger Token")

        @app.get("/api/v1/health", response_model=HealthResponse)
        async def health() -> HealthResponse:
            return HealthResponse(
                uptime_seconds=time.monotonic() - self._start_time,
                active_sessions=len(self._sessions),
            )

        @app.post(
            "/api/v1/message",
            response_model=MessageResponse,
            dependencies=[Depends(verify_token)],
        )
        async def send_message(req: MessageRequest, request: Request) -> MessageResponse:  # noqa: ARG001
            # Rate limiting -- per Client-IP
            client_ip = request.client.host if request.client else "unknown"
            rate_key = f"api_{client_ip}"
            if not await self._rate_limiter.check(rate_key):
                raise HTTPException(status_code=429, detail="Too Many Requests")
            if not self._handler:
                raise HTTPException(status_code=503, detail="Handler nicht bereit")

            session_id = req.session_id or str(uuid4())
            start = time.monotonic()

            incoming = IncomingMessage(
                text=req.text,
                channel="api",
                session_id=session_id,
                user_id="api_user",
                metadata=req.metadata,
            )

            response = await self._handler(incoming)
            duration_ms = int((time.monotonic() - start) * 1000)

            # Session tracken
            now = datetime.now(UTC).isoformat()
            if session_id not in self._sessions:
                self._sessions[session_id] = SessionInfo(
                    session_id=session_id,
                    created_at=now,
                    message_count=0,
                    last_activity=now,
                )
            info = self._sessions[session_id]
            info.message_count += 1
            info.last_activity = now

            return MessageResponse(
                text=response.text,
                session_id=session_id,
                timestamp=now,
                duration_ms=duration_ms,
            )

        @app.get(
            "/api/v1/sessions",
            response_model=list[SessionInfo],
            dependencies=[Depends(verify_token)],
        )
        async def list_sessions() -> list[SessionInfo]:
            return list(self._sessions.values())

        @app.get(
            "/api/v1/approvals/pending",
            dependencies=[Depends(verify_token)],
        )
        async def pending_approvals() -> list[str]:
            return list(self._pending_approvals.keys())

        @app.post(
            "/api/v1/approvals/respond",
            dependencies=[Depends(verify_token)],
        )
        async def respond_approval(resp: ApprovalResponse) -> dict[str, str]:
            future = self._pending_approvals.get(resp.request_id)
            if not future:
                raise HTTPException(status_code=404, detail="Approval-Anfrage nicht gefunden")
            if not future.done():
                future.set_result(resp.approved)
            return {"status": "processed"}

        return app

    @property
    def app(self) -> Any:
        """FastAPI-App-Instanz (für Tests und Embedding in Gateway)."""
        if self._app is None:
            self._app = self._create_app()
        return self._app
