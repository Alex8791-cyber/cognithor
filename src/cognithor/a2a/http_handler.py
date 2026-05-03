"""A2A HTTP Handler -- FastAPI routes for A2A Protocol RC v1.0.

Provides the HTTP endpoints:
  GET  /.well-known/agent.json  -- Agent Card Discovery
  POST /a2a                     -- JSON-RPC 2.0 Dispatch
  POST /a2a/stream              -- SSE Streaming (message/stream)
  GET  /a2a/health              -- Health Check

OPTIONAL: Only registered when the A2A adapter is active.
Import-safe: FastAPI is an optional dependency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from cognithor.a2a.types import A2A_CONTENT_TYPE, A2A_PROTOCOL_VERSION, A2A_VERSION_HEADER
from cognithor.security.rate_limiter import RateLimiter
from cognithor.utils.logging import get_logger

if TYPE_CHECKING:
    from starlette.requests import Request

log = get_logger(__name__)


class A2AHTTPHandler:
    """HTTP transport layer for the A2A adapter.

    Can either run as a standalone server (aiohttp/FastAPI)
    or hook into the existing gateway FastAPI app.
    """

    def __init__(self, adapter: Any) -> None:
        self.adapter = adapter
        self._rate_limiter = RateLimiter()

    # ── Response Helpers ─────────────────────────────────────────

    def _response_headers(self) -> dict[str, str]:
        """Standard response headers for A2A responses."""
        return {
            "Content-Type": A2A_CONTENT_TYPE,
            A2A_VERSION_HEADER: A2A_PROTOCOL_VERSION,
        }

    def _extract_token(self, auth_header: str) -> str | None:
        """Extracts bearer token from Authorization header."""
        if auth_header and auth_header.startswith("Bearer "):
            return auth_header[7:]
        return None

    # ── Route Handlers (Framework-agnostisch) ────────────────────

    async def handle_agent_card(self) -> dict[str, Any]:
        """GET /.well-known/agent.json -- Agent Card Discovery."""
        return cast("dict[str, Any]", self.adapter.get_agent_card())

    async def handle_jsonrpc(
        self,
        body: dict[str, Any],
        auth_header: str = "",
        client_version: str = "",
    ) -> dict[str, Any]:
        """POST /a2a -- JSON-RPC 2.0 Dispatch."""
        return await self.adapter.handle_a2a_request(  # type: ignore[no-any-return]
            body,
            auth_header=auth_header,
            client_version=client_version,
        )

    async def handle_health(self) -> dict[str, Any]:
        """GET /a2a/health -- Health Check."""
        stats = self.adapter.stats()
        return {
            "status": "ok" if self.adapter.enabled else "disabled",
            "protocol_version": A2A_PROTOCOL_VERSION,
            "enabled": self.adapter.enabled,
            "server_running": stats.get("server", {}).get("running", False),
        }

    # ── FastAPI Registration ─────────────────────────────────────

    def register_routes(self, app: Any) -> None:
        """Registers A2A routes in a FastAPI app.

        Args:
            app: FastAPI instance (or APIRouter)
        """
        try:
            from starlette.responses import JSONResponse, StreamingResponse
        except ImportError:
            log.warning("a2a_http_starlette_not_available")
            return

        handler = self  # Closure reference

        @app.get("/.well-known/agent.json")  # type: ignore[misc]
        async def well_known_agent_card() -> JSONResponse:
            """Agent Card Discovery (A2A RC v1.0)."""
            card = await handler.handle_agent_card()
            return JSONResponse(
                content=card,
                headers=handler._response_headers(),
            )

        @app.post("/a2a")  # type: ignore[misc]
        async def a2a_jsonrpc(request: Request) -> JSONResponse:
            """JSON-RPC 2.0 Endpoint (A2A RC v1.0)."""
            client_ip = request.client.host if request.client else "unknown"
            if not await handler._rate_limiter.check(client_ip):
                return JSONResponse(
                    content={
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {"code": -32000, "message": "Too Many Requests"},
                    },
                    status_code=429,
                    headers=handler._response_headers(),
                )
            try:
                body = await request.json()
            except Exception:
                return JSONResponse(
                    content={
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {"code": -32700, "message": "Parse error"},
                    },
                    status_code=400,
                    headers=handler._response_headers(),
                )

            auth = request.headers.get("Authorization", "")
            version = request.headers.get(A2A_VERSION_HEADER, "")

            result = await handler.adapter.handle_a2a_request(
                body,
                auth_header=auth,
                client_version=version,
            )
            status = 200
            if "error" in result:
                code = result.get("error", {}).get("code", 0)
                if code == -32004:  # UNAUTHORIZED
                    status = 401
                elif code == -32005:  # INCOMPATIBLE_VERSION
                    status = 400

            return JSONResponse(
                content=result,
                status_code=status,
                headers=handler._response_headers(),
            )

        @app.post("/a2a/stream")  # type: ignore[misc]
        async def a2a_stream(request: Request) -> StreamingResponse:
            """SSE Streaming Endpoint (message/stream)."""
            try:
                body = await request.json()
            except Exception:

                async def error_gen() -> Any:
                    yield 'event: error\ndata: {"code": -32700, "message": "Parse error"}\n\n'

                return StreamingResponse(
                    error_gen(),
                    media_type="text/event-stream",
                )

            auth = request.headers.get("Authorization", "")
            token = handler._extract_token(auth)

            async def event_generator() -> Any:
                async for event in handler.adapter.handle_stream_request(body, auth_token=token):
                    yield event

            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers={A2A_VERSION_HEADER: A2A_PROTOCOL_VERSION},
            )

        @app.get("/a2a/health")  # type: ignore[misc]
        async def a2a_health() -> JSONResponse:
            """Health Check."""
            result = await handler.handle_health()
            return JSONResponse(content=result)

        log.info("a2a_http_routes_registered")

    # ── Standalone Server ────────────────────────────────────────

    async def start_standalone(self, host: str = "127.0.0.1", port: int = 3002) -> None:
        """Starts a standalone A2A HTTP server.

        Used when Jarvis has no gateway HTTP server
        but still needs to receive A2A requests.
        """
        try:
            from aiohttp import web
        except ImportError:
            log.warning("a2a_aiohttp_not_installed_using_fallback")
            # Fallback: simple asyncio-based server
            await self._start_minimal_server(host, port)
            return

        app = web.Application()
        handler = self

        async def well_known(request: web.Request) -> web.Response:
            card = await handler.handle_agent_card()
            import json

            return web.Response(
                text=json.dumps(card),
                content_type=A2A_CONTENT_TYPE,
                headers={A2A_VERSION_HEADER: A2A_PROTOCOL_VERSION},
            )

        async def jsonrpc(request: web.Request) -> web.Response:
            import json

            try:
                body = await request.json()
            except Exception:
                return web.Response(
                    text=json.dumps(
                        {
                            "jsonrpc": "2.0",
                            "id": None,
                            "error": {"code": -32700, "message": "Parse error"},
                        }
                    ),
                    status=400,
                    content_type=A2A_CONTENT_TYPE,
                    headers={A2A_VERSION_HEADER: A2A_PROTOCOL_VERSION},
                )
            auth = request.headers.get("Authorization", "")
            version = request.headers.get(A2A_VERSION_HEADER, "")
            result = await handler.adapter.handle_a2a_request(
                body,
                auth_header=auth,
                client_version=version,
            )
            return web.Response(
                text=json.dumps(result),
                content_type=A2A_CONTENT_TYPE,
                headers={A2A_VERSION_HEADER: A2A_PROTOCOL_VERSION},
            )

        async def health(request: web.Request) -> web.Response:
            result = await handler.handle_health()
            import json

            return web.Response(text=json.dumps(result), content_type="application/json")

        app.router.add_get("/.well-known/agent.json", well_known)
        app.router.add_post("/a2a", jsonrpc)
        app.router.add_get("/a2a/health", health)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()
        log.info("a2a_standalone_server_started", host=host, port=port)

    async def _start_minimal_server(self, host: str, port: int) -> None:
        """Minimal asyncio-based HTTP server as fallback."""
        import asyncio
        import json

        async def handle_connection(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ) -> None:
            try:
                data = await reader.read(65536)
                request_line = data.decode("utf-8", errors="replace").split("\r\n")[0]
                method, path, _ = request_line.split(" ", 2)

                if method == "GET" and path == "/.well-known/agent.json":
                    card = await self.handle_agent_card()
                    body = json.dumps(card)
                elif method == "POST" and path == "/a2a":
                    # Extract JSON-RPC body from HTTP request
                    header_end = data.find(b"\r\n\r\n")
                    rpc_body = data[header_end + 4 :] if header_end != -1 else b""
                    try:
                        rpc_request = json.loads(rpc_body.decode("utf-8"))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        body = json.dumps(
                            {
                                "jsonrpc": "2.0",
                                "id": None,
                                "error": {"code": -32700, "message": "Parse error"},
                            }
                        )
                    else:
                        result = await self.adapter.handle_a2a_request(rpc_request)
                        body = json.dumps(result)
                elif method == "GET" and path == "/a2a/health":
                    result = await self.handle_health()
                    body = json.dumps(result)
                else:
                    body = json.dumps({"error": "Not found"})

                body_bytes = body.encode("utf-8")
                header = (
                    f"HTTP/1.1 200 OK\r\n"
                    f"Content-Type: {A2A_CONTENT_TYPE}\r\n"
                    f"Content-Length: {len(body_bytes)}\r\n"
                    f"\r\n"
                ).encode()
                writer.write(header + body_bytes)
                await writer.drain()
            except Exception:
                pass  # Cleanup — error response send failure is non-critical
            finally:
                writer.close()

        _server = await asyncio.start_server(handle_connection, host, port)
        log.info("a2a_minimal_server_started", host=host, port=port)
