"""VS Code Extension API routes — /api/v1/chat/completions endpoint.

Provides structured chat completions with code context awareness
for the Cognithor VS Code Extension.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from fastapi.responses import JSONResponse

if TYPE_CHECKING:
    from fastapi import FastAPI, Request

log = logging.getLogger(__name__)


def register_vscode_routes(
    app: FastAPI,
    gateway: Any,
    deps: list[Any],
) -> None:
    """Register VS Code Extension API endpoints."""

    @app.post("/api/v1/chat/completions", dependencies=deps)
    async def chat_completions(request: Request) -> JSONResponse:
        """
        VS Code Extension chat completions endpoint.

        Accepts a message with optional code context and returns
        a structured response with code blocks.

        Request:
            {
                "message": "Explain this function",
                "sessionId": "uuid",
                "model": "auto",
                "language": "de",
                "context": {
                    "filePath": "/path/to/file.py",
                    "language": "python",
                    "selectedCode": "def foo(): ...",
                    "surroundingCode": "...",
                    "cursorLine": 42,
                    "projectFiles": ["/path/to/utils.py"]
                }
            }

        Response:
            {
                "message": "The function foo...",
                "codeBlocks": [{"language": "python", "code": "...", "explanation": "..."}],
                "model": "qwen3:32b",
                "tokenCount": 347,
                "sessionId": "uuid"
            }
        """
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid JSON body"},
            )

        message = body.get("message", "").strip()
        if not message:
            return JSONResponse(
                status_code=400,
                content={"error": "Empty message"},
            )

        session_id = body.get("sessionId", f"vscode_{int(time.time())}")
        context = body.get("context", {})

        # Build enriched prompt from code context
        prompt_parts = [message]

        if context.get("selectedCode"):
            prompt_parts.append(
                f"\n\n```{context.get('language', '')}\n{context['selectedCode']}\n```"
            )

        if context.get("filePath"):
            prompt_parts.append(f"\nDatei: {context['filePath']}")

        if context.get("surroundingCode"):
            prompt_parts.append(
                f"\nKontext:\n```{context.get('language', '')}\n"
                f"{context['surroundingCode'][:2000]}\n```"
            )

        enriched_message = "\n".join(prompt_parts)

        # Send through gateway
        from cognithor.models import IncomingMessage

        incoming = IncomingMessage(
            text=enriched_message,
            channel="vscode",
            user_id="vscode_user",
            session_id=session_id,
        )

        start = time.monotonic()
        try:
            response = await gateway.handle_message(incoming)
        except Exception as exc:
            log.error("vscode_chat_error", error=str(exc))
            return JSONResponse(
                status_code=500,
                content={"error": str(exc)},
            )
        duration_ms = int((time.monotonic() - start) * 1000)

        # Extract code blocks from response
        code_blocks = _extract_code_blocks(response.text or "")

        result = {
            "message": response.text or "",
            "codeBlocks": code_blocks,
            "model": getattr(response, "model", "unknown"),
            "tokenCount": getattr(response, "token_count", 0) or len((response.text or "").split()),
            "sessionId": session_id,
            "durationMs": duration_ms,
        }

        log.info(
            "vscode_chat_completion",
            session=session_id[:8],
            duration_ms=duration_ms,
            code_blocks=len(code_blocks),
        )

        return JSONResponse(content=result)


def _extract_code_blocks(text: str) -> list[dict[str, str]]:
    """Extract fenced code blocks from markdown text."""
    import re

    blocks = []
    pattern = r"```(\w*)\n(.*?)```"
    for match in re.finditer(pattern, text, re.DOTALL):
        lang = match.group(1) or "text"
        code = match.group(2).strip()
        if code:
            blocks.append(
                {
                    "language": lang,
                    "code": code,
                }
            )
    return blocks
