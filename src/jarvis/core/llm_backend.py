"""LLM backend abstraction: Unified interface for various LLM providers.

Enables switching between Ollama (local), OpenAI-compatible API,
and Anthropic Claude API -- without changes to the rest of the system.

The ModelRouter continues to use its own logic for model selection,
but delegates the actual communication to the configured backend.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, AsyncIterator

import httpx

from jarvis.utils.logging import get_logger

if TYPE_CHECKING:
    from jarvis.config import JarvisConfig

log = get_logger(__name__)


# ============================================================================
# Typen und Datenklassen
# ============================================================================


class LLMBackendType(StrEnum):
    """Supported LLM backends."""

    OLLAMA = "ollama"
    OPENAI = "openai"  # OpenAI-kompatible APIs (OpenAI, Together, Groq, vLLM, ...)
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


@dataclass
class ChatResponse:
    """Unified response from all backends.

    Attributes:
        content: Response text.
        tool_calls: Optional tool call list (function calling).
        model: Model used.
        usage: Token consumption (prompt_tokens, completion_tokens, total_tokens).
        raw: Original backend response for debugging.
    """

    content: str = ""
    tool_calls: list[dict[str, Any]] | None = None
    model: str = ""
    usage: dict[str, int] | None = None
    raw: dict[str, Any] | None = None


@dataclass
class EmbedResponse:
    """Unified embedding response."""

    embedding: list[float]
    model: str = ""


class LLMBackendError(Exception):
    """Error communicating with the LLM backend."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


# ============================================================================
# Abstrakte Basis
# ============================================================================


class LLMBackend(ABC):
    """Abstract interface for LLM providers.

    Each backend implements chat(), chat_stream() and embed().
    """

    @property
    @abstractmethod
    def backend_type(self) -> LLMBackendType:
        """Typ des Backends."""
        ...

    @abstractmethod
    async def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        format_json: bool = False,
    ) -> ChatResponse:
        """Sendet eine Chat-Anfrage und wartet auf die vollständige Antwort."""
        ...

    @abstractmethod
    async def chat_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> AsyncIterator[str]:
        """Streamt die Antwort Token für Token."""
        ...

    @abstractmethod
    async def embed(
        self,
        model: str,
        text: str,
    ) -> EmbedResponse:
        """Erstellt ein Embedding für den gegebenen Text."""
        ...

    @abstractmethod
    async def is_available(self) -> bool:
        """Prüft ob das Backend erreichbar ist."""
        ...

    @abstractmethod
    async def list_models(self) -> list[str]:
        """Listet alle verfügbaren Modelle."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Gibt Ressourcen frei."""
        ...


# ============================================================================
# Ollama-Backend (lokal)
# ============================================================================


class OllamaBackend(LLMBackend):
    """Ollama REST API backend for local models.

    Wraps the existing OllamaClient into the unified interface.
    Supports all Ollama features: chat, streaming, embeddings, tools.
    """

    def __init__(self, base_url: str, timeout: int = 120, keep_alive: str = "30m") -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._keep_alive = keep_alive
        self._client: httpx.AsyncClient | None = None

    @property
    def backend_type(self) -> LLMBackendType:
        return LLMBackendType.OLLAMA

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(
                    connect=10.0,
                    read=float(self._timeout),
                    write=30.0,
                    pool=10.0,
                ),
                limits=httpx.Limits(max_connections=5, max_keepalive_connections=2),
                trust_env=False,
            )
        return self._client

    async def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        format_json: bool = False,
    ) -> ChatResponse:
        client = await self._ensure_client()

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "top_p": top_p},
            "keep_alive": self._keep_alive,
        }
        if tools:
            payload["tools"] = tools
        if format_json:
            payload["format"] = "json"

        start = time.monotonic()
        try:
            resp = await client.post("/api/chat", json=payload)
            if resp.status_code != 200:
                raise LLMBackendError(
                    f"Ollama HTTP {resp.status_code}: {resp.text[:500]}",
                    status_code=resp.status_code,
                )
            data = resp.json()
            duration_ms = int((time.monotonic() - start) * 1000)
            log.debug("ollama_chat", model=model, duration_ms=duration_ms)

            msg = data.get("message", {})
            tool_calls = msg.get("tool_calls") or None

            return ChatResponse(
                content=msg.get("content", ""),
                tool_calls=tool_calls,
                model=model,
                usage={
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                    "total_tokens": data.get("prompt_eval_count", 0)
                    + data.get("eval_count", 0),
                },
                raw=data,
            )
        except httpx.TimeoutException as exc:
            raise LLMBackendError(f"Ollama Timeout nach {self._timeout}s") from exc
        except httpx.ConnectError as exc:
            raise LLMBackendError(f"Ollama nicht erreichbar: {self._base_url}") from exc

    async def chat_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> AsyncIterator[str]:
        client = await self._ensure_client()
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": temperature, "top_p": top_p},
            "keep_alive": self._keep_alive,
        }

        async with client.stream("POST", "/api/chat", json=payload) as resp:
            if resp.status_code != 200:
                raise LLMBackendError(
                    f"Ollama Stream HTTP {resp.status_code}",
                    status_code=resp.status_code,
                )
            import json as _json

            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                chunk = _json.loads(line)
                token = chunk.get("message", {}).get("content", "")
                if token:
                    yield token

    async def embed(self, model: str, text: str) -> EmbedResponse:
        client = await self._ensure_client()
        resp = await client.post(
            "/api/embed",
            json={"model": model, "input": text},
        )
        if resp.status_code != 200:
            raise LLMBackendError(f"Ollama Embed HTTP {resp.status_code}")
        data = resp.json()
        embeddings = data.get("embeddings", [[]])
        return EmbedResponse(
            embedding=embeddings[0] if embeddings else [],
            model=model,
        )

    async def is_available(self) -> bool:
        try:
            client = await self._ensure_client()
            resp = await client.get("/api/tags", timeout=5.0)
            return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException, OSError):
            return False

    async def list_models(self) -> list[str]:
        try:
            client = await self._ensure_client()
            resp = await client.get("/api/tags")
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            return []

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None


# ============================================================================
# OpenAI-kompatibles Backend (OpenAI, Together, Groq, vLLM, LM Studio, ...)
# ============================================================================


class OpenAIBackend(LLMBackend):
    """OpenAI-compatible chat completions backend.

    Works with all providers that support the OpenAI API format:
    OpenAI, Together AI, Groq, Fireworks, vLLM, LM Studio, Ollama/OpenAI mode.

    Args:
        api_key: API key (can be empty for local servers).
        base_url: API endpoint (default: OpenAI).
        timeout: Request timeout in seconds.
    """

    # Modelle die temperature/top_p nicht unterstützen (OpenAI Reasoning Models)
    _REASONING_MODEL_PREFIXES = ("o1", "o3", "o4")

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://api.openai.com/v1",
        timeout: int = 120,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    @property
    def backend_type(self) -> LLMBackendType:
        return LLMBackendType.OPENAI

    def _is_reasoning_model(self, model: str) -> bool:
        """Prüft ob ein Modell ein Reasoning-Model ist (kein temperature/top_p).

        Erkennt: o1, o3, o4, gpt-5*, gpt-5.1*, gpt-5.2* und Varianten.
        """
        model_lower = model.lower()
        # Prefix-Check: "o1-...", "o3-...", "o4-mini-..."
        for prefix in self._REASONING_MODEL_PREFIXES:
            if model_lower == prefix or model_lower.startswith(f"{prefix}-"):
                return True
        # GPT-5+ Modelle: gpt-5, gpt-5-mini, gpt-5.1, gpt-5.2, gpt-5.2-pro, etc.
        # Alle GPT-5+ Varianten unterstützen kein benutzerdefiniertes temperature
        if model_lower.startswith("gpt-5"):
            return True
        return False

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            headers: dict[str, str] = {"Content-Type": "application/json"}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers=headers,
                timeout=httpx.Timeout(connect=10.0, read=float(self._timeout), write=30.0, pool=10.0),
                trust_env=False,
            )
        return self._client

    async def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        format_json: bool = False,
    ) -> ChatResponse:
        client = await self._ensure_client()

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }

        # Reasoning-Models (o1, o3, o4, gpt-5.2, ...) unterstützen
        # temperature/top_p nicht — nur den Default (1.0)
        if not self._is_reasoning_model(model):
            payload["temperature"] = temperature
            payload["top_p"] = top_p

        if tools:
            # OpenAI-Format: tools als Function-Definitionen
            payload["tools"] = tools
        if format_json:
            payload["response_format"] = {"type": "json_object"}

        start = time.monotonic()
        try:
            resp = await client.post("/chat/completions", json=payload)

            # Retry ohne temperature/top_p bei 400 "unsupported_value"
            if resp.status_code == 400 and "temperature" in resp.text and "temperature" in payload:
                log.info("openai_retry_without_temperature", model=model)
                payload.pop("temperature", None)
                payload.pop("top_p", None)
                resp = await client.post("/chat/completions", json=payload)

            if resp.status_code != 200:
                raise LLMBackendError(
                    f"OpenAI HTTP {resp.status_code}: {resp.text[:500]}",
                    status_code=resp.status_code,
                )
            data = resp.json()
            duration_ms = int((time.monotonic() - start) * 1000)
            log.debug("openai_chat", model=model, duration_ms=duration_ms)

            choice = data.get("choices", [{}])[0]
            msg = choice.get("message", {})
            tool_calls_raw = msg.get("tool_calls")

            # Tool-Calls ins Jarvis-Format konvertieren
            tool_calls = None
            if tool_calls_raw:
                import json as _json

                tool_calls = []
                for tc in tool_calls_raw:
                    fn = tc.get("function", {})
                    args = fn.get("arguments", "{}")
                    try:
                        parsed_args = _json.loads(args) if isinstance(args, str) else args
                    except _json.JSONDecodeError:
                        parsed_args = {"raw": args}
                    tool_calls.append({
                        "function": {"name": fn.get("name", ""), "arguments": parsed_args},
                    })

            usage = data.get("usage", {})
            return ChatResponse(
                content=msg.get("content", "") or "",
                tool_calls=tool_calls,
                model=data.get("model", model),
                usage={
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
                raw=data,
            )
        except httpx.TimeoutException as exc:
            raise LLMBackendError(f"OpenAI Timeout nach {self._timeout}s") from exc
        except httpx.ConnectError as exc:
            raise LLMBackendError(f"OpenAI nicht erreichbar: {self._base_url}") from exc

    async def chat_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> AsyncIterator[str]:
        client = await self._ensure_client()
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        if not self._is_reasoning_model(model):
            payload["temperature"] = temperature
            payload["top_p"] = top_p

        async with client.stream("POST", "/chat/completions", json=payload) as resp:
            if resp.status_code != 200:
                raise LLMBackendError(f"OpenAI Stream HTTP {resp.status_code}")
            import json as _json

            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break
                chunk = _json.loads(data_str)
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                token = delta.get("content", "")
                if token:
                    yield token

    async def embed(self, model: str, text: str) -> EmbedResponse:
        client = await self._ensure_client()
        resp = await client.post(
            "/embeddings",
            json={"model": model, "input": text},
        )
        if resp.status_code != 200:
            raise LLMBackendError(f"OpenAI Embed HTTP {resp.status_code}")
        data = resp.json()
        embedding = data.get("data", [{}])[0].get("embedding", [])
        return EmbedResponse(embedding=embedding, model=model)

    async def is_available(self) -> bool:
        try:
            client = await self._ensure_client()
            resp = await client.get("/models", timeout=10.0)
            return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException, OSError):
            return False

    async def list_models(self) -> list[str]:
        try:
            client = await self._ensure_client()
            resp = await client.get("/models")
            resp.raise_for_status()
            return [m["id"] for m in resp.json().get("data", [])]
        except Exception:
            return []

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None


# ============================================================================
# Anthropic-Backend (Claude API)
# ============================================================================


class AnthropicBackend(LLMBackend):
    """Anthropic Messages API backend for Claude models.

    Uses the native Anthropic API (not OpenAI-compatible).
    Supports tool use in Anthropic format.

    Args:
        api_key: Anthropic API key.
        timeout: Request timeout in seconds.
        max_tokens: Maximum output tokens (Anthropic requires this parameter).
    """

    API_URL = "https://api.anthropic.com/v1"
    API_VERSION = "2023-06-01"

    def __init__(
        self,
        api_key: str,
        timeout: int = 120,
        max_tokens: int = 4096,
    ) -> None:
        self._api_key = api_key
        self._timeout = timeout
        self._max_tokens = max_tokens
        self._client: httpx.AsyncClient | None = None

    @property
    def backend_type(self) -> LLMBackendType:
        return LLMBackendType.ANTHROPIC

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.API_URL,
                headers={
                    "x-api-key": self._api_key,
                    "anthropic-version": self.API_VERSION,
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(connect=10.0, read=float(self._timeout), write=30.0, pool=10.0),
                trust_env=False,
            )
        return self._client

    async def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        format_json: bool = False,
    ) -> ChatResponse:
        client = await self._ensure_client()

        # Anthropic: system message separat, nicht in messages
        system_text = ""
        api_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                sys_content = msg.get("content", "")
                if isinstance(sys_content, list):
                    sys_content = " ".join(
                        block.get("text", "")
                        for block in sys_content
                        if isinstance(block, dict) and block.get("type") == "text"
                    )
                system_text += sys_content + "\n"
            else:
                content = msg.get("content", "")
                api_messages.append({"role": msg["role"], "content": content})

        payload: dict[str, Any] = {
            "model": model,
            "messages": api_messages,
            "max_tokens": self._max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if system_text.strip():
            payload["system"] = system_text.strip()
        if tools:
            # Anthropic Tool-Format
            payload["tools"] = self._convert_tools_to_anthropic(tools)

        start = time.monotonic()
        try:
            resp = await client.post("/messages", json=payload)
            if resp.status_code != 200:
                raise LLMBackendError(
                    f"Anthropic HTTP {resp.status_code}: {resp.text[:500]}",
                    status_code=resp.status_code,
                )
            data = resp.json()
            duration_ms = int((time.monotonic() - start) * 1000)
            log.debug("anthropic_chat", model=model, duration_ms=duration_ms)

            # Antwort zusammenbauen
            content_parts = []
            tool_calls = []
            for block in data.get("content", []):
                if block.get("type") == "text":
                    content_parts.append(block.get("text", ""))
                elif block.get("type") == "tool_use":
                    tool_calls.append({
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": block.get("input", {}),
                        },
                    })

            usage = data.get("usage", {})
            return ChatResponse(
                content="\n".join(content_parts),
                tool_calls=tool_calls if tool_calls else None,
                model=data.get("model", model),
                usage={
                    "prompt_tokens": usage.get("input_tokens", 0),
                    "completion_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("input_tokens", 0)
                    + usage.get("output_tokens", 0),
                },
                raw=data,
            )
        except httpx.TimeoutException as exc:
            raise LLMBackendError(f"Anthropic Timeout nach {self._timeout}s") from exc
        except httpx.ConnectError as exc:
            raise LLMBackendError("Anthropic API nicht erreichbar") from exc

    async def chat_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> AsyncIterator[str]:
        client = await self._ensure_client()

        system_text = ""
        api_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                sys_content = msg.get("content", "")
                if isinstance(sys_content, list):
                    sys_content = " ".join(
                        block.get("text", "")
                        for block in sys_content
                        if isinstance(block, dict) and block.get("type") == "text"
                    )
                system_text += sys_content + "\n"
            else:
                content = msg.get("content", "")
                api_messages.append({"role": msg["role"], "content": content})

        payload: dict[str, Any] = {
            "model": model,
            "messages": api_messages,
            "max_tokens": self._max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": True,
        }
        if system_text.strip():
            payload["system"] = system_text.strip()

        import json as _json

        async with client.stream("POST", "/messages", json=payload) as resp:
            if resp.status_code != 200:
                raise LLMBackendError(f"Anthropic Stream HTTP {resp.status_code}")
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                chunk = _json.loads(line[6:])
                if chunk.get("type") == "content_block_delta":
                    delta = chunk.get("delta", {})
                    if delta.get("type") == "text_delta":
                        yield delta.get("text", "")

    async def embed(self, model: str, text: str) -> EmbedResponse:
        # Anthropic bietet keine Embedding-API an.
        # Fallback: Ollama oder OpenAI-Embeddings verwenden.
        raise LLMBackendError(
            "Anthropic bietet keine Embedding-API. "
            "Verwende Ollama oder OpenAI für Embeddings."
        )

    async def is_available(self) -> bool:
        try:
            client = await self._ensure_client()
            # Use invalid empty body -- a 400 means the API is reachable and auth is valid.
            # A 401 means bad key, connection errors mean unreachable.
            # This avoids wasting tokens on a real completion.
            resp = await client.post(
                "/messages",
                json={"model": "claude-sonnet-4-20250514", "max_tokens": 1, "messages": []},
                timeout=10.0,
            )
            # 200 = somehow worked, 400 = API reachable (validation error), both mean available
            return resp.status_code in (200, 400)
        except (httpx.ConnectError, httpx.TimeoutException, OSError):
            return False

    async def list_models(self) -> list[str]:
        # Statische Liste -- Anthropic hat keinen dynamischen Modell-Endpunkt
        return [
            "claude-sonnet-4-20250514",
            "claude-haiku-4-5-20251001",
            "claude-opus-4-5-20250918",
        ]

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    @staticmethod
    def _convert_tools_to_anthropic(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Konvertiert MCP/OpenAI Tool-Schemas ins Anthropic-Format."""
        result = []
        for tool in tools:
            if "function" in tool:
                fn = tool["function"]
                result.append({
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
                })
            elif "name" in tool:
                result.append({
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "input_schema": tool.get("inputSchema", {"type": "object", "properties": {}}),
                })
        return result


# ============================================================================
# Google Gemini Backend
# ============================================================================


class GeminiBackend(LLMBackend):
    """Google Gemini API backend.

    Uses the native Gemini REST API (generativelanguage.googleapis.com).
    Supports chat, streaming, embeddings, and tool calling.

    Args:
        api_key: Google Gemini API key.
        timeout: Request timeout in seconds.
    """

    API_URL = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, api_key: str, timeout: int = 120) -> None:
        self._api_key = api_key
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    @property
    def backend_type(self) -> LLMBackendType:
        return LLMBackendType.GEMINI

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=10.0,
                    read=float(self._timeout),
                    write=30.0,
                    pool=10.0,
                ),
                trust_env=False,
            )
        return self._client

    @staticmethod
    def _convert_messages(
        messages: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, Any]]]:
        """Konvertiert OpenAI-Format Messages zu Gemini-Format.

        Returns:
            Tuple von (system_instruction_text, gemini_contents).
        """
        system_parts: list[str] = []
        contents: list[dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                if isinstance(content, list):
                    system_parts.extend(
                        block.get("text", "")
                        for block in content
                        if isinstance(block, dict) and block.get("type") == "text"
                    )
                else:
                    system_parts.append(str(content))
            else:
                gemini_role = "model" if role == "assistant" else "user"
                if isinstance(content, str):
                    contents.append({
                        "role": gemini_role,
                        "parts": [{"text": content}],
                    })
                elif isinstance(content, list):
                    parts = []
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            parts.append({"text": block.get("text", "")})
                        else:
                            parts.append({"text": str(block)})
                    contents.append({"role": gemini_role, "parts": parts})

        return "\n".join(system_parts), contents

    @staticmethod
    def _convert_tools_to_gemini(
        tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Konvertiert OpenAI-Style Tools zu Gemini functionDeclarations."""
        declarations: list[dict[str, Any]] = []
        for tool in tools:
            if "function" in tool:
                fn = tool["function"]
                declarations.append({
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
                })
            elif "name" in tool:
                declarations.append({
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("inputSchema", {"type": "object", "properties": {}}),
                })
        return declarations

    async def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        format_json: bool = False,
    ) -> ChatResponse:
        client = await self._ensure_client()

        system_text, contents = self._convert_messages(messages)

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "topP": top_p,
            },
        }
        if system_text.strip():
            payload["systemInstruction"] = {
                "parts": [{"text": system_text.strip()}],
            }
        if tools:
            payload["tools"] = [{
                "functionDeclarations": self._convert_tools_to_gemini(tools),
            }]
        if format_json:
            payload["generationConfig"]["responseMimeType"] = "application/json"

        url = f"{self.API_URL}/models/{model}:generateContent?key={self._api_key}"

        start = time.monotonic()
        try:
            resp = await client.post(url, json=payload)
            if resp.status_code != 200:
                raise LLMBackendError(
                    f"Gemini HTTP {resp.status_code}: {resp.text[:500]}",
                    status_code=resp.status_code,
                )
            data = resp.json()
            duration_ms = int((time.monotonic() - start) * 1000)
            log.debug("gemini_chat", model=model, duration_ms=duration_ms)

            # Antwort parsen
            candidates = data.get("candidates", [])
            if not candidates:
                return ChatResponse(content="", model=model, raw=data)

            candidate = candidates[0]
            content_parts = candidate.get("content", {}).get("parts", [])

            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            for part in content_parts:
                if "text" in part:
                    text_parts.append(part["text"])
                elif "functionCall" in part:
                    fc = part["functionCall"]
                    tool_calls.append({
                        "function": {
                            "name": fc.get("name", ""),
                            "arguments": fc.get("args", {}),
                        },
                    })

            usage_meta = data.get("usageMetadata", {})
            return ChatResponse(
                content="\n".join(text_parts),
                tool_calls=tool_calls if tool_calls else None,
                model=model,
                usage={
                    "prompt_tokens": usage_meta.get("promptTokenCount", 0),
                    "completion_tokens": usage_meta.get("candidatesTokenCount", 0),
                    "total_tokens": usage_meta.get("totalTokenCount", 0),
                },
                raw=data,
            )
        except httpx.TimeoutException as exc:
            raise LLMBackendError(f"Gemini Timeout nach {self._timeout}s") from exc
        except httpx.ConnectError as exc:
            raise LLMBackendError("Gemini API nicht erreichbar") from exc

    async def chat_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> AsyncIterator[str]:
        client = await self._ensure_client()

        system_text, contents = self._convert_messages(messages)

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "topP": top_p,
            },
        }
        if system_text.strip():
            payload["systemInstruction"] = {
                "parts": [{"text": system_text.strip()}],
            }

        url = (
            f"{self.API_URL}/models/{model}:streamGenerateContent"
            f"?alt=sse&key={self._api_key}"
        )

        import json as _json

        async with client.stream("POST", url, json=payload) as resp:
            if resp.status_code != 200:
                raise LLMBackendError(
                    f"Gemini Stream HTTP {resp.status_code}",
                    status_code=resp.status_code,
                )
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if not data_str.strip():
                    continue
                chunk = _json.loads(data_str)
                candidates = chunk.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    for part in parts:
                        text = part.get("text", "")
                        if text:
                            yield text

    async def embed(self, model: str, text: str) -> EmbedResponse:
        client = await self._ensure_client()
        url = f"{self.API_URL}/models/{model}:embedContent?key={self._api_key}"
        payload = {
            "content": {
                "parts": [{"text": text}],
            },
        }
        resp = await client.post(url, json=payload)
        if resp.status_code != 200:
            raise LLMBackendError(
                f"Gemini Embed HTTP {resp.status_code}: {resp.text[:500]}",
                status_code=resp.status_code,
            )
        data = resp.json()
        embedding = data.get("embedding", {}).get("values", [])
        return EmbedResponse(embedding=embedding, model=model)

    async def is_available(self) -> bool:
        try:
            client = await self._ensure_client()
            url = f"{self.API_URL}/models?key={self._api_key}"
            resp = await client.get(url, timeout=10.0)
            return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException, OSError):
            return False

    async def list_models(self) -> list[str]:
        try:
            client = await self._ensure_client()
            url = f"{self.API_URL}/models?key={self._api_key}"
            resp = await client.get(url)
            resp.raise_for_status()
            return [
                m.get("name", "").replace("models/", "")
                for m in resp.json().get("models", [])
            ]
        except Exception:
            return []

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None


# ============================================================================
# Factory
# ============================================================================


def create_backend(config: JarvisConfig) -> LLMBackend:
    """Erstellt das konfigurierte LLM-Backend.

    Liest `config.llm_backend` und gibt die passende Implementierung zurück.
    Default: Ollama (lokal, keine API-Keys nötig).
    """
    backend_type = getattr(config, "llm_backend_type", "ollama")

    match backend_type:
        case "openai":
            return OpenAIBackend(
                api_key=getattr(config, "openai_api_key", ""),
                base_url=getattr(config, "openai_base_url", "https://api.openai.com/v1"),
                timeout=config.ollama.timeout_seconds,
            )
        case "anthropic":
            return AnthropicBackend(
                api_key=getattr(config, "anthropic_api_key", ""),
                timeout=config.ollama.timeout_seconds,
                max_tokens=getattr(config, "anthropic_max_tokens", 4096),
            )
        case "gemini":
            return GeminiBackend(
                api_key=getattr(config, "gemini_api_key", ""),
                timeout=config.ollama.timeout_seconds,
            )
        case "groq":
            return OpenAIBackend(
                api_key=getattr(config, "groq_api_key", ""),
                base_url="https://api.groq.com/openai/v1",
                timeout=config.ollama.timeout_seconds,
            )
        case "deepseek":
            return OpenAIBackend(
                api_key=getattr(config, "deepseek_api_key", ""),
                base_url="https://api.deepseek.com/v1",
                timeout=config.ollama.timeout_seconds,
            )
        case "mistral":
            return OpenAIBackend(
                api_key=getattr(config, "mistral_api_key", ""),
                base_url="https://api.mistral.ai/v1",
                timeout=config.ollama.timeout_seconds,
            )
        case "together":
            return OpenAIBackend(
                api_key=getattr(config, "together_api_key", ""),
                base_url="https://api.together.xyz/v1",
                timeout=config.ollama.timeout_seconds,
            )
        case "openrouter":
            return OpenAIBackend(
                api_key=getattr(config, "openrouter_api_key", ""),
                base_url="https://openrouter.ai/api/v1",
                timeout=config.ollama.timeout_seconds,
            )
        case "xai":
            return OpenAIBackend(
                api_key=getattr(config, "xai_api_key", ""),
                base_url="https://api.x.ai/v1",
                timeout=config.ollama.timeout_seconds,
            )
        case "cerebras":
            return OpenAIBackend(
                api_key=getattr(config, "cerebras_api_key", ""),
                base_url="https://api.cerebras.ai/v1",
                timeout=config.ollama.timeout_seconds,
            )
        case "github":
            return OpenAIBackend(
                api_key=getattr(config, "github_api_key", ""),
                base_url="https://models.inference.ai.azure.com",
                timeout=config.ollama.timeout_seconds,
            )
        case "bedrock":
            return OpenAIBackend(
                api_key=getattr(config, "bedrock_api_key", ""),
                base_url="https://bedrock-runtime.us-east-1.amazonaws.com/v1",
                timeout=config.ollama.timeout_seconds,
            )
        case "huggingface":
            return OpenAIBackend(
                api_key=getattr(config, "huggingface_api_key", ""),
                base_url="https://api-inference.huggingface.co/v1",
                timeout=config.ollama.timeout_seconds,
            )
        case "moonshot":
            return OpenAIBackend(
                api_key=getattr(config, "moonshot_api_key", ""),
                base_url="https://api.moonshot.cn/v1",
                timeout=config.ollama.timeout_seconds,
            )
        case _:
            return OllamaBackend(
                base_url=config.ollama.base_url,
                timeout=config.ollama.timeout_seconds,
                keep_alive=config.ollama.keep_alive,
            )
