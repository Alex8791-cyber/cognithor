"""Model Router: Model selection and multi-backend communication.

Responsible for:
  - Model selection based on task type and complexity [B§8.2]
  - Async communication with LLM backends (chat + embeddings) [B§8.1]
  - Supports Ollama (local), OpenAI-compatible, Anthropic Claude
  - Streaming (token by token) [B§8]
  - Fallback on model error (32B -> 8B) [B§8]

Migration:
  OllamaClient is retained for backward compatibility.
  New projects should use `llm_backend.create_backend()` + `ModelRouter.from_backend()`.

Bible reference: §8 (Model Router)
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

import httpx

from jarvis.models import Message, MessageRole
from jarvis.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from jarvis.config import JarvisConfig

log = get_logger(__name__)


class OllamaError(Exception):
    """Error communicating with Ollama."""

    def __init__(self, message: str, status_code: int | None = None):
        """Initialisiert die Modell-Konfiguration."""
        super().__init__(message)
        self.status_code = status_code


class OllamaClient:
    """Async HTTP client for the Ollama REST API.

    Supports chat completions (with and without streaming),
    tool calling, and embeddings.
    """

    def __init__(self, config: JarvisConfig) -> None:
        """Initialisiert den Ollama API-Client."""
        self._base_url = config.ollama.base_url.rstrip("/")
        self._timeout = config.ollama.timeout_seconds
        self._keep_alive = config.ollama.keep_alive
        self._client: httpx.AsyncClient | None = None

    async def _ensure_client(self) -> httpx.AsyncClient:
        """Lazy-Initialisierung des HTTP-Clients."""
        if self._client is None or self._client.is_closed:
            # Explicitly disable reading proxy settings from the environment.
            # Without `trust_env=False` httpx may attempt to use SOCKS proxies
            # which require optional dependencies (e.g. socksio) that are not
            # installed in the test environment.
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(
                    connect=10.0,
                    read=float(self._timeout),
                    write=30.0,
                    pool=10.0,
                ),
                limits=httpx.Limits(
                    max_connections=5,
                    max_keepalive_connections=2,
                ),
                trust_env=False,
            )
        return self._client

    async def close(self) -> None:
        """Schließt den HTTP-Client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def is_available(self) -> bool:
        """Prüft ob Ollama erreichbar ist."""
        try:
            client = await self._ensure_client()
            resp = await client.get("/api/tags", timeout=5.0)
            return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException, OSError):
            return False

    async def list_models(self) -> list[str]:
        """Listet alle lokal verfügbaren Modelle."""
        try:
            client = await self._ensure_client()
            resp = await client.get("/api/tags")
            resp.raise_for_status()
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception as exc:
            log.warning("ollama_list_models_failed", error=str(exc))
            return []

    async def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False,
        format_json: bool = False,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Sendet eine Chat-Completion-Anfrage an Ollama.

        Args:
            model: Modellname (z.B. "qwen3:32b")
            messages: Chat-History als Liste von Dicts
            tools: MCP-Tool-Schemas für Function-Calling
            temperature: Kreativitäts-Parameter
            top_p: Nucleus-Sampling-Parameter
            stream: Ob Token-für-Token gestreamt werden soll
            format_json: Ob die Antwort als JSON erzwungen werden soll

        Returns:
            Ollama-Response als Dict mit 'message' Key.

        Raises:
            OllamaError: Bei Kommunikations- oder Server-Fehlern.
        """
        client = await self._ensure_client()

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,  # Non-streaming für diese Methode
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                **(options or {}),
            },
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
                body = resp.text[:500]
                raise OllamaError(
                    f"Ollama HTTP {resp.status_code}: {body}",
                    status_code=resp.status_code,
                )

            result = resp.json()
            duration_ms = int((time.monotonic() - start) * 1000)

            log.debug(
                "ollama_chat_complete",
                model=model,
                duration_ms=duration_ms,
                eval_count=result.get("eval_count", 0),
                has_tool_calls=bool(result.get("message", {}).get("tool_calls")),
            )

            return result  # type: ignore[no-any-return]

        except httpx.TimeoutException as exc:
            duration_ms = int((time.monotonic() - start) * 1000)
            log.error("ollama_timeout", model=model, duration_ms=duration_ms)
            raise OllamaError(f"Ollama Timeout nach {duration_ms}ms") from exc
        except httpx.ConnectError as exc:
            raise OllamaError(f"Ollama nicht erreichbar unter {self._base_url}") from exc

    async def chat_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> AsyncIterator[str]:
        """Streaming Chat-Completion -- Token für Token.

        Yields:
            Einzelne Text-Tokens als Strings.
        """
        client = await self._ensure_client()

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
            },
            "keep_alive": self._keep_alive,
        }

        if tools:
            payload["tools"] = tools

        try:
            async with client.stream("POST", "/api/chat", json=payload) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    raise OllamaError(
                        f"Ollama HTTP {resp.status_code}: {body[:500].decode(errors='replace')}",
                        status_code=resp.status_code,
                    )

                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                        content = chunk.get("message", {}).get("content", "")
                        if content:
                            yield content
                        if chunk.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue

        except httpx.TimeoutException as exc:
            raise OllamaError("Ollama Streaming Timeout") from exc
        except httpx.ConnectError as exc:
            raise OllamaError(f"Ollama nicht erreichbar unter {self._base_url}") from exc

    async def embed(
        self,
        model: str,
        text: str,
    ) -> list[float]:
        """Erzeugt einen Embedding-Vektor für einen Text.

        Args:
            model: Embedding-Modellname (z.B. "nomic-embed-text")
            text: Zu embeddender Text

        Returns:
            Embedding-Vektor als Liste von Floats (768d für nomic).
        """
        client = await self._ensure_client()

        try:
            resp = await client.post(
                "/api/embed",
                json={"model": model, "input": text},
                timeout=30.0,
            )

            if resp.status_code != 200:
                raise OllamaError(
                    f"Embedding fehlgeschlagen: HTTP {resp.status_code}",
                    status_code=resp.status_code,
                )

            data = resp.json()
            embeddings = data.get("embeddings", [])
            if not embeddings:
                raise OllamaError("Keine Embeddings in Ollama-Antwort")

            return embeddings[0]  # type: ignore[no-any-return]

        except httpx.TimeoutException as exc:
            raise OllamaError("Embedding Timeout") from exc

    async def embed_batch(
        self,
        model: str,
        texts: list[str],
    ) -> list[list[float]]:
        """Erzeugt Embeddings für mehrere Texte.

        Args:
            model: Embedding-Modellname
            texts: Liste zu embeddender Texte

        Returns:
            Liste von Embedding-Vektoren.
        """
        client = await self._ensure_client()

        try:
            resp = await client.post(
                "/api/embed",
                json={"model": model, "input": texts},
                timeout=60.0,
            )

            if resp.status_code != 200:
                raise OllamaError(
                    f"Batch-Embedding fehlgeschlagen: HTTP {resp.status_code}",
                    status_code=resp.status_code,
                )

            data = resp.json()
            result: list[list[float]] = data.get("embeddings", [])
            return result

        except httpx.TimeoutException as exc:
            raise OllamaError("Batch-Embedding Timeout") from exc


class ModelRouter:
    """Selects the best model based on task type and complexity. [B§8.2]

    Supports two initialization paths:
      1. Legacy: ModelRouter(config, OllamaClient) -- backward compatible
      2. New:    ModelRouter.from_backend(config, LLMBackend) -- multi-provider
    """

    def __init__(self, config: JarvisConfig, client: OllamaClient) -> None:
        """Initialisiert den Model-Router mit Ollama-Client und Modell-Zuordnung."""
        self._config = config
        self._client = client
        self._backend: Any | None = None  # Optional: LLMBackend-Instanz
        self._available_models: set[str] = set()

    @classmethod
    def from_backend(cls, config: JarvisConfig, backend: Any) -> "ModelRouter":
        """Erstellt einen ModelRouter mit einem LLMBackend statt OllamaClient.

        Args:
            config: Jarvis-Konfiguration.
            backend: LLMBackend-Instanz (aus jarvis.core.llm_backend).

        Returns:
            Konfigurierter ModelRouter.
        """
        # Dummy-OllamaClient erstellen (wird nicht genutzt wenn backend gesetzt)
        dummy_client = OllamaClient(config)
        instance = cls(config, dummy_client)
        instance._backend = backend
        return instance

    @property
    def backend(self) -> Any | None:
        """Gibt das konfigurierte LLMBackend zurück (None wenn legacy-Modus)."""
        return self._backend

    async def initialize(self) -> None:
        """Prüft welche Modelle verfügbar sind.

        Nutzt das LLMBackend wenn vorhanden, sonst den OllamaClient.
        """
        if self._backend is not None:
            models = await self._backend.list_models()
        else:
            models = await self._client.list_models()
        self._available_models = set(models)
        log.info(
            "model_router_initialized",
            available_models=sorted(self._available_models),
        )

    def select_model(
        self,
        task_type: str = "general",
        complexity: str = "medium",
    ) -> str:
        """Wählt das beste verfügbare Modell. [B§8.2]

        Args:
            task_type: Art der Aufgabe (planning, reflection, code, simple_tool_call,
                       summarization, embedding, general)
            complexity: Komplexität (low, medium, high)

        Returns:
            Ollama-Modellname (z.B. "qwen3:32b")
        """
        # Prüfe zunächst, ob ein Override für den gegebenen task_type existiert.
        # Der Schlüssel in model_overrides.skill_models kann den task_type
        # (z. B. "planning", "reflection", "code") überschreiben. So können
        # Anwender in der Konfiguration alternative Modelle für bestimmte
        # Aufgabentypen definieren.
        override = None
        try:
            override = self._config.model_overrides.skill_models.get(task_type)
        except Exception:
            override = None

        if override:
            model_name = override
        else:
            # Direkte Zuordnung nach Aufgabentyp
            match task_type:
                case "planning" | "reflection":
                    model_name = self._config.models.planner.name
                case "code":
                    model_name = self._config.models.coder.name
                case "simple_tool_call" | "summarization":
                    model_name = self._config.models.executor.name
                case "embedding":
                    model_name = self._config.models.embedding.name
                case _:
                    # Komplexität entscheidet
                    if complexity == "high":
                        model_name = self._config.models.planner.name
                    elif complexity == "low":
                        model_name = self._config.models.executor.name
                    else:
                        model_name = self._config.models.planner.name

        # Fallback wenn Modell nicht verfügbar
        if self._available_models and model_name not in self._available_models:
            fallback = self._find_fallback(model_name)
            if fallback:
                log.warning(
                    "model_fallback",
                    requested=model_name,
                    using=fallback,
                )
                return fallback

        return model_name

    def _find_fallback(self, requested: str) -> str | None:
        """Findet ein Fallback-Modell wenn das gewünschte nicht verfügbar ist."""
        # Fallback-Kette: 32B → 8B → was auch immer da ist
        fallback_chain = [
            self._config.models.planner.name,
            self._config.models.executor.name,
        ]
        for fallback in fallback_chain:
            if fallback in self._available_models and fallback != requested:
                return fallback

        # Letzter Versuch: irgendetwas das nicht embedding ist
        for model in self._available_models:
            if "embed" not in model.lower():
                return model

        return None

    def get_model_config(self, model_name: str) -> dict[str, Any]:
        """Gibt die Konfigurations-Parameter für ein Modell zurück."""
        configs = {
            self._config.models.planner.name: self._config.models.planner,
            self._config.models.executor.name: self._config.models.executor,
            self._config.models.coder.name: self._config.models.coder,
            self._config.models.embedding.name: self._config.models.embedding,
        }
        config = configs.get(model_name)
        if config:
            return {
                "temperature": config.temperature,
                "top_p": config.top_p,
                "context_window": config.context_window,
            }
        # Default-Werte für unbekannte Modelle
        return {"temperature": 0.7, "top_p": 0.9, "context_window": 32768}


def messages_to_ollama(messages: list[Message]) -> list[dict[str, Any]]:
    """Konvertiert Jarvis-Messages in Ollama-Format.

    Args:
        messages: Liste von Jarvis-Messages

    Returns:
        Liste von Dicts im Ollama-Chat-Format
    """
    result = []
    for msg in messages:
        entry: dict[str, Any] = {
            "role": msg.role.value,
            "content": msg.content,
        }
        # Tool-Ergebnisse brauchen spezielle Felder
        if msg.role == MessageRole.TOOL and msg.name:
            entry["role"] = "tool"  # Ollama erwartet "tool" nicht "tool"
            # Manche Modelle erwarten Tool-Name
        result.append(entry)
    return result
