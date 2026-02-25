"""Unified LLM Client: Adapter zwischen OllamaClient-Interface und LLMBackend.

Löst das Verdrahtungsproblem: Planner, Reflector und Gateway nutzen
alle `self._ollama.chat()` mit Ollama-spezifischem Response-Format.
Dieser Adapter stellt dasselbe Interface bereit, leitet aber an
das konfigurierte LLMBackend weiter.

Usage:
    # Gateway erstellt den Client basierend auf Config:
    client = UnifiedLLMClient.create(config)

    # Planner/Reflector nutzen ihn wie bisher:
    response = await client.chat(model="qwen3:32b", messages=[...])
    text = response.get("message", {}).get("content", "")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from jarvis.core.llm_backend import LLMBackendError
from jarvis.core.model_router import OllamaClient, OllamaError
from jarvis.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from jarvis.config import JarvisConfig

log = get_logger(__name__)


class UnifiedLLMClient:
    """Adapter der das OllamaClient-Interface auf beliebige LLM-Backends mappt.

    Gibt Responses immer im Ollama-Dict-Format zurück, damit
    Planner/Reflector/etc. nicht geändert werden müssen.

    Unterstützt: Ollama (direkt), OpenAI, Anthropic (via LLMBackend).
    """

    def __init__(
        self,
        ollama_client: OllamaClient,
        backend: Any | None = None,
    ) -> None:
        """Erstellt den unified Client.

        Args:
            ollama_client: Fallback OllamaClient (immer vorhanden).
            backend: Optionales LLMBackend aus llm_backend.py.
                     Wenn None, wird direkt OllamaClient genutzt.
        """
        self._ollama = ollama_client
        self._backend = backend
        self._backend_type: str = "ollama"

        if backend is not None:
            self._backend_type = getattr(backend, "backend_type", "unknown")
            if hasattr(self._backend_type, "value"):
                self._backend_type = self._backend_type.value

    @classmethod
    def create(cls, config: "JarvisConfig") -> "UnifiedLLMClient":
        """Factory: Erstellt den passenden Client basierend auf der Config.

        Args:
            config: Jarvis-Konfiguration mit llm_backend_type.

        Returns:
            Konfigurierter UnifiedLLMClient.
        """
        ollama_client = OllamaClient(config)

        backend = None
        if config.llm_backend_type != "ollama":
            try:
                from jarvis.core.llm_backend import create_backend

                backend = create_backend(config)
                log.info(
                    "unified_client_created",
                    backend=config.llm_backend_type,
                )
            except Exception as exc:
                log.warning(
                    "llm_backend_creation_failed",
                    backend=config.llm_backend_type,
                    error=str(exc),
                    fallback="ollama",
                )
                backend = None

        return cls(ollama_client, backend)

    # ========================================================================
    # Chat (Hauptmethode -- von Planner/Reflector aufgerufen)
    # ========================================================================

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
        """Chat-Completion im Ollama-Response-Format.

        Leitet an das konfigurierte Backend weiter und konvertiert
        die Antwort in das Ollama-Dict-Format:

            {
                "message": {
                    "role": "assistant",
                    "content": "...",
                    "tool_calls": [...]
                },
                "model": "...",
                "done": true
            }

        Raises:
            OllamaError: Bei jedem Backend-Fehler (einheitliche Exception).
        """
        if self._backend is None:
            # Direkt an OllamaClient weiterleiten
            return await self._ollama.chat(
                model=model,
                messages=messages,
                tools=tools,
                temperature=temperature,
                top_p=top_p,
                stream=stream,
                format_json=format_json,
                options=options,
            )

        # Via LLMBackend
        try:
            response = await self._backend.chat(
                model=model,
                messages=messages,
                tools=tools,
                temperature=temperature,
                top_p=top_p,
                format_json=format_json,
            )

            # ChatResponse → Ollama-Dict konvertieren
            result: dict[str, Any] = {
                "message": {
                    "role": "assistant",
                    "content": response.content,
                },
                "model": response.model or model,
                "done": True,
            }

            # Tool-Calls übernehmen
            if response.tool_calls:
                result["message"]["tool_calls"] = response.tool_calls

            # Usage-Info übernehmen
            if response.usage:
                result["prompt_eval_count"] = response.usage.get("prompt_tokens", 0)
                result["eval_count"] = response.usage.get("completion_tokens", 0)

            return result

        except Exception as exc:
            # Alle Backend-Fehler als OllamaError wrappen
            # damit Planner/Reflector catch-Blöcke weiter funktionieren
            raise OllamaError(
                f"LLM-Backend-Fehler ({self._backend_type}): {exc}",
                status_code=getattr(exc, "status_code", None),
            ) from exc

    # ========================================================================
    # Chat-Streaming
    # ========================================================================

    async def chat_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> AsyncIterator[dict[str, Any]]:
        """Streaming-Chat im Ollama-Chunk-Format.

        Yields:
            Dicts im Format: {"message": {"content": "token"}, "done": false}
        """
        if self._backend is None:
            async for token in self._ollama.chat_stream(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
            ):
                yield {
                    "message": {"role": "assistant", "content": token},
                    "done": False,
                }
            yield {"message": {"role": "assistant", "content": ""}, "done": True}
            return

        try:
            async for token in self._backend.chat_stream(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
            ):
                yield {
                    "message": {"role": "assistant", "content": token},
                    "done": False,
                }

            # End-Marker
            yield {"message": {"role": "assistant", "content": ""}, "done": True}

        except Exception as exc:
            raise OllamaError(
                f"LLM-Stream-Fehler ({self._backend_type}): {exc}",
            ) from exc

    # ========================================================================
    # Embeddings
    # ========================================================================

    async def embed(self, model: str, text: str) -> dict[str, Any]:
        """Embedding im Ollama-Format: {"embedding": [0.1, 0.2, ...]}."""
        if self._backend is None:
            vec = await self._ollama.embed(model, text)
            return {"embedding": vec} if not isinstance(vec, dict) else vec

        try:
            response = await self._backend.embed(model, text)
            return {"embedding": response.embedding}
        except (NotImplementedError, LLMBackendError):
            # Anthropic hat kein Embedding → Ollama-Fallback
            log.info("embedding_fallback_to_ollama", backend=self._backend_type)
            vec = await self._ollama.embed(model, text)
            return {"embedding": vec} if not isinstance(vec, dict) else vec
        except Exception as exc:
            raise OllamaError(f"Embedding-Fehler: {exc}") from exc

    async def batch_embed(self, model: str, texts: list[str]) -> list[dict[str, Any]]:
        """Batch-Embedding. Nutzt Backend wenn möglich, sonst OllamaClient."""
        if self._backend is None:
            vecs = await self._ollama.embed_batch(model, texts)
            return [{"embedding": v} if not isinstance(v, dict) else v for v in vecs]

        # LLMBackend hat kein batch_embed → sequentiell
        results = []
        for text in texts:
            result = await self.embed(model, text)
            results.append(result)
        return results

    # ========================================================================
    # Meta-Methoden (von Gateway/ModelRouter benötigt)
    # ========================================================================

    async def is_available(self) -> bool:
        """Prüft ob das LLM-Backend erreichbar ist."""
        if self._backend is not None:
            try:
                return await self._backend.is_available()
            except Exception:
                return False
        return await self._ollama.is_available()

    async def list_models(self) -> list[str]:
        """Listet verfügbare Modelle."""
        if self._backend is not None:
            try:
                return await self._backend.list_models()
            except Exception:
                return []
        return await self._ollama.list_models()

    async def close(self) -> None:
        """Schließt alle Verbindungen."""
        if self._backend is not None:
            try:
                await self._backend.close()
            except Exception:
                pass
        await self._ollama.close()

    @property
    def backend_type(self) -> str:
        """Gibt den aktiven Backend-Typ zurück."""
        return self._backend_type

    @property
    def has_embedding_support(self) -> bool:
        """Prüft ob das aktive Backend Embeddings unterstützt.

        Anthropic hat keine Embeddings -- dann wird der Ollama-Fallback genutzt.
        """
        if self._backend_type == "anthropic":
            return False  # Ollama-Fallback wird in embed() automatisch genutzt
        return True
