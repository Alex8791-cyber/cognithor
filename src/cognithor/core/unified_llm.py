"""Unified LLM Client: Adapter between OllamaClient interface and LLMBackend.

Solves the wiring problem: Planner, Reflector and Gateway all use
`self._ollama.chat()` with Ollama-specific response format.
This adapter provides the same interface but delegates to
the configured LLMBackend.

Usage:
    # Gateway erstellt den Client basierend auf Config:
    client = UnifiedLLMClient.create(config)

    # Planner/Reflector nutzen ihn wie bisher:
    response = await client.chat(model="qwen3:32b", messages=[...])
    text = response.get("message", {}).get("content", "")
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any

from cognithor.core.llm_backend import LLMBackendError, LLMBadRequestError, VLLMNotReadyError
from cognithor.core.model_router import OllamaClient, OllamaError
from cognithor.utils.circuit_breaker import CircuitBreaker, CircuitBreakerOpen, CircuitState
from cognithor.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from cognithor.config import CognithorConfig

log = get_logger(__name__)


class BackendStatus(StrEnum):
    """Public-facing health state surfaced to the UI."""

    OK = "ok"
    DEGRADED = "degraded"


class UnifiedLLMClient:
    """Adapter der das OllamaClient-Interface auf beliebige LLM-Backends mappt.

    Always returns responses in Ollama dict format so that
    Planner/Reflector/etc. don't need to be changed.

    Supports: Ollama (direct), OpenAI, Anthropic (via LLMBackend).
    """

    def __init__(
        self,
        ollama_client: OllamaClient | None,
        backend: Any | None = None,
        config: CognithorConfig | None = None,
        *,
        _breaker_recovery_timeout: float = 60.0,
    ) -> None:
        """Erstellt den unified Client.

        Args:
            ollama_client: Optionaler OllamaClient (nur bei Ollama-Modus oder Fallback).
            backend: Optionales LLMBackend aus llm_backend.py.
                     Wenn None und ollama_client vorhanden, wird direkt OllamaClient genutzt.
            config: CognithorConfig for on-demand per-task backend creation.
            _breaker_recovery_timeout: Seconds until the circuit moves from OPEN to HALF_OPEN.
        """
        self._ollama = ollama_client
        self._backend = backend
        self._config = config
        self._backend_type: str = "ollama"
        self._backend_cache: dict[str, Any] = {}

        if backend is not None:
            self._backend_type = getattr(backend, "backend_type", "unknown")
            if hasattr(self._backend_type, "value"):
                self._backend_type = self._backend_type.value

        self.vllm_breaker = CircuitBreaker(
            name="llm_backend_vllm",
            failure_threshold=3,
            recovery_timeout=_breaker_recovery_timeout,
            half_open_max_calls=1,
            excluded_exceptions=(LLMBadRequestError,),
        )
        self.ollama_breaker = CircuitBreaker(
            name="llm_backend_ollama",
            failure_threshold=3,
            recovery_timeout=_breaker_recovery_timeout,
            half_open_max_calls=1,
            excluded_exceptions=(LLMBadRequestError,),
        )
        self.backend_status: BackendStatus = BackendStatus.OK

    @classmethod
    def create(cls, config: CognithorConfig) -> UnifiedLLMClient:
        """Factory: Erstellt den passenden Client basierend auf der Config.

        Args:
            config: Jarvis-Konfiguration mit llm_backend_type.

        Returns:
            Konfigurierter UnifiedLLMClient.
        """
        backend = None
        ollama_client: OllamaClient | None = None

        if config.llm_backend_type != "ollama":
            try:
                from cognithor.core.llm_backend import create_backend

                backend = create_backend(config)
                log.info(
                    "unified_client_created",
                    backend=config.llm_backend_type,
                )
            except Exception as exc:
                log.error(
                    "llm_backend_creation_failed",
                    backend=config.llm_backend_type,
                    error=str(exc),
                )
                raise OllamaError(
                    f"LLM-Backend '{config.llm_backend_type}' konnte nicht "
                    f"initialisiert werden: {exc}. "
                    f"Bitte API-Key und Konfiguration pruefen."
                ) from exc
        else:
            # Ollama-Modus: OllamaClient erstellen
            ollama_client = OllamaClient(config)

        return cls(ollama_client, backend, config=config)

    # ========================================================================
    # Per-task backend resolution
    # ========================================================================

    def _lookup_backend_for_model(self, model: str) -> str:
        """Check if any ModelConfig.backend is set for the given model name."""
        if self._config is None:
            return ""
        for role in ("planner", "executor", "coder", "coder_fast", "embedding"):
            cfg = getattr(self._config.models, role, None)
            if cfg is not None and cfg.name == model:
                return getattr(cfg, "backend", "") or ""
        return ""

    def _resolve_backend(self, backend_override: str) -> Any | None:
        """Returns the LLMBackend for a per-task override, or None for Ollama.

        Backends are lazily created and cached by provider name so that
        repeated calls for the same provider reuse the connection.
        """
        if not backend_override or backend_override == "ollama":
            return None  # use Ollama path

        # If the override matches the global backend, reuse it
        if self._backend is not None and backend_override == self._backend_type:
            return self._backend

        # Lazy-create and cache
        if backend_override in self._backend_cache:
            return self._backend_cache[backend_override]

        if self._config is None:
            log.warning("per_task_backend_no_config", override=backend_override)
            return self._backend

        try:
            from cognithor.core.llm_backend import create_backend

            # Temporarily override the backend type in a copy
            temp_config = self._config.model_copy(update={"llm_backend_type": backend_override})
            new_backend = create_backend(temp_config)
            self._backend_cache[backend_override] = new_backend
            log.info(
                "per_task_backend_created",
                provider=backend_override,
            )
            return new_backend
        except Exception as exc:
            log.warning(
                "per_task_backend_failed",
                provider=backend_override,
                error=str(exc),
                fallback="global",
            )
            return self._backend

    def _breaker_for(self, backend_type: str) -> CircuitBreaker:
        if backend_type == "vllm":
            return self.vllm_breaker
        return self.ollama_breaker

    def _refresh_status(self) -> None:
        breaker = self._breaker_for(self._backend_type)
        if breaker.state == CircuitState.closed:
            self.backend_status = BackendStatus.OK
        else:
            self.backend_status = BackendStatus.DEGRADED

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
        backend_override: str = "",
        images: list[str] | None = None,
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

        Args:
            backend_override: Per-task backend provider name (e.g. "openai").
                Empty string uses the global backend.

        Raises:
            OllamaError: Bei jedem Backend-Fehler (einheitliche Exception).
        """
        # Context-Window Preflight: check before sending to provider
        if self._config is not None:
            try:
                # Cache model router to avoid re-creating on every call
                if not hasattr(self, "_model_router_cache"):
                    from cognithor.core.model_router import ModelRouter

                    if self._ollama is not None:
                        self._model_router_cache = ModelRouter(self._config, self._ollama)
                    else:
                        self._model_router_cache = None
                if self._model_router_cache is None:
                    raise ImportError("No model router available")
                _model_cfg = self._model_router_cache.get_model_config(model)
                _ctx_window = _model_cfg.get("context_window", 0)
                if _ctx_window > 0:
                    from cognithor.core.preflight import preflight_check

                    _max_out = (options or {}).get("num_predict", 4096)
                    # Don't pass system separately — it's already in messages
                    preflight_check(
                        model,
                        messages,
                        _ctx_window,
                        tools=tools,
                        max_output_tokens=_max_out,
                    )
            except ImportError:
                pass  # preflight not available
            except Exception as exc:
                # ContextWindowExceeded propagates; other errors are non-fatal
                from cognithor.core.preflight import ContextWindowExceeded

                if isinstance(exc, ContextWindowExceeded):
                    raise
                log.debug("preflight_check_failed", error=str(exc))

        # Resolve per-task backend: explicit override > model-config lookup > global
        if not backend_override and self._config is not None:
            backend_override = self._lookup_backend_for_model(model)
        effective_backend = (
            self._resolve_backend(backend_override) if backend_override else self._backend
        )

        if effective_backend is None:
            # Direkt an OllamaClient weiterleiten
            if self._ollama is None:
                _cfg_backend = getattr(self._config, "llm_backend_type", "ollama")
                raise OllamaError(
                    f"Kein LLM-Backend verfuegbar. "
                    f"Backend-Typ: '{_cfg_backend}', Modell: '{model}'. "
                    f"Bitte Backend-Konfiguration und API-Key pruefen."
                )
            breaker = self.ollama_breaker
            try:
                result = await breaker.call(
                    self._ollama.chat(
                        model=model,
                        messages=messages,
                        tools=tools,
                        temperature=temperature,
                        top_p=top_p,
                        stream=stream,
                        format_json=format_json,
                        options=options,
                    )
                )
            finally:
                self._refresh_status()
            return result

        # Via LLMBackend
        is_image_request = bool(images)
        effective_backend_type = getattr(effective_backend, "backend_type", self._backend_type)
        if hasattr(effective_backend_type, "value"):
            effective_backend_type = effective_backend_type.value
        breaker = self._breaker_for(str(effective_backend_type))
        try:
            backend_call_kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "tools": tools,
                "temperature": temperature,
                "top_p": top_p,
                "format_json": format_json,
            }
            if images is not None:
                backend_call_kwargs["images"] = images
            response = await breaker.call(effective_backend.chat(**backend_call_kwargs))
        except LLMBadRequestError:
            self._refresh_status()
            raise
        except (VLLMNotReadyError, CircuitBreakerOpen) as exc:
            self._refresh_status()
            if is_image_request:
                # Images cannot be processed by Ollama fallback — hard error
                if isinstance(exc, CircuitBreakerOpen):
                    raise VLLMNotReadyError(
                        "vLLM offline — cannot process image",
                        recovery_hint="Start vLLM from LLM Backends settings.",
                    ) from exc
                raise
            # Text-only: transparent fallback to Ollama
            if self._ollama is not None:
                log.warning("vllm_fallback_to_ollama", error=str(exc))
                self._refresh_status()
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
            # No Ollama available — wrap and raise
            bt = backend_override or self._backend_type
            raise OllamaError(
                f"LLM-Backend-Fehler ({bt}): {exc}",
                status_code=getattr(exc, "status_code", None),
            ) from exc
        except Exception as exc:
            self._refresh_status()
            # Alle anderen Backend-Fehler als OllamaError wrappen
            # so Planner/Reflector catch blocks keep working
            bt = backend_override or self._backend_type
            raise OllamaError(
                f"LLM-Backend-Fehler ({bt}): {exc}",
                status_code=getattr(exc, "status_code", None),
            ) from exc
        else:
            self._refresh_status()

        # ChatResponse → Ollama-Dict konvertieren
        result_dict: dict[str, Any] = {
            "message": {
                "role": "assistant",
                "content": response.content,
            },
            "model": response.model or model,
            "done": True,
        }

        # Transfer tool calls
        if response.tool_calls:
            result_dict["message"]["tool_calls"] = response.tool_calls

        # Transfer usage info
        if response.usage:
            result_dict["prompt_eval_count"] = response.usage.get("prompt_tokens", 0)
            result_dict["eval_count"] = response.usage.get("completion_tokens", 0)

        return result_dict

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
            if self._ollama is None:
                raise OllamaError("Kein LLM-Backend verfügbar (weder API noch Ollama)")
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
            if self._ollama is None:
                raise OllamaError("Kein LLM-Backend verfügbar für Embeddings")
            vec = await self._ollama.embed(model, text)
            return {"embedding": vec} if not isinstance(vec, dict) else vec

        try:
            response = await self._backend.embed(model, text)
            return {"embedding": response.embedding}
        except (NotImplementedError, LLMBackendError):
            # Backend has no embedding -> Ollama fallback only if available
            if self._ollama is not None:
                log.info("embedding_fallback_to_ollama", backend=self._backend_type)
                vec = await self._ollama.embed(model, text)
                return {"embedding": vec} if not isinstance(vec, dict) else vec
            raise
        except Exception as exc:
            raise OllamaError(f"Embedding-Fehler: {exc}") from exc

    async def batch_embed(self, model: str, texts: list[str]) -> list[dict[str, Any]]:
        """Batch embedding. Uses backend if possible, otherwise OllamaClient."""
        if self._backend is None:
            if self._ollama is None:
                raise OllamaError("Kein LLM-Backend verfügbar für Embeddings")
            vecs = await self._ollama.embed_batch(model, texts)
            return [{"embedding": v} if not isinstance(v, dict) else v for v in vecs]

        # LLMBackend hat kein batch_embed → sequentiell
        results = []
        for text in texts:
            result = await self.embed(model, text)
            results.append(result)
        return results

    # ========================================================================
    # Meta methods (needed by Gateway/ModelRouter)
    # ========================================================================

    async def is_available(self) -> bool:
        """Check whether the LLM backend is reachable."""
        if self._backend is not None:
            try:
                return await self._backend.is_available()
            except Exception:
                return False
        if self._ollama is not None:
            return await self._ollama.is_available()
        return False

    async def list_models(self) -> list[str]:
        """List available models."""
        if self._backend is not None:
            try:
                return await self._backend.list_models()
            except Exception:
                return []
        if self._ollama is not None:
            return await self._ollama.list_models()
        return []

    async def close(self) -> None:
        """Close all connections."""
        if self._backend is not None:
            try:
                await self._backend.close()
            except Exception as exc:
                log.debug(
                    "backend_close_error", error=str(exc)
                )  # Cleanup — failure is non-critical
        if self._ollama is not None:
            await self._ollama.close()

    @property
    def backend_type(self) -> str:
        """Return the active backend type."""
        return self._backend_type

    @property
    def has_embedding_support(self) -> bool:
        """Check whether the active backend supports embeddings.

        Anthropic hat keine Embeddings -- dann wird der Ollama-Fallback genutzt.
        """
        if self._backend_type == "anthropic":
            return False  # Ollama-Fallback wird in embed() automatisch genutzt
        return True
