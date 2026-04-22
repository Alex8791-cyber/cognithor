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

import contextvars
import dataclasses
import json
import time
from typing import TYPE_CHECKING, Any

import httpx

from cognithor.models import Message, MessageRole
from cognithor.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from cognithor.config import CognithorConfig

log = get_logger(__name__)


@dataclasses.dataclass(frozen=True)
class ConciergeProfile:
    """Urgency-based model selection profile.

    Each profile maps a human-friendly urgency label to a concrete
    Ollama model and a short description.
    """

    name: str
    model: str
    description: str


# Default concierge profiles using local Ollama models.
# The ``asap`` profile uses the largest model for maximum quality,
# ``balanced`` trades off speed vs quality, and ``no_hurry`` picks the
# smallest model for fast / low-resource responses.
CONCIERGE_PROFILES: dict[str, ConciergeProfile] = {
    "asap": ConciergeProfile(
        name="asap",
        model="qwen3:32b",
        description="Maximum quality, largest model (planner-class)",
    ),
    "balanced": ConciergeProfile(
        name="balanced",
        model="qwen3:8b",
        description="Good balance of speed and quality (executor-class)",
    ),
    "no_hurry": ConciergeProfile(
        name="no_hurry",
        model="qwen3:1.7b",
        description="Fastest response, smallest model",
    ),
}

# Per-task urgency override using ContextVar for concurrency safety.
_urgency_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("_urgency", default=None)

# Per-task coding override using ContextVar for concurrency safety.
# Each async task (asyncio.Task / contextvars.copy_context()) gets its own
# value, preventing one request's coding override from leaking into another.
_coding_override_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_coding_override", default=None
)


def _prepare_image_payload(images: list[str]) -> list[str]:
    """Normalize a mixed list of image paths + raw base64 strings.

    Each entry may be:
      - An absolute or relative filesystem path to an image file
      - A string already in base64 encoding (identified by being
        decodable as valid base64 and longer than 256 characters).

    Returns a list of base64 strings suitable for Ollama's
    ``messages[i].images`` field. Unreadable entries are skipped with a
    warning so one bad attachment never tanks a whole turn.
    """
    import base64 as _b64
    from pathlib import Path as _Path

    out: list[str] = []
    for entry in images:
        if not entry:
            continue
        # Heuristic: does this look like an already-encoded blob?
        # base64 strings don't contain path separators or dots in the
        # traditional file-path sense, and are typically very long.
        stripped = entry.strip()
        if len(stripped) > 256 and "/" not in stripped and "\\" not in stripped:
            try:
                _b64.b64decode(stripped, validate=True)
                out.append(stripped)
                continue
            except Exception:
                pass  # Not base64 — fall through to path handling.
        try:
            raw = _Path(entry).read_bytes()
            out.append(_b64.b64encode(raw).decode("ascii"))
        except Exception as exc:
            log.warning("image_payload_unreadable", entry=entry, error=str(exc))
    return out


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

    def __init__(self, config: CognithorConfig) -> None:
        """Initialisiert den Ollama API-Client."""
        self._base_url = config.ollama.base_url.rstrip("/")
        self._timeout = config.ollama.timeout_seconds
        self._keep_alive = config.ollama.keep_alive
        self._client: httpx.AsyncClient | None = None

        # PII redactor — opt-in via config.security.pii_redactor.enabled.
        # Instantiate once here so the regex patterns compile exactly
        # once per OllamaClient lifetime.
        self._pii_redactor = None
        pii_cfg = config.security.pii_redactor
        if pii_cfg.enabled and pii_cfg.categories:
            from cognithor.security.pii_redactor import PIIRedactor

            self._pii_redactor = PIIRedactor(
                categories=pii_cfg.categories,
                replacement_template=pii_cfg.replacement_template,
            )
            self._pii_log_redactions = pii_cfg.log_redactions
        else:
            self._pii_log_redactions = False

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
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def is_available(self) -> bool:
        """Check whether Ollama is reachable."""
        try:
            client = await self._ensure_client()
            resp = await client.get("/api/tags", timeout=5.0)
            return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException, OSError):
            return False

    async def list_models(self) -> list[str]:
        """List all locally available models."""
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
        images: list[str] | None = None,
    ) -> dict[str, Any]:
        """Sendet eine Chat-Completion-Anfrage an Ollama.

        Args:
            model: Modellname (z.B. "qwen3:32b")
            messages: Chat-History als Liste von Dicts
            tools: MCP tool schemas for function calling
            temperature: Creativity parameter
            top_p: Nucleus-Sampling-Parameter
            stream: Whether to stream token by token
            format_json: Ob die Antwort als JSON erzwungen werden soll
            images: Optional list of image file paths OR pre-encoded
                base64 strings. Attached to the LAST user message so the
                VLM (e.g. qwen3.6:27b) can see them. Ollama's chat API
                expects ``messages[i].images = [<base64>, ...]``.

        Returns:
            Ollama-Response als Dict mit 'message' Key.

        Raises:
            OllamaError: Bei Kommunikations- oder Server-Fehlern.
        """
        client = await self._ensure_client()

        # Attach images to the most-recent user message so the VLM can see
        # them. This mutates a *copy* to keep the caller's list intact.
        if images:
            encoded = _prepare_image_payload(images)
            if encoded:
                new_messages: list[dict[str, Any]] = [dict(m) for m in messages]
                for i in range(len(new_messages) - 1, -1, -1):
                    if new_messages[i].get("role") == "user":
                        existing = list(new_messages[i].get("images") or [])
                        new_messages[i] = {
                            **new_messages[i],
                            "images": [*existing, *encoded],
                        }
                        break
                else:
                    # No user message yet — create one so the image has a home.
                    new_messages.append({"role": "user", "content": "", "images": list(encoded)})
                messages = new_messages

        # Strip PII from outbound messages if redactor is enabled.
        if self._pii_redactor is not None:
            messages, matches = self._pii_redactor.redact_messages(messages)
            if matches and self._pii_log_redactions:
                by_cat: dict[str, int] = {}
                for m in matches:
                    by_cat[m.category] = by_cat.get(m.category, 0) + 1
                log.info(
                    "pii_redacted_in_chat",
                    model=model,
                    total=len(matches),
                    by_category=by_cat,
                )

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,  # Non-streaming for this method
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
                if resp.status_code == 404:
                    raise OllamaError(
                        f"Model '{model}' not found. Please download it first: ollama pull {model}",
                        status_code=404,
                    )
                if resp.status_code == 429:
                    raise OllamaError(
                        "Service is currently overloaded (rate limit). "
                        "Please wait a moment and try again.",
                        status_code=429,
                    )
                raise OllamaError(
                    f"Ollama HTTP {resp.status_code}",
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
        """Streaming chat completion -- token by token.

        Yields:
            Einzelne Text-Tokens als Strings.
        """
        client = await self._ensure_client()

        # Mirror the same PII redaction as the non-streaming path.
        if self._pii_redactor is not None:
            messages, matches = self._pii_redactor.redact_messages(messages)
            if matches and self._pii_log_redactions:
                by_cat: dict[str, int] = {}
                for m in matches:
                    by_cat[m.category] = by_cat.get(m.category, 0) + 1
                log.info(
                    "pii_redacted_in_chat_stream",
                    model=model,
                    total=len(matches),
                    by_category=by_cat,
                )

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
        """Generate an embedding vector for a text.

        Args:
            model: Embedding-Modellname (z.B. "nomic-embed-text")
            text: Zu embeddender Text

        Returns:
            Embedding vector as list of floats (768d for nomic).
        """
        client = await self._ensure_client()

        try:
            resp = await client.post(
                "/api/embed",
                json={"model": model, "input": text},
                timeout=float(self._timeout),
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
        """Generate embeddings for multiple texts.

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
                timeout=float(self._timeout),
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

    def __init__(self, config: CognithorConfig, client: OllamaClient) -> None:
        """Initialisiert den Model-Router mit Ollama-Client und Modell-Zuordnung."""
        self._config = config
        self._client = client
        self._backend: Any | None = None  # Optional: LLMBackend-Instanz
        self._available_models: set[str] = set()
        self._coding_override: str | None = None

    @classmethod
    def from_backend(cls, config: CognithorConfig, backend: Any) -> ModelRouter:
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
        """Return the configured LLMBackend (None in legacy mode)."""
        return self._backend

    def set_coding_override(self, model_name: str) -> None:
        """Setzt einen temporaeren Modell-Override fuer den gesamten PGE-Zyklus.

        Wird vom Gateway gesetzt wenn eine Coding-Aufgabe erkannt wird.
        Alle select_model()-Aufrufe (planning, reflection, etc.) liefern
        dann das Coding-Modell zurueck.

        Uses a ContextVar so concurrent async tasks each get their own
        override value, preventing cross-request contamination.
        """
        _coding_override_var.set(model_name)
        # Keep instance attribute as fallback for legacy / non-async callers.
        self._coding_override = model_name
        log.info("coding_override_set", model=model_name)

    def clear_coding_override(self) -> None:
        """Entfernt den Coding-Override nach dem PGE-Zyklus."""
        current = _coding_override_var.get()
        if current:
            log.info("coding_override_cleared", model=current)
        _coding_override_var.set(None)
        # Keep instance attribute in sync for legacy callers.
        self._coding_override = None

    def set_urgency(self, urgency: str) -> None:
        """Set the concierge urgency level for model selection.

        Valid values: ``"asap"``, ``"balanced"``, ``"no_hurry"``.
        When set, :meth:`select_model` will prefer the profile's model
        (except for embeddings and when a coding override is active).

        Uses a ContextVar for concurrency safety, just like the coding
        override.
        """
        if urgency not in CONCIERGE_PROFILES:
            raise ValueError(
                f"Unknown urgency '{urgency}'. Valid options: {sorted(CONCIERGE_PROFILES)}"
            )
        _urgency_var.set(urgency)
        log.info("concierge_urgency_set", urgency=urgency)

    def get_urgency(self) -> str | None:
        """Return the current urgency level, or ``None`` if not set."""
        return _urgency_var.get()

    def clear_urgency(self) -> None:
        """Remove the urgency override."""
        current = _urgency_var.get()
        if current:
            log.info("concierge_urgency_cleared", urgency=current)
        _urgency_var.set(None)

    @staticmethod
    def get_concierge_profile(urgency: str) -> ConciergeProfile | None:
        """Look up a concierge profile by urgency name."""
        return CONCIERGE_PROFILES.get(urgency)

    async def initialize(self) -> None:
        """Check which models are available.

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
        """Select the best available model. [B§8.2]

        Args:
            task_type: Art der Aufgabe (planning, reflection, code, simple_tool_call,
                       summarization, embedding, general)
            complexity: Complexity (low, medium, high)

        Returns:
            Ollama-Modellname (z.B. "qwen3:32b")
        """
        # Coding-Override: Wenn gesetzt, wird fuer ALLE task_types das
        # Coding-Modell verwendet (gesamter PGE-Zyklus bei Coding-Aufgaben).
        # Ausnahme: Embeddings bleiben beim Embedding-Modell.
        #
        # Uses the ContextVar exclusively for concurrency safety.
        # The ContextVar is set by set_coding_override() and is isolated
        # per async task, so concurrent requests cannot interfere.
        # The instance attribute self._coding_override is still maintained
        # for backwards compatibility (external readers) but is NOT used
        # for model selection to prevent cross-task contamination.
        _effective_override = _coding_override_var.get()
        if _effective_override and task_type != "embedding":
            return _effective_override

        # Concierge routing: urgency-based model selection.
        # When an urgency level is set, use the profile's model for all
        # non-embedding tasks. Coding override takes precedence (above).
        _effective_urgency = _urgency_var.get()
        if _effective_urgency and task_type != "embedding":
            profile = CONCIERGE_PROFILES.get(_effective_urgency)
            if profile:
                model_name = profile.model
                # Still apply availability fallback
                if self._available_models and model_name not in self._available_models:
                    fallback = self._find_fallback(model_name)
                    if fallback:
                        log.warning(
                            "concierge_model_fallback",
                            urgency=_effective_urgency,
                            requested=model_name,
                            using=fallback,
                        )
                        return fallback
                return model_name

        # Pruefe zunaechst, ob ein Override fuer den gegebenen task_type existiert.
        # Der Schluessel in model_overrides.skill_models kann den task_type
        # (z. B. "planning", "reflection", "code") ueberschreiben. So koennen
        # Anwender in der Konfiguration alternative Modelle fuer bestimmte
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
                    if complexity == "high":
                        model_name = self._config.models.coder.name
                    else:
                        model_name = self._config.models.coder_fast.name
                case "simple_tool_call" | "summarization":
                    model_name = self._config.models.executor.name
                case "embedding":
                    model_name = self._config.models.embedding.name
                case _:
                    # Complexity decides
                    if complexity == "high":
                        model_name = self._config.models.planner.name
                    elif complexity == "low":
                        model_name = self._config.models.executor.name
                    else:
                        model_name = self._config.models.planner.name

        # Fallback if model not available
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
        """Find a fallback model when the desired one is not available."""
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

    def get_backend_for_task(
        self,
        task_type: str = "general",
        complexity: str = "medium",
    ) -> str:
        """Returns the per-task backend override, or empty string for global backend.

        This enables per-role backend routing: e.g. use Anthropic for planning
        while keeping Ollama for execution.  The ``backend`` field on each
        ``ModelConfig`` drives this.  An empty string means "use the global
        llm_backend_type".
        """
        match task_type:
            case "planning" | "reflection":
                cfg = self._config.models.planner
            case "code":
                cfg = (
                    self._config.models.coder
                    if complexity == "high"
                    else self._config.models.coder_fast
                )
            case "simple_tool_call" | "summarization":
                cfg = self._config.models.executor
            case "embedding":
                cfg = self._config.models.embedding
            case _:
                cfg = (
                    self._config.models.planner
                    if complexity != "low"
                    else self._config.models.executor
                )
        return getattr(cfg, "backend", "") or ""

    def get_model_config(self, model_name: str) -> dict[str, Any]:
        """Return the configuration parameters for a model."""
        configs = {
            self._config.models.planner.name: self._config.models.planner,
            self._config.models.executor.name: self._config.models.executor,
            self._config.models.coder.name: self._config.models.coder,
            self._config.models.coder_fast.name: self._config.models.coder_fast,
            self._config.models.embedding.name: self._config.models.embedding,
        }
        config = configs.get(model_name)
        if config:
            return {
                "temperature": config.temperature,
                "top_p": config.top_p,
                "context_window": config.context_window,
            }
        # Default values for unknown models
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
