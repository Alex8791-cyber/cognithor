"""Embedding-Client mit Provider-Strategie und Cache. [B§4.7, B§12]

Generiert Embeddings via konfigurierbarem Provider (Ollama, OpenAI, Gemini, etc.).
Nutzt Content-Hash als Cache-Key → gleicher Text = kein neuer API-Call.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from jarvis.config import JarvisConfig

logger = logging.getLogger("jarvis.memory.embeddings")

# Default: nomic-embed-text via Ollama
DEFAULT_MODEL = "nomic-embed-text"
DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_DIMENSIONS = 768


@dataclass
class EmbeddingResult:
    """Ergebnis einer Embedding-Berechnung."""

    vector: list[float]
    model: str
    dimensions: int
    cached: bool = False


@dataclass
class EmbeddingStats:
    """Statistiken über Embedding-Operationen."""

    total_requests: int = 0
    cache_hits: int = 0
    api_calls: int = 0
    errors: int = 0

    @property
    def cache_hit_rate(self) -> float:
        """Berechnet die Cache-Hit-Rate (0.0--1.0)."""
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests


# ============================================================================
# EmbeddingProvider ABC + Implementierungen
# ============================================================================


class EmbeddingProvider(ABC):
    """Abstrakte Basisklasse für Embedding-Provider.

    Kapselt die HTTP-Logik für verschiedene Embedding-APIs.
    Cache, LRU und Stats bleiben in EmbeddingClient.
    """

    @abstractmethod
    async def embed_single(self, model: str, text: str) -> list[float]:
        """Erzeugt einen einzelnen Embedding-Vektor."""
        ...

    @abstractmethod
    async def embed_batch_raw(self, model: str, texts: list[str]) -> list[list[float]]:
        """Erzeugt Embeddings für eine Liste von Texten."""
        ...

    async def close(self) -> None:
        """Schließt den HTTP-Client (falls vorhanden)."""


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Embedding-Provider für Ollama (POST /api/embed)."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or getattr(self._client, "is_closed", False):
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout,
                trust_env=False,
            )
        return self._client

    async def embed_single(self, model: str, text: str) -> list[float]:
        client = await self._get_client()
        resp = await client.post(
            "/api/embed",
            json={"model": model, "input": text},
        )
        resp.raise_for_status()
        data = resp.json()
        embeddings = data.get("embeddings", [])
        if not embeddings:
            raise ValueError("Keine Embeddings in Ollama-Antwort")
        return embeddings[0]

    async def embed_batch_raw(self, model: str, texts: list[str]) -> list[list[float]]:
        client = await self._get_client()
        resp = await client.post(
            "/api/embed",
            json={"model": model, "input": texts},
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("embeddings", [])

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None


class OpenAICompatibleEmbeddingProvider(EmbeddingProvider):
    """Embedding-Provider für OpenAI-kompatible APIs (POST /embeddings).

    Funktioniert mit: OpenAI, Mistral, GitHub Models, Bedrock, Together, etc.
    """

    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1", timeout: float = 30.0) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or getattr(self._client, "is_closed", False):
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout,
                headers={"Authorization": f"Bearer {self._api_key}"},
                trust_env=False,
            )
        return self._client

    async def embed_single(self, model: str, text: str) -> list[float]:
        client = await self._get_client()
        resp = await client.post(
            "/embeddings",
            json={"model": model, "input": text},
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("data", [{}])[0].get("embedding", [])

    async def embed_batch_raw(self, model: str, texts: list[str]) -> list[list[float]]:
        client = await self._get_client()
        resp = await client.post(
            "/embeddings",
            json={"model": model, "input": texts},
        )
        resp.raise_for_status()
        data = resp.json()
        # OpenAI gibt data[] als Liste zurück, sortiert nach index
        items = sorted(data.get("data", []), key=lambda x: x.get("index", 0))
        return [item.get("embedding", []) for item in items]

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Embedding-Provider für Google Gemini (embedContent REST API)."""

    API_URL = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, api_key: str, timeout: float = 30.0) -> None:
        self._api_key = api_key
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or getattr(self._client, "is_closed", False):
            self._client = httpx.AsyncClient(
                timeout=self._timeout,
                trust_env=False,
            )
        return self._client

    async def embed_single(self, model: str, text: str) -> list[float]:
        client = await self._get_client()
        url = f"{self.API_URL}/models/{model}:embedContent?key={self._api_key}"
        resp = await client.post(
            url,
            json={"content": {"parts": [{"text": text}]}},
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("embedding", {}).get("values", [])

    async def embed_batch_raw(self, model: str, texts: list[str]) -> list[list[float]]:
        client = await self._get_client()
        url = f"{self.API_URL}/models/{model}:batchEmbedContents?key={self._api_key}"
        requests = [
            {"model": f"models/{model}", "content": {"parts": [{"text": t}]}}
            for t in texts
        ]
        resp = await client.post(url, json={"requests": requests})
        resp.raise_for_status()
        data = resp.json()
        return [
            emb.get("values", [])
            for emb in data.get("embeddings", [])
        ]

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None


class NullEmbeddingProvider(EmbeddingProvider):
    """Dummy-Provider für Backends ohne Embedding-API.

    Wirft NotImplementedError — die Vektor-Suche wird deaktiviert,
    BM25+Graph bleiben funktional (search.py fängt alle Exceptions ab).
    """

    def __init__(self, backend_name: str = "unknown") -> None:
        self._backend_name = backend_name

    async def embed_single(self, model: str, text: str) -> list[float]:
        raise NotImplementedError(
            f"Backend '{self._backend_name}' bietet keine Embedding-API. "
            f"Vektor-Suche deaktiviert, BM25+Graph weiterhin aktiv."
        )

    async def embed_batch_raw(self, model: str, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError(
            f"Backend '{self._backend_name}' bietet keine Embedding-API. "
            f"Vektor-Suche deaktiviert, BM25+Graph weiterhin aktiv."
        )


# ============================================================================
# Factory
# ============================================================================

# Backends die den OpenAI-kompatiblen /embeddings-Endpoint unterstützen
_OPENAI_COMPAT_EMBEDDING_BACKENDS = {"openai", "mistral", "github", "bedrock"}

# Backends ohne eigene Embedding-API
_NO_EMBEDDING_BACKENDS = {"anthropic", "groq", "deepseek", "together", "openrouter", "xai", "cerebras", "huggingface", "moonshot"}


def _get_api_key_and_url(config: JarvisConfig, backend: str) -> tuple[str, str]:
    """Gibt (api_key, base_url) für einen Backend-Typ zurück."""
    from jarvis.config import _PROVIDER_BASE_URLS

    key_map: dict[str, str] = {
        "openai": "openai_api_key",
        "mistral": "mistral_api_key",
        "github": "github_api_key",
        "bedrock": "bedrock_api_key",
        "groq": "groq_api_key",
        "deepseek": "deepseek_api_key",
        "together": "together_api_key",
        "openrouter": "openrouter_api_key",
        "xai": "xai_api_key",
        "cerebras": "cerebras_api_key",
        "huggingface": "huggingface_api_key",
        "moonshot": "moonshot_api_key",
    }

    api_key = getattr(config, key_map.get(backend, ""), "")

    if backend == "openai":
        base_url = getattr(config, "openai_base_url", "https://api.openai.com/v1")
    else:
        base_url = _PROVIDER_BASE_URLS.get(backend, "")

    return api_key, base_url


def create_embedding_provider(config: JarvisConfig) -> EmbeddingProvider:
    """Factory: Erstellt den passenden EmbeddingProvider basierend auf der Config.

    Returns:
        Konfigurierter EmbeddingProvider für das aktive Backend.
    """
    backend = config.llm_backend_type

    if backend == "ollama":
        return OllamaEmbeddingProvider(
            base_url=config.ollama.base_url,
            timeout=float(config.ollama.timeout_seconds),
        )

    if backend == "gemini":
        return GeminiEmbeddingProvider(
            api_key=config.gemini_api_key,
        )

    if backend in _OPENAI_COMPAT_EMBEDDING_BACKENDS:
        api_key, base_url = _get_api_key_and_url(config, backend)
        return OpenAICompatibleEmbeddingProvider(
            api_key=api_key,
            base_url=base_url,
        )

    # Backends ohne Embedding-API: Fallback auf OpenAI wenn Key vorhanden
    if backend in _NO_EMBEDDING_BACKENDS:
        if config.openai_api_key:
            logger.info(
                "Backend '%s' hat keine Embedding-API, nutze OpenAI-Embeddings als Fallback",
                backend,
            )
            return OpenAICompatibleEmbeddingProvider(
                api_key=config.openai_api_key,
                base_url=getattr(config, "openai_base_url", "https://api.openai.com/v1"),
            )
        logger.warning(
            "Backend '%s' hat keine Embedding-API und kein OpenAI-Key gesetzt. "
            "Vektor-Suche deaktiviert, BM25+Graph weiterhin aktiv.",
            backend,
        )
        return NullEmbeddingProvider(backend_name=backend)

    # Unbekanntes Backend → Ollama-Fallback
    logger.warning("Unbekanntes Backend '%s', nutze Ollama-Embedding-Provider", backend)
    return OllamaEmbeddingProvider(
        base_url=config.ollama.base_url,
        timeout=float(config.ollama.timeout_seconds),
    )


# ============================================================================
# EmbeddingClient (Cache + Stats, delegiert an Provider)
# ============================================================================


class EmbeddingClient:
    """Async Embedding-Client mit In-Memory-Cache.

    Delegiert API-Calls an einen konfigurierbaren EmbeddingProvider.
    Cache basiert auf Content-Hash (SHA-256).
    Uses an LRU-bounded OrderedDict to prevent unbounded memory growth.
    """

    _MAX_CACHE_SIZE: int = 50_000

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        dimensions: int = DEFAULT_DIMENSIONS,
        timeout: float = 30.0,
        provider: EmbeddingProvider | None = None,
    ) -> None:
        """Initialisiert den Embedding-Client mit Modell und Cache.

        Args:
            model: Name des Embedding-Modells.
            base_url: Base-URL (nur für Rückwärtskompatibilität, wird ignoriert wenn provider gesetzt).
            dimensions: Erwartete Vektor-Dimensionalität.
            timeout: Request-Timeout in Sekunden.
            provider: Optionaler EmbeddingProvider. Default: OllamaEmbeddingProvider.
        """
        self._model = model
        self._dimensions = dimensions
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._stats = EmbeddingStats()
        # Provider: explizit gesetzt oder Ollama-Default (Rückwärtskompatibilität)
        self._provider = provider or OllamaEmbeddingProvider(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
        )

    @property
    def model(self) -> str:
        """Name des Embedding-Modells."""
        return self._model

    @property
    def dimensions(self) -> int:
        """Dimensionalität der Embedding-Vektoren."""
        return self._dimensions

    @property
    def stats(self) -> EmbeddingStats:
        """Gibt die aktuellen Cache-Statistiken zurück."""
        return self._stats

    def _cache_put(self, key: str, vector: list[float]) -> None:
        """Insert or update a single cache entry, evicting the oldest if full."""
        if key in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._cache[key] = vector
        else:
            self._cache[key] = vector
            while len(self._cache) > self._MAX_CACHE_SIZE:
                self._cache.popitem(last=False)  # evict oldest (FIFO/LRU)

    def load_cache(self, entries: dict[str, list[float]]) -> int:
        """Lädt Embeddings in den In-Memory-Cache.

        Args:
            entries: {content_hash: vector} Dict.

        Returns:
            Anzahl geladener Einträge.
        """
        for key, vector in entries.items():
            self._cache_put(key, vector)
        return len(entries)

    def get_cached(self, content_hash: str) -> list[float] | None:
        """Prüft ob ein Embedding im Cache liegt.

        Promotes the entry to most-recently-used on access (LRU).
        """
        if content_hash in self._cache:
            self._cache.move_to_end(content_hash)
            return self._cache[content_hash]
        return None

    async def embed_text(self, text: str, content_hash: str = "") -> EmbeddingResult:
        """Generiert ein Embedding für einen Text.

        Args:
            text: Der zu embedende Text.
            content_hash: Optional Cache-Key.

        Returns:
            EmbeddingResult mit Vektor.
        """
        self._stats.total_requests += 1

        # Cache-Check (promote to most-recently-used on hit)
        if content_hash and content_hash in self._cache:
            self._stats.cache_hits += 1
            self._cache.move_to_end(content_hash)
            return EmbeddingResult(
                vector=self._cache[content_hash],
                model=self._model,
                dimensions=self._dimensions,
                cached=True,
            )

        # API-Call via Provider
        self._stats.api_calls += 1
        try:
            vector = await self._provider.embed_single(self._model, text)

            # Cache speichern (bounded LRU)
            if content_hash:
                self._cache_put(content_hash, vector)

            return EmbeddingResult(
                vector=vector,
                model=self._model,
                dimensions=len(vector),
                cached=False,
            )

        except (httpx.HTTPError, ValueError, KeyError, NotImplementedError) as e:
            self._stats.errors += 1
            logger.error("Embedding-Fehler für '%s...': %s", text[:50], e)
            raise

    async def embed_batch(
        self,
        texts: list[str],
        content_hashes: list[str] | None = None,
        *,
        batch_size: int = 32,
    ) -> list[EmbeddingResult]:
        """Generiert Embeddings für mehrere Texte.

        Nutzt Cache wo möglich, batcht API-Calls.

        Args:
            texts: Liste von Texten.
            content_hashes: Optional zugehörige Cache-Keys.
            batch_size: Max Texte pro API-Call.

        Returns:
            Liste von EmbeddingResults (gleiche Reihenfolge wie Input).
        """
        if content_hashes is None:
            content_hashes = [""] * len(texts)

        if len(texts) != len(content_hashes):
            raise ValueError("texts und content_hashes müssen gleich lang sein")

        results: list[EmbeddingResult | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        # Schritt 1: Cache-Hits sammeln (promote to most-recently-used)
        for i, (text, h) in enumerate(zip(texts, content_hashes, strict=False)):
            self._stats.total_requests += 1
            if h and h in self._cache:
                self._stats.cache_hits += 1
                self._cache.move_to_end(h)
                results[i] = EmbeddingResult(
                    vector=self._cache[h],
                    model=self._model,
                    dimensions=self._dimensions,
                    cached=True,
                )
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        # Schritt 2: Uncached in Batches embedden via Provider
        for batch_start in range(0, len(uncached_texts), batch_size):
            batch = uncached_texts[batch_start : batch_start + batch_size]
            batch_indices = uncached_indices[batch_start : batch_start + batch_size]

            try:
                self._stats.api_calls += 1
                embeddings = await self._provider.embed_batch_raw(self._model, batch)

                for _j, (idx, vec) in enumerate(zip(batch_indices, embeddings, strict=False)):
                    h = content_hashes[idx]
                    if h:
                        self._cache_put(h, vec)
                    results[idx] = EmbeddingResult(
                        vector=vec,
                        model=self._model,
                        dimensions=len(vec),
                        cached=False,
                    )

            except (httpx.HTTPError, ValueError, NotImplementedError) as e:
                self._stats.errors += 1
                logger.error("Batch-Embedding-Fehler: %s", e)
                # Fehlende Ergebnisse bleiben None (kein Zero-Vektor)

        # None-Eintraege bleiben None -- Aufrufer muss filtern
        return results  # type: ignore[return-value]

    async def close(self) -> None:
        """Schließt den HTTP-Client."""
        await self._provider.close()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Berechnet die Kosinus-Ähnlichkeit zweier Vektoren.

    Returns:
        Wert zwischen -1.0 und 1.0. Höher = ähnlicher.
    """
    if len(a) != len(b):
        return 0.0

    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (norm_a * norm_b)
