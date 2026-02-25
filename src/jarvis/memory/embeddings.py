"""Embedding-Client · Ollama-basiert mit Cache. [B§4.7, B§12]

Generiert Embeddings via Ollama (nomic-embed-text, 768d).
Nutzt Content-Hash als Cache-Key → gleicher Text = kein neuer API-Call.
"""

from __future__ import annotations

import logging
import math
from collections import OrderedDict
from dataclasses import dataclass

import httpx

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
        """Berechnet die Cache-Hit-Rate (0.0–1.0)."""
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests


class EmbeddingClient:
    """Async Embedding-Client mit In-Memory-Cache.

    Nutzt Ollama's /api/embed Endpoint.
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
    ) -> None:
        """Initialisiert den Embedding-Client mit Modell und Cache."""
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._dimensions = dimensions
        self._timeout = timeout
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._stats = EmbeddingStats()
        self._client: httpx.AsyncClient | None = None

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

    async def _get_client(self) -> httpx.AsyncClient:
        """Lazy-initialisiert den httpx AsyncClient.

        When constructing the client we disable reading proxy settings
        from the environment (`trust_env=False`) to avoid requiring
        optional dependencies such as socksio. Without this flag, httpx
        will attempt to use any SOCKS proxies defined in environment
        variables which may not be supported in the runtime environment.
        """
        if self._client is None or getattr(self._client, "is_closed", False):
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout,
                trust_env=False,
            )
        return self._client

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

        # API-Call
        self._stats.api_calls += 1
        try:
            client = await self._get_client()
            resp = await client.post(
                "/api/embed",
                json={"model": self._model, "input": text},
            )
            resp.raise_for_status()
            data = resp.json()

            # Ollama gibt "embeddings" zurück (Liste von Vektoren)
            embeddings = data.get("embeddings", [])
            if not embeddings:
                raise ValueError("Keine Embeddings in Ollama-Antwort")

            vector = embeddings[0]

            # Cache speichern (bounded LRU)
            if content_hash:
                self._cache_put(content_hash, vector)

            return EmbeddingResult(
                vector=vector,
                model=self._model,
                dimensions=len(vector),
                cached=False,
            )

        except (httpx.HTTPError, ValueError, KeyError) as e:
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

        # Schritt 2: Uncached in Batches embedden
        for batch_start in range(0, len(uncached_texts), batch_size):
            batch = uncached_texts[batch_start : batch_start + batch_size]
            batch_indices = uncached_indices[batch_start : batch_start + batch_size]

            try:
                self._stats.api_calls += 1
                client = await self._get_client()
                resp = await client.post(
                    "/api/embed",
                    json={"model": self._model, "input": batch},
                )
                resp.raise_for_status()
                data = resp.json()
                embeddings = data.get("embeddings", [])

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

            except (httpx.HTTPError, ValueError) as e:
                self._stats.errors += 1
                logger.error("Batch-Embedding-Fehler: %s", e)
                # Fehlende Ergebnisse bleiben None (kein Zero-Vektor)

        # None-Eintraege bleiben None — Aufrufer muss filtern
        return results  # type: ignore[return-value]

    async def close(self) -> None:
        """Schließt den HTTP-Client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None


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
