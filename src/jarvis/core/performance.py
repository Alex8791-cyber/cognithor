"""Jarvis · Performance & Skalierbarkeit.

Optimierte Infrastruktur für niedrige Latenz und hohen Durchsatz:

  - VectorStore:           Abstraktionsschicht für Vektor-Datenbanken
  - QueryDecomposer:       Zerlegt komplexe Anfragen für schnelleres RAG
  - LoadBalancer:          Verteilt Anfragen auf mehrere Backends
  - CloudFallback:         Automatischer Fallback auf Cloud-LLMs bei Überlast
  - ResourceOptimizer:     Überwacht und optimiert Ressourcenverbrauch
  - LatencyTracker:        Misst und analysiert Antwortzeiten
  - PerformanceManager:    Hauptklasse

Architektur-Bibel: §18.1 (Performance), §18.2 (Skalierbarkeit)
"""

from __future__ import annotations

import hashlib
import math
import random
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ============================================================================
# Vector Store (Abstraktionsschicht)
# ============================================================================


class VectorBackend(Enum):
    IN_MEMORY = "in_memory"
    CHROMADB = "chromadb"
    FAISS = "faiss"
    QDRANT = "qdrant"
    MILVUS = "milvus"
    PGVECTOR = "pgvector"


@dataclass
class VectorEntry:
    """Ein Eintrag im Vector-Store."""

    entry_id: str
    text: str
    embedding: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""


@dataclass
class SearchResult:
    """Suchergebnis aus dem Vector-Store."""

    entry: VectorEntry
    score: float          # 0-1 (Cosine Similarity)
    distance: float = 0   # Euklidischer Abstand


class VectorStore:
    """In-Memory Vector-Store mit Cosine-Similarity.

    Abstrahiert die Anbindung an externe Vector-DBs.
    In Produktion: Wrapper um ChromaDB/FAISS/Qdrant.
    """

    def __init__(self, backend: VectorBackend = VectorBackend.IN_MEMORY, dimension: int = 384) -> None:
        self._backend = backend
        self._dimension = dimension
        self._entries: dict[str, VectorEntry] = {}
        self._counter = 0

    def add(self, text: str, embedding: list[float], metadata: dict[str, Any] | None = None) -> VectorEntry:
        """Fügt einen Eintrag hinzu."""
        self._counter += 1
        entry = VectorEntry(
            entry_id=f"VEC-{self._counter:06d}",
            text=text,
            embedding=embedding[:self._dimension],
            metadata=metadata or {},
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        self._entries[entry.entry_id] = entry
        return entry

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[SearchResult]:
        """Sucht die ähnlichsten Einträge."""
        query = query_embedding[:self._dimension]
        results = []
        for entry in self._entries.values():
            score = self._cosine_similarity(query, entry.embedding)
            results.append(SearchResult(entry=entry, score=score))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def delete(self, entry_id: str) -> bool:
        return self._entries.pop(entry_id, None) is not None

    def clear(self) -> int:
        count = len(self._entries)
        self._entries.clear()
        return count

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Berechnet Cosine-Similarity zweier Vektoren."""
        if len(a) != len(b) or not a:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(x * x for x in b))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    def stats(self) -> dict[str, Any]:
        return {
            "backend": self._backend.value,
            "dimension": self._dimension,
            "entries": len(self._entries),
        }


# ============================================================================
# Query Decomposer
# ============================================================================


@dataclass
class SubQuery:
    """Eine zerlegte Teilanfrage."""

    query_id: str
    text: str
    query_type: str = "factual"    # factual, analytical, comparison
    priority: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.query_id, "text": self.text, "type": self.query_type, "priority": self.priority}


class QueryDecomposer:
    """Zerlegt komplexe Anfragen in Sub-Queries für paralleles RAG."""

    SPLIT_MARKERS = [" und ", " sowie ", " außerdem ", " darüber hinaus ", " zusätzlich ",
                     " and ", " also ", " furthermore ", " additionally "]

    def decompose(self, query: str) -> list[SubQuery]:
        """Zerlegt eine Anfrage in Teilanfragen."""
        parts = [query]
        for marker in self.SPLIT_MARKERS:
            new_parts = []
            for part in parts:
                new_parts.extend(part.split(marker))
            parts = new_parts

        # Filtere zu kurze Teile
        parts = [p.strip() for p in parts if len(p.strip()) > 10]
        if not parts:
            parts = [query]

        sub_queries = []
        for i, part in enumerate(parts):
            sub_queries.append(SubQuery(
                query_id=f"SQ-{i+1:03d}",
                text=part,
                query_type=self._classify(part),
                priority=1 if i == 0 else 2,
            ))
        return sub_queries

    def _classify(self, text: str) -> str:
        """Klassifiziert eine Anfrage."""
        lower = text.lower()
        if any(w in lower for w in ["vergleich", "unterschied", "vs", "compare"]):
            return "comparison"
        if any(w in lower for w in ["warum", "analyse", "bewert", "why", "analyze"]):
            return "analytical"
        return "factual"

    def stats(self) -> dict[str, Any]:
        return {"split_markers": len(self.SPLIT_MARKERS)}


# ============================================================================
# Load Balancer
# ============================================================================


class BalancingStrategy(Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED = "weighted"
    LATENCY_BASED = "latency_based"


@dataclass
class Backend:
    """Ein LLM-Backend (lokal oder remote)."""

    backend_id: str
    name: str
    url: str
    weight: int = 1
    max_concurrent: int = 4
    current_load: int = 0
    avg_latency_ms: float = 0
    healthy: bool = True
    is_local: bool = True
    total_requests: int = 0
    total_errors: int = 0

    @property
    def load_percent(self) -> float:
        if self.max_concurrent == 0:
            return 100.0
        return round(self.current_load / self.max_concurrent * 100, 1)

    @property
    def error_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return round(self.total_errors / self.total_requests * 100, 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.backend_id,
            "name": self.name,
            "healthy": self.healthy,
            "load": f"{self.load_percent}%",
            "latency_ms": self.avg_latency_ms,
            "local": self.is_local,
        }


class LoadBalancer:
    """Verteilt Anfragen auf mehrere LLM-Backends."""

    def __init__(self, strategy: BalancingStrategy = BalancingStrategy.LATENCY_BASED) -> None:
        self._strategy = strategy
        self._backends: list[Backend] = []
        self._rr_index = 0

    def add_backend(self, backend: Backend) -> None:
        self._backends.append(backend)

    def remove_backend(self, backend_id: str) -> bool:
        before = len(self._backends)
        self._backends = [b for b in self._backends if b.backend_id != backend_id]
        return len(self._backends) < before

    def select_backend(self) -> Backend | None:
        """Wählt das beste Backend basierend auf Strategie."""
        healthy = [b for b in self._backends if b.healthy and b.current_load < b.max_concurrent]
        if not healthy:
            return None

        if self._strategy == BalancingStrategy.ROUND_ROBIN:
            backend = healthy[self._rr_index % len(healthy)]
            self._rr_index += 1
            return backend

        elif self._strategy == BalancingStrategy.LEAST_CONNECTIONS:
            return min(healthy, key=lambda b: b.current_load)

        elif self._strategy == BalancingStrategy.LATENCY_BASED:
            return min(healthy, key=lambda b: b.avg_latency_ms if b.avg_latency_ms > 0 else float("inf"))

        elif self._strategy == BalancingStrategy.WEIGHTED:
            total_weight = sum(b.weight for b in healthy)
            r = random.random() * total_weight
            cumulative = 0
            for b in healthy:
                cumulative += b.weight
                if r <= cumulative:
                    return b
            return healthy[-1]

        return healthy[0]

    def record_request(self, backend_id: str, latency_ms: float, error: bool = False) -> None:
        """Zeichnet ein Request-Ergebnis auf."""
        for b in self._backends:
            if b.backend_id == backend_id:
                b.total_requests += 1
                if error:
                    b.total_errors += 1
                # Gleitender Durchschnitt
                if b.avg_latency_ms == 0:
                    b.avg_latency_ms = latency_ms
                else:
                    b.avg_latency_ms = round(b.avg_latency_ms * 0.9 + latency_ms * 0.1, 1)
                # Auto-Health-Check
                if b.error_rate > 50:
                    b.healthy = False
                break

    def health_check(self) -> list[Backend]:
        """Gibt unhealthy Backends zurück."""
        return [b for b in self._backends if not b.healthy]

    @property
    def backend_count(self) -> int:
        return len(self._backends)

    def stats(self) -> dict[str, Any]:
        return {
            "strategy": self._strategy.value,
            "backends": len(self._backends),
            "healthy": sum(1 for b in self._backends if b.healthy),
            "total_requests": sum(b.total_requests for b in self._backends),
            "avg_latency": round(
                sum(b.avg_latency_ms for b in self._backends) / max(len(self._backends), 1), 1
            ),
        }


# ============================================================================
# Cloud Fallback
# ============================================================================


class CloudProvider(Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GROQ = "groq"
    TOGETHER = "together"
    NONE = "none"              # Nur lokal


@dataclass
class FallbackConfig:
    """Konfiguration für Cloud-Fallback."""

    enabled: bool = False
    provider: CloudProvider = CloudProvider.NONE
    model: str = ""
    max_daily_requests: int = 100
    max_cost_eur_day: float = 5.0
    trigger_latency_ms: int = 5000   # Fallback bei > 5s Latenz
    trigger_load_percent: float = 90  # Fallback bei > 90% Last

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "provider": self.provider.value,
            "model": self.model,
            "max_daily": self.max_daily_requests,
            "max_cost": self.max_cost_eur_day,
        }


class CloudFallback:
    """Automatischer Fallback auf Cloud-LLMs bei lokaler Überlast."""

    def __init__(self, config: FallbackConfig | None = None) -> None:
        self._config = config or FallbackConfig()
        self._daily_requests = 0
        self._daily_cost = 0.0
        self._last_reset = time.strftime("%Y-%m-%d", time.gmtime())
        self._fallback_log: list[dict[str, Any]] = []

    @property
    def config(self) -> FallbackConfig:
        return self._config

    def should_fallback(self, local_latency_ms: float, local_load_percent: float) -> bool:
        """Entscheidet ob Cloud-Fallback nötig ist."""
        if not self._config.enabled:
            return False
        self._reset_daily_if_needed()
        if self._daily_requests >= self._config.max_daily_requests:
            return False
        if self._daily_cost >= self._config.max_cost_eur_day:
            return False
        return (
            local_latency_ms > self._config.trigger_latency_ms
            or local_load_percent > self._config.trigger_load_percent
        )

    def record_fallback(self, cost_eur: float = 0.01) -> None:
        self._daily_requests += 1
        self._daily_cost += cost_eur
        self._fallback_log.append({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "cost": cost_eur,
            "provider": self._config.provider.value,
        })

    def _reset_daily_if_needed(self) -> None:
        today = time.strftime("%Y-%m-%d", time.gmtime())
        if today != self._last_reset:
            self._daily_requests = 0
            self._daily_cost = 0.0
            self._last_reset = today

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": self._config.enabled,
            "provider": self._config.provider.value,
            "daily_requests": self._daily_requests,
            "daily_cost_eur": round(self._daily_cost, 4),
            "total_fallbacks": len(self._fallback_log),
        }


# ============================================================================
# Resource Optimizer
# ============================================================================


@dataclass
class ResourceSnapshot:
    """Momentaufnahme der Ressourcenauslastung."""

    timestamp: str
    cpu_percent: float = 0
    ram_used_gb: float = 0
    ram_total_gb: float = 0
    gpu_percent: float = 0
    vram_used_gb: float = 0
    active_agents: int = 0
    pending_requests: int = 0

    @property
    def ram_percent(self) -> float:
        if self.ram_total_gb == 0:
            return 0.0
        return round(self.ram_used_gb / self.ram_total_gb * 100, 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cpu": f"{self.cpu_percent}%",
            "ram": f"{self.ram_percent}%",
            "gpu": f"{self.gpu_percent}%",
            "agents": self.active_agents,
            "pending": self.pending_requests,
        }


class ResourceOptimizer:
    """Überwacht und optimiert Ressourcenverbrauch."""

    def __init__(self, max_history: int = 100) -> None:
        self._history: deque[ResourceSnapshot] = deque(maxlen=max_history)
        self._thresholds = {
            "cpu_warn": 80.0,
            "ram_warn": 85.0,
            "gpu_warn": 90.0,
        }

    def snapshot(
        self,
        cpu: float = 0, ram_used: float = 0, ram_total: float = 16,
        gpu: float = 0, vram_used: float = 0, agents: int = 0, pending: int = 0,
    ) -> ResourceSnapshot:
        """Erfasst aktuelle Ressourcen."""
        snap = ResourceSnapshot(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            cpu_percent=cpu,
            ram_used_gb=ram_used,
            ram_total_gb=ram_total,
            gpu_percent=gpu,
            vram_used_gb=vram_used,
            active_agents=agents,
            pending_requests=pending,
        )
        self._history.append(snap)
        return snap

    def alerts(self) -> list[str]:
        """Gibt aktive Warnungen zurück."""
        if not self._history:
            return []
        latest = self._history[-1]
        alerts = []
        if latest.cpu_percent > self._thresholds["cpu_warn"]:
            alerts.append(f"⚠️ CPU-Last hoch: {latest.cpu_percent}%")
        if latest.ram_percent > self._thresholds["ram_warn"]:
            alerts.append(f"⚠️ RAM-Auslastung hoch: {latest.ram_percent}%")
        if latest.gpu_percent > self._thresholds["gpu_warn"]:
            alerts.append(f"⚠️ GPU-Last hoch: {latest.gpu_percent}%")
        return alerts

    def recommendations(self) -> list[str]:
        """Gibt Optimierungs-Empfehlungen."""
        if not self._history:
            return ["Noch keine Daten — erste Snapshot erfassen"]
        recs = []
        latest = self._history[-1]
        if latest.ram_percent > 90:
            recs.append("RAM kritisch: Modell-Quantisierung verringern oder Agenten reduzieren")
        if latest.cpu_percent > 90 and latest.gpu_percent < 50:
            recs.append("CPU überlastet, GPU unterfordert: GPU-Offloading aktivieren")
        if latest.pending_requests > 10:
            recs.append("Request-Queue wächst: Cloud-Fallback oder mehr Backends aktivieren")
        if not recs:
            recs.append("✅ System läuft optimal")
        return recs

    def avg_over(self, minutes: int = 5) -> dict[str, float]:
        """Durchschnittswerte über die letzten N Minuten."""
        if not self._history:
            return {"cpu": 0, "ram": 0, "gpu": 0}
        recent = list(self._history)[-minutes * 12:]  # ~5s Intervall
        return {
            "cpu": round(sum(s.cpu_percent for s in recent) / len(recent), 1),
            "ram": round(sum(s.ram_percent for s in recent) / len(recent), 1),
            "gpu": round(sum(s.gpu_percent for s in recent) / len(recent), 1),
        }

    def stats(self) -> dict[str, Any]:
        return {
            "snapshots": len(self._history),
            "alerts": self.alerts(),
            "recommendations": self.recommendations(),
        }


# ============================================================================
# Latency Tracker
# ============================================================================


class LatencyTracker:
    """Misst und analysiert Antwortzeiten."""

    def __init__(self, max_samples: int = 1000) -> None:
        self._samples: deque[float] = deque(maxlen=max_samples)
        self._by_operation: dict[str, deque] = {}

    def record(self, latency_ms: float, operation: str = "inference") -> None:
        self._samples.append(latency_ms)
        if operation not in self._by_operation:
            self._by_operation[operation] = deque(maxlen=1000)
        self._by_operation[operation].append(latency_ms)

    def percentile(self, p: float) -> float:
        """Berechnet das p-te Perzentil."""
        if not self._samples:
            return 0.0
        sorted_s = sorted(self._samples)
        idx = int(len(sorted_s) * p / 100)
        return sorted_s[min(idx, len(sorted_s) - 1)]

    @property
    def p50(self) -> float:
        return self.percentile(50)

    @property
    def p95(self) -> float:
        return self.percentile(95)

    @property
    def p99(self) -> float:
        return self.percentile(99)

    @property
    def avg(self) -> float:
        if not self._samples:
            return 0.0
        return round(sum(self._samples) / len(self._samples), 1)

    def by_operation(self, operation: str) -> dict[str, float]:
        samples = self._by_operation.get(operation, deque())
        if not samples:
            return {"avg": 0, "p50": 0, "p95": 0}
        sorted_s = sorted(samples)
        return {
            "avg": round(sum(sorted_s) / len(sorted_s), 1),
            "p50": sorted_s[len(sorted_s) // 2],
            "p95": sorted_s[int(len(sorted_s) * 0.95)],
        }

    def stats(self) -> dict[str, Any]:
        return {
            "total_samples": len(self._samples),
            "avg_ms": self.avg,
            "p50_ms": self.p50,
            "p95_ms": self.p95,
            "p99_ms": self.p99,
            "operations": list(self._by_operation.keys()),
        }


# ============================================================================
# Performance Manager (Hauptklasse)
# ============================================================================


class PerformanceManager:
    """Hauptklasse: Orchestriert alle Performance-Komponenten."""

    def __init__(self) -> None:
        self._vector_store = VectorStore()
        self._decomposer = QueryDecomposer()
        self._balancer = LoadBalancer()
        self._fallback = CloudFallback()
        self._optimizer = ResourceOptimizer()
        self._latency = LatencyTracker()

    @property
    def vector_store(self) -> VectorStore:
        return self._vector_store

    @property
    def decomposer(self) -> QueryDecomposer:
        return self._decomposer

    @property
    def balancer(self) -> LoadBalancer:
        return self._balancer

    @property
    def fallback(self) -> CloudFallback:
        return self._fallback

    @property
    def optimizer(self) -> ResourceOptimizer:
        return self._optimizer

    @property
    def latency(self) -> LatencyTracker:
        return self._latency

    def process_query(self, query: str, query_embedding: list[float] | None = None) -> dict[str, Any]:
        """Verarbeitet eine Anfrage mit allen Performance-Optimierungen."""
        start = time.time()

        # 1. Query dekomponieren
        sub_queries = self._decomposer.decompose(query)

        # 2. RAG-Suche (falls Embedding vorhanden)
        rag_results = []
        if query_embedding:
            rag_results = self._vector_store.search(query_embedding, top_k=3)

        # 3. Backend auswählen
        backend = self._balancer.select_backend()

        elapsed_ms = round((time.time() - start) * 1000, 1)
        self._latency.record(elapsed_ms, "query_processing")

        return {
            "sub_queries": len(sub_queries),
            "rag_results": len(rag_results),
            "backend": backend.name if backend else "none",
            "latency_ms": elapsed_ms,
        }

    def health(self) -> dict[str, Any]:
        """Gesamter Health-Status."""
        return {
            "vector_store": self._vector_store.stats(),
            "balancer": self._balancer.stats(),
            "fallback": self._fallback.stats(),
            "resources": self._optimizer.stats(),
            "latency": self._latency.stats(),
        }

    def stats(self) -> dict[str, Any]:
        return self.health()
