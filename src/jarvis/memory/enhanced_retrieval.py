"""Enhanced Retrieval: Fortgeschrittene RAG-Techniken über der Hybrid-Suche.

Baut auf der bestehenden HybridSearch (BM25+Vektor+Graph) auf und
ergänzt fünf wesentliche Fähigkeiten:

  1. Query-Dekomposition: Komplexe Fragen in Teilfragen zerlegen
  2. Reciprocal Rank Fusion (RRF): Multi-Query-Ergebnisse intelligent mergen
  3. Corrective RAG: Relevanz-Prüfung mit automatischem Re-Retrieval
  4. Frequenz-Gewichtung: Oft referenzierte Chunks höher ranken
  5. Episodenkompression: Alte Episoden zu Zusammenfassungen verdichten

Architektur:
  User-Query → QueryDecomposer → [sub_query_1, sub_query_2, ...]
             → HybridSearch × N Queries
             → RRF-Merge → Vorläufige Ergebnisse
             → CorrectionStage → Relevanz-Check
             → FrequencyBoost → Finale Ergebnisse

Bibel-Referenz: §4.7 (Enhanced Retrieval), §4.3 (Episodic Compression)
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Callable

from jarvis.models import Chunk, MemorySearchResult, MemoryTier

logger = logging.getLogger("jarvis.memory.enhanced_retrieval")


# ============================================================================
# 1. Query-Dekomposition
# ============================================================================


@dataclass
class DecomposedQuery:
    """Ergebnis der Query-Dekomposition."""

    original: str
    sub_queries: list[str]
    query_type: str = "simple"  # simple, compound, multi_aspect


class QueryDecomposer:
    """Zerlegt komplexe Fragen in mehrere Teilfragen.

    Zwei Modi:
      - Rule-based: Schnell, deterministisch, kein LLM nötig
      - LLM-based: Bessere Qualität für komplexe Fragen

    Rule-based Strategien:
      - Konjunktionen splitten ("X und Y" → "X", "Y")
      - Vergleiche erkennen ("Unterschied zwischen A und B" → "A", "B")
      - Aspekte extrahieren ("Vorteile und Nachteile von X" → "Vorteile X", "Nachteile X")
      - Zeitliche Aspekte ("X früher vs heute" → "X historisch", "X aktuell")
    """

    # Konjunktions-Patterns
    _CONJUNCTION_PATTERNS = [
        re.compile(r"(.+?)\s+(?:und|sowie|als auch|und auch)\s+(.+)", re.IGNORECASE),
    ]

    # Vergleichs-Patterns
    _COMPARISON_PATTERNS = [
        re.compile(
            r"(?:unterschied|vergleich|differenz)\s+zwischen\s+(.+?)\s+und\s+(.+)",
            re.IGNORECASE,
        ),
        re.compile(r"(.+?)\s+(?:vs\.?|versus|gegen|oder)\s+(.+)", re.IGNORECASE),
    ]

    # Aspekt-Patterns
    _ASPECT_PATTERNS = [
        re.compile(
            r"(vor-?\s*und\s*nachteile|pros?\s*(?:und|&)\s*cons?)\s+(?:von\s+)?(.+)",
            re.IGNORECASE,
        ),
    ]

    def __init__(self, llm_fn: Callable[..., Any] | None = None) -> None:
        """Args:
            llm_fn: Optionale async LLM-Funktion für bessere Dekomposition.
        """
        self._llm_fn = llm_fn

    def decompose(self, query: str) -> DecomposedQuery:
        """Zerlegt eine Query in Teilfragen (rule-based).

        Args:
            query: Ursprüngliche Nutzerfrage.

        Returns:
            DecomposedQuery mit original + sub_queries.
        """
        query = query.strip()
        if not query:
            return DecomposedQuery(original=query, sub_queries=[query])

        # Versuche Patterns in Prioritätsreihenfolge
        # 1. Vergleiche
        for pattern in self._COMPARISON_PATTERNS:
            match = pattern.search(query)
            if match:
                a, b = match.group(1).strip(), match.group(2).strip()
                return DecomposedQuery(
                    original=query,
                    sub_queries=[query, a, b],
                    query_type="compound",
                )

        # 2. Aspekte (Vor- und Nachteile)
        for pattern in self._ASPECT_PATTERNS:
            match = pattern.search(query)
            if match:
                topic = match.group(2).strip()
                return DecomposedQuery(
                    original=query,
                    sub_queries=[query, f"Vorteile {topic}", f"Nachteile {topic}"],
                    query_type="multi_aspect",
                )

        # 3. Konjunktionen
        for pattern in self._CONJUNCTION_PATTERNS:
            match = pattern.search(query)
            if match:
                a, b = match.group(1).strip(), match.group(2).strip()
                # Nur splitten wenn beide Teile substanziell (>3 Wörter)
                if len(a.split()) >= 2 and len(b.split()) >= 2:
                    return DecomposedQuery(
                        original=query,
                        sub_queries=[query, a, b],
                        query_type="compound",
                    )

        # 4. Keine Dekomposition möglich → Original behalten
        return DecomposedQuery(
            original=query,
            sub_queries=[query],
            query_type="simple",
        )

    async def decompose_with_llm(self, query: str) -> DecomposedQuery:
        """Zerlegt via LLM (bessere Qualität, langsamer).

        Falls kein LLM verfügbar, fällt auf rule-based zurück.
        """
        if self._llm_fn is None:
            return self.decompose(query)

        prompt = (
            "Zerlege die folgende Frage in 1-3 einfachere Suchanfragen. "
            "Gib jede Suchanfrage auf einer eigenen Zeile aus. "
            "Keine Nummerierung, keine Erklärungen.\n\n"
            f"Frage: {query}\n\nSuchanfragen:"
        )

        try:
            response = await self._llm_fn(prompt)
            lines = [ln.strip() for ln in response.strip().splitlines() if ln.strip()]
            if lines:
                return DecomposedQuery(
                    original=query,
                    sub_queries=[query] + lines[:3],  # Original immer dabei
                    query_type="llm_decomposed",
                )
        except Exception as exc:
            logger.warning("llm_decomposition_failed: %s", exc)

        return self.decompose(query)


# ============================================================================
# 2. Reciprocal Rank Fusion (RRF)
# ============================================================================


def reciprocal_rank_fusion(
    result_lists: list[list[MemorySearchResult]],
    *,
    k: int = 60,
    top_n: int | None = None,
) -> list[MemorySearchResult]:
    """Merged mehrere Ergebnislisten via Reciprocal Rank Fusion.

    RRF ist robust gegen unterschiedliche Score-Skalen und bevorzugt
    Chunks die in mehreren Suchergebnissen auftauchen.

    Formel: RRF_score(d) = Σ 1 / (k + rank_i(d))

    Wobei k=60 (Standard-Konstante die stabile Rankings erzeugt).

    Args:
        result_lists: Liste von Ergebnislisten (je eine pro Sub-Query).
        k: RRF-Konstante (höher = weniger Einfluss der Rankposition).
        top_n: Max Ergebnisse (None = alle).

    Returns:
        Merged + sortierte Ergebnisliste.
    """
    # Chunk-ID → aggregierter RRF-Score
    rrf_scores: dict[str, float] = {}
    # Chunk-ID → bestes MemorySearchResult (für Metadaten)
    best_results: dict[str, MemorySearchResult] = {}

    for result_list in result_lists:
        for rank, result in enumerate(result_list):
            chunk_id = result.chunk.id
            rrf_score = 1.0 / (k + rank + 1)
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + rrf_score

            # Behalte das Ergebnis mit dem höchsten Original-Score
            if chunk_id not in best_results or result.score > best_results[chunk_id].score:
                best_results[chunk_id] = result

    # Sortieren nach RRF-Score
    sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

    # Ergebnisse mit RRF-Score
    merged: list[MemorySearchResult] = []
    for chunk_id in sorted_ids:
        original = best_results[chunk_id]
        # Neues Result mit RRF-Score als Haupt-Score
        merged.append(
            MemorySearchResult(
                chunk=original.chunk,
                score=rrf_scores[chunk_id],
                bm25_score=original.bm25_score,
                vector_score=original.vector_score,
                graph_score=original.graph_score,
                recency_factor=original.recency_factor,
            ),
        )

    if top_n:
        return merged[:top_n]
    return merged


# ============================================================================
# 3. Corrective RAG
# ============================================================================


@dataclass
class RelevanceVerdict:
    """Ergebnis der Relevanz-Prüfung."""

    relevant_results: list[MemorySearchResult]
    irrelevant_results: list[MemorySearchResult]
    confidence: float  # 0.0-1.0 wie sicher die Bewertung ist
    needs_retry: bool  # True wenn zu wenig relevante Ergebnisse


class CorrectiveRAG:
    """Prüft Retrieval-Ergebnisse auf Relevanz und triggert Re-Retrieval.

    Zwei Modi:
      - Heuristic: Schnell, basiert auf Score-Schwellwerten + Overlap
      - LLM-based: LLM bewertet ob jedes Ergebnis zur Frage passt

    Workflow:
      1. Ergebnisse erhalten
      2. Relevanz bewerten (heuristic oder LLM)
      3. Wenn <min_relevant: Alternative Query generieren → Re-Retrieval
      4. Ergebnisse zusammenführen
    """

    def __init__(
        self,
        *,
        min_score_threshold: float = 0.15,
        min_relevant_count: int = 2,
        llm_fn: Callable[..., Any] | None = None,
    ) -> None:
        self._min_score = min_score_threshold
        self._min_relevant = min_relevant_count
        self._llm_fn = llm_fn

    def evaluate_relevance_heuristic(
        self,
        query: str,
        results: list[MemorySearchResult],
    ) -> RelevanceVerdict:
        """Heuristische Relevanz-Bewertung.

        Kriterien:
          - Score über Schwellwert
          - Wort-Overlap zwischen Query und Chunk-Text
          - Entitäts-Overlap
        """
        query_words = set(re.findall(r"\w+", query.lower()))
        relevant: list[MemorySearchResult] = []
        irrelevant: list[MemorySearchResult] = []

        for result in results:
            # Score-Check
            if result.score < self._min_score:
                irrelevant.append(result)
                continue

            # Wort-Overlap-Check
            chunk_words = set(re.findall(r"\w+", result.chunk.text.lower()))
            overlap = len(query_words & chunk_words)
            overlap_ratio = overlap / max(len(query_words), 1)

            # Kombinations-Heuristik
            if result.score >= 0.3 or overlap_ratio >= 0.3:
                relevant.append(result)
            elif result.score >= 0.15 and overlap_ratio >= 0.15:
                relevant.append(result)
            else:
                irrelevant.append(result)

        needs_retry = len(relevant) < self._min_relevant and len(results) > 0
        confidence = len(relevant) / max(len(results), 1)

        return RelevanceVerdict(
            relevant_results=relevant,
            irrelevant_results=irrelevant,
            confidence=confidence,
            needs_retry=needs_retry,
        )

    def generate_alternative_queries(self, original_query: str) -> list[str]:
        """Generiert alternative Suchanfragen (rule-based).

        Strategien:
          - Synonyme/Umformulierungen
          - Kürzere Version (nur Schlüsselwörter)
          - Breitere Version (ohne Einschränkungen)
        """
        words = original_query.split()
        alternatives: list[str] = []

        # Strategie 1: Nur Schlüsselwörter (Stoppwörter entfernen)
        stopwords = {
            "der", "die", "das", "ein", "eine", "ist", "sind", "war", "hat",
            "und", "oder", "aber", "in", "an", "auf", "mit", "von", "zu",
            "für", "über", "nach", "aus", "bei", "um", "wie", "was", "wer",
            "wo", "wann", "warum", "ich", "du", "er", "sie", "es", "wir",
            "mein", "dein", "sein", "ihr", "the", "a", "is", "are", "was",
            "and", "or", "in", "on", "at", "with", "for", "to", "of",
        }
        keywords = [w for w in words if w.lower() not in stopwords and len(w) > 2]
        if keywords and len(keywords) < len(words):
            alternatives.append(" ".join(keywords))

        # Strategie 2: Erste N Wörter (wenn Query lang)
        if len(words) > 6:
            alternatives.append(" ".join(words[:4]))

        # Strategie 3: Letzte N Wörter (oft der eigentliche Kern)
        if len(words) > 4:
            alternatives.append(" ".join(words[-3:]))

        return alternatives


# ============================================================================
# 4. Frequenz-Gewichtung
# ============================================================================


class FrequencyTracker:
    """Trackt wie oft Chunks abgerufen werden.

    Häufig referenzierte Chunks erhalten einen Boost,
    weil sie vermutlich wichtiger sind.

    Formel: frequency_boost = 1.0 + log(1 + access_count) * weight

    Der logarithmische Faktor verhindert dass ein Chunk
    durch häufigen Zugriff unverhältnismäßig dominiert.
    """

    def __init__(self, *, frequency_weight: float = 0.1) -> None:
        self._access_counts: Counter[str] = Counter()
        self._weight = frequency_weight

    @property
    def total_accesses(self) -> int:
        return sum(self._access_counts.values())

    def record_access(self, chunk_id: str) -> None:
        """Registriert einen Zugriff auf einen Chunk."""
        self._access_counts[chunk_id] += 1

    def record_accesses(self, chunk_ids: list[str]) -> None:
        """Registriert Zugriffe auf mehrere Chunks."""
        for cid in chunk_ids:
            self._access_counts[cid] += 1

    def get_count(self, chunk_id: str) -> int:
        """Zugriffszähler für einen Chunk."""
        return self._access_counts.get(chunk_id, 0)

    def boost_factor(self, chunk_id: str) -> float:
        """Berechnet den Frequency-Boost für einen Chunk.

        Returns:
            Boost-Faktor >= 1.0 (1.0 = kein Boost).
        """
        count = self._access_counts.get(chunk_id, 0)
        if count == 0:
            return 1.0
        return 1.0 + math.log(1 + count) * self._weight

    def apply_boost(
        self,
        results: list[MemorySearchResult],
    ) -> list[MemorySearchResult]:
        """Wendet Frequency-Boost auf Suchergebnisse an.

        Args:
            results: Originale Suchergebnisse.

        Returns:
            Ergebnisse mit angepassten Scores, neu sortiert.
        """
        boosted: list[MemorySearchResult] = []
        for result in results:
            boost = self.boost_factor(result.chunk.id)
            boosted.append(
                MemorySearchResult(
                    chunk=result.chunk,
                    score=result.score * boost,
                    bm25_score=result.bm25_score,
                    vector_score=result.vector_score,
                    graph_score=result.graph_score,
                    recency_factor=result.recency_factor,
                ),
            )
        boosted.sort(key=lambda r: r.score, reverse=True)
        return boosted

    def top_accessed(self, n: int = 10) -> list[tuple[str, int]]:
        """Die N am häufigsten abgerufenen Chunks."""
        return self._access_counts.most_common(n)

    def clear(self) -> None:
        """Setzt alle Zähler zurück."""
        self._access_counts.clear()

    def stats(self) -> dict[str, Any]:
        return {
            "tracked_chunks": len(self._access_counts),
            "total_accesses": self.total_accesses,
            "top_5": self.top_accessed(5),
        }


# ============================================================================
# 5. Episodenkompression
# ============================================================================


@dataclass
class CompressedEpisode:
    """Eine komprimierte Episode (Zusammenfassung mehrerer Tage)."""

    start_date: date
    end_date: date
    summary: str
    key_facts: list[str] = field(default_factory=list)
    entities_mentioned: list[str] = field(default_factory=list)
    original_entry_count: int = 0
    compressed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )

    @property
    def date_range(self) -> str:
        return f"{self.start_date.isoformat()} – {self.end_date.isoformat()}"

    @property
    def days_covered(self) -> int:
        return (self.end_date - self.start_date).days + 1


class EpisodicCompressor:
    """Komprimiert alte Episoden zu Zusammenfassungen.

    Workflow:
      1. Episoden älter als retention_days identifizieren
      2. In Wochen-Blöcke gruppieren
      3. Pro Block: LLM-Zusammenfassung erstellen (oder heuristic)
      4. Zusammenfassung ins Semantic Memory speichern
      5. Original-Episoden optional archivieren

    Zwei Modi:
      - LLM-based: Hochwertige Zusammenfassungen
      - Heuristic: Extrahiert Schlüsselsätze und Entitäten

    Args:
        retention_days: Episoden älter als X Tage komprimieren.
        llm_fn: Async-Funktion für LLM-Aufrufe.
    """

    def __init__(
        self,
        *,
        retention_days: int = 30,
        llm_fn: Callable[..., Any] | None = None,
    ) -> None:
        self._retention_days = retention_days
        self._llm_fn = llm_fn

    def identify_compressible(
        self,
        episode_dates: list[date],
        *,
        reference_date: date | None = None,
    ) -> list[date]:
        """Identifiziert Episoden die komprimiert werden können.

        Args:
            episode_dates: Verfügbare Episoden-Daten.
            reference_date: Referenzdatum (default: heute).

        Returns:
            Liste von Daten die komprimiert werden sollten.
        """
        ref = reference_date or date.today()
        cutoff = ref.toordinal() - self._retention_days

        return [d for d in episode_dates if d.toordinal() <= cutoff]

    def group_into_weeks(self, dates: list[date]) -> list[tuple[date, date]]:
        """Gruppiert Daten in Wochen-Blöcke.

        Returns:
            Liste von (start_date, end_date) Tupeln.
        """
        if not dates:
            return []

        sorted_dates = sorted(dates)
        weeks: list[tuple[date, date]] = []
        current_start = sorted_dates[0]
        current_end = sorted_dates[0]

        for d in sorted_dates[1:]:
            # Gleiche Woche wenn weniger als 7 Tage Abstand
            if (d - current_end).days <= 7:
                current_end = d
            else:
                weeks.append((current_start, current_end))
                current_start = d
                current_end = d

        weeks.append((current_start, current_end))
        return weeks

    def compress_heuristic(
        self,
        entries: list[str],
        *,
        start_date: date,
        end_date: date,
        max_sentences: int = 5,
    ) -> CompressedEpisode:
        """Heuristische Kompression: Extrahiert Schlüsselsätze.

        Strategie:
          - Sätze mit Named Entities bevorzugen
          - Längere Sätze (mehr Info) bevorzugen
          - Duplikate entfernen
          - Entitäten extrahieren (Großgeschriebene Wörter)
        """
        all_sentences: list[str] = []
        entity_set: set[str] = set()

        for entry in entries:
            # Sätze splitten
            sentences = re.split(r"[.!?]\s+", entry)
            for sent in sentences:
                sent = sent.strip()
                if len(sent) > 20:  # Zu kurze Sätze ignorieren
                    all_sentences.append(sent)

                    # Entitäten extrahieren (einfache Heuristik: Großbuchstaben-Wörter)
                    for word in re.findall(r"\b[A-ZÄÖÜ][a-zäöüß]{2,}\b", sent):
                        entity_set.add(word)

        # Sätze nach Informationsgehalt ranken
        scored: list[tuple[float, str]] = []
        for sent in all_sentences:
            score = 0.0
            # Länge (normalisiert)
            score += min(len(sent) / 200.0, 1.0) * 0.3
            # Entitäten im Satz
            sent_entities = len(re.findall(r"\b[A-ZÄÖÜ][a-zäöüß]{2,}\b", sent))
            score += min(sent_entities / 3.0, 1.0) * 0.4
            # Zahlen (oft wichtige Fakten)
            numbers = len(re.findall(r"\d+", sent))
            score += min(numbers / 2.0, 1.0) * 0.3
            scored.append((score, sent))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Top-Sätze als Zusammenfassung
        key_sentences = []
        seen_texts: set[str] = set()
        for _score, sent in scored:
            # Duplikate vermeiden
            normalized = sent.lower().strip()
            if normalized not in seen_texts:
                key_sentences.append(sent)
                seen_texts.add(normalized)
            if len(key_sentences) >= max_sentences:
                break

        summary = ". ".join(key_sentences)
        if summary and not summary.endswith("."):
            summary += "."

        return CompressedEpisode(
            start_date=start_date,
            end_date=end_date,
            summary=summary,
            key_facts=key_sentences,
            entities_mentioned=sorted(entity_set),
            original_entry_count=len(entries),
        )

    async def compress_with_llm(
        self,
        entries: list[str],
        *,
        start_date: date,
        end_date: date,
    ) -> CompressedEpisode:
        """LLM-basierte Kompression: Hochwertige Zusammenfassung.

        Falls kein LLM verfügbar, fällt auf heuristic zurück.
        """
        if self._llm_fn is None or not entries:
            return self.compress_heuristic(
                entries, start_date=start_date, end_date=end_date,
            )

        combined = "\n\n".join(entries)
        prompt = (
            f"Fasse die folgenden Episoden vom {start_date} bis {end_date} "
            "in maximal 5 Sätzen zusammen. Behalte die wichtigsten Fakten, "
            "Personen und Entscheidungen. Antworte auf Deutsch.\n\n"
            f"{combined[:3000]}"  # Truncate für Token-Limits
        )

        try:
            summary = await self._llm_fn(prompt)
            entities = sorted(set(
                re.findall(r"\b[A-ZÄÖÜ][a-zäöüß]{2,}\b", summary),
            ))
            return CompressedEpisode(
                start_date=start_date,
                end_date=end_date,
                summary=summary.strip(),
                key_facts=summary.strip().split(". "),
                entities_mentioned=entities,
                original_entry_count=len(entries),
            )
        except Exception as exc:
            logger.warning("llm_compression_failed: %s", exc)
            return self.compress_heuristic(
                entries, start_date=start_date, end_date=end_date,
            )


# ============================================================================
# 6. Enhanced Search Pipeline (orchestriert alles)
# ============================================================================


class EnhancedSearchPipeline:
    """Orchestriert alle Enhanced-Retrieval-Komponenten.

    Nutzung:
        pipeline = EnhancedSearchPipeline(hybrid_search=my_search)
        results = await pipeline.search("Vergleich BU-Tarife WWK vs Allianz")

    Die Pipeline führt automatisch:
      1. Query-Dekomposition (wenn query komplex genug)
      2. Mehrfach-Suche mit HybridSearch
      3. RRF-Merge der Ergebnisse
      4. Corrective RAG Relevanz-Check
      5. Frequency-Boost
      6. Finale Sortierung + Top-K
    """

    def __init__(
        self,
        hybrid_search: Any,  # HybridSearch
        *,
        decomposer: QueryDecomposer | None = None,
        corrective: CorrectiveRAG | None = None,
        frequency_tracker: FrequencyTracker | None = None,
        enable_decomposition: bool = True,
        enable_correction: bool = True,
        enable_frequency_boost: bool = True,
    ) -> None:
        self._search = hybrid_search
        self._decomposer = decomposer or QueryDecomposer()
        self._corrective = corrective or CorrectiveRAG()
        self._frequency = frequency_tracker or FrequencyTracker()
        self._enable_decomposition = enable_decomposition
        self._enable_correction = enable_correction
        self._enable_frequency = enable_frequency_boost

    @property
    def frequency_tracker(self) -> FrequencyTracker:
        return self._frequency

    async def search(
        self,
        query: str,
        *,
        top_k: int = 6,
        tier_filter: MemoryTier | None = None,
    ) -> list[MemorySearchResult]:
        """Führt die vollständige Enhanced-Search-Pipeline aus.

        Args:
            query: Nutzerfrage.
            top_k: Maximale Ergebnisse.
            tier_filter: Optionaler Tier-Filter.

        Returns:
            Optimierte Suchergebnisse.
        """
        # ── Phase 1: Query-Dekomposition ──
        if self._enable_decomposition:
            decomposed = self._decomposer.decompose(query)
            sub_queries = decomposed.sub_queries
        else:
            sub_queries = [query]

        # ── Phase 2: Multi-Query Hybrid Search ──
        all_results: list[list[MemorySearchResult]] = []
        for sq in sub_queries:
            results = await self._search.search(
                sq, top_k=top_k * 2, tier_filter=tier_filter,
            )
            all_results.append(results)

        # ── Phase 3: RRF Merge (oder direkt wenn nur 1 Query) ──
        if len(all_results) > 1:
            merged = reciprocal_rank_fusion(all_results, top_n=top_k * 2)
        elif all_results:
            merged = all_results[0]
        else:
            merged = []

        # ── Phase 4: Corrective RAG ──
        if self._enable_correction and merged:
            verdict = self._corrective.evaluate_relevance_heuristic(query, merged)

            if verdict.needs_retry:
                # Alternative Queries generieren und erneut suchen
                alternatives = self._corrective.generate_alternative_queries(query)
                for alt_query in alternatives[:2]:
                    retry_results = await self._search.search(
                        alt_query, top_k=top_k, tier_filter=tier_filter,
                    )
                    all_results.append(retry_results)

                # Erneut mergen mit allen Ergebnissen
                merged = reciprocal_rank_fusion(all_results, top_n=top_k * 2)
            else:
                # Nur relevante Ergebnisse behalten
                merged = verdict.relevant_results

        # ── Phase 5: Frequency Boost ──
        if self._enable_frequency and merged:
            merged = self._frequency.apply_boost(merged)
            # Zugriffe tracken
            self._frequency.record_accesses([r.chunk.id for r in merged[:top_k]])

        # ── Phase 6: Final Top-K ──
        return merged[:top_k]

    def stats(self) -> dict[str, Any]:
        """Pipeline-Statistiken."""
        return {
            "decomposition_enabled": self._enable_decomposition,
            "correction_enabled": self._enable_correction,
            "frequency_boost_enabled": self._enable_frequency,
            "frequency": self._frequency.stats(),
        }
