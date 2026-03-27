"""Enhanced Retrieval: Advanced RAG techniques on top of hybrid search.

Baut auf der bestehenden HybridSearch (BM25+Vektor+Graph) auf und
ergänzt fünf wesentliche Fähigkeiten:

1. Query-Dekomposition: Komplexe Fragen in Teilfragen zerlegen
1. Reciprocal Rank Fusion (RRF): Multi-Query-Ergebnisse intelligent mergen
1. Corrective RAG: Relevanz-Prüfung mit automatischem Re-Retrieval
1. Frequenz-Gewichtung: Oft referenzierte Chunks höher ranken
1. Episodenkompression: Alte Episoden zu Zusammenfassungen verdichten

Architektur:
User-Query → QueryDecomposer → [sub_query_1, sub_query_2, …]
→ HybridSearch × N Queries
→ RRF-Merge → Vorläufige Ergebnisse
→ CorrectionStage → Relevanz-Check
→ FrequencyBoost → Finale Ergebnisse

Bibel-Referenz: §4.7 (Enhanced Retrieval), §4.3 (Episodic Compression)
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Any

from jarvis.models import MemorySearchResult, MemoryTier

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger("jarvis.memory.enhanced_retrieval")

# ============================================================================
# German Named-Entity Extraction (Heuristic)
# ============================================================================

#
# Problem: In German ALL nouns are capitalized, not just
# proper nouns. A simple regex on uppercase letters recognizes "Tisch",
# "Hund" and "Wetter" as entities -- this adds noise to the Knowledge Graph.
#
# Solution: Two-stage filter
# 1. Find capitalized words (candidates)
# 2. Filter out common German nouns via stoplist
# 3. Apply heuristics for likely Named Entities

# Common German everyday nouns that are NOT Named Entities.
# Extended list with the ~300 most frequent German nouns from
# word frequency lists (DeReWo, SUBTLEX-DE). Umlauts normalized.
_GERMAN_COMMON_NOUNS: frozenset[str] = frozenset(
    {
        # Abstract concepts
        "Anfang",
        "Angebot",
        "Angst",
        "Antwort",
        "Arbeit",
        "Art",
        "Aufgabe",
        "Augenblick",
        "Aussage",
        "Bedeutung",
        "Bedingung",
        "Beginn",
        "Beispiel",
        "Bereich",
        "Bericht",
        "Bewegung",
        "Beziehung",
        "Bild",
        "Chance",
        "Ding",
        "Druck",
        "Eindruck",
        "Einfluss",
        "Ende",
        "Entscheidung",
        "Erfahrung",
        "Erfolg",
        "Ergebnis",
        "Erinnerung",
        "Fall",
        "Frage",
        "Freiheit",
        "Freude",
        "Friede",
        "Gedanke",
        "Gefahr",
        "Geschichte",
        "Gesellschaft",
        "Gesetz",
        "Gewalt",
        "Glaube",
        "Grund",
        "Hilfe",
        "Hinweis",
        "Hoffnung",
        "Idee",
        "Information",
        "Interesse",
        "Kampf",
        "Kenntnis",
        "Kosten",
        "Kraft",
        "Krieg",
        "Kritik",
        "Kultur",
        "Kunst",
        "Lage",
        "Leben",
        "Leistung",
        "Liebe",
        "Lust",
        "Macht",
        "Meinung",
        "Menge",
        "Mittel",
        "Moment",
        "Nacht",
        "Natur",
        "Notwendigkeit",
        "Ordnung",
        "Pflicht",
        "Plan",
        "Politik",
        "Praxis",
        "Prinzip",
        "Problem",
        "Programm",
        "Projekt",
        "Prozess",
        "Punkt",
        "Recht",
        "Rede",
        "Regel",
        "Reihe",
        "Richtung",
        "Risiko",
        "Rolle",
        "Sache",
        "Schritt",
        "Schuld",
        "Schutz",
        "Seite",
        "Sicherheit",
        "Sinn",
        "Situation",
        "Sorge",
        "Spass",
        "Sprache",
        "Staat",
        "Stelle",
        "Stimme",
        "Streit",
        "Stunde",
        "System",
        "Tag",
        "Tat",
        "Teil",
        "Thema",
        "Tod",
        "Traum",
        "Trend",
        "Ursache",
        "Urteil",
        "Verantwortung",
        "Verhalten",
        "Versuch",
        "Vertrauen",
        "Vorstellung",
        "Wahl",
        "Wahrheit",
        "Weg",
        "Weise",
        "Welt",
        "Wert",
        "Wirkung",
        "Wissen",
        "Woche",
        "Wort",
        "Wunsch",
        "Zahl",
        "Zeit",
        "Ziel",
        "Zukunft",
        "Zusammenhang",
        "Zustand",
        # Physical objects and places
        "Auge",
        "Auto",
        "Bau",
        "Baum",
        "Bett",
        "Blatt",
        "Blick",
        "Blut",
        "Boden",
        "Brief",
        "Brot",
        "Buch",
        "Computer",
        "Dach",
        "Dorf",
        "Erde",
        "Essen",
        "Fenster",
        "Feuer",
        "Film",
        "Foto",
        "Garten",
        "Geld",
        "Glas",
        "Haar",
        "Hand",
        "Haus",
        "Herz",
        "Hund",
        "Insel",
        "Karte",
        "Kind",
        "Kirche",
        "Klasse",
        "Kopf",
        "Kreis",
        "Land",
        "Licht",
        "Liste",
        "Luft",
        "Markt",
        "Meer",
        "Messer",
        "Morgen",
        "Mund",
        "Musik",
        "Nase",
        "Nummer",
        "Ohr",
        "Papier",
        "Platz",
        "Post",
        "Raum",
        "Ring",
        "Schloss",
        "Schule",
        "Sonne",
        "Spiel",
        "Stadt",
        "Stein",
        "Stern",
        "Strasse",
        "Stuhl",
        "Tisch",
        "Tuer",
        "Turm",
        "Uhr",
        "Wald",
        "Wand",
        "Wasser",
        "Wetter",
        "Wohnung",
        "Wolke",
        "Zeitung",
        "Zimmer",
        "Zug",
        # People (generic)
        "Arzt",
        "Bauer",
        "Bruder",
        "Chef",
        "Eltern",
        "Feind",
        "Frau",
        "Freund",
        "Gast",
        "Herr",
        "Junge",
        "Kellner",
        "Lehrer",
        "Leute",
        "Mann",
        "Mutter",
        "Nachbar",
        "Onkel",
        "Partner",
        "Schwester",
        "Soldat",
        "Sohn",
        "Tochter",
        "Vater",
        "Volk",
        # Insurance and finance terms (domain-specific, frequent)
        "Antrag",
        "Beitrag",
        "Berater",
        "Beratung",
        "Betrag",
        "Deckung",
        "Dokument",
        "Einnahme",
        "Garantie",
        "Kapital",
        "Konto",
        "Kredit",
        "Kunde",
        "Laufzeit",
        "Police",
        "Praemie",
        "Provision",
        "Rate",
        "Rente",
        "Rendite",
        "Steuer",
        "Tarif",
        "Umsatz",
        "Versicherung",
        "Vertrag",
        "Vorsorge",
        "Zahlung",
        "Zinsen",
        # IT / Technology (frequent in mixed-language texts)
        "Abfrage",
        "Anwendung",
        "Code",
        "Datei",
        "Daten",
        "Datenbank",
        "Fehler",
        "Funktion",
        "Klick",
        "Meldung",
        "Modul",
        "Netzwerk",
        "Nutzer",
        "Schnittstelle",
        "Server",
        "Software",
        "Speicher",
        "Tabelle",
        "Update",
        "Verbindung",
        "Version",
        "Zugang",
        "Zugriff",
    }
)

# Pattern for Named-Entity candidates: uppercase letter + at least 2 lowercase letters
_ENTITY_CANDIDATE_RE = re.compile(r"\b[A-ZÄÖÜ][a-zäöüß]{2,}\b")

# Pattern for strong NE signals (multi-word entities, CamelCase etc.)
_MULTI_WORD_ENTITY_RE = re.compile(r"\b[A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)+\b")

# German articles and pronouns that are irrelevant in multi-word entities
_GERMAN_ARTICLES: frozenset[str] = frozenset(
    {
        "Der",
        "Die",
        "Das",
        "Den",
        "Dem",
        "Des",
        "Ein",
        "Eine",
        "Eines",
        "Einem",
        "Einen",
        "Einer",
        "Kein",
        "Keine",
        "Keines",
        "Keinem",
        "Keinen",
        "Keiner",
        "Ihr",
        "Ihre",
        "Sein",
        "Seine",
        "Mein",
        "Meine",
        "Jeder",
        "Jede",
        "Jedes",
        "Dieser",
        "Diese",
        "Dieses",
        "Welcher",
        "Welche",
        "Welches",
        "Alle",
        "Viele",
        "Einige",
    }
)

# Trailing-punctuation pattern for word cleanup
_TRAILING_PUNCT_RE = re.compile(r'[.,;:!?\"\'"()]+$')


def _clean_word(word: str) -> str:
    """Remove trailing punctuation from a word.

    'Berlin.' -> 'Berlin', 'Allianz,' -> 'Allianz'
    """
    return _TRAILING_PUNCT_RE.sub("", word)


def _extract_german_entities(text: str) -> set[str]:
    """Extrahiert Named Entities aus deutschem Text.

    Strategie:
      1. Alle großgeschriebenen Wörter als Kandidaten
      2. Häufige Alltagsnomen herausfiltern
      3. Mehrwort-Entitäten bevorzugen ("Deutsche Bank", "Rotes Kreuz")
      4. Am Satzanfang stehende Wörter ignorieren (dort ist
         Großschreibung grammatikalisch bedingt, nicht semantisch)
      5. Artikel und Pronomen aus Mehrwort-Entitäten entfernen
      6. Satzzeichen an Wortenden bereinigen ("Berlin." -> "Berlin")

    Returns:
        Menge erkannter Named Entities.
    """
    entities: set[str] = set()

    # Multi-word entities first (highest confidence)
    for match in _MULTI_WORD_ENTITY_RE.finditer(text):
        candidate = match.group()
        # Remove articles/pronouns at the beginning
        words = candidate.split()
        while words and words[0] in _GERMAN_ARTICLES:
            words.pop(0)
        if len(words) < 2:
            continue  # No longer multi-word after article removal
        cleaned = " ".join(words)
        # Only if at least one word is not a common noun
        if any(w not in _GERMAN_COMMON_NOUNS for w in words):
            entities.add(cleaned)

    # Single-word entities
    sentences = re.split(r"[.!?]\s+", text)
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        words = sent.split()
        for i, raw_word in enumerate(words):
            # Remove trailing punctuation ("Berlin." -> "Berlin")
            word = _clean_word(raw_word)

            # Only check capitalized words
            if not _ENTITY_CANDIDATE_RE.fullmatch(word):
                continue

            # Skip word at sentence start -- capitalization
            # is grammatical there, not semantic
            if i == 0:
                continue

            # Skip articles and pronouns
            if word in _GERMAN_ARTICLES:
                continue

            # Filter out common nouns
            if word in _GERMAN_COMMON_NOUNS:
                continue

            # Everything remaining is likely a proper name,
            # product, company, place, etc.
            entities.add(word)

    return entities


def _count_german_entities_in_text(text: str) -> int:
    """Zählt Named Entities in einem Text (für Scoring).

    Schnellere Variante von _extract_german_entities die nur zählt
    statt eine Menge aufzubauen.
    """
    count = 0
    sentences = re.split(r"[.!?]\s+", text)
    for sent in sentences:
        words = sent.strip().split()
        for i, raw_word in enumerate(words):
            if i == 0:
                continue
            word = _clean_word(raw_word)
            if not _ENTITY_CANDIDATE_RE.fullmatch(word):
                continue
            if word in _GERMAN_ARTICLES:
                continue
            if word not in _GERMAN_COMMON_NOUNS:
                count += 1
    return count


# ============================================================================
# 1. Query Decomposition
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
      - Konjunktionen splitten ("X und Y" -> "X", "Y")
      - Vergleiche erkennen ("Unterschied zwischen A und B" -> "A", "B")
      - Aspekte extrahieren ("Vorteile und Nachteile von X" -> "Vorteile X", "Nachteile X")
      - Zeitliche Aspekte ("X früher vs heute" -> "X historisch", "X aktuell")
    """

    # Conjunction patterns
    _CONJUNCTION_PATTERNS = [
        re.compile(r"(.+?)\s+(?:und|sowie|als auch|und auch)\s+(.+)", re.IGNORECASE),
    ]

    # Comparison patterns
    _COMPARISON_PATTERNS = [
        re.compile(
            r"(?:unterschied|vergleich|differenz)\s+zwischen\s+(.+?)\s+und\s+(.+)",
            re.IGNORECASE,
        ),
        re.compile(r"(.+?)\s+(?:vs\.?|versus|gegen|oder)\s+(.+)", re.IGNORECASE),
    ]

    # Aspect patterns
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

        # Try patterns in priority order
        # 1. Comparisons
        for pattern in self._COMPARISON_PATTERNS:
            match = pattern.search(query)
            if match:
                a, b = match.group(1).strip(), match.group(2).strip()
                return DecomposedQuery(
                    original=query,
                    sub_queries=[query, a, b],
                    query_type="compound",
                )

        # 2. Aspects (pros and cons)
        for pattern in self._ASPECT_PATTERNS:
            match = pattern.search(query)
            if match:
                topic = match.group(2).strip()
                return DecomposedQuery(
                    original=query,
                    sub_queries=[query, f"Vorteile {topic}", f"Nachteile {topic}"],
                    query_type="multi_aspect",
                )

        # 3. Conjunctions
        for pattern in self._CONJUNCTION_PATTERNS:
            match = pattern.search(query)
            if match:
                a, b = match.group(1).strip(), match.group(2).strip()
                # Only split if both parts are substantial (>3 words)
                if len(a.split()) >= 2 and len(b.split()) >= 2:
                    return DecomposedQuery(
                        original=query,
                        sub_queries=[query, a, b],
                        query_type="compound",
                    )

        # 4. No decomposition possible -> keep original
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
                    sub_queries=[query, *lines[:3]],  # Original immer dabei
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

    Formel: RRF_score(d) = Sigma 1 / (k + rank_i(d))

    Wobei k=60 (Standard-Konstante die stabile Rankings erzeugt).

    Args:
        result_lists: Liste von Ergebnislisten (je eine pro Sub-Query).
        k: RRF-Konstante (höher = weniger Einfluss der Rankposition).
        top_n: Max Ergebnisse (None = alle).

    Returns:
        Merged + sortierte Ergebnisliste.
    """
    # Chunk-ID -> aggregated RRF score
    rrf_scores: dict[str, float] = {}
    # Chunk-ID -> best MemorySearchResult (for metadata)
    best_results: dict[str, MemorySearchResult] = {}

    for result_list in result_lists:
        for rank, result in enumerate(result_list):
            chunk_id = result.chunk.id
            rrf_score = 1.0 / (k + rank + 1)
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + rrf_score

            # Keep the result with the highest original score
            if chunk_id not in best_results or result.score > best_results[chunk_id].score:
                best_results[chunk_id] = result

    # Sort by RRF score
    sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

    # Results with RRF score
    merged: list[MemorySearchResult] = []
    for chunk_id in sorted_ids:
        original = best_results[chunk_id]
        # New result with RRF score as main score
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
      3. Wenn <min_relevant: Alternative Query generieren -> Re-Retrieval
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

            # Word overlap check
            chunk_words = set(re.findall(r"\w+", result.chunk.text.lower()))
            overlap = len(query_words & chunk_words)
            overlap_ratio = overlap / max(len(query_words), 1)

            # Combination heuristic
            if (
                result.score >= 0.3
                or overlap_ratio >= 0.3
                or (result.score >= 0.15 and overlap_ratio >= 0.15)
            ):
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

        # Strategy 1: Keywords only (remove stopwords)
        stopwords = {
            "der",
            "die",
            "das",
            "ein",
            "eine",
            "ist",
            "sind",
            "war",
            "hat",
            "und",
            "oder",
            "aber",
            "in",
            "an",
            "auf",
            "mit",
            "von",
            "zu",
            "für",
            "über",
            "nach",
            "aus",
            "bei",
            "um",
            "wie",
            "was",
            "wer",
            "wo",
            "wann",
            "warum",
            "ich",
            "du",
            "er",
            "sie",
            "es",
            "wir",
            "mein",
            "dein",
            "sein",
            "ihr",
            "the",
            "a",
            "is",
            "are",
            "and",
            "or",
            "on",
            "at",
            "with",
            "for",
            "to",
            "of",
        }
        keywords = [w for w in words if w.lower() not in stopwords and len(w) > 2]
        if keywords and len(keywords) < len(words):
            alternatives.append(" ".join(keywords))

        # Strategy 2: First N words (when query is long)
        if len(words) > 6:
            alternatives.append(" ".join(words[:4]))

        # Strategy 3: Last N words (often the actual core)
        if len(words) > 4:
            alternatives.append(" ".join(words[-3:]))

        return alternatives


# ============================================================================
# 4. Frequency Weighting
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
        """Calculate the frequency boost for a chunk.

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
# 5. Episode Compression
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
        default_factory=lambda: datetime.now(UTC).isoformat(),
    )

    @property
    def date_range(self) -> str:
        return f"{self.start_date.isoformat()} \u2013 {self.end_date.isoformat()}"

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
            # Same week if less than 7 days apart
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
            # Split sentences
            sentences = re.split(r"[.!?]\s+", entry)
            for sent in sentences:
                sent = sent.strip()
                if len(sent) > 20:  # Ignore sentences that are too short
                    all_sentences.append(sent)

                    # Extract Named Entities (German common nouns filtered)
                    entity_set.update(_extract_german_entities(sent))

        # Rank sentences by information content
        scored: list[tuple[float, str]] = []
        for sent in all_sentences:
            score = 0.0
            # Length (normalized)
            score += min(len(sent) / 200.0, 1.0) * 0.3
            # Named Entities in sentence (German common nouns filtered)
            sent_entities = _count_german_entities_in_text(sent)
            score += min(sent_entities / 3.0, 1.0) * 0.4
            # Numbers (often important facts)
            numbers = len(re.findall(r"\d+", sent))
            score += min(numbers / 2.0, 1.0) * 0.3
            scored.append((score, sent))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Top sentences as summary
        key_sentences = []
        seen_texts: set[str] = set()
        for _score, sent in scored:
            # Avoid duplicates
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
                entries,
                start_date=start_date,
                end_date=end_date,
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
            entities = sorted(_extract_german_entities(summary))
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
                entries,
                start_date=start_date,
                end_date=end_date,
            )


# ============================================================================
# 6. Enhanced Search Pipeline (orchestrates everything)
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
        # ── Phase 1: Query decomposition ──
        if self._enable_decomposition:
            decomposed = self._decomposer.decompose(query)
            sub_queries = decomposed.sub_queries
        else:
            sub_queries = [query]

        # ── Phase 2: Multi-Query Hybrid Search ──
        all_results: list[list[MemorySearchResult]] = []
        for sq in sub_queries:
            results = await self._search.search(
                sq,
                top_k=top_k * 2,
                tier_filter=tier_filter,
            )
            all_results.append(results)

        # ── Phase 3: RRF Merge (or direct if only 1 query) ──
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
                # Generate alternative queries and search again
                alternatives = self._corrective.generate_alternative_queries(query)
                for alt_query in alternatives[:2]:
                    retry_results = await self._search.search(
                        alt_query,
                        top_k=top_k,
                        tier_filter=tier_filter,
                    )
                    all_results.append(retry_results)

                # Re-merge with all results
                merged = reciprocal_rank_fusion(all_results, top_n=top_k * 2)
            else:
                # Keep only relevant results
                merged = verdict.relevant_results

        # ── Phase 5: Frequency Boost ──
        if self._enable_frequency and merged:
            merged = self._frequency.apply_boost(merged)
            # Track accesses
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
