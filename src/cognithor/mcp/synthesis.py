"""Knowledge Synthesis — Quellen-Fusion, Widerspruchserkennung, Zeitlinien.

Orchestriert Vault, Memory, Web und LLM zu einer zusammenhaengenden
Wissensanalyse. Transformiert isolierte Informationsfragmente in
ein kohaerentes, temporales Verstaendnis.

Tools:
  - knowledge_synthesize: Vollstaendige Wissenssynthese zu einem Thema
  - knowledge_contradictions: Widersprueche zwischen gespeichertem und neuem Wissen erkennen
  - knowledge_timeline: Temporale Kette zu einem Thema aufbauen
  - knowledge_gaps: Wissensluecken identifizieren und Recherche vorschlagen

Architektur:
  KnowledgeSynthesizer orchestriert intern:
  1. MemoryTools.search_memory() — Gespeichertes Wissen abrufen
  2. MemoryTools.get_entity() — Entitaeten und Relationen laden
  3. MemoryTools.get_recent_episodes() — Episodisches Gedaechtnis
  4. VaultTools.vault_search() — Vault-Notizen durchsuchen
  5. WebTools.web_search() / search_and_read() — Frisches Web-Wissen
  6. LLM — Synthese, Widerspruchserkennung, Zeitlinien, Lueckenanalyse
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

from cognithor.i18n import t
from cognithor.utils.logging import get_logger

if TYPE_CHECKING:
    from cognithor.config import CognithorConfig

log = get_logger(__name__)

__all__ = [
    "KnowledgeSynthesizer",
    "register_synthesis_tools",
]

# ── Constants (defaults, overridable via config.synthesis.*) ──────────

_DEFAULT_MAX_SOURCE_CHARS = 4000
_DEFAULT_MAX_CONTEXT_CHARS = 25000


class KnowledgeSynthesizer:
    """Orchestriert Wissensquellen und synthetisiert kohaerentes Verstaendnis.

    Injizierte Abhaengigkeiten (via Setter):
      - LLM-Funktion: async (prompt, model) -> str
      - MemoryTools: sync search_memory(), get_entity(), get_recent_episodes()
      - VaultTools: async vault_search(), vault_save()
      - WebTools: async web_search(), search_and_read()
    """

    def __init__(self, config: CognithorConfig | None = None) -> None:
        self._llm_fn: Any = None
        self._llm_model: str = ""
        self._memory_tools: Any = None
        self._vault_tools: Any = None
        self._web_tools: Any = None

        # Read limits from config (with safe defaults)
        _synth = getattr(config, "synthesis", None)
        self._max_source_chars: int = getattr(_synth, "max_source_chars", _DEFAULT_MAX_SOURCE_CHARS)
        self._max_context_chars: int = getattr(
            _synth, "max_context_chars", _DEFAULT_MAX_CONTEXT_CHARS
        )

    # ── Dependency Injection ─────────────────────────────────────────

    def _set_llm_fn(self, llm_fn: Any, model_name: str = "") -> None:
        """Injiziert die LLM-Funktion.

        Args:
            llm_fn: Async-Funktion: (prompt: str, model: str) -> str
            model_name: Standard-Modellname.
        """
        self._llm_fn = llm_fn
        self._llm_model = model_name

    def _set_memory_tools(self, memory_tools: Any) -> None:
        """Injiziert MemoryTools-Instanz (synchrone Methoden)."""
        self._memory_tools = memory_tools

    def _set_vault_tools(self, vault_tools: Any) -> None:
        """Injiziert VaultTools-Instanz (async Methoden)."""
        self._vault_tools = vault_tools

    def _set_web_tools(self, web_tools: Any) -> None:
        """Injiziert WebTools-Instanz (async Methoden)."""
        self._web_tools = web_tools

    def _check_ready(self) -> str | None:
        """Prueft ob alle Abhaengigkeiten injiziert sind.

        Returns:
            Fehlermeldung oder None wenn bereit.
        """
        if self._llm_fn is None:
            return t("tools.synthesis_no_llm")
        if self._memory_tools is None:
            return t("tools.synthesis_no_memory")
        return None

    # ── Source gathering ─────────────────────────────────────────────

    async def _gather_sources(
        self,
        topic: str,
        *,
        include_web: bool = True,
        include_vault: bool = True,
        include_memory: bool = True,
        include_episodes: bool = True,
        web_results: int = 3,
    ) -> dict[str, str]:
        """Sammelt alle verfuegbaren Quellen zu einem Thema.

        Returns:
            Dict mit Quelltyp → Inhalt:
            {
                "memory": "...",
                "entities": "...",
                "episodes": "...",
                "vault": "...",
                "web": "...",
            }
        """
        sources: dict[str, str] = {}

        # 1. Search semantic memory (synchronous)
        if include_memory and self._memory_tools:
            try:
                memory_result = self._memory_tools.search_memory(topic, top_k=8)
                if memory_result and "Fehler" not in memory_result and "Keine" not in memory_result:
                    sources["memory"] = _truncate(memory_result, self._max_source_chars)
                    log.info("synthesis_memory_found", topic=topic[:50], chars=len(memory_result))
            except Exception as exc:
                log.debug("synthesis_memory_error", error=str(exc))

        # 2. Load entities from knowledge graph (synchronous)
        if include_memory and self._memory_tools:
            try:
                # Extract main terms from topic for entity search
                keywords = _extract_keywords(topic)
                entity_parts: list[str] = []
                for kw in keywords[:3]:
                    entity_result = self._memory_tools.get_entity(kw)
                    if entity_result and "Keine Entität" not in entity_result:
                        entity_parts.append(entity_result)
                if entity_parts:
                    sources["entities"] = _truncate(
                        "\n\n".join(entity_parts), self._max_source_chars
                    )
                    log.info("synthesis_entities_found", topic=topic[:50], count=len(entity_parts))
            except Exception as exc:
                log.debug("synthesis_entity_error", error=str(exc))

        # 3. Episodic memory — last 7 days (synchronous)
        if include_episodes and self._memory_tools:
            try:
                episodes = self._memory_tools.get_recent_episodes(days=7)
                if episodes and "Keine Episodic" not in episodes:
                    # Keep only entries relevant to the topic
                    relevant = _filter_relevant_text(episodes, topic)
                    if relevant:
                        sources["episodes"] = _truncate(relevant, self._max_source_chars)
                        log.info("synthesis_episodes_found", topic=topic[:50])
            except Exception as exc:
                log.debug("synthesis_episodes_error", error=str(exc))

        # 4. Search vault (async)
        if include_vault and self._vault_tools:
            try:
                vault_result = await self._vault_tools.vault_search(topic, limit=5)
                if vault_result and "Keine Notizen" not in vault_result:
                    sources["vault"] = _truncate(vault_result, self._max_source_chars)
                    log.info("synthesis_vault_found", topic=topic[:50])
            except Exception as exc:
                log.debug("synthesis_vault_error", error=str(exc))

        # 5. Fetch fresh web knowledge (async)
        if include_web and self._web_tools:
            try:
                web_result = await self._web_tools.search_and_read(
                    topic,
                    num_results=web_results,
                    cross_check=True,
                )
                if web_result and "Keine" not in web_result[:30]:
                    sources["web"] = _truncate(web_result, self._max_source_chars * 2)
                    log.info("synthesis_web_found", topic=topic[:50], chars=len(web_result))
            except Exception as exc:
                log.debug("synthesis_web_error", error=str(exc))

        return sources

    def _format_source_context(self, sources: dict[str, str]) -> str:
        """Formatiert gesammelte Quellen als LLM-Kontext."""
        parts: list[str] = []

        labels = {
            "memory": "GESPEICHERTES WISSEN (Semantic Memory)",
            "entities": "BEKANNTE ENTITÄTEN (Knowledge Graph)",
            "episodes": "EPISODISCHES GEDÄCHTNIS (letzte Tage)",
            "vault": "VAULT-NOTIZEN (Knowledge Vault)",
            "web": "AKTUELLE WEB-RECHERCHE",
        }

        for key in ("memory", "entities", "episodes", "vault", "web"):
            if sources.get(key):
                parts.append(f"### {labels.get(key, key.upper())}\n{sources[key]}")

        combined = "\n\n---\n\n".join(parts)

        # Limit total size
        if len(combined) > self._max_context_chars:
            combined = combined[: self._max_context_chars] + "\n\n[... Kontext gekürzt]"

        return combined

    # ── Tool: knowledge_synthesize ───────────────────────────────────

    async def knowledge_synthesize(
        self,
        topic: str,
        include_web: bool = True,
        depth: str = "standard",
        language: str = "de",
        save_to_vault: bool = False,
    ) -> str:
        """Vollstaendige Wissenssynthese zu einem Thema.

        Sammelt alle Quellen (Memory, Vault, Web), erkennt Widersprueche,
        baut Zeitlinien, bewertet Konfidenz und identifiziert Wissensluecken.

        Args:
            topic: Das zu synthetisierende Thema oder die Fragestellung.
            include_web: Frische Web-Recherche einbeziehen (Default: True).
            depth: Tiefe: 'quick' (nur Memory+Vault), 'standard' (+ Web),
                   'deep' (+ mehr Web-Ergebnisse, detailliertere Analyse).
            language: Ausgabesprache ('de' oder 'en').
            save_to_vault: Synthese-Ergebnis im Vault speichern.

        Returns:
            Strukturierte Wissenssynthese als Markdown.
        """
        error = self._check_ready()
        if error:
            return error

        if not topic.strip():
            return t("tools.synthesis_error_no_topic")

        web_results = {"quick": 0, "standard": 3, "deep": 5}.get(depth, 3)
        do_web = include_web and depth != "quick"

        # 1. Gather sources
        sources = await self._gather_sources(
            topic,
            include_web=do_web,
            web_results=web_results,
        )

        if not sources:
            return t("tools.synthesis_no_info", topic=topic)

        # 2. Format context
        context = self._format_source_context(sources)
        source_summary = ", ".join(f"{k} ({len(v)} Zeichen)" for k, v in sources.items())

        # 3. Build synthesis prompt
        prompt = _build_synthesis_prompt(topic, context, depth, language, list(sources.keys()))

        # 4. Call LLM
        try:
            synthesis = await self._llm_fn(prompt, self._llm_model)
        except Exception as exc:
            log.error("synthesis_llm_failed", topic=topic[:50], error=str(exc))
            return t("tools.synthesis_llm_error", exc=exc)

        # 5. Append metadata
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
        footer = t("tools.synthesis_footer", now=now, sources=source_summary, depth=depth)
        result = synthesis + footer

        # 6. Optionally save to vault
        if save_to_vault and self._vault_tools:
            try:
                vault_title = f"Synthese: {topic[:60]}"
                await self._vault_tools.vault_save(
                    title=vault_title,
                    content=result,
                    tags=(
                        "synthese, "
                        f"{_extract_keywords(topic)[0] if _extract_keywords(topic) else 'wissen'}"
                    ),
                    folder="knowledge",
                )
                result += t("tools.synthesis_vault_saved", vault_title=vault_title)
            except Exception as vault_exc:
                log.warning("synthesis_vault_save_failed", error=str(vault_exc))

        log.info("synthesis_complete", topic=topic[:50], sources=len(sources), chars=len(result))
        return cast("str", result)

    # ── Tool: knowledge_contradictions ───────────────────────────────

    async def knowledge_contradictions(
        self,
        topic: str,
        language: str = "de",
    ) -> str:
        """Erkennt Widersprueche zwischen gespeichertem und neuem Wissen.

        Vergleicht was Jarvis gespeichert hat (Memory + Vault) mit
        aktuellen Web-Informationen und identifiziert Diskrepanzen.

        Args:
            topic: Thema fuer die Widerspruchsanalyse.
            language: Ausgabesprache.

        Returns:
            Liste der Widersprueche mit Quellen und Bewertung.
        """
        error = self._check_ready()
        if error:
            return error

        if not topic.strip():
            return t("tools.synthesis_error_no_topic")

        # Gather stored knowledge (without web)
        stored_sources = await self._gather_sources(
            topic,
            include_web=False,
            include_episodes=True,
        )

        if not stored_sources:
            return t("tools.synthesis_no_stored_info", topic=topic)

        # Gather current web knowledge
        web_sources = await self._gather_sources(
            topic,
            include_memory=False,
            include_vault=False,
            include_episodes=False,
            include_web=True,
            web_results=4,
        )

        if not web_sources:
            return t("tools.synthesis_no_web_info", topic=topic)

        stored_context = self._format_source_context(stored_sources)
        web_context = self._format_source_context(web_sources)

        prompt = _build_contradiction_prompt(topic, stored_context, web_context, language)

        try:
            analysis = await self._llm_fn(prompt, self._llm_model)
        except Exception as exc:
            return t("tools.synthesis_contradiction_error", exc=exc)

        log.info("contradiction_analysis_complete", topic=topic[:50])
        return cast("str", analysis)

    # ── Tool: knowledge_timeline ─────────────────────────────────────

    async def knowledge_timeline(
        self,
        topic: str,
        language: str = "de",
    ) -> str:
        """Baut eine temporale Kette zu einem Thema auf.

        Rekonstruiert die Chronologie aus gespeichertem Wissen,
        Vault-Notizen und Web-Recherche.

        Args:
            topic: Thema fuer die Zeitlinie.
            language: Ausgabesprache.

        Returns:
            Chronologische Zeitlinie mit Kausalverbindungen.
        """
        error = self._check_ready()
        if error:
            return error

        if not topic.strip():
            return t("tools.synthesis_error_no_topic")

        sources = await self._gather_sources(
            topic,
            include_web=True,
            web_results=4,
        )

        if not sources:
            return t("tools.synthesis_no_timeline_info", topic=topic)

        context = self._format_source_context(sources)
        prompt = _build_timeline_prompt(topic, context, language)

        try:
            timeline = await self._llm_fn(prompt, self._llm_model)
        except Exception as exc:
            return t("tools.synthesis_timeline_error", exc=exc)

        log.info("timeline_complete", topic=topic[:50])
        return cast("str", timeline)

    # ── Tool: knowledge_gaps ─────────────────────────────────────────

    async def knowledge_gaps(
        self,
        topic: str,
        language: str = "de",
    ) -> str:
        """Identifiziert Wissensluecken und schlaegt Recherchen vor.

        Analysiert was Jarvis zu einem Thema weiss und was fehlt.
        Gibt konkrete Such-Vorschlaege zurueck.

        Args:
            topic: Thema fuer die Lueckenanalyse.
            language: Ausgabesprache.

        Returns:
            Wissensluecken mit priorisierten Recherche-Vorschlaegen.
        """
        error = self._check_ready()
        if error:
            return error

        if not topic.strip():
            return t("tools.synthesis_error_no_topic")

        # Only stored knowledge — no web, to identify gaps
        sources = await self._gather_sources(
            topic,
            include_web=False,
            include_episodes=True,
        )

        context = (
            self._format_source_context(sources) if sources else "KEINE INFORMATIONEN VORHANDEN."
        )

        prompt = _build_gaps_prompt(topic, context, language)

        try:
            gaps = await self._llm_fn(prompt, self._llm_model)
        except Exception as exc:
            return t("tools.synthesis_gaps_error", exc=exc)

        log.info("gap_analysis_complete", topic=topic[:50])
        return cast("str", gaps)


# ── Prompt builders ────────────────────────────────────────────────────────


def _build_synthesis_prompt(
    topic: str,
    context: str,
    depth: str,
    language: str,
    source_types: list[str],
) -> str:
    """Baut den LLM-Prompt fuer die vollstaendige Wissenssynthese."""
    lang = "Antworte auf Deutsch." if language == "de" else "Answer in English."

    source_info = ", ".join(source_types) if source_types else "keine"

    detail_level = {
        "quick": "Kurz und prägnant, Fokus auf Kernaussagen.",
        "standard": "Ausgewogen, mit Quellenvergleich und Konfidenz.",
        "deep": "Detailliert, mit vollständiger Analyse aller Aspekte.",
    }.get(depth, "Ausgewogen.")

    return f"""Du bist ein Analyst fuer Wissenssynthese. {lang}

AUFGABE: Erstelle eine umfassende Wissenssynthese zum Thema: "{topic}"

Dir stehen folgende Informationsquellen zur Verfuegung: {source_info}
Detailtiefe: {detail_level}

Erstelle deine Synthese mit GENAU diesen Abschnitten:

## Wissenssynthese: {topic}

### Kernerkenntnisse
Die 3-7 wichtigsten Fakten/Erkenntnisse, priorisiert nach Relevanz.
Jede Erkenntnis mit Konfidenz-Indikator:
- ★★★ = durch mehrere Quellen bestaetigt
- ★★☆ = durch eine zuverlaessige Quelle gestuetzt
- ★☆☆ = nur eine Quelle oder unsicher

### Quellenvergleich
Wo stimmen die Quellen ueberein? Wo gibt es Abweichungen?
Nenne konkret, welche Quelle was sagt.

### Widersprueche & Diskrepanzen
Falls Widersprueche zwischen gespeichertem Wissen und neuen Informationen bestehen:
- Was war bekannt vs. was ist neu?
- Welche Information ist wahrscheinlich korrekt und warum?

### Zeitliche Entwicklung
Falls zeitliche Aspekte erkennbar: Was hat sich veraendert?
Kausalkette aufzeigen: X fuehrte zu Y, daraus folgt Z.

### Wissensluecken
Was fehlt noch? Welche Fragen sind offen?
Konkrete Vorschlaege fuer Nachrecherche (als Suchbegriffe).

### Fazit & Empfehlung
2-3 Saetze: Was ist der aktuelle Stand? Was sollte der Nutzer als naechstes tun?

INFORMATIONSQUELLEN:
---
{context}
---

REGELN:
- Erfinde KEINE Informationen. Basiere alles auf den gelieferten Quellen.
- Wenn Quellen fehlen oder widerspruechlich sind, sage das explizit.
- Markiere Unsicherheiten transparent.
- Bevorzuge bei Widerspruechen die Primaerquelle (Web > Memory wenn aktueller)."""


def _build_contradiction_prompt(
    topic: str,
    stored_context: str,
    web_context: str,
    language: str,
) -> str:
    """Baut den Prompt fuer die Widerspruchsanalyse."""
    lang = "Antworte auf Deutsch." if language == "de" else "Answer in English."

    return f"""Du bist ein Analyst fuer Widerspruchserkennung. {lang}

AUFGABE: Vergleiche das gespeicherte Wissen mit aktuellen Web-Informationen
zum Thema: "{topic}"

Identifiziere ALLE Widersprueche, Diskrepanzen und veralteten Informationen.

Struktur deiner Analyse:

## Widerspruchsanalyse: {topic}

### Bestaetigte Fakten
Informationen die sowohl gespeichert als auch aktuell bestaetigt sind.

### Widersprueche
Fuer jeden Widerspruch:
| Aspekt | Gespeichert | Aktuell (Web) | Bewertung |
|--------|-------------|---------------|-----------|
| ...    | ...         | ...           | Welche Version ist korrekt? |

### Veraltete Informationen
Was im Speicher steht, aber nicht mehr aktuell ist.
Empfehlung: Aktualisieren / Loeschen / Beibehalten mit Vermerk.

### Neue Erkenntnisse
Informationen aus dem Web, die noch nicht gespeichert waren.
Empfehlung: Speichern / Ignorieren.

GESPEICHERTES WISSEN:
---
{stored_context}
---

AKTUELLE WEB-INFORMATIONEN:
---
{web_context}
---

Sei praezise. Nenne konkret was gespeichert war und was die Web-Quellen sagen."""


def _build_timeline_prompt(
    topic: str,
    context: str,
    language: str,
) -> str:
    """Baut den Prompt fuer den Zeitlinien-Aufbau."""
    lang = "Antworte auf Deutsch." if language == "de" else "Answer in English."

    return f"""Du bist ein Analyst fuer temporale Muster. {lang}

AUFGABE: Erstelle eine chronologische Zeitlinie zum Thema: "{topic}"

Extrahiere alle Datumspunkte, Ereignisse und Entwicklungen aus den Quellen
und ordne sie chronologisch an.

Struktur:

## Zeitlinie: {topic}

### Chronologie
Fuer jeden Zeitpunkt/Ereignis:
- **[Datum/Zeitraum]** — Was passierte. *(Quelle: ...)*
  → Folge/Auswirkung

### Kausalketten
Zeichne die Ursache-Wirkungs-Beziehungen auf:
1. Ereignis A → fuehrte zu B → bewirkte C

### Trend & Prognose
Basierend auf der Entwicklung:
- Erkennbarer Trend (aufwaerts/abwaerts/stabil/volatil)
- Moegliche naechste Entwicklungen (mit Vorbehalt kennzeichnen)

INFORMATIONSQUELLEN:
---
{context}
---

Wenn kein genaues Datum erkennbar ist, nutze ungefaehre Zeitangaben
("Anfang 2026", "Q4 2025", etc.). Erfinde keine Daten."""


def _build_gaps_prompt(
    topic: str,
    context: str,
    language: str,
) -> str:
    """Baut den Prompt fuer die Wissensluecken-Analyse."""
    lang = "Antworte auf Deutsch." if language == "de" else "Answer in English."

    return f"""Du bist ein Analyst fuer Wissensmanagement. {lang}

AUFGABE: Analysiere, was zum Thema "{topic}" bekannt ist und was fehlt.

Bewerte die Vollstaendigkeit des vorhandenen Wissens und identifiziere
systematisch alle Luecken.

Struktur:

## Wissensstand: {topic}

### Abgedeckte Bereiche
Was ist bereits gut dokumentiert? (kurze Auflistung)

### Identifizierte Wissensluecken
Fuer jede Luecke:
| # | Fehlende Information | Prioritaet | Recherche-Vorschlag |
|---|---------------------|-----------|---------------------|
| 1 | ...                 | HOCH/MITTEL/NIEDRIG | Konkrete Suchbegriffe |

### Empfohlene Recherchen
Priorisierte Liste konkreter Suchbegriffe, die die wichtigsten Luecken
schliessen wuerden. Format als web_search-kompatible Keywords:
1. **[HOCH]** `"Suchbegriff 1"` — Warum wichtig
2. **[MITTEL]** `"Suchbegriff 2"` — Warum wichtig
3. ...

### Vollstaendigkeits-Score
Geschaetzte Abdeckung: X/10 (1 = fast nichts bekannt, 10 = umfassend)
Begruendung in einem Satz.

VORHANDENES WISSEN:
---
{context}
---

Wenn gar kein Wissen vorhanden ist, erstelle eine grundlegende Recherche-Roadmap
mit den wichtigsten Suchbegriffen, um das Thema systematisch zu erschliessen."""


# ── Helper functions ──────────────────────────────────────────────────────


def _truncate(text: str, max_chars: int) -> str:
    """Kuerzt Text auf max_chars, am letzten Satzende."""
    if len(text) <= max_chars:
        return text
    original_len = len(text)
    truncated = text[:max_chars]
    last_period = truncated.rfind(".")
    if last_period > max_chars * 0.5:
        truncated = truncated[: last_period + 1]
    return truncated + f"\n[... gekürzt: {len(truncated)}/{original_len} Zeichen]"


def _extract_keywords(text: str) -> list[str]:
    """Extrahiert die wichtigsten Schluesselwoerter aus einem Text.

    Filtert Stoppwoerter und gibt die relevantesten Begriffe zurueck.
    """
    stop_words = frozenset(
        {
            "der",
            "die",
            "das",
            "ein",
            "eine",
            "und",
            "oder",
            "aber",
            "in",
            "von",
            "zu",
            "mit",
            "auf",
            "für",
            "an",
            "bei",
            "nach",
            "über",
            "aus",
            "wie",
            "was",
            "wer",
            "wo",
            "wann",
            "warum",
            "ist",
            "sind",
            "hat",
            "haben",
            "wird",
            "werden",
            "kann",
            "können",
            "nicht",
            "auch",
            "noch",
            "schon",
            "nur",
            "als",
            "wenn",
            "so",
            "doch",
            "ich",
            "du",
            "er",
            "sie",
            "es",
            "wir",
            "ihr",
            "mein",
            "dein",
            "sein",
            "dem",
            "den",
            "des",
            "einem",
            "einen",
            "einer",
            "the",
            "a",
            "is",
            "are",
            "were",
            "be",
            "been",
            "to",
            "of",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "about",
            "how",
            "what",
            "which",
            "who",
            "when",
            "where",
            "why",
            "and",
            "or",
            "but",
            "not",
            "this",
            "that",
        }
    )

    # Extract words, remove punctuation
    words = re.findall(r"\b[a-zA-ZäöüÄÖÜß]{3,}\b", text)
    # Filter stop words, remove duplicates, preserve order
    seen: set[str] = set()
    keywords: list[str] = []
    for w in words:
        lower = w.lower()
        if lower not in stop_words and lower not in seen:
            seen.add(lower)
            keywords.append(w)
    return keywords[:10]


def _filter_relevant_text(text: str, topic: str) -> str:
    """Filtert nur thematisch relevante Absaetze aus einem Text.

    Teilt den Text in Absaetze und behaelt nur jene,
    die Schluesselwoerter des Themas enthalten.
    """
    keywords = _extract_keywords(topic)
    if not keywords:
        return text

    keyword_set = {kw.lower() for kw in keywords}
    paragraphs = text.split("\n\n")
    relevant: list[str] = []

    for para in paragraphs:
        para_lower = para.lower()
        # At least one keyword must appear
        if any(kw in para_lower for kw in keyword_set):
            relevant.append(para)

    return "\n\n".join(relevant) if relevant else ""


# ── MCP client registration ─────────────────────────────────────────────


def register_synthesis_tools(
    mcp_client: Any,
    config: Any | None = None,
) -> KnowledgeSynthesizer:
    """Registriert Synthesis-Tools beim MCP-Client.

    Args:
        mcp_client: JarvisMCPClient-Instanz.
        config: CognithorConfig (optional).

    Returns:
        KnowledgeSynthesizer-Instanz (Abhaengigkeiten werden spaeter injiziert).
    """
    synth = KnowledgeSynthesizer(config=config)

    mcp_client.register_builtin_handler(
        "knowledge_synthesize",
        synth.knowledge_synthesize,
        description=(
            "Vollständige Wissenssynthese: Sammelt alle Quellen (Memory, Vault, Web), "
            "vergleicht Informationen, erkennt Widersprüche, baut Zeitlinien, "
            "bewertet Konfidenz und identifiziert Wissenslücken. "
            "DAS zentrale Tool um ein Thema umfassend zu verstehen."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Thema oder Fragestellung für die Wissenssynthese",
                },
                "include_web": {
                    "type": "boolean",
                    "description": "Aktuelle Web-Recherche einbeziehen (Default: true)",
                    "default": True,
                },
                "depth": {
                    "type": "string",
                    "enum": ["quick", "standard", "deep"],
                    "description": (
                        "Analysetiefe: quick (nur Memory+Vault), "
                        "standard (+ Web), deep (detailliert)"
                    ),
                    "default": "standard",
                },
                "language": {
                    "type": "string",
                    "description": "Sprache der Synthese (de/en)",
                    "default": "de",
                },
                "save_to_vault": {
                    "type": "boolean",
                    "description": "Synthese im Vault speichern",
                    "default": False,
                },
            },
            "required": ["topic"],
        },
    )

    mcp_client.register_builtin_handler(
        "knowledge_contradictions",
        synth.knowledge_contradictions,
        description=(
            "Vergleicht gespeichertes Wissen (Memory + Vault) mit aktuellen Web-Informationen "
            "und identifiziert Widersprüche, veraltete Fakten und neue Erkenntnisse."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Thema für die Widerspruchsanalyse",
                },
                "language": {
                    "type": "string",
                    "description": "Sprache (de/en)",
                    "default": "de",
                },
            },
            "required": ["topic"],
        },
    )

    mcp_client.register_builtin_handler(
        "knowledge_timeline",
        synth.knowledge_timeline,
        description=(
            "Baut eine chronologische Zeitlinie aus allen verfügbaren Quellen auf. "
            "Zeigt Kausalketten (X führte zu Y) und erkennbare Trends."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Thema für die Zeitlinie",
                },
                "language": {
                    "type": "string",
                    "description": "Sprache (de/en)",
                    "default": "de",
                },
            },
            "required": ["topic"],
        },
    )

    mcp_client.register_builtin_handler(
        "knowledge_gaps",
        synth.knowledge_gaps,
        description=(
            "Analysiert was zu einem Thema bekannt ist und was fehlt. "
            "Gibt einen Vollständigkeits-Score und priorisierte Recherche-Vorschläge."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Thema für die Lückenanalyse",
                },
                "language": {
                    "type": "string",
                    "description": "Sprache (de/en)",
                    "default": "de",
                },
            },
            "required": ["topic"],
        },
    )

    log.info(
        "synthesis_tools_registered",
        tools=[
            "knowledge_synthesize",
            "knowledge_contradictions",
            "knowledge_timeline",
            "knowledge_gaps",
        ],
    )
    return synth
