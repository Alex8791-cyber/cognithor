"""Verified Web Lookup — mehrstufiges Fakten-Pruefverfahren.

Orchestriert mehrere Agenten parallel:
  1. Search-Agent:      Findet relevante URLs
  2. Trafilatura-Agent:  Extrahiert Text aus statischem HTML (schnell)
  3. Browser-Agent:      Extrahiert Text aus JS-gerenderten Seiten (robust)
  4. Fact-Extraction:    LLM extrahiert strukturierte Fakten
  5. Consensus-Agent:    Vergleicht Fakten, berechnet Konfidenz

Ergebnis: Verifizierte Antwort mit Quellenangaben und Konfidenz-Score.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from jarvis.i18n import t
from jarvis.utils.logging import get_logger

if TYPE_CHECKING:
    from jarvis.config import JarvisConfig
    from jarvis.mcp.browser import BrowserTool
    from jarvis.mcp.client import JarvisMCPClient
    from jarvis.mcp.web import WebTools

log = get_logger(__name__)

# ── Konfiguration ────────────────────────────────────────────────────────────

_DEFAULT_NUM_SOURCES = 3
_DEFAULT_MAX_TEXT_PER_SOURCE = 5000
_DEFAULT_BROWSER_TIMEOUT_S = 20
_DEFAULT_EXTRACTION_TIMEOUT_S = 45
_FACT_EXTRACTION_PROMPT_BUDGET = 3000  # chars Kontext fuer LLM


# ── Datenmodelle ─────────────────────────────────────────────────────────────


@dataclass
class ExtractedFact:
    """Ein einzelner Fakten-Claim aus einer Quelle."""

    claim: str
    value: str
    fact_type: str  # "number", "date", "name", "boolean", "text"
    source_url: str = ""


@dataclass
class SourceResult:
    """Ergebnis einer einzelnen Quellen-Extraktion."""

    url: str
    text: str
    method: str  # "trafilatura", "browser", "jina"
    success: bool
    duration_ms: float = 0.0


@dataclass
class VerificationResult:
    """Gesamtergebnis des Pruefverfahrens."""

    answer: str
    confidence: float  # 0.0 - 1.0
    facts: list[ExtractedFact] = field(default_factory=list)
    sources: list[SourceResult] = field(default_factory=list)
    agreement: float = 0.0  # Uebereinstimmung der Quellen
    discrepancies: list[str] = field(default_factory=list)
    duration_ms: float = 0.0


# ── Fact-Extraction Prompt ───────────────────────────────────────────────────

_FACT_EXTRACT_PROMPT = """\
Extract factual claims from the text that answer this question: "{query}"

Rules:
- Extract ONLY concrete facts (numbers, dates, names, yes/no).
- Each fact must have a specific value, not vague descriptions.
- If the text does not contain relevant facts, return an empty list.
- Return ONLY valid JSON, no explanation.

Text:
{text}

Return JSON:
{{"facts": [{{"claim": "short description", "value": "concrete value", "type": "number|date|name|boolean|text"}}]}}"""

_CONSENSUS_PROMPT = """\
Multiple sources were queried to answer: "{query}"

Source facts:
{facts_json}

Task:
1. Determine the most likely correct answer based on source agreement.
2. If sources agree, state the answer with high confidence.
3. If sources disagree, note the discrepancy and prefer the primary/official source.
4. Answer in {language}, in natural spoken language (no bullet points).

Return JSON:
{{"answer": "the verified answer in {language}", "confidence": 0.0-1.0, "discrepancies": ["list of disagreements or empty"]}}"""


# ── VerifiedWebLookup ────────────────────────────────────────────────────────


class VerifiedWebLookup:
    """Mehrstufiges Fakten-Pruefverfahren mit parallelen Agenten."""

    def __init__(self, config: JarvisConfig | None = None) -> None:
        self._config = config
        self._web_tools: WebTools | None = None
        self._browser_tool: BrowserTool | None = None
        self._llm_fn: Any = None
        self._llm_model: str = ""

    # ── Dependency Injection ─────────────────────────────────────────────

    def _set_web_tools(self, web_tools: WebTools) -> None:
        self._web_tools = web_tools

    def _set_browser_tool(self, browser_tool: BrowserTool) -> None:
        self._browser_tool = browser_tool

    def _set_llm_fn(self, llm_fn: Any, model_name: str = "") -> None:
        self._llm_fn = llm_fn
        self._llm_model = model_name

    # ── Hauptmethode ─────────────────────────────────────────────────────

    async def verified_lookup(
        self,
        query: str,
        num_sources: int = _DEFAULT_NUM_SOURCES,
        language: str = "de",
    ) -> str:
        """Fuehrt eine verifizierte Web-Recherche durch.

        Args:
            query: Die Suchanfrage / Faktenfrage.
            num_sources: Anzahl der zu pruefenden Quellen (2-5).
            language: Antwortsprache.

        Returns:
            Formatierter Ergebnis-String mit Antwort, Konfidenz und Quellen.
        """
        start = time.monotonic()
        num_sources = max(2, min(num_sources, 5))

        if self._web_tools is None:
            return t("verified_lookup.no_webtools")

        # ── Stage 1: URLs finden ─────────────────────────────────────────
        try:
            search_text = await self._web_tools.web_search(
                query=query,
                num_results=num_sources + 2,  # Puffer fuer fehlschlagende URLs
                language=language,
            )
        except Exception as exc:
            log.warning("verified_lookup_search_failed", error=str(exc)[:200])
            return t("verified_lookup.search_failed", error=str(exc))

        urls = re.findall(r"URL: (https?://[^\s]+)", search_text)[:num_sources]
        if not urls:
            return t("verified_lookup.no_results", query=query)

        # ── Stage 2: Parallele Extraktion (Trafilatura + Browser) ────────
        source_results = await self._parallel_extract(urls, num_sources)

        successful = [s for s in source_results if s.success and len(s.text) > 50]
        if not successful:
            return t("verified_lookup.no_content", query=query)

        # ── Stage 3: Fakten-Extraktion per LLM ──────────────────────────
        all_facts: list[ExtractedFact] = []
        if self._llm_fn is not None:
            fact_tasks = [self._extract_facts(src.text, query, src.url) for src in successful]
            fact_results = await asyncio.gather(*fact_tasks, return_exceptions=True)
            for result in fact_results:
                if isinstance(result, list):
                    all_facts.extend(result)

        # ── Stage 4: Konsens berechnen ───────────────────────────────────
        verification = await self._build_consensus(
            query,
            all_facts,
            successful,
            language,
        )
        verification.duration_ms = (time.monotonic() - start) * 1000

        # ── Ergebnis formatieren ─────────────────────────────────────────
        return self._format_result(verification, query)

    # ── Stage 2: Parallele Extraktion ────────────────────────────────────

    async def _parallel_extract(
        self,
        urls: list[str],
        num_sources: int,
    ) -> list[SourceResult]:
        """Extrahiert Text aus URLs mit Trafilatura und Browser parallel."""
        results: list[SourceResult] = []

        async def _extract_trafilatura(url: str) -> SourceResult:
            start = time.monotonic()
            try:
                text = await self._web_tools.web_fetch(url, max_chars=_DEFAULT_MAX_TEXT_PER_SOURCE)
                ms = (time.monotonic() - start) * 1000
                success = bool(text and len(text.strip()) > 50)
                return SourceResult(
                    url=url,
                    text=text[:_DEFAULT_MAX_TEXT_PER_SOURCE],
                    method="trafilatura",
                    success=success,
                    duration_ms=ms,
                )
            except Exception as exc:
                ms = (time.monotonic() - start) * 1000
                log.debug("trafilatura_extract_failed", url=url[:80], error=str(exc)[:100])
                return SourceResult(
                    url=url,
                    text="",
                    method="trafilatura",
                    success=False,
                    duration_ms=ms,
                )

        async def _extract_browser(url: str) -> SourceResult:
            if self._browser_tool is None:
                return SourceResult(
                    url=url,
                    text="",
                    method="browser",
                    success=False,
                )
            start = time.monotonic()
            try:
                browser_result = await asyncio.wait_for(
                    self._browser_tool.navigate(url, extract_text=True),
                    timeout=_DEFAULT_BROWSER_TIMEOUT_S,
                )
                ms = (time.monotonic() - start) * 1000
                text = browser_result.text or ""
                success = browser_result.success and len(text.strip()) > 50
                return SourceResult(
                    url=url,
                    text=text[:_DEFAULT_MAX_TEXT_PER_SOURCE],
                    method="browser",
                    success=success,
                    duration_ms=ms,
                )
            except (TimeoutError, asyncio.TimeoutError):
                ms = (time.monotonic() - start) * 1000
                log.debug("browser_extract_timeout", url=url[:80])
                return SourceResult(
                    url=url,
                    text="",
                    method="browser",
                    success=False,
                    duration_ms=ms,
                )
            except Exception as exc:
                ms = (time.monotonic() - start) * 1000
                log.debug("browser_extract_failed", url=url[:80], error=str(exc)[:100])
                return SourceResult(
                    url=url,
                    text="",
                    method="browser",
                    success=False,
                    duration_ms=ms,
                )

        # Trafilatura fuer ALLE URLs + Browser fuer die erste URL parallel
        tasks: list[asyncio.Task] = []
        for url in urls:
            tasks.append(asyncio.ensure_future(_extract_trafilatura(url)))
        # Browser-Agent nur fuer Top-URLs (teurer, aber robuster)
        browser_urls = urls[: min(2, len(urls))]
        for url in browser_urls:
            tasks.append(asyncio.ensure_future(_extract_browser(url)))

        try:
            done = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=_DEFAULT_EXTRACTION_TIMEOUT_S,
            )
        except (TimeoutError, asyncio.TimeoutError):
            log.warning("verified_lookup_extraction_timeout")
            done = []

        # Ergebnisse sammeln und deduplizieren (pro URL das bessere Ergebnis)
        best_per_url: dict[str, SourceResult] = {}
        for item in done:
            if isinstance(item, SourceResult):
                existing = best_per_url.get(item.url)
                if existing is None or (item.success and len(item.text) > len(existing.text)):
                    best_per_url[item.url] = item
                # Auch das "zweitbeste" Ergebnis aufheben wenn es eine
                # andere Methode nutzt (fuer Cross-Check)
                elif (
                    item.success
                    and existing.method != item.method
                    and item.url not in [r.url for r in results if r.method == item.method]
                ):
                    results.append(item)

        results.extend(best_per_url.values())
        return results

    # ── Stage 3: Fakten-Extraktion ───────────────────────────────────────

    async def _extract_facts(
        self,
        text: str,
        query: str,
        source_url: str = "",
    ) -> list[ExtractedFact]:
        """Extrahiert strukturierte Fakten aus Text per LLM."""
        if self._llm_fn is None:
            return []

        # Text kuerzen fuer LLM-Budget
        truncated = text[:_FACT_EXTRACTION_PROMPT_BUDGET]
        prompt = _FACT_EXTRACT_PROMPT.format(query=query, text=truncated)

        try:
            raw = await self._llm_fn(prompt, self._llm_model)
            # <think>...</think> entfernen (qwen3)
            raw = re.sub(r"<think>.*?</think>\s*", "", raw, flags=re.DOTALL)
            # JSON extrahieren
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not json_match:
                return []
            data = json.loads(json_match.group(0))
            facts = []
            for f in data.get("facts", []):
                if isinstance(f, dict) and f.get("value"):
                    facts.append(
                        ExtractedFact(
                            claim=f.get("claim", ""),
                            value=str(f["value"]),
                            fact_type=f.get("type", "text"),
                            source_url=source_url,
                        )
                    )
            return facts
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            log.debug("fact_extraction_parse_error", error=str(exc)[:100])
            return []
        except Exception as exc:
            log.warning("fact_extraction_failed", error=str(exc)[:200])
            return []

    # ── Stage 4: Konsens ─────────────────────────────────────────────────

    async def _build_consensus(
        self,
        query: str,
        facts: list[ExtractedFact],
        sources: list[SourceResult],
        language: str,
    ) -> VerificationResult:
        """Berechnet Konsens ueber extrahierte Fakten."""
        if not facts and not sources:
            return VerificationResult(
                answer=t("verified_lookup.no_facts"),
                confidence=0.0,
            )

        # Heuristischer Konsens: gleiche Werte zaehlen
        value_counts: dict[str, list[str]] = {}  # value -> [source_urls]
        for f in facts:
            normalized = f.value.strip().lower().rstrip(".")
            value_counts.setdefault(normalized, []).append(f.source_url)

        # Berechne Agreement-Score
        if value_counts:
            max_agreement = max(len(urls) for urls in value_counts.values())
            total_sources = len({s.url for s in sources if s.success})
            agreement = max_agreement / max(total_sources, 1)
        else:
            agreement = 0.0

        # Diskrepanzen finden
        discrepancies: list[str] = []
        if len(value_counts) > 1:
            sorted_values = sorted(value_counts.items(), key=lambda x: -len(x[1]))
            for val, urls in sorted_values[1:]:
                majority_val = sorted_values[0][0]
                if val != majority_val:
                    discrepancies.append(
                        t(
                            "verified_lookup.discrepancy_item",
                            value_a=val,
                            count_a=len(urls),
                            value_b=majority_val,
                            count_b=len(sorted_values[0][1]),
                        )
                    )

        # LLM-basierter Konsens (wenn verfuegbar)
        if self._llm_fn is not None and facts:
            answer, confidence, llm_discrepancies = await self._llm_consensus(
                query,
                facts,
                language,
            )
            discrepancies.extend(llm_discrepancies)
        else:
            # Fallback: Textzusammenfassung aus bestem Ergebnis
            best_source = max(sources, key=lambda s: len(s.text)) if sources else None
            answer = best_source.text[:500] if best_source else t("verified_lookup.no_answer")
            confidence = agreement * 0.7  # Ohne LLM konservativere Konfidenz

        return VerificationResult(
            answer=answer,
            confidence=min(confidence, 1.0),
            facts=facts,
            sources=sources,
            agreement=agreement,
            discrepancies=discrepancies,
        )

    async def _llm_consensus(
        self,
        query: str,
        facts: list[ExtractedFact],
        language: str,
    ) -> tuple[str, float, list[str]]:
        """LLM-gestuetzter Konsens ueber Fakten.

        Returns:
            (answer, confidence, discrepancies)
        """
        facts_data = [
            {"claim": f.claim, "value": f.value, "type": f.fact_type, "source": f.source_url}
            for f in facts[:20]  # Max 20 Fakten
        ]
        prompt = _CONSENSUS_PROMPT.format(
            query=query,
            facts_json=json.dumps(facts_data, ensure_ascii=False, indent=2),
            language=language,
        )

        try:
            raw = await self._llm_fn(prompt, self._llm_model)
            raw = re.sub(r"<think>.*?</think>\s*", "", raw, flags=re.DOTALL)
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not json_match:
                return raw.strip(), 0.5, []
            data = json.loads(json_match.group(0))
            return (
                data.get("answer", raw.strip()),
                float(data.get("confidence", 0.5)),
                data.get("discrepancies", []),
            )
        except Exception as exc:
            log.warning("llm_consensus_failed", error=str(exc)[:200])
            # Fallback: bester Fakt
            if facts:
                return facts[0].claim, 0.4, []
            return t("verified_lookup.consensus_failed"), 0.2, []

    # ── Formatierung ─────────────────────────────────────────────────────

    @staticmethod
    def _format_result(result: VerificationResult, query: str) -> str:
        """Formatiert das Ergebnis als lesbaren String fuer den Planner."""
        parts: list[str] = []

        # Header
        conf_pct = int(result.confidence * 100)
        parts.append(t("verified_lookup.header", confidence=conf_pct))
        parts.append(result.answer)

        # Quellen
        successful = [s for s in result.sources if s.success]
        if successful:
            parts.append(t("verified_lookup.sources_header", count=len(successful)))
            for i, src in enumerate(successful, 1):
                parts.append(f"- [{i}] {src.url} ({src.method}, {src.duration_ms:.0f}ms)")

        # Uebereinstimmung
        if result.facts:
            agree_pct = int(result.agreement * 100)
            parts.append(t("verified_lookup.agreement_header", agreement=agree_pct))
            if result.agreement >= 0.8:
                parts.append(t("verified_lookup.agreement_high"))
            elif result.agreement >= 0.5:
                parts.append(t("verified_lookup.agreement_medium"))
            else:
                parts.append(t("verified_lookup.agreement_low"))

        # Diskrepanzen
        if result.discrepancies:
            parts.append(t("verified_lookup.discrepancies_header"))
            for disc in result.discrepancies:
                parts.append(f"- {disc}")

        # Performance
        parts.append(t("verified_lookup.duration", ms=f"{result.duration_ms:.0f}"))

        return "\n".join(parts)


# ── MCP-Tool-Registrierung ───────────────────────────────────────────────────


def register_verified_lookup_tools(
    mcp_client: JarvisMCPClient,
    config: JarvisConfig | None = None,
) -> VerifiedWebLookup:
    """Registriert das verified_web_lookup Tool.

    Returns:
        VerifiedWebLookup-Instanz (Dependencies werden spaeter injiziert).
    """
    lookup = VerifiedWebLookup(config)

    async def _handle_verified_lookup(**kwargs: Any) -> str:
        return await lookup.verified_lookup(
            query=kwargs.get("query", ""),
            num_sources=kwargs.get("num_sources", _DEFAULT_NUM_SOURCES),
            language=kwargs.get("language", "de"),
        )

    mcp_client.register_builtin_handler(
        tool_name="verified_web_lookup",
        handler=_handle_verified_lookup,
        description=(
            "Mehrstufiges Fakten-Pruefverfahren: Sucht im Web, extrahiert "
            "Fakten parallel via Trafilatura und Browser, vergleicht "
            "Quellen und liefert verifizierte Antwort mit Konfidenz-Score. "
            "Fuer Faktenfragen mit Zahlen, Daten oder konkreten Aussagen."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Die Faktenfrage oder Suchanfrage.",
                },
                "num_sources": {
                    "type": "integer",
                    "description": "Anzahl der zu pruefenden Quellen (2-5).",
                    "default": _DEFAULT_NUM_SOURCES,
                },
                "language": {
                    "type": "string",
                    "description": "Antwortsprache (de, en, zh).",
                    "default": "de",
                },
            },
            "required": ["query"],
        },
    )

    log.info("verified_lookup_tool_registered")
    return lookup
