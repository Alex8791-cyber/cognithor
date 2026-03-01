"""Reflection cycle -- learning from experience. [B§6]

The Reflector analyzes completed sessions and produces:
  - Success evaluation (score 0-1)
  - Extracted facts -> Semantic Memory
  - Procedure candidates -> Procedural Memory
  - Session summary -> Episodic Memory

Inspired by Reflexion (Shinn 2023), SAGE (2024), RMM (2025).
"""

from __future__ import annotations

import json
import re
from datetime import date
from typing import TYPE_CHECKING, Any

from jarvis.core.model_router import ModelRouter, OllamaClient, OllamaError
from jarvis.models import (
    AgentResult,
    Entity,
    ExtractedFact,
    GateStatus,
    ProcedureCandidate,
    Relation,
    ReflectionResult,
    SessionContext,
    SessionSummary,
    WorkingMemory,
)
from jarvis.utils.logging import get_logger

if TYPE_CHECKING:
    from jarvis.audit import AuditLogger
    from jarvis.config import JarvisConfig

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------

#: Minimale Iterationen, ab denen Reflexion sinnvoll ist.
MIN_ITERATIONS_FOR_REFLECTION = 1

#: Minimale Tool-Aufrufe, ab denen eine Prozedur-Synthese versucht wird.
MIN_TOOL_CALLS_FOR_PROCEDURE = 2

#: Maximale Zeichen für den Reflection-Input (Context-Budget).
MAX_REFLECTION_INPUT_CHARS = 12_000

# ---------------------------------------------------------------------------
# Memory Sanitization
# ---------------------------------------------------------------------------

_INJECTION_PATTERNS = [
    r"#\s*SYSTEM:", r"<\|im_start\|>", r"<\|im_end\|>",
    r"\[INST\]", r"\[/INST\]", r"<<SYS>>", r"<</SYS>>",
    r"Human:", r"Assistant:", r"<\|user\|>", r"<\|assistant\|>",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)


def _sanitize_memory_text(text: str, max_len: int = 5000) -> str:
    """Bereinigt LLM-Output vor Speicherung in Memory-Tiers."""
    if not text:
        return ""
    # Null-Bytes und Steuerzeichen entfernen (ausser \n, \t)
    text = "".join(c for c in text if c == "\n" or c == "\t" or (ord(c) >= 32))
    # Prompt-Injection-Marker entfernen
    text = _INJECTION_RE.sub("[SANITIZED]", text)
    # Laenge begrenzen
    return text[:max_len]


def _safe_float(value: Any, default: float) -> float:
    """Konvertiert einen Wert sicher zu float.

    LLMs liefern manchmal nicht-numerische Strings wie "high" statt 0.9.
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# Reflector
# ---------------------------------------------------------------------------


class ReflectorError(Exception):
    """Error in the Reflector."""


class Reflector:
    """Analyzes completed sessions and extracts knowledge. [B§6.1]

    The Reflector is called at the end of an agent cycle and
    writes insights into all relevant memory tiers.
    """

    def __init__(
        self,
        config: JarvisConfig,
        ollama: Any,
        model_router: ModelRouter,
        audit_logger: AuditLogger | None = None,
        episodic_store: Any = None,
        causal_analyzer: Any = None,
        weight_optimizer: Any = None,
        reward_calculator: Any = None,
        cost_tracker: Any = None,
    ) -> None:
        """Initialisiert den Reflector mit LLM-Client und Model-Router.

        Args:
            config: Jarvis-Konfiguration.
            ollama: LLM-Client (OllamaClient oder UnifiedLLMClient).
            model_router: Model-Router für Modellauswahl.
            audit_logger: Optionaler AuditLogger für LLM-Call-Protokollierung.
            episodic_store: Optionaler EpisodicStore für Langzeit-Episoden.
            causal_analyzer: Optionaler CausalAnalyzer für Tool-Sequenz-Lernen.
            weight_optimizer: Optionaler SearchWeightOptimizer für Such-Feedback.
            reward_calculator: Optionaler RewardCalculator fuer Composite-Scores.
            cost_tracker: Optionaler CostTracker fuer LLM-Kosten-Tracking.
        """
        self._config = config
        self._ollama = ollama
        self._router = model_router
        self._audit_logger = audit_logger
        self._episodic_store = episodic_store
        self._causal_analyzer = causal_analyzer
        self._weight_optimizer = weight_optimizer
        self._reward_calculator = reward_calculator
        self._cost_tracker = cost_tracker

    # ------------------------------------------------------------------
    # Öffentliche API
    # ------------------------------------------------------------------

    def should_reflect(self, agent_result: AgentResult) -> bool:
        """Entscheidet ob eine Reflexion sinnvoll ist.

        Einfache Frage-Antwort-Sessions ohne Tool-Calls
        brauchen keine Reflexion.
        """
        if agent_result.total_iterations < MIN_ITERATIONS_FOR_REFLECTION:
            return False
        if not agent_result.plans:
            return False
        # Nur reflektieren wenn mindestens ein Tool aufgerufen wurde
        has_tool_calls = any(p.has_actions for p in agent_result.plans)
        return has_tool_calls

    def match_procedures(
        self,
        user_message: str,
        procedural_memory: Any,
        max_results: int = 2,
        min_score: float = 0.3,
    ) -> list[str]:
        """Findet passende Prozeduren für eine User-Nachricht. [B§6.3]

        Extrahiert Keywords aus der Nachricht und sucht in den
        gespeicherten Prozeduren nach Matches.

        Args:
            user_message: Die eingehende User-Nachricht
            procedural_memory: ProceduralMemory-Instanz
            max_results: Maximale Anzahl zurückgegebener Prozeduren
            min_score: Minimaler Match-Score (0--1)

        Returns:
            Liste von Prozedur-Texten (body), bereit zur Injection
            in WorkingMemory.injected_procedures.
        """
        keywords = self.extract_keywords(user_message)
        if not keywords:
            return []

        matches = procedural_memory.find_by_keywords(keywords)
        results: list[str] = []

        for meta, body, score in matches[:max_results]:
            if score < min_score:
                continue
            # Prozeduren mit genug Daten und schlechter Erfolgsquote überspringen
            if meta.total_uses >= 3 and meta.success_rate < 0.5:
                log.debug(
                    "procedure_skipped_unreliable",
                    name=meta.name,
                    success_rate=meta.success_rate,
                )
                continue
            results.append(body)
            log.debug(
                "procedure_matched",
                name=meta.name,
                score=score,
                uses=meta.total_uses,
            )

        return results

    @staticmethod
    def extract_keywords(text: str) -> list[str]:
        """Extrahiert Suchbegriffe aus einer User-Nachricht. [B§6.3]

        Einfache Stopwort-Filterung für Deutsch und Englisch.
        Gibt max. 8 Keywords zurück (die längsten bevorzugt).
        """
        # Normalisieren
        text = text.lower().strip()
        # Satzzeichen entfernen
        text = re.sub(r"[^\w\säöüß]", " ", text)
        # Tokens
        tokens = text.split()

        # Deutsche + englische Stopwörter
        stop_words = {
            # Deutsch
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
            "unser",
            "euer",
            "der",
            "die",
            "das",
            "ein",
            "eine",
            "und",
            "oder",
            "aber",
            "wenn",
            "weil",
            "dass",
            "ist",
            "sind",
            "war",
            "hat",
            "haben",
            "wird",
            "werden",
            "kann",
            "können",
            "soll",
            "sollen",
            "muss",
            "müssen",
            "nicht",
            "kein",
            "keine",
            "mit",
            "von",
            "für",
            "auf",
            "aus",
            "bei",
            "nach",
            "über",
            "unter",
            "zwischen",
            "durch",
            "gegen",
            "ohne",
            "bis",
            "seit",
            "wie",
            "was",
            "wer",
            "wo",
            "wann",
            "warum",
            "bitte",
            "mal",
            "mir",
            "mich",
            "dir",
            "dich",
            "ihm",
            "ihn",
            "uns",
            "den",
            "dem",
            "des",
            "noch",
            "schon",
            "auch",
            "nur",
            "mehr",
            "sehr",
            "hier",
            "dort",
            "jetzt",
            "dann",
            "also",
            "denn",
            "doch",
            "gerne",
            "gern",
            "könntest",
            "könnten",
            "kannst",
            "möchte",
            "möchten",
            "würde",
            "würden",
            "einen",
            "einer",
            "einem",
            "bin",
            "bist",
            "wäre",
            "waren",
            "hatte",
            "hatten",
            # Englisch
            "the",
            "a",
            "an",
            "is",
            "are",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "can",
            "may",
            "might",
            "shall",
            "not",
            "and",
            "or",
            "but",
            "if",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "as",
            "into",
            "about",
            "this",
            "that",
            "these",
            "those",
            "it",
            "its",
            "my",
            "your",
            "his",
            "her",
            "our",
            "their",
            "me",
            "him",
            "them",
            "us",
            "who",
            "what",
            "where",
            "when",
            "why",
            "how",
            "please",
            "just",
            "very",
            "too",
            "so",
            "than",
            "then",
        }

        # Filtern: Stopwörter raus, min. 3 Zeichen
        filtered = [t for t in tokens if t not in stop_words and len(t) >= 3]

        # Längste Keywords bevorzugen (informativer)
        filtered.sort(key=len, reverse=True)

        return filtered[:8]

    async def reflect(
        self,
        session: SessionContext,
        working_memory: WorkingMemory,
        agent_result: AgentResult,
    ) -> ReflectionResult:
        """Führt den vollständigen Reflexions-Zyklus durch. [B§6.1]

        Args:
            session: Session-Kontext der abgeschlossenen Session
            working_memory: Working Memory mit Chat-History
            agent_result: Ergebnis des Agent-Zyklus

        Returns:
            ReflectionResult mit Bewertung, Fakten und Prozedur-Kandidat.
        """
        log.info(
            "reflection_start",
            session_id=session.session_id,
            iterations=agent_result.total_iterations,
        )

        # Reflexions-Input zusammenstellen
        session_text = self._format_session_for_reflection(working_memory, agent_result)

        # LLM-basierte Reflexion
        model = self._router.select_model("reflection", "medium")
        model_config = self._router.get_model_config(model)

        system_prompt = self._build_reflection_prompt()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": session_text},
        ]

        try:
            response = await self._ollama.chat(
                model=model,
                messages=messages,
                temperature=model_config.get("temperature", 0.3),
                top_p=model_config.get("top_p", 0.9),
            )
        except OllamaError as exc:
            log.error("reflection_llm_error", error=str(exc))
            if self._audit_logger:
                self._audit_logger.log_tool_call(
                    "llm_reflect", {"model": model, "session": session.session_id[:8]},
                    result=f"ERROR: {exc}", success=False,
                )
            return self._fallback_reflection(session, agent_result)

        # LLM-Kosten aufzeichnen
        if self._cost_tracker is not None:
            try:
                input_tokens = response.get("prompt_eval_count", 0)
                output_tokens = response.get("eval_count", 0)
                if input_tokens or output_tokens:
                    self._cost_tracker.record_llm_call(
                        model=model,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        session_id=session.session_id,
                    )
            except Exception as exc:
                log.debug("cost_tracking_failed", error=str(exc))

        # Antwort parsen
        assistant_text = response.get("message", {}).get("content", "")
        result = self._parse_reflection(assistant_text, session.session_id)

        if self._audit_logger:
            self._audit_logger.log_tool_call(
                "llm_reflect", {"model": model, "session": session.session_id[:8]},
                result=f"score={result.success_score}, facts={len(result.extracted_facts)}",
                success=True,
            )

        log.info(
            "reflection_complete",
            session_id=session.session_id,
            score=result.success_score,
            has_procedure=result.has_procedure,
            fact_count=len(result.extracted_facts),
        )

        # --- Integration: Episodic Store, Causal Learning, Weight Optimizer ---
        tool_sequence = [tr.tool_name for tr in agent_result.tool_results]

        # Reward-Score berechnen (ersetzt einfachen success_score wenn verfuegbar)
        reward_score = result.success_score
        if self._reward_calculator is not None:
            try:
                failed = sum(1 for tr in agent_result.tool_results if not tr.success)
                unique = len(set(tool_sequence))
                reward_score = self._reward_calculator.calculate_reward(
                    success_score=result.success_score,
                    total_tools=len(tool_sequence),
                    failed_tools=failed,
                    unique_tools=unique,
                    total_tool_calls=len(tool_sequence),
                    duration_seconds=agent_result.total_duration_ms / 1000.0,
                )
            except Exception:
                pass

        # Episodic Store: Langzeit-Episode speichern
        if self._episodic_store and result.session_summary:
            try:
                self._episodic_store.store_episode(
                    session_id=session.session_id,
                    topic=result.session_summary.goal,
                    content=result.evaluation,
                    outcome=result.session_summary.outcome,
                    tool_sequence=tool_sequence,
                    success_score=reward_score,
                    tags=result.session_summary.tools_used,
                )
            except Exception as exc:
                log.debug("episodic_store_write_failed", error=str(exc))

        # Causal Analyzer: Tool-Sequenz mit Erfolg korrelieren
        if self._causal_analyzer and tool_sequence:
            try:
                self._causal_analyzer.record_sequence(
                    session_id=session.session_id,
                    tool_sequence=tool_sequence,
                    success_score=reward_score,
                    model_used=agent_result.model_used,
                )
            except Exception as exc:
                log.debug("causal_record_failed", error=str(exc))

        # Weight Optimizer: Such-Feedback (wenn Reflexion Suche enthielt)
        if self._weight_optimizer and result.success_score > 0:
            try:
                self._weight_optimizer.record_outcome(
                    query=session.session_id[:16],
                    channel_contributions={"vector": 0.5, "bm25": 0.3, "graph": 0.2},
                    feedback_score=result.success_score,
                )
            except Exception as exc:
                log.debug("weight_optimizer_feedback_failed", error=str(exc))

        return result

    async def apply(
        self,
        result: ReflectionResult,
        memory_manager: Any,
    ) -> dict[str, int]:
        """Schreibt Reflexionsergebnisse in die Memory-Tiers. [B§6.1]

        Args:
            result: Das ReflectionResult aus reflect()
            memory_manager: MemoryManager-Instanz

        Returns:
            Dict mit Anzahl geschriebener Einträge pro Tier.
        """
        counts: dict[str, int] = {
            "episodic": 0,
            "semantic": 0,
            "procedural": 0,
        }

        # 1. Session-Zusammenfassung → Episodic Memory
        if result.session_summary:
            await self._write_episodic(result, memory_manager)
            counts["episodic"] = 1

        # 2. Extrahierte Fakten → Semantic Memory
        if result.extracted_facts:
            counts["semantic"] = await self._write_semantic(result.extracted_facts, memory_manager)

        # 3. Prozedur-Kandidat → Procedural Memory
        if result.procedure_candidate:
            await self._write_procedural(result, memory_manager)
            counts["procedural"] = 1

        log.info(
            "reflection_applied",
            session_id=result.session_id,
            counts=counts,
        )

        return counts

    # ------------------------------------------------------------------
    # Prompt-Building
    # ------------------------------------------------------------------

    def _build_reflection_prompt(self) -> str:
        """Erstellt den System-Prompt für die Reflexion. [B§6.2]"""
        return """Du bist der Reflector. Analysiere die abgeschlossene Session.

Antworte AUSSCHLIESSLICH als valides JSON-Objekt (kein Markdown, keine Erklärung).

Struktur:
{
  "success_score": 0.0-1.0,
  "evaluation": "Kurze Bewertung ob das Ziel erreicht wurde",
  "extracted_facts": [
    {
      "entity_name": "Name der Entität",
      "entity_type": "person|company|product|project|concept",
      "attribute_key": "Attributname (optional)",
      "attribute_value": "Attributwert (optional)",
      "relation_type": "Beziehungstyp (optional, z.B. hat_police, arbeitet_bei)",
      "relation_target": "Ziel-Entität der Beziehung (optional)"
    }
  ],
  "procedure_candidate": {
    "name": "kebab-case-name",
    "trigger_keywords": ["keyword1", "keyword2"],
    "prerequisite_text": "Was wird vorher gebraucht",
    "steps_text": "Nummerierte Schritte",
    "learned_text": "Was haben wir gelernt",
    "failure_patterns": ["Muster die zu Fehlern führen"],
    "tools_required": ["tool1", "tool2"],
    "is_update": false
  },
  "session_summary": {
    "goal": "Was war das Ziel",
    "outcome": "Was wurde erreicht",
    "key_decisions": ["Entscheidung 1"],
    "open_items": ["Offener Punkt 1"],
    "tools_used": ["tool1"],
    "duration_ms": 0
  },
  "failure_analysis": "Was lief schief (leer wenn alles gut)",
  "improvement_suggestions": ["Verbesserungsvorschlag 1"]
}

Regeln:
- success_score: 0.0 = totaler Misserfolg, 1.0 = perfekt
- extracted_facts: NUR konkrete, neue Fakten. Keine Vermutungen.
- procedure_candidate: NUR wenn ein wiederholbares Muster erkennbar ist \
(mind. 2 Tool-Schritte). Sonst null.
- Bei einfachen Sessions: Kurze Reflexion reicht. Nicht alles muss gefüllt sein.
- Antworte auf Deutsch."""

    def _format_session_for_reflection(
        self,
        working_memory: WorkingMemory,
        agent_result: AgentResult,
    ) -> str:
        """Formatiert die Session-Daten für den Reflection-Prompt."""
        parts: list[str] = []

        # Ziel(e) aus den Plänen
        goals = [p.goal for p in agent_result.plans if p.goal]
        if goals:
            parts.append(f"ZIELE: {'; '.join(goals)}")

        # Chat-History (kompakt)
        history_lines: list[str] = []
        for msg in working_memory.chat_history[-20:]:  # Letzte 20 Nachrichten
            role = msg.role.value.upper()
            text = msg.content[:500]  # Kürzen
            history_lines.append(f"[{role}] {text}")
        if history_lines:
            parts.append("VERLAUF:\n" + "\n".join(history_lines))

        # Tool-Ergebnisse
        result_lines: list[str] = []
        for tr in agent_result.tool_results[-10:]:
            status = "OK" if tr.success else "FEHLER"
            content = tr.content[:300]
            result_lines.append(f"[{tr.tool_name}] {status}: {content}")
        if result_lines:
            parts.append("TOOL-ERGEBNISSE:\n" + "\n".join(result_lines))

        # Plan-Details
        plan_lines: list[str] = []
        for plan in agent_result.plans:
            if plan.steps:
                for i, step in enumerate(plan.steps, 1):
                    plan_lines.append(
                        f"  {i}. {step.tool}({step.params}) [Risiko: {step.risk_estimate}]"
                    )
        if plan_lines:
            parts.append("PLAN-SCHRITTE:\n" + "\n".join(plan_lines))

        # Fehler & Blockierungen aus Audit
        blocks = [
            a for a in agent_result.audit_entries if a.decision_status == GateStatus.BLOCK
        ]
        if blocks:
            block_lines = [f"BLOCKIERT: {a.action_tool} -- {a.decision_reason}" for a in blocks[:5]]
            parts.append("\n".join(block_lines))

        # Zusammenfügen und kürzen
        text = "\n\n".join(parts)
        if len(text) > MAX_REFLECTION_INPUT_CHARS:
            text = text[:MAX_REFLECTION_INPUT_CHARS] + "\n[... gekürzt]"

        # Meta-Info
        meta = (
            f"\nMETA: {agent_result.total_iterations} Iterationen, "
            f"{len(agent_result.tool_results)} Tool-Aufrufe, "
            f"{agent_result.total_duration_ms}ms Gesamtdauer, "
            f"Erfolg: {agent_result.success}"
        )
        text += meta

        return text

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_reflection(self, text: str, session_id: str) -> ReflectionResult:
        """Parst die LLM-Antwort in ein ReflectionResult."""
        # JSON extrahieren
        data = self._extract_json(text)
        if data is None:
            log.warning("reflection_parse_failed", text_preview=text[:200])
            return ReflectionResult(
                session_id=session_id,
                success_score=0.5,
                evaluation="Reflexion konnte nicht geparst werden.",
            )

        try:
            return self._build_result_from_dict(data, session_id)
        except (KeyError, ValueError, TypeError) as exc:
            log.warning("reflection_build_failed", error=str(exc))
            return ReflectionResult(
                session_id=session_id,
                success_score=0.5,
                evaluation=f"Reflexion teilweise fehlgeschlagen: {exc}",
            )

    def _build_result_from_dict(self, data: dict[str, Any], session_id: str) -> ReflectionResult:
        """Baut ein ReflectionResult aus einem geparsten Dict."""
        # Extrahierte Fakten
        facts: list[ExtractedFact] = []
        for fact_data in data.get("extracted_facts", []):
            if not fact_data.get("entity_name"):
                continue
            facts.append(
                ExtractedFact(
                    entity_name=fact_data["entity_name"],
                    entity_type=fact_data.get("entity_type", "unknown"),
                    attribute_key=fact_data.get("attribute_key", ""),
                    attribute_value=fact_data.get("attribute_value", ""),
                    relation_type=fact_data.get("relation_type"),
                    relation_target=fact_data.get("relation_target"),
                    confidence=_safe_float(fact_data.get("confidence", 0.8), 0.8),
                    source_session=session_id,
                )
            )

        # Prozedur-Kandidat
        proc_candidate: ProcedureCandidate | None = None
        proc_data = data.get("procedure_candidate")
        if proc_data and proc_data.get("name"):
            proc_candidate = ProcedureCandidate(
                name=proc_data["name"],
                trigger_keywords=proc_data.get("trigger_keywords", []),
                prerequisite_text=proc_data.get("prerequisite_text", ""),
                steps_text=proc_data.get("steps_text", ""),
                learned_text=proc_data.get("learned_text", ""),
                failure_patterns=proc_data.get("failure_patterns", []),
                tools_required=proc_data.get("tools_required", []),
                is_update=bool(proc_data.get("is_update", False)),
            )

        # Session-Summary
        summary: SessionSummary | None = None
        summary_data = data.get("session_summary")
        if summary_data and summary_data.get("goal"):
            summary = SessionSummary(
                goal=summary_data["goal"],
                outcome=summary_data.get("outcome", ""),
                key_decisions=summary_data.get("key_decisions", []),
                open_items=summary_data.get("open_items", []),
                tools_used=summary_data.get("tools_used", []),
                duration_ms=int(_safe_float(summary_data.get("duration_ms", 0), 0)),
            )

        # Score clampen
        score = max(0.0, min(1.0, _safe_float(data.get("success_score", 0.5), 0.5)))

        return ReflectionResult(
            session_id=session_id,
            success_score=score,
            evaluation=data.get("evaluation", ""),
            extracted_facts=facts,
            procedure_candidate=proc_candidate,
            session_summary=summary,
            failure_analysis=data.get("failure_analysis", ""),
            improvement_suggestions=data.get("improvement_suggestions", []),
        )

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any] | None:
        """Extrahiert JSON aus LLM-Text (mit oder ohne Markdown-Fences)."""
        # Markdown-Code-Block entfernen
        cleaned = re.sub(r"```json\s*", "", text)
        cleaned = re.sub(r"```\s*$", "", cleaned.strip())

        # Versuche vollständigen Text als JSON
        try:
            data = json.loads(cleaned)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

        # Suche nach erstem { ... } Block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                pass

        return None

    # ------------------------------------------------------------------
    # Fallback (ohne LLM)
    # ------------------------------------------------------------------

    def _fallback_reflection(
        self, session: SessionContext, agent_result: AgentResult
    ) -> ReflectionResult:
        """Erstellt eine minimale Reflexion wenn das LLM nicht verfügbar ist."""
        tools_used = list({tr.tool_name for tr in agent_result.tool_results})
        goals = [p.goal for p in agent_result.plans if p.goal]

        return ReflectionResult(
            session_id=session.session_id,
            success_score=0.7 if agent_result.success else 0.3,
            evaluation="Automatische Reflexion (LLM nicht verfügbar).",
            session_summary=SessionSummary(
                goal=goals[0] if goals else "Unbekannt",
                outcome="Abgeschlossen" if agent_result.success else "Fehlgeschlagen",
                tools_used=tools_used,
                duration_ms=agent_result.total_duration_ms,
            ),
        )

    # ------------------------------------------------------------------
    # Memory-Schreib-Operationen
    # ------------------------------------------------------------------

    async def _write_episodic(
        self,
        result: ReflectionResult,
        memory_manager: Any,
    ) -> None:
        """Schreibt Session-Zusammenfassung ins Episodic Memory."""
        summary = result.session_summary
        if not summary:
            return

        lines: list[str] = []
        lines.append(f"- **Ziel:** {_sanitize_memory_text(summary.goal)}")
        lines.append(f"- **Ergebnis:** {_sanitize_memory_text(summary.outcome)}")
        lines.append(f"- **Score:** {result.success_score:.1f}")

        if summary.key_decisions:
            lines.append("- **Entscheidungen:** " + "; ".join(_sanitize_memory_text(d) for d in summary.key_decisions))
        if summary.open_items:
            lines.append("- **Offen:** " + "; ".join(summary.open_items))
        if summary.tools_used:
            lines.append("- **Tools:** " + ", ".join(summary.tools_used))

        content = "\n".join(lines)
        topic = f"Session {result.session_id[:8]}"

        episodic = memory_manager.episodic
        episodic.append_entry(topic, content)

        log.debug("reflection_wrote_episodic", date=date.today().isoformat())

    async def _write_semantic(
        self,
        facts: list[ExtractedFact],
        memory_manager: Any,
    ) -> int:
        """Schreibt extrahierte Fakten ins Semantic Memory."""
        indexer = memory_manager.index
        written = 0

        def _find_entity_by_name(name: str) -> Entity | None:
            """Sucht eine Entität per Name (exakter Match bevorzugt)."""
            results = indexer.search_entities(name=name)
            # Exakten Match bevorzugen
            for e in results:
                if e.name.lower() == name.lower():
                    return e
            return results[0] if results else None

        def _ensure_entity(name: str, entity_type: str, source: str, attrs: dict | None = None) -> str:
            """Erstellt oder findet eine Entität, gibt die ID zurück."""
            existing = _find_entity_by_name(name)
            if existing:
                return existing.id
            entity = Entity(
                type=entity_type,
                name=name,
                attributes=attrs or {},
                source_file=source,
            )
            indexer.upsert_entity(entity)
            return entity.id

        for fact in facts:
            # Sanitize fact fields before storage
            fact_entity_name = _sanitize_memory_text(fact.entity_name, max_len=500)
            fact_attr_value = _sanitize_memory_text(fact.attribute_value, max_len=2000) if fact.attribute_value else fact.attribute_value
            source_ref = f"reflection:{fact.source_session}"

            # Entität anlegen/finden
            existing = _find_entity_by_name(fact_entity_name)
            if existing:
                entity_id = existing.id
                # Attribute updaten wenn vorhanden
                if fact.attribute_key and fact_attr_value:
                    attrs = dict(existing.attributes)
                    attrs[fact.attribute_key] = fact_attr_value
                    updated = existing.model_copy(update={"attributes": attrs})
                    indexer.upsert_entity(updated)
                    written += 1
            else:
                attrs = {}
                if fact.attribute_key and fact_attr_value:
                    attrs[fact.attribute_key] = fact_attr_value
                entity_id = _ensure_entity(
                    fact_entity_name, fact.entity_type, source_ref, attrs,
                )
                written += 1

            # Relation anlegen wenn vorhanden
            if fact.relation_type and fact.relation_target:
                safe_target = _sanitize_memory_text(fact.relation_target, max_len=500)
                target_id = _ensure_entity(safe_target, "unknown", source_ref)
                relation = Relation(
                    source_entity=entity_id,
                    relation_type=fact.relation_type,
                    target_entity=target_id,
                    source_file=source_ref,
                )
                indexer.upsert_relation(relation)
                written += 1

        log.debug("reflection_wrote_semantic", count=written)
        return written

    async def _write_procedural(
        self,
        result: ReflectionResult,
        memory_manager: Any,
    ) -> None:
        """Schreibt Prozedur-Kandidat ins Procedural Memory."""
        candidate = result.procedure_candidate
        if not candidate:
            return

        procedural = memory_manager.procedural

        # Sanitize candidate fields before storage
        safe_name = _sanitize_memory_text(candidate.name, max_len=500)
        safe_steps = _sanitize_memory_text(candidate.steps_text) if candidate.steps_text else ""
        safe_learned = _sanitize_memory_text(candidate.learned_text) if candidate.learned_text else ""

        # Prozedur-Body im SKILL.md-Format aufbauen
        body_parts: list[str] = [f"# {safe_name}\n"]

        if candidate.prerequisite_text:
            body_parts.append(f"## Voraussetzungen\n{candidate.prerequisite_text}\n")

        if safe_steps:
            body_parts.append(f"## Ablauf\n{safe_steps}\n")

        if safe_learned:
            body_parts.append(f"## Gelerntes\n{safe_learned}\n")

        if candidate.failure_patterns:
            body_parts.append(
                "## Fehler-Muster\n"
                + "\n".join(f"- {p}" for p in candidate.failure_patterns)
                + "\n"
            )

        body = "\n".join(body_parts)

        # Metadata
        from jarvis.models import ProcedureMetadata

        metadata = ProcedureMetadata(
            name=candidate.name,
            trigger_keywords=candidate.trigger_keywords,
            tools_required=candidate.tools_required,
            learned_from=[result.session_id],
            source_file="",
        )

        if candidate.is_update:
            existing = procedural.load_procedure(candidate.name)
            if existing:
                old_meta, old_body = existing
                # Body zusammenführen -- neues ans Ende
                merged_body = old_body.rstrip() + "\n\n---\n\n" + body
                metadata = ProcedureMetadata(
                    name=old_meta.name,
                    trigger_keywords=list(
                        set(old_meta.trigger_keywords + candidate.trigger_keywords)
                    ),
                    tools_required=list(set(old_meta.tools_required + candidate.tools_required)),
                    success_count=old_meta.success_count,
                    failure_count=old_meta.failure_count,
                    total_uses=old_meta.total_uses,
                    avg_score=old_meta.avg_score,
                    learned_from=[*old_meta.learned_from, result.session_id],
                    failure_patterns=old_meta.failure_patterns + candidate.failure_patterns,
                    improvements=old_meta.improvements + result.improvement_suggestions,
                    source_file=old_meta.source_file,
                )
                body = merged_body

        procedural.save_procedure(
            name=candidate.name,
            body=body,
            metadata=metadata,
        )

        # Usage tracken
        procedural.record_usage(
            candidate.name,
            success=result.was_successful,
            score=result.success_score,
        )

        log.debug(
            "reflection_wrote_procedural",
            name=candidate.name,
            is_update=candidate.is_update,
        )
