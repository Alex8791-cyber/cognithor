"""Planner: LLM-based understanding, planning, and reflecting.

The Planner is the "brain" of Jarvis. It:
  - Understands user messages
  - Searches memory for relevant context
  - Creates structured plans (ActionPlan)
  - Interprets tool results
  - Formulates responses

The Planner has NO direct access to tools or files.
It can only read (memory) and think (create plans).

Bible reference: §3.1 (Planner), §3.4 (Cycle)
"""

from __future__ import annotations

import json
import re
import time
from typing import TYPE_CHECKING, Any

from jarvis.core.model_router import ModelRouter, OllamaClient, OllamaError
from jarvis.models import (
    ActionPlan,
    PlannedAction,
    RiskLevel,
    ToolResult,
    WorkingMemory,
)
from jarvis.utils.logging import get_logger

if TYPE_CHECKING:
    from jarvis.audit import AuditLogger
    from jarvis.config import JarvisConfig

log = get_logger(__name__)


# =============================================================================
# System-Prompts
# =============================================================================


# =============================================================================
# System-Prompts (optimiert für Qwen3)
# =============================================================================

SYSTEM_PROMPT = """\
Du bist Jarvis, ein autonomes Agent-Betriebssystem aus dem Cognithor-Projekt \
(entwickelt von Alexander Söllner). Du bist intelligent, kreativ und vielseitig.
Du bist der Planner -- du verstehst Anfragen und entscheidest, ob du direkt \
antworten oder einen Tool-Plan erstellen musst.

## Deine Rolle
- Du bist ein leistungsfähiger KI-Agent, der eigenständig denken, planen und \
Probleme lösen kann. Du kannst Code schreiben, im Web recherchieren, Dateien \
verwalten und Shell-Befehle ausführen.
- Wenn du Dateien lesen/schreiben, Befehle ausführen oder im Wissen suchen musst, \
erstellst du einen Plan. Der Executor führt ihn aus.
- Du sprichst Deutsch. {owner_name} duzt dich.
- Denke Schritt für Schritt nach, bevor du antwortest.
- Unterschätze deine Fähigkeiten NICHT. Du kannst Code generieren, Software \
erstellen, Webrecherchen durchführen und komplexe Aufgaben autonom lösen.

## Verfügbare Tools
{tools_section}

## Antwort-Format

WICHTIG: Wähle GENAU EINE Option. Vermische NIEMALS Text und JSON.

### OPTION A -- Direkte Antwort
Für Wissensfragen, Erklärungen, Meinungen, Smalltalk, Nachfragen.
Antworte einfach als normaler Text. KEIN JSON, KEIN Code-Block.

### OPTION B -- Tool-Plan
Für alles was Dateien, Shell, Web, Memory oder Dokument-Erstellung erfordert.
Antworte mit EXAKT diesem JSON-Format in einem ```json Block:

```json
{{
  "goal": "Was soll erreicht werden",
  "reasoning": "Warum dieser Ansatz (1 Satz)",
  "steps": [
    {{
      "tool": "EXAKTER_TOOL_NAME",
      "params": {{"param_name": "wert"}},
      "rationale": "Warum dieser Schritt"
    }}
  ],
  "confidence": 0.85
}}
```

### Beispiel: User sagt „Was weißt du über Projekt Alpha?"
```json
{{
  "goal": "Informationen zu Projekt Alpha aus Memory abrufen",
  "reasoning": "Projektdaten sind im Semantic Memory gespeichert.",
  "steps": [
    {{
      "tool": "search_memory",
      "params": {{"query": "Projekt Alpha"}},
      "rationale": "Memory nach allen Informationen zu Projekt Alpha durchsuchen"
    }}
  ],
  "confidence": 0.9
}}
```

### Beispiel: User fragt nach einem aktuellen Ereignis
```json
{{
  "goal": "Aktuelle Informationen über das Ereignis recherchieren",
  "reasoning": "Faktenfrage zu einem aktuellen Ereignis -- mein Wissen könnte veraltet sein.",
  "steps": [
    {{
      "tool": "search_and_read",
      "params": {{"query": "USA Venezuela Maduro Militäroperation 2026", "num_results": 3}},
      "rationale": "Web-Recherche mit Keywords, Seiteninhalte lesen für vollständige Informationen"
    }}
  ],
  "confidence": 0.9
}}
```

### Beispiel: User sagt „Erstelle ein Kündigungsschreiben als PDF"
```json
{{
  "goal": "Kündigungsschreiben als PDF erstellen",
  "reasoning": "Der User will ein Dokument erstellt bekommen.",
  "steps": [
    {{
      "tool": "document_export",
      "params": {{
        "content": "Sehr geehrte Damen und Herren,\\n\\nhiermit kündige ich ...",
        "format": "pdf",
        "title": "Kündigung",
        "filename": "kuendigung"
      }},
      "rationale": "PDF-Dokument mit dem Kündigungstext generieren"
    }}
  ],
  "confidence": 0.95
}}
```

### Beispiel: User sagt „Was ist eine API?"
Direkte Textantwort (Option A): „Eine API ist eine Programmierschnittstelle..."

## Entscheidungshilfe

| Anfrage enthält... | Option | Typisches Tool |
|---------------------|--------|----------------|
| Allgemeine Erklärung, Smalltalk, Meinung | A | -- |
| Aktuelle Ereignisse, Politik, Nachrichten, Fakten, „wann", „was ist passiert" | B | search_and_read (bevorzugt) oder web_news_search |
| „Datei", „lesen", „erstellen", „schreiben" | B | read_file / write_file |
| „Verzeichnis", „Ordner", „auflisten" | B | list_directory |
| „Befehl", „ausführen", „Shell", „Code" | B | exec_command |
| „suchen", „googlen", „Web", „recherchiere" | B | search_and_read |
| „erinnern", „Memory", „was weißt du über" | B | search_memory |
| „speichern", „merken" | B | save_to_memory |
| „Kontakt", „Entität" | B | get_entity / add_entity |
| „Prozedur", „wie mache ich" | B | search_procedures |
| „PDF", „DOCX", „Brief", „Schreiben", „Dokument", „Kündigung", „Vertrag", „Bewerbung", „erstelle als" | B | document_export |
| Unklare Anfrage | A | -- (nachfragen) |

WICHTIG: Wenn eine Frage sich auf aktuelle Ereignisse, politische Geschehnisse, \
Nachrichten, Daten oder Fakten bezieht, die sich ändern können, nutze IMMER \
search_and_read statt aus dem Gedächtnis zu antworten. Dein Wissen kann veraltet sein. \
Antworte bei Faktenfragen NIEMALS aus dem Gedächtnis -- nutze IMMER ein Such-Tool.

### Tipps für bessere Suchergebnisse
- **Bevorzuge search_and_read** statt web_search -- es liest die Seiteninhalte und liefert \
dir den vollen Text, nicht nur kurze Snippets. Nutze es für alle Faktenfragen.
- Bei aktuellen Nachrichten: web_news_search mit `"timelimit": "w"`.
- Formuliere die Suchanfrage als KEYWORDS, NICHT als Frage. \
Beispiel: Statt „Wann hat die USA den venezolanischen Präsidenten entführt?" → \
`"USA Maduro Venezuela Entführung 2026"` oder `"US military operation Venezuela Maduro"`.
- Setze `"timelimit": "m"` bei aktuellen Ereignissen.
- Bei unklaren Ergebnissen: Zweite Suche mit anderen Keywords oder auf Englisch.

## Regeln
- Verwende NUR Tool-Namen aus der obigen Liste. Erfinde KEINE Tools.
- Jeder Step braucht „tool", „params" und „rationale".
- Bei mehreren Steps: Logische Reihenfolge. Ergebnisse fließen in Folgeschritte.
- confidence: 0.0--1.0. Unter 0.5 = besser nachfragen.
- Im Zweifel: OPTION A wählen und nachfragen.
- Antworte ENTWEDER als Text ODER als JSON-Plan. Niemals beides vermischen.
- Wenn dir eine Prozedur im Kontext angezeigt wird, folge deren Ablauf.
- WICHTIG: Wenn im Kontext bereits "AKTUELLE FAKTEN AUS DEM INTERNET" oder \
"Web-Suchergebnisse" stehen, nutze diese Informationen DIREKT in deiner Antwort \
(Option A). Du brauchst dann KEINEN neuen Such-Plan. Die Suchergebnisse sind AKTUELL \
und KORREKT -- dein Trainingswissen ist dagegen VERALTET. Basiere deine Antwort \
AUSSCHLIEẞLICH auf den bereitgestellten Suchergebnissen.

## Aktuelles Datum und Uhrzeit
{current_datetime}

## Kontext
{context_section}
"""

REPLAN_PROMPT = """\
## Bisherige Ergebnisse

{results_section}

## Aufgabe
Ursprüngliches Ziel: {original_goal}

## WICHTIGE REGELN für die Auswertung
- Wenn ein Tool ERFOLGREICH war (✓), NUTZE dessen Ergebnis in deiner Antwort.
- Ignoriere blockierte oder fehlgeschlagene Schritte (✗), wenn andere Schritte \
das Ziel bereits erreicht haben.
- Gib dem User NIEMALS Anleitungen, Dinge manuell zu tun, wenn du die Antwort \
bereits aus den Ergebnissen ableiten kannst.
- Du bist ein autonomer Agent -- du löst Probleme selbst, du delegierst NICHT an den User.

### KRITISCH -- Umgang mit Suchergebnissen (web_search, web_news_search, search_and_read)
- Wenn ein Suchergebnis vorliegt, sind die SUCHERGEBNISSE deine EINZIGE Faktenquelle.
- Dein Trainingswissen ist VERALTET. Die Suchergebnisse sind AKTUELL und KORREKT.
- Vertraue den Suchergebnissen, AUCH wenn sie deinem Vorwissen widersprechen.
- Wenn die Suchergebnisse ein Ereignis beschreiben, dann IST es passiert.
- Erfinde KEINE Fakten, die nicht in den Suchergebnissen stehen.
- Zitiere konkrete Informationen (Daten, Namen, Orte) DIREKT aus den Ergebnissen.
- Sage NIEMALS „es gibt keinen Beleg", „das ist fiktiv" oder „das ist nicht passiert", \
wenn die Suchergebnisse das Gegenteil belegen.
- Bezeichne Suchergebnisse NIEMALS als „hypothetisch" oder „fiktional".

Analysiere die bisherigen Ergebnisse und entscheide dich für GENAU EINE Option:

**OPTION 1 -- Aufgabe erledigt** → Formuliere eine hilfreiche Antwort als normaler Text. \
KEIN JSON. Fasse die ERFOLGREICHEN Ergebnisse zusammen und beantworte die ursprüngliche Frage. \
Nutze konkrete Daten aus den Ergebnissen.

**OPTION 2 -- Weitere Schritte nötig** → Erstelle einen neuen JSON-Plan (```json Block). \
Nutze die bisherigen Ergebnisse als Kontext. Plane nur die FEHLENDEN Schritte.

**OPTION 3 -- Fehler aufgetreten** → NUR wenn ALLE Schritte fehlgeschlagen sind. \
Analysiere die Ursache und erstelle einen neuen Plan mit anderem Ansatz.

Antworte ENTWEDER als Text ODER als JSON-Plan. Niemals beides vermischen.
"""

ESCALATION_PROMPT = """\
Die Aktion "{tool}" wurde vom Gatekeeper blockiert.
Grund: {reason}

Formuliere eine kurze, höfliche Nachricht auf Deutsch:
1. Was du versucht hast
2. Warum es blockiert wurde (verständlich, nicht technisch)
3. Was der Benutzer tun kann (z.B. Genehmigung erteilen, Alternative vorschlagen)

Maximal 3 Sätze.
"""


class PlannerError(Exception):
    """Error in the Planner."""


class Planner:
    """LLM-based Planner. Understands, plans, reflects. [B§3.1]"""

    def __init__(
        self,
        config: JarvisConfig,
        ollama: Any,
        model_router: ModelRouter,
        audit_logger: AuditLogger | None = None,
        causal_analyzer: Any = None,
        task_profiler: Any = None,
        cost_tracker: Any = None,
    ) -> None:
        """Initialisiert den Planner mit LLM-Client und Model-Router.

        Args:
            config: Jarvis-Konfiguration.
            ollama: LLM-Client (OllamaClient oder UnifiedLLMClient).
                    Muss `chat(model, messages, **kwargs)` unterstützen.
            model_router: Model-Router für Modellauswahl.
            audit_logger: Optionaler AuditLogger für LLM-Call-Protokollierung.
            causal_analyzer: Optionaler CausalAnalyzer für Tool-Vorschlaege.
            task_profiler: Optionaler TaskProfiler fuer Selbsteinschaetzung.
            cost_tracker: Optionaler CostTracker fuer LLM-Kosten-Tracking.
        """
        self._config = config
        self._ollama = ollama
        self._router = model_router
        self._audit_logger = audit_logger
        self._causal_analyzer = causal_analyzer
        self._task_profiler = task_profiler
        self._cost_tracker = cost_tracker

    def _record_cost(self, response: dict[str, Any], model: str, session_id: str = "") -> None:
        """Records LLM call cost if cost_tracker is available."""
        if self._cost_tracker is None:
            return
        try:
            input_tokens = response.get("prompt_eval_count", 0)
            output_tokens = response.get("eval_count", 0)
            if input_tokens or output_tokens:
                self._cost_tracker.record_llm_call(
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    session_id=session_id,
                )
        except Exception as exc:
            log.debug("cost_tracking_failed", error=str(exc))

    async def plan(
        self,
        user_message: str,
        working_memory: WorkingMemory,
        tool_schemas: dict[str, Any],
    ) -> ActionPlan:
        """Erstellt einen Plan für eine User-Nachricht.

        Args:
            user_message: Die Nachricht des Users
            working_memory: Aktiver Session-Kontext (Memory, History)
            tool_schemas: Verfügbare Tools als JSON-Schema

        Returns:
            ActionPlan mit Schritten oder einer direkten Antwort.
        """
        model = self._router.select_model("planning", "high")
        model_config = self._router.get_model_config(model)

        # System-Prompt bauen
        system_prompt = self._build_system_prompt(
            working_memory=working_memory,
            tool_schemas=tool_schemas,
        )

        # Messages zusammenbauen
        messages = self._build_messages(
            system_prompt=system_prompt,
            working_memory=working_memory,
            user_message=user_message,
        )

        # LLM aufrufen
        _plan_start = time.monotonic()
        try:
            response = await self._ollama.chat(
                model=model,
                messages=messages,
                temperature=model_config.get("temperature", 0.7),
                top_p=model_config.get("top_p", 0.9),
                options={"num_predict": getattr(self._config.planner, "response_token_budget", 3000)},
            )
        except OllamaError as exc:
            _plan_ms = int((time.monotonic() - _plan_start) * 1000)
            log.error("planner_llm_error", error=str(exc))
            if self._audit_logger:
                self._audit_logger.log_tool_call(
                    "llm_plan", {"model": model, "goal": user_message[:100]},
                    result=f"ERROR: {exc}", success=False,
                    duration_ms=float(_plan_ms),
                )
            return ActionPlan(
                goal=user_message,
                reasoning="LLM-Fehler -- kann nicht planen",
                direct_response=f"Entschuldigung, ich hatte ein technisches Problem: {exc}",
                confidence=0.0,
            )

        _plan_ms = int((time.monotonic() - _plan_start) * 1000)
        self._record_cost(response, model, session_id=working_memory.session_id)
        if self._audit_logger:
            self._audit_logger.log_tool_call(
                "llm_plan", {"model": model, "goal": user_message[:100]},
                result=f"OK ({_plan_ms}ms)", success=True,
                duration_ms=float(_plan_ms),
            )

        # Antwort parsen
        assistant_text = response.get("message", {}).get("content", "")

        # Prüfe ob die Antwort Tool-Calls enthält (Ollama native)
        tool_calls = response.get("message", {}).get("tool_calls", [])
        if tool_calls:
            return self._parse_tool_calls(tool_calls, user_message)

        # Prüfe ob JSON-Plan in der Antwort steckt
        plan = self._extract_plan(assistant_text, user_message)
        return plan

    async def replan(
        self,
        original_goal: str,
        results: list[ToolResult],
        working_memory: WorkingMemory,
        tool_schemas: dict[str, Any],
    ) -> ActionPlan:
        """Erstellt einen neuen Plan basierend auf bisherigen Ergebnissen. [B§3.4]

        Wird aufgerufen wenn der Agent-Loop weitere Iterationen braucht.
        """
        model = self._router.select_model("planning", "high")
        model_config = self._router.get_model_config(model)

        # Ergebnisse formatieren
        results_text = self._format_results(results)

        # System-Prompt + Replan-Prompt
        system_prompt = self._build_system_prompt(
            working_memory=working_memory,
            tool_schemas=tool_schemas,
        )

        replan_text = REPLAN_PROMPT.format(
            results_section=results_text,
            original_goal=original_goal,
        )

        # Messages mit bisheriger History + Replan-Prompt
        messages = self._build_messages(
            system_prompt=system_prompt,
            working_memory=working_memory,
            user_message=replan_text,
        )

        try:
            response = await self._ollama.chat(
                model=model,
                messages=messages,
                temperature=model_config.get("temperature", 0.7),
                top_p=model_config.get("top_p", 0.9),
                options={"num_predict": getattr(self._config.planner, "response_token_budget", 3000)},
            )
        except OllamaError as exc:
            log.error("planner_replan_error", error=str(exc))
            return ActionPlan(
                goal=original_goal,
                direct_response="Entschuldigung, ich konnte den Plan nicht fortsetzen.",
                confidence=0.0,
            )

        self._record_cost(response, model, session_id=working_memory.session_id)
        assistant_text = response.get("message", {}).get("content", "")

        # Prüfe ob Tool-Calls in der Antwort
        tool_calls = response.get("message", {}).get("tool_calls", [])
        if tool_calls:
            return self._parse_tool_calls(tool_calls, original_goal)

        return self._extract_plan(assistant_text, original_goal)

    async def generate_escalation(
        self,
        tool: str,
        reason: str,
        working_memory: WorkingMemory,
    ) -> str:
        """Generiert eine Eskalations-Nachricht wenn ein Tool 3x blockiert wurde. [B§3.4]"""
        model = self._router.select_model("simple_tool_call", "low")

        messages = [
            {"role": "system", "content": "Du bist Jarvis. Erkläre höflich auf Deutsch."},
            {"role": "user", "content": ESCALATION_PROMPT.format(tool=tool, reason=reason)},
        ]

        try:
            response = await self._ollama.chat(
                model=model, messages=messages,
                options={"num_predict": getattr(self._config.planner, "response_token_budget", 3000)},
            )
            self._record_cost(response, model, session_id=working_memory.session_id)
            content: str = response.get("message", {}).get("content", "")
            return content
        except OllamaError:
            return (
                f"Ich habe mehrfach versucht, '{tool}' auszuführen, "
                f"aber es wurde blockiert: {reason}. "
                "Bitte hilf mir, das anders zu lösen."
            )

    async def formulate_response(
        self,
        user_message: str,
        results: list[ToolResult],
        working_memory: WorkingMemory,
    ) -> str:
        """Formuliert eine finale Antwort basierend auf Tool-Ergebnissen.

        Wird am Ende des Agent-Loops aufgerufen, wenn alle Tools
        ausgeführt wurden und eine zusammenfassende Antwort nötig ist.
        """
        model = self._router.select_model("planning", "medium")

        results_text = self._format_results(results)

        # Prüfe ob Suchergebnisse unter den Tool-Ergebnissen sind
        has_search_results = any(
            r.tool_name in ("web_search", "web_news_search", "search_and_read", "web_fetch")
            and r.success
            for r in results
        )

        if has_search_results:
            # Extrahiere den tatsächlichen Such-Content für die Antwort
            search_content_parts = []
            for r in results:
                if r.tool_name in ("web_search", "web_news_search", "search_and_read", "web_fetch") and r.success:
                    search_content_parts.append(r.content[:5000])
            search_content_block = "\n\n".join(search_content_parts)

            prompt = (
                f"Der User hat gefragt: {user_message}\n\n"
                f"## Suchergebnisse aus dem Internet (AKTUELLE FAKTEN)\n\n"
                f"{search_content_block}\n\n"
                f"## Anweisungen\n"
                f"Beantworte die Frage des Users AUSSCHLIEẞLICH auf Basis der obigen Suchergebnisse.\n"
                f"REGELN:\n"
                f"1. Die Suchergebnisse sind AKTUELL und KORREKT. Dein Trainingswissen ist VERALTET.\n"
                f"2. Wenn die Suchergebnisse ein Ereignis beschreiben, dann IST es passiert.\n"
                f"3. Sage NIEMALS 'es gibt keinen Beleg' oder 'das ist nicht passiert', wenn die "
                f"Suchergebnisse das Gegenteil zeigen.\n"
                f"4. Zitiere konkrete Daten, Namen, Orte und Fakten DIREKT aus den Suchergebnissen.\n"
                f"5. Erfinde KEINE Details, die nicht in den Suchergebnissen stehen.\n"
                f"6. Antworte auf Deutsch, prägnant und faktenbasiert."
            )
        else:
            prompt = (
                f"Der User hat gefragt: {user_message}\n\n"
                f"Du hast folgende Aktionen ausgeführt und Ergebnisse erhalten:\n\n"
                f"{results_text}\n\n"
                f"Formuliere jetzt eine hilfreiche Antwort auf Deutsch.\n"
                f"WICHTIG: Nutze die ERFOLGREICHEN Ergebnisse (✓) direkt in deiner Antwort. "
                f"Ignoriere fehlgeschlagene/blockierte Schritte, wenn das Ziel trotzdem erreicht wurde. "
                f"Gib dem User KEINE Anleitungen für Dinge, die du bereits erledigt hast."
            )

        if has_search_results:
            system_content = (
                "Du bist Jarvis, ein autonomer Agent. Du beantwortest Fragen auf Deutsch.\n"
                "KRITISCHE REGEL: Dein Trainingswissen ist VERALTET. "
                "Bei Suchergebnissen aus dem Internet basiert deine Antwort AUSSCHLIEẞLICH "
                "auf den gefundenen Informationen. Die Suchergebnisse sind die WAHRHEIT. "
                "Widerspricht dein Vorwissen den Suchergebnissen, sind die Suchergebnisse KORREKT. "
                "Du darfst Suchergebnisse NICHT als 'fiktiv', 'hypothetisch' oder 'unbelegte "
                "Behauptung' bezeichnen."
            )
        else:
            system_content = (
                "Du bist Jarvis, ein autonomer Agent. Antworte hilfreich auf Deutsch. "
                "Du nutzt Tool-Ergebnisse direkt und gibst dem User NICHT Anleitungen, "
                "Dinge selbst zu tun. Du löst Probleme eigenständig."
            )

        messages = [
            {"role": "system", "content": system_content},
        ]

        # Kontext aus Working Memory einfügen
        if working_memory.core_memory_text:
            messages.append(
                {
                    "role": "system",
                    "content": f"Dein Hintergrund:\n{working_memory.core_memory_text[:500]}",
                }
            )

        messages.append({"role": "user", "content": prompt})

        try:
            response = await self._ollama.chat(model=model, messages=messages)
            self._record_cost(response, model, session_id=working_memory.session_id)
            content: str = response.get("message", {}).get("content", "")
            return content
        except OllamaError:
            # Fallback: Rohe Ergebnisse als Antwort
            return results_text

    # =========================================================================
    # Private Methoden
    # =========================================================================

    def _build_system_prompt(
        self,
        working_memory: WorkingMemory,
        tool_schemas: dict[str, Any],
    ) -> str:
        """Baut den System-Prompt mit Kontext und Tools."""
        # Tools-Section
        if tool_schemas:
            tools_lines = []
            for name, schema in tool_schemas.items():
                desc = schema.get("description", "Keine Beschreibung")
                params = schema.get("inputSchema", {}).get("properties", {})
                param_list = ", ".join(f"{k}: {v.get('type', '?')}" for k, v in params.items())
                tools_lines.append(f"- **{name}**({param_list}): {desc}")
            tools_section = "\n".join(tools_lines)
        else:
            tools_section = "Keine Tools verfügbar."

        # Context-Section (Memory)
        context_parts: list[str] = []

        if working_memory.core_memory_text:
            context_parts.append(f"### Kern-Wissen\n{working_memory.core_memory_text}")

        if working_memory.injected_memories:
            mem_texts = []
            for mem in working_memory.injected_memories[:6]:
                mem_texts.append(f"- [{mem.chunk.memory_tier.value}] {mem.chunk.text[:200]}")
            context_parts.append("### Relevantes Wissen\n" + "\n".join(mem_texts))

        if working_memory.injected_procedures:
            for proc in working_memory.injected_procedures[:2]:
                if "Web-Suchergebnis" in proc:
                    # Presearch: Web-Ergebnisse mit passender Überschrift und mehr Platz
                    context_parts.append(
                        f"### AKTUELLE FAKTEN AUS DEM INTERNET (vertraue diesen Daten!)\n"
                        f"{proc[:3000]}"
                    )
                else:
                    context_parts.append(f"### Relevante Prozedur (folge diesem Ablauf!)\n{proc[:600]}")

        # Causal-Learning-Vorschlaege (wenn verfuegbar)
        if self._causal_analyzer is not None:
            try:
                top_sequences = self._causal_analyzer.get_sequence_scores(min_occurrences=2)
                if top_sequences:
                    hints = [" → ".join(s.subsequence) for s in top_sequences[:3]]
                    context_parts.append(
                        f"### Erfahrungsbasierte Tool-Empfehlungen\n"
                        f"Erfolgreiche Tool-Muster: {'; '.join(hints)}"
                    )
            except Exception:
                pass

        # Capability-basierte Selbsteinschaetzung (wenn TaskProfiler verfuegbar)
        if self._task_profiler is not None:
            try:
                cap = self._task_profiler.get_capability_profile()
                if cap and (getattr(cap, "strengths", None) or getattr(cap, "weaknesses", None)):
                    parts = []
                    if cap.strengths:
                        parts.append(f"Staerken: {', '.join(cap.strengths[:3])}")
                    if cap.weaknesses:
                        parts.append(f"Schwaechen: {', '.join(cap.weaknesses[:3])}")
                    context_parts.append(
                        "### Selbsteinschaetzung\n" + " | ".join(parts)
                    )
            except Exception:
                pass

        context_section = "\n\n".join(context_parts) if context_parts else "Kein Kontext geladen."

        # Aktuelles Datum und Uhrzeit
        from datetime import datetime
        now = datetime.now()
        current_datetime = now.strftime("%A, %d. %B %Y, %H:%M Uhr")

        return SYSTEM_PROMPT.format(
            tools_section=tools_section,
            context_section=context_section,
            current_datetime=current_datetime,
            owner_name=self._config.owner_name,
        )

    def _build_messages(
        self,
        system_prompt: str,
        working_memory: WorkingMemory,
        user_message: str,
    ) -> list[dict[str, Any]]:
        """Baut die Message-Liste für Ollama."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
        ]

        # Chat-History einfügen (neueste zuerst, bis Budget erschöpft)
        for msg in working_memory.chat_history:
            messages.append(
                {
                    "role": msg.role.value,
                    "content": msg.content,
                }
            )

        # Aktuelle User-Nachricht
        messages.append(
            {
                "role": "user",
                "content": user_message,
            }
        )

        return messages

    def _extract_plan(self, text: str, goal: str) -> ActionPlan:
        """Extrahiert einen ActionPlan aus der LLM-Antwort.

        Versucht JSON zu parsen. Wenn kein JSON gefunden wird,
        wird der Text als direkte Antwort interpretiert.
        """
        # Versuche JSON-Block zu finden (```json ... ```)
        json_match = re.search(
            r"```(?:json)?\s*\n?(.*?)\n?\s*```",
            text,
            re.DOTALL,
        )

        if json_match:
            json_str = json_match.group(1).strip()
            try:
                data = json.loads(json_str)
                return self._parse_plan_json(data, goal)
            except json.JSONDecodeError as exc:
                log.warning("planner_json_parse_failed", error=str(exc), text=json_str[:200])

        # Versuche rohen JSON zu parsen (ohne Code-Block)
        # Finde erstes { und letztes }
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace >= 0 and last_brace > first_brace:
            json_str = text[first_brace : last_brace + 1]
            try:
                data = json.loads(json_str)
                if "steps" in data or "goal" in data:
                    return self._parse_plan_json(data, goal)
            except json.JSONDecodeError:
                pass

        # Kein JSON gefunden → direkte Antwort
        return ActionPlan(
            goal=goal,
            reasoning="Direkte Antwort (kein Tool-Call nötig)",
            direct_response=text.strip(),
            confidence=0.8,
        )

    def _parse_plan_json(self, data: dict[str, Any], goal: str) -> ActionPlan:
        """Parst ein JSON-Dict in einen ActionPlan.

        Robust gegen fehlende oder unerwartete Felder.
        """
        steps: list[PlannedAction] = []

        for step_data in data.get("steps", []):
            if not isinstance(step_data, dict):
                continue
            try:
                step = PlannedAction(
                    tool=step_data.get("tool", "unknown"),
                    params=step_data.get("params", {}),
                    rationale=step_data.get("rationale", ""),
                    depends_on=step_data.get("depends_on", []),
                    risk_estimate=step_data.get("risk_estimate", RiskLevel.ORANGE),
                    rollback=step_data.get("rollback"),
                )
                steps.append(step)
            except Exception as exc:
                log.warning("planner_step_parse_failed", error=str(exc))
                continue

        return ActionPlan(
            goal=data.get("goal", goal),
            reasoning=data.get("reasoning", ""),
            steps=steps,
            memory_context=data.get("memory_context", []),
            confidence=min(max(data.get("confidence", 0.5), 0.0), 1.0),
            direct_response=data.get("direct_response"),
        )

    def _parse_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
        goal: str,
    ) -> ActionPlan:
        """Parst Ollama-native Tool-Calls in einen ActionPlan."""
        steps: list[PlannedAction] = []

        for tc in tool_calls:
            func = tc.get("function", {})
            step = PlannedAction(
                tool=func.get("name", "unknown"),
                params=func.get("arguments", {}),
                rationale="Tool-Call vom Modell vorgeschlagen",
                risk_estimate=RiskLevel.ORANGE,  # Konservativ
            )
            steps.append(step)

        return ActionPlan(
            goal=goal,
            reasoning="Plan basiert auf Modell-Tool-Calls",
            steps=steps,
            confidence=0.7,
        )

    # Tools deren Ergebnisse mehr Kontext brauchen (größeres Content-Limit)
    _HIGH_CONTEXT_TOOLS: frozenset[str] = frozenset({
        "web_search", "web_news_search", "search_and_read", "web_fetch",
    })

    def _format_results(self, results: list[ToolResult]) -> str:
        """Formatiert Tool-Ergebnisse als lesbaren Text."""
        if not results:
            return "Keine Ergebnisse."

        parts: list[str] = []
        for i, r in enumerate(results, 1):
            status = "✓" if r.success else "✗"
            # Suchergebnisse bekommen mehr Platz (4000 Zeichen),
            # andere Tools bleiben bei 1000 Zeichen
            limit = 4000 if r.tool_name in self._HIGH_CONTEXT_TOOLS else 1000
            content = r.content[:limit]
            if r.truncated or len(r.content) > limit:
                content += "\n[... Output gekürzt]"
            parts.append(f"### Schritt {i}: {r.tool_name} [{status}]\n{content}")

        return "\n\n".join(parts)
