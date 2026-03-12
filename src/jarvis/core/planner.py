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

import asyncio
import json
import re
import time
from typing import TYPE_CHECKING, Any

from jarvis.core.model_router import ModelRouter, OllamaClient, OllamaError
from jarvis.models import (
    ActionPlan,
    MessageRole,
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

## Sprachstil
- Antworte in natuerlicher, gesprochener Sprache -- so wie ein Mensch in einem Gespraech reden wuerde.
- Vermeide Aufzaehlungen, Bullet-Points und technische Formatierung, wenn nicht explizit verlangt. Formuliere fliessende Saetze.
- Sei direkt und praegnant, aber nicht roboterhaft. Kurze, klare Saetze statt verschachtelter Konstruktionen.
- Du darfst umgangssprachlich sein -- "also", "na ja", "schau mal", "okay" klingt menschlicher.
- Wenn du etwas erklaerst, stell dir vor du redest mit einem Freund: locker, verstaendlich, auf den Punkt.
- Bei Faktenfragen: 2-3 Saetze, nicht als Liste. Nur bei expliziten Listen-Anfragen darfst du Aufzaehlungen nutzen.

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
| „Skill", „Skills", „was kannst du", „welche Tools", „Fähigkeiten" | B | list_skills |
| „PDF", „DOCX", „Brief", „Schreiben", „Dokument", „Kündigung", „Vertrag", „Bewerbung", „erstelle als" | B | document_export |
| „Code", „Script", „Programm", „debugge", „programmiere" | B | run_python / write_file |
| „analysiere Code", „Code-Review", „Code prüfen" | B | analyze_code |
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

## Autonomer Coding-Modus
Wenn der User Code schreiben, Software erstellen oder debuggen will, \
arbeitest du AUTONOM in einer Schleife:
1. Schreiben (write_file / run_python)
2. Ausführen (run_python / exec_command)
3. Analysieren (analyze_code)
4. Korrigieren (edit_file)
5. Wiederholen bis fehlerfrei

WICHTIG für Code-Ausführung:
- Nutze IMMER run_python statt exec_command für Python/Script-Code.
- Schreibe Code IMMER in Python, NICHT in Bash/Shell-Skripten.
- run_python nimmt Code direkt als String -- kein Umweg über Dateien nötig.
- exec_command nur für System-Befehle (ls, git, pip install, etc.), NICHT für Scripts.

Regeln für Coding:
- Gib bei einem Fehler NIEMALS auf. Analysiere den Fehler und erstelle einen Fix-Plan.
- Teste Code IMMER nach dem Schreiben mit run_python.
- Nutze analyze_code für Qualitäts- und Sicherheitschecks.
- Arbeite autonom bis das Ergebnis FEHLERFREI ist oder der User unterbricht.
- Erst nach 3x identischem Fehler abbrechen und dem User berichten.

## Regeln
- Verwende NUR Tool-Namen aus der obigen Liste. Erfinde KEINE Tools.
- Jeder Step braucht „tool", „params" und „rationale".
- Bei mehreren Steps: Logische Reihenfolge. Ergebnisse fließen in Folgeschritte.
- confidence: 0.0--1.0. Unter 0.5 = besser nachfragen.
- Im Zweifel: OPTION A wählen und nachfragen.
- Antworte ENTWEDER als Text ODER als JSON-Plan. Niemals beides vermischen.
- Wenn dir eine Prozedur im Kontext angezeigt wird, folge deren Ablauf.
- SELBSTAUSKUNFT: Wenn der User nach deinen Skills, Tools, Fähigkeiten oder \
Können fragt, nutze IMMER list_skills (Option B). Beantworte solche Fragen \
NIEMALS aus dem Gedächtnis -- du weißt nicht, welche Skills installiert sind, \
ohne das Tool aufzurufen. Dein INVENTAR in der Core Memory zeigt den aktuellen Stand.
- INVENTAR-PFLEGE: Wenn du einen neuen Skill erstellst (create_skill) oder eine \
Prozedur speicherst, aktualisiere anschließend den INVENTAR-Abschnitt in CORE.md \
via edit_file. Halte die Liste immer aktuell.
- WICHTIG: Wenn im Kontext bereits "AKTUELLE FAKTEN AUS DEM INTERNET" oder \
"Web-Suchergebnisse" stehen, nutze diese Informationen DIREKT in deiner Antwort \
(Option A). Du brauchst dann KEINEN neuen Such-Plan. Die Suchergebnisse sind AKTUELL \
und KORREKT -- dein Trainingswissen ist dagegen VERALTET. Basiere deine Antwort \
AUSSCHLIEẞLICH auf den bereitgestellten Suchergebnissen.

## Aktuelles Datum und Uhrzeit
{current_datetime}
{personality_section}
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
Analysiere den Fehler GENAU (Fehlermeldung, Zeile, Ursache). Erstelle einen konkreten \
Fix-Plan mit anderem Ansatz. GIB NICHT AUF -- versuche mindestens 3 verschiedene Ansätze. \
Erst nach 3x identischem Fehler abbrechen und dem User berichten.

## KRITISCH -- NIEMALS REPLAN-Text ohne Tool-Schritte ausgeben
- Antworte ENTWEDER als Text (Option 1) ODER als JSON-Plan (Option 2/3). \
Niemals beides vermischen.
- VERBOTEN: Texte wie "REPLAN-GRUND:", "KORRIGIERTER PLAN:", "BETROFFENE SCHRITTE:", \
"AKTUALISIERTE RISIKOBEWERTUNG:" OHNE einen JSON-Plan mit konkreten Tool-Aufrufen.
- Wenn du weitere Schritte brauchst, MUSST du einen ```json Plan mit "steps" ausgeben. \
Reine Textbeschreibungen von Plänen sind NUTZLOS -- nur Tool-Aufrufe im JSON-Format \
werden tatsächlich ausgeführt.
- Du bist ein AUTONOMER Agent. Handle. Beschreibe nicht, was du tun würdest -- tu es.
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
        personality_engine: Any = None,
        prompt_evolution: Any = None,
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
            personality_engine: Optionale PersonalityEngine fuer warme Antworten.
            prompt_evolution: Optionale PromptEvolutionEngine fuer A/B-Tests.
        """
        self._config = config
        self._ollama = ollama
        self._router = model_router
        self._audit_logger = audit_logger
        self._causal_analyzer = causal_analyzer
        self._task_profiler = task_profiler
        self._cost_tracker = cost_tracker
        self._personality_engine = personality_engine
        self._prompt_evolution = prompt_evolution
        self._current_prompt_version_id: str | None = None

        # Tool-Descriptions-Cache (#40 Optimierung)
        self._cached_tools_section: str | None = None
        self._cached_tools_hash: int = 0

        # Circuit Breaker für LLM-Calls (#42 Optimierung)
        from jarvis.utils.circuit_breaker import CircuitBreaker

        self._llm_circuit_breaker = CircuitBreaker(
            name="planner_llm",
            failure_threshold=5,
            recovery_timeout=15.0,
            half_open_max_calls=2,
        )

        # Prompts von Disk laden (mit Fallback auf hardcoded Konstanten)
        self._system_prompt_template = self._load_prompt_from_file(
            "SYSTEM_PROMPT.md", SYSTEM_PROMPT
        )
        self._replan_prompt_template = self._load_prompt_from_file(
            "REPLAN_PROMPT.md", REPLAN_PROMPT, fallback_txt="REPLAN_PROMPT.txt"
        )
        self._escalation_prompt_template = self._load_prompt_from_file(
            "ESCALATION_PROMPT.md", ESCALATION_PROMPT, fallback_txt="ESCALATION_PROMPT.txt"
        )

    def _load_prompt_from_file(self, filename: str, fallback: str, fallback_txt: str = "") -> str:
        """Lädt einen Prompt von Disk (.md bevorzugt, .txt als Migration-Fallback)."""
        try:
            prompts_dir = self._config.jarvis_home / "prompts"
            path = prompts_dir / filename
            if path.exists():
                content = path.read_text(encoding="utf-8").strip()
                if content and isinstance(content, str):
                    return content
            # Migration: alte .txt-Datei als Fallback prüfen
            if fallback_txt:
                txt_path = prompts_dir / fallback_txt
                if txt_path.exists():
                    content = txt_path.read_text(encoding="utf-8").strip()
                    if content and isinstance(content, str):
                        return content
        except Exception:
            pass
        return fallback

    def reload_prompts(self) -> None:
        """Lädt alle Prompt-Templates neu von Disk."""
        self._system_prompt_template = self._load_prompt_from_file(
            "SYSTEM_PROMPT.md", SYSTEM_PROMPT
        )
        self._replan_prompt_template = self._load_prompt_from_file(
            "REPLAN_PROMPT.md", REPLAN_PROMPT, fallback_txt="REPLAN_PROMPT.txt"
        )
        self._escalation_prompt_template = self._load_prompt_from_file(
            "ESCALATION_PROMPT.md", ESCALATION_PROMPT, fallback_txt="ESCALATION_PROMPT.txt"
        )
        log.info("planner_prompts_reloaded")

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

        # LLM aufrufen (mit Circuit Breaker, #42 Optimierung)
        _plan_start = time.monotonic()
        try:
            from jarvis.utils.circuit_breaker import CircuitBreakerOpen

            response = await self._llm_circuit_breaker.call(
                self._ollama.chat(
                    model=model,
                    messages=messages,
                    temperature=model_config.get("temperature", 0.7),
                    top_p=model_config.get("top_p", 0.9),
                    options={
                        "num_predict": getattr(self._config.planner, "response_token_budget", 3000)
                    },
                )
            )
        except CircuitBreakerOpen as exc:
            _plan_ms = int((time.monotonic() - _plan_start) * 1000)
            log.warning("planner_circuit_open", remaining_s=f"{exc.remaining_seconds:.1f}")
            return ActionPlan(
                goal=user_message,
                reasoning="LLM nicht erreichbar (Circuit Breaker offen)",
                direct_response=(
                    "Das Sprachmodell ist wiederholt nicht erreichbar. "
                    f"Automatischer Retry in {exc.remaining_seconds:.0f}s.\n\n"
                    "Prüfe bitte:\n"
                    "• Läuft Ollama? → `ollama serve`\n"
                    "• Hat das System genug RAM/VRAM?"
                ),
                confidence=0.0,
            )
        except OllamaError as exc:
            _plan_ms = int((time.monotonic() - _plan_start) * 1000)
            log.error("planner_llm_error", error=str(exc), status_code=exc.status_code)
            if self._audit_logger:
                self._audit_logger.log_tool_call(
                    "llm_plan",
                    {"model": model, "goal": user_message[:100]},
                    result=f"ERROR: {exc}",
                    success=False,
                    duration_ms=float(_plan_ms),
                )
            # Specific message for model-not-found (404)
            if exc.status_code == 404:
                return ActionPlan(
                    goal=user_message,
                    reasoning=f"Modell '{model}' nicht gefunden (HTTP 404)",
                    direct_response=(
                        f"Das Sprachmodell '{model}' ist nicht installiert. "
                        f"Bitte lade es herunter:\n\n"
                        f"  ollama pull {model}\n\n"
                        f"Danach starte mich neu oder versuch es einfach erneut."
                    ),
                    confidence=0.0,
                )
            return ActionPlan(
                goal=user_message,
                reasoning="LLM-Fehler -- kann nicht planen",
                direct_response=(
                    f"Ich konnte deine Anfrage nicht verarbeiten — das Sprachmodell "
                    f"hat einen Fehler gemeldet (HTTP {exc.status_code}).\n\n"
                    f"Mögliche Ursachen:\n"
                    f"• Ollama ist nicht gestartet → `ollama serve`\n"
                    f"• Das Modell '{model}' ist nicht installiert → `ollama pull {model}`\n"
                    f"• Nicht genug VRAM/RAM für das Modell\n\n"
                    f"Fehlerdetails: {str(exc)[:200]}"
                ),
                confidence=0.0,
            )

        _plan_ms = int((time.monotonic() - _plan_start) * 1000)
        self._record_cost(response, model, session_id=working_memory.session_id)
        if self._audit_logger:
            self._audit_logger.log_tool_call(
                "llm_plan",
                {"model": model, "goal": user_message[:100]},
                result=f"OK ({_plan_ms}ms)",
                success=True,
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

        # Retry once if JSON parsing failed (LLM produced malformed JSON)
        if plan.parse_failed:
            log.warning("planner_json_retry", model=model, goal=user_message[:80])
            retry_hint = (
                "\n\nIMPORTANT: Your previous response contained malformed JSON that could not be parsed. "
                "Please respond with VALID JSON only. Use this exact format:\n"
                '```json\n{"goal": "...", "steps": [{"tool": "...", "args": {...}, "purpose": "..."}]}\n```'
            )
            retry_messages = list(messages)
            retry_messages.append({"role": "assistant", "content": assistant_text})
            retry_messages.append({"role": "user", "content": retry_hint})
            try:
                retry_response = await self._ollama.chat(
                    model=model,
                    messages=retry_messages,
                    temperature=max(0.3, model_config.get("temperature", 0.7) - 0.3),
                    top_p=model_config.get("top_p", 0.9),
                    options={
                        "num_predict": getattr(self._config.planner, "response_token_budget", 3000)
                    },
                )
                retry_text = retry_response.get("message", {}).get("content", "")
                self._record_cost(retry_response, model, session_id=working_memory.session_id)
                retry_plan = self._extract_plan(retry_text, user_message)
                if not retry_plan.parse_failed:
                    log.info("planner_json_retry_success")
                    return retry_plan
                log.warning("planner_json_retry_also_failed")
            except Exception as _retry_exc:
                log.warning("planner_json_retry_error", error=str(_retry_exc))

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

        replan_text = self._replan_prompt_template.format(
            results_section=results_text,
            original_goal=original_goal,
        )

        # Messages mit bisheriger History + Replan-Prompt
        messages = self._build_messages(
            system_prompt=system_prompt,
            working_memory=working_memory,
            user_message=replan_text,
        )

        _replan_attempts = 2
        response = None
        for _attempt in range(_replan_attempts):
            try:
                response = await self._ollama.chat(
                    model=model,
                    messages=messages,
                    temperature=model_config.get("temperature", 0.7),
                    top_p=model_config.get("top_p", 0.9),
                    options={
                        "num_predict": getattr(self._config.planner, "response_token_budget", 3000)
                    },
                )
                break
            except OllamaError as exc:
                log.warning("planner_replan_error", error=str(exc), attempt=_attempt + 1)
                if _attempt + 1 >= _replan_attempts:
                    return ActionPlan(
                        goal=original_goal,
                        direct_response=(
                            f"Ich konnte den Plan nicht fortsetzen — das Sprachmodell meldet "
                            f"Fehler nach {_replan_attempts} Versuchen.\n\n"
                            f"Fehler: {str(exc)[:200]}"
                        ),
                        confidence=0.0,
                    )
                await asyncio.sleep(1.0)  # Kurze Pause vor Retry

        self._record_cost(response, model, session_id=working_memory.session_id)
        assistant_text = response.get("message", {}).get("content", "")

        # Prüfe ob Tool-Calls in der Antwort
        tool_calls = response.get("message", {}).get("tool_calls", [])
        if tool_calls:
            return self._parse_tool_calls(tool_calls, original_goal)

        plan = self._extract_plan(assistant_text, original_goal)

        # Retry once if JSON parsing failed during replan
        if plan.parse_failed:
            log.warning("planner_replan_json_retry", model=model)
            retry_hint = (
                "\n\nIMPORTANT: Your previous response contained malformed JSON. "
                "Please respond with VALID JSON only. Use this exact format:\n"
                '```json\n{"goal": "...", "steps": [{"tool": "...", "args": {...}, "purpose": "..."}]}\n```'
            )
            retry_messages = list(messages)
            retry_messages.append({"role": "assistant", "content": assistant_text})
            retry_messages.append({"role": "user", "content": retry_hint})
            try:
                retry_response = await self._ollama.chat(
                    model=model,
                    messages=retry_messages,
                    temperature=max(0.3, model_config.get("temperature", 0.7) - 0.3),
                    top_p=model_config.get("top_p", 0.9),
                    options={
                        "num_predict": getattr(self._config.planner, "response_token_budget", 3000)
                    },
                )
                retry_text = retry_response.get("message", {}).get("content", "")
                self._record_cost(retry_response, model, session_id=working_memory.session_id)
                retry_plan = self._extract_plan(retry_text, original_goal)
                if not retry_plan.parse_failed:
                    log.info("planner_replan_json_retry_success")
                    return retry_plan
                log.warning("planner_replan_json_retry_also_failed")
            except Exception as _retry_exc:
                log.warning("planner_replan_json_retry_error", error=str(_retry_exc))

        return plan

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
            {
                "role": "user",
                "content": self._escalation_prompt_template.format(tool=tool, reason=reason),
            },
        ]

        try:
            response = await self._ollama.chat(
                model=model,
                messages=messages,
                options={
                    "num_predict": getattr(self._config.planner, "response_token_budget", 3000)
                },
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
        model = self._router.select_model("summarization", "medium")

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
                if (
                    r.tool_name in ("web_search", "web_news_search", "search_and_read", "web_fetch")
                    and r.success
                ):
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
                f"6. Antworte auf Deutsch, prägnant und faktenbasiert.\n"
                f"7. Antworte in natürlicher, gesprochener Sprache -- wie ein Mensch im Gespräch. "
                f"Keine Bullet-Points oder Listen, sondern fließende Sätze."
            )
        else:
            prompt = (
                f"Der User hat gefragt: {user_message}\n\n"
                f"Du hast folgende Aktionen ausgeführt und Ergebnisse erhalten:\n\n"
                f"{results_text}\n\n"
                f"Formuliere jetzt eine hilfreiche Antwort auf Deutsch in natuerlicher, gesprochener Sprache.\n"
                f"WICHTIG: Nutze die ERFOLGREICHEN Ergebnisse (✓) direkt in deiner Antwort. "
                f"Ignoriere fehlgeschlagene/blockierte Schritte, wenn das Ziel trotzdem erreicht wurde. "
                f"Gib dem User KEINE Anleitungen fuer Dinge, die du bereits erledigt hast. "
                f"Antworte wie ein Mensch im Gespraech -- fliessende Saetze, keine Bullet-Points oder Listen."
            )

        # Aktuelles Datum/Uhrzeit für korrekte zeitliche Bezüge
        from datetime import datetime

        now = datetime.now()
        current_dt = now.strftime("%A, %d. %B %Y, %H:%M Uhr")
        date_line = f"\nHeutiges Datum und Uhrzeit: {current_dt}\n"

        if has_search_results:
            system_content = (
                "Du bist Jarvis, ein autonomer Agent. Du beantwortest Fragen auf Deutsch "
                "in natuerlicher, gesprochener Sprache -- wie ein Mensch im Gespraech.\n"
                f"{date_line}"
                "KRITISCHE REGEL: Dein Trainingswissen ist VERALTET. "
                "Bei Suchergebnissen aus dem Internet basiert deine Antwort AUSSCHLIEẞLICH "
                "auf den gefundenen Informationen. Die Suchergebnisse sind die WAHRHEIT. "
                "Widerspricht dein Vorwissen den Suchergebnissen, sind die Suchergebnisse KORREKT. "
                "Du darfst Suchergebnisse NICHT als 'fiktiv', 'hypothetisch' oder 'unbelegte "
                "Behauptung' bezeichnen."
            )
        else:
            system_content = (
                "Du bist Jarvis, ein autonomer Agent. Antworte hilfreich auf Deutsch "
                "in natuerlicher, gesprochener Sprache -- wie ein Mensch im Gespraech.\n"
                f"{date_line}"
                "Du nutzt Tool-Ergebnisse direkt und gibst dem User NICHT Anleitungen, "
                "Dinge selbst zu tun. Du loest Probleme eigenstaendig."
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

        for _fmt_attempt in range(2):
            try:
                response = await self._ollama.chat(model=model, messages=messages)
                self._record_cost(response, model, session_id=working_memory.session_id)
                content: str = response.get("message", {}).get("content", "")
                return content
            except OllamaError as exc:
                log.warning(
                    "formulate_response_llm_error", error=str(exc), attempt=_fmt_attempt + 1
                )
                if _fmt_attempt == 0:
                    await asyncio.sleep(1.0)  # Retry nach kurzer Pause
                    continue
                # Zweiter Fehlschlag: Ergebnisse als Fallback direkt zurueckgeben
                raw_results = "\n".join(
                    f"[{r.tool_name}] {r.content[:300]}" for r in results if r.success
                )
                if raw_results:
                    return f"Hier sind die Ergebnisse (Zusammenfassung fehlgeschlagen):\n\n{raw_results}"
                return (
                    "Ich konnte die Ergebnisse leider nicht zusammenfassen — "
                    "das Sprachmodell antwortet fehlerhaft. "
                    "Prüfe, ob Ollama läuft und genug Ressourcen hat."
                )
        return ""  # Unreachable, aber fuer Type-Checker

    # =========================================================================
    # Private Methoden
    # =========================================================================

    def _build_system_prompt(
        self,
        working_memory: WorkingMemory,
        tool_schemas: dict[str, Any],
    ) -> str:
        """Baut den System-Prompt mit Kontext und Tools."""
        # Tools-Section (gecacht, #40 Optimierung)
        if tool_schemas:
            schemas_hash = hash(frozenset(tool_schemas.keys()))
            if schemas_hash != self._cached_tools_hash or self._cached_tools_section is None:
                tools_lines = []
                for name, schema in tool_schemas.items():
                    desc = schema.get("description", "No description")
                    props = schema.get("inputSchema", {}).get("properties", {})
                    required = set(schema.get("inputSchema", {}).get("required", []))
                    parts = []
                    for k, v in props.items():
                        typ = v.get("type", "?")
                        req = " [required]" if k in required else ""
                        parts.append(f"{k}: {typ}{req}")
                    param_list = ", ".join(parts)
                    tools_lines.append(f"- **{name}**({param_list}): {desc}")
                self._cached_tools_section = "\n".join(tools_lines)
                self._cached_tools_hash = schemas_hash
            tools_section = self._cached_tools_section
        else:
            tools_section = "No tools available."

        # Context-Section (Memory) — Relevanz-Ranking nach Score (#41 Optimierung)
        context_parts: list[str] = []

        if working_memory.core_memory_text:
            context_parts.append(f"### Kern-Wissen\n{working_memory.core_memory_text}")

        if working_memory.injected_memories:
            # Nach Score sortieren (höchster zuerst) mit Token-Budget
            sorted_mems = sorted(
                working_memory.injected_memories,
                key=lambda m: getattr(m, "score", 0.0),
                reverse=True,
            )
            mem_texts = []
            budget_chars = 1200  # ~300 Tokens Budget für Memories
            used = 0
            for mem in sorted_mems:
                line = f"- [{mem.chunk.memory_tier.value}] {mem.chunk.text[:200]}"
                if used + len(line) > budget_chars:
                    break
                mem_texts.append(line)
                used += len(line)
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
                    context_parts.append(
                        f"### Relevante Prozedur (folge diesem Ablauf!)\n{proc[:600]}"
                    )

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
                    context_parts.append("### Selbsteinschaetzung\n" + " | ".join(parts))
            except Exception:
                pass

        context_section = "\n\n".join(context_parts) if context_parts else "Kein Kontext geladen."

        # Aktuelles Datum und Uhrzeit
        from datetime import datetime

        now = datetime.now()
        current_datetime = now.strftime("%A, %d. %B %Y, %H:%M Uhr")

        # Personality block (optional)
        personality_section = ""
        if self._personality_engine is not None:
            try:
                personality_section = self._personality_engine.build_personality_block()
            except Exception:
                pass

        # Prompt-Evolution A/B-Test (wenn aktiv)
        if self._prompt_evolution is not None:
            try:
                version_id, template = self._prompt_evolution.get_active_version(
                    "system_prompt",
                    getattr(working_memory, "session_id", None) or "default",
                )
                self._current_prompt_version_id = version_id
                return template.format(
                    tools_section=tools_section,
                    context_section=context_section,
                    current_datetime=current_datetime,
                    owner_name=self._config.owner_name,
                    personality_section=personality_section,
                )
            except Exception:
                pass  # Fallback auf Standard-Template

        return self._system_prompt_template.format(
            tools_section=tools_section,
            context_section=context_section,
            current_datetime=current_datetime,
            owner_name=self._config.owner_name,
            personality_section=personality_section,
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
            role = msg.role.value
            content = msg.content

            # TOOL-Messages als assistant mit Präfix darstellen,
            # da nicht alle LLM-Backends die "tool"-Rolle unterstützen
            if msg.role == MessageRole.TOOL:
                role = "assistant"
                tool_label = msg.name or "tool"
                content = f"[Ergebnis von {tool_label}]\n{content}"

            messages.append(
                {
                    "role": role,
                    "content": content,
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

    @staticmethod
    def _sanitize_json_escapes(json_str: str) -> str:
        """Repariert ungültige Escape-Sequenzen in LLM-generiertem JSON.

        LLMs erzeugen häufig Bash/Regex-Code mit Backslashes (\\s, \\d, \\b, etc.)
        die in JSON-Strings ungültig sind. Diese Methode verdoppelt alleinstehende
        Backslashes die kein gültiges JSON-Escape bilden.

        Gültige JSON-Escapes: \\\", \\\\, \\/, \\b, \\f, \\n, \\r, \\t, \\uXXXX
        """
        # Ersetze Backslashes die kein gültiges JSON-Escape einleiten
        # Negative Lookahead: Backslash gefolgt von etwas das KEIN gültiges Escape ist
        return re.sub(
            r'\\(?!["\\/bfnrtu])',
            r"\\\\",
            json_str,
        )

    def _try_parse_json(self, json_str: str) -> dict[str, Any] | None:
        """Versucht JSON zu parsen, mit mehreren Fallback-Strategien.

        Reihenfolge:
          1. Normales json.loads (strict)
          2. Escape-Sanitierung (\\s, \\d etc. → \\\\s, \\\\d)
          3. strict=False (erlaubt Steuerzeichen wie literale Newlines in Strings)
          4. Sanitierung + strict=False (Kombination)
        """
        # 1. Normales Parsing
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

        # 2. Ungültige Escapes reparieren
        sanitized = self._sanitize_json_escapes(json_str)
        try:
            data = json.loads(sanitized)
            log.debug("planner_json_sanitized", strategy="escape_fix")
            return data
        except json.JSONDecodeError:
            pass

        # 3. strict=False (erlaubt Control-Characters in Strings)
        try:
            data = json.loads(json_str, strict=False)
            log.debug("planner_json_sanitized", strategy="strict_false")
            return data
        except json.JSONDecodeError:
            pass

        # 4. Beides kombiniert
        try:
            data = json.loads(sanitized, strict=False)
            log.debug("planner_json_sanitized", strategy="escape_fix+strict_false")
            return data
        except json.JSONDecodeError as exc:
            log.warning("planner_json_parse_failed", error=str(exc), text=json_str[:200])
            return None

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
            data = self._try_parse_json(json_str)
            if data is not None:
                return self._parse_plan_json(data, goal)

        # Versuche rohen JSON zu parsen (ohne Code-Block)
        # Finde erstes { und letztes }
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace >= 0 and last_brace > first_brace:
            json_str = text[first_brace : last_brace + 1]
            data = self._try_parse_json(json_str)
            if data is not None and ("steps" in data or "goal" in data):
                return self._parse_plan_json(data, goal)

        # Detect if LLM *attempted* JSON but it was malformed:
        # presence of braces or json markers signals a parse failure,
        # not a genuine direct answer.
        _has_json_markers = (
            first_brace is not None
            and first_brace >= 0
            or json_match is not None
            or '"steps"' in text
            or '"goal"' in text
        )

        if _has_json_markers:
            log.warning(
                "planner_json_parse_failed_fallback",
                text_len=len(text),
                text_preview=text[:200],
            )
            return ActionPlan(
                goal=goal,
                reasoning="JSON parse failed — needs retry",
                direct_response=text.strip(),
                confidence=0.1,
                parse_failed=True,
            )

        # Kein JSON gefunden → echte direkte Antwort (z.B. "Was ist 2+2?" → "4")
        log.debug("planner_no_json_found", text_len=len(text))
        return ActionPlan(
            goal=goal,
            reasoning="Direkte Antwort (kein Tool-Call noetig)",
            direct_response=text.strip(),
            confidence=0.5,
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
    _HIGH_CONTEXT_TOOLS: frozenset[str] = frozenset(
        {
            "web_search",
            "web_news_search",
            "search_and_read",
            "web_fetch",
            "media_analyze_image",
            "media_extract_text",
            "analyze_code",
            "run_python",
        }
    )

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
