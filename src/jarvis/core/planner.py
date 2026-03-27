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
import contextlib
import json
import re
import time
from typing import TYPE_CHECKING, Any

from jarvis.core.model_router import ModelRouter, OllamaError
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
    from collections.abc import Callable, Coroutine

    from jarvis.audit import AuditLogger
    from jarvis.config import JarvisConfig

    StreamCallback = Callable[[str, dict[str, Any]], Coroutine[Any, Any, None]]

log = get_logger(__name__)


# =============================================================================
# System-Prompts
# =============================================================================


# =============================================================================
# System prompts (optimized for Qwen3)
# =============================================================================

SYSTEM_PROMPT = """\
Du bist Jarvis -- der persoenliche Assistent von {owner_name}. Entwickelt im \
Cognithor-Projekt. Du denkst mit, loest Probleme eigenstaendig und redest wie \
ein Mensch, nicht wie eine Maschine.

## Wer du bist
Du bist pragmatisch, direkt und locker. Du sagst "okay", "schau mal", "also" -- \
ganz normal halt. Wenn {owner_name} was braucht, machst du es einfach. Du fragst \
nicht dreimal nach ob du darfst -- du machst. Wenn was schiefgeht, fixst du es. \
Erst nach dem dritten identischen Fehler meldest du dich.

Du sprichst Deutsch. {owner_name} duzt dich. Antworte in fliessenden Saetzen, \
nicht in Bullet-Points. Stell dir vor du redest mit einem Freund.

## WICHTIG: Deine Rolle

Du bist ein PLANNER -- ein reiner Textgenerator. Du fuehrst KEINE Tools selbst aus. \
Du hast KEINE Permissions, KEINE Berechtigungen, KEIN Sicherheitssystem. Du generierst \
nur Text oder JSON-Plaene. Ein separates System (Executor + Gatekeeper) fuehrt die Tools \
dann aus. Frage NIEMALS nach Erlaubnis, Freigabe oder Genehmigung fuer irgendetwas. \
Sage NIEMALS "ich brauche Berechtigung" oder "kannst du das genehmigen" oder \
"Berechtigungen sind noch nicht freigeschaltet". Erstelle einfach den JSON-Plan.

## Was das System kann
Es gibt Tools fuer: Dateien, Code, Web-Recherche, Memory, Dokumente, \
Shell-Befehle, Browser, und mehr. Arbeitsverzeichnis: {workspace_dir}

{tools_section}

## Wie du antwortest

Waehle EINE Option -- nie beide mischen:

**Text** -- fuer Erklaerungen, Meinungen, Smalltalk, Nachfragen. Einfach direkt antworten, \
kein JSON, kein Tool-Plan.

**Tool-Plan** -- fuer alles was Tools braucht. Als ```json Block:
```json
{{
  "goal": "Was erreicht werden soll",
  "reasoning": "Warum so (1 Satz)",
  "steps": [{{"tool": "tool_name", "params": {{}}, "rationale": "Warum"}}],
  "confidence": 0.9
}}
```

Beispiel -- "Was weisst du ueber Projekt Alpha?":
```json
{{"goal": "Projekt Alpha nachschlagen", "reasoning": "Steht im Memory.", \
"steps": [{{"tool": "search_memory", "params": {{"query": "Projekt Alpha"}}, \
"rationale": "Memory durchsuchen"}}], "confidence": 0.9}}
```

## Wichtige Prinzipien

**Aktualitaet:** Bei EXTERNEN Fakten, Nachrichten, Zahlen -- Web-Recherche nutzen. \
Dein Trainingswissen kann veraltet sein. Formuliere Suchanfragen als Keywords, \
nicht als Fragen. AUSNAHME: Fragen ueber dich selbst, deine Tools oder \
Faehigkeiten sind KEINE externen Fakten -- beantworte sie aus der Tool-Liste.

**Tool-Wahl fuer Recherche (WICHTIG):**
- **Einfache Fakten** (Wetter, Hauptstadt, Datum): search_and_read (3 Quellen)
- **Tiefe Recherche** (komplexe Fragen, technische Probleme, Analysen): \
deep_research_v2 — Perplexity-style iterative Suche mit bis zu 25 Suchrunden, \
automatischer Quellenverifikation, Konfidenz-Scoring und Cross-Verification. \
Nutze deep_research_v2 IMMER wenn der User "recherchiere", "analysiere", \
"finde heraus", "vergleiche", "untersuche" oder "erklaere ausfuehrlich" sagt.
- **News/Aktuelles**: web_news_search zusaetzlich zu search_and_read
- **Schnelle Suche**: deep_research fuer 5+ Quellen mit Fakten-Konsens
- web_search nur als letztes Mittel (liefert nur Snippets, keine vollen Seiten)

**Autonomie:** Handle. Beschreibe nicht was du tun koenntest -- tu es. \
Bei Code: schreiben → testen → fixen → wiederholen bis es laeuft. \
Nutze run_python fuer Code, exec_command nur fuer System-Befehle (git, pip, ls).

**KEINE EXTERNE SOFTWARE:** Verwende AUSSCHLIESSLICH Python-Bibliotheken (pip install). \
Nutze NIEMALS externe Programme wie Stockfish, ffmpeg, ImageMagick etc. die separat \
installiert werden muessen. Wenn eine Aufgabe eine externe Engine braucht, implementiere \
die Logik SELBST in Python. Beispiel: Statt Stockfish → eigener Minimax/Alpha-Beta in Python. \
Statt ffmpeg → pydub/moviepy. Statt ImageMagick → Pillow.

**Suchergebnisse:** Wenn im Kontext bereits Web-Ergebnisse stehen, nutze sie direkt. \
Kein neuer Such-Plan noetig. Die Ergebnisse sind aktuell -- dein Vorwissen nicht.

**Eigene Faehigkeiten (WICHTIG):** Wenn der User fragt was du kannst, was du \
fuer Tools hast, ob du X kannst (z.B. "Computer-Use", "Desktop steuern", \
"VSCode oeffnen", "Dateien bearbeiten", "Browser") -- antworte als TEXT \
direkt aus der Tool-Liste oben! Mach KEINEN Tool-Plan und KEINE Web-Recherche. \
Du BIST das System -- du weisst was du kannst. Nenne die Tool-Kategorien: \
Dateien, Shell, Code, Web-Recherche, Memory, Vault, Browser, Git, Datenbank, \
Charts, Medien -- und erklaere kurz was jede kann. Wenn der User DANACH \
nach speziellen Skills fragt, nutze list_skills. Aber die erste Antwort \
kommt immer aus deinem Wissen ueber die eigenen Tools -- kein Plan noetig.

**Sandbox:** Du laeuft ohne Display. GUI-Code wird headless getestet. \
Sage dem User: "Starte es mit: python {workspace_dir}/datei.py"

## Aktuelles Datum und Uhrzeit
{current_datetime}
{personality_section}
## Kontext
{context_section}
"""

REPLAN_PROMPT = """\
## Bisherige Ergebnisse

{results_section}

## Ziel: {original_goal}

Analysiere die bisherigen Ergebnisse kritisch:

**Qualitaetspruefung** (fuer Fakten/Recherche-Aufgaben):
- Hast du genug Quellen? (mindestens 2-3 verschiedene)
- Stimmen die Quellen ueberein? Gibt es Widersprueche?
- Fehlen wichtige Aspekte oder Perspektiven?
- Waere eine tiefere Recherche mit search_and_read oder deep_research sinnvoll?

Falls die Ergebnisse duenn, widerspruechlich oder unvollstaendig sind → erstelle einen \
neuen ```json Plan mit weiteren Recherche-Schritten. Nutze search_and_read (liest volle Seiten) \
oder deep_research (Multi-Quellen-Synthese mit Fakten-Konsens) fuer tiefere Analyse.

**Entscheidung:**

**Fehler fixen?** → Wenn ein Tool fehlgeschlagen ist (exit_code != 0, Error, Traceback), \
analysiere den Fehler und erstelle einen neuen ```json Plan der den Bug behebt. \
Bei Code-Aufgaben: Fehler lesen → Code korrigieren → erneut testen → wiederholen bis es laeuft. \
Gib NICHT auf wenn Code einen Fehler hat — fixe ihn autonom.

**Vertiefen?** → Wenn die bisherigen Ergebnisse nur oberflaechlich sind, erstelle einen \
Follow-Up-Plan mit deep_research oder weiteren search_and_read Aufrufen.

**Fertig?** → Antworte dem User direkt als Text. Nutze die erfolgreichen Ergebnisse (checkmark). \
Ignoriere fehlgeschlagene Schritte (x) wenn das Ziel trotzdem erreicht wurde. \
Gib keine Anleitungen fuer Dinge die du bereits erledigt hast.

**Noch nicht fertig?** → Erstelle einen neuen ```json Plan mit den fehlenden Schritten.

**Alles fehlgeschlagen?** → Analysiere den Fehler, probiere einen KOMPLETT anderen Ansatz. \
Nutze keine externen Programme die installiert werden muessen — verwende nur Python-Bibliotheken. \
Wenn ein Ansatz nicht funktioniert, wechsle die Strategie radikal. Gib NICHT auf — \
iteriere weiter bis die Aufgabe funktioniert. Der User will KEIN manuelles Eingreifen.

Suchergebnisse aus dem Web sind Fakten -- vertraue ihnen, auch wenn sie deinem \
Vorwissen widersprechen. Zitiere konkrete Daten direkt aus den Ergebnissen.

Waehle EINE Option: Text ODER JSON-Plan. Nie beides mischen.
"""

ESCALATION_PROMPT = """\
Ich wollte "{tool}" ausfuehren, aber der Sicherheitscheck hat das blockiert.
Grund: {reason}

Erklaere dem User in 2-3 Saetzen was passiert ist und was er tun kann. \
Locker, verstaendlich, keine technischen Details.
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

        # Context-window for Ollama num_ctx (default from model config)
        try:
            cw = self._config.models.planner.context_window
            self._context_window: int = cw if isinstance(cw, int) else 32768
        except Exception:
            self._context_window = 32768

        # Circuit breaker for LLM calls (#42 optimization)
        from jarvis.utils.circuit_breaker import CircuitBreaker

        self._llm_circuit_breaker = CircuitBreaker(
            name="planner_llm",
            failure_threshold=5,
            recovery_timeout=15.0,
            half_open_max_calls=2,
        )

        # Prompts von Disk laden (mit Fallback auf hardcoded Konstanten)
        self._system_prompt_template = self._load_prompt_from_file(
            "SYSTEM_PROMPT.md", SYSTEM_PROMPT, preset_key="plannerSystem"
        )
        self._replan_prompt_template = self._load_prompt_from_file(
            "REPLAN_PROMPT.md",
            REPLAN_PROMPT,
            fallback_txt="REPLAN_PROMPT.txt",
            preset_key="replanPrompt",
        )
        self._escalation_prompt_template = self._load_prompt_from_file(
            "ESCALATION_PROMPT.md",
            ESCALATION_PROMPT,
            fallback_txt="ESCALATION_PROMPT.txt",
            preset_key="escalationPrompt",
        )

    def _load_prompt_from_file(
        self, filename: str, fallback: str, fallback_txt: str = "", preset_key: str = ""
    ) -> str:
        """Lädt einen Prompt von Disk (.md bevorzugt, .txt als Migration-Fallback).

        Priorität: Disk .md → Disk .txt → i18n Preset → Hardcoded Fallback.
        """
        try:
            prompts_dir = self._config.jarvis_home / "prompts"
            path = prompts_dir / filename
            if path.exists():
                content = path.read_text(encoding="utf-8").strip()
                if content and isinstance(content, str):
                    return content
            # Migration: check old .txt file as fallback
            if fallback_txt:
                txt_path = prompts_dir / fallback_txt
                if txt_path.exists():
                    content = txt_path.read_text(encoding="utf-8").strip()
                    if content and isinstance(content, str):
                        return content
        except Exception:
            pass
        # i18n Preset Fallback (curated translations)
        if preset_key:
            try:
                from jarvis.i18n.prompt_presets import get_preset

                preset = get_preset(getattr(self._config, "language", "de"))
                if preset and preset_key in preset:
                    return preset[preset_key]
            except Exception:
                pass
        return fallback

    def reload_prompts(self) -> None:
        """Lädt alle Prompt-Templates neu von Disk."""
        self._system_prompt_template = self._load_prompt_from_file(
            "SYSTEM_PROMPT.md", SYSTEM_PROMPT, preset_key="plannerSystem"
        )
        self._replan_prompt_template = self._load_prompt_from_file(
            "REPLAN_PROMPT.md",
            REPLAN_PROMPT,
            fallback_txt="REPLAN_PROMPT.txt",
            preset_key="replanPrompt",
        )
        self._escalation_prompt_template = self._load_prompt_from_file(
            "ESCALATION_PROMPT.md",
            ESCALATION_PROMPT,
            fallback_txt="ESCALATION_PROMPT.txt",
            preset_key="escalationPrompt",
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
        *,
        model_override: str | None = None,
        temperature_override: float | None = None,
        top_p_override: float | None = None,
    ) -> ActionPlan:
        """Erstellt einen Plan für eine User-Nachricht.

        Args:
            user_message: Die Nachricht des Users
            working_memory: Aktiver Session-Kontext (Memory, History)
            tool_schemas: Verfügbare Tools als JSON-Schema

        Returns:
            ActionPlan mit Schritten oder einer direkten Antwort.
        """
        if model_override:
            model = model_override
        else:
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
                    temperature=temperature_override
                    if temperature_override is not None
                    else model_config.get("temperature", 0.7),
                    top_p=top_p_override
                    if top_p_override is not None
                    else model_config.get("top_p", 0.9),
                    options=self._build_llm_options(),
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

        # Check if the response contains tool calls (Ollama native)
        tool_calls = response.get("message", {}).get("tool_calls", [])
        if tool_calls:
            return self._parse_tool_calls(tool_calls, user_message)

        # Check if JSON plan is embedded in the response
        plan = self._extract_plan(assistant_text, user_message)

        # Retry once if JSON parsing failed (LLM produced malformed JSON)
        if plan.parse_failed:
            log.warning("planner_json_retry", model=model, goal=user_message[:80])
            retry_hint = (
                "\n\nIMPORTANT: Your previous response contained "
                "malformed JSON that could not be parsed. "
                "Please respond with VALID JSON only. "
                "Use this exact format:\n"
                '```json\n{"goal": "...", "steps": '
                '[{"tool": "...", "args": {...}, '
                '"purpose": "..."}]}\n```'
            )
            retry_messages = list(messages)
            retry_messages.append({"role": "assistant", "content": assistant_text})
            retry_messages.append({"role": "user", "content": retry_hint})
            try:
                retry_response = await self._ollama.chat(
                    model=model,
                    messages=retry_messages,
                    temperature=temperature_override
                    if temperature_override is not None
                    else max(0.3, model_config.get("temperature", 0.7) - 0.3),
                    top_p=top_p_override
                    if top_p_override is not None
                    else model_config.get("top_p", 0.9),
                    options=self._build_llm_options(),
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
        *,
        model_override: str | None = None,
        temperature_override: float | None = None,
        top_p_override: float | None = None,
    ) -> ActionPlan:
        """Erstellt einen neuen Plan basierend auf bisherigen Ergebnissen. [B§3.4]

        Wird aufgerufen wenn der Agent-Loop weitere Iterationen braucht.
        """
        if model_override:
            model = model_override
        else:
            model = self._router.select_model("planning", "high")
        model_config = self._router.get_model_config(model)

        # Format results
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
                    temperature=temperature_override
                    if temperature_override is not None
                    else model_config.get("temperature", 0.7),
                    top_p=top_p_override
                    if top_p_override is not None
                    else model_config.get("top_p", 0.9),
                    options=self._build_llm_options(),
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

        # Check if tool calls are in the response
        tool_calls = response.get("message", {}).get("tool_calls", [])
        if tool_calls:
            return self._parse_tool_calls(tool_calls, original_goal)

        plan = self._extract_plan(assistant_text, original_goal)

        # Retry once if JSON parsing failed during replan
        if plan.parse_failed:
            log.warning("planner_replan_json_retry", model=model)
            retry_hint = (
                "\n\nIMPORTANT: Your previous response contained "
                "malformed JSON. "
                "Please respond with VALID JSON only. "
                "Use this exact format:\n"
                '```json\n{"goal": "...", "steps": '
                '[{"tool": "...", "args": {...}, '
                '"purpose": "..."}]}\n```'
            )
            retry_messages = list(messages)
            retry_messages.append({"role": "assistant", "content": assistant_text})
            retry_messages.append({"role": "user", "content": retry_hint})
            try:
                retry_response = await self._ollama.chat(
                    model=model,
                    messages=retry_messages,
                    temperature=temperature_override
                    if temperature_override is not None
                    else max(0.3, model_config.get("temperature", 0.7) - 0.3),
                    top_p=top_p_override
                    if top_p_override is not None
                    else model_config.get("top_p", 0.9),
                    options=self._build_llm_options(),
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
                options=self._build_llm_options(),
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
        messages = self._build_formulate_messages(user_message, results, working_memory)

        for _fmt_attempt in range(2):
            try:
                response = await self._ollama.chat(
                    model=model, messages=messages, options=self._build_llm_options()
                )
                self._record_cost(response, model, session_id=working_memory.session_id)
                content: str = response.get("message", {}).get("content", "")

                # Four Questions response validation (advisory, non-blocking)
                try:
                    from jarvis.core.response_validator import ResponseValidator

                    _validator = ResponseValidator()
                    _val_result = _validator.validate(content, user_message, results)
                    if not _val_result.passed:
                        log.info(
                            "response_validation_warn",
                            score=_val_result.score,
                            issues=_val_result.issues,
                            consistency=_val_result.consistency_score,
                            coverage=_val_result.coverage_score,
                            assumptions=_val_result.assumption_score,
                            evidence=_val_result.evidence_score,
                        )
                    else:
                        log.debug(
                            "response_validation_ok",
                            score=_val_result.score,
                        )
                except Exception:
                    log.debug("response_validation_skipped", exc_info=True)

                return content
            except OllamaError as exc:
                log.warning(
                    "formulate_response_llm_error", error=str(exc), attempt=_fmt_attempt + 1
                )
                if _fmt_attempt == 0:
                    await asyncio.sleep(1.0)  # Retry nach kurzer Pause
                    continue
                # Second failure: return results directly as fallback
                raw_results = "\n".join(
                    f"[{r.tool_name}] {r.content[:300]}" for r in results if r.success
                )
                if raw_results:
                    return (
                        "Hier sind die Ergebnisse "
                        "(Zusammenfassung fehlgeschlagen):"
                        f"\n\n{raw_results}"
                    )
                return (
                    "Ich konnte die Ergebnisse leider nicht zusammenfassen — "
                    "das Sprachmodell antwortet fehlerhaft. "
                    "Prüfe, ob Ollama läuft und genug Ressourcen hat."
                )
        return ""  # Unreachable, aber fuer Type-Checker

    async def formulate_response_stream(
        self,
        user_message: str,
        results: list[ToolResult],
        working_memory: WorkingMemory,
        stream_callback: StreamCallback,
    ) -> str:
        """Streaming-Variante von formulate_response().

        Sendet Tokens via stream_callback an den Client, waehrend die
        Antwort generiert wird. Gibt den vollstaendigen Text zurueck.

        Args:
            user_message: Urspruengliche User-Nachricht.
            results: Tool-Ergebnisse aus dem PGE-Zyklus.
            working_memory: Aktueller Kontext.
            stream_callback: Async callback fuer stream_token Events.

        Returns:
            Vollstaendiger Antwort-Text.
        """
        # Pruefe ob chat_stream verfuegbar ist
        if not hasattr(self._ollama, "chat_stream"):
            # Fallback: nicht-streamende Variante
            return await self.formulate_response(user_message, results, working_memory)

        model = self._router.select_model("summarization", "medium")
        messages = self._build_formulate_messages(user_message, results, working_memory)

        try:
            full_text = ""
            async for chunk in self._ollama.chat_stream(
                model=model,
                messages=messages,
                temperature=0.7,
                top_p=0.9,
            ):
                token = chunk.get("message", {}).get("content", "")
                if token and not chunk.get("done", False):
                    full_text += token
                    try:
                        await stream_callback("stream_token", {"token": token})
                    except Exception:
                        log.debug("stream_callback_failed", exc_info=True)

            # Post-processing: Personality, /think-Block entfernen, etc.
            content = full_text.strip()
            if not content:
                # Streaming lieferte keinen Text — Fallback
                return await self.formulate_response(user_message, results, working_memory)

            # /think-Bloecke entfernen (Qwen3)
            content = re.sub(
                r"<think>.*?</think>",
                "",
                content,
                flags=re.DOTALL,
            ).strip()

            # Personality-Engine Postprocessing (same as formulate_response)
            if self._personality_engine is not None:
                try:
                    content = self._personality_engine.enhance_response(
                        content,
                        context={"user_message": user_message},
                    )
                except Exception:
                    log.debug("personality_enhance_failed", exc_info=True)

            return content

        except OllamaError as exc:
            log.warning("formulate_stream_error", error=str(exc))
            # Fallback: nicht-streamende Variante
            return await self.formulate_response(user_message, results, working_memory)

    def _build_formulate_messages(
        self,
        user_message: str,
        results: list[ToolResult],
        working_memory: WorkingMemory,
    ) -> list[dict[str, Any]]:
        """Baut die Messages fuer formulate_response (shared by stream/non-stream)."""
        results_text = self._format_results(results)

        has_search_results = any(
            r.tool_name in ("web_search", "web_news_search", "search_and_read", "web_fetch")
            and r.success
            for r in results
        )

        if not results:
            prompt = (
                f"Der User hat gesagt: {user_message}\n\n"
                f"Beantworte die Nachricht des Users direkt und hilfreich "
                f"auf Deutsch in natuerlicher, gesprochener Sprache.\n"
                f"Generiere KEINE Planungs-Metaebene, KEIN REPLAN-Format. "
                f"Antworte wie ein Mensch im Gespraech."
            )
        elif has_search_results:
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
                f"Beantworte die Frage des Users AUSSCHLIEẞLICH "
                f"auf Basis der obigen Suchergebnisse.\n"
                f"REGELN:\n"
                f"1. Die Suchergebnisse sind AKTUELL und KORREKT. "
                f"Dein Trainingswissen ist VERALTET.\n"
                f"2. Wenn die Suchergebnisse ein Ereignis beschreiben, "
                f"dann IST es passiert.\n"
                f"3. Sage NIEMALS 'es gibt keinen Beleg' oder "
                f"'das ist nicht passiert', wenn die "
                f"Suchergebnisse das Gegenteil zeigen.\n"
                f"4. Zitiere konkrete Daten, Namen, Orte und Fakten "
                f"DIREKT aus den Suchergebnissen.\n"
                f"5. Erfinde KEINE Details, die nicht in den "
                f"Suchergebnissen stehen.\n"
                f"6. Antworte auf Deutsch, praegnant und faktenbasiert.\n"
                f"7. Antworte in natuerlicher, gesprochener Sprache "
                f"-- wie ein Mensch im Gespraech. "
                f"Keine Bullet-Points oder Listen, sondern "
                f"fliessende Saetze."
            )
        else:
            prompt = (
                f"Der User hat gefragt: {user_message}\n\n"
                f"Du hast folgende Aktionen ausgefuehrt und "
                f"Ergebnisse erhalten:\n\n"
                f"{results_text}\n\n"
                f"Formuliere jetzt eine hilfreiche Antwort auf Deutsch "
                f"in natuerlicher, gesprochener Sprache.\n"
                f"WICHTIG: Nutze die ERFOLGREICHEN Ergebnisse "
                f"direkt in deiner Antwort. "
                f"Ignoriere fehlgeschlagene/blockierte Schritte, "
                f"wenn das Ziel trotzdem erreicht wurde. "
                f"Gib dem User KEINE Anleitungen fuer Dinge, "
                f"die du bereits erledigt hast. "
                f"Antworte wie ein Mensch im Gespraech -- "
                f"fliessende Saetze, keine Bullet-Points."
            )

        from datetime import datetime

        now = datetime.now()
        current_dt = now.strftime("%A, %d. %B %Y, %H:%M Uhr")
        date_line = f"\nHeutiges Datum und Uhrzeit: {current_dt}\n"

        if has_search_results:
            system_content = (
                "Du bist Jarvis, ein autonomer Agent. Du beantwortest "
                "Fragen auf Deutsch "
                "in natuerlicher, gesprochener Sprache -- wie ein "
                "Mensch im Gespraech.\n"
                f"{date_line}"
                "KRITISCHE REGEL: Dein Trainingswissen ist VERALTET. "
                "Bei Suchergebnissen aus dem Internet basiert deine "
                "Antwort AUSSCHLIEẞLICH "
                "auf den gefundenen Informationen. Die Suchergebnisse "
                "sind die WAHRHEIT. "
                "Widerspricht dein Vorwissen den Suchergebnissen, sind "
                "die Suchergebnisse KORREKT. "
                "Du darfst Suchergebnisse NICHT als 'fiktiv', "
                "'hypothetisch' oder 'unbelegte "
                "Behauptung' bezeichnen."
            )
        else:
            system_content = (
                "Du bist Jarvis, ein autonomer Agent. Antworte "
                "hilfreich auf Deutsch "
                "in natuerlicher, gesprochener Sprache -- wie ein "
                "Mensch im Gespraech.\n"
                f"{date_line}"
                "Du nutzt Tool-Ergebnisse direkt und gibst dem User "
                "NICHT Anleitungen, "
                "Dinge selbst zu tun. Du loest Probleme eigenstaendig."
            )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_content},
        ]

        if working_memory.core_memory_text:
            messages.append(
                {
                    "role": "system",
                    "content": f"Dein Hintergrund:\n{working_memory.core_memory_text[:500]}",
                }
            )

        messages.append({"role": "user", "content": prompt})
        return messages

    # =========================================================================
    # Private Methoden
    # =========================================================================

    def _build_llm_options(self) -> dict[str, Any]:
        """Build Ollama ``options`` dict including ``num_ctx`` and ``num_predict``."""
        return {
            "num_ctx": self._context_window,
            "num_predict": getattr(self._config.planner, "response_token_budget", 3000),
        }

    def _build_system_prompt(
        self,
        working_memory: WorkingMemory,
        tool_schemas: dict[str, Any],
    ) -> str:
        """Baut den System-Prompt mit Kontext und Tools."""
        # Tools-Section (gecacht, #40 Optimierung)
        # Compact mode for small context windows (≤16K tokens)
        compact = self._context_window <= 16384
        if tool_schemas:
            cache_key = (hash(frozenset(tool_schemas.keys())), compact)
            if cache_key != self._cached_tools_hash or self._cached_tools_section is None:
                if compact:
                    # Compact: only tool names + short descriptions, no params/examples
                    tools_lines = []
                    for name, schema in tool_schemas.items():
                        desc = schema.get("description", "")
                        # Truncate description to ~60 chars
                        short = desc[:60].rstrip() + ("..." if len(desc) > 60 else "")
                        tools_lines.append(f"- `{name}`: {short}")
                    self._cached_tools_section = "\n".join(tools_lines)
                else:
                    # Full mode: prefer ToolRegistryDB for localized descriptions with examples
                    db_section = None
                    try:
                        from jarvis.mcp.tool_registry_db import ToolRegistryDB

                        db_path = self._config.jarvis_home / "tool_registry.db"
                        if db_path.exists():
                            registry_db = ToolRegistryDB(db_path)
                            language = getattr(self._config, "language", "de")
                            db_section = registry_db.get_tool_prompt_section("planner", language)
                            registry_db.close()
                    except Exception:
                        pass

                    if db_section:
                        self._cached_tools_section = db_section
                    else:
                        # Fallback: hand-rolled schema parsing
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
                self._cached_tools_hash = cache_key
            tools_section = self._cached_tools_section
        else:
            tools_section = "No tools available."

        # Context-Section (Memory) — Relevanz-Ranking nach Score (#41 Optimierung)
        # Budget scales with context window (compact mode for small models)
        context_parts: list[str] = []
        core_budget = 2000 if compact else 0  # 0 = unlimited
        mem_budget = 600 if compact else 1200  # ~150 / ~300 Tokens
        proc_budget = 1500 if compact else 3000
        proc_skill_budget = 300 if compact else 600

        if working_memory.core_memory_text:
            core_text = working_memory.core_memory_text
            if core_budget and len(core_text) > core_budget:
                core_text = core_text[:core_budget] + "\n[...]"
            context_parts.append(f"### Kern-Wissen\n{core_text}")

        if working_memory.injected_memories:
            # Sort by score (highest first) with token budget
            sorted_mems = sorted(
                working_memory.injected_memories,
                key=lambda m: getattr(m, "score", 0.0),
                reverse=True,
            )
            mem_texts = []
            used = 0
            for mem in sorted_mems:
                line = f"- [{mem.chunk.memory_tier.value}] {mem.chunk.text[:200]}"
                if used + len(line) > mem_budget:
                    break
                mem_texts.append(line)
                used += len(line)
            context_parts.append("### Relevantes Wissen\n" + "\n".join(mem_texts))

        if working_memory.injected_procedures:
            for proc in working_memory.injected_procedures[:2]:
                if "Web-Suchergebnis" in proc:
                    # Presearch: web results with matching heading and more space
                    context_parts.append(
                        f"### AKTUELLE FAKTEN AUS DEM INTERNET (vertraue diesen Daten!)\n"
                        f"{proc[:proc_budget]}"
                    )
                else:
                    context_parts.append(
                        f"### Relevante Prozedur (folge diesem Ablauf!)\n{proc[:proc_skill_budget]}"
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
            with contextlib.suppress(Exception):
                personality_section = self._personality_engine.build_personality_block()

        # Cognitive identity injection
        identity_section = ""
        if hasattr(self, "_identity_layer") and self._identity_layer is not None:
            try:
                _id_ctx = self._identity_layer.enrich_context(
                    user_message="",  # Already enriched in gateway, just get state
                )
                identity_section = _id_ctx.get("cognitive_context", "")
            except Exception:
                pass

        if identity_section:
            context_section += f"\n\n### Kognitive Identitaet\n{identity_section}"

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
                    workspace_dir=str(self._config.jarvis_home / "workspace"),
                    personality_section=personality_section,
                )
            except Exception:
                pass  # Fallback auf Standard-Template

        return self._system_prompt_template.format(
            tools_section=tools_section,
            context_section=context_section,
            current_datetime=current_datetime,
            owner_name=self._config.owner_name,
            workspace_dir=str(self._config.jarvis_home / "workspace"),
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

        # Insert chat history (newest first, until budget exhausted)
        for msg in working_memory.chat_history:
            role = msg.role.value
            content = msg.content

            # Render TOOL messages as assistant with prefix,
            # since not all LLM backends support the "tool" role
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
        # Replace backslashes that do not introduce a valid JSON escape
        # Negative lookahead: backslash followed by something that is NOT a valid escape
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

        # 2. Fix invalid escapes
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
        # Try to find JSON block (```json ... ```)
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

        # Try to parse raw JSON (without code block)
        # Find first { and last }
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
        # Tightened check: einzelne { in Freitext (z.B. "Python {dict}")
        # are NOT an indicator of broken JSON. We check for
        # kombinierte Signale: { + JSON-Keys, oder ``` + json-Marker.
        _has_plan_keys = '"steps"' in text or '"goal"' in text
        _has_json_block = json_match is not None
        _has_braces_with_keys = (
            first_brace is not None
            and first_brace >= 0
            and last_brace > first_brace
            and _has_plan_keys
        )
        _has_json_markers = _has_json_block or _has_braces_with_keys or _has_plan_keys

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

        # No JSON found -- check if the planner is asking for permission
        _permission_keywords = [
            "permission",
            "berechtigung",
            "freigabe",
            "erlaubnis",
            "genehmig",
            "allow",
            "approve",
            "write-permission",
            "schreibberechtigung",
        ]
        _lower = text.lower()
        _is_permission_ask = any(kw in _lower for kw in _permission_keywords)

        if _is_permission_ask:
            # Check if there's actual code in the response we can extract
            import re as _re

            _code_blocks = _re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, _re.DOTALL)
            if _code_blocks:
                # Claude wrote the code but wrapped it in permission text.
                # Extract the code and auto-create a write_file plan.
                _code = max(_code_blocks, key=len)  # Take the largest code block
                # Try to find a filename in the text
                _fname_match = _re.search(r"[`'\"](\w+\.py)[`'\"]", text)
                _filename = _fname_match.group(1) if _fname_match else "program.py"

                log.info(
                    "planner_permission_auto_plan",
                    filename=_filename,
                    code_len=len(_code),
                )
                return ActionPlan(
                    goal=goal,
                    reasoning=(
                        f"Auto-extracted code from permission response → write_file({_filename})"
                    ),
                    steps=[
                        PlannedAction(
                            tool="write_file",
                            params={
                                "path": f"{{workspace_dir}}/{_filename}",
                                "content": _code.strip(),
                            },
                            rationale=f"Code schreiben: {_filename}",
                        ),
                        PlannedAction(
                            tool="run_python",
                            params={"code": f"print('File {_filename} written successfully')"},
                            rationale="Bestätigung",
                        ),
                    ],
                    confidence=0.8,
                )

            # No code blocks found — try to find a filename and create
            # a run_python plan that generates the code
            _fname_match = _re.search(r"[`'\"](\w+\.py)[`'\"]", text)
            _filename = _fname_match.group(1) if _fname_match else None

            # Permission text without code — Claude described what it would
            # build but didn't include the code. Pass through as direct response
            # so user sees the description, and let the replan handle next steps.
            log.warning(
                "planner_permission_ask_passthrough",
                text_preview=text[:200],
            )
            # Strip permission-related sentences but keep the rest
            import re as _re2

            _cleaned = _re2.sub(
                r"(?i)(ich brauche|bitte (erlaube|genehmige|klick)|"
                r"sobald du die genehmigst|write-?permission|"
                r"schreibberechtigung|erlaubnis für).*?[.\n]",
                "",
                text,
            ).strip()
            return ActionPlan(
                goal=goal,
                reasoning="Direkte Antwort (Code-Beschreibung ohne ausfuehrbaren Code)",
                direct_response=_cleaned or text.strip(),
                confidence=0.5,
            )

        # Echte direkte Antwort (z.B. "Was ist 2+2?" → "4")
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

    # Tools whose results need more context (larger content limit)
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
