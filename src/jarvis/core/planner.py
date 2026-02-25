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
Du bist Jarvis, ein lokales Agent-Betriebssystem.
Du bist der Planner -- du verstehst Anfragen und entscheidest, ob du direkt \
antworten oder einen Tool-Plan erstellen musst.

## Deine Rolle
- Du hast KEINEN direkten Zugriff auf Dateien, Shell oder Internet.
- Wenn du Dateien lesen/schreiben, Befehle ausführen oder im Wissen suchen musst, \
erstellst du einen Plan. Der Executor führt ihn aus.
- Du sprichst Deutsch. {owner_name} duzt dich.
- Denke Schritt für Schritt nach, bevor du antwortest.

## Verfügbare Tools
{tools_section}

## Antwort-Format

WICHTIG: Wähle GENAU EINE Option. Vermische NIEMALS Text und JSON.

### OPTION A -- Direkte Antwort
Für Wissensfragen, Erklärungen, Meinungen, Smalltalk, Nachfragen.
Antworte einfach als normaler Text. KEIN JSON, KEIN Code-Block.

### OPTION B -- Tool-Plan
Für alles was Dateien, Shell, Web oder Memory erfordert.
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

### Beispiel: User sagt „Was ist eine API?"
Direkte Textantwort (Option A): „Eine API ist eine Programmierschnittstelle..."

## Entscheidungshilfe

| Anfrage enthält... | Option | Typisches Tool |
|---------------------|--------|----------------|
| Wissensfrage, Erklärung, Meinung | A | -- |
| „Datei", „lesen", „erstellen", „schreiben" | B | read_file / write_file |
| „Verzeichnis", „Ordner", „auflisten" | B | list_directory |
| „Befehl", „ausführen", „Shell" | B | exec_command |
| „suchen", „googlen", „Web" | B | web_search |
| „erinnern", „Memory", „was weißt du über" | B | search_memory |
| „speichern", „merken" | B | save_to_memory |
| „Kontakt", „Entität" | B | get_entity / add_entity |
| „Prozedur", „wie mache ich" | B | search_procedures |
| Unklare Anfrage | A | -- (nachfragen) |

## Regeln
- Verwende NUR Tool-Namen aus der obigen Liste. Erfinde KEINE Tools.
- Jeder Step braucht „tool", „params" und „rationale".
- Bei mehreren Steps: Logische Reihenfolge. Ergebnisse fließen in Folgeschritte.
- confidence: 0.0--1.0. Unter 0.5 = besser nachfragen.
- Im Zweifel: OPTION A wählen und nachfragen.
- Antworte ENTWEDER als Text ODER als JSON-Plan. Niemals beides vermischen.
- Wenn dir eine Prozedur im Kontext angezeigt wird, folge deren Ablauf.

## Kontext
{context_section}
"""

REPLAN_PROMPT = """\
## Bisherige Ergebnisse

{results_section}

## Aufgabe
Ursprüngliches Ziel: {original_goal}

Analysiere die bisherigen Ergebnisse und entscheide dich für GENAU EINE Option:

**OPTION 1 -- Aufgabe erledigt** → Formuliere eine hilfreiche Antwort als normaler Text. \
KEIN JSON. Fasse die Ergebnisse zusammen und beantworte die ursprüngliche Frage. \
Nutze konkrete Daten aus den Ergebnissen.

**OPTION 2 -- Weitere Schritte nötig** → Erstelle einen neuen JSON-Plan (```json Block). \
Nutze die bisherigen Ergebnisse als Kontext. Plane nur die FEHLENDEN Schritte.

**OPTION 3 -- Fehler aufgetreten** → Analysiere die Ursache. Wenn ein anderer Ansatz \
möglich ist, erstelle einen neuen Plan. Wenn nicht, erkläre das Problem klar.

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
        prompt = (
            f"Der User hat gefragt: {user_message}\n\n"
            f"Du hast folgende Aktionen ausgeführt und Ergebnisse erhalten:\n\n"
            f"{results_text}\n\n"
            f"Formuliere jetzt eine hilfreiche Antwort auf Deutsch."
        )

        messages = [
            {"role": "system", "content": "Du bist Jarvis. Antworte hilfreich auf Deutsch."},
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

        return SYSTEM_PROMPT.format(
            tools_section=tools_section,
            context_section=context_section,
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

    def _format_results(self, results: list[ToolResult]) -> str:
        """Formatiert Tool-Ergebnisse als lesbaren Text."""
        if not results:
            return "Keine Ergebnisse."

        parts: list[str] = []
        for i, r in enumerate(results, 1):
            status = "✓" if r.success else "✗"
            content = r.content[:1000]  # Maximal 1000 Zeichen pro Ergebnis
            if r.truncated:
                content += "\n[... Output gekürzt]"
            parts.append(f"### Schritt {i}: {r.tool_name} [{status}]\n{content}")

        return "\n\n".join(parts)
