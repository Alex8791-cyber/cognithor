"""MCP Prompts: Exponiert Jarvis-Prompt-Templates als MCP-Prompts.

MCP-Prompts sind wiederverwendbare Prompt-Templates die externe
Clients nutzen können um strukturierte LLM-Anfragen zu stellen.

Registrierte Prompts:
  - analyze_document    → Dokument-Analyse mit Sprach- und Fokuswahl
  - summarize           → Text-Zusammenfassung mit konfigurierbarer Länge
  - insurance_advisor   → Versicherungsberatung (BU, bAV, etc.)
  - code_review         → Code-Review mit Sicherheitsfokus
  - translate           → Übersetzung mit Kontextbewahrung
  - brainstorm          → Strukturiertes Brainstorming
  - explain_concept     → Konzepterklärung für verschiedene Zielgruppen
  - daily_briefing      → Tägliches Briefing aus Memory

OPTIONAL: Wird nur registriert wenn MCP-Server-Modus aktiviert ist.

Bibel-Referenz: §5.5.3 (MCP Prompts)
"""

from __future__ import annotations

from typing import Any

from jarvis.mcp.server import (
    JarvisMCPServer,
    MCPPrompt,
    MCPPromptArgument,
)
from jarvis.utils.logging import get_logger

log = get_logger(__name__)


class JarvisPromptProvider:
    """Stellt Jarvis-Prompt-Templates als MCP-Prompts bereit."""

    def __init__(self) -> None:
        self._registered_count = 0

    def register_all(self, server: JarvisMCPServer) -> int:
        """Registriert alle Prompt-Templates beim MCP-Server.

        Returns:
            Anzahl registrierter Prompts.
        """
        prompts = self._build_prompts()

        for prompt in prompts:
            server.register_prompt(prompt)

        self._registered_count = len(prompts)
        log.info("mcp_prompts_registered", count=len(prompts))
        return len(prompts)

    def _build_prompts(self) -> list[MCPPrompt]:
        """Erstellt alle Prompt-Templates."""
        return [
            # ── Dokument-Analyse ────────────────────────────────
            MCPPrompt(
                name="analyze_document",
                description="Analysiert ein Dokument mit konfigurierbarem Fokus",
                arguments=[
                    MCPPromptArgument(
                        name="content",
                        description="Der zu analysierende Text",
                        required=True,
                    ),
                    MCPPromptArgument(
                        name="focus",
                        description="Analysefokus: summary, key_points, risks, actions, sentiment",
                        required=False,
                    ),
                    MCPPromptArgument(
                        name="language",
                        description="Ausgabesprache (de, en)",
                        required=False,
                    ),
                ],
                handler=self._prompt_analyze_document,
            ),

            # ── Zusammenfassung ─────────────────────────────────
            MCPPrompt(
                name="summarize",
                description="Erstellt eine strukturierte Zusammenfassung",
                arguments=[
                    MCPPromptArgument(
                        name="content",
                        description="Der zusammenzufassende Text",
                        required=True,
                    ),
                    MCPPromptArgument(
                        name="length",
                        description="Gewünschte Länge: short (2-3 Sätze), medium (1 Absatz), long (mehrere Absätze)",
                        required=False,
                    ),
                    MCPPromptArgument(
                        name="style",
                        description="Stil: professional, casual, technical, executive",
                        required=False,
                    ),
                ],
                handler=self._prompt_summarize,
            ),

            # ── Versicherungsberatung ───────────────────────────
            MCPPrompt(
                name="insurance_advisor",
                description="Versicherungsberatung für BU, bAV, Risikoleben und weitere Produkte",
                arguments=[
                    MCPPromptArgument(
                        name="product",
                        description="Produkttyp: bu, bav, rlv, pkv, sachversicherung",
                        required=True,
                    ),
                    MCPPromptArgument(
                        name="customer_info",
                        description="Kundeninformationen (Alter, Beruf, Einkommen etc.)",
                        required=False,
                    ),
                    MCPPromptArgument(
                        name="question",
                        description="Spezifische Frage des Kunden",
                        required=False,
                    ),
                ],
                handler=self._prompt_insurance_advisor,
            ),

            # ── Code-Review ─────────────────────────────────────
            MCPPrompt(
                name="code_review",
                description="Führt ein umfassendes Code-Review mit Sicherheitsfokus durch",
                arguments=[
                    MCPPromptArgument(
                        name="code",
                        description="Der zu reviewende Code",
                        required=True,
                    ),
                    MCPPromptArgument(
                        name="language",
                        description="Programmiersprache (python, javascript, rust, etc.)",
                        required=False,
                    ),
                    MCPPromptArgument(
                        name="focus",
                        description="Fokus: security, performance, readability, all",
                        required=False,
                    ),
                ],
                handler=self._prompt_code_review,
            ),

            # ── Übersetzung ─────────────────────────────────────
            MCPPrompt(
                name="translate",
                description="Kontextsensitive Übersetzung mit Fachbegriff-Bewahrung",
                arguments=[
                    MCPPromptArgument(
                        name="content",
                        description="Der zu übersetzende Text",
                        required=True,
                    ),
                    MCPPromptArgument(
                        name="target_language",
                        description="Zielsprache (de, en, fr, es, etc.)",
                        required=True,
                    ),
                    MCPPromptArgument(
                        name="domain",
                        description="Fachgebiet: insurance, legal, technical, medical, general",
                        required=False,
                    ),
                ],
                handler=self._prompt_translate,
            ),

            # ── Brainstorming ───────────────────────────────────
            MCPPrompt(
                name="brainstorm",
                description="Strukturiertes Brainstorming mit verschiedenen Methoden",
                arguments=[
                    MCPPromptArgument(
                        name="topic",
                        description="Das Thema für das Brainstorming",
                        required=True,
                    ),
                    MCPPromptArgument(
                        name="method",
                        description="Methode: freeform, six_hats, scamper, swot, mind_map",
                        required=False,
                    ),
                    MCPPromptArgument(
                        name="constraints",
                        description="Einschränkungen oder Rahmenbedingungen",
                        required=False,
                    ),
                ],
                handler=self._prompt_brainstorm,
            ),

            # ── Konzepterklärung ────────────────────────────────
            MCPPrompt(
                name="explain_concept",
                description="Erklärt ein Konzept für verschiedene Zielgruppen",
                arguments=[
                    MCPPromptArgument(
                        name="concept",
                        description="Das zu erklärende Konzept",
                        required=True,
                    ),
                    MCPPromptArgument(
                        name="audience",
                        description="Zielgruppe: beginner, intermediate, expert, child, executive",
                        required=False,
                    ),
                    MCPPromptArgument(
                        name="use_analogies",
                        description="Analogien verwenden: true/false",
                        required=False,
                    ),
                ],
                handler=self._prompt_explain_concept,
            ),

            # ── Tägliches Briefing ──────────────────────────────
            MCPPrompt(
                name="daily_briefing",
                description="Erstellt ein tägliches Briefing aus Memory-Daten und Kontext",
                arguments=[
                    MCPPromptArgument(
                        name="focus_areas",
                        description="Fokusbereiche: tasks, meetings, insights, all",
                        required=False,
                    ),
                    MCPPromptArgument(
                        name="date",
                        description="Datum für das Briefing (YYYY-MM-DD)",
                        required=False,
                    ),
                ],
                handler=self._prompt_daily_briefing,
            ),
        ]

    # ── Prompt Handlers ──────────────────────────────────────────

    def _prompt_analyze_document(
        self,
        content: str = "",
        focus: str = "key_points",
        language: str = "de",
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Generiert Messages für Dokument-Analyse."""
        lang_instruction = "Antworte auf Deutsch." if language == "de" else f"Respond in {language}."
        focus_map = {
            "summary": "Erstelle eine prägnante Zusammenfassung.",
            "key_points": "Extrahiere die wichtigsten Kernaussagen als strukturierte Liste.",
            "risks": "Identifiziere Risiken, Probleme und offene Punkte.",
            "actions": "Extrahiere konkrete Handlungsempfehlungen und nächste Schritte.",
            "sentiment": "Analysiere den Ton, die Stimmung und die Haltung des Textes.",
        }
        focus_text = focus_map.get(focus, focus_map["key_points"])

        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"{lang_instruction}\n\n{focus_text}\n\n---\n\n{content}",
                },
            }
        ]

    def _prompt_summarize(
        self,
        content: str = "",
        length: str = "medium",
        style: str = "professional",
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Generiert Messages für Zusammenfassung."""
        length_map = {
            "short": "in 2-3 Sätzen",
            "medium": "in einem Absatz (ca. 100 Wörter)",
            "long": "in mehreren Absätzen (ca. 300 Wörter)",
        }
        style_map = {
            "professional": "professionell und sachlich",
            "casual": "locker und verständlich",
            "technical": "technisch präzise mit Fachbegriffen",
            "executive": "knapp und entscheidungsorientiert",
        }

        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": (
                        f"Fasse den folgenden Text zusammen -- "
                        f"{length_map.get(length, length_map['medium'])}, "
                        f"{style_map.get(style, style_map['professional'])}.\n\n"
                        f"---\n\n{content}"
                    ),
                },
            }
        ]

    def _prompt_insurance_advisor(
        self,
        product: str = "bu",
        customer_info: str = "",
        question: str = "",
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Generiert Messages für Versicherungsberatung."""
        product_context = {
            "bu": "Berufsunfähigkeitsversicherung (BU) -- Absicherung der Arbeitskraft",
            "bav": "Betriebliche Altersversorgung (bAV) -- Arbeitgeberfinanzierte Vorsorge",
            "rlv": "Risikolebensversicherung (RLV) -- Hinterbliebenenabsicherung",
            "pkv": "Private Krankenversicherung (PKV) -- Gesundheitsschutz",
            "sachversicherung": "Sachversicherungen -- Haftpflicht, Hausrat, Wohngebäude",
        }

        system_msg = (
            "Du bist ein erfahrener Versicherungsberater (IHK-zertifiziert) "
            "mit Spezialisierung auf den deutschen Markt. "
            "Beantworte Fragen sachlich, präzise und kundenfreundlich. "
            "Nenne relevante steuerliche Aspekte und gesetzliche Regelungen. "
            "Weise darauf hin, dass eine individuelle Beratung empfohlen wird."
        )

        user_text = f"Produktbereich: {product_context.get(product, product)}\n"
        if customer_info:
            user_text += f"\nKundeninformationen: {customer_info}\n"
        if question:
            user_text += f"\nFrage: {question}"
        else:
            user_text += "\nGib eine allgemeine Übersicht und typische Beratungspunkte."

        return [
            {"role": "user", "content": {"type": "text", "text": f"{system_msg}\n\n{user_text}"}},
        ]

    def _prompt_code_review(
        self,
        code: str = "",
        language: str = "python",
        focus: str = "all",
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Generiert Messages für Code-Review."""
        focus_instructions = {
            "security": "Fokussiere auf Sicherheitslücken: Injection, Path Traversal, XSS, Auth-Probleme, unsichere Deserialisierung.",
            "performance": "Fokussiere auf Performance: N+1 Queries, unnötige Allokationen, algorithmische Komplexität, Caching-Möglichkeiten.",
            "readability": "Fokussiere auf Lesbarkeit: Naming, Struktur, Kommentare, Single Responsibility, DRY.",
            "all": "Prüfe Sicherheit, Performance, Lesbarkeit und Best Practices.",
        }

        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": (
                        f"Code-Review ({language}).\n"
                        f"{focus_instructions.get(focus, focus_instructions['all'])}\n\n"
                        f"```{language}\n{code}\n```"
                    ),
                },
            }
        ]

    def _prompt_translate(
        self,
        content: str = "",
        target_language: str = "en",
        domain: str = "general",
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Generiert Messages für Übersetzung."""
        domain_hint = {
            "insurance": "Behalte Versicherungsfachbegriffe bei oder übersetze sie korrekt.",
            "legal": "Behalte juristische Fachbegriffe bei. Beachte länderspezifische Rechtsbegriffe.",
            "technical": "Technische Begriffe präzise übersetzen, Akronyme beibehalten.",
            "medical": "Medizinische Terminologie korrekt übersetzen.",
            "general": "",
        }

        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": (
                        f"Übersetze den folgenden Text nach {target_language}. "
                        f"{domain_hint.get(domain, '')} "
                        f"Bewahre den Ton und die Struktur des Originals.\n\n"
                        f"---\n\n{content}"
                    ),
                },
            }
        ]

    def _prompt_brainstorm(
        self,
        topic: str = "",
        method: str = "freeform",
        constraints: str = "",
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Generiert Messages für Brainstorming."""
        method_instructions = {
            "freeform": "Generiere möglichst viele kreative Ideen ohne Einschränkungen.",
            "six_hats": "Nutze die 6-Hüte-Methode (de Bono): Weiß (Fakten), Rot (Gefühle), Schwarz (Risiken), Gelb (Vorteile), Grün (Kreativ), Blau (Prozess).",
            "scamper": "Nutze SCAMPER: Substitute, Combine, Adapt, Modify, Put to other uses, Eliminate, Reverse.",
            "swot": "Führe eine SWOT-Analyse durch: Strengths, Weaknesses, Opportunities, Threats.",
            "mind_map": "Erstelle eine Mind-Map-Struktur mit Haupt- und Unterthemen.",
        }

        text = f"Brainstorming zum Thema: {topic}\n\n"
        text += method_instructions.get(method, method_instructions["freeform"])
        if constraints:
            text += f"\n\nRahmenbedingungen: {constraints}"

        return [
            {"role": "user", "content": {"type": "text", "text": text}},
        ]

    def _prompt_explain_concept(
        self,
        concept: str = "",
        audience: str = "beginner",
        use_analogies: str = "true",
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Generiert Messages für Konzepterklärung."""
        audience_style = {
            "beginner": "Erkläre einfach und verständlich, vermeide Fachbegriffe.",
            "intermediate": "Nutze Fachbegriffe, aber erkläre sie kurz.",
            "expert": "Nutze Fachsprache, gehe in die Tiefe, referenziere aktuelle Forschung.",
            "child": "Erkläre so, dass ein 10-Jähriger es versteht. Nutze einfache Wörter und Bilder.",
            "executive": "Erkläre knapp und geschäftsorientiert. Fokus auf Relevanz und Impact.",
        }

        text = f"Erkläre das Konzept: {concept}\n\n"
        text += audience_style.get(audience, audience_style["beginner"])
        if use_analogies.lower() == "true":
            text += " Nutze anschauliche Analogien und Beispiele."

        return [
            {"role": "user", "content": {"type": "text", "text": text}},
        ]

    def _prompt_daily_briefing(
        self,
        focus_areas: str = "all",
        date: str = "",
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Generiert Messages für tägliches Briefing."""
        from datetime import date as date_cls

        target_date = date or date_cls.today().isoformat()

        focus_text = {
            "tasks": "Fokussiere auf offene Aufgaben und Deadlines.",
            "meetings": "Fokussiere auf anstehende Termine und Besprechungen.",
            "insights": "Fokussiere auf neue Erkenntnisse und Lernfortschritte.",
            "all": "Gib einen vollständigen Überblick über Aufgaben, Termine und Erkenntnisse.",
        }

        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": (
                        f"Erstelle ein Tagesbriefing für {target_date}.\n\n"
                        f"{focus_text.get(focus_areas, focus_text['all'])}\n\n"
                        "Nutze die verfügbaren Memory-Daten und strukturiere "
                        "das Briefing übersichtlich mit Prioritäten."
                    ),
                },
            }
        ]

    def stats(self) -> dict[str, Any]:
        """Statistiken des Prompt-Providers."""
        return {
            "registered_prompts": self._registered_count,
        }
