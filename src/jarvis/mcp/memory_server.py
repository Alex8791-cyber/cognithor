"""Memory-Tools für Jarvis · MCP-Server für das 5-Tier Memory-System.

Exponiert das Cognitive Memory-System als Tool-Calls, damit der Planner
über den normalen MCP-Kanal auf alle Memory-Tiers zugreifen kann.

Tools:
  - search_memory: Hybrid-Suche über alle Memory-Tiers (BM25 + Vektor + Graph)
  - save_to_memory: Speichert Information in den passenden Tier
  - get_entity: Lädt eine Entität mit Relationen aus dem Wissens-Graphen
  - add_entity: Erstellt eine neue Entität im Wissens-Graphen
  - add_relation: Erstellt eine Relation zwischen zwei Entitäten
  - get_core_memory: Gibt die aktuelle Core Memory (CORE.md) zurück
  - get_recent_episodes: Lädt die letzten Tageslog-Einträge
  - search_procedures: Sucht nach gelernten Prozeduren/Skills
  - record_procedure_usage: Meldet Erfolg/Misserfolg einer Prozedur
  - memory_stats: Gibt Gesamtstatistiken des Memory-Systems zurück

Bibel-Referenz: §5.3 (jarvis-memory Server)
"""

from __future__ import annotations

import json
from datetime import date
from typing import TYPE_CHECKING, Any

from jarvis.models import Entity, MemorySearchResult, MemoryTier, Relation
from jarvis.utils.logging import get_logger

if TYPE_CHECKING:
    from jarvis.memory.manager import MemoryManager

log = get_logger(__name__)


class MemoryToolsError(Exception):
    """Fehler bei Memory-Tool-Operationen."""


class MemoryTools:
    """Memory-Operationen als Tool-Calls für den Planner. [B§5.3]

    Wrapper um den MemoryManager, der alle Operationen als
    einfache Funktionen mit String-Ein/Ausgabe exponiert.
    """

    def __init__(self, memory: MemoryManager) -> None:
        """Initialisiert MemoryTools mit einem MemoryManager."""
        self._memory = memory

    # ── Search ───────────────────────────────────────────────────

    def search_memory(
        self,
        query: str,
        top_k: int = 6,
        tier: str = "",
    ) -> str:
        """Durchsucht das Memory-System mit Hybrid-Suche.

        Kombiniert BM25 (lexikalisch), Vektor (semantisch) und
        Graph-Traversal für optimale Ergebnisse. Nutzt synchronen
        BM25-Fallback wenn kein Embedding-Server verfügbar.

        Args:
            query: Natürlichsprachliche Suchanfrage.
            top_k: Maximale Anzahl Ergebnisse (1-20, Default: 6).
            tier: Optionaler Filter auf einen Tier
                  ("core"|"episodic"|"semantic"|"procedural"|"").
                  Leer = alle Tiers durchsuchen.

        Returns:
            Formatierte Suchergebnisse mit Score, Quelle und Text.
        """
        if not query.strip():
            return "Fehler: Leere Suchanfrage."

        top_k = max(1, min(20, top_k))

        # Tier-Filter parsen
        tier_filter = None
        if tier:
            try:
                tier_filter = MemoryTier(tier.lower())
            except ValueError:
                valid = ", ".join(t.value for t in MemoryTier)
                return f"Fehler: Unbekannter Tier '{tier}'. Gültig: {valid}"

        # Synchrone BM25-Suche (immer verfügbar, kein Embedding nötig)
        results = self._memory.search_memory_sync(query, top_k=top_k)

        if not results:
            return f"Keine Ergebnisse für: '{query}'"

        # Tier-Filter anwenden (post-hoc, da search_memory_sync keinen hat)
        if tier_filter:
            results = [r for r in results if r.chunk.memory_tier == tier_filter]

        return self._format_results(results)

    @staticmethod
    def _format_results(results: list[MemorySearchResult]) -> str:
        """Formatiert Suchergebnisse für den Planner.

        Args:
            results: Liste von MemorySearchResult.

        Returns:
            Menschenlesbarer Text mit allen Ergebnissen.
        """
        lines: list[str] = [f"### {len(results)} Ergebnis(se)\n"]

        for i, r in enumerate(results, 1):
            chunk = r.chunk
            tier_label = chunk.memory_tier.value if chunk.memory_tier else "?"
            ts = ""
            if chunk.timestamp:
                ts = f" · {chunk.timestamp.strftime('%Y-%m-%d')}"

            lines.append(
                f"**[{i}]** Score: {r.score:.3f} · "
                f"Tier: {tier_label} · "
                f"Quelle: `{chunk.source_path}`{ts}"
            )
            lines.append(chunk.text.strip())
            lines.append("")

        return "\n".join(lines)

    # ── Save ─────────────────────────────────────────────────────

    def save_to_memory(
        self,
        content: str,
        tier: str = "episodic",
        topic: str = "",
        source_path: str = "",
    ) -> str:
        """Speichert Information in den passenden Memory-Tier.

        Args:
            content: Der zu speichernde Text.
            tier: Ziel-Tier ("episodic"|"semantic"|"procedural").
                  Core ist nicht direkt beschreibbar.
            topic: Thema/Überschrift (für Episodic: wird als Titel verwendet).
            source_path: Optionaler Quellpfad (für Semantic/Procedural).

        Returns:
            Bestätigungsnachricht.
        """
        if not content.strip():
            return "Fehler: Leerer Inhalt."

        tier_lower = tier.lower()

        if tier_lower == "episodic":
            return self._save_episodic(content, topic)
        elif tier_lower == "semantic":
            return self._save_semantic(content, source_path)
        elif tier_lower == "procedural":
            return self._save_procedural(content, source_path)
        elif tier_lower == "core":
            return "Fehler: Core Memory ist nicht direkt beschreibbar. Nutze die CORE.md manuell."
        else:
            valid = "episodic, semantic, procedural"
            return f"Fehler: Unbekannter Tier '{tier}'. Gültig: {valid}"

    def _save_episodic(self, content: str, topic: str) -> str:
        """Schreibt einen Eintrag in den heutigen Tageslog.

        Args:
            content: Der Eintrag-Text.
            topic: Überschrift des Eintrags.

        Returns:
            Bestätigungsnachricht.
        """
        topic = topic or "Notiz"
        self._memory.episodic.append_entry(topic=topic, content=content)
        today = date.today().isoformat()
        return f"Episodic Memory gespeichert: [{today}] {topic}"

    def _save_semantic(self, content: str, source_path: str) -> str:
        """Indexiert Text als semantisches Wissen.

        Args:
            content: Der zu indexierende Text.
            source_path: Virtueller Quellpfad.

        Returns:
            Bestätigungsnachricht mit Chunk-Anzahl.
        """
        if not source_path:
            source_path = f"knowledge/auto/{date.today().isoformat()}.md"

        count = self._memory.index_text(content, source_path, MemoryTier.SEMANTIC)
        return f"Semantic Memory indexiert: {count} Chunk(s) unter '{source_path}'"

    def _save_procedural(self, content: str, source_path: str) -> str:
        """Speichert eine Prozedur/Skill.

        Args:
            content: YAML-Frontmatter + Markdown-Body der Prozedur.
            source_path: Dateiname (z.B. "bu-angebot-erstellen.md").

        Returns:
            Bestätigungsnachricht.
        """
        if not source_path:
            source_path = f"auto-{date.today().isoformat()}.md"

        # Sicherstellen dass es mit .md endet
        if not source_path.endswith(".md"):
            source_path += ".md"

        proc_dir = self._memory.procedural._dir
        target = proc_dir / source_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

        # Auch indexieren
        count = self._memory.index_text(content, str(target), MemoryTier.PROCEDURAL)
        return f"Procedural Memory gespeichert: '{source_path}' ({count} Chunk(s))"

    # ── Entity/Relation (Wissens-Graph) ──────────────────────────

    def get_entity(self, name: str) -> str:
        """Lädt eine Entität mit allen Relationen aus dem Wissens-Graphen.

        Args:
            name: Name der Entität (Teilmatch, case-insensitive).

        Returns:
            Formatierte Entität mit Attributen und Relationen.
        """
        if not name.strip():
            return "Fehler: Leerer Name."

        # Suche in DB
        entities = self._memory.index.search_entities(name)

        if not entities:
            return f"Keine Entität gefunden für: '{name}'"

        lines: list[str] = []
        for entity in entities[:5]:  # Max 5 Treffer
            lines.append(f"### {entity.name}")
            lines.append(f"- **Typ:** {entity.type}")

            if entity.attributes:
                attrs = entity.attributes
                if isinstance(attrs, str):
                    try:
                        attrs = json.loads(attrs)
                    except json.JSONDecodeError:
                        attrs = {}
                for k, v in attrs.items():
                    lines.append(f"- **{k}:** {v}")

            # Relationen laden
            relations = self._memory.index.get_relations_for_entity(entity.id)
            if relations:
                lines.append("\n**Relationen:**")
                for rel in relations:
                    if rel.source_entity == entity.id:
                        other_id = rel.target_entity
                        arrow = "→"
                    else:
                        other_id = rel.source_entity
                        arrow = "←"
                    other = self._memory.index.get_entity_by_id(other_id)
                    other_name = other.name if other else other_id
                    lines.append(f"- {rel.relation_type} {arrow} {other_name}")

            lines.append("")

        return "\n".join(lines)

    def add_entity(
        self,
        name: str,
        entity_type: str,
        attributes: str = "{}",
        source_file: str = "",
    ) -> str:
        """Erstellt eine neue Entität im Wissens-Graphen.

        Args:
            name: Anzeigename der Entität.
            entity_type: Typ (z.B. "person", "company", "product", "project").
            attributes: JSON-String mit zusätzlichen Attributen.
            source_file: Quell-Datei aus der die Entität gelernt wurde.

        Returns:
            Bestätigungsnachricht mit Entity-ID.
        """
        if not name.strip():
            return "Fehler: Leerer Name."

        # Attribute parsen
        try:
            attrs = json.loads(attributes) if attributes else {}
        except json.JSONDecodeError:
            return f"Fehler: Ungültiges JSON in attributes: {attributes}"

        entity = Entity(
            type=entity_type or "unknown",
            name=name.strip(),
            attributes=attrs,
            source_file=source_file,
        )
        self._memory.index.upsert_entity(entity)

        log.info("entity_created", name=name, type=entity_type, id=entity.id)
        return f"Entität erstellt: {name} (Typ: {entity_type}, ID: {entity.id})"

    def add_relation(
        self,
        source_name: str,
        relation_type: str,
        target_name: str,
        attributes: str = "{}",
    ) -> str:
        """Erstellt eine Relation zwischen zwei Entitäten.

        Sucht die Entitäten anhand des Namens. Beide müssen existieren.

        Args:
            source_name: Name der Quell-Entität.
            relation_type: Art der Beziehung (z.B. "hat_police", "arbeitet_bei").
            target_name: Name der Ziel-Entität.
            attributes: JSON-String mit zusätzlichen Attributen.

        Returns:
            Bestätigungsnachricht.
        """
        # Entitäten suchen
        sources = self._memory.index.search_entities(source_name)
        if not sources:
            return f"Fehler: Quell-Entität '{source_name}' nicht gefunden."

        targets = self._memory.index.search_entities(target_name)
        if not targets:
            return f"Fehler: Ziel-Entität '{target_name}' nicht gefunden."

        try:
            attrs = json.loads(attributes) if attributes else {}
        except json.JSONDecodeError:
            return f"Fehler: Ungültiges JSON in attributes: {attributes}"

        relation = Relation(
            source_entity=sources[0].id,
            relation_type=relation_type,
            target_entity=targets[0].id,
            attributes=attrs,
        )
        self._memory.index.upsert_relation(relation)

        log.info(
            "relation_created",
            source=source_name,
            relation=relation_type,
            target=target_name,
        )
        return f"Relation erstellt: {source_name} --[{relation_type}]→ {target_name}"

    # ── Core Memory ──────────────────────────────────────────────

    def get_core_memory(self) -> str:
        """Gibt die aktuelle Core Memory (CORE.md) zurück.

        Returns:
            Vollständiger Inhalt der CORE.md.
        """
        content = self._memory.core.load()
        if not content:
            return "(Core Memory ist leer. Erstelle CORE.md in ~/.jarvis/memory/)"
        return content

    # ── Episodic ─────────────────────────────────────────────────

    def get_recent_episodes(self, days: int = 3) -> str:
        """Lädt die letzten Tageslog-Einträge.

        Args:
            days: Anzahl Tage zurück (1-30, Default: 3).

        Returns:
            Formatierte Tageslog-Einträge.
        """
        days = max(1, min(30, days))

        recent = self._memory.episodic.get_recent(days=days)

        if not recent:
            return f"Keine Episodic-Einträge der letzten {days} Tage."

        lines: list[str] = []
        for d, content in recent:
            lines.append(f"## {d.isoformat()}")
            lines.append(content.strip())
            lines.append("")

        return "\n".join(lines)

    # ── Procedural ───────────────────────────────────────────────

    def search_procedures(self, query: str, top_k: int = 3) -> str:
        """Sucht nach gelernten Prozeduren/Skills.

        Args:
            query: Suchtext (matched gegen Trigger-Keywords und Inhalt).
            top_k: Maximale Anzahl Ergebnisse (Default: 3).

        Returns:
            Formatierte Liste relevanter Prozeduren.
        """
        if not query.strip():
            return "Fehler: Leere Suchanfrage."

        keywords = query.strip().split()
        top_k = max(1, min(10, top_k))
        results = self._memory.procedural.find_by_keywords(keywords)

        if not results:
            return f"Keine Prozeduren gefunden für: '{query}'"

        results = results[:top_k]

        lines: list[str] = [f"### {len(results)} Prozedur(en)\n"]
        for meta, body, score in results:
            lines.append(
                f"**{meta.name}** · Erfolgsrate: {meta.success_rate:.0%} · "
                f"Nutzungen: {meta.total_uses} · Relevanz: {score:.1f}"
            )
            # Body kürzen auf 500 Zeichen
            body_short = body[:500] + "…" if len(body) > 500 else body
            lines.append(body_short)
            lines.append("")

        return "\n".join(lines)

    def record_procedure_usage(
        self,
        name: str,
        success: bool,
        score: float = 0.0,
        session_id: str = "",
    ) -> str:
        """Meldet Erfolg oder Misserfolg einer Prozedur.

        Aktualisiert die Nutzungsstatistik (success_rate, total_uses etc.).

        Args:
            name: Name der Prozedur (Dateiname ohne .md).
            success: True wenn erfolgreich, False bei Fehler.
            score: Bewertungs-Score (0.0-1.0).
            session_id: Session-ID in der die Nutzung stattfand.

        Returns:
            Aktualisierte Statistik der Prozedur.
        """
        if not name.strip():
            return "Fehler: Leerer Prozedurname."

        result = self._memory.procedural.record_usage(
            name=name.strip(),
            success=success,
            score=score,
            session_id=session_id,
        )

        if result is None:
            return f"Fehler: Prozedur '{name}' nicht gefunden."

        status = "Erfolg" if success else "Fehlschlag"
        return (
            f"Prozedur '{name}': {status} erfasst. "
            f"Gesamt: {result.total_uses}x, "
            f"Erfolgsrate: {result.success_rate:.0%}"
        )

    # ── Stats ────────────────────────────────────────────────────

    def memory_stats(self) -> str:
        """Gibt Gesamtstatistiken des Memory-Systems zurück.

        Returns:
            Formatierte Übersicht aller Memory-Tiers und Indizes.
        """
        stats = self._memory.stats()

        lines = [
            "### Memory-System Status\n",
            f"- **Chunks:** {stats['chunks']}",
            f"- **Embeddings:** {stats['embeddings']}",
            f"- **Entitäten:** {stats['entities']}",
            f"- **Relationen:** {stats['relations']}",
            f"- **Prozeduren:** {stats['procedures']} ({stats['procedures_reliable']} zuverlässig)",
            f"- **Episode-Tage:** {stats['episode_dates']}",
            f"- **Core Memory geladen:** {'Ja' if stats['core_memory_loaded'] else 'Nein'}",
            (
                f"- **Embedding-Cache:** {stats['embedding_cache_hits']} Hits"
                f" / {stats['embedding_api_calls']} API-Calls"
            ),
            f"- **Initialisiert:** {'Ja' if stats['initialized'] else 'Nein'}",
        ]

        return "\n".join(lines)


# ── Registration ─────────────────────────────────────────────────


def register_memory_tools(
    mcp_client: Any,
    memory: MemoryManager,
) -> MemoryTools:
    """Registriert Memory-Tools beim MCP-Client.

    Args:
        mcp_client: JarvisMCPClient-Instanz.
        memory: Initialisierter MemoryManager.

    Returns:
        MemoryTools-Instanz für direkten Zugriff.
    """
    mt = MemoryTools(memory)

    # ── search_memory ────────────────────────────────────────────
    mcp_client.register_builtin_handler(
        "search_memory",
        mt.search_memory,
        description=(
            "Durchsucht das Jarvis-Gedächtnis (alle 5 Tiers). "
            "Nutzt Hybrid-Suche: BM25 (lexikalisch) + Vektor (semantisch) + Graph. "
            "Immer als ERSTES aufrufen wenn du Kontext brauchst."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Suchanfrage in natürlicher Sprache",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Max Ergebnisse (1-20)",
                    "default": 6,
                },
                "tier": {
                    "type": "string",
                    "description": "Optionaler Filter: core|episodic|semantic|procedural",
                    "default": "",
                },
            },
            "required": ["query"],
        },
    )

    # ── save_to_memory ───────────────────────────────────────────
    mcp_client.register_builtin_handler(
        "save_to_memory",
        mt.save_to_memory,
        description=(
            "Speichert neue Information ins Jarvis-Gedächtnis. "
            "Tier wählen: 'episodic' für Tageslog, 'semantic' für Fakten, "
            "'procedural' für gelernte Abläufe."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Der zu speichernde Text",
                },
                "tier": {
                    "type": "string",
                    "description": "Ziel: episodic|semantic|procedural",
                    "default": "episodic",
                },
                "topic": {
                    "type": "string",
                    "description": "Thema/Überschrift (für Episodic)",
                    "default": "",
                },
                "source_path": {
                    "type": "string",
                    "description": "Quellpfad (für Semantic/Procedural)",
                    "default": "",
                },
            },
            "required": ["content"],
        },
    )

    # ── get_entity ───────────────────────────────────────────────
    mcp_client.register_builtin_handler(
        "get_entity",
        mt.get_entity,
        description=(
            "Lädt eine Entität aus dem Wissens-Graphen mit allen Attributen "
            "und Relationen. Sucht per Name (Teilmatch möglich)."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name der Entität (z.B. 'Müller')",
                },
            },
            "required": ["name"],
        },
    )

    # ── add_entity ───────────────────────────────────────────────
    mcp_client.register_builtin_handler(
        "add_entity",
        mt.add_entity,
        description=(
            "Erstellt eine neue Entität im Wissens-Graphen. "
            "Typen: person, company, product, project, etc."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Anzeigename",
                },
                "entity_type": {
                    "type": "string",
                    "description": "Typ: person|company|product|project|...",
                },
                "attributes": {
                    "type": "string",
                    "description": (
                        'Zusätzliche Attribute als JSON-String, z.B. {"beruf": "Ingenieur"}'
                    ),
                    "default": "{}",
                },
                "source_file": {
                    "type": "string",
                    "description": "Quelldatei",
                    "default": "",
                },
            },
            "required": ["name", "entity_type"],
        },
    )

    # ── add_relation ─────────────────────────────────────────────
    mcp_client.register_builtin_handler(
        "add_relation",
        mt.add_relation,
        description=(
            "Erstellt eine Beziehung zwischen zwei Entitäten im Wissens-Graphen. "
            "Beispiel: Müller --[nutzt]→ Cloud-Pro"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "source_name": {
                    "type": "string",
                    "description": "Name der Quell-Entität",
                },
                "relation_type": {
                    "type": "string",
                    "description": "Art der Beziehung (z.B. hat_police, arbeitet_bei)",
                },
                "target_name": {
                    "type": "string",
                    "description": "Name der Ziel-Entität",
                },
                "attributes": {
                    "type": "string",
                    "description": "Zusätzliche Attribute als JSON",
                    "default": "{}",
                },
            },
            "required": ["source_name", "relation_type", "target_name"],
        },
    )

    # ── get_core_memory ──────────────────────────────────────────
    mcp_client.register_builtin_handler(
        "get_core_memory",
        mt.get_core_memory,
        description="Gibt die CORE.md zurück (Identität, Regeln, Präferenzen).",
        input_schema={
            "type": "object",
            "properties": {},
        },
    )

    # ── get_recent_episodes ──────────────────────────────────────
    mcp_client.register_builtin_handler(
        "get_recent_episodes",
        mt.get_recent_episodes,
        description="Lädt die Tageslog-Einträge der letzten Tage.",
        input_schema={
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "description": "Anzahl Tage zurück (1-30)",
                    "default": 3,
                },
            },
        },
    )

    # ── search_procedures ────────────────────────────────────────
    mcp_client.register_builtin_handler(
        "search_procedures",
        mt.search_procedures,
        description=(
            "Sucht nach gelernten Prozeduren/Skills. "
            "Prozeduren enthalten bewährte Abläufe für wiederkehrende Aufgaben."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Suchtext (z.B. 'Projekt', 'Email')",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Max Ergebnisse (1-10)",
                    "default": 3,
                },
            },
            "required": ["query"],
        },
    )

    # ── record_procedure_usage ───────────────────────────────────
    mcp_client.register_builtin_handler(
        "record_procedure_usage",
        mt.record_procedure_usage,
        description="Meldet Erfolg/Misserfolg einer Prozedur für Lern-Tracking.",
        input_schema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Prozedurname (ohne .md)",
                },
                "success": {
                    "type": "boolean",
                    "description": "true = Erfolg, false = Fehlschlag",
                },
                "score": {
                    "type": "number",
                    "description": "Bewertungs-Score (0.0-1.0)",
                    "default": 0.0,
                },
                "session_id": {
                    "type": "string",
                    "description": "Session-ID",
                    "default": "",
                },
            },
            "required": ["name", "success"],
        },
    )

    # ── memory_stats ─────────────────────────────────────────────
    mcp_client.register_builtin_handler(
        "memory_stats",
        mt.memory_stats,
        description="Zeigt Statistiken des Memory-Systems (Chunks, Entitäten, Prozeduren etc.).",
        input_schema={
            "type": "object",
            "properties": {},
        },
    )

    _tool_names = [
        "search_memory",
        "save_to_memory",
        "get_entity",
        "add_entity",
        "add_relation",
        "get_core_memory",
        "get_recent_episodes",
        "search_procedures",
        "record_procedure_usage",
        "memory_stats",
    ]
    log.info("memory_tools_registered", tools=_tool_names)
    return mt
