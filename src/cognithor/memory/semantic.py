"""Semantic Memory · Tier 3 -- Knowledge graph. [B§4.4]

Entities (people, companies, products, projects) and their relationships.
Stored in SQLite (index) + Markdown (source of truth).
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cognithor.models import Entity, Relation

if TYPE_CHECKING:
    from cognithor.memory.indexer import MemoryIndex

logger = logging.getLogger("cognithor.memory.semantic")


class SemanticMemory:
    """Verwaltet den Wissens-Graphen · Entitaeten + Relationen.

    Source of Truth: ~/.cognithor/memory/knowledge/*.md
    Index: SQLite (entities + relations Tabellen)
    """

    def __init__(self, knowledge_dir: str | Path, index: MemoryIndex) -> None:
        """Initialisiert SemanticMemory mit Knowledge-Verzeichnis und Index."""
        self._dir = Path(knowledge_dir)
        self._index = index

    @property
    def directory(self) -> Path:
        """Gibt das Knowledge-Verzeichnis zurueck."""
        return self._dir

    def ensure_directory(self) -> None:
        """Erstellt Verzeichnisstruktur."""
        for sub in ["kunden", "produkte", "projekte"]:
            (self._dir / sub).mkdir(parents=True, exist_ok=True)

    # ── Entity CRUD ──────────────────────────────────────────────

    def add_entity(
        self,
        name: str,
        entity_type: str,
        *,
        attributes: dict[str, Any] | None = None,
        source_file: str = "",
        confidence: float = 1.0,
        provenance_source_type: str | None = None,
        provenance_source_id: str | None = None,
        provenance_notes: str = "",
    ) -> Entity:
        """Erstellt eine neue Entitaet.

        Args:
            name: Name der Entitaet.
            entity_type: Typ (person, company, product, project).
            attributes: Zusaetzliche Attribute.
            source_file: Quelldatei.
            confidence: Vertrauenswert 0-1.
            provenance_source_type: Optional TRUST-9 ``SourceType`` value
                (e.g. ``"chat_utterance"``, ``"tool_output"``,
                ``"agent_inference"``). When provided together with
                ``provenance_source_id``, a tag is written to the
                canonical ``PROVENANCE_LEDGER`` keyed by ``entity.id``.
            provenance_source_id: Stable id of the upstream source
                (audit-log entry id, message id, config path, etc).
                Required when ``provenance_source_type`` is set.
            provenance_notes: Short free-text breadcrumb for the audit
                log. MUST NOT contain prompt or response content.

        Returns:
            Die erstellte Entity.
        """
        entity = Entity(
            type=entity_type,
            name=name,
            attributes=attributes or {},
            source_file=source_file,
            confidence=confidence,
        )
        self._index.upsert_entity(entity)
        logger.info("Entity erstellt: %s (%s) [%s]", name, entity_type, entity.id[:8])
        if provenance_source_type and provenance_source_id:
            self._tag_provenance(
                item_id=entity.id,
                source_type_value=provenance_source_type,
                source_id=provenance_source_id,
                confidence=confidence,
                notes=provenance_notes,
            )
        return entity

    @staticmethod
    def _tag_provenance(
        *,
        item_id: str,
        source_type_value: str,
        source_id: str,
        confidence: float,
        notes: str,
    ) -> None:
        """Best-effort TRUST-9 provenance tag.

        Failures are logged and swallowed — provenance recording must
        never break entity creation. Unknown ``source_type_value`` is
        coerced to :data:`SourceType.UNKNOWN`.
        """
        from cognithor.memory.provenance import (
            PROVENANCE_LEDGER,
            ProvenanceTag,
            SourceType,
        )

        try:
            try:
                source_type = SourceType(source_type_value)
            except ValueError:
                source_type = SourceType.UNKNOWN
            tag = ProvenanceTag(
                source_type=source_type,
                source_id=source_id,
                confidence=confidence,
                notes=notes,
            )
            PROVENANCE_LEDGER.tag(item_id, tag)
        except ValueError as exc:
            logger.debug(
                "semantic_memory_provenance_skipped item_id=%s error=%s",
                item_id,
                exc,
            )

    def update_entity(
        self,
        entity_id: str,
        *,
        name: str | None = None,
        attributes: dict[str, Any] | None = None,
        confidence: float | None = None,
    ) -> Entity | None:
        """Aktualisiert eine Entitaet.

        Returns:
            Die aktualisierte Entity oder None wenn nicht gefunden.
        """
        entity = self._index.get_entity_by_id(entity_id)
        if entity is None:
            return None

        if name is not None:
            entity.name = name
        if attributes is not None:
            entity.attributes.update(attributes)
        if confidence is not None:
            entity.confidence = confidence
        entity.updated_at = datetime.now()

        self._index.upsert_entity(entity)
        return entity

    def get_entity(self, entity_id: str) -> Entity | None:
        """Laedt eine Entitaet."""
        return self._index.get_entity_by_id(entity_id)

    def find_entities(
        self,
        name: str | None = None,
        entity_type: str | None = None,
    ) -> list[Entity]:
        """Sucht Entitaeten."""
        return self._index.search_entities(name=name, entity_type=entity_type)

    def delete_entity(self, entity_id: str) -> bool:
        """Loescht eine Entitaet und ihre Relationen."""
        return self._index.delete_entity(entity_id)

    # ── Relation CRUD ────────────────────────────────────────────

    def add_relation(
        self,
        source_id: str,
        relation_type: str,
        target_id: str,
        *,
        attributes: dict[str, Any] | None = None,
        source_file: str = "",
        confidence: float = 1.0,
    ) -> Relation | None:
        """Erstellt eine neue Relation zwischen zwei Entitaeten.

        Args:
            source_id: Quell-Entity ID.
            relation_type: Beziehungstyp (z.B. "hat_police", "arbeitet_bei").
            target_id: Ziel-Entity ID.

        Returns:
            Die erstellte Relation oder None wenn Entitaeten nicht existieren.
        """
        # Pruefe ob beide Entitaeten existieren
        if self._index.get_entity_by_id(source_id) is None:
            logger.warning("Source-Entity %s nicht gefunden", source_id)
            return None
        if self._index.get_entity_by_id(target_id) is None:
            logger.warning("Target-Entity %s nicht gefunden", target_id)
            return None

        relation = Relation(
            source_entity=source_id,
            relation_type=relation_type,
            target_entity=target_id,
            attributes=attributes or {},
            source_file=source_file,
            confidence=confidence,
        )
        self._index.upsert_relation(relation)
        logger.info(
            "Relation erstellt: %s -[%s]-> %s",
            source_id[:8],
            relation_type,
            target_id[:8],
        )
        return relation

    def get_relations(
        self,
        entity_id: str,
        relation_type: str | None = None,
    ) -> list[Relation]:
        """Alle Relationen einer Entitaet."""
        return self._index.get_relations_for_entity(entity_id, relation_type)

    def get_neighbors(self, entity_id: str, max_depth: int = 1) -> list[Entity]:
        """Nachbar-Entitaeten im Graph."""
        return self._index.graph_traverse(entity_id, max_depth=max_depth)

    # ── Convenience ──────────────────────────────────────────────

    def get_entity_with_relations(
        self,
        entity_id: str,
    ) -> tuple[Entity | None, list[tuple[Relation, Entity]]]:
        """Laedt eine Entitaet mit all ihren Relationen und verbundenen Entitaeten.

        Returns:
            (entity, [(relation, connected_entity), ...])
        """
        entity = self._index.get_entity_by_id(entity_id)
        if entity is None:
            return None, []

        relations = self._index.get_relations_for_entity(entity_id)
        connected: list[tuple[Relation, Entity]] = []

        for rel in relations:
            other_id = rel.target_entity if rel.source_entity == entity_id else rel.source_entity
            other = self._index.get_entity_by_id(other_id)
            if other:
                connected.append((rel, other))

        return entity, connected

    def export_graph_summary(self) -> str:
        """Exportiert eine lesbare Zusammenfassung des Wissens-Graphen.

        Returns:
            Markdown-formatierte Uebersicht.
        """
        entities = self._index.search_entities()
        if not entities:
            return "# Wissens-Graph\n\nKeine Entitäten vorhanden.\n"

        lines = ["# Wissens-Graph\n"]

        # Gruppiert nach Typ
        by_type: dict[str, list[Entity]] = {}
        for e in entities:
            by_type.setdefault(e.type, []).append(e)

        for etype, ents in sorted(by_type.items()):
            lines.append(f"\n## {etype.title()} ({len(ents)})\n")
            for e in sorted(ents, key=lambda x: x.name):
                relations = self._index.get_relations_for_entity(e.id)
                rel_strs = []
                for r in relations:
                    other_id = r.target_entity if r.source_entity == e.id else r.source_entity
                    other = self._index.get_entity_by_id(other_id)
                    if other:
                        rel_strs.append(f"{r.relation_type} → {other.name}")

                attrs_str = ""
                if e.attributes:
                    attrs_str = f" ({', '.join(f'{k}={v}' for k, v in e.attributes.items())})"

                lines.append(f"- **{e.name}**{attrs_str}")
                for rs in rel_strs:
                    lines.append(f"  - {rs}")

        return "\n".join(lines)

    # ── Stats ────────────────────────────────────────────────────

    def stats(self) -> dict[str, int]:
        """Statistiken."""
        return {
            "entities": self._index.count_entities(),
            "relations": self._index.count_relations(),
        }
