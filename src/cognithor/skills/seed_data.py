"""Seed data for the Skill Marketplace.

Populiert die Marketplace-Datenbank mit den eingebauten Prozeduren
aus ``data/procedures/``. Wird beim ersten Start oder bei leerem
Marketplace automatisch ausgefuehrt.

Architecture reference: SS14 (Skills & Ecosystem)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import yaml

from cognithor.utils.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

log = get_logger(__name__)

# Mapping of procedure categories to marketplace categories
_CATEGORY_MAP: dict[str, str] = {
    "productivity": "produktivitaet",
    "research": "daten",
    "analysis": "daten",
    "general": "sonstiges",
    "development": "entwicklung",
    "communication": "kommunikation",
    "automation": "automatisierung",
    "media": "medien",
    "insurance": "versicherung",
    "finance": "finanzen",
    "integration": "integration",
}


def seed_marketplace(
    store: Any,
    procedures_dir: Path,
    *,
    publisher_id: str = "jarvis-builtin",
    publisher_name: str = "Jarvis Built-in",
) -> int:
    """Populiert die Marketplace-DB mit den eingebauten Prozeduren.

    Liest alle .md-Dateien aus dem angegebenen Verzeichnis, parst
    YAML-Frontmatter und erstellt Listings in der Datenbank.

    Bereits existierende Listings (nach package_id) werden aktualisiert.

    Args:
        store: MarketplaceStore-Instanz.
        procedures_dir: Verzeichnis mit Prozedur-Markdown-Dateien.
        publisher_id: Publisher-ID fuer alle Seed-Listings.
        publisher_name: Anzeigename des Publishers.

    Returns:
        Anzahl der geseedeten Listings.
    """
    if not procedures_dir.exists():
        log.warning("seed_procedures_dir_not_found", path=str(procedures_dir))
        return 0

    count = 0
    for md_file in sorted(procedures_dir.glob("*.md")):
        try:
            listing = _parse_procedure_to_listing(
                md_file,
                publisher_id=publisher_id,
                publisher_name=publisher_name,
            )
            if listing:
                store.save_listing(listing)
                count += 1
                log.debug("seed_listing_saved", package_id=listing["package_id"])
        except Exception as exc:
            log.warning(
                "seed_listing_error",
                file=str(md_file),
                error=str(exc),
            )

    log.info("marketplace_seeded", count=count, dir=str(procedures_dir))
    return count


def _parse_procedure_to_listing(
    path: Path,
    *,
    publisher_id: str = "jarvis-builtin",
    publisher_name: str = "Jarvis Built-in",
) -> dict[str, Any] | None:
    """Parst eine Prozedur-Markdown-Datei und erstellt ein Listing-Dict."""
    content = path.read_text(encoding="utf-8")

    frontmatter: dict[str, Any] = {}
    body = content

    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            try:
                frontmatter = yaml.safe_load(parts[1]) or {}
            except yaml.YAMLError:
                return None
            body = parts[2].strip()

    name = frontmatter.get("name", path.stem)
    if not name:
        return None

    # Map category
    raw_category = frontmatter.get("category", "general")
    category = _CATEGORY_MAP.get(raw_category, raw_category)

    # Use trigger keywords as tags
    triggers = frontmatter.get("trigger_keywords", [])
    if isinstance(triggers, str):
        triggers = [t.strip() for t in triggers.split(",")]

    # Beschreibung: erste nicht-leere Zeile des Body oder Name
    description = ""
    for line in body.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            description = line
            break
    if not description:
        description = name

    # Icon based on category
    _CATEGORY_ICONS: dict[str, str] = {
        "produktivitaet": "⚡",
        "daten": "📊",
        "entwicklung": "💻",
        "kommunikation": "💬",
        "automatisierung": "🤖",
        "medien": "🎨",
        "versicherung": "🛡️",
        "finanzen": "💰",
        "integration": "🔗",
        "sonstiges": "📦",
    }
    icon = _CATEGORY_ICONS.get(category, "📦")

    return {
        "package_id": f"builtin-{path.stem}",
        "name": name,
        "description": description,
        "publisher_id": publisher_id,
        "publisher_name": publisher_name,
        "version": "1.0.0",
        "category": category,
        "tags": triggers,
        "icon": icon,
        "is_featured": frontmatter.get("priority", 0) >= 5,
        "is_verified": True,
        "featured_reason": "Built-in Prozedur" if frontmatter.get("priority", 0) >= 5 else "",
    }
