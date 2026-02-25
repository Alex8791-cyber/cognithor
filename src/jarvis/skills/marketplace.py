"""Skill-Marketplace: Kuratierter Marktplatz für Jarvis-Skills.

Erweitert das P2P-Ökosystem um:
  - SkillMarketplace: Zentrale Anlaufstelle für Suche, Browse, Install
  - FeaturedFeed: Kuratierte Empfehlungen und Trending-Ranking
  - CategoryBrowser: Hierarchische Kategorie-Navigation
  - SkillDetail: Detaillierte Skill-Informationen inkl. Reviews
  - OneClickInstaller: Vereinfachte Installation

Bibel-Referenz: §14 (Skills & Ecosystem)
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from jarvis.utils.logging import get_logger

log = get_logger(__name__)


# ============================================================================
# Datenmodelle
# ============================================================================


class SkillCategory(Enum):
    """Haupt-Kategorien für Skills."""

    INSURANCE = "versicherung"
    FINANCE = "finanzen"
    PRODUCTIVITY = "produktivitaet"
    COMMUNICATION = "kommunikation"
    DATA = "daten"
    DEVELOPMENT = "entwicklung"
    AUTOMATION = "automatisierung"
    MEDIA = "medien"
    INTEGRATION = "integration"
    OTHER = "sonstiges"


@dataclass
class SkillListing:
    """Ein Skill-Eintrag im Marketplace."""

    package_id: str
    name: str
    description: str
    publisher_id: str
    publisher_name: str = ""
    version: str = "1.0.0"
    category: SkillCategory = SkillCategory.OTHER
    tags: list[str] = field(default_factory=list)
    icon: str = ""  # Emoji oder URL
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Statistiken
    install_count: int = 0
    rating_sum: float = 0.0
    rating_count: int = 0
    review_count: int = 0

    # Featured/Curated
    is_featured: bool = False
    is_verified: bool = False
    featured_reason: str = ""

    # Permissions & Security
    required_permissions: list[str] = field(default_factory=list)
    security_scan_passed: bool = False
    security_scan_report: dict[str, Any] = field(default_factory=dict)

    @property
    def average_rating(self) -> float:
        if self.rating_count == 0:
            return 0.0
        return round(self.rating_sum / self.rating_count, 1)

    @property
    def popularity_score(self) -> float:
        """Gewichteter Score aus Installs, Rating und Aktualität."""
        age_days = max(1, (datetime.now(UTC) - self.created_at).days)
        recency_boost = 1.0 / (1.0 + age_days / 30.0)
        install_score = min(self.install_count / 10.0, 10.0)
        rating_score = self.average_rating * 2
        return round(install_score + rating_score + recency_boost * 5, 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "package_id": self.package_id,
            "name": self.name,
            "description": self.description,
            "publisher_id": self.publisher_id,
            "publisher_name": self.publisher_name,
            "version": self.version,
            "category": self.category.value,
            "tags": self.tags,
            "icon": self.icon,
            "install_count": self.install_count,
            "average_rating": self.average_rating,
            "review_count": self.review_count,
            "is_featured": self.is_featured,
            "is_verified": self.is_verified,
            "popularity_score": self.popularity_score,
        }


@dataclass
class SkillReview:
    """Eine Nutzer-Bewertung eines Skills."""

    review_id: str
    package_id: str
    reviewer_id: str
    reviewer_name: str = ""
    rating: int = 5  # 1-5
    comment: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    helpful_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "review_id": self.review_id,
            "package_id": self.package_id,
            "reviewer_name": self.reviewer_name,
            "rating": self.rating,
            "comment": self.comment,
            "created_at": self.created_at.isoformat(),
            "helpful_count": self.helpful_count,
        }


@dataclass
class CategoryInfo:
    """Informationen über eine Skill-Kategorie."""

    category: SkillCategory
    display_name: str
    icon: str
    description: str
    skill_count: int = 0


# Standard-Kategorien mit Anzeige-Infos
CATEGORY_INFOS: dict[SkillCategory, CategoryInfo] = {
    SkillCategory.INSURANCE: CategoryInfo(
        SkillCategory.INSURANCE, "Versicherung", "🛡️",
        "BU, KV, LV Rechner und Vergleiche",
    ),
    SkillCategory.FINANCE: CategoryInfo(
        SkillCategory.FINANCE, "Finanzen", "💰",
        "Steuer, Investment, Buchhaltung",
    ),
    SkillCategory.PRODUCTIVITY: CategoryInfo(
        SkillCategory.PRODUCTIVITY, "Produktivität", "⚡",
        "Kalender, Aufgaben, Zeitmanagement",
    ),
    SkillCategory.COMMUNICATION: CategoryInfo(
        SkillCategory.COMMUNICATION, "Kommunikation", "💬",
        "E-Mail, Chat, CRM Integration",
    ),
    SkillCategory.DATA: CategoryInfo(
        SkillCategory.DATA, "Daten", "📊",
        "Analyse, Reporting, Visualisierung",
    ),
    SkillCategory.DEVELOPMENT: CategoryInfo(
        SkillCategory.DEVELOPMENT, "Entwicklung", "💻",
        "Code-Tools, Testing, Deployment",
    ),
    SkillCategory.AUTOMATION: CategoryInfo(
        SkillCategory.AUTOMATION, "Automatisierung", "🤖",
        "Workflows, Cronjobs, Pipelines",
    ),
    SkillCategory.MEDIA: CategoryInfo(
        SkillCategory.MEDIA, "Medien", "🎨",
        "Bilder, Videos, Audio, Dokumente",
    ),
    SkillCategory.INTEGRATION: CategoryInfo(
        SkillCategory.INTEGRATION, "Integration", "🔗",
        "API-Anbindungen, Webhooks, Importer",
    ),
    SkillCategory.OTHER: CategoryInfo(
        SkillCategory.OTHER, "Sonstiges", "📦",
        "Weitere Skills",
    ),
}


# ============================================================================
# SkillMarketplace
# ============================================================================


class SkillMarketplace:
    """Kuratierter Skill-Marktplatz.

    Bietet:
    - Suche (Volltext + Filter)
    - Browse (Kategorien, Featured, Trending)
    - Reviews und Ratings
    - One-Click-Install Tracking
    - Kuratierte Empfehlungen
    """

    def __init__(self) -> None:
        self._listings: dict[str, SkillListing] = {}
        self._reviews: dict[str, list[SkillReview]] = defaultdict(list)
        self._installed: dict[str, set[str]] = defaultdict(set)  # user→set(pkg)
        self._review_counter = 0
        self._verified_publishers: set[str] = set()
        self._banned_publishers: set[str] = set()
        self._recall_log: list[dict[str, Any]] = []

    # --- Listings CRUD ---

    def publish(self, listing: SkillListing) -> SkillListing:
        """Publiziert oder aktualisiert einen Skill im Marketplace."""
        if listing.publisher_id in self._banned_publishers:
            msg = f"Publisher '{listing.publisher_id}' is banned"
            raise ValueError(msg)
        listing.updated_at = datetime.now(UTC)
        self._listings[listing.package_id] = listing
        log.info("skill_published", package_id=listing.package_id, name=listing.name)
        return listing

    def get_listing(self, package_id: str) -> SkillListing | None:
        return self._listings.get(package_id)

    def remove_listing(self, package_id: str) -> bool:
        if package_id in self._listings:
            del self._listings[package_id]
            return True
        return False

    @property
    def listing_count(self) -> int:
        return len(self._listings)

    # --- Suche ---

    def search(
        self,
        query: str = "",
        *,
        category: SkillCategory | None = None,
        tags: list[str] | None = None,
        verified_only: bool = False,
        min_rating: float = 0.0,
        sort_by: str = "relevance",  # relevance, popularity, rating, newest
        max_results: int = 20,
    ) -> list[SkillListing]:
        """Durchsucht den Marketplace."""
        results = list(self._listings.values())

        # Filter: Kategorie
        if category:
            results = [r for r in results if r.category == category]

        # Filter: Tags
        if tags:
            results = [
                r for r in results
                if any(t in r.tags for t in tags)
            ]

        # Filter: Verified
        if verified_only:
            results = [r for r in results if r.is_verified]

        # Filter: Min-Rating
        if min_rating > 0:
            results = [r for r in results if r.average_rating >= min_rating]

        # Filter: Volltext
        if query:
            q_lower = query.lower()
            results = [
                r for r in results
                if q_lower in r.name.lower()
                or q_lower in r.description.lower()
                or any(q_lower in t.lower() for t in r.tags)
            ]

        # Sortierung
        if sort_by == "popularity":
            results.sort(key=lambda r: r.popularity_score, reverse=True)
        elif sort_by == "rating":
            results.sort(key=lambda r: r.average_rating, reverse=True)
        elif sort_by == "newest":
            results.sort(key=lambda r: r.created_at, reverse=True)
        elif sort_by == "installs":
            results.sort(key=lambda r: r.install_count, reverse=True)
        else:  # relevance
            results.sort(key=lambda r: r.popularity_score, reverse=True)

        return results[:max_results]

    # --- Kategorien ---

    def categories(self) -> list[CategoryInfo]:
        """Gibt alle Kategorien mit Skill-Anzahl zurück."""
        counts: dict[SkillCategory, int] = defaultdict(int)
        for listing in self._listings.values():
            counts[listing.category] += 1

        infos = []
        for cat, info in CATEGORY_INFOS.items():
            info_copy = CategoryInfo(
                category=info.category,
                display_name=info.display_name,
                icon=info.icon,
                description=info.description,
                skill_count=counts.get(cat, 0),
            )
            infos.append(info_copy)
        return infos

    def by_category(self, category: SkillCategory) -> list[SkillListing]:
        """Alle Skills einer Kategorie, nach Popularität sortiert."""
        return self.search(category=category, sort_by="popularity")

    # --- Featured & Trending ---

    def featured(self, max_results: int = 10) -> list[SkillListing]:
        """Kuratierte Empfehlungen."""
        featured = [l for l in self._listings.values() if l.is_featured]
        featured.sort(key=lambda r: r.popularity_score, reverse=True)
        return featured[:max_results]

    def trending(self, max_results: int = 10) -> list[SkillListing]:
        """Trending: Hohe Install-Rate + gute Bewertung + neulich aktualisiert."""
        all_listings = list(self._listings.values())
        all_listings.sort(key=lambda r: r.popularity_score, reverse=True)
        return all_listings[:max_results]

    def newest(self, max_results: int = 10) -> list[SkillListing]:
        """Neueste Skills."""
        return self.search(sort_by="newest", max_results=max_results)

    def top_rated(self, max_results: int = 10) -> list[SkillListing]:
        """Bestbewertete Skills (min 2 Reviews)."""
        rated = [l for l in self._listings.values() if l.rating_count >= 2]
        rated.sort(key=lambda r: r.average_rating, reverse=True)
        return rated[:max_results]

    # --- Reviews ---

    def add_review(
        self,
        package_id: str,
        reviewer_id: str,
        rating: int,
        comment: str = "",
        reviewer_name: str = "",
    ) -> SkillReview | None:
        """Fügt eine Bewertung hinzu. 1-5 Sterne."""
        listing = self._listings.get(package_id)
        if not listing:
            return None

        if rating < 1 or rating > 5:
            return None

        # Duplikat-Check
        existing = [r for r in self._reviews[package_id] if r.reviewer_id == reviewer_id]
        if existing:
            return None  # Schon bewertet

        self._review_counter += 1
        review = SkillReview(
            review_id=f"review_{self._review_counter}",
            package_id=package_id,
            reviewer_id=reviewer_id,
            reviewer_name=reviewer_name,
            rating=rating,
            comment=comment,
        )
        self._reviews[package_id].append(review)

        # Listing-Statistiken aktualisieren
        listing.rating_sum += rating
        listing.rating_count += 1
        listing.review_count += 1

        return review

    def get_reviews(self, package_id: str) -> list[SkillReview]:
        return self._reviews.get(package_id, [])

    # --- Installation ---

    def record_install(self, package_id: str, user_id: str) -> bool:
        """Registriert eine Installation."""
        listing = self._listings.get(package_id)
        if not listing:
            return False

        if package_id not in self._installed[user_id]:
            listing.install_count += 1
            self._installed[user_id].add(package_id)
            return True
        return False  # Schon installiert

    def is_installed(self, package_id: str, user_id: str) -> bool:
        return package_id in self._installed.get(user_id, set())

    def user_installed(self, user_id: str) -> list[SkillListing]:
        """Alle installierten Skills eines Users."""
        pkg_ids = self._installed.get(user_id, set())
        return [self._listings[pid] for pid in pkg_ids if pid in self._listings]

    # --- Feature-Management ---

    def set_featured(
        self,
        package_id: str,
        featured: bool = True,
        reason: str = "",
    ) -> bool:
        """Markiert einen Skill als featured."""
        listing = self._listings.get(package_id)
        if not listing:
            return False
        listing.is_featured = featured
        listing.featured_reason = reason
        return True

    def set_verified(self, package_id: str, verified: bool = True) -> bool:
        """Markiert einen Skill als verifiziert."""
        listing = self._listings.get(package_id)
        if not listing:
            return False
        listing.is_verified = verified
        return True

    # --- Statistiken ---

    def stats(self) -> dict[str, Any]:
        all_listings = list(self._listings.values())
        return {
            "total_skills": len(all_listings),
            "featured_count": sum(1 for l in all_listings if l.is_featured),
            "verified_count": sum(1 for l in all_listings if l.is_verified),
            "total_installs": sum(l.install_count for l in all_listings),
            "total_reviews": sum(len(r) for r in self._reviews.values()),
            "categories_used": len(set(l.category for l in all_listings)),
            "unique_publishers": len(set(l.publisher_id for l in all_listings)),
        }

    # ------------------------------------------------------------------
    # Publisher-Verifizierung
    # ------------------------------------------------------------------

    def verify_publisher(self, publisher_id: str, *, verified: bool = True) -> int:
        """Markiert alle Skills eines Publishers als verifiziert/unverifiziert.

        Returns:
            Anzahl betroffener Skills.
        """
        count = 0
        for listing in self._listings.values():
            if listing.publisher_id == publisher_id:
                listing.is_verified = verified
                count += 1
        if count:
            self._verified_publishers.add(publisher_id) if verified else \
                self._verified_publishers.discard(publisher_id)
        return count

    def is_publisher_verified(self, publisher_id: str) -> bool:
        return publisher_id in self._verified_publishers

    # ------------------------------------------------------------------
    # Emergency Recall
    # ------------------------------------------------------------------

    def recall_skill(
        self,
        package_id: str,
        reason: str = "",
        *,
        ban_publisher: bool = False,
    ) -> dict[str, Any]:
        """Zieht einen Skill aus dem Marktplatz zurück (Emergency Recall).

        Returns:
            Recall-Report mit Betroffenen.
        """
        listing = self._listings.get(package_id)
        if not listing:
            return {"recalled": False, "error": "package_not_found"}

        publisher_id = listing.publisher_id
        recalled_ids = [package_id]

        # Skill entfernen
        del self._listings[package_id]

        # Optional: Alle Skills des Publishers bannen
        if ban_publisher and publisher_id:
            self._banned_publishers.add(publisher_id)
            for pid in list(self._listings.keys()):
                if self._listings[pid].publisher_id == publisher_id:
                    recalled_ids.append(pid)
                    del self._listings[pid]

        recall_entry = {
            "recalled": True,
            "package_id": package_id,
            "reason": reason,
            "publisher_id": publisher_id,
            "publisher_banned": ban_publisher,
            "recalled_count": len(recalled_ids),
            "recalled_ids": recalled_ids,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        self._recall_log.append(recall_entry)
        log.warning("skill_recalled", package_id=package_id, reason=reason)
        return recall_entry

    @property
    def recall_log(self) -> list[dict[str, Any]]:
        return list(self._recall_log)

    @property
    def banned_publishers(self) -> set[str]:
        return set(self._banned_publishers)

    # ------------------------------------------------------------------
    # Permission-Display (Sandbox-Rechte)
    # ------------------------------------------------------------------

    def set_permissions(
        self,
        package_id: str,
        permissions: list[str],
    ) -> bool:
        """Setzt die benötigten Berechtigungen eines Skills.

        Args:
            package_id: Skill-ID
            permissions: z.B. ["filesystem:read", "network:allow", "shell:execute"]
        """
        listing = self._listings.get(package_id)
        if not listing:
            return False
        listing.required_permissions = list(permissions)
        return True

    def get_permissions(self, package_id: str) -> list[str]:
        """Gibt die benötigten Berechtigungen eines Skills zurück."""
        listing = self._listings.get(package_id)
        if not listing:
            return []
        return list(getattr(listing, "required_permissions", []))

    # ------------------------------------------------------------------
    # Security-Scan-Hook
    # ------------------------------------------------------------------

    def set_scan_result(
        self,
        package_id: str,
        *,
        passed: bool,
        scan_report: dict[str, Any] | None = None,
    ) -> bool:
        """Speichert das Ergebnis eines Security-Scans."""
        listing = self._listings.get(package_id)
        if not listing:
            return False
        listing.security_scan_passed = passed
        listing.security_scan_report = scan_report or {}
        return True

    def needs_scan(self, package_id: str) -> bool:
        """Prüft ob ein Skill einen Security-Scan benötigt."""
        listing = self._listings.get(package_id)
        if not listing:
            return False
        return not getattr(listing, "security_scan_passed", False)
