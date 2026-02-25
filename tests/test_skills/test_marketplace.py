"""Tests für Skill-Marketplace.

Testet: Publishing, Suche, Kategorien, Featured, Trending,
Reviews, Installation, Statistiken.
"""

from __future__ import annotations

import pytest

from jarvis.skills.marketplace import (
    CATEGORY_INFOS,
    SkillCategory,
    SkillListing,
    SkillMarketplace,
    SkillReview,
)


@pytest.fixture
def market() -> SkillMarketplace:
    return SkillMarketplace()


@pytest.fixture
def populated_market(market: SkillMarketplace) -> SkillMarketplace:
    """Marketplace mit Beispiel-Skills."""
    market.publish(SkillListing(
        package_id="bu_calc", name="BU-Rechner",
        description="Berufsunfähigkeits-Tarifvergleich mit 40+ Tarifen",
        publisher_id="p_alex", publisher_name="Alex",
        category=SkillCategory.INSURANCE,
        tags=["bu", "tarif", "vergleich"],
        icon="🛡️", is_verified=True,
    ))
    market.publish(SkillListing(
        package_id="kv_check", name="KV-Tarifcheck",
        description="Krankenversicherungs-Optimierung",
        publisher_id="p_alex", publisher_name="Alex",
        category=SkillCategory.INSURANCE,
        tags=["kv", "krankenversicherung"],
        icon="🏥", is_featured=True, is_verified=True,
    ))
    market.publish(SkillListing(
        package_id="pdf_gen", name="PDF-Generator",
        description="Professionelle Angebots-PDFs erstellen",
        publisher_id="p_bob", publisher_name="Bob",
        category=SkillCategory.PRODUCTIVITY,
        tags=["pdf", "angebot", "dokument"],
        icon="📄",
    ))
    market.publish(SkillListing(
        package_id="crm_sync", name="CRM-Synchronisation",
        description="Kundendaten mit externem CRM abgleichen",
        publisher_id="p_carol", publisher_name="Carol",
        category=SkillCategory.INTEGRATION,
        tags=["crm", "sync", "kunde"],
        icon="🔗", is_featured=True,
    ))
    market.publish(SkillListing(
        package_id="code_review", name="Code-Review-Bot",
        description="Automatische Code-Reviews mit AI",
        publisher_id="p_bob", publisher_name="Bob",
        category=SkillCategory.DEVELOPMENT,
        tags=["code", "review", "ai"],
        icon="💻",
    ))
    return market


# ============================================================================
# Publishing
# ============================================================================


class TestPublishing:
    def test_publish(self, market: SkillMarketplace) -> None:
        listing = market.publish(SkillListing(
            package_id="test", name="Test", description="A test",
            publisher_id="p1",
        ))
        assert listing.package_id == "test"
        assert market.listing_count == 1

    def test_get_listing(self, populated_market: SkillMarketplace) -> None:
        listing = populated_market.get_listing("bu_calc")
        assert listing is not None
        assert listing.name == "BU-Rechner"

    def test_get_nonexistent(self, market: SkillMarketplace) -> None:
        assert market.get_listing("nope") is None

    def test_remove(self, populated_market: SkillMarketplace) -> None:
        assert populated_market.remove_listing("bu_calc")
        assert populated_market.get_listing("bu_calc") is None

    def test_update(self, populated_market: SkillMarketplace) -> None:
        listing = populated_market.get_listing("bu_calc")
        assert listing is not None
        listing.description = "Neue Beschreibung"
        populated_market.publish(listing)
        updated = populated_market.get_listing("bu_calc")
        assert updated is not None
        assert updated.description == "Neue Beschreibung"


# ============================================================================
# Suche
# ============================================================================


class TestSearch:
    def test_search_all(self, populated_market: SkillMarketplace) -> None:
        results = populated_market.search()
        assert len(results) == 5

    def test_search_by_query(self, populated_market: SkillMarketplace) -> None:
        results = populated_market.search("BU")
        assert len(results) == 1
        assert results[0].package_id == "bu_calc"

    def test_search_by_description(self, populated_market: SkillMarketplace) -> None:
        results = populated_market.search("Tarifvergleich")
        assert len(results) == 1

    def test_search_by_category(self, populated_market: SkillMarketplace) -> None:
        results = populated_market.search(category=SkillCategory.INSURANCE)
        assert len(results) == 2

    def test_search_by_tags(self, populated_market: SkillMarketplace) -> None:
        results = populated_market.search(tags=["pdf"])
        assert len(results) == 1
        assert results[0].package_id == "pdf_gen"

    def test_search_verified_only(self, populated_market: SkillMarketplace) -> None:
        results = populated_market.search(verified_only=True)
        assert all(r.is_verified for r in results)
        assert len(results) == 2

    def test_search_min_rating(self, populated_market: SkillMarketplace) -> None:
        populated_market.add_review("bu_calc", "u1", 5)
        populated_market.add_review("pdf_gen", "u1", 2)

        results = populated_market.search(min_rating=4.0)
        assert len(results) == 1
        assert results[0].package_id == "bu_calc"

    def test_search_sort_by_newest(self, populated_market: SkillMarketplace) -> None:
        results = populated_market.search(sort_by="newest")
        # Alle haben fast gleiche Erstellung, aber Reihenfolge stabil
        assert len(results) == 5

    def test_search_sort_by_installs(self, populated_market: SkillMarketplace) -> None:
        populated_market.record_install("pdf_gen", "user1")
        populated_market.record_install("pdf_gen", "user2")
        results = populated_market.search(sort_by="installs")
        assert results[0].package_id == "pdf_gen"

    def test_search_max_results(self, populated_market: SkillMarketplace) -> None:
        results = populated_market.search(max_results=2)
        assert len(results) == 2


# ============================================================================
# Kategorien
# ============================================================================


class TestCategories:
    def test_categories_with_counts(self, populated_market: SkillMarketplace) -> None:
        cats = populated_market.categories()
        assert len(cats) == len(SkillCategory)

        insurance = next(c for c in cats if c.category == SkillCategory.INSURANCE)
        assert insurance.skill_count == 2
        assert insurance.icon == "🛡️"

    def test_by_category(self, populated_market: SkillMarketplace) -> None:
        results = populated_market.by_category(SkillCategory.INSURANCE)
        assert len(results) == 2

    def test_category_infos_complete(self) -> None:
        for cat in SkillCategory:
            assert cat in CATEGORY_INFOS
            info = CATEGORY_INFOS[cat]
            assert info.display_name
            assert info.icon


# ============================================================================
# Featured & Trending
# ============================================================================


class TestFeaturedTrending:
    def test_featured(self, populated_market: SkillMarketplace) -> None:
        results = populated_market.featured()
        assert all(r.is_featured for r in results)
        assert len(results) == 2  # kv_check + crm_sync

    def test_trending(self, populated_market: SkillMarketplace) -> None:
        # Installs + Ratings = höherer Score
        populated_market.record_install("bu_calc", "u1")
        populated_market.record_install("bu_calc", "u2")
        populated_market.record_install("bu_calc", "u3")
        populated_market.add_review("bu_calc", "u1", 5)

        results = populated_market.trending(3)
        assert len(results) == 3
        assert results[0].package_id == "bu_calc"

    def test_newest(self, populated_market: SkillMarketplace) -> None:
        results = populated_market.newest(3)
        assert len(results) == 3

    def test_top_rated(self, populated_market: SkillMarketplace) -> None:
        # Braucht min 2 Reviews
        populated_market.add_review("bu_calc", "u1", 5)
        populated_market.add_review("bu_calc", "u2", 4)
        populated_market.add_review("pdf_gen", "u1", 3)
        populated_market.add_review("pdf_gen", "u2", 2)

        top = populated_market.top_rated()
        assert len(top) == 2
        assert top[0].package_id == "bu_calc"  # 4.5 > 2.5

    def test_set_featured(self, populated_market: SkillMarketplace) -> None:
        assert populated_market.set_featured("bu_calc", True, "Skill der Woche")
        listing = populated_market.get_listing("bu_calc")
        assert listing is not None
        assert listing.is_featured
        assert listing.featured_reason == "Skill der Woche"

    def test_set_verified(self, populated_market: SkillMarketplace) -> None:
        assert populated_market.set_verified("pdf_gen")
        listing = populated_market.get_listing("pdf_gen")
        assert listing is not None
        assert listing.is_verified


# ============================================================================
# Reviews
# ============================================================================


class TestReviews:
    def test_add_review(self, populated_market: SkillMarketplace) -> None:
        review = populated_market.add_review(
            "bu_calc", "user_1", 5, "Großartiger Rechner!",
        )
        assert review is not None
        assert review.rating == 5

    def test_review_updates_listing(self, populated_market: SkillMarketplace) -> None:
        populated_market.add_review("bu_calc", "u1", 5)
        populated_market.add_review("bu_calc", "u2", 3)
        listing = populated_market.get_listing("bu_calc")
        assert listing is not None
        assert listing.average_rating == 4.0
        assert listing.review_count == 2

    def test_duplicate_review_blocked(self, populated_market: SkillMarketplace) -> None:
        populated_market.add_review("bu_calc", "u1", 5)
        assert populated_market.add_review("bu_calc", "u1", 3) is None

    def test_invalid_rating_blocked(self, populated_market: SkillMarketplace) -> None:
        assert populated_market.add_review("bu_calc", "u1", 0) is None
        assert populated_market.add_review("bu_calc", "u2", 6) is None

    def test_review_nonexistent_skill(self, populated_market: SkillMarketplace) -> None:
        assert populated_market.add_review("nope", "u1", 5) is None

    def test_get_reviews(self, populated_market: SkillMarketplace) -> None:
        populated_market.add_review("bu_calc", "u1", 5, "Super!")
        populated_market.add_review("bu_calc", "u2", 4, "Gut")
        reviews = populated_market.get_reviews("bu_calc")
        assert len(reviews) == 2


# ============================================================================
# Installation
# ============================================================================


class TestInstallation:
    def test_record_install(self, populated_market: SkillMarketplace) -> None:
        assert populated_market.record_install("bu_calc", "user_1")
        listing = populated_market.get_listing("bu_calc")
        assert listing is not None
        assert listing.install_count == 1

    def test_duplicate_install_ignored(self, populated_market: SkillMarketplace) -> None:
        populated_market.record_install("bu_calc", "user_1")
        assert not populated_market.record_install("bu_calc", "user_1")
        listing = populated_market.get_listing("bu_calc")
        assert listing is not None
        assert listing.install_count == 1

    def test_is_installed(self, populated_market: SkillMarketplace) -> None:
        populated_market.record_install("bu_calc", "u1")
        assert populated_market.is_installed("bu_calc", "u1")
        assert not populated_market.is_installed("bu_calc", "u2")

    def test_user_installed(self, populated_market: SkillMarketplace) -> None:
        populated_market.record_install("bu_calc", "u1")
        populated_market.record_install("pdf_gen", "u1")
        installed = populated_market.user_installed("u1")
        assert len(installed) == 2

    def test_install_nonexistent(self, populated_market: SkillMarketplace) -> None:
        assert not populated_market.record_install("nope", "u1")


# ============================================================================
# SkillListing Model
# ============================================================================


class TestSkillListing:
    def test_average_rating(self) -> None:
        l = SkillListing(
            package_id="t", name="T", description="T", publisher_id="p",
            rating_sum=12.0, rating_count=3,
        )
        assert l.average_rating == 4.0

    def test_average_rating_zero(self) -> None:
        l = SkillListing(
            package_id="t", name="T", description="T", publisher_id="p",
        )
        assert l.average_rating == 0.0

    def test_popularity_score(self) -> None:
        l = SkillListing(
            package_id="t", name="T", description="T", publisher_id="p",
            install_count=50, rating_sum=20, rating_count=5,
        )
        assert l.popularity_score > 0

    def test_to_dict(self) -> None:
        l = SkillListing(
            package_id="t", name="Test", description="D", publisher_id="p",
            category=SkillCategory.INSURANCE, tags=["a", "b"],
        )
        d = l.to_dict()
        assert d["package_id"] == "t"
        assert d["category"] == "versicherung"
        assert "popularity_score" in d


# ============================================================================
# Statistiken
# ============================================================================


class TestStats:
    def test_stats_empty(self, market: SkillMarketplace) -> None:
        s = market.stats()
        assert s["total_skills"] == 0

    def test_stats_populated(self, populated_market: SkillMarketplace) -> None:
        populated_market.add_review("bu_calc", "u1", 5)
        populated_market.record_install("bu_calc", "u1")

        s = populated_market.stats()
        assert s["total_skills"] == 5
        assert s["featured_count"] == 2
        assert s["verified_count"] == 2
        assert s["total_installs"] == 1
        assert s["total_reviews"] == 1
        assert s["unique_publishers"] == 3


# ============================================================================
# Publisher-Verifizierung
# ============================================================================


class TestPublisherVerification:
    def test_verify_publisher(self, populated_market: SkillMarketplace) -> None:
        count = populated_market.verify_publisher("p_alex")
        assert count >= 1
        assert populated_market.is_publisher_verified("p_alex")

    def test_unverify_publisher(self, populated_market: SkillMarketplace) -> None:
        populated_market.verify_publisher("p_alex")
        populated_market.verify_publisher("p_alex", verified=False)
        assert not populated_market.is_publisher_verified("p_alex")

    def test_verify_unknown_publisher(self, market: SkillMarketplace) -> None:
        count = market.verify_publisher("nonexistent")
        assert count == 0


# ============================================================================
# Emergency Recall
# ============================================================================


class TestEmergencyRecall:
    def test_recall_skill(self, populated_market: SkillMarketplace) -> None:
        initial = populated_market.listing_count
        result = populated_market.recall_skill("bu_calc", reason="malware")
        assert result["recalled"] is True
        assert result["reason"] == "malware"
        assert populated_market.listing_count == initial - 1
        assert populated_market.get_listing("bu_calc") is None

    def test_recall_nonexistent(self, market: SkillMarketplace) -> None:
        result = market.recall_skill("nope")
        assert result["recalled"] is False

    def test_recall_with_publisher_ban(self, populated_market: SkillMarketplace) -> None:
        result = populated_market.recall_skill(
            "bu_calc", reason="abuse", ban_publisher=True,
        )
        assert result["recalled"] is True
        assert result["publisher_banned"] is True
        # All skills from that publisher should be removed
        publisher_id = result["publisher_id"]
        remaining = [l for l in populated_market._listings.values()
                     if l.publisher_id == publisher_id]
        assert len(remaining) == 0

    def test_banned_publisher_cannot_publish(self, populated_market: SkillMarketplace) -> None:
        # bu_calc has publisher_id="p_alex"
        populated_market.recall_skill("bu_calc", ban_publisher=True)
        with pytest.raises(ValueError, match="banned"):
            populated_market.publish(SkillListing(
                package_id="new_skill", name="New", description="D",
                publisher_id="p_alex",
            ))

    def test_recall_log(self, populated_market: SkillMarketplace) -> None:
        populated_market.recall_skill("bu_calc", reason="test")
        log = populated_market.recall_log
        assert len(log) == 1
        assert log[0]["reason"] == "test"


# ============================================================================
# Permission-Display & Security-Scan
# ============================================================================


class TestPermissionsAndSecurity:
    def test_set_and_get_permissions(self, populated_market: SkillMarketplace) -> None:
        perms = ["filesystem:read", "network:allow"]
        assert populated_market.set_permissions("bu_calc", perms) is True
        assert populated_market.get_permissions("bu_calc") == perms

    def test_get_permissions_nonexistent(self, market: SkillMarketplace) -> None:
        assert market.get_permissions("nope") == []

    def test_set_scan_result(self, populated_market: SkillMarketplace) -> None:
        assert populated_market.set_scan_result(
            "bu_calc", passed=True, scan_report={"issues": 0},
        ) is True
        listing = populated_market.get_listing("bu_calc")
        assert listing is not None
        assert listing.security_scan_passed is True

    def test_needs_scan_new_skill(self, populated_market: SkillMarketplace) -> None:
        assert populated_market.needs_scan("bu_calc") is True

    def test_needs_scan_after_pass(self, populated_market: SkillMarketplace) -> None:
        populated_market.set_scan_result("bu_calc", passed=True)
        assert populated_market.needs_scan("bu_calc") is False
