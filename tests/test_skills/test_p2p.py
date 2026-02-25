"""Tests für P2P Skill Distribution.

Testet:
  - PeerNode: Identität, Heartbeat, Staleness
  - PeerRegistry: Register, Heartbeat, Cleanup, Eviction
  - SkillIndex: Publish, Search, Versioning, Merge
  - ReputationTracker: Scoring, Auto-Quarantäne, Trust-Levels
  - SubscriptionFeed: Abonnements, Benachrichtigungen
  - SkillExchange: End-to-End Publish → Search → Install → Feedback
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from jarvis.skills.package import (
    PackageSigner,
    SkillManifest,
    SkillPackage,
    TrustLevel,
)
from jarvis.skills.p2p import (
    IndexEntry,
    PeerNode,
    PeerRegistry,
    ReputationEvent,
    ReputationTracker,
    SkillExchange,
    SkillIndex,
    Subscription,
    SubscriptionFeed,
)


# ============================================================================
# PeerNode
# ============================================================================


class TestPeerNode:
    """Peer-Identität und Status."""

    def test_basic_creation(self) -> None:
        peer = PeerNode(peer_id="abc123", display_name="Jarvis-1")
        assert peer.peer_id == "abc123"
        assert peer.is_online
        assert not peer.is_stale

    def test_touch_updates_timestamp(self) -> None:
        peer = PeerNode(peer_id="test")
        old_ts = peer.last_seen
        peer.touch()
        assert peer.last_seen >= old_ts

    def test_stale_detection(self) -> None:
        peer = PeerNode(peer_id="old")
        peer.last_seen = (
            datetime.now(timezone.utc) - timedelta(hours=2)
        ).isoformat()
        assert peer.is_stale


# ============================================================================
# PeerRegistry
# ============================================================================


class TestPeerRegistry:
    """Verwaltung bekannter Peers."""

    @pytest.fixture
    def registry(self) -> PeerRegistry:
        return PeerRegistry(max_peers=5)

    def test_register_new_peer(self, registry: PeerRegistry) -> None:
        peer = PeerNode(peer_id="p1", display_name="Peer 1")
        assert registry.register(peer) is True
        assert registry.peer_count == 1

    def test_register_duplicate(self, registry: PeerRegistry) -> None:
        peer = PeerNode(peer_id="p1")
        registry.register(peer)
        assert registry.register(peer) is False  # Update, nicht neu
        assert registry.peer_count == 1

    def test_get_peer(self, registry: PeerRegistry) -> None:
        peer = PeerNode(peer_id="p1", display_name="Test")
        registry.register(peer)
        found = registry.get("p1")
        assert found is not None
        assert found.display_name == "Test"

    def test_remove_peer(self, registry: PeerRegistry) -> None:
        registry.register(PeerNode(peer_id="p1"))
        assert registry.remove("p1") is True
        assert registry.remove("p1") is False  # Schon entfernt

    def test_heartbeat(self, registry: PeerRegistry) -> None:
        registry.register(PeerNode(peer_id="p1"))
        assert registry.heartbeat("p1") is True
        assert registry.heartbeat("unknown") is False

    def test_list_online(self, registry: PeerRegistry) -> None:
        registry.register(PeerNode(peer_id="p1"))
        registry.register(PeerNode(peer_id="p2"))
        online = registry.list_online()
        assert len(online) == 2

    def test_cleanup_stale(self, registry: PeerRegistry) -> None:
        old_peer = PeerNode(peer_id="old")
        old_peer.last_seen = (
            datetime.now(timezone.utc) - timedelta(hours=5)
        ).isoformat()
        registry._peers["old"] = old_peer

        registry.register(PeerNode(peer_id="fresh"))
        removed = registry.cleanup_stale(max_age_seconds=3600)
        assert removed == 1
        assert registry.peer_count == 1

    def test_eviction_on_max_peers(self) -> None:
        registry = PeerRegistry(max_peers=3)
        for i in range(5):
            p = PeerNode(peer_id=f"p{i}")
            if i < 2:
                p.last_seen = (
                    datetime.now(timezone.utc) - timedelta(hours=3)
                ).isoformat()
            registry.register(p)

        assert registry.peer_count <= 5


# ============================================================================
# SkillIndex
# ============================================================================


class TestSkillIndex:
    """Verteiltes Skill-Verzeichnis."""

    @pytest.fixture
    def index(self) -> SkillIndex:
        idx = SkillIndex()

        m1 = SkillManifest(
            name="bu_vergleich", version="1.0.0",
            description="BU-Tarifvergleich für Versicherungen",
            author="Alexander",
            trigger_keywords=["BU", "Berufsunfähigkeit", "Tarif"],
            category="insurance",
        )
        idx.publish(IndexEntry(
            package_id="bu_vergleich@1.0.0:abc",
            manifest=m1, publisher_id="peer_1",
        ))

        m2 = SkillManifest(
            name="web_scraper", version="2.0.0",
            description="Webseiten-Scraping",
            author="Dev",
            trigger_keywords=["scrape", "website", "crawl"],
            category="tools",
        )
        idx.publish(IndexEntry(
            package_id="web_scraper@2.0.0:def",
            manifest=m2, publisher_id="peer_2",
        ))

        return idx

    def test_entry_count(self, index: SkillIndex) -> None:
        assert index.entry_count == 2

    def test_search_by_keyword(self, index: SkillIndex) -> None:
        results = index.search("BU Tarif")
        assert len(results) >= 1
        assert results[0].manifest.name == "bu_vergleich"

    def test_search_by_category(self, index: SkillIndex) -> None:
        results = index.search(category="insurance")
        assert len(results) == 1
        assert results[0].manifest.name == "bu_vergleich"

    def test_search_by_description(self, index: SkillIndex) -> None:
        results = index.search("Versicherung")
        assert len(results) >= 1

    def test_search_empty_returns_all(self, index: SkillIndex) -> None:
        results = index.search()
        assert len(results) == 2

    def test_search_no_match(self, index: SkillIndex) -> None:
        results = index.search("Quantenphysik")
        assert len(results) == 0

    def test_get_versions(self, index: SkillIndex) -> None:
        # Zweite Version hinzufügen
        m = SkillManifest(
            name="bu_vergleich", version="1.1.0",
            description="Update", author="Alexander",
            category="insurance",
        )
        index.publish(IndexEntry(
            package_id="bu_vergleich@1.1.0:xyz",
            manifest=m, publisher_id="peer_1",
        ))

        versions = index.get_versions("bu_vergleich")
        assert len(versions) == 2
        assert versions[0].manifest.version == "1.1.0"  # Neueste zuerst

    def test_get_latest(self, index: SkillIndex) -> None:
        latest = index.get_latest("bu_vergleich")
        assert latest is not None
        assert latest.manifest.name == "bu_vergleich"

    def test_get_latest_nonexistent(self, index: SkillIndex) -> None:
        assert index.get_latest("nonexistent") is None

    def test_remove(self, index: SkillIndex) -> None:
        assert index.remove("bu_vergleich@1.0.0:abc") is True
        assert index.entry_count == 1
        assert index.remove("bu_vergleich@1.0.0:abc") is False

    def test_merge_from(self, index: SkillIndex) -> None:
        m = SkillManifest(
            name="new_skill", version="1.0.0",
            description="Neuer Skill", author="Remote",
            category="tools",
        )
        remote_entries = [
            IndexEntry(
                package_id="new_skill@1.0.0:ghi",
                manifest=m, publisher_id="peer_3",
            ),
        ]

        new_count = index.merge_from(remote_entries)
        assert new_count == 1
        assert index.entry_count == 3

    def test_merge_deduplicate(self, index: SkillIndex) -> None:
        # Gleichen Entry nochmal mergen
        m = SkillManifest(
            name="bu_vergleich", version="1.0.0",
            description="Duplikat", author="Alexander",
            category="insurance",
        )
        duplicate = [
            IndexEntry(
                package_id="bu_vergleich@1.0.0:abc",
                manifest=m, publisher_id="peer_1",
            ),
        ]
        new_count = index.merge_from(duplicate)
        assert new_count == 0


# ============================================================================
# ReputationTracker
# ============================================================================


class TestReputationTracker:
    """Vertrauenssystem."""

    @pytest.fixture
    def tracker(self) -> ReputationTracker:
        return ReputationTracker()

    def test_initial_score(self, tracker: ReputationTracker) -> None:
        assert tracker.get_score("unknown") == 0.0

    def test_install_success_increases(self, tracker: ReputationTracker) -> None:
        score = tracker.record("pkg1", ReputationEvent.INSTALL_SUCCESS)
        assert score == 1.0

    def test_install_failure_decreases(self, tracker: ReputationTracker) -> None:
        score = tracker.record("pkg1", ReputationEvent.INSTALL_FAILURE)
        assert score == -2.0

    def test_malware_report_heavy_penalty(self, tracker: ReputationTracker) -> None:
        score = tracker.record("evil_pkg", ReputationEvent.MALWARE_REPORT)
        assert score == -5.0

    def test_auto_quarantine_package(self, tracker: ReputationTracker) -> None:
        # Package: Threshold = -3
        tracker.record("pkg@1.0:abc", ReputationEvent.INSTALL_FAILURE)
        tracker.record("pkg@1.0:abc", ReputationEvent.NEGATIVE_FEEDBACK)
        assert tracker.is_quarantined("pkg@1.0:abc")

    def test_auto_quarantine_peer(self, tracker: ReputationTracker) -> None:
        # Peer: Threshold = -5
        tracker.record("peer_evil", ReputationEvent.MALWARE_REPORT)
        assert tracker.is_quarantined("peer_evil")

    def test_is_trusted(self, tracker: ReputationTracker) -> None:
        tracker.record("good_peer", ReputationEvent.INSTALL_SUCCESS)
        assert tracker.is_trusted("good_peer")

        tracker.record("bad_peer", ReputationEvent.MALWARE_REPORT)
        assert not tracker.is_trusted("bad_peer")

    def test_trust_level(self, tracker: ReputationTracker) -> None:
        for _ in range(5):
            tracker.record("popular", ReputationEvent.INSTALL_SUCCESS)
            tracker.record("popular", ReputationEvent.POSITIVE_FEEDBACK)

        profile = tracker.get_profile("popular")
        assert profile is not None
        assert profile.trust_level == TrustLevel.VERIFIED

    def test_manual_quarantine_and_release(self, tracker: ReputationTracker) -> None:
        tracker.quarantine("suspect")
        assert tracker.is_quarantined("suspect")

        tracker.release("suspect")
        assert not tracker.is_quarantined("suspect")

    def test_top_trusted(self, tracker: ReputationTracker) -> None:
        tracker.record("a", ReputationEvent.INSTALL_SUCCESS)
        tracker.record("b", ReputationEvent.INSTALL_SUCCESS)
        tracker.record("b", ReputationEvent.INSTALL_SUCCESS)

        top = tracker.top_trusted(2)
        assert len(top) == 2
        assert top[0].entity_id == "b"  # Höherer Score

    def test_quarantined_list(self, tracker: ReputationTracker) -> None:
        tracker.quarantine("bad1")
        tracker.quarantine("bad2")
        assert len(tracker.quarantined_list()) == 2


# ============================================================================
# SubscriptionFeed
# ============================================================================


class TestSubscriptionFeed:
    """Skill-Abonnements und Benachrichtigungen."""

    @pytest.fixture
    def feed(self) -> SubscriptionFeed:
        return SubscriptionFeed()

    def test_subscribe(self, feed: SubscriptionFeed) -> None:
        sub = feed.subscribe("peer_1", category="insurance")
        assert sub.subscriber_id == "peer_1"
        assert feed.subscription_count == 1

    def test_notification_on_matching_category(self, feed: SubscriptionFeed) -> None:
        feed.subscribe("peer_1", category="insurance")

        entry = IndexEntry(
            package_id="new_bu@1.0:abc",
            manifest=SkillManifest(
                name="new_bu", version="1.0.0",
                description="Neue BU", author="a", category="insurance",
            ),
            publisher_id="peer_2",
        )
        notified = feed.check_new_entry(entry)
        assert "peer_1" in notified
        assert feed.notification_count == 1

    def test_no_notification_wrong_category(self, feed: SubscriptionFeed) -> None:
        feed.subscribe("peer_1", category="tools")

        entry = IndexEntry(
            package_id="pkg@1.0:abc",
            manifest=SkillManifest(
                name="pkg", version="1.0.0",
                description="x", author="a", category="insurance",
            ),
            publisher_id="peer_2",
        )
        notified = feed.check_new_entry(entry)
        assert len(notified) == 0

    def test_keyword_subscription(self, feed: SubscriptionFeed) -> None:
        feed.subscribe("peer_1", keyword="BU")

        entry = IndexEntry(
            package_id="bu_new@1.0:abc",
            manifest=SkillManifest(
                name="bu_new", version="1.0.0",
                description="x", author="a",
                trigger_keywords=["BU", "Tarif"],
            ),
            publisher_id="peer_2",
        )
        notified = feed.check_new_entry(entry)
        assert "peer_1" in notified

    def test_author_subscription(self, feed: SubscriptionFeed) -> None:
        feed.subscribe("peer_1", author_id="trusted_dev")

        entry = IndexEntry(
            package_id="pkg@1.0:abc",
            manifest=SkillManifest(
                name="pkg", version="1.0.0", description="x", author="a",
            ),
            publisher_id="trusted_dev",
        )
        notified = feed.check_new_entry(entry)
        assert "peer_1" in notified

    def test_unsubscribe_all(self, feed: SubscriptionFeed) -> None:
        feed.subscribe("peer_1", category="a")
        feed.subscribe("peer_1", keyword="b")
        feed.subscribe("peer_2", category="c")

        removed = feed.unsubscribe_all("peer_1")
        assert removed == 2
        assert feed.subscription_count == 1

    def test_clear_notifications(self, feed: SubscriptionFeed) -> None:
        feed.subscribe("peer_1", category="test")
        entry = IndexEntry(
            package_id="x@1.0:abc",
            manifest=SkillManifest(
                name="xxx", version="1.0.0", description="x",
                author="a", category="test",
            ),
            publisher_id="p",
        )
        feed.check_new_entry(entry)
        assert feed.notification_count == 1

        feed.clear_notifications()
        assert feed.notification_count == 0


# ============================================================================
# SkillExchange (End-to-End)
# ============================================================================


class TestSkillExchange:
    """End-to-End: Publish → Search → Install → Feedback."""

    @pytest.fixture
    def exchange(self, tmp_path: Path) -> SkillExchange:
        ex = SkillExchange(
            tmp_path / "skills",
            private_key="test_exchange_key",
            require_signatures=False,
        )
        ex.set_identity(PeerNode(peer_id="local", display_name="Test-Node"))
        return ex

    def test_publish(self, exchange: SkillExchange) -> None:
        manifest = SkillManifest(
            name="my_skill", version="1.0.0",
            description="Ein toller Skill", author="Alexander",
            trigger_keywords=["toll"],
            category="general",
        )
        pkg = exchange.publish(manifest, "x = 1\ny = 2")

        assert pkg is not None
        assert exchange.index.entry_count == 1

    def test_search_after_publish(self, exchange: SkillExchange) -> None:
        manifest = SkillManifest(
            name="bu_helper", version="1.0.0",
            description="BU-Hilfe", author="Alexander",
            trigger_keywords=["BU", "Berufsunfähigkeit"],
            category="insurance",
        )
        exchange.publish(manifest, "result = 'BU ok'")

        results = exchange.search("BU")
        assert len(results) == 1
        assert results[0].manifest.name == "bu_helper"

    def test_install_after_publish(self, exchange: SkillExchange) -> None:
        manifest = SkillManifest(
            name="installable", version="1.0.0",
            description="Installierbar", author="a",
        )
        pkg = exchange.publish(manifest, "x = 42")
        assert pkg is not None

        result = exchange.install(pkg.package_id)
        assert result.success
        assert exchange.installer.installed_count == 1

    def test_install_quarantined_blocked(self, exchange: SkillExchange) -> None:
        manifest = SkillManifest(
            name="quarantined_pkg", version="1.0.0",
            description="Böse", author="a",
        )
        pkg = exchange.publish(manifest, "x = 1")
        exchange.report_malware(pkg.package_id)

        result = exchange.install(pkg.package_id)
        assert not result.success
        assert "quarantiniert" in result.message

    def test_feedback_updates_reputation(self, exchange: SkillExchange) -> None:
        manifest = SkillManifest(
            name="rated_skill", version="1.0.0",
            description="Bewertbar", author="a",
        )
        pkg = exchange.publish(manifest, "x = 1")

        exchange.report_feedback(pkg.package_id, positive=True)
        score = exchange.reputation.get_score(pkg.package_id)
        assert score > 0

        exchange.report_feedback(pkg.package_id, positive=False)
        new_score = exchange.reputation.get_score(pkg.package_id)
        assert new_score < score

    def test_malware_report_quarantines(self, exchange: SkillExchange) -> None:
        manifest = SkillManifest(
            name="evil_skill", version="1.0.0",
            description="Böser Skill", author="Hacker",
        )
        pkg = exchange.publish(manifest, "x = 1")
        exchange.report_malware(pkg.package_id)

        assert exchange.reputation.is_quarantined(pkg.package_id)

    def test_search_filters_quarantined(self, exchange: SkillExchange) -> None:
        m1 = SkillManifest(
            name="good_skill", version="1.0.0",
            description="Gut", author="a", category="test",
        )
        m2 = SkillManifest(
            name="bad_skill", version="1.0.0",
            description="Schlecht", author="a", category="test",
        )
        exchange.publish(m1, "x = 1")
        pkg2 = exchange.publish(m2, "y = 2")
        exchange.report_malware(pkg2.package_id)

        results = exchange.search(category="test")
        names = [r.manifest.name for r in results]
        assert "good_skill" in names
        assert "bad_skill" not in names

    def test_peer_sync(self, tmp_path: Path) -> None:
        ex1 = SkillExchange(tmp_path / "s1", require_signatures=False)
        ex2 = SkillExchange(tmp_path / "s2", require_signatures=False)

        # Ex1 publiziert
        m = SkillManifest(
            name="shared_skill", version="1.0.0",
            description="Geteilt", author="a",
        )
        ex1.publish(m, "x = 1")

        # Ex2 synchronisiert
        entries = ex1.get_index_for_sync()
        new = ex2.sync_with_peer(entries)
        assert new == 1
        assert ex2.index.entry_count == 1

    def test_subscription_with_exchange(self, exchange: SkillExchange) -> None:
        exchange.subscriptions.subscribe("local", category="insurance")

        manifest = SkillManifest(
            name="bu_auto", version="1.0.0",
            description="Auto BU", author="a",
            category="insurance",
        )
        exchange.publish(manifest, "x = 1")

        notifs = exchange.subscriptions.get_notifications()
        assert len(notifs) >= 1

    def test_stats(self, exchange: SkillExchange) -> None:
        stats = exchange.stats()
        assert "peers_total" in stats
        assert "index_entries" in stats
        assert "installed" in stats
        assert "quarantined" in stats

    def test_publish_dangerous_rejected(self, exchange: SkillExchange) -> None:
        manifest = SkillManifest(
            name="dangerous_pub", version="1.0.0",
            description="Böse", author="a",
        )
        pkg = exchange.publish(manifest, "eval('evil')")
        assert pkg is None  # Wird vom CodeAnalyzer blockiert
