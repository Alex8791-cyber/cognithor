"""P2P Skill Distribution: Verteiltes Netzwerk für Jarvis-Skills.

Ermöglicht die dezentrale Verteilung von Skill-Paketen ohne
zentralen Server. Jede Jarvis-Instanz kann Skills publizieren,
suchen und herunterladen.

Komponenten:

  1. PeerNode: Lokaler Knoten mit Identität und Schlüsselpaar.
     Jede Jarvis-Instanz hat einen eindeutigen Peer.

  2. PeerRegistry: Verwaltung bekannter Peers mit Heartbeat
     und Cleanup von inaktiven Nodes.

  3. SkillIndex: Verteiltes Verzeichnis aller verfügbaren Skills.
     DHT-ähnliche Struktur: Jeder Peer kennt einen Teil des Index.
     Suche über Keyword-Matching + Kategorien.

  4. ReputationTracker: Vertrauenssystem basierend auf:
     - Erfolgreiche Installationen (+1)
     - Fehlgeschlagene Installationen (-2)
     - Positives User-Feedback (+1)
     - Negatives Feedback / Malware-Report (-5)
     - Alter des Pakets (ältere = mehr Vertrauen)

  5. SkillExchange: Orchestriert den kompletten Workflow:
     Publish → Sign → Index → Search → Download → Verify → Install

  6. SubscriptionFeed: Automatische Benachrichtigung über neue
     Skills in abonnierten Kategorien.

Sicherheit:
  - Alle Pakete MÜSSEN signiert sein
  - Code-Analyse vor Installation (CodeAnalyzer)
  - Reputation-basiertes Vertrauen
  - Sandbox-Isolation pro Skill
  - Schnelle Isolation von Malware-Paketen (Quarantäne)

Bibel-Referenz: §6.5 (P2P Skill Distribution)
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from jarvis.skills.package import (
    AnalysisVerdict,
    CodeAnalyzer,
    InstallResult,
    PackageBuilder,
    PackageInstaller,
    PackageSigner,
    PackageSignature,
    SkillManifest,
    SkillPackage,
    TrustLevel,
)
from jarvis.skills.circles import CircleManager
from jarvis.skills.marketplace import SkillMarketplace
from jarvis.skills.updater import SkillUpdater

logger = logging.getLogger("jarvis.skills.p2p")


# ============================================================================
# Peer Identity
# ============================================================================


@dataclass
class PeerNode:
    """Ein Knoten im P2P-Netzwerk.

    Jede Jarvis-Instanz ist ein PeerNode mit eindeutiger ID
    und optionalem Schlüsselpaar für Signierung.
    """

    peer_id: str  # SHA-256[:16] vom öffentlichen Schlüssel
    display_name: str = ""
    address: str = ""  # host:port oder URL
    public_key: str = ""  # Hex-encodierter öffentlicher Schlüssel

    # Status
    last_seen: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    is_online: bool = True
    skills_published: int = 0

    @property
    def is_stale(self) -> bool:
        """Peer gilt als stale wenn >1h nicht gesehen."""
        try:
            last = datetime.fromisoformat(self.last_seen)
            age = (datetime.now(timezone.utc) - last).total_seconds()
            return age > 3600
        except (ValueError, TypeError):
            return True

    def touch(self) -> None:
        """Aktualisiert last_seen."""
        self.last_seen = datetime.now(timezone.utc).isoformat()
        self.is_online = True


# ============================================================================
# Peer Registry
# ============================================================================


class PeerRegistry:
    """Verwaltung bekannter Peers im Netzwerk.

    Hält eine Liste aktiver Peers, entfernt inaktive,
    und bietet Discovery-Funktionen.
    """

    def __init__(self, max_peers: int = 200) -> None:
        self._peers: dict[str, PeerNode] = {}
        self._max_peers = max_peers

    @property
    def peer_count(self) -> int:
        return len(self._peers)

    @property
    def online_count(self) -> int:
        return sum(1 for p in self._peers.values() if p.is_online)

    def register(self, peer: PeerNode) -> bool:
        """Registriert oder aktualisiert einen Peer.

        Returns:
            True wenn neu registriert.
        """
        is_new = peer.peer_id not in self._peers
        peer.touch()
        self._peers[peer.peer_id] = peer

        # Max-Limit: Älteste offline Peers entfernen
        if len(self._peers) > self._max_peers:
            self._evict_stale()

        return is_new

    def get(self, peer_id: str) -> PeerNode | None:
        return self._peers.get(peer_id)

    def remove(self, peer_id: str) -> bool:
        return self._peers.pop(peer_id, None) is not None

    def list_online(self) -> list[PeerNode]:
        """Alle aktuell erreichbaren Peers."""
        return [p for p in self._peers.values() if p.is_online and not p.is_stale]

    def list_all(self) -> list[PeerNode]:
        return list(self._peers.values())

    def heartbeat(self, peer_id: str) -> bool:
        """Aktualisiert den Zeitstempel eines Peers.

        Returns:
            True wenn Peer bekannt.
        """
        peer = self._peers.get(peer_id)
        if peer:
            peer.touch()
            return True
        return False

    def cleanup_stale(self, max_age_seconds: int = 7200) -> int:
        """Entfernt Peers die zu lange nicht gesehen wurden.

        Args:
            max_age_seconds: Maximales Alter (Default: 2h).

        Returns:
            Anzahl entfernter Peers.
        """
        to_remove = []
        now = datetime.now(timezone.utc)

        for pid, peer in self._peers.items():
            try:
                last = datetime.fromisoformat(peer.last_seen)
                if (now - last).total_seconds() > max_age_seconds:
                    to_remove.append(pid)
            except (ValueError, TypeError):
                to_remove.append(pid)

        for pid in to_remove:
            self._peers.pop(pid, None)

        return len(to_remove)

    def _evict_stale(self) -> None:
        """Entfernt die ältesten stale Peers bis unter max_peers."""
        stale = sorted(
            [(pid, p) for pid, p in self._peers.items() if p.is_stale],
            key=lambda x: x[1].last_seen,
        )
        while len(self._peers) > self._max_peers and stale:
            pid, _ = stale.pop(0)
            self._peers.pop(pid, None)


# ============================================================================
# Skill Index (DHT-ähnlich)
# ============================================================================


@dataclass
class IndexEntry:
    """Ein Eintrag im verteilten Skill-Index."""

    package_id: str  # name@version:hash
    manifest: SkillManifest
    publisher_id: str  # PeerNode.peer_id
    published_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    download_count: int = 0
    signature_valid: bool = False


class SkillIndex:
    """Verteiltes Verzeichnis verfügbarer Skills.

    Jeder Peer hält eine lokale Kopie des Index.
    Synchronisation erfolgt über Peer-Exchange:
      - Beim Verbinden: Index-Diff austauschen
      - Beim Publizieren: Broadcast an bekannte Peers
    """

    def __init__(self) -> None:
        self._entries: dict[str, IndexEntry] = {}  # package_id → IndexEntry
        self._by_name: dict[str, list[str]] = defaultdict(list)  # name → [package_ids]
        self._by_category: dict[str, list[str]] = defaultdict(list)
        self._by_keyword: dict[str, list[str]] = defaultdict(list)

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    def publish(self, entry: IndexEntry) -> None:
        """Fügt ein Skill-Paket dem Index hinzu.

        Args:
            entry: Index-Eintrag.
        """
        self._entries[entry.package_id] = entry
        self._by_name[entry.manifest.name].append(entry.package_id)
        self._by_category[entry.manifest.category].append(entry.package_id)

        for kw in entry.manifest.trigger_keywords:
            self._by_keyword[kw.lower()].append(entry.package_id)

    def search(
        self,
        query: str = "",
        *,
        category: str = "",
        max_results: int = 20,
    ) -> list[IndexEntry]:
        """Sucht im Index nach Skills.

        Args:
            query: Suchtext (matched gegen Name, Keywords, Beschreibung).
            category: Optional Kategorie-Filter.
            max_results: Maximale Ergebnisse.

        Returns:
            Sortierte Liste nach Relevanz.
        """
        candidates: dict[str, float] = {}

        if category:
            for pid in self._by_category.get(category, []):
                candidates[pid] = candidates.get(pid, 0) + 0.5

        if query:
            query_lower = query.lower()
            query_words = set(query_lower.split())

            for pid, entry in self._entries.items():
                score = 0.0

                # Name-Match
                if query_lower in entry.manifest.name:
                    score += 1.0
                elif any(w in entry.manifest.name for w in query_words):
                    score += 0.6

                # Keyword-Match
                for kw in entry.manifest.trigger_keywords:
                    if query_lower in kw.lower() or kw.lower() in query_lower:
                        score += 0.8
                        break

                # Beschreibung
                desc_lower = entry.manifest.description.lower()
                overlap = sum(1 for w in query_words if w in desc_lower)
                score += overlap * 0.3

                if score > 0:
                    candidates[pid] = candidates.get(pid, 0) + score

        if not query and not category:
            # Alles zurückgeben, sortiert nach Datum
            entries = sorted(
                self._entries.values(),
                key=lambda e: e.published_at,
                reverse=True,
            )
            return entries[:max_results]

        # Sortieren nach Score
        ranked = sorted(
            candidates.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        results = []
        for pid, _ in ranked[:max_results]:
            entry = self._entries.get(pid)
            if entry:
                results.append(entry)

        return results

    def get_versions(self, name: str) -> list[IndexEntry]:
        """Alle Versionen eines Skills.

        Returns:
            Sortiert nach Version (neueste zuerst).
        """
        pids = self._by_name.get(name, [])
        entries = [self._entries[pid] for pid in pids if pid in self._entries]
        def _semver_key(e: IndexEntry) -> tuple[int, ...]:
            try:
                return tuple(int(x) for x in e.manifest.version.split("."))
            except (ValueError, AttributeError):
                return (0,)
        entries.sort(key=_semver_key, reverse=True)
        return entries

    def get_latest(self, name: str) -> IndexEntry | None:
        """Neueste Version eines Skills."""
        versions = self.get_versions(name)
        return versions[0] if versions else None

    def remove(self, package_id: str) -> bool:
        """Entfernt ein Paket aus dem Index."""
        entry = self._entries.pop(package_id, None)
        if not entry:
            return False

        # Index-Listen aufräumen
        name = entry.manifest.name
        if name in self._by_name:
            self._by_name[name] = [p for p in self._by_name[name] if p != package_id]
        cat = entry.manifest.category
        if cat in self._by_category:
            self._by_category[cat] = [p for p in self._by_category[cat] if p != package_id]

        return True

    def merge_from(self, other_entries: list[IndexEntry]) -> int:
        """Merged Einträge eines anderen Peers in den lokalen Index.

        Returns:
            Anzahl neuer Einträge.
        """
        new_count = 0
        for entry in other_entries:
            if entry.package_id not in self._entries:
                self.publish(entry)
                new_count += 1
        return new_count


# ============================================================================
# Reputation Tracker
# ============================================================================


class ReputationEvent(Enum):
    """Reputation-beeinflussende Ereignisse."""

    INSTALL_SUCCESS = "install_success"  # +1
    INSTALL_FAILURE = "install_failure"  # -2
    POSITIVE_FEEDBACK = "positive_feedback"  # +1
    NEGATIVE_FEEDBACK = "negative_feedback"  # -3
    MALWARE_REPORT = "malware_report"  # -5
    SKILL_PUBLISHED = "skill_published"  # +0.5


_REPUTATION_SCORES: dict[ReputationEvent, float] = {
    ReputationEvent.INSTALL_SUCCESS: 1.0,
    ReputationEvent.INSTALL_FAILURE: -2.0,
    ReputationEvent.POSITIVE_FEEDBACK: 1.0,
    ReputationEvent.NEGATIVE_FEEDBACK: -3.0,
    ReputationEvent.MALWARE_REPORT: -5.0,
    ReputationEvent.SKILL_PUBLISHED: 0.5,
}


@dataclass
class ReputationProfile:
    """Reputationsprofil eines Peers oder Pakets."""

    entity_id: str  # Peer-ID oder Package-ID
    score: float = 0.0
    events: list[tuple[str, float]] = field(default_factory=list)  # (event, delta)
    quarantined: bool = False  # Isoliert nach Malware-Report

    @property
    def trust_level(self) -> TrustLevel:
        if self.quarantined:
            return TrustLevel.UNKNOWN
        if self.score >= 10:
            return TrustLevel.VERIFIED
        if self.score >= 3:
            return TrustLevel.COMMUNITY
        return TrustLevel.UNKNOWN


class ReputationTracker:
    """Vertrauenssystem für Peers und Pakete.

    Tracks:
      - Peer-Reputation (basierend auf publizierten Paketen)
      - Paket-Reputation (basierend auf Installationen + Feedback)

    Quarantäne:
      - Peers mit Score < -5 werden automatisch quarantiniert
      - Pakete mit Score < -3 werden quarantiniert
    """

    def __init__(
        self,
        *,
        peer_quarantine_threshold: float = -5.0,
        package_quarantine_threshold: float = -3.0,
    ) -> None:
        self._profiles: dict[str, ReputationProfile] = {}
        self._peer_threshold = peer_quarantine_threshold
        self._package_threshold = package_quarantine_threshold

    def record(self, entity_id: str, event: ReputationEvent) -> float:
        """Zeichnet ein Reputations-Ereignis auf.

        Args:
            entity_id: Peer-ID oder Package-ID.
            event: Reputations-Ereignis.

        Returns:
            Neuer Score.
        """
        profile = self._profiles.setdefault(
            entity_id, ReputationProfile(entity_id=entity_id),
        )

        delta = _REPUTATION_SCORES[event]
        profile.score += delta
        profile.events.append((event.value, delta))

        # Auto-Quarantäne prüfen
        if ":" in entity_id:
            # Package-ID (name@version:hash)
            if profile.score <= self._package_threshold:
                profile.quarantined = True
                logger.warning("Paket quarantiniert: %s (score=%.1f)", entity_id, profile.score)
        else:
            # Peer-ID
            if profile.score <= self._peer_threshold:
                profile.quarantined = True
                logger.warning("Peer quarantiniert: %s (score=%.1f)", entity_id, profile.score)

        return profile.score

    def get_score(self, entity_id: str) -> float:
        profile = self._profiles.get(entity_id)
        return profile.score if profile else 0.0

    def get_profile(self, entity_id: str) -> ReputationProfile | None:
        return self._profiles.get(entity_id)

    def is_trusted(self, entity_id: str) -> bool:
        """Prüft ob ein Entity (Peer/Paket) vertrauenswürdig ist."""
        profile = self._profiles.get(entity_id)
        if not profile:
            return True  # Unbekannt = neutral
        return not profile.quarantined and profile.score >= 0

    def is_quarantined(self, entity_id: str) -> bool:
        profile = self._profiles.get(entity_id)
        return profile.quarantined if profile else False

    def quarantine(self, entity_id: str) -> None:
        """Manuelles Quarantinieren."""
        profile = self._profiles.setdefault(
            entity_id, ReputationProfile(entity_id=entity_id),
        )
        profile.quarantined = True

    def release(self, entity_id: str) -> None:
        """Quarantäne aufheben."""
        profile = self._profiles.get(entity_id)
        if profile:
            profile.quarantined = False

    def top_trusted(self, n: int = 10) -> list[ReputationProfile]:
        """Die vertrauenswürdigsten Entities."""
        ranked = sorted(
            [p for p in self._profiles.values() if not p.quarantined],
            key=lambda p: p.score,
            reverse=True,
        )
        return ranked[:n]

    def quarantined_list(self) -> list[ReputationProfile]:
        return [p for p in self._profiles.values() if p.quarantined]


# ============================================================================
# Subscription Feed
# ============================================================================


@dataclass
class Subscription:
    """Ein Skill-Abonnement für automatische Updates."""

    subscriber_id: str  # Eigene Peer-ID
    category: str = ""  # Abonnierte Kategorie
    keyword: str = ""  # Abonniertes Keyword
    author_id: str = ""  # Abonnierter Herausgeber
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )


class SubscriptionFeed:
    """Automatische Benachrichtigungen über neue Skills.

    Peers können Kategorien, Keywords oder Herausgeber abonnieren.
    Neue Pakete werden gegen Abonnements gematched.
    """

    def __init__(self) -> None:
        self._subscriptions: list[Subscription] = []
        self._notifications: list[tuple[str, IndexEntry]] = []  # (sub_desc, entry)

    @property
    def subscription_count(self) -> int:
        return len(self._subscriptions)

    @property
    def notification_count(self) -> int:
        return len(self._notifications)

    def subscribe(
        self,
        subscriber_id: str,
        *,
        category: str = "",
        keyword: str = "",
        author_id: str = "",
    ) -> Subscription:
        """Erstellt ein neues Abonnement.

        Mindestens eines von category, keyword oder author_id muss gesetzt sein.
        """
        sub = Subscription(
            subscriber_id=subscriber_id,
            category=category,
            keyword=keyword,
            author_id=author_id,
        )
        self._subscriptions.append(sub)
        return sub

    def unsubscribe_all(self, subscriber_id: str) -> int:
        """Entfernt alle Abonnements eines Subscribers."""
        before = len(self._subscriptions)
        self._subscriptions = [
            s for s in self._subscriptions if s.subscriber_id != subscriber_id
        ]
        return before - len(self._subscriptions)

    def check_new_entry(self, entry: IndexEntry) -> list[str]:
        """Prüft ob ein neuer Index-Eintrag Abonnements auslöst.

        Args:
            entry: Neuer Skill-Index-Eintrag.

        Returns:
            Liste der benachrichtigten Subscriber-IDs.
        """
        notified: list[str] = []

        for sub in self._subscriptions:
            matched = False

            if sub.category and sub.category == entry.manifest.category:
                matched = True
            if sub.keyword and sub.keyword.lower() in [
                kw.lower() for kw in entry.manifest.trigger_keywords
            ]:
                matched = True
            if sub.author_id and sub.author_id == entry.publisher_id:
                matched = True

            if matched:
                desc = f"Neuer Skill: {entry.manifest.qualified_name}"
                self._notifications.append((desc, entry))
                notified.append(sub.subscriber_id)

        return notified

    def get_notifications(self, limit: int = 50) -> list[tuple[str, IndexEntry]]:
        """Holt die neuesten Benachrichtigungen."""
        return self._notifications[-limit:]

    def clear_notifications(self) -> None:
        self._notifications.clear()


# ============================================================================
# Skill Exchange (Orchestrator)
# ============================================================================


class SkillExchange:
    """Zentrale Orchestrierung des P2P Skill-Austauschs.

    Verbindet alle Komponenten:
      PeerRegistry + SkillIndex + ReputationTracker
      + PackageBuilder + PackageInstaller + SubscriptionFeed

    Usage:
        exchange = SkillExchange(skills_dir=Path("~/.jarvis/skills/p2p"))
        exchange.set_identity(my_peer)

        # Publizieren
        exchange.publish(manifest, code, tests)

        # Suchen + Installieren
        results = exchange.search("BU-Tarifvergleich")
        result = exchange.install(results[0].package_id, package_bytes)

        # Feedback
        exchange.report_feedback("pkg_id", positive=True)
    """

    def __init__(
        self,
        skills_dir: Path,
        *,
        private_key: str = "",
        require_signatures: bool = True,
    ) -> None:
        self._skills_dir = skills_dir
        self._skills_dir.mkdir(parents=True, exist_ok=True)

        # Identität
        self._identity: PeerNode | None = None
        self._signer: PackageSigner | None = None
        if private_key:
            self._signer = PackageSigner(private_key)

        # Komponenten
        self._peers = PeerRegistry()
        self._index = SkillIndex()
        self._reputation = ReputationTracker()
        self._subscriptions = SubscriptionFeed()
        self._analyzer = CodeAnalyzer()
        self._installer = PackageInstaller(
            skills_dir,
            require_signature=require_signatures,
            analyzer=self._analyzer,
            signer=self._signer,
        )
        self._builder = PackageBuilder(signer=self._signer, analyzer=self._analyzer)

        # Paket-Speicher (simuliert P2P-Transfer)
        self._packages: dict[str, SkillPackage] = {}

        # Trusted Circles (Web-of-Trust Ökosystem)
        self._circles = CircleManager()

        # Kuratierter Marketplace
        self._marketplace = SkillMarketplace()

        # Auto-Update-Mechanismus
        self._updater = SkillUpdater()

    # ── Properties ───────────────────────────────────────────────

    @property
    def peers(self) -> PeerRegistry:
        return self._peers

    @property
    def index(self) -> SkillIndex:
        return self._index

    @property
    def reputation(self) -> ReputationTracker:
        return self._reputation

    @property
    def subscriptions(self) -> SubscriptionFeed:
        return self._subscriptions

    @property
    def circles(self) -> CircleManager:
        return self._circles

    @property
    def marketplace(self) -> SkillMarketplace:
        return self._marketplace

    @property
    def updater(self) -> SkillUpdater:
        return self._updater

    @property
    def installer(self) -> PackageInstaller:
        return self._installer

    @property
    def identity(self) -> PeerNode | None:
        return self._identity

    # ── Setup ────────────────────────────────────────────────────

    def set_identity(self, peer: PeerNode) -> None:
        """Setzt die lokale Identität."""
        self._identity = peer
        self._peers.register(peer)

    # ── Publish ──────────────────────────────────────────────────

    def publish(
        self,
        manifest: SkillManifest,
        code: str,
        test_code: str = "",
        documentation: str = "",
    ) -> SkillPackage | None:
        """Publiziert einen neuen Skill im Netzwerk.

        1. Paket bauen + signieren
        2. Im Index registrieren
        3. Subscriptions benachrichtigen

        Returns:
            Fertiges SkillPackage oder None bei Fehler.
        """
        try:
            package = self._builder.build(
                manifest, code, test_code, documentation,
            )
        except ValueError as exc:
            logger.error("Publish fehlgeschlagen: %s", exc)
            return None

        # Im Index registrieren
        publisher_id = self._identity.peer_id if self._identity else "local"
        entry = IndexEntry(
            package_id=package.package_id,
            manifest=manifest,
            publisher_id=publisher_id,
            signature_valid=package.is_signed,
        )
        self._index.publish(entry)

        # Paket lokal speichern
        self._packages[package.package_id] = package

        # Reputation aufbauen
        self._reputation.record(publisher_id, ReputationEvent.SKILL_PUBLISHED)

        # Subscriptions prüfen
        notified = self._subscriptions.check_new_entry(entry)
        if notified:
            logger.info(
                "Skill publiziert + %d Subscriber benachrichtigt: %s",
                len(notified), package.package_id,
            )

        return package

    # ── Search ───────────────────────────────────────────────────

    def search(
        self,
        query: str = "",
        *,
        category: str = "",
        max_results: int = 20,
        trust_filter: bool = False,
        min_trust_score: float = 0.0,
    ) -> list[IndexEntry]:
        """Sucht im Skill-Index.

        Filtert quarantinierte Pakete und Herausgeber automatisch.
        Optional: Trust-Filter über Trusted Circles.
        """
        results = self._index.search(query, category=category, max_results=max_results * 2)

        # Quarantinierte filtern
        filtered = []
        for entry in results:
            if self._reputation.is_quarantined(entry.package_id):
                continue
            if self._reputation.is_quarantined(entry.publisher_id):
                continue
            filtered.append(entry)

        # Optional: Trust-basiertes Ranking über Circles
        if trust_filter and self._identity:
            publisher_map = {e.package_id: e.publisher_id for e in filtered}
            trusted = self._circles.filter_trusted_packages(
                [e.package_id for e in filtered],
                publisher_map,
                self._identity.peer_id,
                min_score=min_trust_score,
            )
            trusted_ids = [pkg_id for pkg_id, _score in trusted]
            # Sortiere filtered nach trusted-Reihenfolge
            id_order = {pid: i for i, pid in enumerate(trusted_ids)}
            filtered = sorted(
                [e for e in filtered if e.package_id in id_order],
                key=lambda e: id_order[e.package_id],
            )

        return filtered[:max_results]

    # ── Install ──────────────────────────────────────────────────

    def install(
        self,
        package_id: str,
        package_data: bytes | None = None,
    ) -> InstallResult:
        """Installiert ein Skill-Paket.

        1. Paket laden (lokal oder aus Bytes)
        2. Reputation prüfen
        3. An Installer delegieren
        4. Reputation aktualisieren

        Args:
            package_id: Paket-ID aus dem Index.
            package_data: Rohe Paket-Bytes (bei P2P-Download).

        Returns:
            InstallResult.
        """
        # Paket laden
        package: SkillPackage | None = None
        if package_data:
            try:
                package = SkillPackage.from_bytes(package_data)
            except Exception as exc:
                return InstallResult(
                    success=False,
                    package_id=package_id,
                    message=f"Paket-Deserialisierung fehlgeschlagen: {exc}",
                )
        else:
            package = self._packages.get(package_id)

        if not package:
            return InstallResult(
                success=False,
                package_id=package_id,
                message="Paket nicht gefunden",
            )

        # Reputation prüfen
        if self._reputation.is_quarantined(package_id):
            return InstallResult(
                success=False,
                package_id=package_id,
                message="Paket ist quarantiniert (Malware-Verdacht)",
            )

        # Installieren
        result = self._installer.install(package)

        # Reputation aktualisieren
        event = (
            ReputationEvent.INSTALL_SUCCESS if result.success
            else ReputationEvent.INSTALL_FAILURE
        )
        self._reputation.record(package_id, event)

        # Publisher-Reputation
        entry = self._index._entries.get(package_id)
        if entry:
            self._reputation.record(entry.publisher_id, event)

        return result

    # ── Feedback ─────────────────────────────────────────────────

    def report_feedback(
        self,
        package_id: str,
        *,
        positive: bool = True,
    ) -> float:
        """Meldet User-Feedback zu einem Paket.

        Returns:
            Neuer Reputation-Score.
        """
        event = (
            ReputationEvent.POSITIVE_FEEDBACK if positive
            else ReputationEvent.NEGATIVE_FEEDBACK
        )
        return self._reputation.record(package_id, event)

    def report_malware(self, package_id: str) -> None:
        """Meldet ein Paket als Malware.

        Paket wird sofort quarantiniert.
        """
        self._reputation.record(package_id, ReputationEvent.MALWARE_REPORT)
        self._reputation.quarantine(package_id)

        # Publisher auch bestrafen
        entry = self._index._entries.get(package_id)
        if entry:
            self._reputation.record(entry.publisher_id, ReputationEvent.MALWARE_REPORT)

        logger.warning("Malware gemeldet: %s", package_id)

    # ── Peer Exchange ────────────────────────────────────────────

    def sync_with_peer(self, peer_entries: list[IndexEntry]) -> int:
        """Synchronisiert den Index mit Einträgen eines anderen Peers.

        Returns:
            Anzahl neuer Einträge.
        """
        new = self._index.merge_from(peer_entries)
        if new:
            logger.info("Index-Sync: %d neue Einträge", new)
        return new

    def get_index_for_sync(self) -> list[IndexEntry]:
        """Gibt den lokalen Index für Peer-Sync zurück."""
        return list(self._index._entries.values())

    # ── Stats ────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Netzwerk-Statistiken."""
        circle_stats = self._circles.stats()
        market_stats = self._marketplace.stats()
        return {
            "peers_total": self._peers.peer_count,
            "peers_online": self._peers.online_count,
            "index_entries": self._index.entry_count,
            "installed": self._installer.installed_count,
            "subscriptions": self._subscriptions.subscription_count,
            "quarantined": len(self._reputation.quarantined_list()),
            "local_packages": len(self._packages),
            "circles": circle_stats["circles"],
            "circle_members": circle_stats["total_members"],
            "curated_skills": circle_stats["total_curated_skills"],
            "collections": circle_stats["collections"],
            "marketplace_skills": market_stats["total_skills"],
            "marketplace_installs": market_stats["total_installs"],
            "marketplace_reviews": market_stats["total_reviews"],
        }
