"""SQLite persistence for the Skill Marketplace.

Speichert Marketplace-Listings, Reviews, Reputation und Install-History
in einer lokalen SQLite-Datenbank. Verwendet WAL-Mode und busy_timeout
analog zu session_store.py.

Architecture reference: SS14 (Skills & Ecosystem)
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jarvis.db import SQLITE_BUSY_TIMEOUT_MS
from jarvis.security.encrypted_db import encrypted_connect

try:
    from jarvis.security.encrypted_db import compatible_row_factory
except ImportError:
    compatible_row_factory = lambda: sqlite3.Row
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

# ============================================================================
# Schema
# ============================================================================

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS listings (
    package_id      TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    description     TEXT NOT NULL DEFAULT '',
    publisher_id    TEXT NOT NULL DEFAULT '',
    publisher_name  TEXT NOT NULL DEFAULT '',
    version         TEXT NOT NULL DEFAULT '1.0.0',
    category        TEXT NOT NULL DEFAULT 'sonstiges',
    tags            TEXT NOT NULL DEFAULT '[]',
    icon            TEXT NOT NULL DEFAULT '',
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL,
    install_count   INTEGER NOT NULL DEFAULT 0,
    rating_sum      REAL NOT NULL DEFAULT 0.0,
    rating_count    INTEGER NOT NULL DEFAULT 0,
    review_count    INTEGER NOT NULL DEFAULT 0,
    is_featured     INTEGER NOT NULL DEFAULT 0,
    is_verified     INTEGER NOT NULL DEFAULT 0,
    featured_reason TEXT NOT NULL DEFAULT '',
    recalled        INTEGER NOT NULL DEFAULT 0,
    recall_reason   TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS reviews (
    review_id   TEXT PRIMARY KEY,
    package_id  TEXT NOT NULL,
    reviewer_id TEXT NOT NULL,
    rating      INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5),
    comment     TEXT NOT NULL DEFAULT '',
    created_at  REAL NOT NULL,
    FOREIGN KEY (package_id) REFERENCES listings(package_id)
);

CREATE INDEX IF NOT EXISTS idx_reviews_package
    ON reviews(package_id, created_at DESC);

CREATE UNIQUE INDEX IF NOT EXISTS idx_reviews_unique
    ON reviews(package_id, reviewer_id);

CREATE TABLE IF NOT EXISTS reputation (
    peer_id     TEXT PRIMARY KEY,
    score       REAL NOT NULL DEFAULT 0.0,
    updated_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS reputation_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    peer_id     TEXT NOT NULL,
    delta       REAL NOT NULL,
    reason      TEXT NOT NULL DEFAULT '',
    created_at  REAL NOT NULL,
    FOREIGN KEY (peer_id) REFERENCES reputation(peer_id)
);

CREATE INDEX IF NOT EXISTS idx_reputation_log_peer
    ON reputation_log(peer_id, created_at DESC);

CREATE TABLE IF NOT EXISTS install_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    package_id  TEXT NOT NULL,
    version     TEXT NOT NULL DEFAULT '',
    user_id     TEXT NOT NULL DEFAULT '',
    installed_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_install_history_user
    ON install_history(user_id, installed_at DESC);

CREATE INDEX IF NOT EXISTS idx_install_history_package
    ON install_history(package_id);

CREATE INDEX IF NOT EXISTS idx_listings_category
    ON listings(category);

CREATE INDEX IF NOT EXISTS idx_listings_featured
    ON listings(is_featured) WHERE is_featured = 1;

CREATE INDEX IF NOT EXISTS idx_listings_recalled
    ON listings(recalled) WHERE recalled = 1;
"""

_MIGRATION_COMMUNITY = """\
-- source column for listings (builtin vs community)
ALTER TABLE listings ADD COLUMN source TEXT NOT NULL DEFAULT 'builtin';

CREATE INDEX IF NOT EXISTS idx_listings_source
    ON listings(source) WHERE source IS NOT NULL;

-- Publisher table for community marketplace
CREATE TABLE IF NOT EXISTS publishers (
    github_username TEXT PRIMARY KEY,
    github_id       INTEGER NOT NULL DEFAULT 0,
    display_name    TEXT NOT NULL DEFAULT '',
    verified        INTEGER NOT NULL DEFAULT 0,
    reputation_score REAL NOT NULL DEFAULT 50.0,
    trust_level     TEXT NOT NULL DEFAULT 'untrusted',
    skills_published INTEGER NOT NULL DEFAULT 0,
    abuse_reports   INTEGER NOT NULL DEFAULT 0,
    recalls         INTEGER NOT NULL DEFAULT 0,
    registered_at   REAL NOT NULL,
    updated_at      REAL NOT NULL
);

-- Remote recalls for community skills
CREATE TABLE IF NOT EXISTS recalls_remote (
    recall_id       TEXT PRIMARY KEY,
    skill_name      TEXT NOT NULL,
    reason          TEXT NOT NULL DEFAULT '',
    severity        TEXT NOT NULL DEFAULT 'high',
    issued_at       REAL NOT NULL,
    fetched_at      REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_recalls_remote_skill
    ON recalls_remote(skill_name);
"""


def _now() -> float:
    """Current UTC timestamp as float."""
    return datetime.now(UTC).timestamp()


def _iso(ts: float) -> str:
    """Unix timestamp to ISO string."""
    return datetime.fromtimestamp(ts, tz=UTC).isoformat()


# ============================================================================
# MarketplaceStore
# ============================================================================


class MarketplaceStore:
    """SQLite-based marketplace store.

    Thread-safe via SQLite WAL mode and check_same_thread=False.
    Lazily initializes the connection on first access.
    """

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        """Lazily initialize the DB connection and schema."""
        if self._conn is None:
            self._conn = encrypted_connect(
                str(self._db_path),
                check_same_thread=False,
            )
            self._conn.row_factory = compatible_row_factory()
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(_SCHEMA)
            self._migrate_community()
            log.info("marketplace_db_opened", path=str(self._db_path))
        return self._conn

    def close(self) -> None:
        """Close the DB connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Listings
    # ------------------------------------------------------------------

    def save_listing(self, listing: dict) -> str:
        """Save or update a listing.

        Args:
            listing: Dict mit mindestens ``package_id`` und ``name``.

        Returns:
            Die package_id.
        """
        now = _now()
        package_id = listing.get("package_id", str(uuid.uuid4()))
        tags = listing.get("tags", [])
        tags_json = json.dumps(tags, ensure_ascii=False) if isinstance(tags, list) else str(tags)

        self.conn.execute(
            """
            INSERT INTO listings
                (package_id, name, description, publisher_id, publisher_name,
                 version, category, tags, icon, created_at, updated_at,
                 install_count, rating_sum, rating_count, review_count,
                 is_featured, is_verified, featured_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(package_id) DO UPDATE SET
                name=excluded.name,
                description=excluded.description,
                publisher_id=excluded.publisher_id,
                publisher_name=excluded.publisher_name,
                version=excluded.version,
                category=excluded.category,
                tags=excluded.tags,
                icon=excluded.icon,
                updated_at=excluded.updated_at,
                is_featured=excluded.is_featured,
                is_verified=excluded.is_verified,
                featured_reason=excluded.featured_reason
            """,
            (
                package_id,
                listing.get("name", ""),
                listing.get("description", ""),
                listing.get("publisher_id", ""),
                listing.get("publisher_name", ""),
                listing.get("version", "1.0.0"),
                listing.get("category", "sonstiges"),
                tags_json,
                listing.get("icon", ""),
                listing.get("created_at", now),
                now,
                listing.get("install_count", 0),
                listing.get("rating_sum", 0.0),
                listing.get("rating_count", 0),
                listing.get("review_count", 0),
                int(listing.get("is_featured", False)),
                int(listing.get("is_verified", False)),
                listing.get("featured_reason", ""),
            ),
        )
        self.conn.commit()
        return package_id

    def get_listing(self, package_id: str) -> dict | None:
        """Load a single listing."""
        row = self.conn.execute(
            "SELECT * FROM listings WHERE package_id = ? AND recalled = 0",
            (package_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_listing(row)

    def search_listings(
        self,
        query: str = "",
        category: str = "",
        min_rating: float = 0.0,
        sort: str = "relevance",
        limit: int = 20,
    ) -> list[dict]:
        """Search listings with optional filters.

        Args:
            query: Volltextsuche in Name, Beschreibung, Tags.
            category: Kategorie-Filter.
            min_rating: Minimum-Durchschnittsbewertung.
            sort: Sortierung -- ``relevance``, ``newest``, ``rating``,
                  ``installs``, ``popularity``.
            limit: Maximale Ergebnisse.

        Returns:
            Liste von Listing-Dicts.
        """
        conditions = ["recalled = 0"]
        params: list[Any] = []

        if query:
            conditions.append(
                "(LOWER(name) LIKE ? OR LOWER(description) LIKE ? OR LOWER(tags) LIKE ?)"
            )
            q = f"%{query.lower()}%"
            params.extend([q, q, q])

        if category:
            conditions.append("category = ?")
            params.append(category)

        if min_rating > 0:
            conditions.append(
                "CASE WHEN rating_count > 0 THEN rating_sum / rating_count ELSE 0.0 END >= ?"
            )
            params.append(min_rating)

        where = " AND ".join(conditions)

        # Sorting
        order_map = {
            "relevance": "install_count DESC, updated_at DESC",
            "newest": "created_at DESC",
            "rating": "CASE WHEN rating_count > 0 THEN rating_sum / rating_count ELSE 0.0 END DESC",
            "installs": "install_count DESC",
            "popularity": "install_count DESC, rating_sum DESC",
        }
        if sort not in order_map:
            sort = "relevance"
        order = order_map[sort]

        sql = f"SELECT * FROM listings WHERE {where} ORDER BY {order} LIMIT ?"
        params.append(limit)

        rows = self.conn.execute(sql, params).fetchall()
        return [self._row_to_listing(r) for r in rows]

    def get_featured(self, limit: int = 10) -> list[dict]:
        """Return featured listings."""
        rows = self.conn.execute(
            """
            SELECT * FROM listings
            WHERE is_featured = 1 AND recalled = 0
            ORDER BY install_count DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [self._row_to_listing(r) for r in rows]

    def get_trending(self, days: int = 7, limit: int = 10) -> list[dict]:
        """Trending-Listings basierend auf kuerzlichen Installationen.

        Sortiert nach install_count absteigend, gefiltert nach
        Listings die innerhalb der letzten *days* Tage aktualisiert wurden.
        """
        cutoff = _now() - days * 86400
        rows = self.conn.execute(
            """
            SELECT * FROM listings
            WHERE updated_at >= ? AND recalled = 0
            ORDER BY install_count DESC, rating_sum DESC
            LIMIT ?
            """,
            (cutoff, limit),
        ).fetchall()
        return [self._row_to_listing(r) for r in rows]

    def increment_install_count(self, package_id: str) -> None:
        """Increment the installation counter by 1."""
        self.conn.execute(
            "UPDATE listings SET install_count = install_count + 1, "
            "updated_at = ? WHERE package_id = ?",
            (_now(), package_id),
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Reviews
    # ------------------------------------------------------------------

    def save_review(
        self,
        package_id: str,
        reviewer_id: str,
        rating: int,
        comment: str = "",
    ) -> str:
        """Save a review. Duplikate (gleicher reviewer_id + package_id)
        werden per UNIQUE-Index abgelehnt.

        Returns:
            Die review_id.

        Raises:
            sqlite3.IntegrityError: Bei Duplikat.
            ValueError: Bei ungueltiger Rating-Zahl.
        """
        if rating < 1 or rating > 5:
            msg = f"Rating must be between 1 and 5, got {rating}"
            raise ValueError(msg)

        review_id = f"review_{uuid.uuid4().hex[:12]}"
        now = _now()

        self.conn.execute(
            """
            INSERT INTO reviews (review_id, package_id, reviewer_id, rating, comment, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (review_id, package_id, reviewer_id, rating, comment, now),
        )

        # Update listing statistics
        self.conn.execute(
            """
            UPDATE listings
            SET rating_sum = rating_sum + ?,
                rating_count = rating_count + 1,
                review_count = review_count + 1,
                updated_at = ?
            WHERE package_id = ?
            """,
            (rating, now, package_id),
        )
        self.conn.commit()
        return review_id

    def get_reviews(self, package_id: str, limit: int = 20) -> list[dict]:
        """Load reviews for a package."""
        rows = self.conn.execute(
            """
            SELECT * FROM reviews
            WHERE package_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (package_id, limit),
        ).fetchall()
        return [
            {
                "review_id": r["review_id"],
                "package_id": r["package_id"],
                "reviewer_id": r["reviewer_id"],
                "rating": r["rating"],
                "comment": r["comment"],
                "created_at": _iso(r["created_at"]),
            }
            for r in rows
        ]

    def get_average_rating(self, package_id: str) -> float:
        """Calculate the average rating of a package."""
        row = self.conn.execute(
            "SELECT rating_sum, rating_count FROM listings WHERE package_id = ?",
            (package_id,),
        ).fetchone()
        if row is None or row["rating_count"] == 0:
            return 0.0
        return round(row["rating_sum"] / row["rating_count"], 1)

    # ------------------------------------------------------------------
    # Reputation
    # ------------------------------------------------------------------

    def update_reputation(
        self,
        peer_id: str,
        delta: float,
        reason: str = "",
    ) -> float:
        """Update the reputation score of a peer.

        Args:
            peer_id: Peer-Identifikator.
            delta: Score-Aenderung (positiv oder negativ).
            reason: Begruendung fuer die Aenderung.

        Returns:
            Neuer Score.
        """
        now = _now()

        # Create or update reputation entry
        self.conn.execute(
            """
            INSERT INTO reputation (peer_id, score, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(peer_id) DO UPDATE SET
                score = score + ?,
                updated_at = ?
            """,
            (peer_id, delta, now, delta, now),
        )

        # Log entry
        self.conn.execute(
            """
            INSERT INTO reputation_log (peer_id, delta, reason, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (peer_id, delta, reason, now),
        )
        self.conn.commit()

        return self.get_reputation(peer_id)

    def get_reputation(self, peer_id: str) -> float:
        """Return the current reputation score."""
        row = self.conn.execute(
            "SELECT score FROM reputation WHERE peer_id = ?",
            (peer_id,),
        ).fetchone()
        return row["score"] if row else 0.0

    # ------------------------------------------------------------------
    # Install History
    # ------------------------------------------------------------------

    def record_install(
        self,
        package_id: str,
        version: str = "",
        user_id: str = "",
    ) -> None:
        """Record an installation."""
        self.conn.execute(
            """
            INSERT INTO install_history (package_id, version, user_id, installed_at)
            VALUES (?, ?, ?, ?)
            """,
            (package_id, version, user_id, _now()),
        )
        self.conn.commit()

    def get_install_history(
        self,
        user_id: str,
        limit: int = 50,
    ) -> list[dict]:
        """Return the installation history of a user."""
        rows = self.conn.execute(
            """
            SELECT ih.*, l.name AS listing_name
            FROM install_history ih
            LEFT JOIN listings l ON ih.package_id = l.package_id
            WHERE ih.user_id = ?
            ORDER BY ih.installed_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
        return [
            {
                "package_id": r["package_id"],
                "version": r["version"],
                "user_id": r["user_id"],
                "installed_at": _iso(r["installed_at"]),
                "name": r["listing_name"] or r["package_id"],
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Recall
    # ------------------------------------------------------------------

    def recall_listing(self, package_id: str, reason: str = "") -> None:
        """Mark a listing as recalled."""
        self.conn.execute(
            """
            UPDATE listings
            SET recalled = 1, recall_reason = ?, updated_at = ?
            WHERE package_id = ?
            """,
            (reason, _now(), package_id),
        )
        self.conn.commit()
        log.warning("listing_recalled", package_id=package_id, reason=reason)

    def get_recalled(self) -> list[dict]:
        """Return all recalled listings."""
        rows = self.conn.execute(
            "SELECT * FROM listings WHERE recalled = 1 ORDER BY updated_at DESC",
        ).fetchall()
        return [self._row_to_listing(r) for r in rows]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Return aggregated marketplace statistics."""
        c = self.conn

        total = c.execute(
            "SELECT COUNT(*) FROM listings WHERE recalled = 0",
        ).fetchone()[0]
        total_recalled = c.execute(
            "SELECT COUNT(*) FROM listings WHERE recalled = 1",
        ).fetchone()[0]
        total_installs = c.execute(
            "SELECT COALESCE(SUM(install_count), 0) FROM listings WHERE recalled = 0",
        ).fetchone()[0]
        total_reviews = c.execute(
            "SELECT COUNT(*) FROM reviews",
        ).fetchone()[0]
        total_publishers = c.execute(
            "SELECT COUNT(DISTINCT publisher_id) FROM listings "
            "WHERE recalled = 0 AND publisher_id != ''",
        ).fetchone()[0]
        total_categories = c.execute(
            "SELECT COUNT(DISTINCT category) FROM listings WHERE recalled = 0",
        ).fetchone()[0]
        featured_count = c.execute(
            "SELECT COUNT(*) FROM listings WHERE is_featured = 1 AND recalled = 0",
        ).fetchone()[0]
        verified_count = c.execute(
            "SELECT COUNT(*) FROM listings WHERE is_verified = 1 AND recalled = 0",
        ).fetchone()[0]

        return {
            "total_listings": total,
            "total_recalled": total_recalled,
            "total_installs": total_installs,
            "total_reviews": total_reviews,
            "total_publishers": total_publishers,
            "total_categories": total_categories,
            "featured_count": featured_count,
            "verified_count": verified_count,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Community-Migration
    # ------------------------------------------------------------------

    def _migrate_community(self) -> None:
        """Apply community marketplace migrations (idempotent)."""
        assert self._conn is not None
        c = self._conn

        # Add source column (idempotent check)
        cols = {r[1] for r in c.execute("PRAGMA table_info(listings)").fetchall()}
        if "source" not in cols:
            try:
                c.executescript(_MIGRATION_COMMUNITY)
                log.info("community_migration_applied")
            except Exception as exc:
                # Tabellen existieren bereits — ignorieren (catches both sqlite3 and sqlcipher3)
                if "already exists" not in str(exc) and "duplicate column" not in str(exc):
                    log.warning("community_migration_warning", error=str(exc))
        else:
            # Ensure publishers + recalls_remote exist
            with contextlib.suppress(Exception):  # catches both sqlite3 and sqlcipher3
                c.executescript(
                    _MIGRATION_COMMUNITY.split("ALTER TABLE")[0]
                    + """
                    CREATE TABLE IF NOT EXISTS publishers (
                        github_username TEXT PRIMARY KEY,
                        github_id       INTEGER NOT NULL DEFAULT 0,
                        display_name    TEXT NOT NULL DEFAULT '',
                        verified        INTEGER NOT NULL DEFAULT 0,
                        reputation_score REAL NOT NULL DEFAULT 50.0,
                        trust_level     TEXT NOT NULL DEFAULT 'untrusted',
                        skills_published INTEGER NOT NULL DEFAULT 0,
                        abuse_reports   INTEGER NOT NULL DEFAULT 0,
                        recalls         INTEGER NOT NULL DEFAULT 0,
                        registered_at   REAL NOT NULL DEFAULT 0,
                        updated_at      REAL NOT NULL DEFAULT 0
                    );
                    CREATE TABLE IF NOT EXISTS recalls_remote (
                        recall_id       TEXT PRIMARY KEY,
                        skill_name      TEXT NOT NULL,
                        reason          TEXT NOT NULL DEFAULT '',
                        severity        TEXT NOT NULL DEFAULT 'high',
                        issued_at       REAL NOT NULL,
                        fetched_at      REAL NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_recalls_remote_skill
                        ON recalls_remote(skill_name);
                    """
                )

    # ------------------------------------------------------------------
    # Publishers
    # ------------------------------------------------------------------

    def save_publisher(self, publisher: dict) -> str:
        """Save or update a publisher.

        Returns:
            github_username des Publishers.
        """
        now = _now()
        username = publisher.get("github_username", "")
        self.conn.execute(
            """
            INSERT INTO publishers
                (github_username, github_id, display_name, verified,
                 reputation_score, trust_level, skills_published,
                 abuse_reports, recalls, registered_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(github_username) DO UPDATE SET
                github_id=excluded.github_id,
                display_name=excluded.display_name,
                verified=excluded.verified,
                reputation_score=excluded.reputation_score,
                trust_level=excluded.trust_level,
                skills_published=excluded.skills_published,
                abuse_reports=excluded.abuse_reports,
                recalls=excluded.recalls,
                updated_at=excluded.updated_at
            """,
            (
                username,
                publisher.get("github_id", 0),
                publisher.get("display_name", ""),
                int(publisher.get("verified", False)),
                publisher.get("reputation_score", 50.0),
                publisher.get("trust_level", "untrusted"),
                publisher.get("skills_published", 0),
                publisher.get("abuse_reports", 0),
                publisher.get("recalls", 0),
                publisher.get("registered_at", now),
                now,
            ),
        )
        self.conn.commit()
        return username

    def get_publisher(self, github_username: str) -> dict | None:
        """Load a publisher."""
        row = self.conn.execute(
            "SELECT * FROM publishers WHERE github_username = ?",
            (github_username,),
        ).fetchone()
        if row is None:
            return None
        return {
            "github_username": row["github_username"],
            "github_id": row["github_id"],
            "display_name": row["display_name"],
            "verified": bool(row["verified"]),
            "reputation_score": row["reputation_score"],
            "trust_level": row["trust_level"],
            "skills_published": row["skills_published"],
            "abuse_reports": row["abuse_reports"],
            "recalls": row["recalls"],
            "registered_at": _iso(row["registered_at"]) if row["registered_at"] else "",
            "updated_at": _iso(row["updated_at"]) if row["updated_at"] else "",
        }

    # ------------------------------------------------------------------
    # Remote Recalls
    # ------------------------------------------------------------------

    def save_remote_recall(self, recall: dict) -> None:
        """Save a remote recall."""
        self.conn.execute(
            """
            INSERT OR REPLACE INTO recalls_remote
                (recall_id, skill_name, reason, severity, issued_at, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                recall.get("recall_id", str(uuid.uuid4())),
                recall.get("skill_name", ""),
                recall.get("reason", ""),
                recall.get("severity", "high"),
                recall.get("issued_at", _now()),
                _now(),
            ),
        )
        self.conn.commit()

    def get_remote_recalls(self) -> list[dict]:
        """Return all remote recalls."""
        rows = self.conn.execute(
            "SELECT * FROM recalls_remote ORDER BY issued_at DESC",
        ).fetchall()
        return [
            {
                "recall_id": r["recall_id"],
                "skill_name": r["skill_name"],
                "reason": r["reason"],
                "severity": r["severity"],
                "issued_at": _iso(r["issued_at"]),
                "fetched_at": _iso(r["fetched_at"]),
            }
            for r in rows
        ]

    def is_skill_recalled_remote(self, skill_name: str) -> bool:
        """Check if a skill is remotely recalled."""
        row = self.conn.execute(
            "SELECT 1 FROM recalls_remote WHERE skill_name = ? LIMIT 1",
            (skill_name,),
        ).fetchone()
        return row is not None

    @staticmethod
    def _row_to_listing(row: sqlite3.Row) -> dict:
        """Convert a DB row to a listing dict."""
        tags_raw = row["tags"]
        try:
            tags = json.loads(tags_raw) if tags_raw else []
        except (json.JSONDecodeError, TypeError):
            tags = []

        rating_count = row["rating_count"]
        rating_sum = row["rating_sum"]
        avg_rating = round(rating_sum / rating_count, 1) if rating_count > 0 else 0.0

        return {
            "package_id": row["package_id"],
            "name": row["name"],
            "description": row["description"],
            "publisher_id": row["publisher_id"],
            "publisher_name": row["publisher_name"],
            "version": row["version"],
            "category": row["category"],
            "tags": tags,
            "icon": row["icon"],
            "created_at": _iso(row["created_at"]),
            "updated_at": _iso(row["updated_at"]),
            "install_count": row["install_count"],
            "average_rating": avg_rating,
            "rating_count": rating_count,
            "review_count": row["review_count"],
            "is_featured": bool(row["is_featured"]),
            "is_verified": bool(row["is_verified"]),
            "featured_reason": row["featured_reason"],
            "recalled": bool(row["recalled"]),
            "recall_reason": row["recall_reason"],
            "source": row["source"] if "source" in row.keys() else "builtin",
        }
