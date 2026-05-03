"""Generic lead data models.

Source-agnostic — a ``Lead`` carries a ``source_id`` that names the origin
("reddit", "hn", "discord", "rss", ...). Reddit-specific fields (``subreddit``,
``upvotes``, ``num_comments``) are preserved as optional legacy fields to
keep the on-disk SQLite schema stable across the Reddit-Lead-Hunter-Pro
extraction. New sources that don't have a notion of a subreddit leave it
empty.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum


class LeadStatus(str, Enum):
    NEW = "new"
    REVIEWED = "reviewed"
    REPLIED = "replied"
    ARCHIVED = "archived"


@dataclass
class Lead:
    post_id: str
    source_id: str
    title: str
    url: str
    intent_score: int

    # Optional fields (default first, required fields above).
    body: str = ""
    author: str = ""
    created_utc: float = 0.0
    score_reason: str = ""
    reply_draft: str = ""
    reply_final: str = ""
    status: LeadStatus = LeadStatus.NEW
    scan_id: str = ""
    received_at: float = field(default_factory=time.time)

    # Reddit-legacy fields. Present for schema back-compat; other sources
    # leave these empty.
    subreddit: str = ""
    upvotes: int = 0
    num_comments: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "post_id": self.post_id,
            "source_id": self.source_id,
            "title": self.title,
            "url": self.url,
            "intent_score": self.intent_score,
            "body": self.body,
            "author": self.author,
            "created_utc": self.created_utc,
            "score_reason": self.score_reason,
            "reply_draft": self.reply_draft,
            "status": self.status.value,
            "scan_id": self.scan_id,
            "received_at": self.received_at,
            "subreddit": self.subreddit,
            "upvotes": self.upvotes,
            "num_comments": self.num_comments,
        }


@dataclass
class LeadStats:
    total: int = 0
    new: int = 0
    reviewed: int = 0
    replied: int = 0
    archived: int = 0
    avg_score: float = 0.0
    top_subreddits: list[tuple[str, int]] = field(default_factory=list)
    total_scans: int = 0


@dataclass
class ScanResult:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    subreddits_scanned: list[str] = field(default_factory=list)  # legacy name kept
    posts_checked: int = 0
    leads_found: int = 0
    posts_skipped_duplicate: int = 0
    posts_skipped_low_score: int = 0
    started_at: float = field(default_factory=time.time)
    finished_at: float = 0.0
    trigger: str = "unknown"

    def summary(self) -> str:
        return (
            f"{self.posts_checked} posts checked, {self.leads_found} leads found, "
            f"{self.posts_skipped_duplicate} dup, {self.posts_skipped_low_score} low-score"
        )
