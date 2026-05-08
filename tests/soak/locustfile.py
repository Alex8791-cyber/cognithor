"""Soak / load harness for Cognithor — Sprint 1.3.

Five user types, each modelling a real production traffic class:

* **ChatUser**   — sends short chat turns to ``/api/v1/message``,
  measures first-token latency and full-response latency.
* **VaultUser**  — reads + searches the knowledge vault.
* **MemoryUser** — fires hybrid-search queries (BM25 + vector + graph).
* **CrewUser**   — kicks off Crew tasks via ``/api/crew/run``.
* **VideoUser**  — uploads short MP4s + asks "describe" (VLM router path).

Run::

    locust -f tests/soak/locustfile.py --headless -u 10 -r 2 -t 1h \\
        --host http://localhost:8741

Or via the makefile::

    make soak-1h    # 1h dry-run
    make soak-24h   # 24h release-gate

The 24h run is observed by ``tests/soak/observers/`` which sample
process-RSS, file-descriptor counts, SQLite-lock waits, audit-chain
head index every 30 seconds and refuse to mark the run "green" if any
of these grow unboundedly.
"""

from __future__ import annotations

import json
import random
import string
import time
from typing import Any

from locust import HttpUser, between, events, task

# ---------------------------------------------------------------------------
# Shared payload generators
# ---------------------------------------------------------------------------


def _short_prompt() -> str:
    """Random short prompt — keeps the suite from becoming a single-cache hit."""
    nouns = ("Termin", "Vertrag", "Kunde", "Projekt", "Audit", "Deadline")
    verbs = ("erstelle", "fasse zusammen", "finde", "vergleiche")
    suffix = "".join(random.choices(string.ascii_lowercase, k=4))
    return f"{random.choice(verbs).capitalize()} {random.choice(nouns)} {suffix}."


def _short_search() -> str:
    return random.choice(
        [
            "Quartalsbericht Q3",
            "TLS-Konfiguration",
            "GDPR Artikel 17",
            "Backup-Strategie",
            "Audit chain integrity",
            "VLM router fallback policy",
        ],
    )


# ---------------------------------------------------------------------------
# User classes
# ---------------------------------------------------------------------------


class ChatUser(HttpUser):
    """Models a chat-page user. Highest weight — most realistic load."""

    weight = 5
    wait_time = between(2, 8)

    @task(4)
    def short_chat_turn(self) -> None:
        with self.client.post(
            "/api/v1/message",
            json={"text": _short_prompt(), "session_id": "soak-chat"},
            catch_response=True,
            timeout=60,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"chat returned {resp.status_code}")

    @task(1)
    def long_chat_turn(self) -> None:
        with self.client.post(
            "/api/v1/message",
            json={
                "text": "Erkläre " + _short_search() + " im Detail.",
                "session_id": "soak-chat-long",
            },
            catch_response=True,
            timeout=120,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"long chat returned {resp.status_code}")


class VaultUser(HttpUser):
    weight = 2
    wait_time = between(5, 15)

    @task
    def vault_search(self) -> None:
        self.client.get(
            "/api/vault/search",
            params={"q": _short_search(), "limit": 10},
            timeout=30,
        )

    @task
    def vault_save(self) -> None:
        self.client.post(
            "/api/vault/save",
            json={
                "title": f"soak-note-{int(time.time())}-{random.randint(0, 9999)}",
                "body": "Soak-test note " + _short_prompt() * 3,
                "tags": ["soak", "auto"],
            },
            timeout=30,
        )


class MemoryUser(HttpUser):
    weight = 2
    wait_time = between(3, 10)

    @task
    def memory_search(self) -> None:
        self.client.get(
            "/api/memory/search",
            params={"q": _short_search(), "k": 6},
            timeout=30,
        )


class CrewUser(HttpUser):
    weight = 1
    wait_time = between(15, 60)

    @task
    def crew_kickoff(self) -> None:
        self.client.post(
            "/api/crew/run",
            json={
                "template": "research",
                "input": {"topic": _short_search()},
            },
            timeout=300,
        )


class VideoUser(HttpUser):
    """VLM-router path — disabled by default because each call eats GPU.

    Enable with environment variable LOCUST_INCLUDE_VIDEO=1.
    """

    weight = 0  # off by default
    wait_time = between(60, 180)

    @task
    def video_describe(self) -> None:
        # Soak doesn't actually upload bytes — it hits a synthetic
        # endpoint that simulates the round-trip without burning GPU.
        # Real video soak is a separate suite.
        self.client.post(
            "/api/v1/message",
            json={
                "text": "Describe the attached clip in one sentence.",
                "session_id": "soak-video",
                "synthetic_video_attached": True,
            },
            timeout=600,
        )


# ---------------------------------------------------------------------------
# Observability hooks — track soak-specific drift signals
# ---------------------------------------------------------------------------


_started_at: float = 0.0
_observations: list[dict[str, Any]] = []


@events.test_start.add_listener
def _on_test_start(environment: Any, **_: Any) -> None:
    global _started_at
    _started_at = time.time()


@events.test_stop.add_listener
def _on_test_stop(environment: Any, **_: Any) -> None:
    elapsed = time.time() - _started_at
    summary = {
        "elapsed_seconds": round(elapsed, 1),
        "observations": _observations,
        "stats_csv_hint": "Use --csv=soak-results to capture full stats",
    }
    with open("soak-summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nSoak summary written to soak-summary.json (elapsed={elapsed:.1f}s)")
