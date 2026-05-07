"""Schema migration round-trip tests — Sprint 2.2.

Round-trip protocol per DB:

    1. Create at version V_n with fixture data.
    2. Migrate forward V_n → V_n+1.
    3. Verify post-forward invariants (fixture data preserved or
       transformed predictably).
    4. Migrate backward V_n+1 → V_n.
    5. Verify post-backward state matches step 1's state by content.

The harness enumerates DBs from the registry below; new DBs are added
by appending a :class:`MigrationCase` entry. Each case is a tuple of
(name, builder, fixture, forward_step, backward_step, verifier).

Cognithor today writes most of its forward migrations inline in module
init code (e.g. ``CREATE TABLE IF NOT EXISTS …``) — not as a numbered
``schema_versions/`` ladder. The round-trip cases below are written as
**property tests on the data-preservation invariant**: even without an
explicit downgrade, dropping then re-creating a table from fixture data
should be lossless. Future migrations introduce explicit V_n / V_n+1
pairs that this harness picks up automatically.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest


@dataclass(frozen=True)
class MigrationCase:
    """One round-trip case.

    ``builder``  — creates a fresh DB at ``path`` for the OLD schema.
    ``fixture``  — inserts representative rows.
    ``forward``  — migrates OLD → NEW.
    ``backward`` — migrates NEW → OLD.
    ``verifier`` — returns a structured snapshot for equality compare.
    """

    name: str
    builder: Callable[[Path], None]
    fixture: Callable[[Path], None]
    forward: Callable[[Path], None]
    backward: Callable[[Path], None]
    verifier: Callable[[Path], object]


# ---------------------------------------------------------------------------
# Case 1: audit.jsonl-shadow-table — represents the migration that
# would happen if Cognithor moved audit storage from JSONL to SQLite.
# Deliberately synthetic so the test doesn't take a hard dep on a
# specific module's internal schema.
# ---------------------------------------------------------------------------


def _audit_v1_build(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE audit_v1 (seq INTEGER PRIMARY KEY, ts TEXT, "
        "category TEXT, payload TEXT)"
    )
    conn.commit()
    conn.close()


def _audit_v1_fixture(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    conn.executemany(
        "INSERT INTO audit_v1 VALUES (?, ?, ?, ?)",
        [
            (1, "2026-05-07T12:00:00", "system", '{"event":"start"}'),
            (2, "2026-05-07T12:00:01", "skill", '{"event":"skill_loaded"}'),
            (3, "2026-05-07T12:00:02", "reflection", '{"event":"causal"}'),
        ],
    )
    conn.commit()
    conn.close()


def _audit_v1_to_v2_forward(path: Path) -> None:
    """V1 → V2: add `prev_hash` column for hash-chain (SEC-HIGH-5)."""
    conn = sqlite3.connect(str(path))
    conn.execute("ALTER TABLE audit_v1 RENAME TO audit_v2")
    conn.execute("ALTER TABLE audit_v2 ADD COLUMN prev_hash TEXT")
    conn.execute(
        "UPDATE audit_v2 SET prev_hash = '0' WHERE prev_hash IS NULL"
    )
    conn.commit()
    conn.close()


def _audit_v2_to_v1_backward(path: Path) -> None:
    """V2 → V1: drop `prev_hash` column.

    SQLite < 3.35 didn't support DROP COLUMN; the safe pattern is to
    rebuild the table. Modern SQLite supports it directly — we use the
    rebuild pattern for portability.
    """
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE audit_v1 (seq INTEGER PRIMARY KEY, ts TEXT, "
        "category TEXT, payload TEXT)"
    )
    conn.execute(
        "INSERT INTO audit_v1 (seq, ts, category, payload) "
        "SELECT seq, ts, category, payload FROM audit_v2"
    )
    conn.execute("DROP TABLE audit_v2")
    conn.commit()
    conn.close()


def _audit_v1_snapshot(path: Path) -> list[tuple[int, str, str, str]]:
    conn = sqlite3.connect(str(path))
    rows = conn.execute(
        "SELECT seq, ts, category, payload FROM audit_v1 ORDER BY seq"
    ).fetchall()
    conn.close()
    return rows


CASES = [
    MigrationCase(
        name="audit_v1_v2_round_trip",
        builder=_audit_v1_build,
        fixture=_audit_v1_fixture,
        forward=_audit_v1_to_v2_forward,
        backward=_audit_v2_to_v1_backward,
        verifier=_audit_v1_snapshot,
    ),
]


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_round_trip_lossless(case: MigrationCase, tmp_path: Path) -> None:
    """Forward → backward → forward must preserve data."""
    db = tmp_path / f"{case.name}.sqlite"
    case.builder(db)
    case.fixture(db)

    snapshot_before = case.verifier(db)
    case.forward(db)
    case.backward(db)
    snapshot_after = case.verifier(db)

    assert snapshot_before == snapshot_after, (
        f"Migration round-trip {case.name!r} lost data.\n"
        f"  before: {snapshot_before!r}\n"
        f"  after:  {snapshot_after!r}"
    )

    # Forward again to verify the rebuilt OLD schema is still upgradable.
    case.forward(db)
    case.backward(db)
    snapshot_after2 = case.verifier(db)
    assert snapshot_before == snapshot_after2, (
        f"Migration {case.name!r} not idempotent across two round-trips."
    )


def test_registry_has_at_least_one_case() -> None:
    """Sentinel: don't ship an empty round-trip suite."""
    assert len(CASES) >= 1
    # Names must be unique so pytest IDs don't collide.
    assert len({c.name for c in CASES}) == len(CASES)


def test_each_case_has_distinct_forward_and_backward() -> None:
    """Sanity: forward != backward (else round-trip is meaningless)."""
    for case in CASES:
        assert case.forward is not case.backward, case.name
