"""Tests for the SQLite-backed FingerprintStore (TRUST-7 disk persistence)."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from cognithor.security.fingerprint import (
    BinaryKind,
    FingerprintLedger,
    ToolFingerprint,
)
from cognithor.security.fingerprint_store import (
    FingerprintStore,
    FingerprintStoreError,
)


def _hash(seed: str) -> str:
    return hashlib.sha256(seed.encode()).hexdigest()


def _fp(
    *,
    name: str = "web_fetch",
    kind: BinaryKind = BinaryKind.TOOL,
    content_hash: str | None = None,
    version: str = "1.0.0",
    captured_at: datetime | None = None,
    source_path: str = "",
    upstream_url: str = "",
    notes: str = "",
) -> ToolFingerprint:
    return ToolFingerprint(
        name=name,
        kind=kind,
        content_hash=content_hash or _hash(f"{name}:{version}"),
        version=version,
        captured_at=captured_at or datetime(2026, 5, 9, 12, 0, 0, tzinfo=UTC),
        source_path=source_path,
        upstream_url=upstream_url,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Lifecycle + schema
# ---------------------------------------------------------------------------


class TestStoreLifecycle:
    def test_creates_schema_in_memory(self) -> None:
        with FingerprintStore(":memory:") as store:
            assert store.schema_version == 1
            assert len(store) == 0

    def test_context_manager_persists_to_disk(self, tmp_path) -> None:
        db = tmp_path / "x.sqlite"
        with FingerprintStore(db) as store:
            store.register(_fp(name="web_fetch"))

        with FingerprintStore(db) as store:
            assert len(store) == 1
            assert store.names() == ["web_fetch"]

    def test_close_idempotent(self) -> None:
        store = FingerprintStore(":memory:")
        store.close()
        store.close()  # must not raise

    def test_reopen_existing_does_not_reset(self, tmp_path) -> None:
        db = tmp_path / "x.sqlite"
        with FingerprintStore(db) as store:
            store.register(_fp())
        with FingerprintStore(db) as store:
            assert len(store) == 1


# ---------------------------------------------------------------------------
# Schema-version guard
# ---------------------------------------------------------------------------


class TestSchemaVersionGuard:
    def test_rejects_newer_schema(self, tmp_path) -> None:
        db = tmp_path / "future.sqlite"
        FingerprintStore(db).close()
        conn = sqlite3.connect(db)
        try:
            conn.execute(
                "INSERT INTO _schema_meta (version, applied_at) VALUES (?, ?)",
                (999, datetime.now(UTC).isoformat()),
            )
            conn.commit()
        finally:
            conn.close()

        with pytest.raises(FingerprintStoreError, match="version 999"):
            FingerprintStore(db)

    def test_accepts_equal_version(self, tmp_path) -> None:
        db = tmp_path / "x.sqlite"
        FingerprintStore(db).close()
        FingerprintStore(db).close()  # second open at same version is no-op


# ---------------------------------------------------------------------------
# Register + remove
# ---------------------------------------------------------------------------


class TestRegisterAndRemove:
    def test_register_returns_true_first_time(self) -> None:
        with FingerprintStore(":memory:") as store:
            assert store.register(_fp()) is True
            assert len(store) == 1

    def test_register_idempotent_on_same_hash(self) -> None:
        """Mirrors the in-memory contract: re-registering an identical
        hash is a no-op and returns ``False``. SQLite UNIQUE on
        ``content_hash`` enforces this at the storage layer."""
        fp = _fp()
        with FingerprintStore(":memory:") as store:
            assert store.register(fp) is True
            assert store.register(fp) is False
            assert len(store) == 1

    def test_register_new_hash_for_existing_name_appends(self) -> None:
        """Two different SHA-256s under the same logical name MUST
        both land in storage — that's the smoking-gun query
        ``divergent_names()`` exists to surface."""
        v1 = _fp(name="web_fetch", version="1.0.0", content_hash=_hash("v1"))
        v2 = _fp(name="web_fetch", version="1.1.0", content_hash=_hash("v2"))
        with FingerprintStore(":memory:") as store:
            assert store.register(v1) is True
            assert store.register(v2) is True
            assert len(store) == 2

    def test_full_field_round_trip(self) -> None:
        original = _fp(
            name="qwen3:30b",
            kind=BinaryKind.MODEL,
            content_hash=_hash("qwen3-30b-v1"),
            version="2026.04.16",
            captured_at=datetime(2026, 5, 9, 14, 30, 15, tzinfo=UTC),
            source_path="/models/qwen3.bin",
            upstream_url="https://huggingface.co/Qwen/Qwen3-30B",
            notes="planner default",
        )
        with FingerprintStore(":memory:") as store:
            store.register(original)
            recovered = store.get(original.content_hash)
            assert recovered == original

    def test_remove_returns_true_when_present(self) -> None:
        with FingerprintStore(":memory:") as store:
            fp = _fp()
            store.register(fp)
            assert store.remove(fp.content_hash) is True
            assert len(store) == 0

    def test_remove_returns_false_when_absent(self) -> None:
        with FingerprintStore(":memory:") as store:
            assert store.remove(_hash("nope")) is False

    def test_clear_drops_everything(self) -> None:
        with FingerprintStore(":memory:") as store:
            for i in range(5):
                store.register(_fp(content_hash=_hash(f"x{i}")))
            assert len(store) == 5
            store.clear()
            assert len(store) == 0


# ---------------------------------------------------------------------------
# Lookup parity with FingerprintLedger
# ---------------------------------------------------------------------------


class TestLookupParity:
    def test_get_present_and_absent(self) -> None:
        with FingerprintStore(":memory:") as store:
            fp = _fp()
            store.register(fp)
            assert store.get(fp.content_hash) == fp
            assert store.get(_hash("missing")) is None

    def test_contains_only_accepts_strings(self) -> None:
        """Mirrors the in-memory ledger's ``__contains__`` shape:
        non-string queries silently return False (don't raise) so a
        caller passing a misshaped value gets the same answer
        regardless of source."""
        with FingerprintStore(":memory:") as store:
            fp = _fp()
            store.register(fp)
            assert (fp.content_hash in store) is True
            assert (_hash("missing") in store) is False
            assert (123 in store) is False  # type: ignore[operator]
            assert (None in store) is False  # type: ignore[operator]

    def test_history_oldest_first(self) -> None:
        base = datetime(2026, 5, 9, 10, 0, 0, tzinfo=UTC)
        with FingerprintStore(":memory:") as store:
            store.register(
                _fp(
                    name="web_fetch",
                    content_hash=_hash("v1"),
                    captured_at=base,
                )
            )
            store.register(
                _fp(
                    name="web_fetch",
                    content_hash=_hash("v2"),
                    captured_at=base + timedelta(hours=1),
                )
            )
            chain = store.history("web_fetch")
            assert [fp.content_hash for fp in chain] == [_hash("v1"), _hash("v2")]

    def test_history_missing_name_returns_empty(self) -> None:
        with FingerprintStore(":memory:") as store:
            assert store.history("nope") == ()

    def test_names_sorted(self) -> None:
        with FingerprintStore(":memory:") as store:
            store.register(_fp(name="zebra", content_hash=_hash("z")))
            store.register(_fp(name="alpha", content_hash=_hash("a")))
            store.register(_fp(name="mango", content_hash=_hash("m")))
            assert store.names() == ["alpha", "mango", "zebra"]

    def test_filter_by_kind(self) -> None:
        with FingerprintStore(":memory:") as store:
            store.register(_fp(name="t1", kind=BinaryKind.TOOL, content_hash=_hash("t1")))
            store.register(_fp(name="t2", kind=BinaryKind.TOOL, content_hash=_hash("t2")))
            store.register(_fp(name="qwen3", kind=BinaryKind.MODEL, content_hash=_hash("m1")))
            tools = store.filter_by_kind(BinaryKind.TOOL)
            assert [fp.name for fp in tools] == ["t1", "t2"]
            models = store.filter_by_kind(BinaryKind.MODEL)
            assert len(models) == 1
            packs = store.filter_by_kind(BinaryKind.PACK)
            assert packs == []

    def test_divergent_names(self) -> None:
        """A name with two distinct hashes must show up; a name with
        only one must not."""
        with FingerprintStore(":memory:") as store:
            # web_fetch — divergent (two hashes)
            store.register(_fp(name="web_fetch", content_hash=_hash("wf-v1")))
            store.register(_fp(name="web_fetch", content_hash=_hash("wf-v2")))
            # stable_tool — only one hash
            store.register(_fp(name="stable_tool", content_hash=_hash("st-v1")))
            assert store.divergent_names() == ["web_fetch"]


# ---------------------------------------------------------------------------
# Snapshot — must match in-memory ledger contract
# ---------------------------------------------------------------------------


class TestSnapshot:
    def test_empty_snapshot(self) -> None:
        with FingerprintStore(":memory:") as store:
            assert store.snapshot() == []

    def test_round_trip_via_json(self) -> None:
        with FingerprintStore(":memory:") as store:
            store.register(
                _fp(
                    name="qwen3:30b",
                    kind=BinaryKind.MODEL,
                    content_hash=_hash("q3"),
                    version="2026.04.16",
                    notes="planner",
                )
            )
            rows = store.snapshot()
            serialised = json.dumps(rows)
            parsed = json.loads(serialised)
            assert isinstance(parsed, list)
            row = parsed[0]
            assert row["name"] == "qwen3:30b"
            assert row["kind"] == "model"
            assert row["short_hash"] == _hash("q3")[:12]
            assert row["version"] == "2026.04.16"

    def test_snapshot_keys_match_in_memory_ledger(self) -> None:
        """Trace-UI consumers must NOT branch on whether the source
        is in-memory or on-disk. Pin the key set."""
        memory_ledger = FingerprintLedger()
        memory_ledger.register(_fp())
        memory_keys = set(memory_ledger.snapshot()[0].keys())

        with FingerprintStore(":memory:") as store:
            store.register(_fp())
            disk_keys = set(store.snapshot()[0].keys())

        assert memory_keys == disk_keys, (
            f"snapshot keys diverge — memory_only={memory_keys - disk_keys}, "
            f"disk_only={disk_keys - memory_keys}"
        )

    def test_snapshot_ordering_matches_in_memory_ledger(self) -> None:
        """Stable ordering: name → captured_at → content_hash. Pin
        the order so Trace-UI rendering is deterministic."""
        base = datetime(2026, 5, 9, 10, 0, 0, tzinfo=UTC)
        events = [
            _fp(name="zebra", content_hash=_hash("z"), captured_at=base),
            _fp(
                name="alpha",
                content_hash=_hash("a1"),
                captured_at=base + timedelta(hours=1),
            ),
            _fp(name="alpha", content_hash=_hash("a2"), captured_at=base),
            _fp(
                name="mango",
                content_hash=_hash("m"),
                captured_at=base + timedelta(hours=2),
            ),
        ]

        memory_ledger = FingerprintLedger()
        for e in events:
            memory_ledger.register(e)

        with FingerprintStore(":memory:") as store:
            for e in events:
                store.register(e)

            mem_order = [(r["name"], r["captured_at"]) for r in memory_ledger.snapshot()]
            disk_order = [(r["name"], r["captured_at"]) for r in store.snapshot()]
            assert mem_order == disk_order


# ---------------------------------------------------------------------------
# File persistence + concurrency
# ---------------------------------------------------------------------------


class TestFilePersistence:
    def test_writes_visible_after_reopen(self, tmp_path) -> None:
        db = tmp_path / "x.sqlite"
        with FingerprintStore(db) as store:
            store.register(_fp(name="ollama", content_hash=_hash("o1")))
            store.register(_fp(name="vllm", content_hash=_hash("v1")))
        with FingerprintStore(db) as store:
            assert len(store) == 2
            names = set(store.names())
            assert names == {"ollama", "vllm"}

    def test_concurrent_writers_share_db(self, tmp_path) -> None:
        db = tmp_path / "shared.sqlite"
        with (
            FingerprintStore(db) as a,
            FingerprintStore(db) as b,
        ):
            a.register(_fp(name="tool-a", content_hash=_hash("a")))
            b.register(_fp(name="tool-b", content_hash=_hash("b")))
        with FingerprintStore(db) as reader:
            assert len(reader) == 2
            assert {n for n in reader.names()} == {"tool-a", "tool-b"}

    def test_concurrent_writers_dedupe_on_unique_hash(self, tmp_path) -> None:
        """Two concurrent writers racing the same hash must converge —
        the SQLite UNIQUE on ``content_hash`` is the authority. One
        ``register`` returns True, the other returns False, and the
        store ends with exactly one row."""
        db = tmp_path / "race.sqlite"
        fp = _fp(content_hash=_hash("dedupe"))
        with (
            FingerprintStore(db) as a,
            FingerprintStore(db) as b,
        ):
            r1 = a.register(fp)
            r2 = b.register(fp)
            assert {r1, r2} == {True, False}
        with FingerprintStore(db) as reader:
            assert len(reader) == 1


# ---------------------------------------------------------------------------
# Indices — query-shape sanity
# ---------------------------------------------------------------------------


class TestIndices:
    def test_all_three_secondary_indices_present(self) -> None:
        """Pin the secondary indices: name, kind, captured_at. The
        primary read path (by content_hash) is covered by the UNIQUE
        constraint already, so it doesn't get its own named index."""
        with FingerprintStore(":memory:") as store:
            cursor = store._conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='index' AND tbl_name='fingerprint_events'"
            )
            names = {row[0] for row in cursor.fetchall()}
            assert "idx_fingerprint_events_name" in names
            assert "idx_fingerprint_events_kind" in names
            assert "idx_fingerprint_events_captured_at" in names

    def test_unique_content_hash_constraint_present(self) -> None:
        """The hash-uniqueness contract is the single most load-bearing
        invariant — without it, ``register()`` idempotency breaks. Pin
        an explicit assertion against the schema."""
        with FingerprintStore(":memory:") as store:
            cursor = store._conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='fingerprint_events'"
            )
            row = cursor.fetchone()
            assert row is not None
            sql = str(row[0])
            assert "UNIQUE" in sql
            assert "content_hash" in sql
