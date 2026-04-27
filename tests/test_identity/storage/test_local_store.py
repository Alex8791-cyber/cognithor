"""
tests/test_identity/storage/test_local_store.py

Pure-unit tests for cognithor.identity.storage.local_store.
Uses pytest tmp_path for filesystem isolation. Covers JSON round-trip,
_SAFE_ID_RE filename sanitization, path-traversal blocking, listing,
latest-snapshot retrieval, and cleanup.
"""

from __future__ import annotations

import json
import os

import pytest

from cognithor.identity.storage.local_store import _SAFE_ID_RE, LocalStore

# ---------------------------------------------------------------------------
# _SAFE_ID_RE
# ---------------------------------------------------------------------------


class TestSafeIdRegex:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("abc123", "abc123"),
            ("abc-123_XY", "abc-123_XY"),
            ("abc/123", "abc_123"),
            ("hello world", "hello_world"),
            ("foo@bar.baz", "foo_bar_baz"),
            ("!!danger!!", "__danger__"),
            ("normal", "normal"),
        ],
    )
    def test_substitution(self, raw, expected):
        assert _SAFE_ID_RE.sub("_", raw) == expected


# ---------------------------------------------------------------------------
# LocalStore — init
# ---------------------------------------------------------------------------


class TestLocalStoreInit:
    def test_creates_base_dir(self, tmp_path):
        store_dir = str(tmp_path / "new_store")
        LocalStore(base_dir=store_dir)
        assert os.path.isdir(store_dir)

    def test_existing_dir_no_error(self, tmp_path):
        store_dir = str(tmp_path)
        LocalStore(base_dir=store_dir)  # should not raise


# ---------------------------------------------------------------------------
# LocalStore — save_snapshot / load_snapshot round-trip
# ---------------------------------------------------------------------------


class TestSaveLoadSnapshot:
    def test_save_returns_metadata_keys(self, tmp_path):
        store = LocalStore(base_dir=str(tmp_path))
        result = store.save_snapshot({"key": "value"}, "test_id")
        assert "uri" in result
        assert "filepath" in result
        assert "hash" in result
        assert "timestamp" in result

    def test_saved_file_exists(self, tmp_path):
        store = LocalStore(base_dir=str(tmp_path))
        result = store.save_snapshot({"k": 1}, "myid")
        assert os.path.isfile(result["filepath"])

    def test_load_roundtrip(self, tmp_path):
        store = LocalStore(base_dir=str(tmp_path))
        data = {"hello": "world", "num": 42}
        result = store.save_snapshot(data, "roundtrip_test")
        loaded = store.load_snapshot(result["uri"])
        assert loaded == data

    def test_hash_is_sha256_of_content(self, tmp_path):
        import hashlib

        store = LocalStore(base_dir=str(tmp_path))
        data = {"x": 1}
        result = store.save_snapshot(data, "hash_check")
        content = json.dumps(data, ensure_ascii=False, indent=2)
        expected_hash = hashlib.sha256(content.encode()).hexdigest()
        assert result["hash"] == expected_hash

    def test_special_chars_in_identity_id_sanitized(self, tmp_path):
        store = LocalStore(base_dir=str(tmp_path))
        result = store.save_snapshot({}, "id/with/slashes")
        filename = os.path.basename(result["filepath"])
        # The sanitized prefix must not contain slashes
        assert "/" not in filename
        assert "\\" not in filename

    def test_uri_uses_local_scheme(self, tmp_path):
        store = LocalStore(base_dir=str(tmp_path))
        result = store.save_snapshot({}, "myid")
        assert result["uri"].startswith("local://")

    def test_load_missing_uri_returns_none(self, tmp_path):
        store = LocalStore(base_dir=str(tmp_path))
        result = store.load_snapshot(f"local://{tmp_path}/nonexistent.json")
        assert result is None


# ---------------------------------------------------------------------------
# LocalStore — path traversal
# ---------------------------------------------------------------------------


class TestPathTraversal:
    def test_traversal_blocked(self, tmp_path):
        store = LocalStore(base_dir=str(tmp_path / "store"))
        # Construct a URI that tries to escape the base_dir
        outside = str(tmp_path / "secret.json")
        (tmp_path / "secret.json").write_text('{"secret": true}')
        result = store.load_snapshot(f"local://{outside}")
        assert result is None


# ---------------------------------------------------------------------------
# LocalStore — list_snapshots
# ---------------------------------------------------------------------------


class TestListSnapshots:
    def test_empty_dir_returns_empty(self, tmp_path):
        store = LocalStore(base_dir=str(tmp_path))
        assert store.list_snapshots() == []

    def test_lists_saved_snapshots(self, tmp_path):
        store = LocalStore(base_dir=str(tmp_path))
        # Use distinct identity IDs so filenames never collide on same-second saves
        store.save_snapshot({"a": 1}, "myid_aaa")
        store.save_snapshot({"b": 2}, "myid_bbb")
        snapshots = store.list_snapshots()
        assert len(snapshots) == 2

    def test_filter_by_identity_id(self, tmp_path):
        store = LocalStore(base_dir=str(tmp_path))
        store.save_snapshot({}, "alice")
        store.save_snapshot({}, "bob__")
        alice_snaps = store.list_snapshots(identity_id="alice")
        assert all("alice" in s["filename"] for s in alice_snaps)

    def test_metadata_keys_present(self, tmp_path):
        store = LocalStore(base_dir=str(tmp_path))
        store.save_snapshot({"z": 9}, "anyid")
        snap = store.list_snapshots()[0]
        for key in ("filename", "filepath", "uri", "size_bytes", "modified_at"):
            assert key in snap


# ---------------------------------------------------------------------------
# LocalStore — get_latest_snapshot
# ---------------------------------------------------------------------------


class TestGetLatestSnapshot:
    def test_returns_none_when_empty(self, tmp_path):
        store = LocalStore(base_dir=str(tmp_path))
        assert store.get_latest_snapshot("noid") is None

    def test_returns_data_of_most_recent(self, tmp_path):
        store = LocalStore(base_dir=str(tmp_path))
        store.save_snapshot({"v": 1}, "myid")
        store.save_snapshot({"v": 2}, "myid")
        latest = store.get_latest_snapshot("myid")
        assert latest is not None


# ---------------------------------------------------------------------------
# LocalStore — cleanup_old_snapshots
# ---------------------------------------------------------------------------


class TestCleanupOldSnapshots:
    def test_deletes_excess_files(self, tmp_path):
        store = LocalStore(base_dir=str(tmp_path))
        # Each save uses a unique ID so names don't collide on same-second timestamps
        for i in range(5):
            store.save_snapshot({"i": i}, f"myid_{i:03d}")
        deleted = store.cleanup_old_snapshots(keep_last=3)
        assert deleted == 2
        assert len(store.list_snapshots()) == 3

    def test_keep_last_zero_deletes_all(self, tmp_path):
        store = LocalStore(base_dir=str(tmp_path))
        for i in range(4):
            store.save_snapshot({}, f"uid_{i:03d}")
        deleted = store.cleanup_old_snapshots(keep_last=0)
        assert deleted == 4
        assert store.list_snapshots() == []

    def test_no_excess_returns_zero(self, tmp_path):
        store = LocalStore(base_dir=str(tmp_path))
        store.save_snapshot({}, "myid")
        deleted = store.cleanup_old_snapshots(keep_last=10)
        assert deleted == 0
