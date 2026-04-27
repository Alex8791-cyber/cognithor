"""
tests/test_identity/storage/test_merkle_batcher.py

Pure-unit tests for cognithor.identity.storage.merkle_batcher.
All tests are deterministic: no external services, no I/O.
Covers add(), flush(), pending_count(), _merkle_root() properties,
and batch-flush behaviour when the configured size is reached.
"""

from __future__ import annotations

import hashlib

from cognithor.identity.storage.merkle_batcher import MerkleBatcher

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _leaf(h: str) -> str:
    """Domain-separated leaf hash — mirrors the implementation's formula."""
    return _sha256(b"\x00" + bytes.fromhex(h))


def _node(left: str, right: str) -> str:
    """Internal node hash — mirrors the implementation's formula."""
    return _sha256(b"\x01" + bytes.fromhex(left) + bytes.fromhex(right))


# A stable 64-char hex string to use as a dummy content hash.
_HASH_A = "a" * 64
_HASH_B = "b" * 64
_HASH_C = "c" * 64
_HASH_D = "d" * 64


# ---------------------------------------------------------------------------
# MerkleBatcher — basic operations
# ---------------------------------------------------------------------------


class TestMerkleBatcherBasics:
    def test_pending_count_starts_at_zero(self):
        b = MerkleBatcher()
        assert b.pending_count() == 0

    def test_add_increments_pending_count(self):
        b = MerkleBatcher(batch_size=10)
        b.add(_HASH_A)
        assert b.pending_count() == 1

    def test_add_returns_none_before_full(self):
        b = MerkleBatcher(batch_size=10)
        for _ in range(9):
            result = b.add(_HASH_A)
        assert result is None

    def test_add_returns_root_when_full(self):
        b = MerkleBatcher(batch_size=3)
        b.add(_HASH_A)
        b.add(_HASH_B)
        root = b.add(_HASH_C)
        assert root is not None
        assert isinstance(root, str)

    def test_pending_cleared_after_auto_flush(self):
        b = MerkleBatcher(batch_size=2)
        b.add(_HASH_A)
        b.add(_HASH_B)
        assert b.pending_count() == 0

    def test_flush_empty_returns_none(self):
        b = MerkleBatcher()
        assert b.flush() is None

    def test_flush_returns_root_string(self):
        b = MerkleBatcher()
        b.add(_HASH_A)
        root = b.flush()
        assert root is not None
        assert len(root) == 64  # sha256 hex

    def test_flush_clears_pending(self):
        b = MerkleBatcher()
        b.add(_HASH_A)
        b.flush()
        assert b.pending_count() == 0


# ---------------------------------------------------------------------------
# MerkleBatcher._merkle_root — tree properties
# ---------------------------------------------------------------------------


class TestMerkleRoot:
    def test_single_element_returns_itself(self):
        h = _leaf(_HASH_A)
        assert MerkleBatcher._merkle_root([h]) == h

    def test_two_elements_are_hashed_together(self):
        la = _leaf(_HASH_A)
        lb = _leaf(_HASH_B)
        expected = _node(la, lb)
        assert MerkleBatcher._merkle_root([la, lb]) == expected

    def test_odd_list_duplicates_last(self):
        """Three leaves: [A, B, C] → pairs: (A,B), (C,C) → root."""
        la = _leaf(_HASH_A)
        lb = _leaf(_HASH_B)
        lc = _leaf(_HASH_C)
        parent1 = _node(la, lb)
        parent2 = _node(lc, lc)
        expected = _node(parent1, parent2)
        assert MerkleBatcher._merkle_root([la, lb, lc]) == expected

    def test_four_elements_balanced_tree(self):
        la, lb, lc, ld = _leaf(_HASH_A), _leaf(_HASH_B), _leaf(_HASH_C), _leaf(_HASH_D)
        p1 = _node(la, lb)
        p2 = _node(lc, ld)
        expected = _node(p1, p2)
        assert MerkleBatcher._merkle_root([la, lb, lc, ld]) == expected

    def test_root_is_deterministic(self):
        la = _leaf(_HASH_A)
        lb = _leaf(_HASH_B)
        r1 = MerkleBatcher._merkle_root([la, lb])
        r2 = MerkleBatcher._merkle_root([la, lb])
        assert r1 == r2

    def test_order_matters(self):
        la = _leaf(_HASH_A)
        lb = _leaf(_HASH_B)
        r_ab = MerkleBatcher._merkle_root([la, lb])
        r_ba = MerkleBatcher._merkle_root([lb, la])
        assert r_ab != r_ba


# ---------------------------------------------------------------------------
# Domain separation — leaf vs internal node
# ---------------------------------------------------------------------------


class TestDomainSeparation:
    def test_leaf_hash_differs_from_raw_hash(self):
        """The 0x00 prefix must change the output versus the raw hash."""
        raw = _HASH_A
        leaf = _leaf(raw)
        assert leaf != raw

    def test_flush_single_hash_matches_leaf_formula(self):
        b = MerkleBatcher(batch_size=5)
        b.add(_HASH_A)
        root = b.flush()
        expected = _leaf(_HASH_A)
        assert root == expected

    def test_flush_two_hashes_matches_tree_formula(self):
        b = MerkleBatcher(batch_size=5)
        b.add(_HASH_A)
        b.add(_HASH_B)
        root = b.flush()
        la = _leaf(_HASH_A)
        lb = _leaf(_HASH_B)
        expected = _node(la, lb)
        assert root == expected
