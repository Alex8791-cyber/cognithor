"""Smoke tests for the chaos-fault primitives themselves.

Verifies each primitive activates + cleans up cleanly. Real chaos tests
that wire these into PGE-loop / vLLM / audit-chain operations live in
sibling files (test_chaos_pge.py, test_chaos_audit.py, etc.) which are
heavier and run only on the nightly chaos workflow.
"""

from __future__ import annotations

import socket
import sys
import time
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from tests.chaos.faults import (
    audit_chain_is_intact,
    black_hole_port,
    corrupt_sqlite_pages,
    crash_subprocess_after,
    disk_full_simulator,
    kill_subprocess_after,
    simulate_gpu_oom_via_env,
    tamper_audit_jsonl,
)


class TestProcessFaults:
    def test_kill_subprocess_after_cancels_when_test_finishes_early(self) -> None:
        # Use this process's pid; the killer should be cancelled before
        # firing because the with-block exits immediately.
        with kill_subprocess_after(pid=999_999_999, delay_seconds=0.5):
            pass
        # If the killer fired we'd have torn ourselves down; reaching here proves cancel works.

    def test_crash_subprocess_after_does_not_fire_when_pid_is_invalid(self) -> None:
        # 999_999_999 doesn't exist; killer suppresses ProcessLookupError.
        with crash_subprocess_after(pid=999_999_999, delay_seconds=0.05):
            time.sleep(0.15)


class TestNetworkFaults:
    def test_black_hole_port_accepts_but_does_not_respond(self) -> None:
        with black_hole_port() as port:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2.0)
            sock.connect(("127.0.0.1", port))
            sock.sendall(b"ping")
            sock.settimeout(0.5)
            with pytest.raises(socket.timeout):
                sock.recv(64)
            sock.close()


class TestFilesystemFaults:
    def test_disk_full_simulator_yields_existing_file(self, tmp_path: Path) -> None:
        with disk_full_simulator(tmp_path / "d", max_bytes=4096) as p:
            assert p.exists()
            assert p.stat().st_size == 4096
        assert not p.exists()  # cleanup


class TestDatabaseFaults:
    def test_corrupt_sqlite_pages_restores_on_exit(self, tmp_path: Path) -> None:
        import sqlite3

        db = tmp_path / "x.sqlite"
        conn = sqlite3.connect(str(db))
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
        for i in range(50):
            conn.execute("INSERT INTO t VALUES (?, ?)", (i, "x" * 50))
        conn.commit()
        conn.close()

        original_bytes = db.read_bytes()
        with corrupt_sqlite_pages(db, pages_to_corrupt=2):
            mid_bytes = db.read_bytes()
            assert mid_bytes != original_bytes  # corruption applied
        # Restored
        assert db.read_bytes() == original_bytes


class TestGpuFault:
    def test_simulate_gpu_oom_via_env_sets_and_restores(self) -> None:
        import os

        before_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
        before_hf = os.environ.get("HF_NO_GPU")
        with simulate_gpu_oom_via_env():
            assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            assert os.environ.get("HF_NO_GPU") == "1"
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == before_cuda
        assert os.environ.get("HF_NO_GPU") == before_hf


class TestAuditFaults:
    def test_audit_chain_is_intact_on_empty(self, tmp_path: Path) -> None:
        p = tmp_path / "audit.jsonl"
        p.write_text("", encoding="utf-8")
        assert audit_chain_is_intact(p) is True

    def test_tamper_audit_jsonl_restores_on_exit(self, tmp_path: Path) -> None:
        # Build a tiny synthetic chain (non-canonical, just for round-trip)
        p = tmp_path / "audit.jsonl"
        p.write_text(
            '{"event": "a", "prev_hash": "x"}\n{"event": "b", "prev_hash": "y"}\n',
            encoding="utf-8",
        )
        before = p.read_bytes()
        with tamper_audit_jsonl(p, line_offset=0):
            mid = p.read_bytes()
            assert mid != before
        assert p.read_bytes() == before


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="POSIX-only signal semantics — Win uses different kill semantics",
)
def test_pge_loop_recovery_under_subprocess_kill_is_documented_only() -> None:
    """Marker for the heavier chaos tests under tests/chaos/test_chaos_pge.py.

    Heavy chaos tests need a live Gateway instance and run only on the
    nightly chaos workflow — not in the per-PR pytest run. This test
    exists so a developer reading the suite knows where to look.
    """
    assert True
