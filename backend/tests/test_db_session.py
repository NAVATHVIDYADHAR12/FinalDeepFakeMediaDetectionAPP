"""Regression tests for the database session helper.

These exist because of a real corruption incident: every query used
`with sqlite3.connect(...) as conn`, which commits but does NOT close. Hundreds
of live handles against one file left the database with an invalid page-1
B-tree header, and the whole app started returning 500s.
"""

from __future__ import annotations

import sqlite3
import threading
import time

import numpy as np
import pytest

import auth
import database as db


class TestSharedConnection:
    """The connection is deliberately long-lived.

    Two earlier designs were both wrong: one leaked a handle per query and
    corrupted the database; the other opened and closed per query, which made
    every write pay for a WAL checkpoint (~700 ms measured). These tests pin
    the behaviour that avoids both.
    """

    def test_the_same_connection_is_reused(self):
        with db.session() as a:
            first = a
        with db.session() as b:
            assert b is first, "each call opened a new connection"

    def test_many_operations_open_at_most_one_connection(self, monkeypatch):
        """The original failure mode was handles accumulating per query."""
        opened = []
        real_connect = sqlite3.connect

        def tracking_connect(*a, **kw):
            conn = real_connect(*a, **kw)
            opened.append(conn)
            return conn

        db.close()                       # force a fresh open we can observe
        monkeypatch.setattr(sqlite3, "connect", tracking_connect)

        for i in range(40):
            db.save_scan({
                "scan_id": f"REUSE{i}", "media_type": "image", "filename": f"{i}.jpg",
                "verdict": "AUTHENTIC", "risk_level": "MINIMAL", "fake_probability": 0.1,
                "authenticity_score": 90.0, "confidence": 0.8, "faces_detected": 1,
                "file_size_bytes": 10, "processing_ms": 1.0, "faces": [],
            })
            db.stats()
            db.recent_scans(limit=5)

        assert len(opened) == 1, (
            f"{len(opened)} connections opened for 120 operations — "
            f"the connection is not being reused"
        )

    def test_writes_are_fast(self):
        """Guards the checkpoint-per-query regression.

        Closing the last connection to a WAL database forces a checkpoint. When
        every query opened and closed its own connection, a single insert cost
        roughly 700 ms. The bound here is deliberately loose — it is catching a
        thousandfold regression, not measuring throughput.
        """
        report = {
            "scan_id": "PERF0", "media_type": "image", "filename": "p.jpg",
            "verdict": "AUTHENTIC", "risk_level": "MINIMAL", "fake_probability": 0.1,
            "authenticity_score": 90.0, "confidence": 0.8, "faces_detected": 1,
            "file_size_bytes": 10, "processing_ms": 1.0,
            "faces": [{"face_id": 1, "crop_preview": "x" * 40_000}],
        }
        db.save_scan(report)                       # warm

        start = time.perf_counter()
        for i in range(20):
            report["scan_id"] = f"PERF{i}"
            db.save_scan(report)
        per_write = (time.perf_counter() - start) / 20 * 1000

        assert per_write < 100, f"{per_write:.0f} ms per write — checkpoint regression"

    def test_commit_happens_on_success(self):
        db.save_scan({
            "scan_id": "COMMIT1", "media_type": "image", "filename": "c.jpg",
            "verdict": "FAKE", "risk_level": "HIGH", "fake_probability": 0.9,
            "authenticity_score": 10.0, "confidence": 0.8, "faces_detected": 1,
            "file_size_bytes": 10, "processing_ms": 1.0, "faces": [],
        })
        db.close()                                  # drop and reopen
        assert db.get_scan("COMMIT1") is not None

    def test_rollback_on_error_leaves_no_partial_write(self):
        db.save_scan({
            "scan_id": "ROLLBACK1", "media_type": "image", "filename": "a.jpg",
            "verdict": "FAKE", "risk_level": "HIGH", "fake_probability": 0.9,
            "authenticity_score": 10.0, "confidence": 0.8, "faces_detected": 1,
            "file_size_bytes": 10, "processing_ms": 1.0, "faces": [],
        })
        with pytest.raises(sqlite3.OperationalError):
            with db.session() as conn:
                conn.execute("DELETE FROM scans")
                conn.execute("SELECT * FROM a_table_that_does_not_exist")

        assert db.get_scan("ROLLBACK1") is not None, "delete should have rolled back"

    def test_changing_the_database_path_reopens(self, tmp_path, monkeypatch):
        """Tests swap cfg.DB_PATH per test; the cached connection must follow."""
        with db.session() as conn:
            original = conn

        monkeypatch.setattr(db.cfg, "DB_PATH", tmp_path / "other.db")
        db.init()
        with db.session() as conn:
            assert conn is not original


class TestSharedLock:
    def test_auth_and_scans_use_the_same_lock(self):
        """Two independent locks over one file is not serialisation."""
        import inspect
        source = inspect.getsource(auth)
        assert "threading.Lock()" not in source, "auth must not hold its own lock"
        assert "from database import session" in source

    def test_concurrent_writes_from_both_modules(self):
        """Scans and accounts written from several threads at once must all
        land, and the database must stay readable."""
        errors = []

        def write_scans(start):
            try:
                for i in range(start, start + 15):
                    db.save_scan({
                        "scan_id": f"T{i}", "media_type": "image", "filename": f"{i}.jpg",
                        "verdict": "AUTHENTIC", "risk_level": "MINIMAL",
                        "fake_probability": 0.1, "authenticity_score": 90.0,
                        "confidence": 0.8, "faces_detected": 1,
                        "file_size_bytes": 10, "processing_ms": 1.0, "faces": [],
                    })
            except Exception as exc:                       # noqa: BLE001
                errors.append(exc)

        def write_identities(start):
            try:
                for i in range(start, start + 15):
                    db.enroll_identity(f"person{i}", np.ones(128, dtype=np.float32))
            except Exception as exc:                       # noqa: BLE001
                errors.append(exc)

        threads = [
            threading.Thread(target=write_scans, args=(0,)),
            threading.Thread(target=write_scans, args=(100,)),
            threading.Thread(target=write_identities, args=(0,)),
            threading.Thread(target=write_identities, args=(100,)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        assert errors == [], f"concurrent writes raised: {errors}"
        assert db.stats()["total_scans"] == 30
        assert len(db.list_identities()) == 30

        with db.session() as conn:
            assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


class TestCorruptionRecovery:
    def test_unreadable_database_is_moved_aside(self, tmp_path, monkeypatch):
        broken = tmp_path / "omniguard.db"
        broken.write_bytes(b"SQLite format 3\x00" + b"\x00garbage" * 200)
        monkeypatch.setattr(db.cfg, "DB_PATH", broken)

        moved = db.init()

        assert moved is not None and "corrupt" in moved
        assert (tmp_path / moved).exists(), "the damaged file must be kept, not deleted"
        # A fresh, usable database now sits at the original path.
        assert db.stats()["total_scans"] == 0

    def test_healthy_database_is_left_alone(self, tmp_path, monkeypatch):
        path = tmp_path / "omniguard.db"
        monkeypatch.setattr(db.cfg, "DB_PATH", path)
        db.init()
        assert db.init() is None, "a healthy database must not be quarantined"

    def test_wal_mode_is_enabled(self):
        with db.session() as conn:
            assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"


class TestPlaceholderHistory:
    def test_placeholder_model_history_is_unverified_without_mutating_storage(self):
        report = {
            "scan_id": "OLD-DUMMY", "media_type": "image", "filename": "old.jpg",
            "verdict": "SUSPICIOUS", "risk_level": "HIGH", "fake_probability": 0.77,
            "authenticity_score": 23.0, "confidence": 0.54, "faces_detected": 1,
            "file_size_bytes": 10, "processing_ms": 2.0,
            "faces": [{"face_id": 0, "verdict": "SUSPICIOUS",
                       "fake_probability": 0.77, "heatmap_preview": "fake-heatmap",
                       "models": [{"arch": "dummy_testnet", "fake_probability": 0.77}]}],
            "findings": ["classifier verdict finding"],
        }
        db.save_scan(report)

        listed = db.recent_scans(limit=10)
        loaded = db.get_scan("OLD-DUMMY")
        stats = db.stats()

        assert listed[0]["verdict"] == "UNVERIFIED"
        assert listed[0]["fake_probability"] is None
        assert loaded["verdict"] == "UNVERIFIED"
        assert loaded["fake_probability"] is None
        assert loaded["faces"][0]["fake_probability"] is None
        assert loaded["faces"][0]["heatmap_preview"] is None
        assert "placeholder classifier" in loaded["findings"][0]["text"]
        assert stats["total_scans"] == 1
        assert stats["unverified_count"] == 1
        assert stats["suspicious"] == stats["fake"] == stats["authentic"] == 0

        with db.session() as conn:
            stored = conn.execute(
                "SELECT verdict, fake_probability FROM scans WHERE scan_id = ?",
                ("OLD-DUMMY",),
            ).fetchone()
        assert stored["verdict"] == "SUSPICIOUS"
        assert stored["fake_probability"] == 0.77

    def test_verdict_filter_uses_the_normalized_verdict(self):
        db.save_scan({
            "scan_id": "OLD-DUMMY", "media_type": "image", "filename": "old.jpg",
            "verdict": "SUSPICIOUS", "risk_level": "HIGH", "fake_probability": 0.77,
            "authenticity_score": 23.0, "confidence": 0.54, "faces_detected": 1,
            "file_size_bytes": 10, "processing_ms": 2.0,
            "models": [{"arch": "dummy_testnet"}], "faces": [],
        })
        assert db.recent_scans(verdict="SUSPICIOUS") == []
        assert db.recent_scans(verdict="UNVERIFIED")[0]["scan_id"] == "OLD-DUMMY"
