"""SQLite persistence for scan history and the enrolled-identity gallery.

SQLite ships with Python, so there is no database server to install or run.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone

import numpy as np

import config as cfg

# One re-entrant lock guards the whole file. auth.py shares it rather than
# holding its own: two independent locks over the same database is not
# serialisation, it just looks like it.
_lock = threading.RLock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS scans (
    scan_id        TEXT PRIMARY KEY,
    created_at     TEXT NOT NULL,
    filename       TEXT NOT NULL,
    media_type     TEXT NOT NULL,
    verdict        TEXT NOT NULL,
    risk_level     TEXT,
    fake_probability  REAL,
    authenticity_score REAL,
    confidence     REAL,
    faces_detected INTEGER DEFAULT 0,
    file_size_bytes INTEGER,
    processing_ms  REAL,
    report_json    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_scans_created ON scans(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_scans_verdict ON scans(verdict);

CREATE TABLE IF NOT EXISTS identities (
    name        TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL,
    embedding   BLOB NOT NULL,
    sample_count INTEGER DEFAULT 1,
    notes       TEXT
);
"""


# One long-lived connection, reused for every query and guarded by _lock.
# `_conn_path` records which database it was opened against, so a change to
# cfg.DB_PATH (tests do this per-test) transparently reopens.
_conn: sqlite3.Connection | None = None
_conn_path = None


def _connection() -> sqlite3.Connection:
    """The shared connection, opened on first use."""
    global _conn, _conn_path

    if _conn is not None and _conn_path == cfg.DB_PATH:
        return _conn

    if _conn is not None:                      # database path changed
        try:
            _conn.close()
        except sqlite3.Error:
            pass

    conn = sqlite3.connect(cfg.DB_PATH, check_same_thread=False, timeout=15)
    conn.row_factory = sqlite3.Row
    # Applied once per connection rather than once per query.
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA foreign_keys = ON")

    _conn, _conn_path = conn, cfg.DB_PATH
    return conn


@contextmanager
def session():
    """Run a statement against the shared connection, committing on success.

    Two earlier versions of this were both wrong, in opposite directions:

    1. `with sqlite3.connect(...) as conn` commits but does NOT close, so every
       query leaked a file handle. Hundreds of live handles against one file is
       how the database once ended up with a corrupt page-1 header.

    2. Opening and closing a connection per query fixed the leak but made every
       write cost ~700 ms. Closing the *last* connection to a WAL database
       forces a checkpoint — measured at 391 ms — so each query paid for one.

    Holding exactly one connection avoids both: nothing accumulates, and the
    checkpoint happens on SQLite's own schedule instead of on every call.
    The lock still serialises access, which is what `check_same_thread=False`
    requires.
    """
    with _lock:
        conn = _connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise


def close() -> None:
    """Close the shared connection. Used at shutdown and between tests."""
    global _conn, _conn_path
    with _lock:
        if _conn is not None:
            try:
                _conn.close()
            finally:
                _conn, _conn_path = None, None


def _quarantine_unreadable() -> str | None:
    """Move an unreadable database aside so the app can start with a fresh one.

    Refusing to start at all would take the whole tool down over a cache of
    past scans. The old file is renamed rather than deleted, so nothing is
    silently destroyed and it can still be examined.
    """
    if not cfg.DB_PATH.exists():
        return None

    conn = None
    try:
        conn = sqlite3.connect(cfg.DB_PATH, timeout=5)
        conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
        return None
    except sqlite3.DatabaseError:
        pass
    finally:
        # Closed in `finally`, not after the query: if the probe raises, an
        # unclosed handle keeps the file locked, and on Windows the rename
        # below then fails with PermissionError.
        if conn is not None:
            conn.close()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    broken = cfg.DB_PATH.with_name(f"{cfg.DB_PATH.stem}.corrupt-{stamp}.db")
    try:
        shutil.move(str(cfg.DB_PATH), str(broken))
    except OSError as exc:
        # Another process still holds it. Starting with a broken database is
        # worse than saying so, so this is raised rather than swallowed.
        raise RuntimeError(
            f"The database at {cfg.DB_PATH} is unreadable and could not be "
            f"moved aside ({exc}). Close any other running copy of the server "
            f"and start again."
        ) from exc

    for suffix in ("-wal", "-shm"):
        cfg.DB_PATH.with_name(cfg.DB_PATH.name + suffix).unlink(missing_ok=True)
    return broken.name


def init() -> str | None:
    """Create the schema. Returns the quarantined filename if one was moved."""
    # Release any handle on the old file before the probe tries to rename it.
    close()
    moved = _quarantine_unreadable()
    with session() as conn:
        conn.executescript(_SCHEMA)
        # v2: an honest out-of-domain result has no neural probability. Older
        # databases declared both score columns NOT NULL, so rebuild this small
        # history table once while preserving every report.
        columns = {row[1]: row for row in conn.execute("PRAGMA table_info(scans)")}
        if (columns.get("fake_probability", (None,) * 4)[3]
                or columns.get("authenticity_score", (None,) * 4)[3]):
            conn.executescript("""
                CREATE TABLE scans_v2 (
                    scan_id TEXT PRIMARY KEY, created_at TEXT NOT NULL,
                    filename TEXT NOT NULL, media_type TEXT NOT NULL,
                    verdict TEXT NOT NULL, risk_level TEXT,
                    fake_probability REAL, authenticity_score REAL,
                    confidence REAL, faces_detected INTEGER DEFAULT 0,
                    file_size_bytes INTEGER, processing_ms REAL,
                    report_json TEXT NOT NULL
                );
                INSERT INTO scans_v2 SELECT * FROM scans;
                DROP TABLE scans;
                ALTER TABLE scans_v2 RENAME TO scans;
                CREATE INDEX idx_scans_created ON scans(created_at DESC);
                CREATE INDEX idx_scans_verdict ON scans(verdict);
            """)
    return moved


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------------ scans
def save_scan(report: dict) -> None:
    """Persist a report. The face embeddings are stripped - they are large and
    only meaningful during the request that produced them."""
    slim = json.loads(json.dumps(report, default=str))
    for face in slim.get("faces", []):
        face.pop("_embedding", None)

    with session() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO scans
               (scan_id, created_at, filename, media_type, verdict, risk_level,
                fake_probability, authenticity_score, confidence, faces_detected,
                file_size_bytes, processing_ms, report_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                report["scan_id"], _now(), report["filename"], report["media_type"],
                report["verdict"], report.get("risk_level"),
                report["fake_probability"], report["authenticity_score"],
                report.get("confidence"),
                report.get("faces_detected", report.get("people_detected", 0)),
                report.get("file_size_bytes"), report.get("processing_ms"),
                json.dumps(slim, default=str),
            ),
        )


def _uses_placeholder_classifier(value) -> bool:
    """Return true if any model metadata in a persisted report is a test stub."""
    if isinstance(value, dict):
        arch = str(value.get("arch", "")).lower()
        if value.get("dummy") is True or arch.startswith("dummy_"):
            return True
        return any(_uses_placeholder_classifier(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_uses_placeholder_classifier(item) for item in value)
    return False


def _is_placeholder_report(report: dict) -> bool:
    return (str(report.get("engine", "")).lower().startswith("dummy")
            or _uses_placeholder_classifier(report))


def _mark_placeholder_unverified(report: dict) -> dict:
    """Mask fake classifier outputs in old reports without changing stored data."""
    report = dict(report)
    report.update({
        "verdict": "UNVERIFIED",
        "risk_level": "UNKNOWN",
        "fake_probability": None,
        "authenticity_score": None,
        "confidence": None,
        "model_agreement": None,
        "models": [],
    })
    warning = (
        "This historical result used an untrained placeholder classifier. "
        "Its verdict and score are not valid; run it again after loading trained models."
    )
    model_findings = (
        "face manipulation detected", "possible manipulation",
        "no manipulation signature detected by the model ensemble",
        "models disagree on this sample",
    )
    preserved_findings = []
    for finding in report.get("findings", []):
        text = finding.get("text", "") if isinstance(finding, dict) else str(finding)
        if text and not text.lower().startswith(model_findings):
            preserved_findings.append(
                finding if isinstance(finding, dict)
                else {"severity": "info", "text": text}
            )
    report["findings"] = [{"severity": "medium", "text": warning}, *preserved_findings]
    for face in report.get("faces", []):
        face.update({
            "verdict": "UNVERIFIED", "fake_probability": None,
            "authenticity_score": None, "confidence": None,
            "model_agreement": None, "models": [], "heatmap_preview": None,
        })
    if "frame_scores" in report:
        report["frame_scores"] = [
            {**frame, "verdict": "UNVERIFIED", "fake_probability": None,
             "authenticity_score": None, "confidence": None, "models": []}
            for frame in report["frame_scores"]
        ]
    report["most_suspicious_frame"] = None
    report["tracks"] = [
        {key: value for key, value in track.items()
         if key not in {"mean_fake_probability", "max_fake_probability", "verdict"}}
        for track in report.get("tracks", [])
    ]
    return report


def recent_scans(limit: int = 20, offset: int = 0,
                 verdict: str | None = None) -> list[dict]:
    query = ("SELECT scan_id, created_at, filename, media_type, verdict, risk_level,"
             " fake_probability, authenticity_score, confidence, faces_detected,"
             " file_size_bytes, processing_ms, report_json FROM scans"
             " ORDER BY created_at DESC")

    with session() as conn:
        rows = conn.execute(query).fetchall()

    scans = []
    for row in rows:
        scan = dict(row)
        report = json.loads(scan.pop("report_json"))
        if _is_placeholder_report(report):
            scan.update({
                "verdict": "UNVERIFIED", "risk_level": "UNKNOWN",
                "fake_probability": None, "authenticity_score": None,
                "confidence": None,
            })
        if verdict and scan["verdict"] != verdict.upper():
            continue
        scans.append(scan)
    return scans[offset:offset + limit]


def get_scan(scan_id: str) -> dict | None:
    with session() as conn:
        row = conn.execute(
            "SELECT report_json FROM scans WHERE scan_id = ?", (scan_id,)
        ).fetchone()
    if not row:
        return None
    report = json.loads(row["report_json"])
    return _mark_placeholder_unverified(report) if _is_placeholder_report(report) else report


def delete_scan(scan_id: str) -> bool:
    with session() as conn:
        return conn.execute("DELETE FROM scans WHERE scan_id = ?", (scan_id,)).rowcount > 0


def stats() -> dict:
    """Aggregates that drive the dashboard tiles and the donut chart."""
    with session() as conn:
        rows = conn.execute(
            "SELECT verdict, media_type, processing_ms, created_at, report_json FROM scans"
        ).fetchall()

    total = len(rows)
    verified = []
    unverified = 0
    for row in rows:
        if _is_placeholder_report(json.loads(row["report_json"])):
            unverified += 1
        else:
            verified.append(row)

    by_verdict: dict[str, int] = {}
    for row in verified:
        by_verdict[row["verdict"]] = by_verdict.get(row["verdict"], 0) + 1
    by_type = {
        kind: sum(1 for row in rows if row["media_type"] == kind)
        for kind in {row["media_type"] for row in rows}
    }
    durations = [row["processing_ms"] for row in verified if row["processing_ms"] is not None]
    avg_ms = sum(durations) / len(durations) if durations else 0.0
    daily: dict[str, dict] = {}
    for row in verified:
        day = row["created_at"][:10]
        entry = daily.setdefault(day, {"day": day, "total": 0, "fake": 0})
        entry["total"] += 1
        if row["verdict"] == "FAKE":
            entry["fake"] += 1
    trend = sorted(daily.values(), key=lambda entry: entry["day"], reverse=True)[:14]

    authentic = by_verdict.get("AUTHENTIC", 0)
    suspicious = by_verdict.get("SUSPICIOUS", 0)
    fake = by_verdict.get("FAKE", 0)
    pct = lambda n: round(n / len(verified) * 100, 1) if verified else 0.0

    return {
        "total_scans": total,
        "verified_scans": len(verified),
        "unverified_count": unverified,
        "authentic": authentic,
        "suspicious": suspicious,
        "fake": fake,
        "authentic_pct": pct(authentic),
        "suspicious_pct": pct(suspicious),
        "fake_pct": pct(fake),
        "by_media_type": by_type,
        "avg_processing_ms": round(avg_ms, 1),
        "trend": list(reversed(trend)),
    }


# ------------------------------------------------------------------- identities
def enroll_identity(name: str, embedding: np.ndarray, notes: str | None = None) -> dict:
    """Add or update a known face. Re-enrolling averages the embeddings, which
    makes the identity more robust across pose and lighting."""
    vec = np.asarray(embedding, dtype=np.float32)

    with session() as conn:
        row = conn.execute(
            "SELECT embedding, sample_count FROM identities WHERE name = ?", (name,)
        ).fetchone()

        if row:
            existing = np.frombuffer(row["embedding"], dtype=np.float32)
            n = row["sample_count"]
            vec = (existing * n + vec) / (n + 1)
            count = n + 1
        else:
            count = 1

        conn.execute(
            """INSERT OR REPLACE INTO identities (name, created_at, embedding, sample_count, notes)
               VALUES (?,?,?,?,?)""",
            (name, _now(), vec.astype(np.float32).tobytes(), count, notes),
        )

    return {"name": name, "sample_count": count, "dimensions": int(vec.size)}


def load_gallery() -> dict[str, np.ndarray]:
    with session() as conn:
        rows = conn.execute("SELECT name, embedding FROM identities").fetchall()
    return {r["name"]: np.frombuffer(r["embedding"], dtype=np.float32) for r in rows}


def list_identities() -> list[dict]:
    with session() as conn:
        return [
            dict(r) for r in conn.execute(
                "SELECT name, created_at, sample_count, notes FROM identities ORDER BY name"
            ).fetchall()
        ]


def delete_identity(name: str) -> bool:
    with session() as conn:
        return conn.execute("DELETE FROM identities WHERE name = ?", (name,)).rowcount > 0
