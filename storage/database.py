# ─────────────────────────────────────────────
#  storage/database.py
#  SQLite schema + all CRUD helpers
#  Privacy: raw face images NEVER stored here
# ─────────────────────────────────────────────

import sqlite3
import json
import hashlib
import time
from datetime import datetime
from pathlib import Path
import numpy as np
from loguru import logger

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import DB_PATH, MAX_FACE_DB_SIZE


def get_connection() -> sqlite3.Connection:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")   # better concurrent reads
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Create all tables if they don't exist."""
    conn = get_connection()
    cur = conn.cursor()

    # ── Face embeddings (privacy: no raw image) ──
    cur.execute("""
        CREATE TABLE IF NOT EXISTS face_embeddings (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            face_hash       TEXT    UNIQUE NOT NULL,   -- SHA256 of embedding bytes
            embedding_blob  BLOB    NOT NULL,           -- numpy float32 array
            first_seen      REAL    NOT NULL,           -- unix timestamp
            last_seen       REAL    NOT NULL,
            seen_count      INTEGER DEFAULT 1,
            alert_issued    INTEGER DEFAULT 0,          -- 1 = yellow warning sent
            source_video    TEXT,
            notes           TEXT
        )
    """)

    # ── Tracked objects per frame ──────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS tracked_objects (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            track_id        INTEGER NOT NULL,
            frame_number    INTEGER NOT NULL,
            timestamp       REAL    NOT NULL,
            x1              REAL, y1 REAL, x2 REAL, y2 REAL,
            confidence      REAL,
            zone_id         TEXT,
            speed_px_s      REAL,
            source_video    TEXT
        )
    """)

    # ── Events (behaviours detected) ──────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id        TEXT    UNIQUE NOT NULL,   -- UUID
            timestamp       REAL    NOT NULL,
            event_type      TEXT    NOT NULL,           -- loitering, running, fighting…
            alert_level     TEXT    NOT NULL,           -- GREEN/YELLOW/ORANGE/RED
            track_id        INTEGER,
            face_hash       TEXT,                       -- linked face if identified
            zone_id         TEXT,
            description     TEXT,                       -- Worker agent natural language
            manager_verdict TEXT,                       -- Manager agent classification
            raw_json        TEXT,                       -- Full context JSON blob
            source_video    TEXT,
            acknowledged    INTEGER DEFAULT 0
        )
    """)

    # ── Audit trail ───────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS audit_trail (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       REAL    NOT NULL,
            action          TEXT    NOT NULL,
            details         TEXT,
            source          TEXT    DEFAULT 'system'
        )
    """)

    # ── Indexes ───────────────────────────────
    cur.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp   ON events(timestamp)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_events_level       ON events(alert_level)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_tracked_track_id   ON tracked_objects(track_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_face_hash          ON face_embeddings(face_hash)")

    conn.commit()
    conn.close()
    logger.info(f"Database initialised at {DB_PATH}")


# ══════════════════════════════════════════════
#  FACE RE-ID
# ══════════════════════════════════════════════

def embedding_to_blob(embedding: np.ndarray) -> bytes:
    return embedding.astype(np.float32).tobytes()

def blob_to_embedding(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)

def hash_embedding(embedding: np.ndarray) -> str:
    return hashlib.sha256(embedding.astype(np.float32).tobytes()).hexdigest()[:16]

def get_all_embeddings() -> list[dict]:
    """Return all stored face embeddings for similarity search."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT face_hash, embedding_blob, seen_count, alert_issued, first_seen FROM face_embeddings"
    ).fetchall()
    conn.close()
    return [
        {
            "face_hash":     r["face_hash"],
            "embedding":     blob_to_embedding(r["embedding_blob"]),
            "seen_count":    r["seen_count"],
            "alert_issued":  r["alert_issued"],
            "first_seen":    r["first_seen"],
        }
        for r in rows
    ]

def upsert_face(embedding: np.ndarray, source_video: str = "") -> dict:
    """
    Insert new face or update existing.
    Returns: {"face_hash": str, "is_new": bool, "seen_count": int, "alert_issued": bool}
    """
    face_hash = hash_embedding(embedding)
    blob      = embedding_to_blob(embedding)
    now       = time.time()
    conn      = get_connection()

    existing = conn.execute(
        "SELECT id, seen_count, alert_issued FROM face_embeddings WHERE face_hash = ?",
        (face_hash,)
    ).fetchone()

    if existing:
        new_count = existing["seen_count"] + 1
        conn.execute(
            "UPDATE face_embeddings SET last_seen=?, seen_count=? WHERE face_hash=?",
            (now, new_count, face_hash)
        )
        conn.commit()
        conn.close()
        return {
            "face_hash":    face_hash,
            "is_new":       False,
            "seen_count":   new_count,
            "alert_issued": bool(existing["alert_issued"]),
        }
    else:
        # Prune if over limit
        count = conn.execute("SELECT COUNT(*) FROM face_embeddings").fetchone()[0]
        if count >= MAX_FACE_DB_SIZE:
            conn.execute(
                "DELETE FROM face_embeddings WHERE id IN "
                "(SELECT id FROM face_embeddings ORDER BY last_seen ASC LIMIT 100)"
            )

        conn.execute(
            "INSERT INTO face_embeddings (face_hash, embedding_blob, first_seen, last_seen, source_video) "
            "VALUES (?, ?, ?, ?, ?)",
            (face_hash, blob, now, now, source_video)
        )
        conn.commit()
        conn.close()
        return {
            "face_hash":    face_hash,
            "is_new":       True,
            "seen_count":   1,
            "alert_issued": False,
        }

def mark_alert_issued(face_hash: str):
    conn = get_connection()
    conn.execute("UPDATE face_embeddings SET alert_issued=1 WHERE face_hash=?", (face_hash,))
    conn.commit()
    conn.close()


# ══════════════════════════════════════════════
#  EVENTS
# ══════════════════════════════════════════════

def insert_event(event: dict) -> int:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO events
            (event_id, timestamp, event_type, alert_level, track_id, face_hash,
             zone_id, description, manager_verdict, raw_json, source_video)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        event.get("event_id"),
        event.get("timestamp", time.time()),
        event.get("event_type"),
        event.get("alert_level"),
        event.get("track_id"),
        event.get("face_hash"),
        event.get("zone_id"),
        event.get("description"),
        event.get("manager_verdict"),
        json.dumps(event.get("raw_json", {})),
        event.get("source_video"),
    ))
    row_id = cur.lastrowid
    conn.commit()
    conn.close()
    return row_id

def get_recent_events(limit: int = 50, level: str = None) -> list[dict]:
    conn = get_connection()
    if level:
        rows = conn.execute(
            "SELECT * FROM events WHERE alert_level=? ORDER BY timestamp DESC LIMIT ?",
            (level, limit)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM events ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_event_counts_by_level() -> dict:
    conn = get_connection()
    rows = conn.execute(
        "SELECT alert_level, COUNT(*) as cnt FROM events GROUP BY alert_level"
    ).fetchall()
    conn.close()
    return {r["alert_level"]: r["cnt"] for r in rows}


# ══════════════════════════════════════════════
#  TRACKED OBJECTS
# ══════════════════════════════════════════════

def insert_tracked_object(obj: dict):
    conn = get_connection()
    conn.execute("""
        INSERT INTO tracked_objects
            (track_id, frame_number, timestamp, x1, y1, x2, y2, confidence, zone_id, speed_px_s, source_video)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        obj["track_id"], obj["frame_number"], obj["timestamp"],
        obj["x1"], obj["y1"], obj["x2"], obj["y2"],
        obj.get("confidence"), obj.get("zone_id"), obj.get("speed_px_s"), obj.get("source_video")
    ))
    conn.commit()
    conn.close()


# ══════════════════════════════════════════════
#  AUDIT TRAIL
# ══════════════════════════════════════════════

def audit_log(action: str, details: str = "", source: str = "system"):
    conn = get_connection()
    conn.execute(
        "INSERT INTO audit_trail (timestamp, action, details, source) VALUES (?, ?, ?, ?)",
        (time.time(), action, details, source)
    )
    conn.commit()
    conn.close()

def get_audit_trail(limit: int = 100) -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM audit_trail ORDER BY timestamp DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


if __name__ == "__main__":
    init_db()
    print("✅ Database initialised successfully.")
