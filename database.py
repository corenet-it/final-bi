from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterable

from config import DEFAULT_DB_PATH
from logging_utils import get_logger


logger = get_logger("database")


SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS videos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_url TEXT NOT NULL,
    title TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS roi_zones (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    points_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id INTEGER,
    roi_zone_id INTEGER,
    frame_number INTEGER NOT NULL,
    timestamp_sec REAL NOT NULL,
    detected_at TEXT NOT NULL,
    class_name TEXT NOT NULL,
    confidence REAL NOT NULL,
    track_id INTEGER,
    vehicle_make TEXT,
    vehicle_model TEXT,
    vehicle_color TEXT,
    vehicle_body_type TEXT,
    vehicle_model_confidence REAL,
    vehicle_crop_path TEXT,
    vehicle_recognition_source TEXT,
    x1 REAL NOT NULL,
    y1 REAL NOT NULL,
    x2 REAL NOT NULL,
    y2 REAL NOT NULL,
    cx REAL NOT NULL,
    cy REAL NOT NULL,
    FOREIGN KEY(video_id) REFERENCES videos(id),
    FOREIGN KEY(roi_zone_id) REFERENCES roi_zones(id)
);

CREATE TABLE IF NOT EXISTS track_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    detection_id INTEGER,
    event_type TEXT NOT NULL,
    event_value TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(detection_id) REFERENCES detections(id)
);

CREATE TABLE IF NOT EXISTS chat_queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_detections_time ON detections(detected_at);
CREATE INDEX IF NOT EXISTS idx_detections_class ON detections(class_name);
CREATE INDEX IF NOT EXISTS idx_detections_track ON detections(track_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_detection_track
ON detections(video_id, roi_zone_id, class_name, track_id)
WHERE track_id IS NOT NULL;
"""


def utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


def ensure_schema_migrations(conn: sqlite3.Connection) -> None:
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(detections)").fetchall()}
    new_columns = {
        "vehicle_make": "TEXT",
        "vehicle_model": "TEXT",
        "vehicle_color": "TEXT",
        "vehicle_body_type": "TEXT",
        "vehicle_model_confidence": "REAL",
        "vehicle_crop_path": "TEXT",
        "vehicle_recognition_source": "TEXT",
    }
    for name, column_type in new_columns.items():
        if name not in columns:
            conn.execute(f"ALTER TABLE detections ADD COLUMN {name} {column_type}")


@contextmanager
def connect(db_path: Path | str = DEFAULT_DB_PATH):
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: Path | str = DEFAULT_DB_PATH) -> None:
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)
        ensure_schema_migrations(conn)
    logger.info("db_initialized path=%s", db_path)


def create_video(source_url: str, title: str | None = None, db_path: Path | str = DEFAULT_DB_PATH) -> int:
    init_db(db_path)
    with connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO videos(source_url, title, created_at) VALUES (?, ?, ?)",
            (source_url, title, utc_now()),
        )
        return int(cur.lastrowid)


def upsert_roi_zone(name: str, points_json: str, db_path: Path | str = DEFAULT_DB_PATH) -> int:
    init_db(db_path)
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO roi_zones(name, points_json, created_at)
            VALUES (?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET points_json = excluded.points_json
            """,
            (name, points_json, utc_now()),
        )
        row = conn.execute("SELECT id FROM roi_zones WHERE name = ?", (name,)).fetchone()
        return int(row["id"])


def get_roi_zones(db_path: Path | str = DEFAULT_DB_PATH) -> list[dict]:
    init_db(db_path)
    with connect(db_path) as conn:
        rows = conn.execute("SELECT * FROM roi_zones ORDER BY name").fetchall()
        return [dict(row) for row in rows]


def insert_detection(
    *,
    video_id: int | None,
    roi_zone_id: int | None,
    frame_number: int,
    timestamp_sec: float,
    detected_at: str,
    class_name: str,
    confidence: float,
    track_id: int | None,
    bbox: tuple[float, float, float, float],
    center: tuple[float, float],
    db_path: Path | str = DEFAULT_DB_PATH,
    vehicle_make: str | None = None,
    vehicle_model: str | None = None,
    vehicle_color: str | None = None,
    vehicle_body_type: str | None = None,
    vehicle_model_confidence: float | None = None,
    vehicle_crop_path: str | None = None,
    vehicle_recognition_source: str | None = None,
) -> int:
    init_db(db_path)
    with connect(db_path) as conn:
        before = conn.total_changes
        cur = conn.execute(
            """
            INSERT OR IGNORE INTO detections(
                video_id, roi_zone_id, frame_number, timestamp_sec, detected_at,
                class_name, confidence, track_id, vehicle_make, vehicle_model,
                vehicle_color, vehicle_body_type, vehicle_model_confidence,
                vehicle_crop_path, vehicle_recognition_source, x1, y1, x2, y2, cx, cy
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                video_id,
                roi_zone_id,
                frame_number,
                timestamp_sec,
                detected_at,
                class_name,
                confidence,
                track_id,
                vehicle_make,
                vehicle_model,
                vehicle_color,
                vehicle_body_type,
                vehicle_model_confidence,
                vehicle_crop_path,
                vehicle_recognition_source,
                bbox[0],
                bbox[1],
                bbox[2],
                bbox[3],
                center[0],
                center[1],
            ),
        )
        if conn.total_changes == before:
            logger.info("detection_duplicate_ignored class=%s track_id=%s", class_name, track_id)
            return 0
        logger.info("detection_inserted id=%s class=%s track_id=%s", cur.lastrowid, class_name, track_id)
        return int(cur.lastrowid)


def insert_many_detections(records: Iterable[dict], db_path: Path | str = DEFAULT_DB_PATH) -> int:
    init_db(db_path)
    rows = list(records)
    if not rows:
        return 0
    defaults = {
        "vehicle_make": None,
        "vehicle_model": None,
        "vehicle_color": None,
        "vehicle_body_type": None,
        "vehicle_model_confidence": None,
        "vehicle_crop_path": None,
        "vehicle_recognition_source": None,
    }
    for row in rows:
        for key, value in defaults.items():
            row.setdefault(key, value)
    with connect(db_path) as conn:
        before = conn.total_changes
        conn.executemany(
            """
            INSERT OR IGNORE INTO detections(
                video_id, roi_zone_id, frame_number, timestamp_sec, detected_at,
                class_name, confidence, track_id, vehicle_make, vehicle_model,
                vehicle_color, vehicle_body_type, vehicle_model_confidence,
                vehicle_crop_path, vehicle_recognition_source, x1, y1, x2, y2, cx, cy
            )
            VALUES (:video_id, :roi_zone_id, :frame_number, :timestamp_sec, :detected_at,
                    :class_name, :confidence, :track_id, :vehicle_make, :vehicle_model,
                    :vehicle_color, :vehicle_body_type, :vehicle_model_confidence,
                    :vehicle_crop_path, :vehicle_recognition_source, :x1, :y1, :x2, :y2, :cx, :cy)
            """,
            rows,
        )
        inserted = conn.total_changes - before
        ignored = len(rows) - inserted
        logger.info("detections_batch_inserted inserted=%s ignored=%s requested=%s", inserted, ignored, len(rows))
        return inserted


def save_chat_query(question: str, answer: str, db_path: Path | str = DEFAULT_DB_PATH) -> None:
    init_db(db_path)
    with connect(db_path) as conn:
        conn.execute(
            "INSERT INTO chat_queries(question, answer, created_at) VALUES (?, ?, ?)",
            (question, answer, utc_now()),
        )
    logger.info("chat_saved question_chars=%s answer_chars=%s", len(question), len(answer))


def fetch_chat_history(db_path: Path | str = DEFAULT_DB_PATH, limit: int = 8) -> list[dict]:
    init_db(db_path)
    with connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT question, answer, created_at
            FROM chat_queries
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in reversed(rows)]


def clear_chat_history(db_path: Path | str = DEFAULT_DB_PATH) -> None:
    init_db(db_path)
    with connect(db_path) as conn:
        conn.execute("DELETE FROM chat_queries")
    logger.info("chat_history_cleared path=%s", db_path)


def fetch_detections(db_path: Path | str = DEFAULT_DB_PATH) -> list[dict]:
    init_db(db_path)
    with connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT d.*, rz.name AS roi_name, v.source_url
            FROM detections d
            LEFT JOIN roi_zones rz ON rz.id = d.roi_zone_id
            LEFT JOIN videos v ON v.id = d.video_id
            ORDER BY d.detected_at DESC, d.frame_number DESC
            """
        ).fetchall()
        return [dict(row) for row in rows]
