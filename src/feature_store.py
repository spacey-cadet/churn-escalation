"""
Free-tier feature store: SQLite standing in for the Redis (online) + partitioned
Parquet-on-S3 (offline) pair described in the roadmap.

- Online table: one row per entity_id, overwritten on every write (Redis semantics).
  Used for low-latency "current value" lookups at serving time.
- Offline table: append-only, full history retained, one row per (entity_id, ts).
  Used for point-in-time-correct training joins.
- Both writes happen in the same transaction (`write_features`), which is the
  free-tier version of the roadmap's "pipeline updates both stores concurrently."
- `reconcile()` samples entities and diffs the online "latest" value against what
  the offline history says was true as of now, catching the exact class of sync bug
  the roadmap calls out as a thing "most candidates don't mention."

This is one SQLite file, so it is trivially inspectable (`sqlite3 feature_store.sqlite`)
and requires no server process, unlike Redis/S3.
"""
import sqlite3
import json
from datetime import datetime, timezone
from pathlib import Path
from contextlib import contextmanager

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS online_features (
    entity_id TEXT PRIMARY KEY,
    features_json TEXT NOT NULL,
    as_of TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS offline_features (
    entity_id TEXT NOT NULL,
    features_json TEXT NOT NULL,
    ts TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_offline_entity_ts ON offline_features(entity_id, ts);

CREATE TABLE IF NOT EXISTS inference_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id TEXT,
    features_json TEXT NOT NULL,
    calibrated_score REAL NOT NULL,
    tier TEXT NOT NULL,
    model_version TEXT NOT NULL,
    served_by TEXT NOT NULL,       -- 'champion' or 'challenger'
    low_confidence INTEGER NOT NULL,
    request_ts TEXT NOT NULL
);
"""


@contextmanager
def _conn():
    config.FEATURE_STORE_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(config.FEATURE_STORE_DB))
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def write_features(entity_id: str, features: dict, ts: datetime | None = None) -> None:
    """Dual write: overwrite the online row, append to offline history. One call,
    one transaction -- this is the 'pipeline updates both stores concurrently' step."""
    ts = ts or datetime.now(timezone.utc)
    ts_str = ts.isoformat()
    payload = json.dumps(features)
    with _conn() as conn:
        conn.execute(
            "INSERT INTO online_features (entity_id, features_json, as_of) VALUES (?, ?, ?) "
            "ON CONFLICT(entity_id) DO UPDATE SET features_json=excluded.features_json, as_of=excluded.as_of",
            (entity_id, payload, ts_str),
        )
        conn.execute(
            "INSERT INTO offline_features (entity_id, features_json, ts) VALUES (?, ?, ?)",
            (entity_id, payload, ts_str),
        )


def read_online(entity_id: str) -> dict | None:
    """Low-latency 'current value' lookup -- the Redis-equivalent read path."""
    with _conn() as conn:
        row = conn.execute(
            "SELECT features_json, as_of FROM online_features WHERE entity_id = ?", (entity_id,)
        ).fetchone()
    if row is None:
        return None
    features, as_of = row
    return {"features": json.loads(features), "as_of": as_of}


def point_in_time_join(entity_id: str, as_of_timestamp: datetime) -> dict | None:
    """CORRECT training-join semantics: the most recent feature snapshot that was
    already true AT OR BEFORE as_of_timestamp -- never a value from the future
    relative to the label being joined to. This is what Feast's point-in-time joins
    do, and what prevents feature leakage / training-serving skew."""
    with _conn() as conn:
        row = conn.execute(
            "SELECT features_json, ts FROM offline_features "
            "WHERE entity_id = ? AND ts <= ? ORDER BY ts DESC LIMIT 1",
            (entity_id, as_of_timestamp.isoformat()),
        ).fetchone()
    if row is None:
        return None
    features, ts = row
    return {"features": json.loads(features), "ts": ts}


def reconcile(entity_ids: list[str], tolerance: float = 0.01) -> list[dict]:
    """Periodic reconciliation job: for each entity, compare the online 'latest'
    value against the offline 'as of right now' value. Any disagreement beyond
    `tolerance` on a numeric feature is a sync bug -- catch it before it causes
    silent training-serving skew."""
    now = datetime.now(timezone.utc)
    mismatches = []
    for entity_id in entity_ids:
        online = read_online(entity_id)
        offline = point_in_time_join(entity_id, now)
        if online is None or offline is None:
            continue
        for key, online_val in online["features"].items():
            offline_val = offline["features"].get(key)
            if isinstance(online_val, (int, float)) and isinstance(offline_val, (int, float)):
                if abs(online_val - offline_val) > tolerance:
                    mismatches.append({
                        "entity_id": entity_id, "feature": key,
                        "online_value": online_val, "offline_value": offline_val,
                    })
    return mismatches


def log_inference(entity_id, features: dict, calibrated_score: float, tier: str,
                   model_version: str, served_by: str, low_confidence: bool) -> None:
    """Every live prediction gets logged here. This is what drift_monitor.py reads
    to run the KS test / PSI check against the training baseline, and what backs
    the champion/challenger comparison in production."""
    with _conn() as conn:
        conn.execute(
            "INSERT INTO inference_log (entity_id, features_json, calibrated_score, tier, "
            "model_version, served_by, low_confidence, request_ts) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (entity_id, json.dumps(features), calibrated_score, tier, model_version,
             served_by, int(low_confidence), datetime.now(timezone.utc).isoformat()),
        )


def read_inference_log(limit: int = 10_000):
    """Returns the logged inference requests as a list of dicts, most recent first."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT entity_id, features_json, calibrated_score, tier, model_version, "
            "served_by, low_confidence, request_ts FROM inference_log "
            "ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    out = []
    for entity_id, features_json, score, tier, version, served_by, low_conf, ts in rows:
        out.append({
            "entity_id": entity_id, "features": json.loads(features_json),
            "calibrated_score": score, "tier": tier, "model_version": version,
            "served_by": served_by, "low_confidence": bool(low_conf), "request_ts": ts,
        })
    return out
