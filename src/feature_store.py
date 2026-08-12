"""
AWS-backed feature store: DynamoDB standing in for the Redis (online) + S3 (offline)
pair described in the roadmap -- and now also standing in for the SQLite file this
repo used for its free-tier build.

This is a drop-in replacement: every function keeps its old name and signature, so
nothing outside this file needs to change. src/serving/app.py, src/drift_monitor.py,
scripts/*, and tests/test_pipeline.py all call these functions exactly as before.

- Online table: one item per entity_id, overwritten on every write (Redis semantics).
- Offline table: append-only, one item per (entity_id, ts). Point-in-time join is a
  native DynamoDB Query (entity_id = ? AND ts <= ?, sorted descending, limit 1) --
  no full-table scan, same guarantee the SQL version gave.
- Inference log: single logical partition ("LOG") with a time-ordered sort key, so
  "most recent N" is a Query, not a Scan. This IS a real, documented scaling ceiling
  (a single DynamoDB partition tops out around 1000 WCU/3000 RCU) -- completely fine
  at hobby/portfolio traffic, and worth revisiting with a sharded key (e.g.
  "LOG#<hour>") the moment sustained high write throughput is a real requirement.
  Flagging it here rather than hiding it, in the spirit of this repo's existing
  free-tier substitution table.
- Feature/score payloads are stored as a JSON string attribute, not a native
  DynamoDB Map -- this sidesteps DynamoDB's float-vs-Decimal marshalling entirely
  and keeps this file close to line-for-line identical to the SQLite version.
"""
import json
import os
import uuid
from datetime import datetime, timezone

import boto3
from boto3.dynamodb.conditions import Key

ONLINE_TABLE = os.environ.get("DYNAMODB_ONLINE_TABLE", "churn-online-features")
OFFLINE_TABLE = os.environ.get("DYNAMODB_OFFLINE_TABLE", "churn-offline-features")
LOG_TABLE = os.environ.get("DYNAMODB_LOG_TABLE", "churn-inference-log")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

_dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)


def _online_table():
    return _dynamodb.Table(ONLINE_TABLE)


def _offline_table():
    return _dynamodb.Table(OFFLINE_TABLE)


def _log_table():
    return _dynamodb.Table(LOG_TABLE)


def write_features(entity_id: str, features: dict, ts: datetime | None = None) -> None:
    """Dual write: overwrite the online item, append to offline history. Two
    put_item calls rather than one SQLite transaction -- if you need atomicity
    across both tables, wrap this in dynamodb.meta.client.transact_write_items
    instead. Not done here: a partial write (offline succeeds, online fails)
    just means a momentarily stale 'current value' read, not corrupted
    history, and that tradeoff keeps this function simple."""
    ts = ts or datetime.now(timezone.utc)
    ts_str = ts.isoformat()
    payload = json.dumps(features)

    _online_table().put_item(Item={
        "entity_id": entity_id,
        "features_json": payload,
        "as_of": ts_str,
    })
    _offline_table().put_item(Item={
        "entity_id": entity_id,
        "ts": ts_str,
        "features_json": payload,
    })


def read_online(entity_id: str) -> dict | None:
    """Low-latency 'current value' lookup -- the Redis-equivalent read path."""
    resp = _online_table().get_item(Key={"entity_id": entity_id})
    item = resp.get("Item")
    if item is None:
        return None
    return {"features": json.loads(item["features_json"]), "as_of": item["as_of"]}


def point_in_time_join(entity_id: str, as_of_timestamp: datetime) -> dict | None:
    """CORRECT training-join semantics: the most recent feature snapshot that was
    already true AT OR BEFORE as_of_timestamp -- never a value from the future
    relative to the label being joined to. A native DynamoDB Query with a sorted
    key condition does the same job the SQL WHERE ... ORDER BY ts DESC LIMIT 1
    did, and is the single most important behavior to keep correct in this file --
    see tests/test_pipeline.py's future-leakage test."""
    resp = _offline_table().query(
        KeyConditionExpression=Key("entity_id").eq(entity_id) & Key("ts").lte(as_of_timestamp.isoformat()),
        ScanIndexForward=False,
        Limit=1,
    )
    items = resp.get("Items", [])
    if not items:
        return None
    item = items[0]
    return {"features": json.loads(item["features_json"]), "ts": item["ts"]}


def reconcile(entity_ids: list[str], tolerance: float = 0.01) -> list[dict]:
    """Unchanged logic from the SQLite version -- it only ever calls read_online
    and point_in_time_join above, both of which now hit DynamoDB instead."""
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
    """Every live prediction gets logged here, same as before. Stored under a
    single logical partition ('LOG') with a time-ordered sort key so
    read_inference_log can Query instead of Scan -- see the module docstring
    for the scaling caveat that comes with that choice. calibrated_score is
    stored as a string (not DynamoDB's native Number type) purely to avoid
    boto3's float->Decimal marshalling requirement -- same reasoning as
    storing features as a JSON string."""
    request_ts = datetime.now(timezone.utc).isoformat()
    _log_table().put_item(Item={
        "log_shard": "LOG",
        "sk": f"{request_ts}#{uuid.uuid4().hex}",
        "entity_id": entity_id,
        "features_json": json.dumps(features),
        "calibrated_score": str(calibrated_score),
        "tier": tier,
        "model_version": model_version,
        "served_by": served_by,
        "low_confidence": low_confidence,
        "request_ts": request_ts,
    })


def read_inference_log(limit: int = 10_000):
    """Returns the logged inference requests as a list of dicts, most recent
    first -- a Query on the single log partition, descending, capped at
    limit, mirroring the old `ORDER BY id DESC LIMIT ?`."""
    resp = _log_table().query(
        KeyConditionExpression=Key("log_shard").eq("LOG"),
        ScanIndexForward=False,
        Limit=limit,
    )
    out = []
    for item in resp.get("Items", []):
        out.append({
            "entity_id": item["entity_id"],
            "features": json.loads(item["features_json"]),
            "calibrated_score": float(item["calibrated_score"]),
            "tier": item["tier"],
            "model_version": item["model_version"],
            "served_by": item["served_by"],
            "low_confidence": bool(item["low_confidence"]),
            "request_ts": item["request_ts"],
        })
    return out
