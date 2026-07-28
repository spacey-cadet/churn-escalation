"""
Feature Store Synchronization: online (low-latency) vs offline (high-throughput).

Simulates:
    - Online store: a plain dict standing in for Redis -- key-value OVERWRITE of the
      latest feature vector per entity_id, sub-10ms lookup semantics.
    - Offline store: an append-only Parquet file preserving full chronological history,
      for leak-free point-in-time-correct training joins.
    - A reconciliation job: periodically samples entity/timestamp pairs, compares the
      online value against what the offline history says WAS true at that instant, and
      alerts on divergence beyond a numeric tolerance -- catching sync bugs before they
      cause silent training-serving skew.

Run: python phase3_mlops/feature_store_sync.py
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RNG = np.random.default_rng(9)


class OnlineStore:
    """Stand-in for Redis: overwrites the latest value per key. No history."""
    def __init__(self):
        self._store = {}

    def write(self, entity_id: str, feature_value: float, timestamp: datetime):
        self._store[entity_id] = {"value": feature_value, "as_of": timestamp}

    def read(self, entity_id: str):
        return self._store.get(entity_id)


class OfflineStore:
    """Stand-in for partitioned Parquet on S3: append-only, full history retained."""
    def __init__(self):
        self._rows = []

    def append(self, entity_id: str, feature_value: float, timestamp: datetime):
        self._rows.append({"entity_id": entity_id, "value": feature_value, "timestamp": timestamp})

    def as_of(self, entity_id: str, as_of_timestamp: datetime):
        """Point-in-time lookup: the most recent value AT OR BEFORE as_of_timestamp."""
        candidates = [r for r in self._rows if r["entity_id"] == entity_id and r["timestamp"] <= as_of_timestamp]
        if not candidates:
            return None
        return max(candidates, key=lambda r: r["timestamp"])


def simulate_dual_write(n_entities=50, n_updates_per_entity=5, inject_bug=False):
    online = OnlineStore()
    offline = OfflineStore()
    start = datetime(2026, 1, 1)

    for i in range(n_entities):
        entity_id = f"cust_{i:04d}"
        for u in range(n_updates_per_entity):
            ts = start + timedelta(hours=u * 6)
            value = float(RNG.normal(3.0, 1.0))

            offline.append(entity_id, value, ts)

            # Inject a sync bug on purpose for a handful of entities: online write
            # silently uses a stale/incorrect value (simulating a race condition or a
            # bug fix applied to one path but not the other).
            if inject_bug and i % 10 == 0 and u == n_updates_per_entity - 1:
                online.write(entity_id, value + 5.0, ts)  # WRONG on purpose
            else:
                online.write(entity_id, value, ts)

    return online, offline


def reconciliation_job(online: OnlineStore, offline: OfflineStore, entity_ids, tolerance=0.01):
    """Samples entities, compares online 'latest' value against offline 'as-of-now' value."""
    now = datetime(2026, 1, 2)  # pretend "now" is after all writes
    mismatches = []
    for entity_id in entity_ids:
        online_record = online.read(entity_id)
        offline_record = offline.as_of(entity_id, now)
        if online_record is None or offline_record is None:
            continue
        diff = abs(online_record["value"] - offline_record["value"])
        if diff > tolerance:
            mismatches.append({
                "entity_id": entity_id,
                "online_value": round(online_record["value"], 4),
                "offline_value": round(offline_record["value"], 4),
                "diff": round(diff, 4),
            })
    return mismatches


def main():
    entity_ids = [f"cust_{i:04d}" for i in range(50)]

    print("=== Run 1: clean dual write, no bug injected ===")
    online, offline = simulate_dual_write(inject_bug=False)
    mismatches = reconciliation_job(online, offline, entity_ids)
    print(f"Mismatches found: {len(mismatches)} (expect 0)\n")

    print("=== Run 2: dual write WITH an injected sync bug ===")
    online_bad, offline_bad = simulate_dual_write(inject_bug=True)
    mismatches_bad = reconciliation_job(online_bad, offline_bad, entity_ids)
    print(f"Mismatches found: {len(mismatches_bad)}")
    for m in mismatches_bad[:5]:
        print(f"  ALERT: {m}")
    print("\nThis is the defensive check most candidates don't mention: a periodic")
    print("reconciliation job sampling entity/timestamp pairs and diffing online vs.")
    print("offline. Naming this signals real operating scars, not textbook knowledge.")


if __name__ == "__main__":
    main()
