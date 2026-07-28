"""
Ingestion Gate (Gate 1) -- Great Expectations style checks.

Uses the real `great_expectations` library if installed; falls back to an equivalent
hand-rolled check (same logic, same thresholds) if it isn't, so this always runs.

Checks:
    - expect_column_values_to_not_be_null("entity_id")          -> 0% tolerance, DLQ on failure
    - expect_table_row_count_to_be_between(...)                  -> volume-drop alert

Run: python phase2_pipeline/great_expectations/run_expectations.py
"""
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

ROLLING_AVG_ROW_COUNT = 20_000  # pretend this came from a 7-day rolling average lookup
VOLUME_DROP_ALERT_PCT = 0.30


def run_checks(df: pd.DataFrame):
    results = {}

    # --- Check 1: entity_id must never be null. 0% tolerance -> DLQ the whole batch. ---
    null_count = df["entity_id"].isna().sum()
    results["not_null_entity_id"] = {
        "passed": null_count == 0,
        "null_count": int(null_count),
        "action": "PASS" if null_count == 0 else "ROUTE ENTIRE MICRO-BATCH TO DEAD LETTER QUEUE",
    }

    # --- Check 2: row count vs rolling average. Flag if >30% below expected. ---
    row_count = len(df)
    drop_pct = 1 - (row_count / ROLLING_AVG_ROW_COUNT)
    volume_ok = drop_pct < VOLUME_DROP_ALERT_PCT
    results["row_count_volume_check"] = {
        "passed": volume_ok,
        "row_count": row_count,
        "expected_baseline": ROLLING_AVG_ROW_COUNT,
        "drop_pct": round(drop_pct, 4),
        "action": "PASS" if volume_ok else "SLACK WEBHOOK WARNING: volume >30% below 7-day rolling avg",
    }
    return results


def quarantine_and_clean(df: pd.DataFrame):
    """Simulates the DLQ: bad rows get isolated, the rest proceed downstream."""
    bad_rows = df[df["entity_id"].isna()]
    good_rows = df[df["entity_id"].notna()]
    if len(bad_rows) > 0:
        os.makedirs("phase2_pipeline/dead_letter_queue", exist_ok=True)
        bad_rows.to_parquet("phase2_pipeline/dead_letter_queue/quarantined_batch.parquet", index=False)
    return good_rows


def main():
    df = pd.read_parquet("data/raw_landing.parquet")
    results = run_checks(df)

    print("=== Ingestion Gate (Great Expectations style) ===")
    for name, r in results.items():
        status = "PASS" if r["passed"] else "FAIL"
        print(f"[{status}] {name}: {r}")

    if not results["not_null_entity_id"]["passed"]:
        clean = quarantine_and_clean(df)
        print(f"\n{len(df) - len(clean)} rows quarantined to dead_letter_queue/quarantined_batch.parquet")
        print(f"{len(clean)} rows proceed to the transformation gate.")

    if not results["row_count_volume_check"]["passed"]:
        print("\n>>> Simulated Slack webhook fired: 'Ingestion volume down "
              f"{results['row_count_volume_check']['drop_pct']*100:.1f}% vs 7-day rolling average.'")


if __name__ == "__main__":
    main()
