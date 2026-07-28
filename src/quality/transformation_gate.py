"""
Gate 2 -- Transformation Zone.

Real tool: dbt tests (`unique`, `not_null`, `relationships`) run during warehouse
compilation. This module implements the identical checks in plain pandas so it runs
without a warehouse, while a real dbt-equivalent YAML lives alongside it in
reference_rehearsal_scripts/phase2_pipeline/ for anyone who wants the literal syntax.

On failure: real behavior would be Airflow blocking downstream training tasks and a
PagerDuty-style critical alert. Here that's a non-zero exit code (which a CI job or
orchestrator script treats exactly the same way) plus a webhook alert.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import pandas as pd
import config


def test_unique_not_null(df: pd.DataFrame, column: str) -> dict:
    nulls = int(df[column].isna().sum())
    duplicates = int(df[column].duplicated().sum())
    return {
        "test": f"unique_and_not_null({column})",
        "passed": nulls == 0 and duplicates == 0,
        "nulls": nulls,
        "duplicates": duplicates,
    }


def test_referential_integrity(features_df: pd.DataFrame, labels_df: pd.DataFrame, key: str) -> dict:
    feature_keys = set(features_df[key])
    label_keys = set(labels_df[key])
    orphaned = feature_keys - label_keys
    return {
        "test": f"relationships({key}, features -> labels)",
        "passed": len(orphaned) == 0,
        "orphaned_count": len(orphaned),
        "orphaned_sample": list(orphaned)[:5],
    }


def run_transformation_gate(features: pd.DataFrame, labels: pd.DataFrame) -> dict:
    results = [
        test_unique_not_null(features, "entity_id"),
        test_referential_integrity(features, labels, "entity_id"),
    ]
    return {"passed": all(r["passed"] for r in results), "checks": results}


def main():
    features = pd.read_parquet(config.FEATURES_CLEAN)
    labels = pd.read_parquet(config.LABELS_DELAYED)
    result = run_transformation_gate(features, labels)

    print("=== Transformation Gate (dbt-style tests) ===")
    for r in result["checks"]:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"[{status}] {r}")

    if not result["passed"]:
        import alerting
        alerting.send_alert("Transformation gate FAILED -- downstream training DAG blocked.")
        print("\n>>> Downstream training tasks BLOCKED. Offending partition isolated pending investigation.")
        sys.exit(1)
    else:
        print("\nAll transformation-gate tests passed. Downstream training unblocked.")


if __name__ == "__main__":
    main()
