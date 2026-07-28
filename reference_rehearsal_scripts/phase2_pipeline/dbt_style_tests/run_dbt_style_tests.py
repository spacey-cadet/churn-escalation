"""
Transformation Gate (Gate 2) -- dbt-style tests, without needing a real dbt project.

Real dbt would run these as YAML-declared tests during warehouse compilation
(`unique`, `not_null`, relationship tests). This script implements the identical
logic and failure behavior in plain pandas so you can run it without a warehouse,
while still being able to say "this is exactly what a dbt `unique` + `relationships`
test schema would check."

On failure: simulates Airflow blocking downstream training tasks + firing a
PagerDuty-style critical alert + isolating the offending partition.

Run: python phase2_pipeline/dbt_style_tests/run_dbt_style_tests.py
"""
import pandas as pd


def test_unique_not_null(df: pd.DataFrame, column: str):
    nulls = df[column].isna().sum()
    duplicates = df[column].duplicated().sum()
    passed = (nulls == 0) and (duplicates == 0)
    return {
        "test": f"unique_and_not_null({column})",
        "passed": passed,
        "nulls": int(nulls),
        "duplicates": int(duplicates),
    }


def test_referential_integrity(features_df: pd.DataFrame, labels_df: pd.DataFrame, key: str):
    """Equivalent of a dbt `relationships` test: every feature row's key must exist
    in the master/labels tracking table (or vice versa, depending on direction)."""
    feature_keys = set(features_df[key])
    label_keys = set(labels_df[key])
    orphaned = feature_keys - label_keys
    passed = len(orphaned) == 0
    return {
        "test": f"relationships({key}, features -> labels)",
        "passed": passed,
        "orphaned_count": len(orphaned),
        "orphaned_sample": list(orphaned)[:5],
    }


def main():
    features = pd.read_parquet("data/features_clean.parquet")
    labels = pd.read_parquet("data/labels_delayed.parquet")

    results = [
        test_unique_not_null(features, "entity_id"),
        test_referential_integrity(features, labels, "entity_id"),
    ]

    print("=== Transformation Gate (dbt-style tests) ===")
    any_failed = False
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        if not r["passed"]:
            any_failed = True
        print(f"[{status}] {r}")

    if any_failed:
        print("\n>>> AIRFLOW: downstream training DAG tasks BLOCKED.")
        print(">>> PAGERDUTY: critical alert fired to on-call.")
        print(">>> Offending table partition isolated pending investigation.")
    else:
        print("\nAll transformation-gate tests passed. Downstream training DAG unblocked.")


if __name__ == "__main__":
    main()
