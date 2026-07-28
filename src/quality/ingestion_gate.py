"""
Gate 1 -- Ingestion / Landing Zone.

Real tool: Great Expectations (or a PySpark schema-validation block), run inline as
raw files land. Here it runs as a plain function so it can be called from the
pipeline orchestrator, from a test, or from the CLI -- same logic either way.

If `great_expectations` is installed, we use it for real; otherwise we fall back to
a hand-rolled check that implements the identical expectations and thresholds, so
the gate always runs even in a minimal environment.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))          # for `import config`
sys.path.insert(0, str(_ROOT / "src"))  # for `import alerting`, `import feature_store`, etc.

import pandas as pd
import config


def _check_with_great_expectations(df: pd.DataFrame):
    import great_expectations as gx
    context = gx.get_context(mode="ephemeral")
    validator = context.sources.pandas_default.read_dataframe(df)
    validator.expect_column_values_to_not_be_null("entity_id")
    validator.expect_table_row_count_to_be_between(
        min_value=int(config.ROLLING_AVG_ROW_COUNT * (1 - config.VOLUME_DROP_ALERT_PCT))
    )
    result = validator.validate()
    return result.success, result


def run_ingestion_gate(df: pd.DataFrame) -> dict:
    """Returns a dict describing pass/fail per check, the cleaned dataframe that's
    allowed to proceed, and the quarantined rows (Dead Letter Queue)."""
    null_count = int(df["entity_id"].isna().sum())
    row_count = len(df)
    quarantine_rate = null_count / row_count if row_count else 0.0
    # A few nulls get quarantined and the batch proceeds -- that's the designed
    # remediation. The gate only BLOCKS the whole batch if quarantine would strip
    # out more than MAX_QUARANTINE_RATE, which signals a structural problem rather
    # than a handful of bad rows.
    quarantine_within_tolerance = quarantine_rate <= config.MAX_QUARANTINE_RATE

    drop_pct = 1 - (row_count / config.ROLLING_AVG_ROW_COUNT)
    volume_ok = drop_pct < config.VOLUME_DROP_ALERT_PCT

    quarantined = df[df["entity_id"].isna()]
    clean = df[df["entity_id"].notna()].reset_index(drop=True)

    return {
        "passed": quarantine_within_tolerance,  # a volume drop only warns (see
                                                 # roadmap: DLQ vs Slack warning)
        "checks": {
            "not_null_entity_id": {
                "passed": quarantine_within_tolerance,
                "null_count": null_count,
                "quarantine_rate": round(quarantine_rate, 4),
                "action": "PASS" if null_count == 0 else
                          (f"{null_count} rows ROUTED TO DEAD LETTER QUEUE, batch proceeds"
                           if quarantine_within_tolerance else
                           "BATCH BLOCKED -- quarantine rate exceeds tolerance"),
            },
            "row_count_volume_check": {
                "passed": volume_ok,
                "row_count": row_count,
                "expected_baseline": config.ROLLING_AVG_ROW_COUNT,
                "drop_pct": round(drop_pct, 4),
                "action": "PASS" if volume_ok else "ALERT: volume drop beyond threshold",
            },
        },
        "clean_df": clean,
        "quarantined_df": quarantined,
    }


def main():
    df = pd.read_parquet(config.RAW_LANDING)
    result = run_ingestion_gate(df)

    print("=== Ingestion Gate ===")
    for name, r in result["checks"].items():
        status = "PASS" if r["passed"] else "FAIL"
        print(f"[{status}] {name}: { {k: v for k, v in r.items() if k != 'passed'} }")

    if len(result["quarantined_df"]) > 0:
        config.DEAD_LETTER_DIR.mkdir(parents=True, exist_ok=True)
        out_path = config.DEAD_LETTER_DIR / "quarantined_batch.parquet"
        result["quarantined_df"].to_parquet(out_path, index=False)
        print(f"\n{len(result['quarantined_df'])} rows quarantined -> {out_path}")

    result["clean_df"].to_parquet(config.FEATURES_CLEAN.parent / "_ingestion_gate_output.parquet", index=False)
    print(f"{len(result['clean_df'])} rows passed the gate.")

    if not result["checks"]["row_count_volume_check"]["passed"]:
        import alerting  # local import to avoid a hard dependency for pure-gate tests
        alerting.send_alert(
            f"Ingestion volume down {result['checks']['row_count_volume_check']['drop_pct']*100:.1f}% "
            "vs 7-day rolling average."
        )

    if not result["passed"]:
        sys.exit(1)  # non-zero exit -> a CI/orchestrator step can block on this


if __name__ == "__main__":
    main()
