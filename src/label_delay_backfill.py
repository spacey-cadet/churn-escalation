"""
Bridges the 21-day label-delay gap. Concept drift (the feature->label relationship
changing) is invisible to drift_monitor.py's KS/PSI checks on inputs and scores --
it only shows up once ground truth arrives. This closes that loop, and doubles as
the source of "hard/mislabeled" examples that get folded into the next retrain (see
README's retraining-cadence section).

Run: python -m src.label_delay_backfill
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

import config
import registry


def backfill_against_ground_truth(predictions_df: pd.DataFrame, labels_df: pd.DataFrame) -> dict:
    """predictions_df: entity_id, calibrated_score, tier (from the inference log).
    labels_df: entity_id, label (ground truth that has now arrived, 21 days later)."""
    joined = predictions_df.merge(labels_df[["entity_id", "label"]], on="entity_id", how="inner")
    if joined.empty:
        return {"status": "no_matching_labels"}

    preds = (joined["calibrated_score"] >= 0.5).astype(int)
    return {
        "status": "checked",
        "n_matched": int(len(joined)),
        "precision": float(precision_score(joined["label"], preds)),
        "recall": float(recall_score(joined["label"], preds)),
        "f1": float(f1_score(joined["label"], preds)),
    }


def main():
    import feature_store
    logged = feature_store.read_inference_log(limit=50_000)
    if not logged:
        print("No inference log entries yet -- run some requests through the serving API first.")
        return

    predictions_df = pd.DataFrame([
        {"entity_id": r["entity_id"], "calibrated_score": r["calibrated_score"], "tier": r["tier"]}
        for r in logged if r["entity_id"] is not None
    ])
    labels_df = pd.read_parquet(config.LABELS_DELAYED)[["entity_id", "label"]]

    result = backfill_against_ground_truth(predictions_df, labels_df)
    print("=== Label-delay backfill (ground truth now available) ===")
    print(result)

    if result.get("status") == "checked" and result["f1"] < 0.5:
        import alerting
        alerting.send_alert(
            f"Backfilled F1 = {result['f1']:.4f} on {result['n_matched']} matched predictions -- "
            "this is a concept-drift signal, not just a data-drift one. Consider retraining soon.",
            severity="critical",
        )


if __name__ == "__main__":
    main()
