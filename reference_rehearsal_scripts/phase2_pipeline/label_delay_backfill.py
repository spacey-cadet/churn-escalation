"""
Label Delay Handling.

Ground truth (did they actually churn/escalate) arrives 21 days after the prediction
in our simulated world. This script implements the three bridging mechanisms from
the roadmap:

1. PSI (Population Stability Index) on the model's OUTPUT SCORE distribution over
   time -- doesn't need labels, often precedes a labeled performance drop.
2. A small human-in-the-loop sample -- manually "labeled" fast read on a subset.
3. An automated backfill job -- recomputes true precision/recall once real labels land,
   confirming or dismissing the earlier PSI-based alert against ground truth.

Run: python phase2_pipeline/label_delay_backfill.py
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score

RNG = np.random.default_rng(3)
FEATURES = ["support_tickets_30d", "avg_message_length", "satisfaction_score",
            "days_since_last_login", "tenure_days", "monthly_spend"]


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two score distributions."""
    breakpoints = np.quantile(expected, np.linspace(0, 1, bins + 1))
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf
    expected_pct = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_pct = np.histogram(actual, breakpoints)[0] / len(actual)
    expected_pct = np.clip(expected_pct, 1e-6, None)
    actual_pct = np.clip(actual_pct, 1e-6, None)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def main():
    df = pd.read_parquet("data/features_clean.parquet")
    model = LogisticRegression(max_iter=1000).fit(df[FEATURES], df["label"])

    # "Training-time" score distribution (what we expected to see at deploy time)
    training_scores = model.predict_proba(df[FEATURES])[:, 1]

    # --- Step 1: PSI on live scores, no labels needed, available immediately ---
    live_df = df.sample(3000, random_state=11).copy()
    live_df["satisfaction_score"] *= RNG.normal(0.85, 0.05, len(live_df))  # slight real shift
    live_scores = model.predict_proba(live_df[FEATURES])[:, 1]

    psi_value = psi(training_scores, live_scores)
    print(f"PSI(training scores, live scores) = {psi_value:.4f}")
    if psi_value > 0.25:
        interpretation = "MAJOR shift -- investigate immediately, do not wait for labels"
    elif psi_value > 0.10:
        interpretation = "MODERATE shift -- early warning, worth a closer look"
    else:
        interpretation = "stable"
    print(f"Interpretation: {interpretation}\n")

    # --- Step 2: human-in-the-loop fast read on a small sample ---
    hitl_sample = live_df.sample(min(50, len(live_df)), random_state=5)
    # Simulate reviewers manually labeling this small batch with high (but imperfect) accuracy
    manual_labels = hitl_sample["label"].copy()
    noise_idx = manual_labels.sample(frac=0.05, random_state=1).index  # 5% reviewer error
    manual_labels.loc[noise_idx] = 1 - manual_labels.loc[noise_idx]
    hitl_preds = (model.predict_proba(hitl_sample[FEATURES])[:, 1] >= 0.5).astype(int)
    hitl_f1 = f1_score(manual_labels, hitl_preds)
    print(f"Human-in-the-loop fast read (n={len(hitl_sample)}): F1 = {hitl_f1:.4f}")
    print("This gives a low-volume-but-fast signal LONG before the 21-day label delay resolves.\n")

    # --- Step 3: automated backfill job once real labels finally arrive ---
    print("--- Simulating day 21: automated backfill job recomputes true metrics ---")
    true_preds = (model.predict_proba(live_df[FEATURES])[:, 1] >= 0.5).astype(int)
    true_precision = precision_score(live_df["label"], true_preds)
    true_recall = recall_score(live_df["label"], true_preds)
    true_f1 = f1_score(live_df["label"], true_preds)
    print(f"Backfilled precision: {true_precision:.4f}")
    print(f"Backfilled recall:    {true_recall:.4f}")
    print(f"Backfilled F1:        {true_f1:.4f}")

    print("\nNow the loop closes: does the backfilled F1 confirm or dismiss the earlier")
    print(f"PSI alert ({interpretation})? Compare against your production SLA threshold")
    print("and either confirm the retraining trigger or stand down the alert.")


if __name__ == "__main__":
    main()
