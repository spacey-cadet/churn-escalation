"""
Multi-threshold cascade routing.

A single cutoff assumes every decision is binary, but production routing usually isn't.
This implements the tiered system from the roadmap:

    score < t_low                    -> auto-resolve / no action
    t_low <= score < t_high          -> route to review queue (cheap: junior agent)
    score >= t_high                  -> escalate directly to senior human (expensive)

Each threshold is optimized independently against ITS OWN cost matrix, not forced to
share a single cutoff.

Run: python phase1_modeling/cascade_routing.py
(requires calibration_and_thresholds.py to have run first, for the calibrated model)
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def calibrated_scores(model, platt, X):
    raw = model.predict_proba(X)[:, 1]
    return platt.predict_proba(raw.reshape(-1, 1))[:, 1]


def find_optimal_threshold(y_true, scores, cost_fn, cost_fp, grid=None):
    """Grid-search a single threshold against a given cost matrix."""
    if grid is None:
        grid = np.linspace(0.01, 0.99, 98)
    best_t, best_cost = None, np.inf
    for t in grid:
        preds = (scores >= t).astype(int)
        fn = ((preds == 0) & (y_true == 1)).sum()
        fp = ((preds == 1) & (y_true == 0)).sum()
        cost = fn * cost_fn + fp * cost_fp
        if cost < best_cost:
            best_cost = cost
            best_t = t
    return best_t, best_cost


def main():
    model = joblib.load("phase1_modeling/champion_model.joblib")
    holdout = pd.read_parquet("phase1_modeling/holdout_test_set.parquet")
    FEATURES = [c for c in holdout.columns if c != "label"]

    calib_df, test_df = train_test_split(holdout, test_size=0.5, stratify=holdout["label"], random_state=7)
    raw_calib = model.predict_proba(calib_df[FEATURES])[:, 1]
    platt = LogisticRegression().fit(raw_calib.reshape(-1, 1), calib_df["label"])

    scores = calibrated_scores(model, platt, test_df[FEATURES])
    y_true = test_df["label"].values

    # --- Threshold 1 (t_low): cost of a missed auto-resolve ---
    # If we auto-resolve someone who was actually at risk, that's a real miss (FN),
    # but auto-resolving a happy user correctly costs nothing. Auto-resolving someone
    # borderline who then escalates anyway is expensive -> penalize FN heavily here,
    # because "below t_low" means "we did nothing."
    t_low, cost_low = find_optimal_threshold(y_true, scores, cost_fn=8, cost_fp=1)

    # --- Threshold 2 (t_high): cost of an unnecessary senior escalation ---
    # Above t_high routes straight to the most expensive, scarcest resource (a senior
    # human). Here the FP cost (wasting senior time on a false alarm) matters more
    # relative to FN, because anything missed here still gets caught by the review
    # queue below it -- it's not a total miss, just a slower path.
    t_high, cost_high = find_optimal_threshold(y_true, scores, cost_fn=3, cost_fp=6)

    if t_high < t_low:
        t_low, t_high = t_high, t_low  # keep them ordered

    tier = np.where(scores < t_low, "auto_resolve",
            np.where(scores < t_high, "review_queue", "senior_escalation"))

    result = pd.DataFrame({"score": scores, "label": y_true, "tier": tier})
    print(f"t_low  = {t_low:.4f}  (auto-resolve / review boundary)")
    print(f"t_high = {t_high:.4f}  (review / senior-escalation boundary)")
    print()
    print("Tier distribution and actual positive rate within each tier:")
    print(result.groupby("tier")["label"].agg(["count", "mean"]).rename(columns={"mean": "positive_rate"}))

    recall_at_or_above_low = result[result["tier"] != "auto_resolve"]["label"].sum() / result["label"].sum()
    print(f"\nRecall captured by (review_queue + senior_escalation): {recall_at_or_above_low:.4f}")
    print("This is the number to quote when someone asks 'how many at-risk customers")
    print("does auto-resolve silently let through?' -- 1 minus this recall.")


if __name__ == "__main__":
    main()
