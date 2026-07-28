"""
Serving Gate (Gate 3) -- KS drift test, and the data-drift vs concept-drift distinction
made tangible instead of memorized.

Part A: DATA DRIFT
    Shift the input feature distribution (e.g. users start submitting longer messages)
    and run a Kolmogorov-Smirnov test comparing live vs. training baseline. This is
    checkable WITHOUT ever needing ground-truth labels.

Part B: CONCEPT DRIFT
    Keep the INPUT distribution identical, but change the underlying relationship
    between features and the label (what used to predict escalation no longer does).
    This is invisible to the KS test on inputs -- it only shows up once you compute
    live precision/recall against arriving ground truth.

Run: python phase2_pipeline/drift_detection.py
"""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

RNG = np.random.default_rng(7)


def part_a_data_drift():
    print("=== Part A: DATA DRIFT (input distribution shift) ===")
    baseline = pd.read_parquet("data/features_clean.parquet")["avg_message_length"]

    # Simulate 30 days later: users start writing longer messages (a real product/behavior shift)
    live_shifted = baseline.sample(2000, random_state=1) * RNG.normal(1.35, 0.05, 2000)
    live_unshifted = baseline.sample(2000, random_state=2)  # control: no shift

    stat_shifted, p_shifted = ks_2samp(baseline, live_shifted)
    stat_unshifted, p_unshifted = ks_2samp(baseline, live_unshifted)

    print(f"Shifted live traffic   -> KS stat={stat_shifted:.4f}, p={p_shifted:.6f}  "
          f"{'DRIFT DETECTED (p<0.05)' if p_shifted < 0.05 else 'no drift'}")
    print(f"Unshifted live traffic -> KS stat={stat_unshifted:.4f}, p={p_unshifted:.6f}  "
          f"{'DRIFT DETECTED (p<0.05)' if p_unshifted < 0.05 else 'no drift'}")
    print("Notice: this check needed ZERO ground-truth labels -- that's exactly why it")
    print("can run hourly in production while labels are still 21 days away.\n")
    if p_shifted < 0.05:
        print(">>> MLflow flag set: automated retraining loop triggered.\n")


def part_b_concept_drift():
    print("=== Part B: CONCEPT DRIFT (same inputs, relationship to label changes) ===")
    df = pd.read_parquet("data/features_clean.parquet")
    FEATURES = ["support_tickets_30d", "avg_message_length", "satisfaction_score",
                "days_since_last_login", "tenure_days", "monthly_spend"]

    model = LogisticRegression(max_iter=1000)
    model.fit(df[FEATURES], df["label"])

    # Same feature distribution, but flip the meaning: pretend a product change means
    # "high satisfaction_score" now paradoxically correlates with silent churn (e.g.
    # customers who stopped complaining because they've already decided to leave).
    concept_shifted = df.copy()
    flip_mask = concept_shifted["satisfaction_score"] > 4.0
    concept_shifted.loc[flip_mask, "label"] = 1 - concept_shifted.loc[flip_mask, "label"]

    preds_normal = model.predict(df[FEATURES])
    preds_shifted_data = model.predict(concept_shifted[FEATURES])  # inputs identical either way

    f1_normal = f1_score(df["label"], preds_normal)
    f1_after_concept_drift = f1_score(concept_shifted["label"], preds_shifted_data)

    # KS test on the inputs shows NOTHING, because we never touched the inputs.
    stat, p = ks_2samp(df[FEATURES[0]], concept_shifted[FEATURES[0]])
    print(f"KS test on inputs: stat={stat:.4f}, p={p:.6f}  (p=1.0 -> distributions are IDENTICAL)")
    print(f"Live F1 before concept drift: {f1_normal:.4f}")
    print(f"Live F1 after concept drift:  {f1_after_concept_drift:.4f}")
    print("\nThis is the point: the KS test is blind to this failure. Concept drift only")
    print("shows up once ground-truth labels arrive and precision/recall degrade --")
    print("which is exactly why label-delay handling (next script) matters.")


if __name__ == "__main__":
    part_a_data_drift()
    part_b_concept_drift()
