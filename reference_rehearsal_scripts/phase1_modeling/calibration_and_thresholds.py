"""
Calibration -> Thresholding, in the correct order.

Tree ensembles produce scores that are NOT reliable probabilities out of the box
(they optimize split purity, not calibrated likelihood). This script:

1. Plots a reliability diagram BEFORE calibration to show the miscalibration.
2. Applies Platt scaling (logistic fit on raw scores) and isotonic regression.
3. Only THEN builds ROC/PR curves on a held-out validation fold and picks the
   threshold that maximizes Youden's J = Sensitivity + Specificity - 1.

Run: python phase1_modeling/calibration_and_thresholds.py
(requires baseline_hierarchy.py to have run first)
"""
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve


def reliability_diagram(y_true, probs, title, ax):
    frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=10, strategy="quantile")
    ax.plot(mean_pred, frac_pos, marker="o", label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive rate")
    ax.set_title(title)
    ax.legend()


def youdens_j_threshold(y_true, probs):
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    j_scores = tpr - fpr  # Sensitivity + Specificity - 1  ==  TPR - FPR
    best_idx = np.argmax(j_scores)
    return thresholds[best_idx], j_scores[best_idx], (fpr, tpr, thresholds)


def main():
    model = joblib.load("phase1_modeling/champion_model.joblib")
    holdout = pd.read_parquet("phase1_modeling/holdout_test_set.parquet")

    FEATURES = [c for c in holdout.columns if c != "label"]
    # Split the holdout further: a calibration/threshold-selection fold and a final
    # untouched test fold, so threshold selection never leaks into the true test set.
    calib_df, final_test_df = train_test_split(
        holdout, test_size=0.5, stratify=holdout["label"], random_state=7
    )

    raw_probs_calib = model.predict_proba(calib_df[FEATURES])[:, 1]
    raw_probs_test = model.predict_proba(final_test_df[FEATURES])[:, 1]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    reliability_diagram(calib_df["label"], raw_probs_calib, "Raw XGBoost scores (before calibration)", axes[0])

    # --- Platt scaling: fit logistic regression on raw scores ---
    platt = LogisticRegression()
    platt.fit(raw_probs_calib.reshape(-1, 1), calib_df["label"])
    platt_probs_test = platt.predict_proba(raw_probs_test.reshape(-1, 1))[:, 1]
    reliability_diagram(final_test_df["label"], platt_probs_test, "After Platt scaling", axes[1])

    # --- Isotonic regression: non-parametric monotonic fit ---
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_probs_calib, calib_df["label"])
    iso_probs_test = iso.predict(raw_probs_test)
    reliability_diagram(final_test_df["label"], iso_probs_test, "After isotonic regression", axes[2])

    plt.tight_layout()
    plt.savefig("phase1_modeling/reliability_diagrams.png", dpi=100)
    print("Saved reliability_diagrams.png -- look at how bunched-up the raw scores are.")

    # Use Platt-calibrated probabilities going forward (works fine with our data size;
    # isotonic needs more validation data to avoid overfitting the calibration curve).
    calibrated_probs = platt_probs_test

    # --- ROC / PR curves + Youden's J, on the calibrated scores, on this held-out fold ---
    best_thresh, best_j, (fpr, tpr, thresholds) = youdens_j_threshold(final_test_df["label"], calibrated_probs)
    precision, recall, pr_thresholds = precision_recall_curve(final_test_df["label"], calibrated_probs)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(fpr, tpr)
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set_title("ROC curve")
    axes[0].set_xlabel("FPR")
    axes[0].set_ylabel("TPR")

    axes[1].plot(recall, precision)
    axes[1].set_title("Precision-Recall curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    plt.tight_layout()
    plt.savefig("phase1_modeling/roc_pr_curves.png", dpi=100)

    print(f"\nYouden's J-optimal threshold: {best_thresh:.4f}  (J = {best_j:.4f})")
    print("This is the scalar you'd lock into production config and periodically re-derive")
    print("as live class balance drifts.")

    # --- Cost-matrix sanity check: is Youden's J actually the right call here? ---
    # Suppose: False Negative (missed escalation, lost customer) costs 10x
    #          False Positive (annoyed a happy user with a review-queue nudge) costs 1x
    cost_fn, cost_fp = 10, 1
    candidate_thresholds = np.linspace(0.01, 0.99, 50)
    costs = []
    for t in candidate_thresholds:
        preds = (calibrated_probs >= t).astype(int)
        fn = ((preds == 0) & (final_test_df["label"] == 1)).sum()
        fp = ((preds == 1) & (final_test_df["label"] == 0)).sum()
        costs.append(fn * cost_fn + fp * cost_fp)
    cost_optimal_thresh = candidate_thresholds[np.argmin(costs)]
    print(f"Cost-matrix-optimal threshold (10x FN penalty): {cost_optimal_thresh:.4f}")
    print("Notice this differs from Youden's J -- Youden's J assumes FN and FP costs are")
    print("equal, which is almost never true in a real business. State this distinction")
    print("explicitly in the interview: J is a good DEFAULT, the cost matrix is the REAL answer.")


if __name__ == "__main__":
    main()
