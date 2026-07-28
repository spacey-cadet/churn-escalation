"""
Train -> calibrate -> derive cascade thresholds -> register, as one pipeline step.

This is what you run every time you retrain (manually, on refreshed/re-curated
data). It doesn't touch production on its own -- it only registers a new version.
Promotion happens separately, after the champion-challenger gate in
scripts/evaluate_and_promote.py passes.

Run: python -m src.modeling.train
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

import config
import registry


def load_data():
    df = pd.read_parquet(config.FEATURES_CLEAN)
    X = df[config.FEATURES]
    y = df["label"]
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    # Holdout gets split again downstream: one fold for calibration/threshold
    # selection, one fold that's never touched until final promotion-gate eval.
    return X_train, y_train, X_holdout, y_holdout


def train_xgboost(X_train, y_train) -> tuple:
    hyperparameters = dict(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        scale_pos_weight=float((y_train == 0).sum() / (y_train == 1).sum()),
        eval_metric="logloss", random_state=42,
    )
    model = XGBClassifier(**hyperparameters)
    model.fit(X_train, y_train)
    return model, hyperparameters


def calibrate(model, X_calib, y_calib) -> LogisticRegression:
    """Platt scaling: fit a logistic regression on the model's raw scores. Chosen
    over isotonic regression here because our calibration fold is small; isotonic
    needs more data to avoid overfitting the calibration curve itself."""
    raw_scores = model.predict_proba(X_calib)[:, 1]
    platt = LogisticRegression()
    platt.fit(raw_scores.reshape(-1, 1), y_calib)
    return platt


def calibrated_scores(model, calibrator, X) -> np.ndarray:
    raw = model.predict_proba(X)[:, 1]
    return calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]


def find_optimal_threshold(y_true, scores, cost_fn, cost_fp, grid=None):
    if grid is None:
        grid = np.linspace(0.01, 0.99, 98)
    best_t, best_cost = None, np.inf
    for t in grid:
        preds = (scores >= t).astype(int)
        fn = ((preds == 0) & (y_true == 1)).sum()
        fp = ((preds == 1) & (y_true == 0)).sum()
        cost = fn * cost_fn + fp * cost_fp
        if cost < best_cost:
            best_cost, best_t = cost, t
    return float(best_t), float(best_cost)


def derive_cascade_thresholds(y_true, scores) -> dict:
    t_low, cost_low = find_optimal_threshold(y_true, scores, **config.COST_MATRIX_LOW)
    t_high, cost_high = find_optimal_threshold(y_true, scores, **config.COST_MATRIX_HIGH)
    if t_high < t_low:
        t_low, t_high = t_high, t_low
    return {"t_low": t_low, "t_high": t_high, "cost_low": cost_low, "cost_high": cost_high}


def main():
    X_train, y_train, X_holdout, y_holdout = load_data()

    # Split holdout: calibration/threshold fold vs. a truly untouched final-test fold.
    X_calib, X_final_test, y_calib, y_final_test = train_test_split(
        X_holdout, y_holdout, test_size=0.5, stratify=y_holdout, random_state=7
    )

    model, hyperparameters = train_xgboost(X_train, y_train)
    calibrator = calibrate(model, X_calib, y_calib)

    calib_scores_for_thresholds = calibrated_scores(model, calibrator, X_calib)
    thresholds = derive_cascade_thresholds(y_calib.values, calib_scores_for_thresholds)

    final_scores = calibrated_scores(model, calibrator, X_final_test)
    metrics = {
        "pr_auc": float(average_precision_score(y_final_test, final_scores)),
        "roc_auc": float(roc_auc_score(y_final_test, final_scores)),
        "n_train": int(len(X_train)),
        "n_final_test": int(len(X_final_test)),
        "positive_rate_train": float(y_train.mean()),
    }

    data_version = f"features_clean@{config.FEATURES_CLEAN.stat().st_mtime_ns}"
    version = registry.register(
        model=model, calibrator=calibrator, thresholds=thresholds,
        metrics=metrics, data_version=data_version, hyperparameters=hyperparameters,
    )

    print(f"Registered new model version: {version}")
    print(f"  PR-AUC (final test fold): {metrics['pr_auc']:.4f}")
    print(f"  ROC-AUC (final test fold): {metrics['roc_auc']:.4f}")
    print(f"  Cascade thresholds: t_low={thresholds['t_low']:.4f}  t_high={thresholds['t_high']:.4f}")
    print("\nThis version is registered but NOT promoted. Run "
          "scripts/evaluate_and_promote.py to run the champion-challenger gate.")
    return version


if __name__ == "__main__":
    main()
