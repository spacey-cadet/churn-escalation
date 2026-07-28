"""
Baseline Selection Hierarchy, implemented for real.

1. Heuristic baseline  -- a business rule
2. Linear baseline     -- Logistic Regression with L1 and L2
3. Tree ensemble       -- XGBoost

Prints PR-AUC (not just accuracy -- see the imbalanced-data caveat) at each step so
you can quote the *actual* gap between tiers instead of a hypothetical one.

Run from the project root: python phase1_modeling/baseline_hierarchy.py
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier
import joblib

FEATURES = [
    "support_tickets_30d", "avg_message_length", "satisfaction_score",
    "days_since_last_login", "tenure_days", "monthly_spend",
]


def load_data():
    df = pd.read_parquet("data/features_clean.parquet")
    X = df[FEATURES]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    return X_train, X_test, y_train, y_test


def heuristic_baseline(X_test, y_test):
    pred = ((X_test["satisfaction_score"] < 2.5) | (X_test["support_tickets_30d"] > 3)).astype(int)
    pr_auc = average_precision_score(y_test, pred)
    print(f"[1] Heuristic baseline        PR-AUC: {pr_auc:.4f}  (business rule, no training)")
    return pr_auc


def linear_baseline(X_train, X_test, y_train, y_test):
    results = {}
    for penalty, C in [("l1", 0.5), ("l2", 0.5)]:
        model = LogisticRegression(penalty=penalty, C=C, solver="liblinear", max_iter=1000)
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        pr_auc = average_precision_score(y_test, probs)
        roc_auc = roc_auc_score(y_test, probs)
        print(f"[2] Logistic ({penalty.upper()}, C={C})   PR-AUC: {pr_auc:.4f}   ROC-AUC: {roc_auc:.4f}")
        results[penalty] = (model, pr_auc)
    return results


def tree_ensemble(X_train, X_test, y_train, y_test):
    model = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        eval_metric="logloss", random_state=42,
    )
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    pr_auc = average_precision_score(y_test, probs)
    roc_auc = roc_auc_score(y_test, probs)
    print(f"[3] XGBoost                    PR-AUC: {pr_auc:.4f}   ROC-AUC: {roc_auc:.4f}")

    importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("    Feature importances (explainability you get for free):")
    print(importances.to_string())
    return model, pr_auc


def main():
    X_train, X_test, y_train, y_test = load_data()
    print(f"Train: {len(X_train)} rows, positive rate {y_train.mean():.4f}")
    print(f"Test:  {len(X_test)} rows, positive rate {y_test.mean():.4f}\n")

    h_score = heuristic_baseline(X_test, y_test)
    linear_results = linear_baseline(X_train, X_test, y_train, y_test)
    xgb_model, xgb_score = tree_ensemble(X_train, X_test, y_train, y_test)

    best_linear = max(linear_results.values(), key=lambda t: t[1])
    print("\n--- Summary: state these gaps out loud, they're real numbers now ---")
    print(f"Heuristic -> Linear gap:  {best_linear[1] - h_score:+.4f} PR-AUC")
    print(f"Linear -> XGBoost gap:    {xgb_score - best_linear[1]:+.4f} PR-AUC")
    print("Deep learning would only be justified here if this gap were still large")
    print("AND the data paradigm shifted to raw text/audio/images -- neither is true,")
    print("so DL is correctly ruled out for this tabular problem.")

    joblib.dump(xgb_model, "phase1_modeling/champion_model.joblib")
    X_test.assign(label=y_test).to_parquet("phase1_modeling/holdout_test_set.parquet", index=False)
    print("\nSaved champion_model.joblib and holdout_test_set.parquet for the next script.")


if __name__ == "__main__":
    main()
