"""
Point-in-Time Leakage Demo -- the single most important exercise in this whole project.

Run this TWICE:
    1. USE_LEAKY_JOIN = True   -> naive join uses the CURRENT/LATEST feature value for
       every historical label, leaking future information into the past.
    2. USE_LEAKY_JOIN = True  -> point-in-time-correct join uses only the feature value
       that was TRUE AS OF each label's timestamp.

Watch the offline AUC collapse from the leaky version to the correct version. That gap
IS training-serving skew / feature leakage, made numeric instead of theoretical.

Run: python phase3_mlops/point_in_time_leakage_demo.py
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

RNG = np.random.default_rng(21)

# <<< TOGGLE THIS AND RE-RUN >>>
USE_LEAKY_JOIN = True


def build_feature_history(n_entities=800):
    """Each entity has a 'purchases_last_30d' feature history, and a label event
    (did they churn) that happens partway through. Critically: AFTER a churn event,
    the feature value collapses (churned customers stop purchasing) -- this is the
    classic leakage trap. A feature's CURRENT/LATEST value is partly CAUSED BY the
    outcome itself, so joining "current value" to a past label smuggles the answer
    into the input."""
    rows = []
    label_events = []
    start = datetime(2026, 1, 1)

    for i in range(n_entities):
        entity_id = f"cust_{i:04d}"
        base = RNG.uniform(8, 12)  # pre-event baseline purchase level, weakly informative

        label_day = int(RNG.integers(20, 100))
        label_ts = start + timedelta(days=label_day)
        # True underlying signal at prediction time is WEAK on purpose -- realistic difficulty.
        pre_event_signal = base + RNG.normal(0, 1.5)
        prob = 1 / (1 + np.exp(-(pre_event_signal - 10) / 4))  # only mildly separable
        label = int(RNG.random() < prob)
        label_events.append({"entity_id": entity_id, "label_timestamp": label_ts, "label": label})

        for day in range(0, 180, 5):
            ts = start + timedelta(days=day)
            if day <= label_day:
                # Before the event: normal fluctuation around baseline, weak signal only
                value = base + RNG.normal(0, 1.5)
            else:
                # After the event: churners' activity collapses, non-churners keep buying.
                # This post-event value is CAUSED BY the outcome -- classic leakage.
                if label == 1:
                    value = RNG.normal(0.5, 0.3)
                else:
                    value = base + RNG.normal(0, 1.5) + 1.0  # stays healthy, ticks up slightly
            rows.append({"entity_id": entity_id, "timestamp": ts, "purchases_last_30d": value})

    return pd.DataFrame(rows), pd.DataFrame(label_events)


def leaky_join(feature_history: pd.DataFrame, labels: pd.DataFrame):
    """WRONG: joins each label to the LATEST (most recent / 'current') feature value,
    regardless of whether that value existed yet at label time."""
    latest = feature_history.sort_values("timestamp").groupby("entity_id").tail(1)
    return labels.merge(latest[["entity_id", "purchases_last_30d"]], on="entity_id", how="left")


def point_in_time_join(feature_history: pd.DataFrame, labels: pd.DataFrame):
    """CORRECT: for each label, use only the feature value that was valid AS OF that
    label's own timestamp -- exactly what Feast's point-in-time joins do."""
    merged_rows = []
    for _, label_row in labels.iterrows():
        candidates = feature_history[
            (feature_history["entity_id"] == label_row["entity_id"]) &
            (feature_history["timestamp"] <= label_row["label_timestamp"])
        ]
        if candidates.empty:
            continue
        latest_valid = candidates.sort_values("timestamp").iloc[-1]
        merged_rows.append({
            "entity_id": label_row["entity_id"],
            "label": label_row["label"],
            "purchases_last_30d": latest_valid["purchases_last_30d"],
        })
    return pd.DataFrame(merged_rows)


def evaluate(joined_df: pd.DataFrame, label_col="label"):
    X = joined_df[["purchases_last_30d"]].values
    y = joined_df[label_col].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1)
    model = LogisticRegression().fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, probs)


def main():
    feature_history, labels = build_feature_history()

    if USE_LEAKY_JOIN:
        print(">>> Running with USE_LEAKY_JOIN = True (naive 'current value' join) <<<\n")
        joined = leaky_join(feature_history, labels)
        joined = joined.dropna()
        auc = evaluate(joined)
        print(f"Offline AUC (LEAKY join): {auc:.4f}")
        print("This looks great -- because the model is partly 'seeing the future':")
        print("purchase totals accumulated AFTER the label event are leaking into training.")
        print("\nNow flip USE_LEAKY_JOIN to False at the top of this file and re-run.")
    else:
        print(">>> Running with USE_LEAKY_JOIN = True (point-in-time-correct join) <<<\n")
        joined = point_in_time_join(feature_history, labels)
        auc = evaluate(joined)
        print(f"Offline AUC (point-in-time-correct join): {auc:.4f}")
        print("This is the REAL, honest offline metric -- the one that will actually")
        print("survive contact with production, because at inference time we could")
        print("never have seen feature values from after the prediction moment.")


if __name__ == "__main__":
    main()
