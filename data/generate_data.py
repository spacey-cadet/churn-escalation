"""
Generates a synthetic, imbalanced "customer escalation risk" dataset.

Produces:
    data/raw_landing.parquet     -- looks like what lands in S3/Kafka before any cleaning
    data/features_clean.parquet  -- what it looks like after the transformation gate
    data/labels_delayed.parquet  -- ground-truth labels that "arrive" 21 days after prediction

Positive rate is deliberately ~3% to force you to deal with class imbalance for real,
the way the roadmap's "imbalanced data caveat" describes.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RNG = np.random.default_rng(42)
N = 20_000
POSITIVE_RATE = 0.03


def generate_raw_landing():
    n_pos = int(N * POSITIVE_RATE)
    n_neg = N - n_pos

    def make_block(n, is_positive):
        base_tickets = RNG.poisson(3.5 if is_positive else 0.8, n)
        msg_len = RNG.normal(420 if is_positive else 140, 80, n).clip(10)
        satisfaction = RNG.normal(2.1 if is_positive else 4.2, 0.6, n).clip(1, 5)
        days_since_login = RNG.exponential(14 if is_positive else 3, n)
        tenure_days = RNG.exponential(200, n)
        monthly_spend = RNG.normal(45, 15, n).clip(0)
        return pd.DataFrame({
            "support_tickets_30d": base_tickets,
            "avg_message_length": msg_len,
            "satisfaction_score": satisfaction,
            "days_since_last_login": days_since_login,
            "tenure_days": tenure_days,
            "monthly_spend": monthly_spend,
            "label": int(is_positive),
        })

    pos = make_block(n_pos, True)
    neg = make_block(n_neg, False)
    df = pd.concat([pos, neg], ignore_index=True)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    df["entity_id"] = [f"cust_{i:06d}" for i in range(len(df))]

    event_start = datetime(2026, 1, 1)
    df["event_timestamp"] = [
        event_start + timedelta(minutes=int(m))
        for m in RNG.integers(0, 60 * 24 * 90, size=len(df))
    ]

    dirty_idx = RNG.choice(len(df), size=25, replace=False)
    df.loc[dirty_idx, "entity_id"] = None

    df = df.sort_values("event_timestamp").reset_index(drop=True)
    return df


def main():
    raw = generate_raw_landing()
    raw.to_parquet("data/raw_landing.parquet", index=False)

    clean = raw.dropna(subset=["entity_id"]).reset_index(drop=True)
    clean.to_parquet("data/features_clean.parquet", index=False)

    labels = clean[["entity_id", "event_timestamp", "label"]].copy()
    labels["label_observed_timestamp"] = labels["event_timestamp"] + timedelta(days=21)
    labels.to_parquet("data/labels_delayed.parquet", index=False)

    print(f"raw_landing: {len(raw)} rows, {raw['entity_id'].isna().sum()} null entity_ids (intentional)")
    print(f"features_clean: {len(clean)} rows, positive rate = {clean['label'].mean():.4f}")
    print(f"labels_delayed: {len(labels)} rows, delayed by 21 days")


if __name__ == "__main__":
    main()
