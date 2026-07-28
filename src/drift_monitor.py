"""
Serving Gate (Gate 3) equivalent, run as a scheduled job (see
.github/workflows/drift_check.yml) instead of an always-on streaming job -- the
free-tier version of "hourly KS test" is "KS test on a cron schedule."

Two independent checks, on purpose, because they catch different failure modes:

1. DATA DRIFT: KS test comparing each live logged feature's distribution (pulled
   from feature_store.read_inference_log) against the training baseline. Needs NO
   ground-truth labels, so it can run continuously even though this project's
   labels arrive 21 days late.

2. LABEL-DELAY PROXY: Population Stability Index (PSI) on the model's OUTPUT SCORE
   distribution over time. A shift here often precedes a labeled performance drop,
   and -- like the KS test -- needs no labels either.

Concept drift (the relationship between features and label changing) is NOT
detectable by either of these; it only shows up once real labels arrive, which is
what src/label_delay_backfill.py (adapted from the rehearsal scripts) is for.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

import config
import feature_store
import alerting


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    breakpoints = np.quantile(expected, np.linspace(0, 1, bins + 1))
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf
    expected_pct = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_pct = np.histogram(actual, breakpoints)[0] / len(actual)
    expected_pct = np.clip(expected_pct, 1e-6, None)
    actual_pct = np.clip(actual_pct, 1e-6, None)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def check_data_drift() -> dict:
    baseline = pd.read_parquet(config.FEATURES_CLEAN)
    logged = feature_store.read_inference_log(limit=50_000)

    if len(logged) < 30:
        return {"status": "insufficient_data", "n_logged": len(logged)}

    logged_features = pd.DataFrame([r["features"] for r in logged])
    results = {}
    any_drift = False
    for col in config.FEATURES:
        if col not in logged_features.columns:
            continue
        stat, p = ks_2samp(baseline[col], logged_features[col])
        drifted = bool(p < config.KS_PVALUE_ALERT_THRESHOLD)
        any_drift = any_drift or drifted
        results[col] = {"ks_stat": round(float(stat), 4), "p_value": round(float(p), 6), "drifted": drifted}

    return {"status": "checked", "n_logged": len(logged), "any_drift": bool(any_drift), "per_feature": results}


def check_score_psi() -> dict:
    baseline = pd.read_parquet(config.FEATURES_CLEAN)
    logged = feature_store.read_inference_log(limit=50_000)
    if len(logged) < 30:
        return {"status": "insufficient_data", "n_logged": len(logged)}

    # Training-time reference: use the champion's calibrated scores over the
    # training set as the "expected" distribution.
    import registry
    champion_version = registry.get_pointer(config.PRODUCTION_POINTER)
    if champion_version is None:
        return {"status": "no_production_model"}

    bundle = registry.load(champion_version)
    raw = bundle["model"].predict_proba(baseline[config.FEATURES])[:, 1]
    expected_scores = bundle["calibrator"].predict_proba(raw.reshape(-1, 1))[:, 1]

    actual_scores = np.array([r["calibrated_score"] for r in logged])
    psi_value = psi(expected_scores, actual_scores)

    if psi_value > config.PSI_MAJOR_THRESHOLD:
        interpretation = "MAJOR shift -- investigate immediately, do not wait for labels"
    elif psi_value > config.PSI_MODERATE_THRESHOLD:
        interpretation = "MODERATE shift -- early warning"
    else:
        interpretation = "stable"

    return {"status": "checked", "psi": round(psi_value, 4), "interpretation": interpretation, "n_logged": len(logged)}


def main():
    print("=== Data drift check (KS test, no labels needed) ===")
    drift_result = check_data_drift()
    print(drift_result)
    if drift_result.get("any_drift"):
        drifted_cols = [c for c, r in drift_result["per_feature"].items() if r["drifted"]]
        alerting.send_alert(
            f"Data drift detected on features: {drifted_cols}. "
            "Consider moving the next retrain up sooner.", severity="critical",
        )

    print("\n=== Score PSI check (label-delay proxy) ===")
    psi_result = check_score_psi()
    print(psi_result)
    if psi_result.get("status") == "checked" and psi_result["psi"] > config.PSI_MODERATE_THRESHOLD:
        alerting.send_alert(f"Score PSI = {psi_result['psi']:.4f} ({psi_result['interpretation']})")


if __name__ == "__main__":
    main()
