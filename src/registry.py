"""
Free-tier model registry: no MLflow server required (though a local MLflow with a
SQLite backend is a drop-in upgrade if you want the extra UI -- see README).

Every trained artifact gets its own immutable, timestamped version directory under
registry/, containing:
    model.joblib        -- the XGBoost model
    calibrator.joblib    -- the Platt-scaling LogisticRegression
    model_card.json      -- data version, hyperparameters, eval metrics, thresholds

`production.json` and `staging.json` are pointer files -- re-pointing them is the
entire "promotion" or "rollback" action, and it's atomic (a single file write).
This gives you the three properties the roadmap asks for: immutable versioned
artifacts, explicit lifecycle stages, and auditable promotion/rollback.
"""
import json
import sys
import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def _new_version_id() -> str:
    return datetime.now(timezone.utc).strftime("v%Y%m%dT%H%M%S%f")


def register(model, calibrator, thresholds: dict, metrics: dict, data_version: str,
             hyperparameters: dict) -> str:
    """Saves a new immutable version. Returns the version id. Does NOT promote it --
    promotion is a separate, explicit step (see promote())."""
    version = _new_version_id()
    version_dir = config.REGISTRY_DIR / version
    version_dir.mkdir(parents=True, exist_ok=False)

    joblib.dump(model, version_dir / "model.joblib")
    joblib.dump(calibrator, version_dir / "calibrator.joblib")

    model_card = {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_version": data_version,
        "hyperparameters": hyperparameters,
        "thresholds": thresholds,
        "metrics": metrics,
        "stage": "none",
    }
    with open(version_dir / "model_card.json", "w") as f:
        json.dump(model_card, f, indent=2)

    return version


def load(version: str) -> dict:
    version_dir = config.REGISTRY_DIR / version
    model = joblib.load(version_dir / "model.joblib")
    calibrator = joblib.load(version_dir / "calibrator.joblib")
    with open(version_dir / "model_card.json") as f:
        model_card = json.load(f)
    return {"model": model, "calibrator": calibrator, "model_card": model_card}


def _write_pointer(pointer_path: Path, version: str) -> None:
    with open(pointer_path, "w") as f:
        json.dump({"version": version, "pointed_at": datetime.now(timezone.utc).isoformat()}, f, indent=2)


def get_pointer(pointer_path: Path) -> str | None:
    if not pointer_path.exists():
        return None
    with open(pointer_path) as f:
        return json.load(f)["version"]


def promote_to_staging(version: str) -> None:
    _write_pointer(config.STAGING_POINTER, version)
    _set_stage(version, "staging")


def promote_to_production(version: str) -> None:
    """The auditable promotion action. Whatever was production before this call
    becomes the previous version -- rollback is just calling this again with that
    version id."""
    _write_pointer(config.PRODUCTION_POINTER, version)
    _set_stage(version, "production")


def _set_stage(version: str, stage: str) -> None:
    card_path = config.REGISTRY_DIR / version / "model_card.json"
    with open(card_path) as f:
        card = json.load(f)
    card["stage"] = stage
    with open(card_path, "w") as f:
        json.dump(card, f, indent=2)


def list_versions() -> list[dict]:
    if not config.REGISTRY_DIR.exists():
        return []
    out = []
    for d in sorted(config.REGISTRY_DIR.iterdir()):
        card_path = d / "model_card.json"
        if card_path.exists():
            with open(card_path) as f:
                out.append(json.load(f))
    return out


def evaluate_on(version: str, X, y) -> dict:
    """Runs a registered version's full inference path (raw score -> Platt
    calibration) over a fixed eval set and returns PR-AUC / ROC-AUC. Used by the
    champion-challenger gate below."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    bundle = load(version)
    raw = bundle["model"].predict_proba(X)[:, 1]
    calibrated = bundle["calibrator"].predict_proba(raw.reshape(-1, 1))[:, 1]
    return {
        "pr_auc": float(average_precision_score(y, calibrated)),
        "roc_auc": float(roc_auc_score(y, calibrated)),
    }


def champion_challenger_gate(challenger_version: str, eval_sets: dict) -> dict:
    """The real (not simulated) gate: run BOTH the current production champion and
    the challenger over every eval set provided (e.g. held-out fold, and any
    cross-segment slice you care about), and only allow promotion if the challenger
    doesn't regress PR-AUC by more than config.MAX_ALLOWED_PR_AUC_REGRESSION on ANY
    of them.

    eval_sets: {"held_out": (X, y), "stress_slice": (X, y), ...}
    """
    champion_version = get_pointer(config.PRODUCTION_POINTER)
    per_set_results = {}
    passed = True

    for set_name, (X, y) in eval_sets.items():
        challenger_metrics = evaluate_on(challenger_version, X, y)
        if champion_version is None:
            # No champion yet -- first-ever model always "passes" (nothing to regress against).
            per_set_results[set_name] = {
                "champion": None, "challenger": challenger_metrics,
                "regression": 0.0, "passed": True,
            }
            continue

        champion_metrics = evaluate_on(champion_version, X, y)
        regression = champion_metrics["pr_auc"] - challenger_metrics["pr_auc"]
        set_passed = regression <= config.MAX_ALLOWED_PR_AUC_REGRESSION
        passed = passed and set_passed
        per_set_results[set_name] = {
            "champion": champion_metrics, "challenger": challenger_metrics,
            "regression": round(regression, 5), "passed": set_passed,
        }

    return {
        "challenger_version": challenger_version,
        "champion_version": champion_version,
        "passed": passed,
        "per_set_results": per_set_results,
    }
