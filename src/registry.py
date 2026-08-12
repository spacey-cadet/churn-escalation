"""
AWS-backed model registry: S3 standing in for the local `registry/` directory.
Same function signatures as the original, so nothing outside this file needs to
change (scripts/evaluate_and_promote.py, src/modeling/train.py, src/serving/app.py,
src/drift_monitor.py all call these functions exactly as before).

Every trained artifact gets its own immutable key prefix under `versions/<version>/`
in the registry bucket:
    model.joblib
    calibrator.joblib
    model_card.json

`pointers/production.json` and `pointers/staging.json` are the promotion/rollback
pointers, same idea as before -- re-pointing them (a single put_object call) is the
entire "deploy" or "rollback" action.

NOTE: config.PRODUCTION_POINTER / config.STAGING_POINTER now hold POINTER NAMES
("production" / "staging"), not local file Paths -- see the updated config.py.
"""
import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import boto3
import joblib
from botocore.exceptions import ClientError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

BUCKET = os.environ.get("S3_REGISTRY_BUCKET", "churn-registry")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

_s3 = boto3.client("s3", region_name=AWS_REGION)


def _new_version_id() -> str:
    return datetime.now(timezone.utc).strftime("v%Y%m%dT%H%M%S%f")


def _version_prefix(version: str) -> str:
    return f"versions/{version}/"


def _pointer_key(name: str) -> str:
    return f"pointers/{name}.json"


def _put_joblib(key: str, obj) -> None:
    buf = io.BytesIO()
    joblib.dump(obj, buf)
    buf.seek(0)
    _s3.put_object(Bucket=BUCKET, Key=key, Body=buf.getvalue())


def _get_joblib(key: str):
    resp = _s3.get_object(Bucket=BUCKET, Key=key)
    return joblib.load(io.BytesIO(resp["Body"].read()))


def _put_json(key: str, data: dict) -> None:
    _s3.put_object(Bucket=BUCKET, Key=key, Body=json.dumps(data, indent=2).encode())


def _get_json(key: str) -> dict:
    resp = _s3.get_object(Bucket=BUCKET, Key=key)
    return json.loads(resp["Body"].read())


def register(model, calibrator, thresholds: dict, metrics: dict, data_version: str,
             hyperparameters: dict) -> str:
    """Saves a new immutable version. Returns the version id. Does NOT promote it --
    promotion is a separate, explicit step (see promote_to_production())."""
    version = _new_version_id()
    prefix = _version_prefix(version)

    _put_joblib(prefix + "model.joblib", model)
    _put_joblib(prefix + "calibrator.joblib", calibrator)

    model_card = {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_version": data_version,
        "hyperparameters": hyperparameters,
        "thresholds": thresholds,
        "metrics": metrics,
        "stage": "none",
    }
    _put_json(prefix + "model_card.json", model_card)
    return version


def load(version: str) -> dict:
    prefix = _version_prefix(version)
    model = _get_joblib(prefix + "model.joblib")
    calibrator = _get_joblib(prefix + "calibrator.joblib")
    model_card = _get_json(prefix + "model_card.json")
    return {"model": model, "calibrator": calibrator, "model_card": model_card}


def _write_pointer(pointer_name: str, version: str) -> None:
    _put_json(_pointer_key(pointer_name), {
        "version": version, "pointed_at": datetime.now(timezone.utc).isoformat(),
    })


def get_pointer(pointer_name: str) -> str | None:
    try:
        return _get_json(_pointer_key(pointer_name))["version"]
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return None
        raise


def promote_to_staging(version: str) -> None:
    _write_pointer(config.STAGING_POINTER, version)
    _set_stage(version, "staging")


def promote_to_production(version: str) -> None:
    """The auditable promotion action. Whatever was production before this call
    becomes the previous version -- rollback is just calling this again with
    that version id."""
    _write_pointer(config.PRODUCTION_POINTER, version)
    _set_stage(version, "production")


def _set_stage(version: str, stage: str) -> None:
    key = _version_prefix(version) + "model_card.json"
    card = _get_json(key)
    card["stage"] = stage
    _put_json(key, card)


def list_versions() -> list[dict]:
    """Lists all registered versions, oldest first. Version ids are
    lexicographically sortable timestamps (vYYYYMMDDTHHMMSSffffff), so an
    explicit sort on the prefix preserves chronological order without
    depending on S3 ListObjectsV2's ordering behavior (which happens to be
    lexicographic too, but relying on that implicitly is fragile)."""
    paginator = _s3.get_paginator("list_objects_v2")
    prefixes = set()
    for page in paginator.paginate(Bucket=BUCKET, Prefix="versions/", Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            prefixes.add(cp["Prefix"])

    out = []
    for prefix in sorted(prefixes):
        try:
            out.append(_get_json(prefix + "model_card.json"))
        except ClientError:
            continue  # partially-written version, e.g. an interrupted register() call
    return out


def evaluate_on(version: str, X, y) -> dict:
    """Unchanged from the local version -- pure logic over load()'s output."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    bundle = load(version)
    raw = bundle["model"].predict_proba(X)[:, 1]
    calibrated = bundle["calibrator"].predict_proba(raw.reshape(-1, 1))[:, 1]
    return {
        "pr_auc": float(average_precision_score(y, calibrated)),
        "roc_auc": float(roc_auc_score(y, calibrated)),
    }


def champion_challenger_gate(challenger_version: str, eval_sets: dict) -> dict:
    """Unchanged from the local version -- pure logic over get_pointer()/evaluate_on()."""
    champion_version = get_pointer(config.PRODUCTION_POINTER)
    per_set_results = {}
    passed = True

    for set_name, (X, y) in eval_sets.items():
        challenger_metrics = evaluate_on(challenger_version, X, y)
        if champion_version is None:
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
