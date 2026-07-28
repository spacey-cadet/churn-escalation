"""
Production inference API.

Run locally:   uvicorn src.serving.app:app --reload --port 8000
Run in Docker: see Dockerfile at repo root (this is exactly what HF Spaces/any
               free container host runs).

What this endpoint does, end to end, per request:
  1. Validates the request body (pydantic) -- the serving-gate equivalent of Gate 1.
  2. Picks champion or challenger via a session/entity-hash-sticky canary split, so
     the SAME entity_id always lands on the same model version for the duration of
     a rollout (no mid-conversation-equivalent model flip).
  3. Runs calibrated inference, applies the two-threshold cascade
     (auto_resolve / review_queue / senior_escalation), and flags low-confidence /
     ambiguous scores near either threshold.
  4. Logs the request to the feature store (SQLite) for drift monitoring and the
     eventual label-delay backfill.
  5. Returns a bounded-latency response -- inference itself is a fast tree-model
     forward pass, but the pattern here (return-then-log, never block on non-
     essential work) is what protects a live caller from latency spikes.
"""
import sys
import time
import hashlib
from pathlib import Path
from contextlib import asynccontextmanager

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import config
import registry
import feature_store

_MODELS: dict = {}  # cache of loaded {"champion": bundle, "challenger": bundle|None}


def _load_models():
    champion_version = registry.get_pointer(config.PRODUCTION_POINTER)
    if champion_version is None:
        raise RuntimeError(
            "No production model registered yet. Run `python -m src.modeling.train` "
            "then `python scripts/evaluate_and_promote.py` before starting the server."
        )
    bundle = {"champion": {"version": champion_version, **registry.load(champion_version)}}

    staging_version = registry.get_pointer(config.STAGING_POINTER)
    if staging_version and config.CANARY_PCT > 0:
        bundle["challenger"] = {"version": staging_version, **registry.load(staging_version)}
    else:
        bundle["challenger"] = None
    return bundle


@asynccontextmanager
async def lifespan(app: FastAPI):
    _MODELS.update(_load_models())
    yield


app = FastAPI(title="Churn Escalation Detector", lifespan=lifespan)


class PredictionRequest(BaseModel):
    entity_id: str = Field(..., min_length=1, max_length=64)
    support_tickets_30d: float = Field(..., ge=0, le=1000)
    avg_message_length: float = Field(..., ge=0, le=20_000)
    satisfaction_score: float = Field(..., ge=1, le=5)
    days_since_last_login: float = Field(..., ge=0, le=3650)
    tenure_days: float = Field(..., ge=0, le=20_000)
    monthly_spend: float = Field(..., ge=0, le=1_000_000)


class PredictionResponse(BaseModel):
    entity_id: str
    calibrated_score: float
    tier: str
    low_confidence: bool
    served_by: str
    model_version: str
    latency_ms: float


def _pick_model(entity_id: str) -> str:
    """Session/entity-sticky canary split: hash the entity_id into [0, 100), route
    to the challenger if it's below CANARY_PCT and a challenger is actually staged.
    Hashing (not random.random() per request) means the SAME entity always gets the
    SAME model version for the life of the rollout -- no inconsistent experience
    across repeated requests from one user."""
    if _MODELS.get("challenger") is None or config.CANARY_PCT <= 0:
        return "champion"
    bucket = int(hashlib.sha256(entity_id.encode()).hexdigest(), 16) % 100
    return "challenger" if bucket < config.CANARY_PCT else "champion"


def _score_and_route(bundle: dict, features: dict) -> dict:
    X = np.array([[features[f] for f in config.FEATURES]])
    raw = bundle["model"].predict_proba(X)[:, 1]
    calibrated = float(bundle["calibrator"].predict_proba(raw.reshape(-1, 1))[:, 1][0])

    thresholds = bundle["model_card"]["thresholds"]
    t_low, t_high = thresholds["t_low"], thresholds["t_high"]

    if calibrated < t_low:
        tier = "auto_resolve"
    elif calibrated < t_high:
        tier = "review_queue"
    else:
        tier = "senior_escalation"

    near_low = abs(calibrated - t_low) <= config.LOW_CONFIDENCE_BAND
    near_high = abs(calibrated - t_high) <= config.LOW_CONFIDENCE_BAND
    low_confidence = near_low or near_high

    return {"calibrated_score": calibrated, "tier": tier, "low_confidence": low_confidence}


@app.get("/health")
def health():
    champion = _MODELS.get("champion")
    return {
        "status": "ok",
        "champion_version": champion["version"] if champion else None,
        "challenger_version": _MODELS["challenger"]["version"] if _MODELS.get("challenger") else None,
        "canary_pct": config.CANARY_PCT,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    start = time.perf_counter()

    if not _MODELS.get("champion"):
        raise HTTPException(status_code=503, detail="Model not loaded.")

    served_by = _pick_model(req.entity_id)
    bundle = _MODELS[served_by]
    features = req.model_dump(exclude={"entity_id"})

    result = _score_and_route(bundle, features)
    latency_ms = (time.perf_counter() - start) * 1000

    # Log for drift monitoring + backfill. Fire-and-forget in spirit: a local SQLite
    # write is sub-millisecond, so it's safe to do inline here; if this endpoint
    # ever sat in front of a slower store, this call is exactly what you'd move to
    # a background task so it can never add latency to the user-facing response.
    feature_store.write_features(req.entity_id, features)
    feature_store.log_inference(
        entity_id=req.entity_id, features=features,
        calibrated_score=result["calibrated_score"], tier=result["tier"],
        model_version=bundle["version"], served_by=served_by,
        low_confidence=result["low_confidence"],
    )

    return PredictionResponse(
        entity_id=req.entity_id,
        calibrated_score=result["calibrated_score"],
        tier=result["tier"],
        low_confidence=result["low_confidence"],
        served_by=served_by,
        model_version=bundle["version"],
        latency_ms=round(latency_ms, 3),
    )


@app.post("/admin/reload")
def reload_models():
    """Call this after promoting a new version (scripts/evaluate_and_promote.py)
    or after changing CANARY_PCT, so the running server picks up the new
    production/staging pointers without a full restart."""
    _MODELS.update(_load_models())
    return health()
