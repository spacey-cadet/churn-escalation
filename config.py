"""
Single source of truth for paths and tunables used across the pipeline.
Nothing here talks to a paid service. Everything is a local file, a local
SQLite DB, or an optional webhook URL read from the environment.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# --- data ---
DATA_DIR = ROOT / "data"
RAW_LANDING = DATA_DIR / "raw_landing.parquet"
FEATURES_CLEAN = DATA_DIR / "features_clean.parquet"
LABELS_DELAYED = DATA_DIR / "labels_delayed.parquet"
DEAD_LETTER_DIR = DATA_DIR / "dead_letter_queue"

FEATURES = [
    "support_tickets_30d", "avg_message_length", "satisfaction_score",
    "days_since_last_login", "tenure_days", "monthly_spend",
]

# --- model registry (local, file-based -- see src/registry.py) ---
REGISTRY_DIR = ROOT / "registry"
PRODUCTION_POINTER = REGISTRY_DIR / "production.json"   # which version is "champion"
STAGING_POINTER = REGISTRY_DIR / "staging.json"          # which version is "challenger"

# --- feature store (SQLite standing in for Redis + partitioned Parquet-on-S3) ---
FEATURE_STORE_DB = ROOT / "feature_store_db" / "feature_store.sqlite"

# --- ingestion gate thresholds (Great Expectations equivalent) ---
ROLLING_AVG_ROW_COUNT = 20_000
VOLUME_DROP_ALERT_PCT = 0.30
# A few null entity_ids get quarantined to the DLQ and the batch proceeds without
# them -- that's the DESIGNED remediation, not a pipeline-blocking failure. The
# batch only blocks entirely if quarantine would strip out more than this fraction,
# since that signals something structurally wrong with the whole landing batch
# rather than a handful of bad rows.
MAX_QUARANTINE_RATE = 0.05

# --- cascade cost matrices (Track 1: two independently-optimized thresholds) ---
COST_MATRIX_LOW = {"cost_fn": 8, "cost_fp": 1}   # auto-resolve / review boundary
COST_MATRIX_HIGH = {"cost_fn": 3, "cost_fp": 6}  # review / senior-escalation boundary

# --- drift monitoring ---
KS_PVALUE_ALERT_THRESHOLD = 0.05
PSI_MODERATE_THRESHOLD = 0.10
PSI_MAJOR_THRESHOLD = 0.25

# --- champion-challenger promotion gate ---
# A challenger must not regress PR-AUC by more than this on the eval set to be promotable.
MAX_ALLOWED_PR_AUC_REGRESSION = 0.01

# --- canary rollout (in-process, session/entity-hash-sticky) ---
CANARY_PCT = int(os.environ.get("CANARY_PCT", "0"))  # 0-100, % of traffic to challenger
CANARY_ERROR_RATE_SLA = 0.001  # 0.1%, mirrors the roadmap's rollback trigger

# --- alerting (optional; falls back to stdout if unset) ---
ALERT_WEBHOOK_URL = os.environ.get("ALERT_WEBHOOK_URL", "")

# --- serving ---
LOW_CONFIDENCE_BAND = 0.05  # scores within this distance of a threshold -> "ambiguous"
