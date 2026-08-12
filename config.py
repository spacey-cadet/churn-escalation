"""
Single source of truth for paths, table/bucket names, and tunables used across
the pipeline.

Local files stay local: training data (`data/*.parquet`) and the dead-letter
queue live in the repo checkout (your machine, or the GitHub Actions runner) --
ingestion, training, and the quality gates run periodically on a machine that
has this repo checked out, not inside Lambda.

The feature store and model registry are AWS-backed (DynamoDB + S3) so the
serving API can run in Lambda, where the filesystem is read-only outside
`/tmp` and nothing written there survives between invocations or concurrent
executions. See src/feature_store.py and src/registry.py for the
implementations -- nothing else in the repo needed to change, because both
kept their original function signatures.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# --- data (local -- training/gates run outside Lambda) ---
DATA_DIR = ROOT / "data"
RAW_LANDING = DATA_DIR / "raw_landing.parquet"
FEATURES_CLEAN = DATA_DIR / "features_clean.parquet"
LABELS_DELAYED = DATA_DIR / "labels_delayed.parquet"
DEAD_LETTER_DIR = DATA_DIR / "dead_letter_queue"

FEATURES = [
    "support_tickets_30d", "avg_message_length", "satisfaction_score",
    "days_since_last_login", "tenure_days", "monthly_spend",
]

# --- model registry (S3-backed -- see src/registry.py) ---
# These are now POINTER NAMES, not local file paths -- registry.py maps them to
# S3 keys under pointers/<name>.json. Kept as plain strings (rather than, say,
# an enum) so this file stays a drop-in-compatible source of truth for anything
# that used to do `config.PRODUCTION_POINTER`.
PRODUCTION_POINTER = "production"
STAGING_POINTER = "staging"
S3_REGISTRY_BUCKET = os.environ.get("S3_REGISTRY_BUCKET", "churn-registry")

# --- feature store (DynamoDB-backed -- see src/feature_store.py) ---
DYNAMODB_ONLINE_TABLE = os.environ.get("DYNAMODB_ONLINE_TABLE", "churn-online-features")
DYNAMODB_OFFLINE_TABLE = os.environ.get("DYNAMODB_OFFLINE_TABLE", "churn-offline-features")
DYNAMODB_LOG_TABLE = os.environ.get("DYNAMODB_LOG_TABLE", "churn-inference-log")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

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
