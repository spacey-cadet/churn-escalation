# Churn Escalation Detector -- Free-Tier Production Pipeline

A real, running version of the roadmap in `reference_rehearsal_scripts/` -- not a
simulation of it. Every gate, the registry, the feature store, the canary split,
and the drift monitor are wired together into one pipeline you can actually run,
retrain, and serve from, using nothing but local files, SQLite, and free hosted
tiers (GitHub Actions, Hugging Face Spaces, Slack/Discord webhooks).

Two things shape this build, on purpose:
- **You retrain periodically, by hand**, on refreshed/re-curated data -- not
  continuously. So every retrain goes through a real champion-challenger gate
  before it can reach production, and the model registry is the thing that makes
  "what changed between retrains" answerable, not just a nice-to-have.
- **Inference serves a live caller** (a chat app, a support tool, whatever's in
  front of this API) -- so validation, latency, and per-customer routing
  consistency across a rollout all matter for real, not just as talking points.

## Architecture

```
                          ┌─────────────────────────┐
  data/generate_data.py ->│  raw_landing.parquet      │
  (or your real ingest)   └───────────┬───────────────┘
                                      v
                       ┌───────────────────────────────┐
                       │  Gate 1: Ingestion             │  src/quality/ingestion_gate.py
                       │  null checks, volume checks,   │  -> Dead Letter Queue on bad rows
                       │  DLQ routing                   │  -> Slack/Discord alert on volume drop
                       └───────────────┬───────────────┘
                                      v
                       ┌───────────────────────────────┐
                       │  Gate 2: Transformation        │  src/quality/transformation_gate.py
                       │  unique/not_null, referential  │  -> blocks downstream on failure
                       │  integrity (dbt-style)         │  -> alert fired
                       └───────────────┬───────────────┘
                                      v
                       ┌───────────────────────────────┐
                       │  Train: baseline hierarchy ->  │  src/modeling/train.py
                       │  XGBoost -> Platt calibration  │  -> registers a NEW version,
                       │  -> two cascade thresholds      │     does NOT touch production
                       └───────────────┬───────────────┘
                                      v
                       ┌───────────────────────────────┐
                       │  Champion-Challenger Gate      │  scripts/evaluate_and_promote.py
                       │  eval on held-out + stress      │  -> promotes to production ONLY
                       │  slice, real numbers, real gate │     if it doesn't regress PR-AUC
                       └───────────────┬───────────────┘
                                      v
                       ┌───────────────────────────────┐
                       │  Model Registry                │  src/registry.py + registry/
                       │  immutable versions, model      │  production.json / staging.json
                       │  cards, promotion pointers      │  are the entire "deploy" action
                       └───────────────┬───────────────┘
                                      v
                       ┌───────────────────────────────┐
                       │  Serving API (FastAPI)          │  src/serving/app.py
                       │  pydantic validation, cascade    │  session/entity-sticky canary
                       │  routing, feature store logging  │  split between champion/challenger
                       └───────────────┬───────────────┘
                                      v
                       ┌───────────────────────────────┐
                       │  Feature Store (SQLite)         │  src/feature_store.py
                       │  online (latest) + offline       │  point-in-time joins,
                       │  (history) + inference log        │  reconciliation job
                       └───────────────┬───────────────┘
                                      v
                       ┌───────────────────────────────┐
                       │  Drift Monitor + Backfill        │  src/drift_monitor.py
                       │  KS test + PSI (no labels        │  src/label_delay_backfill.py
                       │  needed) + backfilled F1/P/R       │  (labels, 21 days later)
                       │  once labels arrive               │
                       └───────────────────────────────┘
```

## Free-tier tool mapping

The roadmap describes a system built on AWS, Kubernetes, Istio, Airflow, PagerDuty,
Redis, Feast, and a hosted MLflow. None of those are required to get the same
*mechanics*. Here's the substitution table -- know it, because "why didn't you use
the real thing" is a fair interview question and the honest answer is "the
mechanism is identical, the hosting is just free":

| Roadmap component | Paid/hosted version | What this repo actually uses | Why it's equivalent |
|---|---|---|---|
| Ingestion validation | Great Expectations (managed) | `src/quality/ingestion_gate.py` -- same checks, same thresholds, in plain pandas (real GE also supported if installed, see the module) | Same expectations, same DLQ/alert behavior |
| Transformation tests | dbt Cloud + a warehouse | `src/quality/transformation_gate.py` -- same `unique`/`not_null`/`relationships` logic in pandas; real dbt YAML equivalent is in `reference_rehearsal_scripts/phase2_pipeline/` | Identical test semantics, no warehouse needed |
| Orchestration/blocking | Airflow + PagerDuty | Non-zero exit codes chained in `scripts/run_pipeline.sh`, surfaced as a failed GitHub Actions run | A failing gate stops the pipeline either way |
| Drift detection | Whylogs/Evidently + a streaming job | `src/drift_monitor.py`, run hourly via `.github/workflows/drift_check.yml` (free scheduled Actions) | Same KS test, cron instead of streaming |
| Model registry | MLflow Model Registry (hosted) | `src/registry.py` -- versioned local dirs + JSON model cards + pointer files (`registry/production.json`, `registry/staging.json`). A local MLflow with a SQLite backend is a drop-in upgrade if you want the UI. | Same properties: immutable versions, lifecycle stages, auditable promotion |
| Feature store | Feast + Redis + S3 | `src/feature_store.py` -- one SQLite file, an "online" table with overwrite semantics and an "offline" table that's append-only | Same point-in-time-join and online/offline split, one inspectable file |
| Canary rollout | Kubernetes + Istio traffic splitting | `CANARY_PCT` env var + a SHA256 hash of `entity_id` in `src/serving/app.py` | Same sticky-split behavior, no service mesh |
| Alerting | PagerDuty/Opsgenie | A plain webhook POST in `src/alerting.py` to a free Slack or Discord incoming webhook | Same "someone gets pinged" outcome |
| Hosting | ECS/EKS | Any free container host that runs `Dockerfile` as-is (Hugging Face Spaces' Docker SDK, Fly.io's free allowance, or just `docker run` locally) | Same artifact, different host |
| CI/CD gate | Jenkins/CircleCI | `.github/workflows/ci.yml` -- free on GitHub Actions | Same block-on-failure behavior |

**What's genuinely NOT equivalent**, and would need real infra if you scaled this
past a single-instance deployment: multi-region failover, sub-10ms feature lookups
under heavy concurrent load (SQLite will not out-perform Redis at scale), and fully
automated retraining triggered directly off a drift alert (this repo gets you the
human-in-the-loop version -- a drift alert tells you to retrain *sooner*, not a
robot that retrains *for* you). Be upfront about that distinction in an interview;
it's the honest version of "I know where the free-tier analogy breaks."

## Setup

```bash
pip install -r requirements.txt        # or: pip install --break-system-packages -r requirements.txt
```

No `great_expectations` or `dbt-core` install required -- the gates implement the
same logic natively. Uncomment them in `requirements.txt` if you want the real
CLI/YAML experience too (see `reference_rehearsal_scripts/phase2_pipeline/` for
the literal syntax either would use).

## Running it

```bash
bash scripts/run_pipeline.sh
```

This does, in order: generate synthetic data -> ingestion gate -> transformation
gate -> train + register a new model version -> champion-challenger gate ->
promote to production if it passes. Every step exits non-zero on failure, so a
broken gate stops the run, same as Airflow blocking a downstream DAG.

Then start the API:

```bash
uvicorn src.serving.app:app --reload --port 8000
```

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "entity_id": "cust_00042",
  "support_tickets_30d": 4,
  "avg_message_length": 410,
  "satisfaction_score": 1.9,
  "days_since_last_login": 18,
  "tenure_days": 240,
  "monthly_spend": 38
}'
```

```json
{
  "entity_id": "cust_00042",
  "calibrated_score": 0.89,
  "tier": "senior_escalation",
  "low_confidence": false,
  "served_by": "champion",
  "model_version": "v20260720T181550315582",
  "latency_ms": 1.9
}
```

`tier` is one of `auto_resolve` / `review_queue` / `senior_escalation`, from the
two independently-optimized cascade thresholds (Track 1). `low_confidence` flags
scores sitting within `LOW_CONFIDENCE_BAND` of either threshold -- worth routing to
a human reviewer even if the tier logic alone wouldn't. `served_by` tells you
whether this request landed on the champion or a staged challenger, decided by a
SHA256 hash of `entity_id` so the same customer always lands on the same model
version for the life of a canary rollout.

## Retraining (the cadence this repo actually assumes)

Whenever you've refreshed or re-curated real data into
`data/features_clean.parquet` / `data/labels_delayed.parquet`:

```bash
bash scripts/retrain.sh
```

This re-runs the transformation gate against the new data, trains a new
candidate, and runs it through the champion-challenger gate. It's promoted to
production automatically **only if** it doesn't regress PR-AUC by more than
`config.MAX_ALLOWED_PR_AUC_REGRESSION` on any eval slice (the held-out fold *and*
a harder high-ticket-volume slice, standing in for the roadmap's cross-corpus
check). If it fails, it's staged instead -- inspect it with:

```bash
python3 scripts/evaluate_and_promote.py --version <the_new_version> --dry-run
```

If a running server needs to pick up a newly-promoted version:

```bash
curl -X POST http://localhost:8000/admin/reload
```

### Closing the loop with drift + label delay

Ground truth in this problem shape arrives 21 days late. Two jobs bridge that gap
and both feed the *next* retrain:

```bash
python3 src/drift_monitor.py          # KS test + PSI, no labels needed, run hourly (see .github/workflows/drift_check.yml)
python3 -m src.label_delay_backfill   # once labels land, recompute real precision/recall/F1
```

A drift or backfill alert is your signal to move the next retrain up sooner rather
than waiting out the usual cadence -- and any clips/rows the backfill flags as
genuinely mislabeled or hard are exactly what should get folded into the next
retrain's dataset.

### Canary rollout

```bash
CANARY_PCT=10 uvicorn src.serving.app:app --port 8000   # stage a challenger first:
                                                          #   python3 scripts/evaluate_and_promote.py --version <v> --dry-run
                                                          #   (then registry.promote_to_staging(<v>) -- see scripts/evaluate_and_promote.py)
```

Step `CANARY_PCT` up manually (10 -> 25 -> 50 -> 100), watching
`GET /health` and the drift monitor's error signals between steps. Any anomaly:
set `CANARY_PCT=0` and hit `/admin/reload` -- that's the rollback, and it's
instant because it's just re-pointing which version gets loaded.

## Testing

```bash
pytest tests/ -v
```

Covers: both quality gates (pass and fail paths), point-in-time joins actually
ignoring future writes (the single most important property in the whole repo),
the reconciliation job catching an injected online/offline sync bug, registry
round-tripping, the first-model-always-passes-the-gate edge case, and cascade
threshold ordering.

## CI/CD

`.github/workflows/ci.yml` runs the test suite and the full pipeline (including
the champion-challenger gate) on every push -- free on GitHub Actions.
`.github/workflows/drift_check.yml` runs `src/drift_monitor.py` hourly, the free
scheduled-Actions equivalent of an always-on streaming drift job (note the comment
inside it: you'll need to point it at wherever your deployed API's
`feature_store.sqlite` actually lives).

## Deploying the API

`Dockerfile` builds the serving API as a single container. This runs as-is on:
- **Hugging Face Spaces** (Docker SDK) -- free tier, easiest path to a public URL.
- **Fly.io** or **Railway** free tiers -- `fly launch` / `railway up` against this Dockerfile.
- Locally: `docker build -t churn-detector . && docker run -p 8000:8000 churn-detector`.

Bake a trained registry into the image by running `scripts/run_pipeline.sh` before
`docker build`, or mount a volume with a pre-populated `registry/` directory.

## Repo layout

```
config.py                     Every threshold, path, and tunable in one place
data/                          generate_data.py + the synthetic dataset
src/
  quality/
    ingestion_gate.py          Gate 1
    transformation_gate.py     Gate 2
  modeling/
    train.py                   baseline -> XGBoost -> calibration -> cascade thresholds -> register
  serving/
    app.py                     FastAPI: validation, cascade routing, canary split, logging
  feature_store.py             SQLite online/offline store + point-in-time join + reconciliation
  registry.py                  versioned model registry + champion-challenger gate
  drift_monitor.py             KS test + PSI, no labels needed
  label_delay_backfill.py      real precision/recall/F1 once labels arrive
  alerting.py                  webhook alerts (Slack/Discord), stdout fallback
scripts/
  run_pipeline.sh               full pipeline, first run
  retrain.sh                    the script you actually run on your retrain cadence
  evaluate_and_promote.py       the champion-challenger gate + promotion CLI
tests/test_pipeline.py         pytest suite, run in CI
.github/workflows/             ci.yml (tests + gate), drift_check.yml (scheduled)
Dockerfile                     serving API container
reference_rehearsal_scripts/   the original phase1-4 standalone practice scripts
                                (kept for STAR-story mining -- see phase4_system_design/)
```

## What "done" looks like

You should be able to: run `scripts/run_pipeline.sh` cold and get a served model;
kill a gate on purpose (feed it bad data) and watch it block with a real alert;
retrain, watch the champion-challenger gate actually reject a worse model; flip on
a canary and confirm the same customer always lands on the same version; and pull
up `feature_store.sqlite` in any SQLite browser and see real logged predictions.
That's the operational-fluency bar the roadmap is actually asking for.

