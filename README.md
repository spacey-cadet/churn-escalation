# Churn Escalation Detector — Local Pipeline + AWS Serverless Production

A real, running churn-escalation pipeline. Every gate, the
registry, the feature store, the canary split, and the drift monitor are wired
together end to end. It runs two ways from the **same codebase**:

- **Locally / free-tier hosted** — local files, SQLite, and free hosted tiers
  (GitHub Actions, Hugging Face Spaces, Slack/Discord webhooks). Cheapest path to
  a working, inspectable system; best for development, demos, and interview
  walkthroughs.
- **AWS Serverless** — the same model, feature logic, and gate gets served from a
  Lambda behind a Function URL, with DynamoDB as the online/offline feature store
  and S3 as the model registry, deployed via GitHub Actions using OIDC (no
  long-lived AWS keys). Best for an always-on public endpoint without paying for
  idle compute.

Two things shape this build, on purpose, in **both** deployment modes:

- **You retrain periodically, by hand**, on refreshed/re-curated data — not
  continuously. Every retrain goes through a real champion-challenger gate before
  it can reach production, and the model registry is the thing that makes "what
  changed between retrains" answerable, not just a nice-to-have.
- **Inference serves a live caller** (a chat app, a support tool, whatever's in
  front of this API) — so validation, latency, and per-customer routing
  consistency across a rollout all matter for real, not just as talking points.

---

## Architecture

The pipeline is identical up through the model registry. Where it diverges is
serving and storage — pick a lane at the bottom of the diagram.

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
                    ┌─────────────────┴─────────────────┐
                    v                                     v
      ┌───────────────────────────┐        ┌───────────────────────────────┐
      │  LOCAL: Serving API        │        │  AWS: Serving Lambda            │
      │  FastAPI, uvicorn           │        │  Same app logic, packaged as    │
      │  src/serving/app.py         │        │  a container image, invoked via │
      │  cascade routing, canary     │        │  a Lambda Function URL          │
      │  split, feature logging      │        │  (infra/aws-serverless/)        │
      └───────────────┬───────────┘        └───────────────┬───────────────┘
                       v                                    v
      ┌───────────────────────────┐        ┌───────────────────────────────┐
      │  LOCAL: Feature Store       │        │  AWS: Feature Store              │
      │  feature_store.sqlite       │        │  DynamoDB: churn-online-features,│
      │  online + offline tables     │        │  churn-offline-features,          │
      │  point-in-time joins          │        │  churn-inference-log              │
      └───────────────┬───────────┘        └───────────────┬───────────────┘
                       v                                    v
      ┌───────────────────────────┐        ┌───────────────────────────────┐
      │  LOCAL: Drift Monitor       │        │  AWS: Drift Monitor              │
      │  hourly cron, GitHub          │        │  hourly cron, GitHub              │
      │  Actions (drift_check.yml)  │        │  Actions (drift-check.yml),       │
      │  reads local sqlite           │        │  reads DynamoDB inference log     │
      └────────────────────────────┘        └───────────────────────────────┘
```

Both paths share `config.py`, `src/quality/`, `src/modeling/train.py`,
`src/registry.py`, and the champion-challenger gate — the only things that swap
out are **where state lives** (SQLite file vs. DynamoDB tables, local `registry/`
dir vs. an S3 bucket) and **how the API is invoked** (uvicorn process vs. Lambda
Function URL).

---

## Free-tier tool mapping (local mode)

The roadmap this repo is built against describes a system on AWS, Kubernetes,
Istio, Airflow, PagerDuty, Redis, Feast, and a hosted MLflow. None of those are
required to get the same *mechanics* in local mode. Know this table — "why didn't
you use the real thing" is a fair interview question, and the honest answer is
"the mechanism is identical, the hosting is just free":

| Roadmap component | Paid/hosted version | What local mode actually uses | Why it's equivalent |
|---|---|---|---|
| Ingestion validation | Great Expectations (managed) | `src/quality/ingestion_gate.py` — same checks, same thresholds, in plain pandas | Same expectations, same DLQ/alert behavior |
| Transformation tests | dbt Cloud + a warehouse | `src/quality/transformation_gate.py` — same `unique`/`not_null`/`relationships` logic in pandas | Identical test semantics, no warehouse needed |
| Orchestration/blocking | Airflow + PagerDuty | Non-zero exit codes chained in `scripts/run_pipeline.sh`, surfaced as a failed GitHub Actions run | A failing gate stops the pipeline either way |
| Drift detection | Whylogs/Evidently + a streaming job | `src/drift_monitor.py`, run hourly via `.github/workflows/drift_check.yml` | Same KS test, cron instead of streaming |
| Model registry | MLflow Model Registry (hosted) | `src/registry.py` — versioned local dirs + JSON model cards + pointer files | Same properties: immutable versions, lifecycle stages, auditable promotion |
| Feature store | Feast + Redis + S3 | `src/feature_store.py` — one SQLite file, online overwrite table + offline append-only table | Same point-in-time-join and online/offline split, one inspectable file |
| Canary rollout | Kubernetes + Istio traffic splitting | `CANARY_PCT` env var + SHA256 hash of `entity_id` in `src/serving/app.py` | Same sticky-split behavior, no service mesh |
| Alerting | PagerDuty/Opsgenie | A plain webhook POST in `src/alerting.py` to a free Slack/Discord webhook | Same "someone gets pinged" outcome |
| Hosting | ECS/EKS | Any free container host running `Dockerfile` as-is | Same artifact, different host |
| CI/CD gate | Jenkins/CircleCI | `.github/workflows/ci.yml` — free on GitHub Actions | Same block-on-failure behavior |

**What's genuinely NOT equivalent**, even scaled up: multi-region failover,
sub-10ms feature lookups under heavy concurrent load (SQLite won't out-perform
Redis at scale), and fully automated retraining triggered directly off a drift
alert (this repo is human-in-the-loop by design — a drift alert tells you to
retrain *sooner*, not a robot that retrains *for* you).

---

## AWS Serverless mapping (production mode)

Same pipeline, same gates, same registry logic — swapped onto AWS-managed,
pay-per-use services instead of local files. This is the upgrade path when you
want a durable public endpoint without keeping a machine on.

| Local component | AWS equivalent | Resource name(s) |
|---|---|---|
| Serving process (`uvicorn`) | Lambda, invoked via Function URL | Function: `churn-serving` |
| Container | Same `Dockerfile.lambda`, built via `docker buildx`, pushed to ECR | Repo: `churn-serving` (`395249043027.dkr.ecr.us-east-1.amazonaws.com/churn-serving`) |
| Feature store (SQLite) | DynamoDB, online + offline + inference log tables | `churn-online-features`, `churn-offline-features`, `churn-inference-log` |
| Model registry (`registry/` dir) | S3 bucket, same JSON pointer-file scheme | `churn-registry-395249043027` |
| CI/CD deploy (build → push → redeploy) | GitHub Actions, OIDC-authenticated (no stored AWS keys) | `.github/workflows/deploy-aws-serverless.yml` |
| Drift monitor cron | GitHub Actions scheduled workflow, reads DynamoDB instead of sqlite | `.github/workflows/drift-check.yml` |
| Retrain + promote | Manual `workflow_dispatch`, same gate logic, pings the live Lambda to reload | `.github/workflows/retrain-and-promote.yml` |
| Deploy identity | IAM role assumed via GitHub's OIDC provider | `churn-escalation-deploy-role` (inline policy `churn-escalation-ci-permissions`) |

**Auth model:** GitHub Actions requests a short-lived signed token from GitHub's
own OIDC provider, which AWS trusts directly — `configure-aws-credentials` swaps
that token for temporary AWS credentials via `sts:AssumeRoleWithWebIdentity`.
Nothing long-lived is stored in GitHub Secrets except the role's ARN itself.

**IAM permissions the deploy role needs, at minimum:**
- ECR: `GetAuthorizationToken`, `BatchCheckLayerAvailability`, `BatchGetImage`,
  `GetDownloadUrlForLayer`, `PutImage`, `InitiateLayerUpload`, `UploadLayerPart`,
  `CompleteLayerUpload` — note both read *and* write actions are required, since
  Docker/Buildx performs manifest-existence checks before a push completes.
- Lambda: `UpdateFunctionCode` scoped to the `churn-serving` function ARN.
- DynamoDB: `GetItem`, `PutItem`, `UpdateItem`, `Query` scoped to the three
  tables above.
- S3: `GetObject`, `PutObject`, `ListBucket` scoped to the registry bucket.

---

## Setup

```bash
pip install -r requirements.txt        # or: pip install --break-system-packages -r requirements.txt
```

No `great_expectations` or `dbt-core` install required — the gates implement the
same logic natively. Uncomment them in `requirements.txt` if you want the real
CLI/YAML experience too.

---

## Running it locally

```bash
bash scripts/run_pipeline.sh
```

This does, in order: generate synthetic data → ingestion gate → transformation
gate → train + register a new model version → champion-challenger gate → promote
to production if it passes. Every step exits non-zero on failure, so a broken
gate stops the run, same as Airflow blocking a downstream DAG.

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

`tier` is one of `auto_resolve` / `review_queue` / `senior_escalation`, from two
independently-optimized cascade thresholds. `low_confidence` flags scores sitting
within `LOW_CONFIDENCE_BAND` of either threshold — worth routing to a human
reviewer even if the tier logic alone wouldn't. `served_by` tells you whether this
request landed on the champion or a staged challenger, decided by a SHA256 hash of
`entity_id` so the same customer always lands on the same model version for the
life of a canary rollout.

---

## Running it on AWS

Deployment is push-triggered: any push to the `aws-serverless` branch (excluding
docs-only changes) runs `.github/workflows/deploy-aws-serverless.yml`, which:

1. Assumes `churn-escalation-deploy-role` via OIDC — no stored AWS keys.
2. Logs into ECR via `aws-actions/amazon-ecr-login`.
3. Builds the Lambda-compatible image with `docker buildx` (`--provenance=false
   --sbom=false` — required, or Lambda rejects the multi-manifest image index
   Buildx attaches by default).
4. Pushes to ECR, tagged both `:latest` and `:<commit-sha>`.
5. Calls `aws lambda update-function-code` to point the running function at the
   new image.

```bash
git push origin aws-serverless
gh run watch   # optional, watch the deploy live
```

Once deployed, invoke the same `/predict` contract shown above against the
Lambda's Function URL instead of `localhost:8000`.

---

## Model & data

- **Model:** baseline hierarchy → XGBoost → Platt calibration → two
  independently-tuned cascade thresholds (`auto_resolve` / `review_queue` /
  `senior_escalation`).
- **Training data:** `data/features_clean.parquet` + `data/labels_delayed.parquet`
  — refreshed/re-curated by hand between retrains, not streamed continuously.
  `data/generate_data.py` produces a synthetic stand-in for local development.
- **Label delay:** ground truth in this problem shape arrives **21 days late**.
  `src/label_delay_backfill.py` recomputes real precision/recall/F1 once labels
  land, feeding the *next* retrain rather than the current one.

---

## Retraining (the cadence this repo actually assumes)

**Locally:**

```bash
bash scripts/retrain.sh
```

Re-runs the transformation gate against the refreshed data, trains a new
candidate, and runs it through the champion-challenger gate. Promoted to
production automatically **only if** it doesn't regress PR-AUC by more than
`config.MAX_ALLOWED_PR_AUC_REGRESSION` on any eval slice (the held-out fold and a
harder high-ticket-volume slice). If it fails, it's staged instead:

```bash
python3 scripts/evaluate_and_promote.py --version <the_new_version> --dry-run
```

If a running local server needs to pick up a newly-promoted version:

```bash
curl -X POST http://localhost:8000/admin/reload
```

**On AWS:** trigger `.github/workflows/retrain-and-promote.yml` manually
(`workflow_dispatch` only — retraining is deliberately not automatic). It runs
`src.modeling.train`, then `scripts/evaluate_and_promote.py` against the same
gate, then POSTs to the live Lambda's `/admin/reload` endpoint (via
`SERVING_FUNCTION_URL`) so the deployed function picks up the newly-promoted
pointer without a redeploy.

### Closing the loop with drift + label delay

Two jobs bridge the 21-day label gap and both feed the *next* retrain:

```bash
python3 src/drift_monitor.py          # KS test + PSI, no labels needed
python3 -m src.label_delay_backfill   # once labels land, recompute real precision/recall/F1
```

Locally this runs against `feature_store.sqlite` on an hourly cron
(`drift_check.yml`). On AWS the equivalent (`drift-check.yml`) runs the same
monitor hourly against the live `churn-inference-log` DynamoDB table instead. A
drift or backfill alert is your signal to move the next retrain up sooner rather
than waiting out the usual cadence.

### Canary rollout (local)

```bash
CANARY_PCT=10 uvicorn src.serving.app:app --port 8000   # stage a challenger first:
                                                          #   python3 scripts/evaluate_and_promote.py --version <v> --dry-run
                                                          #   then registry.promote_to_staging(<v>)
```

Step `CANARY_PCT` up manually (10 → 25 → 50 → 100), watching `GET /health` and
the drift monitor's error signals between steps. Any anomaly: set `CANARY_PCT=0`
and hit `/admin/reload` — instant rollback, since it's just re-pointing which
version gets loaded.

*Note on AWS mode:* the Lambda deploy currently rolls a new image straight to
`:latest` with no traffic-split step — canarying at the Lambda layer (e.g. Lambda
aliases + weighted routing, or keeping the same `CANARY_PCT` env-var approach
inside the container) is a natural next step but isn't wired up yet. Stage and
verify with `--dry-run` before promoting on AWS in the meantime.

---

## Testing

```bash
pytest tests/ -v
```

Covers: both quality gates (pass and fail paths), point-in-time joins actually
ignoring future writes (the single most important property in the whole repo),
the reconciliation job catching an injected online/offline sync bug, registry
round-tripping, the first-model-always-passes-the-gate edge case, and cascade
threshold ordering.

---

## CI/CD

**Local mode:**
- `.github/workflows/ci.yml` — runs the test suite and the full pipeline
  (including the champion-challenger gate) on every push.
- `.github/workflows/drift_check.yml` — runs `src/drift_monitor.py` hourly
  against local `feature_store.sqlite`.

**AWS mode:**
- `.github/workflows/deploy-aws-serverless.yml` — build, push, redeploy on every
  push to `aws-serverless`.
- `.github/workflows/drift-check.yml` — runs the same drift monitor hourly
  against the live DynamoDB inference log.
- `.github/workflows/retrain-and-promote.yml` — manual-only retrain + gate +
  live reload.

All three AWS workflows authenticate via the same OIDC role
(`AWS_DEPLOY_ROLE_ARN` secret) rather than separate credentials per workflow.

---

## Deploying the API elsewhere

`Dockerfile` builds the serving API as a single container, usable outside AWS
too. Runs as-is on:

- **Hugging Face Spaces** (Docker SDK) — free tier, easiest path to a public URL.
- **Fly.io** or **Railway** free tiers — `fly launch` / `railway up` against this
  Dockerfile.
- Locally: `docker build -t churn-detector . && docker run -p 8000:8000 churn-detector`.

Bake a trained registry into the image by running `scripts/run_pipeline.sh` before
`docker build`, or mount a volume with a pre-populated `registry/` directory.
`Dockerfile.lambda` is the AWS-specific variant used for the Lambda container
image — same app, packaged for Lambda's runtime contract instead of a long-running
server process.

---

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
                                (same app code serves both local uvicorn and the AWS Lambda image)
  feature_store.py             SQLite online/offline store + point-in-time join + reconciliation
                                (local mode; DynamoDB tables serve the equivalent role on AWS)
  registry.py                  versioned model registry + champion-challenger gate
                                (local dirs; S3 bucket serves the equivalent role on AWS)
  drift_monitor.py             KS test + PSI, no labels needed
  label_delay_backfill.py      real precision/recall/F1 once labels arrive
  alerting.py                  webhook alerts (Slack/Discord), stdout fallback
scripts/
  run_pipeline.sh               full pipeline, first run
  retrain.sh                    the script you actually run on your local retrain cadence
  evaluate_and_promote.py       the champion-challenger gate + promotion CLI (shared by both modes)
tests/test_pipeline.py         pytest suite, run in CI
.github/workflows/
  ci.yml                        local mode: tests + gate, every push
  drift_check.yml               local mode: scheduled drift check
  deploy-aws-serverless.yml     AWS mode: build, push, redeploy on push to aws-serverless
  drift-check.yml               AWS mode: scheduled drift check against DynamoDB
  retrain-and-promote.yml       AWS mode: manual retrain + gate + live reload
Dockerfile                     serving API container (local / HF Spaces / Fly.io / Railway)
Dockerfile.lambda              Lambda-compatible container image for AWS mode
infra/aws-serverless/          AWS-specific build context and deploy assets
reference_rehearsal_scripts/   the original phase1-4 standalone practice scripts
                                (kept for STAR-story mining — see phase4_system_design/)
```

---

## What "done" looks like

You should be able to: run `scripts/run_pipeline.sh` cold and get a served model;
kill a gate on purpose (feed it bad data) and watch it block with a real alert;
retrain, watch the champion-challenger gate actually reject a worse model; flip on
a canary and confirm the same customer always lands on the same version; pull up
`feature_store.sqlite` in any SQLite browser and see real logged predictions —
**and**, on the AWS side, push to `aws-serverless` and watch a real container
build, push to ECR, and redeploy a live Lambda; trigger a manual retrain workflow
and watch it gate a new candidate against production; and query the DynamoDB
inference log table directly and see the same shape of logged predictions as the
local SQLite store. That's the operational-fluency bar the roadmap is actually
asking for — twice, once cheap and once durable.
