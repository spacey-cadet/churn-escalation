# 0001: Local free-tier stack vs. AWS serverless

This repo was originally built to run entirely on local files, SQLite, and free
hosted tiers (see `PIPELINE_README.md`'s own free-tier substitution table). This
document covers the second swap: local -> AWS, for a target budget of well under
$5/month.

| Local component | AWS component | What changed | What's genuinely NOT equivalent |
|---|---|---|---|
| SQLite `online_features` table | DynamoDB on-demand table, PK `entity_id` | `src/feature_store.py` rewritten to use boto3; same function signatures | None at this traffic level |
| SQLite `offline_features` table | DynamoDB on-demand table, PK `entity_id` + SK `ts` | Point-in-time join becomes a native `Query` instead of a SQL `ORDER BY ... LIMIT` | None -- if anything this is a cleaner primitive than the SQL version |
| SQLite `inference_log` table | DynamoDB table, single fixed partition key (`"LOG"`) + time-ordered sort key | `read_inference_log`'s "most recent N" becomes a `Query` on one partition | **Real ceiling**: a single DynamoDB partition tops out around 1000 WCU / 3000 RCU. Fine at hobby/portfolio volume; would need a sharded key (e.g. `LOG#<hour>`) under sustained high write throughput. Not hidden, just not needed yet. |
| Local `registry/` directory + JSON pointer files | S3 bucket, `versions/<id>/...` + `pointers/*.json` | `src/registry.py` rewritten to use boto3; same function signatures | S3 has no server-side "read-modify-write JSON" primitive -- `_set_stage()` does two full round trips instead of an in-place edit. Irrelevant at this write volume. |
| Local container host (HF Spaces/Fly/Railway) running `Dockerfile` | Lambda (container image) + Function URL | New `Dockerfile.lambda`, Mangum wrapper appended to `src/serving/app.py` | Lambda cold starts (~1-2s with XGBoost+pandas import) are slower than an always-warm container. Acceptable for a low-traffic API; would need provisioned concurrency (real cost) to avoid entirely. |
| `.github/workflows/ci.yml` running gates/tests locally-equivalent | Same, but `retrain-and-promote.yml` and `drift_check.yml` now carry AWS credentials via OIDC | Ingestion, training, and both quality gates stay in GitHub Actions -- there is no reason to pay for compute to run something that already runs free on a human retrain cadence | None |
| In-process canary (`CANARY_PCT` + SHA256(entity_id) hash) | **Unchanged** | Champion and challenger models both load into the same Lambda's memory from S3; no traffic-splitting infra needed | None |
| Webhook alerting (`src/alerting.py`) | **Unchanged** | Called from wherever the check runs (GitHub Actions or Lambda) | None |

## What stayed local/GitHub Actions on purpose

Ingestion, the two quality gates, training, and the champion-challenger gate all
run in GitHub Actions, not Lambda. They're periodic and human-triggered by
design (see `PIPELINE_README.md`), so there's no latency or availability reason
to move them into AWS -- doing so would only add cost (Step Functions,
additional Lambdas) for no benefit. The drift monitor stays on its existing
hourly GitHub Actions cron for the same reason; only its data source changed
(DynamoDB instead of a local SQLite file).

## Cost estimate

At hobby/portfolio traffic (well under free-tier request/throughput ceilings on
Lambda and DynamoDB on-demand), the dominant and most easily-overlooked line
item is CloudWatch Logs storage, which is why `main.tf` sets a 14-day retention
explicitly rather than leaving the default (never expire). Realistic total:
**$1-3/month**, comfortably under the $5/month target and inside the $15/month
three-project ceiling covering Project 1 and Project 3 as well.
