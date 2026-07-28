# System Design Answer Template

Use this exact 5-step skeleton for any "design an ML system for X" prompt. Fill in
the bracketed parts using what you actually built in this project — you now have real
numbers and real component names to plug in instead of generic ones.

---

## Prompt example: "Design a fraud / churn / escalation detection system"

### 1. Clarify the business objective and constraints
- What decision does this feed? [e.g., route customers into auto-resolve / review / escalate]
- Latency budget: [real-time inference at request time, or batch scoring overnight?]
- Request volume: [rough QPS or daily volume]
- Label availability: [how long is the delay between prediction and ground truth? — mine: 21 days]

### 2. Define the metric and how it maps to the loss function and threshold
- Business metric: [e.g., minimize (cost of missed escalations) + (cost of wasted senior review time)]
- Why not accuracy: [state the imbalance — my positive rate was ~3%, so accuracy is meaningless]
- Metric used instead: [PR-AUC / F-beta / the actual cost-matrix total]
- Threshold approach: [calibrate first (Platt/isotonic) → build ROC/PR on a held-out fold →
  Youden's J as a default, cost-matrix-optimal threshold as the real answer]
- If tiered routing is relevant: [two independently-optimized thresholds, state both cost matrices]

### 3. Sketch the data flow: ingestion → feature store → training → serving
- Ingestion gate: [tool + specific check + threshold, e.g. Great Expectations, 0% null
  tolerance on entity_id, DLQ on failure]
- Transformation gate: [dbt-style uniqueness/referential integrity tests, Airflow block + PagerDuty on failure]
- Feature store: [online = low-latency KV overwrite (Redis), offline = append-only timestamped
  Parquet, point-in-time joins to prevent leakage — quote your own AUC-collapse number here]
- Training: [baseline hierarchy — heuristic → linear (L1/L2) → tree ensemble, only escalate to DL if X]
- Serving gate: [KS test hourly against training baseline, p<0.05 triggers retraining]

### 4. Call out failure modes and monitoring
- Data drift: [KS test on inputs, doesn't need labels]
- Concept drift: [only visible via degrading live precision/recall once labels arrive — distinguish explicitly]
- Label delay bridging: [PSI on output scores, human-in-the-loop fast sample, automated backfill job]
- Rollout safety: [canary 10→25→50→100, rollback trigger = error rate > 0.1% in a 1hr window —
  quote your own simulation's rollback event here]
- Online/offline skew: [periodic reconciliation job sampling entity/timestamp pairs, alert on divergence]

### 5. Discuss scaling/cost trade-offs (briefly, at the end)
- [e.g., senior-escalation capacity is the real bottleneck, not model latency — the t_high
  threshold is really a capacity-allocation decision disguised as a modeling decision]
- [Redis online store cost vs. read latency trade-off if QPS scales 10x]

---

## Second worked example (do this one yourself before the interview)

Pick a different prompt — e.g. "design a recommendation system" or "design a
content-moderation pipeline" — and fill out the same 5 steps from scratch, reusing
the same component vocabulary (gates, cascades, canary, point-in-time joins) so it's
clear this is a transferable mental model, not a memorized answer about one project.
