# STAR Story Templates

Fill these in using what actually happened when you ran the scripts — not hypotheticals.
Each one points at exactly which script to mine for real numbers.

---

## 1. A time you chose a simpler baseline over a complex model
**Mine this from:** `phase1_modeling/baseline_hierarchy.py` output

- **Situation:** [what was the tabular problem]
- **Task:** [what accuracy/business bar did it need to clear]
- **Action:** [you ran heuristic → logistic (L1/L2) → XGBoost and looked at the actual PR-AUC gap]
- **Result:** [quote your real numbers: "Heuristic -> Linear gap: +X, Linear -> XGBoost gap: +Y" —
  and state explicitly why deep learning was correctly ruled out]

---

## 2. A production incident caused by data drift or a pipeline failure
**Mine this from:** `phase2_pipeline/drift_detection.py` and `great_expectations/run_expectations.py` output

- **Situation:** [pick one: the injected null entity_ids at the ingestion gate, OR the
  simulated message-length drift at the serving gate]
- **Task:** [what needed to happen before bad data reached training/serving]
- **Action:** [which gate caught it, which specific check fired — e.g. "0% null tolerance
  on entity_id fired, 25 rows routed to DLQ" or "KS test p=0.000012, well under 0.05"]
- **Result:** [what happened next — DLQ quarantine + clean rows proceeded, or retraining
  loop triggered via MLflow flag]
- **What you'd change:** [a tightened threshold, a schema contract test, an earlier check]

---

## 3. A rollout you had to roll back
**Mine this from:** `phase3_mlops/canary_rollout_simulation.py` with `INJECT_BAD_MODEL = True`

- **Situation:** [canary at some traffic %, with a specific observed error rate]
- **Task:** [SLA: error rate must stay under 0.1% in a 1-hour window]
- **Action:** [state the exact trigger — "observed error rate X% > threshold 0.1% at N% traffic"]
- **Result:** [automatic rollback via ingress controller, reset to 100% stable baseline,
  zero manual intervention needed]
- **Reframe:** don't frame this as a failure — frame it as evidence the safety mechanism
  worked exactly as designed, catching a bad model before it reached significant traffic.

---

## 4. A disagreement with a stakeholder about a threshold/metric trade-off
**Mine this from:** `phase1_modeling/cascade_routing.py` output (the two independent cost matrices)

- **Situation:** [competing priorities — e.g. "product wants to minimize false positives
  to protect UX, support leadership wants to minimize false negatives to protect retention"]
- **Task:** [reconcile these into a single deployable threshold, or in your case, TWO thresholds]
- **Action:** [reframe the disagreement around your actual cost numbers — "FN cost=8x, FP cost=1x
  for auto-resolve boundary; FN cost=3x, FP cost=6x for senior-escalation boundary" —
  state why these are different because the two decisions have different stakes]
- **Result:** [state your actual t_low / t_high values and the recall captured above
  auto-resolve — this is what "won" the argument: a number, not an opinion]

---

## Delivery tip
Practice saying each Result out loud with the actual number from your run included.
"The gap was about 4 points of PR-AUC" is forgettable. "Linear to XGBoost bought us
0.0623 PR-AUC, which is why we still needed the ensemble" is what operational fluency
sounds like.
