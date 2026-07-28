"""
Run the real champion-challenger gate against the most recently registered (but not
yet promoted) version, and promote it to production if -- and only if -- it clears
the bar on every eval slice.

This is the thing standing between a bad retrain and a bad user experience. Run it
every time after src/modeling/train.py, never promote by hand.

Usage:
    python scripts/evaluate_and_promote.py                 # evaluate + promote if it passes
    python scripts/evaluate_and_promote.py --version v...  # evaluate a specific version
    python scripts/evaluate_and_promote.py --dry-run       # evaluate only, never promote
"""
import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import pandas as pd
from sklearn.model_selection import train_test_split

import config
import registry
import alerting


def build_eval_sets():
    """Two slices, on purpose: the standard held-out fold, plus a harder "high
    ticket volume" slice standing in for the roadmap's "cross-corpus" check --
    a challenger has to hold up on both, not just the easy average case."""
    df = pd.read_parquet(config.FEATURES_CLEAN)
    _, holdout = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=42)

    held_out = (holdout[config.FEATURES], holdout["label"])

    high_volume_slice = holdout[holdout["support_tickets_30d"] >= holdout["support_tickets_30d"].quantile(0.75)]
    stress_slice = (high_volume_slice[config.FEATURES], high_volume_slice["label"])

    return {"held_out": held_out, "high_ticket_volume_slice": stress_slice}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default=None, help="Version to evaluate (default: most recently registered)")
    parser.add_argument("--dry-run", action="store_true", help="Evaluate only, never promote")
    args = parser.parse_args()

    versions = registry.list_versions()
    if not versions:
        print("No registered versions found. Run `python -m src.modeling.train` first.")
        sys.exit(1)

    challenger_version = args.version or versions[-1]["version"]
    print(f"Evaluating challenger: {challenger_version}")

    eval_sets = build_eval_sets()
    gate_result = registry.champion_challenger_gate(challenger_version, eval_sets)

    print(f"Current production (champion): {gate_result['champion_version']}")
    for set_name, r in gate_result["per_set_results"].items():
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{status}] {set_name}: challenger PR-AUC={r['challenger']['pr_auc']:.4f}"
              + (f"  champion PR-AUC={r['champion']['pr_auc']:.4f}  regression={r['regression']:+.4f}"
                 if r["champion"] else "  (no champion yet)"))

    if gate_result["passed"]:
        print(f"\nGate PASSED.")
        if args.dry_run:
            print("(--dry-run set: not promoting)")
        else:
            registry.promote_to_production(challenger_version)
            alerting.send_alert(f"Promoted {challenger_version} to production (champion-challenger gate passed).")
            print(f"Promoted {challenger_version} to production.")
    else:
        print(f"\nGate FAILED -- {challenger_version} regresses PR-AUC beyond "
              f"{config.MAX_ALLOWED_PR_AUC_REGRESSION} on at least one eval slice.")
        registry.promote_to_staging(challenger_version)
        alerting.send_alert(
            f"Champion-challenger gate FAILED for {challenger_version} -- staged, not promoted.",
            severity="critical",
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
