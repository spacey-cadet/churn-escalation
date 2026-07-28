"""
Canary Rollout Simulation: 10% -> 25% -> 50% -> 100%, with a real rollback trigger.

Run this TWICE:
    1. INJECT_BAD_MODEL = False -> clean rollout, watch it climb through all four stages.
    2. INJECT_BAD_MODEL = True  -> the canary's error rate breaches 0.1% partway through,
       and the simulation automatically rolls back to the stable baseline.

This is your real "a rollout you had to roll back" STAR story -- state the exact
trigger, how fast the system reacted, and that the safety mechanism worked as designed.

Run: python phase3_mlops/canary_rollout_simulation.py
"""
import numpy as np

RNG = np.random.default_rng(99)

# <<< TOGGLE THIS AND RE-RUN >>>
INJECT_BAD_MODEL = True

ERROR_RATE_THRESHOLD = 0.001   # 0.1%
WINDOW_REQUESTS = 5000         # requests observed per "1-hour window"
TRAFFIC_STAGES = [10, 25, 50, 100]


def simulate_stage(traffic_pct, baseline_error_rate, canary_error_rate):
    """Simulate WINDOW_REQUESTS at this traffic split and measure the canary's
    observed error rate over its share of traffic."""
    canary_requests = int(WINDOW_REQUESTS * (traffic_pct / 100))
    if canary_requests == 0:
        return 0.0, 0
    errors = RNG.binomial(canary_requests, canary_error_rate)
    observed_rate = errors / canary_requests
    return observed_rate, canary_requests


def run_rollout(inject_bad_model: bool):
    baseline_error_rate = 0.0003  # stable baseline's known error rate
    canary_error_rate = 0.0008 if not inject_bad_model else 0.0035  # bad model breaches threshold

    print(f"Stable baseline error rate: {baseline_error_rate:.4%}")
    print(f"Canary true error rate:     {canary_error_rate:.4%}  "
          f"({'INJECTED BAD MODEL' if inject_bad_model else 'healthy model'})\n")
    print(f"Rollback trigger: observed error rate > {ERROR_RATE_THRESHOLD:.4%} in a monitoring window\n")

    for stage_pct in TRAFFIC_STAGES:
        observed_rate, n_requests = simulate_stage(stage_pct, baseline_error_rate, canary_error_rate)
        status = "OK" if observed_rate <= ERROR_RATE_THRESHOLD else "BREACH"
        print(f"Stage {stage_pct:>3}% traffic  |  {n_requests} requests observed  |  "
              f"error rate {observed_rate:.4%}  |  {status}")

        if observed_rate > ERROR_RATE_THRESHOLD:
            print(f"\n>>> SLA BREACH at {stage_pct}% traffic: observed error rate "
                  f"{observed_rate:.4%} > threshold {ERROR_RATE_THRESHOLD:.4%}")
            print(">>> INGRESS CONTROLLER: instant rollback triggered.")
            print(">>> Traffic weights reset to 100% stable baseline / 0% canary.")
            print(">>> Rollback completed. No further stage progression.")
            return False

        print(f"    Canary holds SLA over this window -> proceeding to next stage.\n")

    print("\nRollout completed: canary promoted to 100% and is now the new stable baseline.")
    return True


if __name__ == "__main__":
    run_rollout(INJECT_BAD_MODEL)
