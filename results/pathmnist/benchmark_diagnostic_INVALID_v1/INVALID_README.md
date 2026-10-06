# THIS SWEEP IS INVALID FOR QUANTITATIVE CONCLUSIONS — PRESERVED AS A DIAGNOSTIC

This directory is the **first** full PathMNIST single-method sweep (594
combos, 151.4 minutes), preserved **unmodified** as a diagnostic artifact.
**Do not use its numbers as research findings.** It is kept, not deleted or
overwritten, because the run itself is what surfaced the bug described
below — see `REPORT.md` in this directory for the full original write-up.

## Why it's invalid

1. **Norm defense: invalid everywhere.** `local_training_seeded` ran one
   full local epoch over however much data each client had.
   Calibration pseudo-clients (val split ÷ 5 ≈ 2,000 samples/client) got
   **8 batches/epoch**; real benchmark clients (train pool ÷ 5) got
   **64 batches/epoch at IID** and **30–130 batches/epoch within a single
   Dirichlet(α=0.1) round** (partition-size imbalance alone). Gradient
   -delta magnitude scales with step count, so Norm thresholds calibrated
   on 8-batch pseudo-clients rejected essentially every real client
   (FPR 96–100%, confirmed in `detector_tpr_fpr_table.csv`), including at
   IID where calibration and test conditions nominally matched.
2. **Absolute accuracy/robustness: invalid everywhere.** 6 rounds was too
   short — clean FedAvg reached only 0.12–0.18 (chance = 0.111) vs. 50.7%
   at STEP 5's 15 rounds.
3. Cosine/Sign Consensus TPR/FPR numbers are directionally suggestive only
   (both are less magnitude-sensitive than Norm, but calibration still used
   the same mismatched step counts, which can shift direction too, not just
   scale).

## What's still true from this run

- Four-way partition/seed/attacker-assignment identity across defenses —
  confirmed by construction, unaffected by the bug.
- Runtime numbers — unaffected by the bug.
- **The core qualitative finding survives**: Regime B (IID-calibrated,
  frozen) thresholds reject the overwhelming majority of honest clients
  under every tested Non-IID condition, even with zero attacker present —
  this is threshold-position-driven, not magnitude-driven, so the
  step-count bug shifts it but did not invent it. It's consistent with the
  honest-gradient characterization study's independent prediction.

## What superseded this

`docs/BENCHMARK_PROTOCOL.md` Addendum 2026-10-07 (g) locks the fix: a
fixed number `K` of local SGD steps per client per round, independent of
dataset size, applied identically to calibration pseudo-clients, IID
clients, Dirichlet clients, and the `g_ref` computation. The corrected
sweep is in `results/pathmnist/benchmark_v2_corrected/` (once produced).
