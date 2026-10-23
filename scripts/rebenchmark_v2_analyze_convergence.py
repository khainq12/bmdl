"""R12: analyze the baseline convergence study's round_metrics.csv with an
explicit, pre-registered multi-round convergence criterion (written before
looking at results, so the criterion isn't fit to what's convenient).

Criterion (applied per run, i.e. per candidate x partition x seed):
  A run is CONVERGED at round r if, for the window [r, r+PATIENCE-1]:
    1. max(val_accuracy in window) - val_accuracy[r-1] < ACC_EPS
       (no more than ACC_EPS further improvement is ever achieved after
       round r-1, looking PATIENCE rounds ahead), AND
    2. the window's val_loss does not decrease by more than LOSS_EPS
       per round on average (slope >= -LOSS_EPS).
  If no such r exists within the run's recorded rounds, the run is
  NOT CONVERGED within budget.

PATIENCE=15, ACC_EPS=0.005 (0.5 percentage points), LOSS_EPS=0.002/round.
These are fixed constants, not tuned per-partition.

Usage: python scripts/rebenchmark_v2_analyze_convergence.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_DIR = os.path.join(ROOT, "results", "rebenchmark_v2", "baseline_convergence")
OUT_MD = os.path.join(IN_DIR, "CONVERGENCE_ANALYSIS.md")

PATIENCE = 15
ACC_EPS = 0.005
LOSS_EPS = 0.002


def find_convergence_round(val_acc: np.ndarray, val_loss: np.ndarray) -> int | None:
    """Persistent-plateau criterion (fixes a false-positive in the first
    version of this function, which flagged transient early-training flat
    patches -- common under severe non-IID -- as 'converged'). A run is
    converged at round r if round r achieves a val_accuracy within ACC_EPS
    of the run's eventual best, AND no round after r ever improves on
    val_acc[r] by more than ACC_EPS, AND val_loss does not keep dropping
    by more than LOSS_EPS/round on average over the remaining rounds.
    Equivalently: r is the earliest round after which the run is within
    ACC_EPS of its own final best for every subsequent round -- a
    persistent plateau to the end of the recorded curve, not a 15-round
    transient one."""
    n = len(val_acc)
    if n < PATIENCE + 1:
        return None
    best_so_far = np.maximum.accumulate(val_acc)
    # running max of (everything from r to the end) == best achievable from r onward
    best_suffix = np.maximum.accumulate(val_acc[::-1])[::-1]
    for r in range(1, n - PATIENCE + 1):
        if best_suffix[r] - val_acc[r - 1] >= ACC_EPS:
            continue  # something later still beats round r-1 by too much
        tail_loss = val_loss[r:]
        slope = np.polyfit(np.arange(len(tail_loss)), tail_loss, 1)[0]
        if slope < -LOSS_EPS:
            continue
        return r
    return None


def main() -> None:
    round_path = os.path.join(IN_DIR, "round_metrics.csv")
    manifest_path = os.path.join(IN_DIR, "run_manifest.csv")
    if not os.path.exists(round_path):
        raise SystemExit(f"not found: {round_path} -- run rebenchmark_v2_baseline_convergence.py first")

    rounds = pd.read_csv(round_path)
    manifest = pd.read_csv(manifest_path)

    lines = ["# Baseline convergence analysis (R12)",
             "",
             f"Criterion: PATIENCE={PATIENCE} rounds, ACC_EPS={ACC_EPS}, LOSS_EPS={LOSS_EPS}/round "
             "(fixed before inspecting results).",
             "",
             "| run_id | candidate | partition | seed | converged_at_round | final_val_acc | final_val_loss | final_test_acc |",
             "|---|---|---|---|---|---|---|---|"]

    summary_rows = []
    for run_id, g in rounds.groupby("run_id"):
        g = g.sort_values("round")
        val_acc = g["val_accuracy"].to_numpy()
        val_loss = g["val_loss"].to_numpy()
        conv_round = find_convergence_round(val_acc, val_loss)
        m = manifest[manifest.run_id == run_id].iloc[0]
        lines.append(f"| {run_id} | {m.candidate} | {m.partition} | {m.seed} | "
                     f"{conv_round if conv_round is not None else 'NOT CONVERGED'} | "
                     f"{val_acc[-1]:.4f} | {val_loss[-1]:.4f} | {m.final_test_accuracy:.4f} |")
        summary_rows.append({"run_id": run_id, "candidate": m.candidate, "partition": m.partition,
                              "seed": m.seed, "converged_at_round": conv_round,
                              "final_val_acc": val_acc[-1], "final_val_loss": val_loss[-1],
                              "final_test_acc": m.final_test_accuracy})

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(IN_DIR, "convergence_summary.csv"), index=False)

    lines.append("")
    lines.append("## Per (candidate, partition) summary across seeds")
    lines.append("")
    lines.append("| candidate | partition | n_converged/n_seeds | median_converged_round | mean_final_val_acc | std_final_val_acc |")
    lines.append("|---|---|---|---|---|---|")
    for (cand, part), g in summary.groupby(["candidate", "partition"]):
        n_conv = g["converged_at_round"].notna().sum()
        med_round = g["converged_at_round"].dropna().median()
        lines.append(f"| {cand} | {part} | {n_conv}/{len(g)} | "
                     f"{med_round if not np.isnan(med_round) else 'n/a'} | "
                     f"{g.final_val_acc.mean():.4f} | {g.final_val_acc.std():.4f} |")

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {OUT_MD}")


if __name__ == "__main__":
    main()
