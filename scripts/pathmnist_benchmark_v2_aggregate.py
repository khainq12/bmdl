"""Aggregate results/pathmnist/benchmark_v2_corrected/per_round_results.csv
into utility / robustness / detector-TPR-FPR / runtime / divergence tables.
Keeps calibration_regime strictly separate throughout (Addendum (f)), and
reports divergence rate as its own column rather than silently averaging
diverged (model-destroyed) rows into ordinary utility numbers (Addendum (i)).

Usage: python scripts/pathmnist_benchmark_v2_aggregate.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "pathmnist", "benchmark_v2_corrected")
N_ROUNDS = 25
ATTACK_FROM_ROUND = 10
PARTITION_ORDER = ["iid", "dirichlet_a1.0", "dirichlet_a0.5", "dirichlet_a0.1"]
DETECTOR_DEFENSES = {"norm", "cosine", "sign_consensus"}


def main() -> None:
    df = pd.read_csv(os.path.join(OUT_DIR, "per_round_results.csv"))
    calib = pd.read_csv(os.path.join(OUT_DIR, "calibration_results.csv"))
    group_cols = ["partition", "attack", "defense", "calibration_regime"]

    # ---------- Divergence: fraction of seeds where this combo was destroyed by round 24 ----------
    final_round = df[df["round"] == N_ROUNDS - 1]
    divergence = (
        final_round.groupby(group_cols)["diverged"]
        .agg(["mean", "sum", "count"])
        .reset_index()
        .rename(columns={"mean": "diverged_fraction_of_seeds", "sum": "n_diverged_seeds", "count": "n_seeds"})
    )
    divergence.to_csv(os.path.join(OUT_DIR, "divergence_table.csv"), index=False)

    # ---------- Utility: final-round accuracy, mean+-std across seeds (diverged included, flagged) ----------
    util = (
        final_round.groupby(group_cols)["accuracy"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "final_acc_mean", "std": "final_acc_std", "count": "n_seeds"})
    )
    util = util.merge(divergence[group_cols + ["diverged_fraction_of_seeds"]], on=group_cols, how="left")
    util.to_csv(os.path.join(OUT_DIR, "utility_table.csv"), index=False)

    # ---------- Robustness: degradation vs no_attack, same (partition,defense,regime) ----------
    baseline = util[util["attack"] == "no_attack"][["partition", "defense", "calibration_regime", "final_acc_mean"]]
    baseline = baseline.rename(columns={"final_acc_mean": "no_attack_acc_mean"})
    rob = util.merge(baseline, on=["partition", "defense", "calibration_regime"], how="left")
    rob["accuracy_drop"] = rob["no_attack_acc_mean"] - rob["final_acc_mean"]
    rob.to_csv(os.path.join(OUT_DIR, "robustness_table.csv"), index=False)

    # ---------- Detector TPR/FPR: pooled over active-attack rounds (or all rounds for no_attack) ----------
    det = df[df["defense"].isin(DETECTOR_DEFENSES)].copy()
    det_active = det[(det["attack"] != "no_attack") & (det["round"] >= ATTACK_FROM_ROUND)]
    det_clean = det[det["attack"] == "no_attack"]
    det_pool = pd.concat([det_active, det_clean], ignore_index=True)

    def _tpr_fpr(g: pd.DataFrame) -> pd.Series:
        tp, fp, tn, fn = g["tp"].sum(), g["fp"].sum(), g["tn"].sum(), g["fn"].sum()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        return pd.Series({"tp": tp, "fp": fp, "tn": tn, "fn": fn, "tpr": tpr, "fpr": fpr})

    tpr_fpr = det_pool.groupby(group_cols).apply(_tpr_fpr, include_groups=False).reset_index()
    tpr_fpr.to_csv(os.path.join(OUT_DIR, "detector_tpr_fpr_table.csv"), index=False)

    # ---------- Runtime: mean per-round runtime (non-diverged rows only, so copy-forwarded rows don't skew it) ----------
    rt_rows = df[~df["diverged"]]
    rt = rt_rows.groupby(group_cols)["runtime_s"].agg(["mean", "std"]).reset_index().rename(
        columns={"mean": "mean_round_runtime_s", "std": "std_round_runtime_s"}
    )
    rt.to_csv(os.path.join(OUT_DIR, "runtime_table.csv"), index=False)

    # ---------- Console summary ----------
    pd.set_option("display.width", 240)
    pd.set_option("display.max_rows", 1000)
    print("=" * 110)
    print("CALIBRATION RESULTS")
    print("=" * 110)
    print(calib[["defense", "partition", "calibration_regime", "threshold", "validation_fpr", "validation_tpr", "used_fallback"]].to_string(index=False))

    print("\n" + "=" * 110)
    print("DIVERGENCE (fraction of seeds where the model was destroyed by round 24)")
    print("=" * 110)
    div_nonzero = divergence[divergence["diverged_fraction_of_seeds"] > 0].sort_values("diverged_fraction_of_seeds", ascending=False)
    print(div_nonzero.to_string(index=False) if len(div_nonzero) else "(none)")

    print("\n" + "=" * 110)
    print("UTILITY: final-round accuracy mean (n=3 seeds), by partition x attack x defense x regime")
    print("=" * 110)
    piv = util.copy()
    piv["partition"] = pd.Categorical(piv["partition"], categories=PARTITION_ORDER, ordered=True)
    for attack in sorted(piv["attack"].unique()):
        print(f"\n--- attack={attack} ---")
        sub = piv[piv["attack"] == attack].sort_values(["defense", "calibration_regime", "partition"])
        print(sub[["partition", "defense", "calibration_regime", "final_acc_mean", "final_acc_std", "diverged_fraction_of_seeds"]].to_string(index=False))

    print("\n" + "=" * 110)
    print("ROBUSTNESS: accuracy drop vs no_attack (same partition/defense/regime)")
    print("=" * 110)
    rob_sorted = rob[rob["attack"] != "no_attack"].sort_values(["defense", "calibration_regime", "attack", "partition"])
    print(rob_sorted[["partition", "attack", "defense", "calibration_regime", "final_acc_mean", "no_attack_acc_mean", "accuracy_drop", "diverged_fraction_of_seeds"]].to_string(index=False))

    print("\n" + "=" * 110)
    print("DETECTOR TPR/FPR (pooled active-attack rounds, or all rounds for no_attack)")
    print("=" * 110)
    print(tpr_fpr.sort_values(["defense", "calibration_regime", "attack", "partition"]).to_string(index=False))

    print("\n" + "=" * 110)
    print("RUNTIME: mean per-round runtime (s), non-diverged rows only")
    print("=" * 110)
    print(rt.groupby(["defense", "calibration_regime"])["mean_round_runtime_s"].mean().to_string())

    print(f"\nSaved tables to {OUT_DIR}")


if __name__ == "__main__":
    main()
