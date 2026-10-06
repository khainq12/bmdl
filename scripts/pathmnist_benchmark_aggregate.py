"""Aggregate results/pathmnist/benchmark/per_round_results.csv into
utility / robustness / detector-TPR-FPR / runtime comparison tables,
keeping calibration_regime (A_condition_specific_oracle vs
B_iid_transfer_frozen vs not_applicable) strictly separate throughout, per
docs/BENCHMARK_PROTOCOL.md Addendum 2026-10-06 (f).

Usage: python scripts/pathmnist_benchmark_aggregate.py
Writes results/pathmnist/benchmark/{utility_table.csv, robustness_table.csv,
detector_tpr_fpr_table.csv, runtime_table.csv} and prints key comparisons.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "pathmnist", "benchmark")
N_ROUNDS = 6
ATTACK_FROM_ROUND = 2
PARTITION_ORDER = ["iid", "dirichlet_a1.0", "dirichlet_a0.5", "dirichlet_a0.1"]
DETECTOR_DEFENSES = {"norm", "cosine", "sign_consensus"}


def main() -> None:
    df = pd.read_csv(os.path.join(OUT_DIR, "per_round_results.csv"))
    calib = pd.read_csv(os.path.join(OUT_DIR, "calibration_results.csv"))

    group_cols = ["partition", "attack", "defense", "calibration_regime"]

    # ---------- Utility: final-round accuracy, mean+-std across seeds ----------
    final_round = df[df["round"] == N_ROUNDS - 1]
    util = (
        final_round.groupby(group_cols)["accuracy"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "final_acc_mean", "std": "final_acc_std", "count": "n_seeds"})
    )
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

    # ---------- Runtime: mean per-round runtime ----------
    rt = df.groupby(group_cols)["runtime_s"].agg(["mean", "std"]).reset_index().rename(
        columns={"mean": "mean_round_runtime_s", "std": "std_round_runtime_s"}
    )
    rt.to_csv(os.path.join(OUT_DIR, "runtime_table.csv"), index=False)

    # ---------- Console summary: utility x partition, split by regime, per defense ----------
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 500)
    print("=" * 100)
    print("CALIBRATION RESULTS (threshold, validation FPR/TPR)")
    print("=" * 100)
    print(calib[["defense", "partition", "calibration_regime", "threshold", "validation_fpr", "validation_tpr", "used_fallback"]].to_string(index=False))

    print("\n" + "=" * 100)
    print("UTILITY: final-round accuracy mean (n=3 seeds), by partition x attack x defense x regime")
    print("=" * 100)
    piv = util.copy()
    piv["partition"] = pd.Categorical(piv["partition"], categories=PARTITION_ORDER, ordered=True)
    for attack in sorted(piv["attack"].unique()):
        print(f"\n--- attack={attack} ---")
        sub = piv[piv["attack"] == attack].sort_values(["defense", "calibration_regime", "partition"])
        print(sub[["partition", "defense", "calibration_regime", "final_acc_mean", "final_acc_std"]].to_string(index=False))

    print("\n" + "=" * 100)
    print("ROBUSTNESS: accuracy drop vs no_attack (same partition/defense/regime)")
    print("=" * 100)
    rob_sorted = rob[rob["attack"] != "no_attack"].sort_values(["defense", "calibration_regime", "attack", "partition"])
    print(rob_sorted[["partition", "attack", "defense", "calibration_regime", "final_acc_mean", "no_attack_acc_mean", "accuracy_drop"]].to_string(index=False))

    print("\n" + "=" * 100)
    print("DETECTOR TPR/FPR (pooled active-attack rounds, or all rounds for no_attack)")
    print("=" * 100)
    print(tpr_fpr.sort_values(["defense", "calibration_regime", "attack", "partition"]).to_string(index=False))

    print("\n" + "=" * 100)
    print("RUNTIME: mean per-round runtime (s)")
    print("=" * 100)
    print(rt.groupby(["defense", "calibration_regime"])["mean_round_runtime_s"].mean().to_string())

    print(f"\nSaved tables to {OUT_DIR}")


if __name__ == "__main__":
    main()
