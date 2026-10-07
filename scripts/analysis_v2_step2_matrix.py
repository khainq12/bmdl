"""STEP 2 (analysis_v2): attack x defense matrix from the CORRECTED
PathMNIST sweep only (results/pathmnist/benchmark_v2_corrected/). Does not
touch the preserved invalid v1 sweep and does not rerun anything.

Reports, mean +- std across the 3 seeds, for every (partition, attack,
defense, calibration_regime):
  - final test accuracy, test loss
  - accuracy degradation vs no_attack (same partition/defense/regime)
  - divergence rate (fraction of seeds diverged by round 24)
  - detector-only: TPR, FPR, accepted-malicious count, rejected-honest
    count, pooled over active-attack rounds (or all rounds for no_attack)
  - mean per-round runtime, total runtime (non-diverged rows only)

Usage: python scripts/analysis_v2_step2_matrix.py
Writes results/pathmnist/analysis_v2/attack_defense_matrix.csv and
results/pathmnist/analysis_v2/attack_defense_matrix.md
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

IN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "pathmnist", "benchmark_v2_corrected")
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "pathmnist", "analysis_v2")
N_ROUNDS = 25
ATTACK_FROM_ROUND = 10
PARTITION_ORDER = ["iid", "dirichlet_a1.0", "dirichlet_a0.5", "dirichlet_a0.1"]
ATTACK_ORDER = ["no_attack", "large_norm", "low_norm", "full_sign_flip", "directional_poisoning", "sparse_coordinate_attack"]
DEFENSE_ORDER = ["fedavg", "norm", "cosine", "sign_consensus", "median", "multi_krum"]
DETECTOR_DEFENSES = {"norm", "cosine", "sign_consensus"}


def per_seed_tpr_fpr(g: pd.DataFrame) -> pd.Series:
    tp, fp, tn, fn = g["tp"].sum(), g["fp"].sum(), g["tn"].sum(), g["fn"].sum()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    return pd.Series({"tp": tp, "fp": fp, "tn": tn, "fn": fn, "tpr": tpr, "fpr": fpr})


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(os.path.join(IN_DIR, "per_round_results.csv"))

    final_round = df[df["round"] == N_ROUNDS - 1].copy()
    group_cols = ["partition", "attack", "defense", "calibration_regime"]

    # --- utility (per-seed, then mean+-std) ---
    util = (
        final_round.groupby(group_cols)["accuracy"]
        .agg(acc_mean="mean", acc_std="std", n_seeds="count")
        .reset_index()
    )
    loss_stats = final_round.groupby(group_cols)["loss"].agg(loss_mean="mean", loss_std="std").reset_index()
    util = util.merge(loss_stats, on=group_cols)

    div = final_round.groupby(group_cols)["diverged"].mean().reset_index().rename(columns={"diverged": "diverged_fraction"})
    util = util.merge(div, on=group_cols)

    # --- robustness: degradation vs no_attack, same partition/defense/regime ---
    baseline = util[util["attack"] == "no_attack"][["partition", "defense", "calibration_regime", "acc_mean"]]
    baseline = baseline.rename(columns={"acc_mean": "no_attack_acc_mean"})
    util = util.merge(baseline, on=["partition", "defense", "calibration_regime"], how="left")
    util["accuracy_degradation"] = util["no_attack_acc_mean"] - util["acc_mean"]

    # --- detector TPR/FPR: per-seed then mean+-std, pooled counts too ---
    det = df[df["defense"].isin(DETECTOR_DEFENSES)].copy()
    det_active = det[(det["attack"] != "no_attack") & (det["round"] >= ATTACK_FROM_ROUND)]
    det_clean = det[det["attack"] == "no_attack"]
    det_pool_rows = pd.concat([det_active, det_clean], ignore_index=True)

    per_seed = det_pool_rows.groupby(group_cols + ["seed"]).apply(per_seed_tpr_fpr, include_groups=False).reset_index()
    det_summary = (
        per_seed.groupby(group_cols)
        .agg(
            tpr_mean=("tpr", "mean"), tpr_std=("tpr", "std"),
            fpr_mean=("fpr", "mean"), fpr_std=("fpr", "std"),
            accepted_malicious_total=("fn", "sum"),  # fn = malicious accepted (false negative = miss)
            rejected_malicious_total=("tp", "sum"),
            rejected_honest_total=("fp", "sum"),
            accepted_honest_total=("tn", "sum"),
        )
        .reset_index()
    )

    # --- runtime: non-diverged rows only, mean+-std per-round, and total (sum of actual compute, non-diverged rows) ---
    rt_rows = df[~df["diverged"]]
    rt_per_seed = rt_rows.groupby(group_cols + ["seed"])["runtime_s"].agg(mean_round="mean", total="sum").reset_index()
    rt_summary = (
        rt_per_seed.groupby(group_cols)
        .agg(
            mean_round_runtime_s=("mean_round", "mean"),
            std_round_runtime_s=("mean_round", "std"),
            mean_total_runtime_s=("total", "mean"),
        )
        .reset_index()
    )

    # --- merge everything ---
    full = util.merge(det_summary, on=group_cols, how="left").merge(rt_summary, on=group_cols, how="left")
    full["partition"] = pd.Categorical(full["partition"], categories=PARTITION_ORDER, ordered=True)
    full["attack"] = pd.Categorical(full["attack"], categories=ATTACK_ORDER, ordered=True)
    full["defense"] = pd.Categorical(full["defense"], categories=DEFENSE_ORDER, ordered=True)
    full = full.sort_values(["partition", "calibration_regime", "attack", "defense"])
    full.to_csv(os.path.join(OUT_DIR, "attack_defense_matrix.csv"), index=False)

    # --- compact markdown matrices: accuracy and divergence, per partition, Regime A (+ not_applicable) ---
    lines = ["# Attack x Defense matrix (compact view)\n"]
    lines.append(
        "Full data with all metrics (mean+-std, TPR/FPR, runtime) in "
        "`attack_defense_matrix.csv`. Compact tables below show final accuracy "
        "and divergence rate; Regime A (condition-specific) and `not_applicable` "
        "(non-detector) rows shown per partition; Regime B shown separately.\n"
    )

    def render_matrix(sub: pd.DataFrame, title: str, value_col: str, fmt: str) -> str:
        pivot = sub.pivot_table(index="attack", columns="defense", values=value_col, observed=True)
        pivot = pivot.reindex(index=ATTACK_ORDER, columns=DEFENSE_ORDER)
        out = [f"### {title}\n"]
        out.append("| attack | " + " | ".join(DEFENSE_ORDER) + " |")
        out.append("|---|" + "---|" * len(DEFENSE_ORDER))
        for atk in ATTACK_ORDER:
            row = [atk]
            for d in DEFENSE_ORDER:
                v = pivot.loc[atk, d] if (atk in pivot.index and d in pivot.columns) else np.nan
                row.append(fmt.format(v) if pd.notna(v) else "-")
            out.append("| " + " | ".join(row) + " |")
        return "\n".join(out) + "\n"

    for p in PARTITION_ORDER:
        sub = full[(full["partition"] == p) & (full["calibration_regime"].isin(["A_condition_specific_oracle", "not_applicable"]))]
        lines.append(f"\n## Partition: {p} (Regime A for detectors)\n")
        lines.append(render_matrix(sub, f"Final accuracy (mean over 3 seeds) — {p}", "acc_mean", "{:.3f}"))
        lines.append(render_matrix(sub, f"Divergence fraction — {p}", "diverged_fraction", "{:.2f}"))

    lines.append("\n## Regime B (IID-calibrated, frozen) — detectors only, non-IID partitions\n")
    for p in ["dirichlet_a1.0", "dirichlet_a0.5", "dirichlet_a0.1"]:
        sub = full[(full["partition"] == p) & (full["calibration_regime"] == "B_iid_transfer_frozen")]
        lines.append(f"\n### {p}\n")
        lines.append(render_matrix(sub, f"Final accuracy, Regime B — {p}", "acc_mean", "{:.3f}"))

    with open(os.path.join(OUT_DIR, "attack_defense_matrix.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Saved {OUT_DIR}/attack_defense_matrix.csv and .md")
    print(f"rows in full matrix: {len(full)}")


if __name__ == "__main__":
    main()
