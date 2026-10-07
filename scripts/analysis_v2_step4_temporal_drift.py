"""STEP 4 (analysis_v2): temporal threshold drift. Uses Trace A (raw
per-client scores, 1 seed) for score-distribution-vs-round plots, and the
real corrected sweep's per_round_results.csv (3 seeds, pooled tp/fp/tn/fn)
for FPR/TPR-vs-round — combining both without rerunning anything.

Usage: python scripts/analysis_v2_step4_temporal_drift.py
Writes results/pathmnist/analysis_v2/figures/temporal_{defense}_{partition}.png
and results/pathmnist/analysis_v2/step4_temporal_drift_diagnosis.md
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANALYSIS_DIR = os.path.join(ROOT, "results", "pathmnist", "analysis_v2")
FIG_DIR = os.path.join(ANALYSIS_DIR, "figures")
REAL_DIR = os.path.join(ROOT, "results", "pathmnist", "benchmark_v2_corrected")

PARTITIONS = ["iid", "dirichlet_a1.0", "dirichlet_a0.5", "dirichlet_a0.1"]
DEFENSES = ["norm", "cosine", "sign_consensus"]
ATTACK_FROM_ROUND = 10
HIGHER_IS_HONEST = {"norm": False, "cosine": True, "sign_consensus": True}


def plot_defense_partition(trace: pd.DataFrame, real: pd.DataFrame, defense: str, partition: str, calib: pd.DataFrame, attack: str = "large_norm") -> dict:
    sub = trace[(trace["defense"] == defense) & (trace["partition"] == partition) & (trace["attack"] == attack)]
    threshold = float(
        calib[(calib["defense"] == defense) & (calib["partition"] == partition) & (calib["calibration_regime"] == "A_condition_specific_oracle")]["threshold"].iloc[0]
    )

    honest = sub[~sub["is_malicious"]]
    malicious = sub[sub["is_malicious"]]

    real_sub = real[(real["defense"] == defense) & (real["partition"] == partition) & (real["attack"] == attack) & (real["calibration_regime"] == "A_condition_specific_oracle")]
    fpr_by_round = real_sub.groupby("round").apply(lambda g: g["fp"].sum() / (g["fp"].sum() + g["tn"].sum()) if (g["fp"].sum() + g["tn"].sum()) > 0 else np.nan, include_groups=False)
    tpr_by_round = real_sub.groupby("round").apply(lambda g: g["tp"].sum() / (g["tp"].sum() + g["fn"].sum()) if (g["tp"].sum() + g["fn"].sum()) > 0 else np.nan, include_groups=False)

    has_huge_malicious = len(malicious) and malicious["score"].abs().max() > 50 * max(1e-6, honest["score"].abs().max())

    if has_huge_malicious:
        fig, axes = plt.subplots(3, 1, figsize=(8, 9.5), sharex=True)
        ax_mal, ax, ax2 = axes
        ax_mal.scatter(malicious["round"], malicious["score"], s=20, alpha=0.8, color="tab:red", marker="x", label="malicious (trace, seed=42)")
        ax_mal.set_yscale("symlog")
        ax_mal.set_ylabel("malicious score (symlog)")
        ax_mal.axvline(ATTACK_FROM_ROUND, color="gray", linestyle=":", alpha=0.5)
        ax_mal.legend(fontsize=8)
        ax_mal.grid(alpha=0.3)
        ax_mal.set_title(f"{defense} — {partition} — attack={attack}")
    else:
        fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
        ax, ax2 = axes
        ax.set_title(f"{defense} — {partition} — attack={attack}")

    ax.scatter(honest["round"], honest["score"], s=10, alpha=0.5, color="tab:blue", label="honest (trace, seed=42)")
    if not has_huge_malicious and len(malicious):
        ax.scatter(malicious["round"], malicious["score"], s=20, alpha=0.8, color="tab:red", marker="x", label="malicious (trace, seed=42)")
    ax.axhline(threshold, color="black", linestyle="--", label=f"threshold={threshold:.3f}")
    ax.axvline(ATTACK_FROM_ROUND, color="gray", linestyle=":", alpha=0.5)
    ax.set_ylabel("honest score (zoomed)" if has_huge_malicious else "score")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax2.plot(fpr_by_round.index, fpr_by_round.values, marker="o", markersize=3, color="tab:orange", label="FPR/round (real sweep, 3 seeds pooled)")
    ax2.plot(tpr_by_round.index, tpr_by_round.values, marker="s", markersize=3, color="tab:green", label="TPR/round (real sweep, 3 seeds pooled)")
    ax2.axvline(ATTACK_FROM_ROUND, color="gray", linestyle=":", alpha=0.5)
    ax2.set_xlabel("round")
    ax2.set_ylabel("rate")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fname = f"temporal_{defense}_{partition}_{attack}.png"
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=140)
    plt.close(fig)

    return {
        "defense": defense,
        "partition": partition,
        "attack": attack,
        "threshold": threshold,
        "honest_score_round0_mean": float(honest[honest["round"] == 0]["score"].mean()) if len(honest[honest["round"] == 0]) else np.nan,
        "honest_score_round24_mean": float(honest[honest["round"] == honest["round"].max()]["score"].mean()) if len(honest) else np.nan,
        "honest_score_drift": None,  # filled below
        "fpr_round0": float(fpr_by_round.get(0, np.nan)),
        "fpr_round24": float(fpr_by_round.iloc[-1]) if len(fpr_by_round) else np.nan,
        "tpr_round10": float(tpr_by_round.get(ATTACK_FROM_ROUND, np.nan)),
        "tpr_round24": float(tpr_by_round.iloc[-1]) if len(tpr_by_round) else np.nan,
    }


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    trace = pd.read_csv(os.path.join(ANALYSIS_DIR, "trace_A_client_scores.csv"))
    real = pd.read_csv(os.path.join(REAL_DIR, "per_round_results.csv"))
    calib = pd.read_csv(os.path.join(REAL_DIR, "calibration_results.csv"))

    summary_rows = []
    for defense in DEFENSES:
        for partition in PARTITIONS:
            row = plot_defense_partition(trace, real, defense, partition, calib, attack="large_norm")
            row["honest_score_drift"] = row["honest_score_round24_mean"] - row["honest_score_round0_mean"]
            summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(ANALYSIS_DIR, "step4_temporal_drift_summary.csv"), index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved plots to {FIG_DIR}/temporal_*.png and summary CSV")


if __name__ == "__main__":
    main()
