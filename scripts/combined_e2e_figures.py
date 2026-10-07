"""STEP 12 figures for combined_e2e_minimal. Reads the CSVs produced by
combined_e2e_analysis.py / combined_e2e_baseline_comparison.py.

Usage: python scripts/combined_e2e_figures.py
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
OUT_DIR = os.path.join(ROOT, "results", "pathmnist", "combined_e2e_minimal")
FIG_DIR = os.path.join(OUT_DIR, "figures")

CANDIDATES = ["c3_static", "c4_static", "c4_drift_aware"]
ATTACKS = ["no_attack", "large_norm", "directional_poisoning", "sparse_coordinate_attack"]
PARTITIONS = ["iid", "dirichlet_a1.0", "dirichlet_a0.5", "dirichlet_a0.1"]
COLORS = {"c3_static": "tab:blue", "c4_static": "tab:green", "c4_drift_aware": "tab:purple"}


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    manifest = pd.read_csv(os.path.join(OUT_DIR, "run_manifest.csv"))
    rounds = pd.read_csv(os.path.join(OUT_DIR, "round_metrics.csv"))
    adaptive = pd.read_csv(os.path.join(OUT_DIR, "adaptive_threshold_trace.csv"))
    by_attack = pd.read_csv(os.path.join(OUT_DIR, "summary_by_attack.csv"))
    by_partition = pd.read_csv(os.path.join(OUT_DIR, "summary_by_partition.csv"))
    cat = pd.read_csv(os.path.join(OUT_DIR, "catastrophic_safety.csv"))
    baseline_cmp_path = os.path.join(OUT_DIR, "baseline_comparability.csv")
    baseline_cmp = pd.read_csv(baseline_cmp_path) if os.path.exists(baseline_cmp_path) else None
    present_attacks = [a for a in ATTACKS if a in manifest["attack"].unique()]
    present_partitions = [p for p in PARTITIONS if p in manifest["partition"].unique()]

    # 1. final accuracy by candidate x attack
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(present_attacks))
    width = 0.25
    for i, cand in enumerate(CANDIDATES):
        vals = [by_attack[(by_attack.candidate == cand) & (by_attack.attack == a)]["final_accuracy_mean"].values for a in present_attacks]
        vals = [v[0] if len(v) else np.nan for v in vals]
        ax.bar(x + i * width, vals, width, label=cand, color=COLORS[cand])
    ax.set_xticks(x + width)
    ax.set_xticklabels(present_attacks, rotation=20, ha="right")
    ax.set_ylabel("final test accuracy (mean across partitions/seeds)")
    ax.set_title("Fig 1: final accuracy by candidate x attack")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "1_final_accuracy_by_attack.png"), dpi=140)
    plt.close(fig)

    # 2. accuracy degradation vs no_attack
    if "no_attack" in present_attacks:
        fig, ax = plt.subplots(figsize=(9, 5))
        base = {c: by_attack[(by_attack.candidate == c) & (by_attack.attack == "no_attack")]["final_accuracy_mean"].values[0] for c in CANDIDATES}
        attacked = [a for a in present_attacks if a != "no_attack"]
        xx = np.arange(len(attacked))
        for i, cand in enumerate(CANDIDATES):
            vals = [base[cand] - by_attack[(by_attack.candidate == cand) & (by_attack.attack == a)]["final_accuracy_mean"].values[0] for a in attacked]
            ax.bar(xx + i * width, vals, width, label=cand, color=COLORS[cand])
        ax.set_xticks(xx + width)
        ax.set_xticklabels(attacked, rotation=20, ha="right")
        ax.set_ylabel("accuracy degradation vs no_attack")
        ax.set_title("Fig 2: accuracy degradation relative to matching no_attack run")
        ax.legend()
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "2_accuracy_degradation.png"), dpi=140)
        plt.close(fig)

    # 3. divergence heatmap (candidate x attack)
    fig, ax = plt.subplots(figsize=(8, 4))
    mat = by_attack.pivot_table(index="candidate", columns="attack", values="divergence_rate").reindex(index=CANDIDATES, columns=present_attacks)
    im = ax.imshow(mat.values, cmap="Reds", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(present_attacks)))
    ax.set_xticklabels(present_attacks, rotation=20, ha="right")
    ax.set_yticks(range(len(CANDIDATES)))
    ax.set_yticklabels(CANDIDATES)
    for i in range(len(CANDIDATES)):
        for j in range(len(present_attacks)):
            v = mat.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center")
    fig.colorbar(im, label="divergence rate")
    ax.set_title("Fig 3: divergence heatmap")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "3_divergence_heatmap.png"), dpi=140)
    plt.close(fig)

    # 4. TPR by attack (malicious_tpr from manifest, averaged by candidate x attack)
    tpr = manifest.groupby(["candidate", "attack"])["malicious_tpr"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(9, 5))
    attacked = [a for a in present_attacks if a != "no_attack"]
    xx = np.arange(len(attacked))
    for i, cand in enumerate(CANDIDATES):
        vals = [tpr[(tpr.candidate == cand) & (tpr.attack == a)]["malicious_tpr"].values for a in attacked]
        vals = [v[0] if len(v) else np.nan for v in vals]
        ax.bar(xx + i * width, vals, width, label=cand, color=COLORS[cand])
    ax.set_xticks(xx + width)
    ax.set_xticklabels(attacked, rotation=20, ha="right")
    ax.set_ylabel("TPR")
    ax.set_title("Fig 4: TPR by attack (real end-to-end trajectory)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "4_tpr_by_attack.png"), dpi=140)
    plt.close(fig)

    # 5. honest FPR by partition
    fpr = manifest[manifest.attack == "no_attack"].groupby(["candidate", "partition"])["honest_fpr"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(9, 5))
    xx = np.arange(len(present_partitions))
    for i, cand in enumerate(CANDIDATES):
        vals = [fpr[(fpr.candidate == cand) & (fpr.partition == p)]["honest_fpr"].values for p in present_partitions]
        vals = [v[0] if len(v) else np.nan for v in vals]
        ax.bar(xx + i * width, vals, width, label=cand, color=COLORS[cand])
    ax.set_xticks(xx + width)
    ax.set_xticklabels(present_partitions, rotation=20)
    ax.set_ylabel("honest FPR (no_attack condition)")
    ax.set_title("Fig 5: honest FPR by partition")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "5_honest_fpr_by_partition.png"), dpi=140)
    plt.close(fig)

    # 6. static vs drift-aware FPR (c4_static vs c4_drift_aware only)
    fig, ax = plt.subplots(figsize=(7, 5))
    for cand, marker in [("c4_static", "o"), ("c4_drift_aware", "s")]:
        vals = [fpr[(fpr.candidate == cand) & (fpr.partition == p)]["honest_fpr"].values for p in present_partitions]
        vals = [v[0] if len(v) else np.nan for v in vals]
        ax.plot(range(len(present_partitions)), vals, marker=marker, label=cand)
    ax.set_xticks(range(len(present_partitions)))
    ax.set_xticklabels(present_partitions, rotation=20)
    ax.set_ylabel("honest FPR")
    ax.set_title("Fig 6: static vs drift-aware Norm gate -- honest FPR")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "6_static_vs_driftaware_fpr.png"), dpi=140)
    plt.close(fig)

    # 7. adaptive threshold vs round (c4_drift_aware, one representative run per partition, large_norm)
    fig, ax = plt.subplots(figsize=(9, 5))
    sub = adaptive[(adaptive.candidate == "c4_drift_aware") & (adaptive.attack == "large_norm") & (adaptive.seed == adaptive.seed.min())]
    for p in present_partitions:
        s = sub[sub.partition == p].sort_values("round")
        if len(s):
            ax.plot(s["round"], s["tau_drift"], marker=".", label=p)
    ax.axvline(10, color="gray", linestyle="--", linewidth=0.8, label="attack starts")
    ax.set_xlabel("round")
    ax.set_ylabel("adaptive tau_drift")
    ax.set_title("Fig 7: adaptive Norm threshold vs round (large_norm, seed=min)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "7_adaptive_threshold_vs_round.png"), dpi=140)
    plt.close(fig)

    # 8. malicious norm vs adaptive threshold (large_norm, c4_drift_aware)
    si_path = os.path.join(OUT_DIR, "self_influence_audit.csv")
    fig, ax = plt.subplots(figsize=(7, 6))
    if os.path.exists(si_path):
        si = pd.read_csv(si_path)
        si_ln = si[si["attack"] == "large_norm"]
        if len(si_ln):
            ax.scatter(si_ln["tau_used"], si_ln["norm_score"], alpha=0.6, s=15)
            lim = max(si_ln["tau_used"].max(), si_ln["norm_score"].quantile(0.95)) * 1.1
            ax.plot([0, lim], [0, lim], "k--", linewidth=0.8, label="malicious_norm = threshold")
    ax.set_xlabel("adaptive tau_drift")
    ax.set_ylabel("malicious norm score")
    ax.set_title("Fig 8: malicious norm vs adaptive threshold (large_norm)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "8_malicious_norm_vs_threshold.png"), dpi=140)
    plt.close(fig)

    # 9/10. clean and attacked learning curves (iid, seed=min)
    seed0 = int(manifest["seed"].min())
    for fig_num, attack_filter, fname, title in [
        (9, "no_attack", "9_clean_learning_curves.png", "Fig 9: clean (no_attack) learning curves"),
        (10, "large_norm", "10_attacked_learning_curves.png", "Fig 10: attacked (large_norm) learning curves"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5))
        r = rounds[(rounds.attack == attack_filter) & (rounds.partition == "iid") & (rounds.seed == seed0)]
        for cand in CANDIDATES:
            s = r[r.candidate == cand].sort_values("round")
            if len(s):
                ax.plot(s["round"], s["accuracy"], marker=".", label=cand, color=COLORS[cand])
        ax.axvline(10, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("round")
        ax.set_ylabel("test accuracy")
        ax.set_title(f"{title} (iid, seed={seed0})")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, fname), dpi=140)
        plt.close(fig)

    # 11. Combined vs reused single-defense baselines
    if baseline_cmp is not None:
        comp = baseline_cmp[baseline_cmp.comparable]
        comp = comp.merge(manifest[["run_id", "candidate", "final_accuracy"]], left_on="combined_run_id", right_on="run_id")
        agg = comp.groupby(["candidate", "attack", "baseline_defense"])[["final_accuracy", "baseline_final_accuracy"]].mean().reset_index()
        fig, axes = plt.subplots(1, len(present_attacks), figsize=(5 * len(present_attacks), 4.5), sharey=True)
        if len(present_attacks) == 1:
            axes = [axes]
        all_methods = CANDIDATES + sorted(agg["baseline_defense"].unique())
        for ax, attack in zip(axes, present_attacks):
            sub = agg[agg.attack == attack]
            vals, labels = [], []
            for cand in CANDIDATES:
                v = sub[sub.candidate == cand]["final_accuracy"]
                vals.append(v.mean() if len(v) else np.nan)
                labels.append(cand)
            for bd in sorted(agg["baseline_defense"].unique()):
                v = sub[sub.baseline_defense == bd]["baseline_final_accuracy"]
                vals.append(v.mean() if len(v) else np.nan)
                labels.append(bd)
            ax.bar(range(len(vals)), vals, color=["tab:purple"] * 3 + ["tab:gray"] * 6)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
            ax.set_title(attack, fontsize=9)
            ax.grid(alpha=0.3, axis="y")
        axes[0].set_ylabel("final accuracy")
        fig.suptitle("Fig 11: Combined candidates vs reused single-defense baselines")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "11_combined_vs_baselines.png"), dpi=140)
        plt.close(fig)

    # 12. self-influence threshold-shift plot
    if os.path.exists(si_path):
        si = pd.read_csv(si_path)
        if len(si):
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.hist(si["relative_shift"].dropna(), bins=30)
            ax.set_xlabel("relative threshold shift (threshold_all - threshold_without_attacker) / threshold_without_attacker")
            ax.set_ylabel("count (attacked client-rounds)")
            ax.set_title("Fig 12: self-influence -- attacker's effect on its own threshold")
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(FIG_DIR, "12_self_influence_shift.png"), dpi=140)
            plt.close(fig)

    print(f"Saved figures to {FIG_DIR}/")


if __name__ == "__main__":
    main()
