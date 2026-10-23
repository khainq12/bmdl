"""Phase 1 (CPU-only) reanalysis of the 504-run benchmark's no_attack subset.
Fixes the gap in the earlier audit: a frozen run (update=0 every round) trivially
satisfies the retrospective convergence criterion (flat curve = both conditions
met), so it was being counted as "converged" even though nothing was learned.
This script explicitly excludes frozen runs from "genuinely converged" and
reports them in their own category.

No GPU, no new training -- reads only existing CSVs.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FB_DIR = os.path.join(ROOT, "results", "rebenchmark_v2", "full_benchmark")
OUT_DIR = os.path.join(FB_DIR, "analysis_artifacts")

PATIENCE, ACC_EPS, LOSS_EPS = 15, 0.005, 0.002
CHECKPOINTS = [150, 200, 240, 280]
DETECTOR_DEFENSES = {"norm", "cosine", "sign_consensus", "c4_drift_aware"}
PRIORITY = ["fedavg", "median", "multi_krum", "c4_drift_aware"]


def find_convergence_round(val_acc, val_loss):
    n = len(val_acc)
    if n < PATIENCE + 1:
        return None
    best_suffix = np.maximum.accumulate(val_acc[::-1])[::-1]
    for r in range(1, n - PATIENCE + 1):
        if best_suffix[r] - val_acc[r - 1] >= ACC_EPS:
            continue
        tail = val_loss[r:]
        slope = np.polyfit(np.arange(len(tail)), tail, 1)[0]
        if slope < -LOSS_EPS:
            continue
        return r
    return None


def classify(acc: np.ndarray, loss: np.ndarray, frozen: bool) -> str:
    if frozen:
        return "frozen"
    conv = find_convergence_round(acc, loss)
    tail_acc = acc[-30:]
    x = np.arange(len(tail_acc))
    acc_slope = np.polyfit(x, tail_acc, 1)[0]
    acc_std = tail_acc.std()
    if conv is not None:
        return "genuinely_converged"
    if acc_std < 1e-4:
        return "frozen"
    if acc_slope > ACC_EPS / PATIENCE:
        return "still_improving"
    if acc_std > 0.02 and abs(acc_slope) <= ACC_EPS / PATIENCE:
        return "oscillating"
    return "plateau"


def main() -> None:
    manifest = pd.read_csv(os.path.join(FB_DIR, "run_manifest.csv"))
    rounds = pd.read_csv(os.path.join(FB_DIR, "per_round_results.csv"))
    freeze = pd.read_csv(os.path.join(OUT_DIR, "freeze_detection.csv"))
    frozen_ids = set(freeze[freeze["frozen_last30"]]["run_id"])

    na = manifest[manifest["attack"] == "no_attack"]
    rows = []
    for _, r in na.iterrows():
        g = rounds[rounds["run_id"] == r["run_id"]].sort_values("round")
        acc = g["accuracy"].to_numpy()
        loss = g["loss"].to_numpy()
        is_frozen = r["run_id"] in frozen_ids
        cls = classify(acc, loss, is_frozen)
        row = {"run_id": r["run_id"], "defense": r["defense"], "partition": r["partition"],
               "seed": r["seed"], "class": cls}
        for cp in CHECKPOINTS:
            idx = min(cp, len(acc) - 1)
            row[f"acc_r{cp}"] = float(acc[idx])
            row[f"loss_r{cp}"] = float(loss[idx])
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "phase1_no_attack_classification.csv"), index=False)

    print("=== Corrected classification (frozen excluded from genuinely_converged) ===")
    print(df["class"].value_counts().to_string())
    print()
    print("=== By defense ===")
    print(pd.crosstab(df["defense"], df["class"]).to_string())
    print()
    print("=== Priority defenses only ===")
    print(pd.crosstab(df[df.defense.isin(PRIORITY)]["defense"], df[df.defense.isin(PRIORITY)]["class"]).to_string())

    # ------------- accuracy/loss trend at each checkpoint, per defense (mean over partition x seed) -------------
    print("\n=== Mean accuracy by defense at each checkpoint round ===")
    trend = df.groupby("defense")[[f"acc_r{cp}" for cp in CHECKPOINTS]].mean().round(4)
    print(trend.to_string())
    trend.to_csv(os.path.join(OUT_DIR, "phase1_accuracy_trend_by_defense.csv"))

    # ------------- ranking stability: does the ORDER of defenses by mean accuracy change across checkpoints? -------------
    print("\n=== Defense ranking (best to worst) at each checkpoint ===")
    rankings = {}
    for cp in CHECKPOINTS:
        order = trend[f"acc_r{cp}"].sort_values(ascending=False).index.tolist()
        rankings[cp] = order
        print(f"round {cp}: {order}")
    # Spearman-style stability check: rank correlation between round-150 order and round-280 order
    from scipy.stats import spearmanr
    r150 = {d: i for i, d in enumerate(rankings[150])}
    r280 = {d: i for i, d in enumerate(rankings[280])}
    defs = list(r150.keys())
    rho, pval = spearmanr([r150[d] for d in defs], [r280[d] for d in defs])
    print(f"\nSpearman rank correlation (round150 vs round280 ordering): rho={rho:.3f}, p={pval:.3f}")

    with open(os.path.join(OUT_DIR, "phase1_rankings.txt"), "w") as f:
        for cp in CHECKPOINTS:
            f.write(f"round {cp}: {rankings[cp]}\n")
        f.write(f"\nSpearman rho(150,280)={rho:.3f} p={pval:.3f}\n")

    # ------------- C4-DA vs Multi-Krum at each checkpoint (paired by partition x seed) -------------
    print("\n=== C4-DA vs Multi-Krum, paired, at each checkpoint ===")
    piv = df.pivot_table(index=["partition", "seed"], columns="defense",
                          values=[f"acc_r{cp}" for cp in CHECKPOINTS])
    for cp in CHECKPOINTS:
        a = piv[(f"acc_r{cp}", "c4_drift_aware")]
        b = piv[(f"acc_r{cp}", "multi_krum")]
        diff = (a - b).dropna()
        print(f"round {cp}: c4-mkrum mean_diff={diff.mean():.4f} std={diff.std():.4f} "
              f"c4_better={int((diff>0).sum())}/{len(diff)}")

    print(f"\nWrote {OUT_DIR}/phase1_no_attack_classification.csv, "
          f"phase1_accuracy_trend_by_defense.csv, phase1_rankings.txt")


if __name__ == "__main__":
    main()
