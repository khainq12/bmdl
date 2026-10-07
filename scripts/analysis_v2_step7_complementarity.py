"""STEP 7 (analysis_v2): complementarity of Norm, Cosine, Sign Consensus.
Uses Trace B (same median-driven stable trajectory, all 3 signals scored
on the SAME generated updates each round). Regime A thresholds (condition
-specific) are the primary view; Regime B noted separately.

Usage: python scripts/analysis_v2_step7_complementarity.py
Writes results/pathmnist/analysis_v2/step7_complementarity_tables.csv and
results/pathmnist/analysis_v2/figures/complementarity_overlap.png
"""

from __future__ import annotations

import itertools
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
ATTACK_FROM_ROUND = 10

LABELS = ["Norm", "Cosine", "Sign"]
ACCEPT_COLS = ["accept_norm_A", "accept_cosine_A", "accept_sign_A"]


def overlap_pattern(row) -> str:
    rej = [not row[c] for c in ACCEPT_COLS]  # True = rejected by that signal
    names = [LABELS[i] for i, r in enumerate(rej) if r]
    if not names:
        return "accepted_by_all"
    return "rejected_by_" + "+".join(names)


def summarize(df: pd.DataFrame, label: str) -> pd.DataFrame:
    df = df.copy()
    df["pattern"] = df.apply(overlap_pattern, axis=1)
    counts = df["pattern"].value_counts().rename("count").reset_index().rename(columns={"index": "pattern"})
    counts["fraction"] = counts["count"] / len(df)
    counts["group"] = label
    counts["n_total"] = len(df)
    return counts


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    pooled_path = os.path.join(ANALYSIS_DIR, "trace_B_complementarity_pooled3seeds.csv")
    single_path = os.path.join(ANALYSIS_DIR, "trace_B_complementarity.csv")
    trace = pd.read_csv(pooled_path if os.path.exists(pooled_path) else single_path)
    print(f"Using {'POOLED 3-seed' if os.path.exists(pooled_path) else 'single-seed'} trace: {trace.shape[0]} rows, seeds={sorted(trace['seed'].unique()) if 'seed' in trace.columns else 'n/a'}")

    malicious_active = trace[(trace["is_malicious"]) & (trace["round"] >= ATTACK_FROM_ROUND) & (trace["attack"] != "no_attack")]
    honest = trace[~trace["is_malicious"]]

    all_rows = []
    all_rows.append(summarize(malicious_active, "malicious_updates_pooled"))
    all_rows.append(summarize(honest, "honest_updates_pooled"))

    for attack in sorted(malicious_active["attack"].unique()):
        sub = malicious_active[malicious_active["attack"] == attack]
        all_rows.append(summarize(sub, f"malicious_updates_{attack}"))

    out = pd.concat(all_rows, ignore_index=True)
    out = out[["group", "pattern", "count", "fraction", "n_total"]]
    out.to_csv(os.path.join(ANALYSIS_DIR, "step7_complementarity_tables.csv"), index=False)
    pd.set_option("display.width", 200)
    print(out.to_string(index=False))

    # --- per-seed consistency check (answers: is this seed=42-specific?) ---
    if "seed" in trace.columns and trace["seed"].nunique() > 1:
        per_seed_rows = []
        for seed in sorted(trace["seed"].unique()):
            s_mal = malicious_active[malicious_active["seed"] == seed] if "seed" in malicious_active.columns else trace[(trace["seed"] == seed) & (trace["is_malicious"]) & (trace["round"] >= ATTACK_FROM_ROUND) & (trace["attack"] != "no_attack")]
            s_honest = honest[honest["seed"] == seed] if "seed" in honest.columns else trace[(trace["seed"] == seed) & (~trace["is_malicious"])]
            r1 = summarize(s_mal, f"malicious_pooled_seed{seed}")
            r2 = summarize(s_honest, f"honest_pooled_seed{seed}")
            per_seed_rows.append(r1)
            per_seed_rows.append(r2)
        per_seed = pd.concat(per_seed_rows, ignore_index=True)[["group", "pattern", "count", "fraction", "n_total"]]
        per_seed.to_csv(os.path.join(ANALYSIS_DIR, "step7_per_seed_consistency.csv"), index=False)
        print("\n=== Per-seed consistency check (malicious_pooled and honest_pooled, by seed) ===")
        print(per_seed.to_string(index=False))

    # --- figure: overlap pattern bar chart, malicious (pooled + per-attack) ---
    patterns_order = [
        "accepted_by_all",
        "rejected_by_Norm",
        "rejected_by_Cosine",
        "rejected_by_Sign",
        "rejected_by_Norm+Cosine",
        "rejected_by_Norm+Sign",
        "rejected_by_Cosine+Sign",
        "rejected_by_Norm+Cosine+Sign",
    ]
    attacks = sorted(malicious_active["attack"].unique())
    fig, ax = plt.subplots(figsize=(11, 6))
    width = 0.1
    x = np.arange(len(patterns_order))
    for i, attack in enumerate(attacks):
        sub = out[out["group"] == f"malicious_updates_{attack}"].set_index("pattern")
        vals = [sub["fraction"].get(p, 0.0) for p in patterns_order]
        ax.bar(x + i * width, vals, width, label=attack)
    ax.set_xticks(x + width * (len(attacks) - 1) / 2)
    ax.set_xticklabels([p.replace("rejected_by_", "rej:\n").replace("accepted_by_all", "accepted\nby all") for p in patterns_order], fontsize=8)
    ax.set_ylabel("fraction of malicious client-rounds")
    ax.set_title("Complementarity: which signal(s) reject each malicious update (Regime A, Trace B)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "complementarity_overlap.png"), dpi=140)
    plt.close(fig)
    print(f"\nSaved figure to {FIG_DIR}/complementarity_overlap.png")


if __name__ == "__main__":
    main()
