"""Analysis for the 192-run C4DriftAware targeted confirmation
(8 seeds x 4 partitions x 6 attacks). Mirrors combined_e2e_analysis.py's
logic, scoped to this single candidate / this output directory.

Usage: python scripts/c4_drift_aware_confirmation_analysis.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "results", "pathmnist", "c4_drift_aware_confirmation")


def compute_detection_rates(clients: pd.DataFrame) -> pd.DataFrame:
    c = clients.copy()
    c["honest"] = (~c["malicious_role"]) | (~c["delta_modified"])
    c["malicious_active"] = c["malicious_role"] & c["attack_active"] & c["delta_modified"]
    rows = []
    for run_id, g in c.groupby("run_id"):
        honest = g[g["honest"]]
        mal = g[g["malicious_active"]]
        rows.append({
            "run_id": run_id,
            "honest_fpr": (~honest["accepted"]).mean() if len(honest) else np.nan,
            "n_honest": len(honest),
            "malicious_tpr": (~mal["accepted"]).mean() if len(mal) else np.nan,
            "n_malicious_active": len(mal),
        })
    return pd.DataFrame(rows)


def main() -> None:
    manifest = pd.read_csv(os.path.join(OUT_DIR, "run_manifest.csv"))
    clients = pd.read_csv(os.path.join(OUT_DIR, "client_decisions.csv"))
    print(f"Loaded {len(manifest)} runs, {len(clients)} client rows")
    assert len(manifest) == 192, f"expected 192 runs, got {len(manifest)}"

    det = compute_detection_rates(clients)
    base_cols = ["run_id", "seed", "partition", "attack", "candidate", "n_rounds", "diverged", "runtime_s", "final_accuracy", "final_loss"]
    manifest = manifest[[c for c in base_cols if c in manifest.columns]]
    manifest = manifest.merge(det, on="run_id", how="left")
    manifest.to_csv(os.path.join(OUT_DIR, "run_manifest.csv"), index=False)

    print("\n=== any numerical divergence anywhere? ===")
    print(manifest["diverged"].any())

    print("\n=== collapse check (honest_fpr > 0.9) ===")
    collapse = manifest[manifest["honest_fpr"] > 0.9]
    print(f"{len(collapse)} / {len(manifest)} runs with honest_fpr > 0.9")
    if len(collapse):
        print(collapse[["run_id", "honest_fpr", "final_accuracy"]].to_string(index=False))

    print("\n=== large_norm catastrophic safety, by seed ===")
    ln = manifest[manifest["attack"] == "large_norm"]
    ln_clients = clients[(clients["attack"] == "large_norm") & (clients["malicious_role"]) & (clients["attack_active"]) & (clients["delta_modified"])]
    for seed in sorted(manifest["seed"].unique()):
        sub = ln_clients[ln_clients["seed"] == seed]
        accepted_any = int(sub["accepted"].sum())
        print(f"  seed {seed}: {accepted_any} accepted malicious large_norm client-rounds (of {len(sub)})")
    total_accepted = int(ln_clients["accepted"].sum())
    print(f"TOTAL across all 8 seeds: {total_accepted} accepted / {len(ln_clients)} large_norm malicious client-rounds")

    print("\n=== TPR by attack (all 8 seeds pooled) ===")
    print(manifest.groupby("attack")["malicious_tpr"].agg(["mean", "std", "min", "max"]).to_string())

    print("\n=== honest FPR by seed (no_attack condition) ===")
    na = manifest[manifest["attack"] == "no_attack"]
    print(na.groupby("seed")["honest_fpr"].agg(["mean", "std", "min", "max"]).to_string())

    print("\n=== honest FPR by partition (pooled 8 seeds) ===")
    print(manifest.groupby("partition")["honest_fpr"].agg(["mean", "std"]).to_string())

    print("\n=== overall summary ===")
    print(manifest[["honest_fpr", "malicious_tpr", "final_accuracy"]].describe().to_string())

    manifest.to_csv(os.path.join(OUT_DIR, "run_manifest.csv"), index=False)
    print(f"\nWrote augmented run_manifest.csv to {OUT_DIR}/")


if __name__ == "__main__":
    main()
