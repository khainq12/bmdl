"""Combined E2E minimal benchmark — analysis pass (STEP 6/7/8/9/10 of the
ChatGPT-authored prompt). Reads run_manifest.csv / round_metrics.csv /
client_decisions.csv / adaptive_threshold_trace.csv from
results/pathmnist/combined_e2e_minimal/ and writes the summary/safety/
self-influence CSVs. Does not touch benchmark_v2_corrected/analysis_v2/
combined_design_v1.

Usage: python scripts/combined_e2e_analysis.py [--pilot]
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "results", "pathmnist", "combined_e2e_minimal")
ATTACK_FROM_ROUND = 10


def load():
    manifest = pd.read_csv(os.path.join(OUT_DIR, "run_manifest.csv"))
    rounds = pd.read_csv(os.path.join(OUT_DIR, "round_metrics.csv"))
    clients = pd.read_csv(os.path.join(OUT_DIR, "client_decisions.csv"))
    adaptive = pd.read_csv(os.path.join(OUT_DIR, "adaptive_threshold_trace.csv"))
    return manifest, rounds, clients, adaptive


def summary_by(manifest: pd.DataFrame, group_cols) -> pd.DataFrame:
    g = manifest.groupby(group_cols)
    out = g.agg(
        n_runs=("run_id", "count"),
        n_diverged=("diverged", "sum"),
        final_accuracy_mean=("final_accuracy", "mean"),
        final_accuracy_std=("final_accuracy", "std"),
        final_loss_mean=("final_loss", "mean"),
        runtime_s_mean=("runtime_s", "mean"),
    ).reset_index()
    out["divergence_rate"] = out["n_diverged"] / out["n_runs"]
    return out


def compute_detection_rates(clients: pd.DataFrame) -> pd.DataFrame:
    """Per run_id: honest FPR and malicious TPR, computed the same way as
    combined_design_v1 (honest := not malicious_role, OR attack_active==False
    even if malicious_role==True for no_attack condition -- avoids the Trace
    B ambiguity by construction here since we log attack_active/delta_modified
    separately already)."""
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


def catastrophic_safety(manifest: pd.DataFrame, clients: pd.DataFrame, rounds: pd.DataFrame) -> pd.DataFrame:
    c = clients.copy()
    c["malicious_active"] = c["malicious_role"] & c["attack_active"] & c["delta_modified"]
    rows = []
    for candidate, g in manifest.groupby("candidate"):
        ln_runs = g[g["attack"] == "large_norm"]["run_id"].tolist()
        any_accept, any_diverge, zero_accept, zero_diverge = 0, 0, 0, 0
        for run_id in ln_runs:
            mal = c[(c["run_id"] == run_id) & (c["malicious_active"])]
            diverged = bool(manifest.loc[manifest["run_id"] == run_id, "diverged"].iloc[0])
            accepted_any = bool(mal["accepted"].any()) if len(mal) else False
            if accepted_any:
                any_accept += 1
                any_diverge += int(diverged)
            else:
                zero_accept += 1
                zero_diverge += int(diverged)
        rows.append({
            "candidate": candidate,
            "large_norm_runs": len(ln_runs),
            "runs_with_ge1_accept": any_accept,
            "p_diverge_given_ge1_accept": (any_diverge / any_accept) if any_accept else np.nan,
            "runs_with_0_accept": zero_accept,
            "p_diverge_given_0_accept": (zero_diverge / zero_accept) if zero_accept else np.nan,
        })
    return pd.DataFrame(rows)


def self_influence_audit(clients: pd.DataFrame) -> pd.DataFrame:
    """Diagnostic only (STEP 9). Only meaningful for c4_drift_aware rows
    where tau_without_self was logged, restricted to attacked rounds."""
    c = clients.copy()
    if "tau_without_self" not in c.columns:
        return pd.DataFrame()
    # delta_modified (NOT attack_active): attack_active is role+round only and is True
    # under no_attack too (same role/round gating, identity perturbation) -- using it here
    # would silently mix genuinely-perturbed rows with unperturbed no_attack rows, exactly
    # the ambiguity flagged for Trace B. delta_modified is True only when a real attack
    # function actually changed the delta.
    sub = c[c["tau_without_self"].notna() & (c["candidate"] == "c4_drift_aware") & (c["delta_modified"])].copy()
    if sub.empty:
        return pd.DataFrame()
    sub["delta_threshold"] = sub["tau_used"] - sub["tau_without_self"]
    eps = 1e-9
    sub["relative_shift"] = sub["delta_threshold"] / sub["tau_without_self"].abs().clip(lower=eps)
    sub["malicious_norm_over_threshold"] = sub["norm_score"] / sub["tau_used"].clip(lower=eps)
    cols = ["run_id", "round", "partition", "attack", "seed", "client_id", "malicious_role",
            "norm_score", "tau_used", "tau_without_self", "delta_threshold", "relative_shift",
            "malicious_norm_over_threshold", "accepted"]
    cols = [c_ for c_ in cols if c_ in sub.columns]
    return sub[cols]


def main() -> None:
    manifest, rounds, clients, adaptive = load()
    print(f"Loaded: {len(manifest)} runs, {len(rounds)} round rows, {len(clients)} client rows, {len(adaptive)} adaptive rows")

    det = compute_detection_rates(clients)
    # idempotent: drop any previously-merged detection columns before re-merging,
    # so reruns of this script don't produce _x/_y duplicate columns.
    manifest = manifest.drop(columns=[c for c in det.columns if c != "run_id" and c in manifest.columns])
    manifest = manifest.merge(det, on="run_id", how="left")
    manifest.to_csv(os.path.join(OUT_DIR, "run_manifest.csv"), index=False)  # augmented in place with detection cols

    summary_by(manifest, ["candidate"]).to_csv(os.path.join(OUT_DIR, "summary_by_run.csv"), index=False)
    summary_by(manifest, ["candidate", "attack"]).to_csv(os.path.join(OUT_DIR, "summary_by_attack.csv"), index=False)
    summary_by(manifest, ["candidate", "partition"]).to_csv(os.path.join(OUT_DIR, "summary_by_partition.csv"), index=False)
    summary_by(manifest, ["candidate", "seed"]).to_csv(os.path.join(OUT_DIR, "summary_by_seed.csv"), index=False)
    print("Wrote summary_by_{run,attack,partition,seed}.csv")

    cat = catastrophic_safety(manifest, clients, rounds)
    cat.to_csv(os.path.join(OUT_DIR, "catastrophic_safety.csv"), index=False)
    print("catastrophic_safety.csv:")
    print(cat.to_string(index=False))

    si = self_influence_audit(clients)
    si.to_csv(os.path.join(OUT_DIR, "self_influence_audit.csv"), index=False)
    print(f"\nself_influence_audit.csv: {len(si)} attacked-round rows (c4_drift_aware)")
    if len(si):
        print(si[["relative_shift", "malicious_norm_over_threshold", "accepted"]].describe().to_string())


if __name__ == "__main__":
    main()
