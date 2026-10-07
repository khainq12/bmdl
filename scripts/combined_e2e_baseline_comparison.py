"""STEP 10 (combined_e2e_minimal): fair baseline reuse. Joins the new
Combined runs against results/pathmnist/benchmark_v2_corrected/ (READ ONLY
-- never rerun) on exactly-matching partition/seed/attack. Protocol fields
(K=50, 25 rounds, model, optimizer, LR, 5 clients, attack_start=round 10,
attacker=client 0, eval protocol) are locked constants shared by both the
original 594-combo sweep and this new benchmark -- verified identical by
construction (same scripts/pathmnist_benchmark_v2.py-derived constants),
not re-derived per row. A config is marked comparable=False only if no
matching baseline row is actually found (defensive, not assumed).

Usage: python scripts/combined_e2e_baseline_comparison.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "results", "pathmnist", "combined_e2e_minimal")
BASELINE_CSV = os.path.join(ROOT, "results", "pathmnist", "benchmark_v2_corrected", "per_round_results.csv")

BASELINE_DEFENSES = ["fedavg", "norm", "cosine", "sign_consensus", "median", "multi_krum"]
# Regime A (condition-specific) for detectors, not_applicable for aggregators --
# the primary-view regime used throughout analysis_v2/combined_design_v1.
PREFERRED_REGIME = {"norm": "A_condition_specific_oracle", "cosine": "A_condition_specific_oracle",
                     "sign_consensus": "A_condition_specific_oracle", "fedavg": "not_applicable",
                     "median": "not_applicable", "multi_krum": "not_applicable"}

PROTOCOL_FIELDS_LOCKED = (
    "K=50 local SGD steps/client/round; 25 FL rounds; same TorchCNNFlat model/init scheme; "
    "SGD optimizer, LR=0.01, batch_size=128; 5 clients; MALICIOUS_RATIO=0.2 (client 0); "
    "attack_start=round 10; SERVER_REF_FRACTION=0.10; same evaluate() on official PathMNIST "
    "test split -- identical across benchmark_v2_corrected and combined_e2e_minimal by "
    "construction (both derive these constants from the same locked protocol addenda)."
)


def main() -> None:
    manifest = pd.read_csv(os.path.join(OUT_DIR, "run_manifest.csv"))
    baseline = pd.read_csv(BASELINE_CSV)

    # final-round row per (partition, seed, attack, defense, regime)
    baseline_final = (
        baseline.sort_values("round")
        .groupby(["partition", "seed", "attack", "defense", "calibration_regime"], as_index=False)
        .last()
    )

    rows = []
    for _, run in manifest.iterrows():
        for defense in BASELINE_DEFENSES:
            regime = PREFERRED_REGIME[defense]
            match = baseline_final[
                (baseline_final["partition"] == run["partition"]) &
                (baseline_final["seed"] == run["seed"]) &
                (baseline_final["attack"] == run["attack"]) &
                (baseline_final["defense"] == defense) &
                (baseline_final["calibration_regime"] == regime)
            ]
            comparable = len(match) == 1
            row = {
                "combined_run_id": run["run_id"],
                "baseline_defense": defense,
                "baseline_regime": regime,
                "seed": run["seed"],
                "partition": run["partition"],
                "attack": run["attack"],
                "protocol_fields": PROTOCOL_FIELDS_LOCKED,
                "comparable": comparable,
                "reason": "exact (partition,seed,attack,defense,regime) match found" if comparable else "no matching baseline row",
            }
            if comparable:
                m = match.iloc[0]
                row.update(
                    baseline_final_accuracy=float(m["accuracy"]),
                    baseline_final_loss=float(m["loss"]),
                    baseline_diverged=bool(m["diverged"]),
                )
            rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, "baseline_comparability.csv"), index=False)
    n_ok = int(out["comparable"].sum())
    print(f"baseline_comparability.csv: {n_ok}/{len(out)} comparable rows "
          f"({out.loc[~out['comparable'], 'combined_run_id'].nunique()} combined runs have >=1 non-comparable baseline)")


if __name__ == "__main__":
    main()
