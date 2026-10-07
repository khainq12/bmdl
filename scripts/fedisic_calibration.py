"""Fed-ISIC2019 calibration: τ/ρ/κ for Norm/Cosine/Sign Consensus, locked
BEFORE any attack benchmarking (ChatGPT's explicit requirement -- "tuyet
doi khong duoc thay Fed-ISIC directional TPR thap roi chinh threshold").
Reuses calibrate_tau/calibrate_rho/calibrate_kappa UNCHANGED (same
functions PathMNIST used) -- only the data source changes.

Calibration population: the 6 centers' own calib slices (carved in
Addendum (j)), used directly as 6 calibration pseudo-clients -- natural
structure preserved rather than re-partitioned into an arbitrary client
count, consistent with Fed-ISIC2019's "no re-partitioning" role.
MALICIOUS_RATIO=0.2 -> 1 of 6 pseudo-clients malicious (large_norm, the
same canonical calibration attack PathMNIST locked -- Addendum (f) --
never the attack under final test, to avoid tuning on test-attack
outcomes). Only ONE calibration condition exists here (no Regime A/B
split like PathMNIST): Fed-ISIC2019 has no partition sweep to transfer
thresholds across (natural federation is the only condition), so Regime
A/B's whole premise (per-partition vs. frozen-IID-transfer) does not
apply -- this calibration is the single locked threshold set.

Usage: python scripts/fedisic_calibration.py
Writes results/fed_isic2019/calibration_results.csv
"""

from __future__ import annotations

import csv
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from benchmark.attacks.suite import apply_attack  # noqa: E402
from benchmark.calibration import calibrate_kappa, calibrate_rho, calibrate_tau  # noqa: E402
from benchmark.datasets.fed_isic2019 import load_fed_isic2019_problem  # noqa: E402
from benchmark.defenses.cosine import cosine_similarity  # noqa: E402
from benchmark.defenses.sign_consensus import sign_consensus_scores  # noqa: E402
from benchmark.models.local_training_image import local_training_fixed_steps_image  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "fed_isic2019")

CALIB_ATTACK = "large_norm"
CALIB_ROUNDS = 5
CALIB_SEED = 1000
FPR_TARGET = 0.05
K_STEPS = 10
BATCH_SIZE = 32
LOCAL_LR = 0.01
MALICIOUS_RATIO = 0.2


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    problem = load_fed_isic2019_problem(seed=CALIB_SEED)
    calib_datasets = problem["calib_datasets"]  # [(Dataset, idx), ...] per center, 6 pseudo-clients
    server_ref_ds, server_ref_idx = problem["server_ref"]
    n_pseudo = len(calib_datasets)
    n_mal = max(1, int(round(MALICIOUS_RATIO * n_pseudo)))
    print(f"{n_pseudo} calibration pseudo-clients (one per natural center), {n_mal} malicious (role), attack={CALIB_ATTACK}")

    model = problem["model_factory"](pretrained=True, seed=CALIB_SEED)
    rng = np.random.default_rng(CALIB_SEED + 500)

    honest_norm, mal_norm = [], []
    honest_cos, mal_cos = [], []
    honest_sign, mal_sign = [], []

    for r in range(CALIB_ROUNDS):
        gw = model.get_weights().copy()

        g_ref_model = problem["model_factory"](pretrained=False, seed=CALIB_SEED)
        g_ref_model.set_weights(gw.copy())
        g_ref = local_training_fixed_steps_image(g_ref_model, server_ref_ds, server_ref_idx, K_STEPS, LOCAL_LR, BATCH_SIZE, rng)

        deltas, mal_flags = [], []
        for cid, (ds, idx) in enumerate(calib_datasets):
            local = problem["model_factory"](pretrained=False, seed=CALIB_SEED)
            local.set_weights(gw.copy())
            d = local_training_fixed_steps_image(local, ds, idx, K_STEPS, LOCAL_LR, BATCH_SIZE, rng)
            is_mal = cid < n_mal and r >= 1  # mirrors PathMNIST calibrate_for_condition's r>=1 gate
            if is_mal:
                d = apply_attack(CALIB_ATTACK, d, rng, {})
            deltas.append(d)
            mal_flags.append(is_mal)

        norm_scores = [float(np.linalg.norm(d)) for d in deltas]
        cos_scores = [cosine_similarity(d, g_ref) for d in deltas]
        sign_scores = sign_consensus_scores(deltas).tolist()

        for i, m in enumerate(mal_flags):
            (mal_norm if m else honest_norm).append(norm_scores[i])
            (mal_cos if m else honest_cos).append(cos_scores[i])
            (mal_sign if m else honest_sign).append(sign_scores[i])

        honest_deltas = [d for d, m in zip(deltas, mal_flags) if not m]
        update = np.mean(honest_deltas, axis=0) if honest_deltas else np.mean(deltas, axis=0)
        model.set_weights(gw + update)
        print(f"round {r}: honest_norm so far={len(honest_norm)} mal_norm so far={len(mal_norm)}")

    tau_result = calibrate_tau(np.array(honest_norm), np.array(mal_norm), fpr_target=FPR_TARGET)
    rho_result = calibrate_rho(np.array(honest_cos), np.array(mal_cos), fpr_target=FPR_TARGET)
    kappa_result = calibrate_kappa(np.array(honest_sign), np.array(mal_sign), fpr_target=FPR_TARGET)

    rows: List[Dict[str, Any]] = []
    for defense, result in [("norm", tau_result), ("cosine", rho_result), ("sign_consensus", kappa_result)]:
        row = dict(result)
        row["defense"] = defense
        row["calibration_regime"] = "natural_federation_single_condition"
        rows.append(row)
        print(f"[calib] {defense:14s} threshold={result['threshold']:.4f} val_fpr={result['validation_fpr']:.3f} val_tpr={result['validation_tpr']:.3f} fallback={result['used_fallback']}")

    with open(os.path.join(OUT_DIR, "calibration_results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {OUT_DIR}/calibration_results.csv")


if __name__ == "__main__":
    main()
