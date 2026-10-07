"""Trace B (analysis_v2): complementarity probe. One stable, non-diverging
trajectory per (partition, attack) — driven by Coordinate Median, the only
aggregator with zero divergence anywhere in the real 594-combo sweep — so
the SAME sequence of honest+malicious updates each round can be scored by
Norm, Cosine, and Sign Consensus simultaneously, without the trajectory
itself diverging differently depending on which single signal would have
driven it. See results/pathmnist/analysis_v2/MISSING_DATA_NOTE.md for the
rationale (this is a diagnostic construction for STEP 7, not a
reproduction of any single real defense's own trajectory — that fidelity
is Trace A's job).

Logs all three detectors' score + accept/reject decision (both Regime A
and, where applicable, Regime B thresholds) for every client every round.

Usage: python scripts/analysis_v2_trace_B.py
Writes results/pathmnist/analysis_v2/trace_B_complementarity.csv
"""

from __future__ import annotations

import csv
import os
import sys
import time
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from benchmark.attacks.suite import apply_attack  # noqa: E402
from benchmark.datasets.pathmnist import _medmnist_split, build_or_load_partitions  # noqa: E402
from benchmark.defenses.cosine import cosine_similarity  # noqa: E402
from benchmark.defenses.sign_consensus import sign_consensus_scores  # noqa: E402
from benchmark.models import local_training_fixed_steps  # noqa: E402
from benchmark.models.torch_cnn import TorchCNNFlat  # noqa: E402
from fl_core.robust_agg import coord_median  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALIB_CSV = os.path.join(ROOT, "results", "pathmnist", "benchmark_v2_corrected", "calibration_results.csv")
OUT_DIR = os.path.join(ROOT, "results", "pathmnist", "analysis_v2")

N_CLIENTS = 5
SEED = int(os.environ.get("ZKFL_SEED", "42"))
PARTITIONS = [("iid", None), ("dirichlet", 1.0), ("dirichlet", 0.5), ("dirichlet", 0.1)]
ATTACKS = ["no_attack", "large_norm", "low_norm", "full_sign_flip", "directional_poisoning", "sparse_coordinate_attack"]

K_STEPS = 50
BATCH_SIZE = 128
LOCAL_LR = 0.01
N_ROUNDS = 25
ATTACK_FROM_ROUND = 10
MALICIOUS_RATIO = 0.2
SERVER_REF_FRACTION = 0.10
DEVICE = os.environ.get("ZKFL_DEVICE", "cuda")


def _tag(partition: str, alpha) -> str:
    return "iid" if partition == "iid" else f"dirichlet_a{alpha}"


def _model(n_classes: int, seed: int) -> TorchCNNFlat:
    return TorchCNNFlat(n_classes=n_classes, seed=seed, device=DEVICE)


def _train(model, X, y, rng) -> np.ndarray:
    return local_training_fixed_steps(model, X, y, K_STEPS, LOCAL_LR, BATCH_SIZE, rng)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    t_start = time.perf_counter()

    calib = pd.read_csv(CALIB_CSV)
    calib_A = calib[calib["calibration_regime"] == "A_condition_specific_oracle"]
    calib_B = calib[calib["calibration_regime"] == "B_iid_transfer_frozen"]

    def threshold_A(defense, tag):
        row = calib_A[(calib_A["defense"] == defense) & (calib_A["partition"] == tag)]
        assert len(row) == 1
        return float(row.iloc[0]["threshold"])

    def threshold_B(defense):
        row = calib_B[calib_B["defense"] == defense]
        assert len(row) == 1
        return float(row.iloc[0]["threshold"])

    norm_B, cos_B, sign_B = threshold_B("norm"), threshold_B("cosine"), threshold_B("sign_consensus")

    print("Loading official PathMNIST splits ...")
    X_train, y_train, info = _medmnist_split("train")
    n_classes = len(info["label"])
    print(f"n_classes={n_classes} K={K_STEPS} batch_size={BATCH_SIZE} device={DEVICE}")

    rows: List[Dict[str, Any]] = []
    combo_count = 0
    n_mal = max(0, int(round(MALICIOUS_RATIO * N_CLIENTS)))
    malicious_client_ids = set(range(n_mal))

    for partition, alpha in PARTITIONS:
        tag = _tag(partition, alpha)
        rec = build_or_load_partitions(len(X_train), y_train, partition, alpha if alpha is not None else 0.5, N_CLIENTS, SERVER_REF_FRACTION, SEED)
        server_ref_idx = np.array(rec["server_ref_idx"])
        client_idx = [np.array(c) for c in rec["client_idx"]]
        server_ref_xy = (X_train[server_ref_idx], y_train[server_ref_idx])
        partition_xy = [(X_train[c], y_train[c]) for c in client_idx]

        tau_A = threshold_A("norm", tag)
        rho_A = threshold_A("cosine", tag)
        kappa_A = threshold_A("sign_consensus", tag)
        is_iid = partition == "iid"

        for attack in ATTACKS:
            model = _model(n_classes, SEED)
            rng = np.random.default_rng(SEED + 9000)
            t0 = time.perf_counter()

            for r in range(N_ROUNDS):
                gw = model.get_weights().copy()
                deltas, mal_flags = [], []
                for cid in range(N_CLIENTS):
                    local = _model(n_classes, SEED)
                    local.set_weights(gw.copy())
                    Xc, yc = partition_xy[cid]
                    d = _train(local, Xc, yc, rng)
                    is_mal = cid in malicious_client_ids and r >= ATTACK_FROM_ROUND
                    if is_mal:
                        d = apply_attack(attack, d, rng, {})
                    deltas.append(d)
                    mal_flags.append(is_mal)

                m = _model(n_classes, SEED)
                m.set_weights(gw.copy())
                g_ref = _train(m, server_ref_xy[0], server_ref_xy[1], rng)

                norm_scores = [float(np.linalg.norm(d)) for d in deltas]
                cos_scores = [cosine_similarity(d, g_ref) for d in deltas]
                sign_scores = sign_consensus_scores(deltas).tolist()

                for cid in range(N_CLIENTS):
                    row = {
                        "seed": SEED,
                        "partition": tag,
                        "attack": attack,
                        "round": r,
                        "client_id": cid,
                        "is_malicious": bool(mal_flags[cid]),
                        "norm_score": norm_scores[cid],
                        "cosine_score": cos_scores[cid],
                        "sign_score": sign_scores[cid],
                        "accept_norm_A": norm_scores[cid] <= tau_A,
                        "accept_cosine_A": cos_scores[cid] >= rho_A,
                        "accept_sign_A": sign_scores[cid] >= kappa_A,
                        "accept_norm_B": (norm_scores[cid] <= norm_B) if not is_iid else None,
                        "accept_cosine_B": (cos_scores[cid] >= cos_B) if not is_iid else None,
                        "accept_sign_B": (sign_scores[cid] >= sign_B) if not is_iid else None,
                    }
                    rows.append(row)

                # trajectory driver: coordinate median (never diverges, independent of the
                # three signals above, which are purely observational in this trace)
                model.set_weights(gw + coord_median(deltas))

            combo_count += 1
            dt = time.perf_counter() - t0
            print(f"[{combo_count:3d}/24] {tag:16s} {attack:24s} {dt:5.1f}s")

        _write_csv(os.path.join(OUT_DIR, f"trace_B_complementarity_seed{SEED}.csv"), rows)
        print(f"  -- checkpointed after {tag}: {len(rows)} rows, elapsed={time.perf_counter()-t_start:.0f}s --")

    _write_csv(os.path.join(OUT_DIR, "trace_B_complementarity.csv"), rows)
    print(f"DONE. {combo_count} combos, {len(rows)} rows, total {time.perf_counter()-t_start:.0f}s")


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        open(path, "w", encoding="utf-8").close()
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
