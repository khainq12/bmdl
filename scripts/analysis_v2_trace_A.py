"""Trace A (analysis_v2): reproduces each real per-defense trajectory for
{norm, cosine, sign_consensus} x 4 partitions x 6 attacks, Regime A only,
seed=42 — IDENTICAL config to the real corrected sweep (K=50, batch=128,
25 rounds, attack from round 10, same frozen thresholds) — but additionally
logs each client's raw score, accept-decision, and delta norm per round.

See results/pathmnist/analysis_v2/MISSING_DATA_NOTE.md for why this is
needed and why it is NOT a rerun of the 594-combo matrix (different output,
same inputs/config, 72 combos vs 594, single seed).

Usage: python scripts/analysis_v2_trace_A.py
Writes results/pathmnist/analysis_v2/trace_A_client_scores.csv
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
from benchmark.defenses import make_defense  # noqa: E402
from benchmark.defenses.cosine import cosine_similarity  # noqa: E402
from benchmark.defenses.sign_consensus import sign_consensus_scores  # noqa: E402
from benchmark.models import local_training_fixed_steps  # noqa: E402
from benchmark.models.torch_cnn import TorchCNNFlat  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALIB_CSV = os.path.join(ROOT, "results", "pathmnist", "benchmark_v2_corrected", "calibration_results.csv")
OUT_DIR = os.path.join(ROOT, "results", "pathmnist", "analysis_v2")

N_CLIENTS = 5
SEED = 42
PARTITIONS = [("iid", None), ("dirichlet", 1.0), ("dirichlet", 0.5), ("dirichlet", 0.1)]
ATTACKS = ["no_attack", "large_norm", "low_norm", "full_sign_flip", "directional_poisoning", "sparse_coordinate_attack"]
DEFENSES = ["norm", "cosine", "sign_consensus"]

K_STEPS = 50
BATCH_SIZE = 128
LOCAL_LR = 0.01
N_ROUNDS = 25
ATTACK_FROM_ROUND = 10
MALICIOUS_RATIO = 0.2
SERVER_REF_FRACTION = 0.10
DEVICE = os.environ.get("ZKFL_DEVICE", "cuda")
DIVERGENCE_LOSS_THRESHOLD = 50.0

SCORE_KEY = {"norm": "tau", "cosine": "rho", "sign_consensus": "kappa"}


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

    def threshold_for(defense, tag):
        row = calib_A[(calib_A["defense"] == defense) & (calib_A["partition"] == tag)]
        assert len(row) == 1, (defense, tag, len(row))
        return float(row.iloc[0]["threshold"])

    print("Loading official PathMNIST splits ...")
    X_train, y_train, info = _medmnist_split("train")
    X_test, y_test, _ = _medmnist_split("test")
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

        for attack in ATTACKS:
            for defense_name in DEFENSES:
                threshold = threshold_for(defense_name, tag)
                model = _model(n_classes, SEED)
                rng = np.random.default_rng(SEED + 7000)
                defense = make_defense(defense_name)
                threshold_ctx = {"tau": None, "rho": None, "kappa": None, "sign_topk": None, "krum_f": 1}
                threshold_ctx[SCORE_KEY[defense_name]] = threshold

                diverged_at = None
                t0 = time.perf_counter()
                for r in range(N_ROUNDS):
                    if diverged_at is not None:
                        # safeguard: skip all training compute, copy the triggering round
                        # forward (mirrors the real sweep's behavior, Addendum (i))
                        last = [
                            rw for rw in rows
                            if rw["partition"] == tag and rw["attack"] == attack
                            and rw["defense"] == defense_name and rw["round"] == diverged_at
                        ]
                        for base in last:
                            rows.append({**base, "round": r})
                        continue

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

                    ctx = dict(threshold_ctx)
                    ctx["rng"] = rng
                    g_ref = None
                    if defense_name == "cosine":
                        Xr, yr = server_ref_xy
                        m = _model(n_classes, SEED)
                        m.set_weights(gw.copy())
                        g_ref = _train(m, Xr, yr, rng)
                        ctx["g_ref"] = g_ref

                    # compute per-client scores explicitly (same formulas the defense uses internally)
                    if defense_name == "norm":
                        scores = [float(np.linalg.norm(d)) for d in deltas]
                    elif defense_name == "cosine":
                        scores = [cosine_similarity(d, g_ref) for d in deltas]
                    else:
                        scores = sign_consensus_scores(deltas).tolist()

                    result = defense.aggregate(deltas, ctx)
                    model.set_weights(gw + result.update)
                    acc, loss = model.evaluate(X_test, y_test)

                    for cid in range(N_CLIENTS):
                        rows.append(
                            {
                                "partition": tag,
                                "attack": attack,
                                "defense": defense_name,
                                "round": r,
                                "client_id": cid,
                                "score": float(scores[cid]),
                                "delta_norm": float(np.linalg.norm(deltas[cid])),
                                "is_malicious": bool(mal_flags[cid]),
                                "accepted": bool(result.accepted[cid]) if result.accepted is not None else None,
                                "threshold": threshold,
                                "accuracy": float(acc),
                                "loss": float(loss),
                            }
                        )

                    if (not np.isfinite(loss)) or loss > DIVERGENCE_LOSS_THRESHOLD:
                        diverged_at = r

                combo_count += 1
                dt = time.perf_counter() - t0
                print(f"[{combo_count:3d}/72] {tag:16s} {attack:24s} {defense_name:14s} {dt:5.1f}s diverged_at={diverged_at}")

        _write_csv(os.path.join(OUT_DIR, "trace_A_client_scores.csv"), rows)
        print(f"  -- checkpointed after {tag}: {len(rows)} rows, elapsed={time.perf_counter()-t_start:.0f}s --")

    _write_csv(os.path.join(OUT_DIR, "trace_A_client_scores.csv"), rows)
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
