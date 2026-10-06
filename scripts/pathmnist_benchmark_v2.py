"""PathMNIST single-method benchmark, v2 — corrected per
docs/BENCHMARK_PROTOCOL.md Addendum 2026-10-07 (g)/(h): fixed-K local
training (no more step-count confound), round budget selected empirically
(scripts/pathmnist_round_budget_study.py), calibration diagnostics gate
passed (scripts/pathmnist_calibration_diagnostics.py) before this runs.

Same experimental grid as the (now-invalid) v1 sweep: 6 defenses x 6 attack
conditions x 4 partitions x 3 seeds, both calibration regimes where
applicable. Combined Attestation is NOT implemented. Results go to a
SEPARATE directory from the preserved v1 diagnostic run.

Usage:
  python scripts/pathmnist_benchmark_v2.py --pilot   # the 4 required pilot conditions
  python scripts/pathmnist_benchmark_v2.py           # full corrected matrix

N_ROUNDS below is set from the round-budget study's selection (see
results/pathmnist/round_budget_study/SELECTION.md) — not a placeholder.
"""

from __future__ import annotations

import csv
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from benchmark.attacks.suite import apply_attack  # noqa: E402
from benchmark.calibration import calibrate_kappa, calibrate_rho, calibrate_tau  # noqa: E402
from benchmark.datasets.pathmnist import (  # noqa: E402
    _medmnist_split,
    build_or_load_partitions,
    partition_dirichlet_indices,
    partition_iid_indices,
)
from benchmark.defenses import make_defense  # noqa: E402
from benchmark.defenses.cosine import cosine_similarity  # noqa: E402
from benchmark.defenses.sign_consensus import sign_consensus_scores  # noqa: E402
from benchmark.models import local_training_fixed_steps  # noqa: E402
from benchmark.models.torch_cnn import TorchCNNFlat  # noqa: E402

PILOT = "--pilot" in sys.argv
OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results",
    "pathmnist",
    "pilot_v2" if PILOT else "benchmark_v2_corrected",
)

N_CLIENTS = 5
SEEDS = [42] if PILOT else [42, 43, 44]

# Pilot = the exact 4 required conditions (docs/BENCHMARK_PROTOCOL.md Addendum (g) gate,
# step 5 of the fix request): IID/no_attack, IID/large_norm, a0.1/no_attack, a0.1/full_sign_flip.
if PILOT:
    _CONDITIONS = [
        ("iid", None, "no_attack"),
        ("iid", None, "large_norm"),
        ("dirichlet", 0.1, "no_attack"),
        ("dirichlet", 0.1, "full_sign_flip"),
    ]
    PARTITIONS = sorted(set((p, a) for p, a, _ in _CONDITIONS), key=lambda x: str(x))
    ATTACKS = sorted(set(atk for _, _, atk in _CONDITIONS))
    DEFENSES = ["fedavg", "norm", "cosine", "sign_consensus"]
else:
    _CONDITIONS = None
    PARTITIONS = [("iid", None), ("dirichlet", 1.0), ("dirichlet", 0.5), ("dirichlet", 0.1)]
    ATTACKS = [
        "no_attack",
        "large_norm",
        "low_norm",
        "full_sign_flip",
        "directional_poisoning",
        "sparse_coordinate_attack",
    ]
    DEFENSES = ["fedavg", "norm", "cosine", "sign_consensus", "median", "multi_krum"]

DETECTOR_DEFENSES = {"norm", "cosine", "sign_consensus"}

# ---- Locked (Addendum (g)/(h)) ----
K_STEPS = 50
BATCH_SIZE = 128
LOCAL_LR = 0.01
# N_ROUNDS is set from the round-budget study's selection; see
# results/pathmnist/round_budget_study/SELECTION.md for the rationale.
N_ROUNDS = int(os.environ.get("ZKFL_N_ROUNDS", "0"))
ATTACK_FROM_ROUND = int(os.environ.get("ZKFL_ATTACK_FROM_ROUND", "0"))

MALICIOUS_RATIO = 0.2
SERVER_REF_FRACTION = 0.10
DEVICE = os.environ.get("ZKFL_DEVICE", "cuda")

CALIB_ATTACK = "large_norm"
CALIB_ROUNDS = 5
CALIB_SEED = 1000
FPR_TARGET = 0.05
N_CALIB_CLIENTS = max(3, N_CLIENTS)

CALIB_FN = {"norm": calibrate_tau, "cosine": calibrate_rho, "sign_consensus": calibrate_kappa}
THRESH_KEY = {"norm": "tau", "cosine": "rho", "sign_consensus": "kappa"}


def _tag(partition: str, alpha) -> str:
    return "iid" if partition == "iid" else f"dirichlet_a{alpha}"


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        open(path, "w", encoding="utf-8").close()
        return
    keys: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _model(n_classes: int, seed: int) -> TorchCNNFlat:
    return TorchCNNFlat(n_classes=n_classes, seed=seed, device=DEVICE)


def _train(model, X, y, rng) -> np.ndarray:
    return local_training_fixed_steps(model, X, y, K_STEPS, LOCAL_LR, BATCH_SIZE, rng)


def calibrate_for_condition(n_classes, defense_name, partition, alpha, X_val, y_val, server_ref_xy):
    rng = np.random.default_rng(CALIB_SEED)
    if partition == "iid":
        calib_idx = partition_iid_indices(len(y_val), N_CALIB_CLIENTS, seed=CALIB_SEED)
    else:
        calib_idx = partition_dirichlet_indices(len(y_val), y_val, N_CALIB_CLIENTS, alpha, seed=CALIB_SEED)
    calib_partitions = [(X_val[c], y_val[c]) for c in calib_idx]

    model = _model(n_classes, CALIB_SEED)
    n_mal = max(1, int(round(MALICIOUS_RATIO * N_CALIB_CLIENTS)))
    honest_scores: List[float] = []
    malicious_scores: List[float] = []

    for r in range(CALIB_ROUNDS):
        gw = model.get_weights().copy()
        g_ref = None
        if defense_name == "cosine":
            Xr, yr = server_ref_xy
            m = _model(n_classes, CALIB_SEED)
            m.set_weights(gw.copy())
            g_ref = _train(m, Xr, yr, rng)

        deltas, mal_flags = [], []
        for cid in range(N_CALIB_CLIENTS):
            local = _model(n_classes, CALIB_SEED)
            local.set_weights(gw.copy())
            Xc, yc = calib_partitions[cid]
            d = _train(local, Xc, yc, rng)
            is_mal = cid < n_mal and r >= 1
            if is_mal:
                d = apply_attack(CALIB_ATTACK, d, rng, {})
            deltas.append(d)
            mal_flags.append(is_mal)

        if defense_name == "norm":
            scores = [float(np.linalg.norm(d)) for d in deltas]
        elif defense_name == "cosine":
            scores = [cosine_similarity(d, g_ref) for d in deltas]
        else:
            scores = sign_consensus_scores(deltas).tolist()

        for s, m_ in zip(scores, mal_flags):
            (malicious_scores if m_ else honest_scores).append(float(s))

        honest_deltas = [d for d, m_ in zip(deltas, mal_flags) if not m_]
        update = np.mean(honest_deltas, axis=0) if honest_deltas else np.mean(deltas, axis=0)
        model.set_weights(gw + update)

    result = CALIB_FN[defense_name](np.array(honest_scores), np.array(malicious_scores), fpr_target=FPR_TARGET)
    result["param"] = THRESH_KEY[defense_name]
    return result


DIVERGENCE_LOSS_THRESHOLD = 50.0  # sane 9-class cross-entropy is ~0-3; found via pilot (Addendum (g)/(h) step 5)


def run_combo(n_classes, partition_xy, server_ref_xy, X_test, y_test, attack_name, defense_name, threshold_ctx, seed, malicious_client_ids):
    """Stops local training early once the model has numerically diverged
    (found during the pilot: an undefended/mis-detected large_norm attack
    can destroy the model in one round, after which further K-step training
    on NaN/huge weights is wasted compute and produces meaningless huge
    -loss rows). Remaining rounds are copy-forward of the diverged state,
    flagged `diverged=True`, so every combo still yields exactly N_ROUNDS
    rows (consistent schema for aggregation) without paying for them."""
    model = _model(n_classes, seed)
    rng = np.random.default_rng(seed + 7000)
    defense = make_defense(defense_name)
    n_clients = len(partition_xy)
    rows = []
    diverged_row: Optional[Dict[str, Any]] = None

    for r in range(N_ROUNDS):
        if diverged_row is not None:
            row = dict(diverged_row)
            row["round"] = r
            rows.append(row)
            continue

        t0 = time.perf_counter()
        gw = model.get_weights().copy()
        deltas, mal_flags = [], []
        for cid in range(n_clients):
            local = _model(n_classes, seed)
            local.set_weights(gw.copy())
            Xc, yc = partition_xy[cid]
            d = _train(local, Xc, yc, rng)
            is_mal = cid in malicious_client_ids and r >= ATTACK_FROM_ROUND
            if is_mal:
                d = apply_attack(attack_name, d, rng, {})
            deltas.append(d)
            mal_flags.append(is_mal)

        ctx = dict(threshold_ctx)
        ctx["rng"] = rng
        if defense_name == "cosine":
            Xr, yr = server_ref_xy
            m = _model(n_classes, seed)
            m.set_weights(gw.copy())
            ctx["g_ref"] = _train(m, Xr, yr, rng)

        result = defense.aggregate(deltas, ctx)
        model.set_weights(gw + result.update)

        acc, loss = model.evaluate(X_test, y_test)
        dt = time.perf_counter() - t0

        tp = fp = tn = fn = None
        if result.is_detector and result.accepted is not None:
            tp = fp = tn = fn = 0
            for is_mal, acc_flag in zip(mal_flags, result.accepted):
                if is_mal and not acc_flag:
                    tp += 1
                elif (not is_mal) and (not acc_flag):
                    fp += 1
                elif (not is_mal) and acc_flag:
                    tn += 1
                else:
                    fn += 1

        row = {
            "round": r,
            "accuracy": float(acc),
            "loss": float(loss),
            "runtime_s": float(dt),
            "n_malicious": int(sum(mal_flags)),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "diverged": False,
        }
        rows.append(row)

        if (not np.isfinite(loss)) or loss > DIVERGENCE_LOSS_THRESHOLD:
            diverged_row = dict(row)
            diverged_row["diverged"] = True

    return rows


def main() -> None:
    if N_ROUNDS <= 0:
        raise SystemExit(
            "N_ROUNDS not set. Pass ZKFL_N_ROUNDS=<budget> ZKFL_ATTACK_FROM_ROUND=<round> "
            "in the environment, per the frozen selection in "
            "results/pathmnist/round_budget_study/SELECTION.md"
        )

    os.makedirs(OUT_DIR, exist_ok=True)
    t_start = time.perf_counter()
    print(f"PILOT={PILOT} N_ROUNDS={N_ROUNDS} ATTACK_FROM_ROUND={ATTACK_FROM_ROUND} K={K_STEPS} batch_size={BATCH_SIZE}")
    print(f"seeds={SEEDS} partitions={[_tag(*p) for p in PARTITIONS]} attacks={ATTACKS} defenses={DEFENSES}")

    print("Loading official PathMNIST splits ...")
    X_train, y_train, info = _medmnist_split("train")
    X_val, y_val, _ = _medmnist_split("val")
    X_test, y_test, _ = _medmnist_split("test")
    n_classes = len(info["label"])
    print(f"train={len(y_train)} val={len(y_val)} test={len(y_test)} n_classes={n_classes} device={DEVICE}")

    regimeA: Dict[Tuple[str, str, Any], Dict[str, Any]] = {}
    calib_rows: List[Dict[str, Any]] = []
    for defense_name in DEFENSES:
        if defense_name not in DETECTOR_DEFENSES:
            continue
        for partition, alpha in PARTITIONS:
            rec = build_or_load_partitions(len(X_train), y_train, partition, alpha if alpha is not None else 0.5, N_CLIENTS, SERVER_REF_FRACTION, CALIB_SEED)
            server_ref_idx = np.array(rec["server_ref_idx"])
            server_ref_xy = (X_train[server_ref_idx], y_train[server_ref_idx])
            result = calibrate_for_condition(n_classes, defense_name, partition, alpha, X_val, y_val, server_ref_xy)
            result.update(calibration_regime="A_condition_specific_oracle", defense=defense_name, partition=_tag(partition, alpha))
            regimeA[(defense_name, partition, alpha)] = result
            calib_rows.append(dict(result))
            print(f"[calib A] {defense_name:14s} {_tag(partition,alpha):16s} threshold={result['threshold']:.4f} val_fpr={result['validation_fpr']:.3f} val_tpr={result['validation_tpr']:.3f}")

    regimeB: Dict[str, Dict[str, Any]] = {}
    iid_rec = build_or_load_partitions(len(X_train), y_train, "iid", 0.5, N_CLIENTS, SERVER_REF_FRACTION, CALIB_SEED)
    iid_ref_idx = np.array(iid_rec["server_ref_idx"])
    iid_server_ref_xy = (X_train[iid_ref_idx], y_train[iid_ref_idx])
    for defense_name in DEFENSES:
        if defense_name not in DETECTOR_DEFENSES:
            continue
        result = calibrate_for_condition(n_classes, defense_name, "iid", None, X_val, y_val, iid_server_ref_xy)
        result.update(calibration_regime="B_iid_transfer_frozen", defense=defense_name, partition="iid_reference_frozen")
        regimeB[defense_name] = result
        calib_rows.append(dict(result))
        print(f"[calib B] {defense_name:14s} iid_reference   threshold={result['threshold']:.4f} val_fpr={result['validation_fpr']:.3f} val_tpr={result['validation_tpr']:.3f}")

    _write_csv(os.path.join(OUT_DIR, "calibration_results.csv"), calib_rows)
    print(f"Calibration done in {time.perf_counter()-t_start:.1f}s.")

    all_rows: List[Dict[str, Any]] = []
    n_mal = max(0, int(round(MALICIOUS_RATIO * N_CLIENTS)))
    malicious_client_ids = set(range(n_mal))
    combo_count = 0

    for partition, alpha in PARTITIONS:
        tag = _tag(partition, alpha)
        for seed in SEEDS:
            rec = build_or_load_partitions(len(X_train), y_train, partition, alpha if alpha is not None else 0.5, N_CLIENTS, SERVER_REF_FRACTION, seed)
            server_ref_idx = np.array(rec["server_ref_idx"])
            client_idx = [np.array(c) for c in rec["client_idx"]]
            server_ref_xy = (X_train[server_ref_idx], y_train[server_ref_idx])
            partition_xy = [(X_train[c], y_train[c]) for c in client_idx]

            for attack in ATTACKS:
                if PILOT and (tag, attack) not in {(_tag(p, a), atk) for p, a, atk in _CONDITIONS}:
                    continue
                for defense_name in DEFENSES:
                    regimes: List[Tuple[str, Optional[Dict[str, Any]]]] = []
                    if defense_name in DETECTOR_DEFENSES:
                        regimes.append(("A_condition_specific_oracle", regimeA[(defense_name, partition, alpha)]))
                        if partition != "iid":
                            regimes.append(("B_iid_transfer_frozen", regimeB[defense_name]))
                    else:
                        regimes.append(("not_applicable", None))

                    for regime_label, calib_result in regimes:
                        threshold_ctx = {"tau": None, "rho": None, "kappa": None, "sign_topk": None, "krum_f": 1}
                        if calib_result is not None:
                            threshold_ctx[calib_result["param"]] = calib_result["threshold"]

                        t0 = time.perf_counter()
                        rows = run_combo(n_classes, partition_xy, server_ref_xy, X_test, y_test, attack, defense_name, threshold_ctx, seed, malicious_client_ids)
                        dt = time.perf_counter() - t0
                        combo_count += 1

                        for row in rows:
                            row.update(
                                partition=tag, seed=seed, attack=attack, defense=defense_name,
                                calibration_regime=regime_label,
                                threshold_used=threshold_ctx.get(THRESH_KEY.get(defense_name, ""), None),
                            )
                            all_rows.append(row)

                        print(f"[{combo_count:4d}] {tag:16s} seed={seed} attack={attack:24s} defense={defense_name:14s} regime={regime_label:28s} {dt:5.1f}s final_acc={rows[-1]['accuracy']:.3f}")

            _write_csv(os.path.join(OUT_DIR, "per_round_results.csv"), all_rows)
            print(f"  -- checkpointed after {tag} seed={seed}: {len(all_rows)} rows so far, elapsed={time.perf_counter()-t_start:.0f}s --")

    _write_csv(os.path.join(OUT_DIR, "per_round_results.csv"), all_rows)
    total = time.perf_counter() - t_start
    print(f"DONE. {combo_count} combos, {len(all_rows)} rows, total time {total:.0f}s ({total/60:.1f} min)")


if __name__ == "__main__":
    main()
