"""FL benchmark runner — ties dataset + model + attack + defense +
calibration together for one RunConfig, per the locked protocol in
docs/BENCHMARK_PROTOCOL.md. Only the synthetic (debug) dataset is wired up
so far; PathMNIST and Fed-ISIC2019 loaders are separate implementation
passes (STEP 5/6).
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

import numpy as np

from benchmark.attacks.suite import apply_attack
from benchmark.calibration import NEEDS_CALIBRATION
from benchmark.config import RunConfig
from benchmark.datasets import DATASET_LOADERS
from benchmark.defenses import make_defense
from benchmark.defenses.cosine import cosine_similarity
from benchmark.defenses.sign_consensus import sign_consensus_scores
from benchmark.models import local_training_seeded


def _compute_server_ref_delta(problem: Dict, global_weights: np.ndarray, cfg: RunConfig, rng: np.random.Generator) -> np.ndarray:
    """g_ref: one local-SGD pass over the trusted server reference subset,
    from the current global weights. See docs/BENCHMARK_PROTOCOL.md Sec 6."""
    X_ref, y_ref = problem["server_ref"]
    model = problem["model_factory"]()
    model.set_weights(global_weights.copy())
    return local_training_seeded(model, X_ref, y_ref, cfg.optim.local_epochs, cfg.optim.local_lr, cfg.optim.batch_size, rng)


def _run_round(
    problem: Dict,
    global_weights: np.ndarray,
    cfg: RunConfig,
    attack_name: str,
    attack_params: Dict[str, Any],
    n_mal: int,
    rng: np.random.Generator,
    round_idx: int,
) -> Tuple[List[np.ndarray], List[bool]]:
    partitions = problem["partitions"]
    deltas, mal_flags = [], []
    for cid in range(len(partitions)):
        model = problem["model_factory"]()
        model.set_weights(global_weights.copy())
        X_c, y_c = partitions[cid]
        delta = local_training_seeded(model, X_c, y_c, cfg.optim.local_epochs, cfg.optim.local_lr, cfg.optim.batch_size, rng)
        is_mal = cid < n_mal and round_idx >= cfg.attack.attack_from_round
        if is_mal:
            delta = apply_attack(attack_name, delta, rng, attack_params)
        deltas.append(delta)
        mal_flags.append(is_mal)
    return deltas, mal_flags


def _partition_for_calibration(X: np.ndarray, y: np.ndarray, n_clients: int, cfg: RunConfig, seed: int):
    from fl_core.model import partition_non_iid

    if cfg.dataset.partition == "iid":
        from benchmark.datasets.synthetic import partition_iid

        return partition_iid(X, y, n_clients, seed=seed)
    return partition_non_iid(X, y, n_clients, alpha=cfg.dataset.dirichlet_alpha, seed=seed)


def _calibrate(problem: Dict, cfg: RunConfig, defense_name: str) -> Dict:
    """Validation-only calibration loop. See docs/BENCHMARK_PROTOCOL.md Sec 8.
    Uses the calib split exclusively; never touches the test split."""
    param_name, calib_fn = NEEDS_CALIBRATION[defense_name]
    rng = np.random.default_rng(cfg.calibration.calib_seed)
    X_calib, y_calib = problem["calib"]
    n_calib_clients = max(3, cfg.dataset.n_clients)
    calib_partitions = _partition_for_calibration(X_calib, y_calib, n_calib_clients, cfg, seed=cfg.calibration.calib_seed)

    model = problem["model_factory"]()
    honest_scores: List[float] = []
    malicious_scores: List[float] = []
    n_mal = max(1, int(round(cfg.attack.malicious_ratio * n_calib_clients))) if cfg.attack.malicious_ratio > 0 else max(1, n_calib_clients // 5)

    for r in range(cfg.calibration.calib_rounds):
        gw = model.get_weights().copy()
        g_ref = _compute_server_ref_delta(problem, gw, cfg, rng) if defense_name == "cosine" else None

        deltas, mal_flags = [], []
        for cid in range(n_calib_clients):
            local = problem["model_factory"]()
            local.set_weights(gw.copy())
            X_c, y_c = calib_partitions[cid]
            delta = local_training_seeded(local, X_c, y_c, cfg.optim.local_epochs, cfg.optim.local_lr, cfg.optim.batch_size, rng)
            is_mal = cid < n_mal and r >= 1  # round 0 always clean (Sec 8.1)
            if is_mal:
                delta = apply_attack(cfg.attack.name, delta, rng, cfg.attack.params)
            deltas.append(delta)
            mal_flags.append(is_mal)

        if defense_name == "sign_consensus":
            topk = cfg.defense.params.get("sign_topk")
            scores = sign_consensus_scores(deltas, sign_topk=topk)
        elif defense_name == "cosine":
            scores = np.array([cosine_similarity(d, g_ref) for d in deltas])
        else:  # norm
            scores = np.array([float(np.linalg.norm(d)) for d in deltas])

        for s, is_mal in zip(scores, mal_flags):
            (malicious_scores if is_mal else honest_scores).append(float(s))

        # Advance the calibration trajectory on honest-only deltas (Addendum
        # 2026-10-05, docs/BENCHMARK_PROTOCOL.md Sec 11) so an unmitigated
        # attack does not diverge the calibration model and contaminate
        # subsequent honest score samples with instability artifacts.
        honest_deltas = [d for d, is_mal in zip(deltas, mal_flags) if not is_mal]
        update = np.mean(honest_deltas, axis=0) if honest_deltas else np.mean(deltas, axis=0)
        model.set_weights(gw + update)

    result = calib_fn(np.array(honest_scores), np.array(malicious_scores), fpr_target=cfg.calibration.fpr_target)
    result["param_name"] = param_name
    return result


def run(cfg: RunConfig) -> Dict[str, Any]:
    if cfg.dataset.name not in DATASET_LOADERS:
        raise ValueError(
            f"dataset '{cfg.dataset.name}' is not wired into the runner yet "
            "(see docs/IMPLEMENTATION_PLAN.md STEP 5/6)"
        )
    problem = DATASET_LOADERS[cfg.dataset.name](cfg)
    defense = make_defense(cfg.defense.name)

    calibration_record = None
    thresholds: Dict[str, float] = {}
    if cfg.defense.name in NEEDS_CALIBRATION:
        calibration_record = _calibrate(problem, cfg, cfg.defense.name)
        thresholds[calibration_record["param_name"]] = calibration_record["threshold"]

    rng = np.random.default_rng(cfg.seed)
    model = problem["model_factory"]()
    n_clients = len(problem["partitions"])
    n_mal = max(0, int(round(cfg.attack.malicious_ratio * n_clients)))

    per_round: List[Dict[str, Any]] = []
    client_scores_rows: List[Dict[str, Any]] = []
    timing_rows: List[Dict[str, Any]] = []

    for r in range(cfg.optim.n_rounds):
        t0 = time.perf_counter()
        gw = model.get_weights().copy()
        deltas, mal_flags = _run_round(problem, gw, cfg, cfg.attack.name, cfg.attack.params, n_mal, rng, r)

        ctx: Dict[str, Any] = {
            "rng": rng,
            "tau": thresholds.get("tau"),
            "rho": thresholds.get("rho"),
            "kappa": thresholds.get("kappa"),
            "sign_topk": cfg.defense.params.get("sign_topk"),
            "krum_f": cfg.defense.params.get("krum_f", 1),
        }
        if cfg.defense.name == "cosine":
            ctx["g_ref"] = _compute_server_ref_delta(problem, gw, cfg, rng)

        result = defense.aggregate(deltas, ctx)
        model.set_weights(gw + result.update)

        X_test, y_test = problem["test"]
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

        per_round.append(
            {
                "round": r,
                "accuracy": float(acc),
                "loss": float(loss),
                "runtime_s": float(dt),
                "n_malicious": int(sum(mal_flags)),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
            }
        )
        timing_rows.append({"round": r, "runtime_s": float(dt)})

        if result.scores is not None and result.accepted is not None:
            for cid, (sc, mal, acc_flag) in enumerate(zip(result.scores, mal_flags, result.accepted)):
                client_scores_rows.append(
                    {
                        "round": r,
                        "client_id": cid,
                        "score": float(sc),
                        "is_malicious": bool(mal),
                        "accepted": bool(acc_flag),
                    }
                )

    return {
        "config": cfg.to_dict(),
        "calibration": calibration_record,
        "per_round": per_round,
        "client_scores": client_scores_rows,
        "timing": timing_rows,
        "final_accuracy": per_round[-1]["accuracy"] if per_round else None,
    }
