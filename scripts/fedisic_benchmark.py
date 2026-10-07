"""Fed-ISIC2019 cross-dataset validation benchmark. Locked matrix (per
ChatGPT cross-check): 2 seeds x {C4-DA v1 (locked, unmodified), FedAvg,
Norm, Multi-Krum} x 4 conditions (no_attack, large_norm,
directional_poisoning, sparse_coordinate_attack) + Cosine x 1 condition
(directional_poisoning only, as the direct single-signal baseline for
C4-DA's directional component) = 34 runs. No ablation here (PathMNIST
already answered the mechanism/necessity question) -- this stage answers
only: does the PathMNIST-locked C4-DA v1 design hold up on 6 natural
medical centers? calibration_results.csv (τ/ρ/κ) was locked BEFORE this
script ever ran, from scripts/fedisic_calibration.py, using only the
canonical large_norm calibration attack -- never retuned against these
results.

benchmark/defenses/combined.py is NOT modified by this script or as a
result of anything it produces.

Usage: python scripts/fedisic_benchmark.py
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
from benchmark.datasets.fed_isic2019 import load_fed_isic2019_problem  # noqa: E402
from benchmark.defenses import make_defense  # noqa: E402
from benchmark.models.local_training_image import local_training_fixed_steps_image  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "results", "fed_isic2019", "main_benchmark")
CALIB_CSV = os.path.join(ROOT, "results", "fed_isic2019", "calibration_results.csv")

SEEDS = [42, 43]
CONDITIONS = ["no_attack", "large_norm", "directional_poisoning", "sparse_coordinate_attack"]
MAIN_DEFENSES = ["c4_drift_aware", "fedavg", "norm", "multi_krum"]
COSINE_ONLY_CONDITION = "directional_poisoning"

N_ROUNDS = 12
ATTACK_FROM_ROUND = 5
K_STEPS = 10
BATCH_SIZE = 32
LOCAL_LR = 0.01
MALICIOUS_CENTER = 1  # fixed, pre-declared (~0.2 of 6 centers = 1); center 0 excluded (server-ref role)
DIVERGENCE_LOSS_THRESHOLD = 50.0

DETECTOR_DEFENSES = {"norm", "cosine", "sign_consensus", "c3_static", "c4_static", "c4_drift_aware",
                     "c4_drift_aware_no_norm", "c4_drift_aware_no_cosine", "c4_drift_aware_no_sign"}


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        if not os.path.exists(path):
            open(path, "w", encoding="utf-8").close()
        return
    keys: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k); keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def load_thresholds():
    calib = pd.read_csv(CALIB_CSV)

    def th(defense):
        row = calib[calib["defense"] == defense]
        assert len(row) == 1
        return float(row.iloc[0]["threshold"])

    return {"tau": th("norm"), "rho": th("cosine"), "kappa": th("sign_consensus")}


def run_one(problem, attack_name, defense_name, thresh, seed):
    model = problem["model_factory"](pretrained=True, seed=seed)
    rng = np.random.default_rng(seed + 8000)
    defense = make_defense(defense_name)
    client_datasets = problem["client_datasets"]
    server_ref_ds, server_ref_idx = problem["server_ref"]
    test_ds = problem["test"]
    n_centers = len(client_datasets)
    needs_gref = defense_name in ("cosine", "c4_drift_aware")

    round_rows, client_rows = [], []
    diverged_at = None
    last_state = None

    for r in range(N_ROUNDS):
        if diverged_at is not None:
            rr = dict(last_state["round_row"]); rr["round"] = r; rr["diverged"] = True
            round_rows.append(rr)
            for crow in last_state["client_rows"]:
                crow2 = dict(crow); crow2["round"] = r; client_rows.append(crow2)
            continue

        t0 = time.perf_counter()
        gw = model.get_weights().copy()
        deltas, mal_role, attack_active_flags, delta_modified_flags = [], [], [], []
        for cid, (ds, idx) in enumerate(client_datasets):
            local = problem["model_factory"](pretrained=False, seed=seed)
            local.set_weights(gw.copy())
            d = local_training_fixed_steps_image(local, ds, idx, K_STEPS, LOCAL_LR, BATCH_SIZE, rng)
            role = cid == MALICIOUS_CENTER
            active = role and r >= ATTACK_FROM_ROUND
            modified = active and attack_name != "no_attack"
            if active:
                d = apply_attack(attack_name, d, rng, {})
            deltas.append(d); mal_role.append(role); attack_active_flags.append(active); delta_modified_flags.append(modified)

        ctx = {"tau": thresh["tau"], "rho": thresh["rho"], "kappa": thresh["kappa"], "krum_f": 1}
        if needs_gref:
            gref_model = problem["model_factory"](pretrained=False, seed=seed)
            gref_model.set_weights(gw.copy())
            ctx["g_ref"] = local_training_fixed_steps_image(gref_model, server_ref_ds, server_ref_idx, K_STEPS, LOCAL_LR, BATCH_SIZE, rng)

        result = defense.aggregate(deltas, ctx)
        model.set_weights(gw + result.update)

        acc, loss = model.evaluate(test_ds, batch_size=64)
        dt = time.perf_counter() - t0
        n_accepted = int(np.sum(result.accepted)) if result.accepted is not None else n_centers

        round_row = {"round": r, "accuracy": float(acc), "loss": float(loss), "runtime_s": float(dt),
                     "n_accepted": n_accepted, "n_rejected": n_centers - n_accepted,
                     "n_malicious_role": int(sum(mal_role)), "n_attack_active": int(sum(attack_active_flags)), "diverged": False}
        round_rows.append(round_row)

        crows_this_round = []
        for cid in range(n_centers):
            accepted_val = bool(result.accepted[cid]) if result.accepted is not None else None
            crow = {"round": r, "center_id": cid, "malicious_role": bool(mal_role[cid]),
                    "attack_active": bool(attack_active_flags[cid]), "delta_modified": bool(delta_modified_flags[cid]),
                    "accepted": accepted_val}
            if result.extra is not None:
                for k, v in result.extra.items():
                    crow[k] = float(v[cid]) if hasattr(v[cid], "item") else v[cid]
            elif result.scores is not None:
                crow["score"] = float(result.scores[cid])
            client_rows.append(crow); crows_this_round.append(crow)

        last_state = {"round_row": round_row, "client_rows": crows_this_round}

        if (not np.isfinite(loss)) or loss > DIVERGENCE_LOSS_THRESHOLD:
            diverged_at = r

    return round_rows, client_rows


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    t_start = time.perf_counter()
    thresh = load_thresholds()
    print(f"thresholds: {thresh}")
    print(f"N_ROUNDS={N_ROUNDS} ATTACK_FROM_ROUND={ATTACK_FROM_ROUND} K_STEPS={K_STEPS} seeds={SEEDS}")

    combos = []
    for seed in SEEDS:
        for attack in CONDITIONS:
            for defense_name in MAIN_DEFENSES:
                combos.append((seed, attack, defense_name))
        combos.append((seed, COSINE_ONLY_CONDITION, "cosine"))
    print(f"Total planned runs: {len(combos)}")

    reuse_ids = set()
    manifest_rows, all_round_rows, all_client_rows = [], [], []
    manifest_path = os.path.join(OUT_DIR, "run_manifest.csv")
    if os.path.exists(manifest_path):
        om = pd.read_csv(manifest_path)
        base_cols = ["run_id", "seed", "attack", "defense", "n_rounds", "diverged", "runtime_s", "final_accuracy", "final_loss"]
        manifest_rows = om[[c for c in base_cols if c in om.columns]].to_dict("records")
        reuse_ids = set(om["run_id"])
        for fname, store in [("round_metrics.csv", all_round_rows), ("client_decisions.csv", all_client_rows)]:
            p = os.path.join(OUT_DIR, fname)
            if os.path.exists(p):
                store.extend(pd.read_csv(p).to_dict("records"))
        print(f"Resuming: {len(reuse_ids)} runs already present")

    combo_count = len(reuse_ids)
    problems_cache = {}

    for seed, attack, defense_name in combos:
        run_id = f"{defense_name}__{attack}__seed{seed}"
        if run_id in reuse_ids:
            continue
        if seed not in problems_cache:
            problems_cache[seed] = load_fed_isic2019_problem(seed=seed)
        problem = problems_cache[seed]

        t0 = time.perf_counter()
        round_rows, client_rows = run_one(problem, attack, defense_name, thresh, seed)
        dt = time.perf_counter() - t0
        combo_count += 1
        diverged = any(rr["diverged"] for rr in round_rows)
        manifest_rows.append({"run_id": run_id, "seed": seed, "attack": attack, "defense": defense_name,
                               "n_rounds": N_ROUNDS, "diverged": diverged, "runtime_s": dt,
                               "final_accuracy": round_rows[-1]["accuracy"], "final_loss": round_rows[-1]["loss"]})
        for rr in round_rows:
            rr2 = dict(rr); rr2.update(run_id=run_id, seed=seed, attack=attack, defense=defense_name); all_round_rows.append(rr2)
        for cr in client_rows:
            cr2 = dict(cr); cr2.update(run_id=run_id, seed=seed, attack=attack, defense=defense_name); all_client_rows.append(cr2)
        print(f"[{combo_count:3d}/{len(combos)}] {run_id:45s} {dt:6.1f}s final_acc={round_rows[-1]['accuracy']:.3f} diverged={diverged}")

        _write_csv(manifest_path, manifest_rows)
        _write_csv(os.path.join(OUT_DIR, "round_metrics.csv"), all_round_rows)
        _write_csv(os.path.join(OUT_DIR, "client_decisions.csv"), all_client_rows)

    total = time.perf_counter() - t_start
    print(f"DONE. {combo_count} total runs, elapsed {total:.0f}s ({total/60:.1f} min)")


if __name__ == "__main__":
    main()
