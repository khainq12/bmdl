"""Fed-ISIC2019 round-budget RE-VERIFICATION under real benchmark conditions
(not clean-FedAvg-only like the original round_budget_study). Runs 3 key
defenses (fedavg, norm, c4_drift_aware) under no_attack, same 2 seeds and
same data split as results/fed_isic2019/main_benchmark/, out to 40 rounds
instead of 12, to check whether accuracy has genuinely plateaued by round 12
or is still measurably rising (as the raw per-run curves in
round_metrics.csv suggested, in contrast to the original 3-seed-averaged
clean-FedAvg-only study).

Reuses the exact same training/defense code as scripts/fedisic_benchmark.py
(not a reimplementation) -- only N_ROUNDS and the defense/attack subset
differ. Does not modify fedisic_benchmark.py or its locked combos list.

Usage: python fedisic_round_extension.py
Writes round_extension.csv (round-by-round accuracy per defense/seed) to
the current directory.
"""

from __future__ import annotations

import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/pq-zkfl-medical"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from benchmark.attacks.suite import apply_attack  # noqa: E402  (unused here, no_attack only, kept for parity)
from benchmark.datasets.fed_isic2019 import load_fed_isic2019_problem  # noqa: E402
from benchmark.defenses import make_defense  # noqa: E402
from benchmark.models.local_training_image import local_training_fixed_steps_image  # noqa: E402

ROOT = os.path.expanduser("~/pq-zkfl-medical")
CALIB_CSV = os.path.join(ROOT, "results", "fed_isic2019", "calibration_results.csv")
OUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "round_extension.csv")

SEEDS = [42, 43]
DEFENSES = ["fedavg", "norm", "c4_drift_aware"]
N_ROUNDS = 40
K_STEPS = 10
BATCH_SIZE = 32
LOCAL_LR = 0.01


def load_thresholds():
    calib = pd.read_csv(CALIB_CSV)

    def th(defense):
        row = calib[calib["defense"] == defense]
        assert len(row) == 1
        return float(row.iloc[0]["threshold"])

    return {"tau": th("norm"), "rho": th("cosine"), "kappa": th("sign_consensus")}


def run_one(problem, defense_name, thresh, seed):
    model = problem["model_factory"](pretrained=True, seed=seed)
    rng = np.random.default_rng(seed + 8000)
    defense = make_defense(defense_name)
    client_datasets = problem["client_datasets"]
    server_ref_ds, server_ref_idx = problem["server_ref"]
    test_ds = problem["test"]
    needs_gref = defense_name in ("cosine", "c4_drift_aware")

    rows = []
    for r in range(N_ROUNDS):
        t0 = time.perf_counter()
        gw = model.get_weights().copy()
        deltas = []
        for ds, idx in client_datasets:
            local = problem["model_factory"](pretrained=False, seed=seed)
            local.set_weights(gw.copy())
            d = local_training_fixed_steps_image(local, ds, idx, K_STEPS, LOCAL_LR, BATCH_SIZE, rng)
            deltas.append(d)

        ctx = {"tau": thresh["tau"], "rho": thresh["rho"], "kappa": thresh["kappa"], "krum_f": 1}
        if needs_gref:
            gref_model = problem["model_factory"](pretrained=False, seed=seed)
            gref_model.set_weights(gw.copy())
            ctx["g_ref"] = local_training_fixed_steps_image(gref_model, server_ref_ds, server_ref_idx, K_STEPS, LOCAL_LR, BATCH_SIZE, rng)

        result = defense.aggregate(deltas, ctx)
        model.set_weights(gw + result.update)
        acc, loss = model.evaluate(test_ds, batch_size=64)
        dt = time.perf_counter() - t0
        rows.append({"defense": defense_name, "seed": seed, "round": r, "accuracy": float(acc), "loss": float(loss), "runtime_s": float(dt)})
        print(f"  {defense_name} seed={seed} round={r:2d} acc={acc:.4f} loss={loss:.4f} dt={dt:.1f}s", flush=True)
    return rows


def main():
    t_start = time.perf_counter()
    thresh = load_thresholds()
    print(f"N_ROUNDS={N_ROUNDS} defenses={DEFENSES} seeds={SEEDS} thresholds={thresh}", flush=True)

    all_rows = []
    if os.path.exists(OUT_CSV):
        all_rows = pd.read_csv(OUT_CSV).to_dict("records")
        done = {(r["defense"], r["seed"]) for r in all_rows if r["round"] == N_ROUNDS - 1}
        print(f"Resuming: {len(done)} (defense,seed) combos already complete", flush=True)
    else:
        done = set()

    problems_cache = {}
    for seed in SEEDS:
        if seed not in problems_cache:
            problems_cache[seed] = load_fed_isic2019_problem(seed=seed)
        problem = problems_cache[seed]
        for defense_name in DEFENSES:
            if (defense_name, seed) in done:
                print(f"skip {defense_name} seed={seed} (already done)", flush=True)
                continue
            rows = run_one(problem, defense_name, thresh, seed)
            all_rows.extend(rows)
            with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
                w.writeheader()
                w.writerows(all_rows)
            print(f"DONE {defense_name} seed={seed}, elapsed so far {time.perf_counter()-t_start:.0f}s", flush=True)

    total = time.perf_counter() - t_start
    print(f"ALL DONE. {len(all_rows)} rows, total {total:.0f}s ({total/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
