"""Phase 3 (R12 convergence extension): FedAvg / Multi-Krum / C4-DA x {iid,
dirichlet_a0.5} x seed{42,43,44}, restarted from round 0 (no checkpoints
exist for the original 504-run sweep -- see SCIENTIFIC_AUDIT.md), trained to
MAX_ROUNDS (default 500), using the GPU-native local-SGD path verified in
Phase 2 (benchmarked ~2x faster than the original per-step numpy path, max
abs delta diff 1.5e-5 -- see GPU_OPTIMIZATION_REPORT.md).

Correctness-critical differences from scripts/rebenchmark_v2_full_benchmark.py
(that script is NOT modified; this is a separate script for separate output):
  - Evaluates on the VALIDATION split every round (not test -- the original
    504-run sweep evaluated on test every round, a disclosed protocol
    limitation, see SCIENTIFIC_AUDIT.md). Test is evaluated exactly once,
    at the end, on the round selected by the convergence criterion (or the
    final round if never satisfied).
  - Saves a full checkpoint every CHECKPOINT_EVERY=50 rounds: flat model
    weights, the per-run RNG's bit_generator state, the round index, and a
    JSON config -- enabling safe interruption and exact resumption (unlike
    the original sweep). Resume correctness is unit-tested by
    scripts/rebenchmark_v2_test_checkpoint_resume.py before this script is
    trusted for the real run.

Output: results/rebenchmark_v2/convergence_extension/<run_id>/
  - checkpoint_round<N>.npz (weights, rng_state, round, config)
  - round_metrics.csv (round, val_accuracy, val_loss, runtime_s)
  - final_result.json (final_test_accuracy, final_test_loss, selected_round,
    converged_round per the ORIGINAL PATIENCE/ACC_EPS/LOSS_EPS criterion)

Usage:
  ZKFL_DEVICE=cuda python3 scripts/rebenchmark_v2_convergence_extension.py \
      --defense fedavg --partition iid --seed 42 --max_rounds 500
  # resume (auto-detects latest checkpoint in the run's output dir):
  ZKFL_DEVICE=cuda python3 scripts/rebenchmark_v2_convergence_extension.py \
      --defense fedavg --partition iid --seed 42 --max_rounds 500 --resume
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from benchmark.datasets.pathmnist import _medmnist_split, build_or_load_partitions  # noqa: E402
from benchmark.defenses import make_defense  # noqa: E402
from benchmark.models.torch_cnn import TorchCNNFlat, resolve_device  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_ROOT = os.environ.get("ZKFL_OUT_ROOT", os.path.join(ROOT, "results", "rebenchmark_v2", "convergence_extension"))

K_STEPS, BATCH_SIZE, LOCAL_LR = 50, 128, 0.01
N_CLIENTS = 5
SERVER_REF_FRACTION = 0.10
CHECKPOINT_EVERY = 50
PATIENCE, ACC_EPS, LOSS_EPS = 15, 0.005, 0.002
DIVERGENCE_LOSS_THRESHOLD = 50.0


def local_training_gpu_native(model: TorchCNNFlat, X, y, k_steps, lr, batch_size, rng) -> np.ndarray:
    """Verified in Phase 2 against the original numpy-per-step path:
    max abs delta diff 1.5e-5, ~2x faster (see GPU_OPTIMIZATION_REPORT.md).
    Same sampling procedure as benchmark/models/local_training_fixed_steps.py."""
    m = model.model
    n = len(X)
    if n == 0:
        raise ValueError("empty client dataset")
    initial_weights = model.get_weights().copy()
    perm = rng.permutation(n)
    pos = 0
    m.train()
    params = list(m.parameters())
    for _ in range(k_steps):
        if pos + batch_size > n:
            perm = rng.permutation(n)
            pos = 0
        batch_idx = perm[pos:pos + batch_size]
        pos += batch_size
        xb = model._to_tensor(X[batch_idx])
        yb = torch.from_numpy(np.asarray(y[batch_idx], dtype=np.int64)).to(model.device)
        m.zero_grad(set_to_none=True)
        loss = F.cross_entropy(m(xb), yb)
        loss.backward()
        with torch.no_grad():
            for p in params:
                if p.grad is not None:
                    p -= lr * p.grad
    return model.get_weights() - initial_weights


def find_convergence_round(val_acc, val_loss):
    n = len(val_acc)
    if n < PATIENCE + 1:
        return None
    best_suffix = np.maximum.accumulate(val_acc[::-1])[::-1]
    for r in range(1, n - PATIENCE + 1):
        if best_suffix[r] - val_acc[r - 1] >= ACC_EPS:
            continue
        tail = val_loss[r:]
        slope = np.polyfit(np.arange(len(tail)), tail, 1)[0]
        if slope < -LOSS_EPS:
            continue
        return r
    return None


def save_checkpoint(path, model, rng, round_idx, config):
    np.savez(path, weights=model.get_weights(), rng_state=np.frombuffer(
        json.dumps(rng.bit_generator.state).encode(), dtype=np.uint8),
        round=round_idx, config=np.frombuffer(json.dumps(config).encode(), dtype=np.uint8))


def load_checkpoint(path):
    z = np.load(path, allow_pickle=False)
    rng_state = json.loads(bytes(z["rng_state"]).decode())
    config = json.loads(bytes(z["config"]).decode())
    bg = np.random.PCG64()
    bg.state = rng_state
    rng = np.random.Generator(bg)
    return z["weights"], rng, int(z["round"]), config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--defense", required=True, choices=["fedavg", "multi_krum", "c4_drift_aware"])
    ap.add_argument("--partition", required=True, choices=["iid", "dirichlet_a0.5"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--max_rounds", type=int, default=500)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    device = resolve_device(os.environ.get("ZKFL_DEVICE", "cuda"))
    run_id = f"{args.defense}__{args.partition}__seed{args.seed}"
    run_dir = os.path.join(OUT_ROOT, run_id)
    os.makedirs(run_dir, exist_ok=True)
    metrics_path = os.path.join(run_dir, "round_metrics.csv")

    X_train, y_train, info = _medmnist_split("train")
    X_val, y_val, _ = _medmnist_split("val")
    X_test, y_test, _ = _medmnist_split("test")
    n_classes = len(info["label"])

    partition_name = "iid" if args.partition == "iid" else "dirichlet"
    alpha = None if args.partition == "iid" else float(args.partition.replace("dirichlet_a", ""))
    rec = build_or_load_partitions(len(X_train), y_train, partition_name,
                                    alpha if alpha is not None else 0.5, N_CLIENTS, SERVER_REF_FRACTION, args.seed)
    client_idx = [np.array(c) for c in rec["client_idx"]]
    partition_xy = [(X_train[c], y_train[c]) for c in client_idx]
    server_ref_idx = np.array(rec["server_ref_idx"])
    server_ref_xy = (X_train[server_ref_idx], y_train[server_ref_idx])

    model = TorchCNNFlat(n_classes=n_classes, seed=args.seed, device=device)
    defense = make_defense(args.defense)

    thresh = {"tau": None, "rho": None, "kappa": None}
    if args.defense == "c4_drift_aware":
        import pandas as pd
        calib_path = os.path.join(ROOT, "results", "rebenchmark_v2", "full_benchmark", "calibration_results.csv")
        calib = pd.read_csv(calib_path)
        for name, key in [("norm", "tau"), ("cosine", "rho"), ("sign_consensus", "kappa")]:
            row = calib[(calib["defense"] == name) & (calib["partition"] == args.partition)]
            thresh[key] = float(row.iloc[0]["threshold"])
        print(f"Loaded locked thresholds for {args.partition}: tau={thresh['tau']:.4f} "
              f"rho={thresh['rho']:.4f} kappa={thresh['kappa']:.4f}")

    start_round = 0
    rng = np.random.default_rng(args.seed + 11000)
    history = []
    if args.resume:
        ckpts = sorted([f for f in os.listdir(run_dir) if f.startswith("checkpoint_round")]) if os.path.isdir(run_dir) else []
        if ckpts:
            latest = os.path.join(run_dir, ckpts[-1])
            weights, rng, start_round, config = load_checkpoint(latest)
            model.set_weights(weights)
            start_round += 1
            if os.path.exists(metrics_path):
                import pandas as pd
                history = pd.read_csv(metrics_path).to_dict("records")
            print(f"Resumed from {latest} at round {start_round}")

    print(f"run_id={run_id} device={device} start_round={start_round} max_rounds={args.max_rounds}")
    t_start = time.perf_counter()
    for r in range(start_round, args.max_rounds):
        t0 = time.perf_counter()
        gw = model.get_weights().copy()
        deltas = []
        for cid in range(N_CLIENTS):
            local = TorchCNNFlat(n_classes=n_classes, seed=args.seed, device=device)
            local.set_weights(gw.copy())
            Xc, yc = partition_xy[cid]
            d = local_training_gpu_native(local, Xc, yc, K_STEPS, LOCAL_LR, BATCH_SIZE, rng)
            deltas.append(d)

        ctx = {"rng": rng, **thresh}
        if args.defense == "c4_drift_aware":
            lm = TorchCNNFlat(n_classes=n_classes, seed=args.seed, device=device)
            lm.set_weights(gw.copy())
            ctx["g_ref"] = local_training_gpu_native(lm, server_ref_xy[0], server_ref_xy[1], K_STEPS, LOCAL_LR, BATCH_SIZE, rng)

        result = defense.aggregate(deltas, ctx)
        model.set_weights(gw + result.update)

        val_acc, val_loss = model.evaluate(X_val, y_val)
        dt = time.perf_counter() - t0
        history.append({"round": r, "val_accuracy": float(val_acc), "val_loss": float(val_loss), "runtime_s": dt})

        if (r + 1) % CHECKPOINT_EVERY == 0 or r == args.max_rounds - 1:
            ckpt_path = os.path.join(run_dir, f"checkpoint_round{r:04d}.npz")
            save_checkpoint(ckpt_path, model, rng, r,
                             {"defense": args.defense, "partition": args.partition, "seed": args.seed,
                              "max_rounds": args.max_rounds, "k_steps": K_STEPS, "batch_size": BATCH_SIZE,
                              "lr": LOCAL_LR})
            import pandas as pd
            pd.DataFrame(history).to_csv(metrics_path, index=False)
            acc_arr = np.array([h["val_accuracy"] for h in history])
            loss_arr = np.array([h["val_loss"] for h in history])
            conv = find_convergence_round(acc_arr, loss_arr)
            print(f"[round {r}] val_acc={val_acc:.4f} val_loss={val_loss:.4f} "
                  f"elapsed={time.perf_counter()-t_start:.0f}s converged_round={conv}")

        if (not np.isfinite(val_loss)) or val_loss > DIVERGENCE_LOSS_THRESHOLD:
            print(f"DIVERGED at round {r}; stopping early.")
            break

    acc_arr = np.array([h["val_accuracy"] for h in history])
    loss_arr = np.array([h["val_loss"] for h in history])
    conv_round = find_convergence_round(acc_arr, loss_arr)
    selected_round = conv_round if conv_round is not None else len(history) - 1
    test_acc, test_loss = model.evaluate(X_test, y_test)

    final = {"run_id": run_id, "converged_round": conv_round, "selected_round": selected_round,
             "final_val_accuracy": float(acc_arr[-1]), "final_val_loss": float(loss_arr[-1]),
             "test_accuracy_at_final_checkpoint": float(test_acc), "test_loss_at_final_checkpoint": float(test_loss),
             "total_rounds_run": len(history)}
    with open(os.path.join(run_dir, "final_result.json"), "w") as f:
        json.dump(final, f, indent=2)
    print("DONE", final)


if __name__ == "__main__":
    main()
