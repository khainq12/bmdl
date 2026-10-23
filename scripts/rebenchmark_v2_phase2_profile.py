"""Phase 2: profile the existing training pipeline and benchmark a candidate
optimization (GPU-resident local SGD instead of per-step numpy round-trips,
plus a persistent-on-device eval tensor cache instead of re-transferring the
eval set from host every round) on a SMALL representative workload. Does not
modify benchmark/models/local_training_fixed_steps.py or torch_cnn.py (the
locked, already-used-for-504-runs files) -- writes an alternative path here
and compares it against the original for both speed AND numerical agreement
before any adoption recommendation is made.

Usage: ZKFL_DEVICE=cuda python3 scripts/rebenchmark_v2_phase2_profile.py
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from benchmark.datasets.pathmnist import _medmnist_split, build_or_load_partitions  # noqa: E402
from benchmark.models.local_training_fixed_steps import local_training_fixed_steps  # noqa: E402
from benchmark.models.torch_cnn import PathMNISTCNN, TorchCNNFlat, resolve_device  # noqa: E402

DEVICE = resolve_device(os.environ.get("ZKFL_DEVICE", "cuda"))
K_STEPS, BATCH_SIZE, LR = 50, 128, 0.01
N_CLIENTS = 5


# ---------------------------------------------------------- GPU-native SGD ---
def local_training_gpu_native(model: TorchCNNFlat, X: np.ndarray, y: np.ndarray, k_steps: int,
                               lr: float, batch_size: int, rng: np.random.Generator) -> np.ndarray:
    """Same sampling procedure as local_training_fixed_steps (shuffle/cycle,
    exactly k_steps batches), but the weight update happens with torch
    tensors in-place on the GPU every step -- numpy/.cpu() conversion
    happens ONCE at the start (to snapshot initial_weights, already required
    for the delta) and never during the K-step loop itself."""
    m = model.model
    n = len(X)
    if n == 0:
        raise ValueError("local_training_gpu_native: empty client dataset")
    initial_weights = model.get_weights().copy()

    perm = rng.permutation(n)
    pos = 0
    m.train()
    with torch.no_grad():
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
        logits = m(xb)
        loss = F.cross_entropy(logits, yb)
        loss.backward()
        with torch.no_grad():
            for p in params:
                if p.grad is not None:
                    p -= lr * p.grad

    delta = model.get_weights() - initial_weights
    return delta


def main() -> None:
    print(f"device={DEVICE}")
    X_train, y_train, info = _medmnist_split("train")
    X_val, y_val, _ = _medmnist_split("val")
    n_classes = len(info["label"])
    rec = build_or_load_partitions(len(X_train), y_train, "iid", 0.5, N_CLIENTS, 0.10, 42)
    client_idx = [np.array(c) for c in rec["client_idx"]]
    partition_xy = [(X_train[c], y_train[c]) for c in client_idx]

    N_PROFILE_ROUNDS = 10

    # ---------------- 1. ORIGINAL path: per-step numpy get/set_weights ----------------
    model = TorchCNNFlat(n_classes=n_classes, seed=42, device=DEVICE)
    rng = np.random.Generator(np.random.PCG64(1000))
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    deltas_orig = []
    for r in range(N_PROFILE_ROUNDS):
        gw = model.get_weights().copy()
        for cid in range(N_CLIENTS):
            local = TorchCNNFlat(n_classes=n_classes, seed=42, device=DEVICE)
            local.set_weights(gw.copy())
            Xc, yc = partition_xy[cid]
            d = local_training_fixed_steps(local, Xc, yc, K_STEPS, LR, BATCH_SIZE, rng)
            if cid == 0 and r == 0:
                deltas_orig.append(d.copy())
        acc, loss = model.evaluate(X_val, y_val)
        # global model intentionally held fixed across profiling rounds (pure timing test,
        # not a training-quality test -- correctness is checked separately below on round 0)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t_orig = time.perf_counter() - t0
    print(f"ORIGINAL (numpy get/set_weights every step): {N_PROFILE_ROUNDS} rounds x {N_CLIENTS} clients = {t_orig:.2f}s "
          f"-> {t_orig/N_PROFILE_ROUNDS:.3f} s/round")

    # ---------------- 2. GPU-NATIVE path ----------------
    model2 = TorchCNNFlat(n_classes=n_classes, seed=42, device=DEVICE)
    rng2 = np.random.Generator(np.random.PCG64(1000))
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    deltas_new = []
    for r in range(N_PROFILE_ROUNDS):
        gw = model2.get_weights().copy()
        for cid in range(N_CLIENTS):
            local = TorchCNNFlat(n_classes=n_classes, seed=42, device=DEVICE)
            local.set_weights(gw.copy())
            Xc, yc = partition_xy[cid]
            d = local_training_gpu_native(local, Xc, yc, K_STEPS, LR, BATCH_SIZE, rng2)
            if cid == 0 and r == 0:
                deltas_new.append(d.copy())
        acc2, loss2 = model2.evaluate(X_val, y_val)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t_new = time.perf_counter() - t0
    print(f"GPU-NATIVE (torch in-place update, no per-step numpy): {N_PROFILE_ROUNDS} rounds x {N_CLIENTS} clients = {t_new:.2f}s "
          f"-> {t_new/N_PROFILE_ROUNDS:.3f} s/round")
    print(f"speedup: {t_orig/t_new:.2f}x")

    # ---------------- 3. correctness check: same seed/data -> same delta (within float32 rounding) ----------------
    diff = np.abs(deltas_orig[0] - deltas_new[0])
    rel = diff / (np.abs(deltas_orig[0]) + 1e-12)
    print(f"\ncorrectness check (client 0, round 0 delta, {len(deltas_orig[0])} params):")
    print(f"  max abs diff: {diff.max():.3e}  mean abs diff: {diff.mean():.3e}")
    print(f"  max rel diff: {rel.max():.3e}  mean rel diff: {rel.mean():.3e}")
    print(f"  original delta norm: {np.linalg.norm(deltas_orig[0]):.6f}  new delta norm: {np.linalg.norm(deltas_new[0]):.6f}")

    # ---------------- 4. eval tensor caching: host->device transfer cost per evaluate() call ----------------
    print("\n--- eval transfer cost (val set, {} images) ---".format(len(X_val)))
    model3 = TorchCNNFlat(n_classes=n_classes, seed=42, device=DEVICE)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        model3.evaluate(X_val, y_val)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t_eval_uncached = (time.perf_counter() - t0) / 20
    print(f"evaluate() [re-transfers X_val from host every call]: {t_eval_uncached*1000:.2f} ms/call")

    # cached version: pre-load val tensors once, run forward only
    xb_cached = model3._to_tensor(X_val)
    yb_cached = torch.from_numpy(np.asarray(y_val, dtype=np.int64)).to(model3.device)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        model3.model.eval()
        with torch.no_grad():
            logits = model3.model(xb_cached)
            _ = F.cross_entropy(logits, yb_cached).item()
            _ = (logits.argmax(dim=1) == yb_cached).float().mean().item()
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t_eval_cached = (time.perf_counter() - t0) / 20
    print(f"evaluate() [pre-cached on-device tensors]: {t_eval_cached*1000:.2f} ms/call")
    print(f"eval speedup from caching: {t_eval_uncached/t_eval_cached:.2f}x")


if __name__ == "__main__":
    main()
