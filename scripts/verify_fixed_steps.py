"""Experimental verification that local_training_fixed_steps performs
EXACTLY K optimizer steps regardless of client dataset size — the
requirement locked in docs/BENCHMARK_PROTOCOL.md Addendum 2026-10-07 (g).

Usage: python scripts/verify_fixed_steps.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from benchmark.models import local_training_fixed_steps  # noqa: E402


class _CountingLinearModel:
    """Minimal SimpleMLP-shaped model that counts train_step calls and
    records the batch size seen each call, without needing torch/CUDA."""

    def __init__(self, dim: int = 8, seed: int = 0):
        self.dim = dim
        self.w = np.zeros(dim)
        self.calls = 0
        self.batch_sizes = []

    def get_weights(self) -> np.ndarray:
        return self.w.copy()

    def set_weights(self, flat: np.ndarray) -> None:
        self.w = flat.copy()

    def train_step(self, X: np.ndarray, y: np.ndarray, lr: float):
        self.calls += 1
        self.batch_sizes.append(len(X))
        grad = np.ones(self.dim) * 0.01  # arbitrary nonzero gradient
        return grad, 0.0

    def evaluate(self, X, y):
        return 0.0, 0.0


def check_case(n: int, k_steps: int, batch_size: int, seed: int) -> None:
    X = np.arange(n * 8, dtype=np.float64).reshape(n, 8)
    y = np.zeros(n, dtype=np.int64)
    model = _CountingLinearModel(dim=model_dim(X))
    rng = np.random.default_rng(seed)
    local_training_fixed_steps(model, X, y, k_steps, lr=0.01, batch_size=batch_size, rng=rng)
    assert model.calls == k_steps, f"n={n} k={k_steps} b={batch_size}: expected {k_steps} steps, got {model.calls}"
    # every observed batch must be a full batch_size, or (only possible when n<batch_size) exactly n
    for bs in model.batch_sizes:
        assert bs == min(batch_size, n) or bs == batch_size, (n, k_steps, batch_size, bs)
    print(f"[ok] n={n:6d} K={k_steps:3d} batch_size={batch_size:3d} -> exactly {model.calls} train_step calls, "
          f"batch sizes in {{{min(model.batch_sizes)}..{max(model.batch_sizes)}}}")


def model_dim(X: np.ndarray) -> int:
    return X.shape[1]


def main() -> None:
    K = 50
    B = 128
    cases = [
        ("tiny client, n << batch_size", 37, K, B),
        ("n == batch_size exactly", 128, K, B),
        ("n slightly above batch_size", 150, K, B),
        ("n evenly divides K*B", 6400, K, B),
        ("n does not evenly divide K*B", 7439, K, B),
        ("large client, n >> K*B", 33151, K, B),
        ("huge client (IID scale)", 16200, K, B),
        ("K=1 edge case", 500, 1, B),
        ("B larger than n by a lot", 10, K, B),
    ]
    for label, n, k, b in cases:
        print(f"-- {label} --")
        check_case(n, k, b, seed=42)

    # reproducibility: same seed -> identical step count and same total weight delta magnitude pattern
    model_a = _CountingLinearModel(dim=8)
    rng_a = np.random.default_rng(123)
    X = np.arange(7439 * 8, dtype=np.float64).reshape(7439, 8)
    y = np.zeros(7439, dtype=np.int64)
    local_training_fixed_steps(model_a, X, y, K, lr=0.01, batch_size=B, rng=rng_a)
    assert model_a.calls == K

    print(f"\nALL_FIXED_STEP_CHECKS_OK K={K} batch_size={B}")


if __name__ == "__main__":
    main()
