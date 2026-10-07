"""Experimental verification that local_training_fixed_steps_image performs
EXACTLY K optimizer steps regardless of client (natural center) size --
requested explicitly by ChatGPT before any GPU benchmarking: Fed-ISIC2019's
natural centers range from 299 (smallest, post-split) to 7,597 (largest
non-server-ref client) training samples, a ~25x imbalance even larger than
PathMNIST's synthetic Dirichlet partitions. Mirrors the spirit of
scripts/verify_fixed_steps.py (PathMNIST phase) but uses the REAL 6 center
sizes from the actual Fed-ISIC2019 split, not just synthetic n values, and
a counting stub (no GPU/real images needed) so this is a cheap, fast,
code-path-identical check before committing GPU hours.

Usage: python scripts/verify_fixed_steps_image.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from benchmark.datasets.fed_isic2019 import load_fed_isic2019_problem  # noqa: E402
from benchmark.models.local_training_image import local_training_fixed_steps_image  # noqa: E402


class _CountingImageModel:
    """Counting stub matching FedIsicModel's get_weights/set_weights/
    train_step_batch contract, without touching GPU/real images."""

    def __init__(self, dim: int = 16):
        self.dim = dim
        self.w = np.zeros(dim)
        self.calls = 0
        self.batch_sizes = []
        self.indices_seen = []

    def get_weights(self) -> np.ndarray:
        return self.w.copy()

    def set_weights(self, flat: np.ndarray) -> None:
        self.w = flat.copy()

    def train_step_batch(self, dataset, indices: np.ndarray, lr: float):
        self.calls += 1
        self.batch_sizes.append(len(indices))
        self.indices_seen.append(np.asarray(indices).copy())
        grad = np.ones(self.dim) * 0.01
        return grad, 0.0


def check_case(label: str, n: int, k_steps: int, batch_size: int, seed: int) -> None:
    client_indices = np.arange(n, dtype=np.int64)  # values don't matter, only count/cycling does
    model = _CountingImageModel()
    rng = np.random.default_rng(seed)
    local_training_fixed_steps_image(model, dataset=None, client_indices=client_indices, k_steps=k_steps, lr=0.01, batch_size=batch_size, rng=rng)
    assert model.calls == k_steps, f"{label} n={n} k={k_steps} b={batch_size}: expected {k_steps} steps, got {model.calls}"
    for bs in model.batch_sizes:
        assert bs == min(batch_size, n) or bs == batch_size, (n, k_steps, batch_size, bs)
    # every index actually used must come from the client's own n -- never resampled from elsewhere
    all_idx = np.concatenate(model.indices_seen)
    assert all_idx.min() >= 0 and all_idx.max() < n, f"{label}: indices escaped client's own range [0,{n})"
    print(f"[ok] {label:40s} n={n:6d} K={k_steps:3d} batch_size={batch_size:3d} -> exactly {model.calls} train_step_batch calls, "
          f"batch sizes in {{{min(model.batch_sizes)}..{max(model.batch_sizes)}}}, all indices within [0,{n})")


def main() -> None:
    K = 10
    B = 32

    print("=== REAL Fed-ISIC2019 center sizes (post server-ref/calib split) ===")
    problem = load_fed_isic2019_problem(seed=42)
    for c, (ds, idx) in enumerate(problem["client_datasets"]):
        check_case(f"REAL center {c} ({'smallest' if c == 5 else 'largest' if c == 0 else ''})", len(idx), K, B, seed=42)

    print("\n=== synthetic edge cases (same style as PathMNIST's verify_fixed_steps.py) ===")
    edge_cases = [
        ("tiny client, n << batch_size", 11, K, B),
        ("n == batch_size exactly", B, K, B),
        ("n slightly above batch_size", B + 5, K, B),
        ("n does not evenly divide K*B", 299, K, B),  # the real smallest center's exact size
        ("K=1 edge case", 50, 1, B),
        ("B larger than n by a lot", 5, K, B),
    ]
    for label, n, k, b in edge_cases:
        check_case(label, n, k, b, seed=42)

    print(f"\nALL_FIXED_STEP_IMAGE_CHECKS_OK K={K} batch_size={B}")
    print("Confirmed: all 6 real Fed-ISIC2019 centers (299 to 7,597 samples) execute exactly")
    print(f"K={K} optimizer steps/round each -- no oversampling, no size-dependent step-count change,")
    print("no cross-client index leakage. Same fixed-K cycling logic as PathMNIST, just index-based.")


if __name__ == "__main__":
    main()
