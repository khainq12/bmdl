"""Fixed-step-count local SGD — the corrected benchmark primitive.

Locked design: docs/BENCHMARK_PROTOCOL.md Addendum 2026-10-07 (g). Fixes
the step-count confound found in the first PathMNIST sweep (preserved at
results/pathmnist/benchmark_diagnostic_INVALID_v1/): local_training_seeded
(still present, unmodified, in local_training.py) ran one full local
epoch — a step count proportional to the client's dataset size — which
made gradient-delta magnitude incomparable across populations of
different sizes (calibration pseudo-clients vs. real clients; large vs.
small Dirichlet clients within the same round).

This primitive performs exactly K SGD steps for every caller, regardless
of dataset size, by cycling through a reshuffled pass of the client's own
data whenever the current pass is exhausted. It never resamples across
clients and never changes what data a client has — only how many times
the client's own data is revisited within one round's K steps.
"""

from __future__ import annotations

import numpy as np


def local_training_fixed_steps(
    model,
    X: np.ndarray,
    y: np.ndarray,
    k_steps: int,
    lr: float,
    batch_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Exactly k_steps SGD updates on (X, y), cycling/reshuffling as needed.

    Sampling procedure (locked, see module docstring / protocol Addendum (g)):
      1. Maintain one shuffled permutation of this client's own indices.
      2. Draw successive batches of size `batch_size` from its front.
      3. When fewer than `batch_size` indices remain (including n < batch_size
         from the start), discard the remainder and draw a fresh shuffled
         permutation, continuing from its start — standard epoch-boundary
         cycling, generalized from "stop after one pass" to "keep cycling
         until k_steps is reached."
      4. The client's own (X, y) is never resized, padded, or resampled
         from another client — only revisited more or fewer times.
    """
    initial_weights = model.get_weights().copy()
    n = len(X)
    if n == 0:
        raise ValueError("local_training_fixed_steps: empty client dataset")

    perm = rng.permutation(n)
    pos = 0
    for _ in range(k_steps):
        if pos + batch_size > n:
            perm = rng.permutation(n)
            pos = 0
        batch_idx = perm[pos : pos + batch_size]
        pos += batch_size
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        grad_vec, _loss = model.train_step(X_batch, y_batch, lr)
        new_weights = model.get_weights() - lr * grad_vec
        model.set_weights(new_weights)

    delta = model.get_weights() - initial_weights
    return delta
