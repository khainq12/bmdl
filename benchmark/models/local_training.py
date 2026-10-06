"""Seeded local SGD step for the new benchmark framework.

This fixes the upstream reproducibility gap documented in
docs/IMPLEMENTATION_PLAN.md Sec 3.9: experiments/run_experiment.py's
local_training() shuffles minibatches with an unseeded
np.random.default_rng(). That script is left untouched (it is the upstream
reproduction path). Every call site in benchmark/ uses this seeded version
instead, taking an explicit rng so determinism is caller-controlled.
"""

from __future__ import annotations

import numpy as np


def local_training_seeded(
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_epochs: int,
    lr: float,
    batch_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    initial_weights = model.get_weights().copy()
    n_samples = len(X)

    for _ in range(n_epochs):
        perm = rng.permutation(n_samples)
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_idx = perm[start:end]
            X_batch, y_batch = X[batch_idx], y[batch_idx]
            grad_vec, _loss = model.train_step(X_batch, y_batch, lr)
            new_weights = model.get_weights() - lr * grad_vec
            model.set_weights(new_weights)

    delta = model.get_weights() - initial_weights
    return delta
