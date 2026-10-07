"""Fixed-step-count local SGD for Dataset-backed clients (Fed-ISIC2019).

Same locked sampling procedure as local_training_fixed_steps.py
(docs/BENCHMARK_PROTOCOL.md Addendum 2026-10-07 (g)), generalized from
in-memory (X, y) arrays to an index list into a torch Dataset (FLamby's
FedIsic2019 instance) -- required because Fed-ISIC2019 images live on disk
at variable aspect ratios and are decoded/cropped per-item, not loaded
into one fixed-shape array. Exactly k_steps SGD updates regardless of the
client's natural size (natural Fed-ISIC2019 centers range from 351 to
9,930 training images -- an even larger imbalance than PathMNIST's
synthetic Dirichlet partitions), via the identical shuffle-and-cycle-on
-exhaustion rule, driven by the same single explicit rng stream (no
separate DataLoader-internal RNG).
"""

from __future__ import annotations

import numpy as np


def local_training_fixed_steps_image(
    model,
    dataset,
    client_indices: np.ndarray,
    k_steps: int,
    lr: float,
    batch_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    initial_weights = model.get_weights().copy()
    n = len(client_indices)
    if n == 0:
        raise ValueError("local_training_fixed_steps_image: empty client dataset")

    perm = rng.permutation(n)
    pos = 0
    for _ in range(k_steps):
        if pos + batch_size > n:
            perm = rng.permutation(n)
            pos = 0
        batch_pos = perm[pos : pos + batch_size]
        pos += batch_size
        batch_idx = client_indices[batch_pos]
        grad_vec, _loss = model.train_step_batch(dataset, batch_idx, lr)
        new_weights = model.get_weights() - lr * grad_vec
        model.set_weights(new_weights)

    delta = model.get_weights() - initial_weights
    return delta
