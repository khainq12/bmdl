"""Synthetic dataset loader — reproduction/debugging role ONLY (locked,
docs/BENCHMARK_PROTOCOL.md Sec 1). Never used for headline defense-comparison
claims; PathMNIST (STEP 5) and Fed-ISIC2019 (STEP 6) carry those.

Builds the four disjoint subsets required by Sec 3 of the protocol: server
trusted reference set, calibration/validation set, client training
partitions, and the test set — and asserts disjointness in code, not just
in prose.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from fl_core.model import SimpleMLP, generate_synthetic_medical_data, partition_non_iid


def partition_iid(X: np.ndarray, y: np.ndarray, n_clients: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """True IID partition: shuffle once, split into n_clients equal chunks."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    chunks = np.array_split(idx, n_clients)
    return [(X[c], y[c]) for c in chunks]


def _assert_disjoint(**named_index_arrays: np.ndarray) -> None:
    sets = {name: set(arr.tolist()) for name, arr in named_index_arrays.items()}
    total = sum(len(s) for s in sets.values())
    union: set = set()
    for s in sets.values():
        union |= s
    if total != len(union):
        raise AssertionError(
            f"dataset split overlap detected among {list(sets)} — "
            "violates docs/BENCHMARK_PROTOCOL.md Sec 3 disjointness rule"
        )


def load_synthetic_problem(cfg) -> Dict:
    ds = cfg.dataset
    X, y = generate_synthetic_medical_data(n_samples=1000, n_features=64, n_classes=4, seed=ds.seed)
    n = len(X)

    rng = np.random.default_rng(ds.seed)
    perm = rng.permutation(n)

    n_test = int(0.15 * n)
    remaining_after_test = n - n_test
    n_calib = int(ds.calib_fraction * remaining_after_test)
    n_ref = int(ds.server_ref_fraction * remaining_after_test)

    test_idx = perm[:n_test]
    rest = perm[n_test:]
    calib_idx = rest[:n_calib]
    ref_idx = rest[n_calib : n_calib + n_ref]
    client_pool_idx = rest[n_calib + n_ref :]

    _assert_disjoint(test=test_idx, calib=calib_idx, server_ref=ref_idx, client_pool=client_pool_idx)

    X_test, y_test = X[test_idx], y[test_idx]
    X_calib, y_calib = X[calib_idx], y[calib_idx]
    X_ref, y_ref = X[ref_idx], y[ref_idx]
    X_pool, y_pool = X[client_pool_idx], y[client_pool_idx]

    if ds.partition == "iid":
        partitions = partition_iid(X_pool, y_pool, ds.n_clients, seed=ds.seed)
    else:
        partitions = partition_non_iid(X_pool, y_pool, ds.n_clients, alpha=ds.dirichlet_alpha, seed=ds.seed)

    n_classes = int(np.max(y)) + 1
    n_features = X.shape[1]

    def model_factory(seed: int = ds.seed):
        return SimpleMLP(n_features, n_classes, seed=seed, hidden=tuple(cfg.model.hidden))

    return {
        "partitions": partitions,
        "test": (X_test, y_test),
        "calib": (X_calib, y_calib),
        "server_ref": (X_ref, y_ref),
        "model_factory": model_factory,
        "n_features": n_features,
        "n_classes": n_classes,
    }
