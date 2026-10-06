"""PathMNIST dataset loader — STEP 5, "main controlled benchmark" role
(locked, docs/BENCHMARK_PROTOCOL.md Sec 1). Official MedMNIST train/val/test
split is used and preserved exactly; see Addendum 2026-10-05 (c) for the
split-to-protocol-role mapping this module implements:

    official train -> {server_ref, client partitions}  (server_ref carved first)
    official val   -> calibration/validation subset, used as-is
    official test  -> held out, never touched here

Client partitions (IID or Dirichlet(alpha)) are generated once per
(partition, alpha, n_clients, seed) and persisted to disk under
data/partitions/pathmnist/ so every defense in a comparison sees the exact
same client data (task brief Sec 2).
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Tuple

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
_PARTITION_DIR = os.path.join(_ROOT, "data", "partitions", "pathmnist")


def _medmnist_split(split: str):
    import medmnist
    from medmnist import INFO

    info = INFO["pathmnist"]
    DataClass = getattr(medmnist, info["python_class"])
    ds = DataClass(split=split, download=True)
    imgs = ds.imgs  # (N, 28, 28, 3) uint8
    labels = ds.labels.ravel().astype(np.int64)
    X = np.ascontiguousarray(imgs.transpose(0, 3, 1, 2))  # -> (N, 3, 28, 28) uint8
    return X, labels, info


def _hash_images(X: np.ndarray) -> List[str]:
    return [hashlib.sha1(x.tobytes()).hexdigest() for x in X]


def partition_iid_indices(n: int, n_clients: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    return [c.astype(np.int64) for c in np.array_split(idx, n_clients)]


def partition_dirichlet_indices(n: int, y: np.ndarray, n_clients: int, alpha: float, seed: int) -> List[np.ndarray]:
    """Reuses the canonical fl_core.model.partition_non_iid Dirichlet
    algorithm (not a reimplementation) by partitioning index positions as a
    stand-in 'X' array, so the resulting partitions are index-addressable
    for disk persistence while remaining bit-identical to the upstream
    partitioner's logic."""
    from fl_core.model import partition_non_iid

    idx_as_X = np.arange(n, dtype=np.int64).reshape(-1, 1)
    parts = partition_non_iid(idx_as_X, y, n_clients, alpha=alpha, seed=seed)
    return [p[0].ravel().astype(np.int64) for p in parts]


def _partition_filename(partition: str, alpha: float, n_clients: int, server_ref_fraction: float, seed: int) -> str:
    alpha_tag = f"alpha{alpha}" if partition == "dirichlet" else "iid"
    return f"{partition}_{alpha_tag}_clients{n_clients}_ref{server_ref_fraction}_seed{seed}.json"


def build_or_load_partitions(
    n_train: int,
    y_train: np.ndarray,
    partition: str,
    alpha: float,
    n_clients: int,
    server_ref_fraction: float,
    seed: int,
    force_rebuild: bool = False,
) -> Dict:
    """Returns {"server_ref_idx": [...], "client_idx": [[...], ...]}, indices
    into the official PathMNIST train array. Persisted to disk; regenerated
    only if the file is missing or force_rebuild=True (deterministic given
    the same seed, so regeneration reproduces the same partition anyway)."""
    os.makedirs(_PARTITION_DIR, exist_ok=True)
    path = os.path.join(_PARTITION_DIR, _partition_filename(partition, alpha, n_clients, server_ref_fraction, seed))

    if os.path.isfile(path) and not force_rebuild:
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_train)
    n_ref = int(server_ref_fraction * n_train)
    server_ref_idx = perm[:n_ref]
    client_pool_idx = perm[n_ref:]

    if partition == "iid":
        client_idx = partition_iid_indices(len(client_pool_idx), n_clients, seed=seed)
        client_idx = [client_pool_idx[c] for c in client_idx]
    elif partition == "dirichlet":
        y_pool = y_train[client_pool_idx]
        rel_idx = partition_dirichlet_indices(len(client_pool_idx), y_pool, n_clients, alpha, seed=seed)
        client_idx = [client_pool_idx[c] for c in rel_idx]
    else:
        raise ValueError(f"unknown partition scheme: {partition}")

    record = {
        "server_ref_idx": server_ref_idx.tolist(),
        "client_idx": [c.tolist() for c in client_idx],
        "meta": {
            "n_train": int(n_train),
            "partition": partition,
            "alpha": alpha if partition == "dirichlet" else None,
            "n_clients": n_clients,
            "server_ref_fraction": server_ref_fraction,
            "seed": seed,
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f)
    return record


def assert_four_way_disjoint(
    server_ref_idx: np.ndarray,
    client_idx: List[np.ndarray],
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> Dict:
    """Programmatic zero-overlap proof: (a) index-set algebra within the
    train array (server_ref vs each client vs each other client), and (b)
    a content-hash duplicate check across train/val/test as a whole, so a
    duplicated image under two different official-split indices would also
    be caught (index-set disjointness alone cannot see across arrays)."""
    client_sets = [set(c.tolist()) for c in client_idx]
    ref_set = set(server_ref_idx.tolist())

    overlaps = {}
    overlaps["server_ref_vs_clients"] = sum(len(ref_set & cs) for cs in client_sets)
    pairwise_client_overlap = 0
    for i in range(len(client_sets)):
        for j in range(i + 1, len(client_sets)):
            pairwise_client_overlap += len(client_sets[i] & client_sets[j])
    overlaps["client_vs_client"] = pairwise_client_overlap

    train_hashes = _hash_images(X_train)
    val_hashes = _hash_images(X_val)
    test_hashes = _hash_images(X_test)
    h_train, h_val, h_test = set(train_hashes), set(val_hashes), set(test_hashes)
    overlaps["content_hash_train_vs_val"] = len(h_train & h_val)
    overlaps["content_hash_train_vs_test"] = len(h_train & h_test)
    overlaps["content_hash_val_vs_test"] = len(h_val & h_test)

    ref_hashes = {train_hashes[i] for i in server_ref_idx.tolist()}
    client_all_idx: set = set()
    for cs in client_sets:
        client_all_idx |= cs
    client_hashes = {train_hashes[i] for i in client_all_idx}
    overlaps["content_hash_server_ref_vs_client_training"] = len(ref_hashes & client_hashes)
    overlaps["content_hash_server_ref_vs_val"] = len(ref_hashes & h_val)
    overlaps["content_hash_server_ref_vs_test"] = len(ref_hashes & h_test)
    overlaps["content_hash_client_training_vs_val"] = len(client_hashes & h_val)
    overlaps["content_hash_client_training_vs_test"] = len(client_hashes & h_test)

    all_zero = all(v == 0 for v in overlaps.values())
    overlaps["all_zero"] = all_zero
    if not all_zero:
        raise AssertionError(f"PathMNIST disjointness violated: {overlaps}")
    return overlaps


def load_pathmnist_problem(cfg) -> Dict:
    ds = cfg.dataset
    X_train, y_train, info = _medmnist_split("train")
    X_val, y_val, _ = _medmnist_split("val")
    X_test, y_test, _ = _medmnist_split("test")

    record = build_or_load_partitions(
        n_train=len(X_train),
        y_train=y_train,
        partition=ds.partition,
        alpha=ds.dirichlet_alpha,
        n_clients=ds.n_clients,
        server_ref_fraction=ds.server_ref_fraction,
        seed=ds.seed,
    )
    server_ref_idx = np.array(record["server_ref_idx"], dtype=np.int64)
    client_idx = [np.array(c, dtype=np.int64) for c in record["client_idx"]]

    X_ref, y_ref = X_train[server_ref_idx], y_train[server_ref_idx]
    partitions = [(X_train[c], y_train[c]) for c in client_idx]

    n_classes = len(info["label"])
    device = getattr(cfg, "device", "cpu")

    def model_factory(seed: int = ds.seed):
        from benchmark.models.torch_cnn import TorchCNNFlat

        return TorchCNNFlat(n_classes=n_classes, seed=seed, device=device)

    return {
        "partitions": partitions,
        "test": (X_test, y_test),
        "calib": (X_val, y_val),
        "server_ref": (X_ref, y_ref),
        "model_factory": model_factory,
        "n_features": None,  # images, not flat features
        "n_classes": n_classes,
        "_audit": {
            "server_ref_idx": server_ref_idx,
            "client_idx": client_idx,
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
            "info": info,
        },
    }
