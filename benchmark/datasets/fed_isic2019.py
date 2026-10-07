"""Fed-ISIC2019 (FLamby) dataset loader -- STEP: natural-client validation
role (locked, docs/BENCHMARK_PROTOCOL.md Sec 1). See Addendum 2026-10-07
(j) for the full data-role mapping this module implements:

    center 0 (largest, BCN)  -> {server_ref slice, client data = remainder}
    each center's remainder  -> {calib slice, client data = remainder}
    pooled official test     -> held out, never touched here

Native FLamby center partition is preserved exactly for client training
data -- no re-partitioning, no Dirichlet re-sampling (docs/BENCHMARK_
PROTOCOL.md Sec 1/3). Requires FLamby installed and the dataset already
downloaded+preprocessed by the user (see Addendum (j); this module does
not and must not attempt to download anything).
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
_SPLIT_DIR = os.path.join(_ROOT, "data", "partitions", "fed_isic2019")

N_CENTERS = 6
SERVER_REF_CENTER = 0  # largest (BCN), fixed and pre-declared -- never re-chosen per condition


def _split_filename(server_ref_fraction: float, calib_fraction: float, seed: int) -> str:
    return f"split_ref{server_ref_fraction}_calib{calib_fraction}_seed{seed}.json"


def build_or_load_split(
    center_sizes: List[int],
    server_ref_fraction: float,
    calib_fraction: float,
    seed: int,
    force_rebuild: bool = False,
) -> Dict:
    """Returns {"server_ref": {center: [idx]}, "calib": {center: [idx]},
    "client": {center: [idx]}} -- indices into each center's OWN training
    Dataset (FedIsic2019(center=c, train=True)), not a global index space."""
    os.makedirs(_SPLIT_DIR, exist_ok=True)
    path = os.path.join(_SPLIT_DIR, _split_filename(server_ref_fraction, calib_fraction, seed))
    if os.path.isfile(path) and not force_rebuild:
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    rng = np.random.default_rng(seed)
    # string keys throughout (not just after a JSON round-trip) so the
    # fresh-build and cache-hit return paths are identical to callers.
    server_ref: Dict[str, List[int]] = {}
    calib: Dict[str, List[int]] = {}
    client: Dict[str, List[int]] = {}

    for c in range(N_CENTERS):
        n = center_sizes[c]
        perm = rng.permutation(n)
        pos = 0
        if c == SERVER_REF_CENTER:
            n_ref = int(server_ref_fraction * n)
            server_ref[str(c)] = perm[:n_ref].tolist()
            pos = n_ref
        else:
            server_ref[str(c)] = []
        remaining = perm[pos:]
        n_calib = int(calib_fraction * len(remaining))
        calib[str(c)] = remaining[:n_calib].tolist()
        client[str(c)] = remaining[n_calib:].tolist()

    record = {
        "server_ref": server_ref,
        "calib": calib,
        "client": client,
        "meta": {
            "center_sizes": center_sizes,
            "server_ref_fraction": server_ref_fraction,
            "server_ref_center": SERVER_REF_CENTER,
            "calib_fraction": calib_fraction,
            "seed": seed,
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f)
    return record


def assert_disjoint(split: Dict) -> Dict:
    """Index-set disjointness per center (server_ref/calib/client are
    index-set partitions of that center's own training pool by
    construction -- verified, not assumed) plus a cross-center sanity
    check that no center's indices exceed its declared size."""
    overlaps = {}
    for c_str, idx in split["server_ref"].items():
        c = int(c_str)
        ref_set = set(idx)
        calib_set = set(split["calib"][c_str])
        client_set = set(split["client"][c_str])
        overlaps[f"center{c}_ref_vs_calib"] = len(ref_set & calib_set)
        overlaps[f"center{c}_ref_vs_client"] = len(ref_set & client_set)
        overlaps[f"center{c}_calib_vs_client"] = len(calib_set & client_set)
        total = len(ref_set) + len(calib_set) + len(client_set)
        overlaps[f"center{c}_total_matches_size"] = total == split["meta"]["center_sizes"][c]

    all_ok = all(v == 0 for k, v in overlaps.items() if "total_matches_size" not in k) and all(
        v for k, v in overlaps.items() if "total_matches_size" in k
    )
    overlaps["all_ok"] = all_ok
    if not all_ok:
        raise AssertionError(f"Fed-ISIC2019 split disjointness violated: {overlaps}")
    return overlaps


def load_fed_isic2019_problem(server_ref_fraction: float = 0.10, calib_fraction: float = 0.15, seed: int = 42) -> Dict:
    from flamby.datasets.fed_isic2019 import FedIsic2019

    train_centers = [FedIsic2019(center=c, train=True, pooled=False) for c in range(N_CENTERS)]
    test_pooled = FedIsic2019(train=False, pooled=True)
    center_sizes = [len(ds) for ds in train_centers]

    split = build_or_load_split(center_sizes, server_ref_fraction, calib_fraction, seed)
    assert_disjoint(split)

    client_datasets = [(train_centers[c], np.array(split["client"][str(c)], dtype=np.int64)) for c in range(N_CENTERS)]
    server_ref_dataset = (train_centers[SERVER_REF_CENTER], np.array(split["server_ref"][str(SERVER_REF_CENTER)], dtype=np.int64))
    calib_datasets = [(train_centers[c], np.array(split["calib"][str(c)], dtype=np.int64)) for c in range(N_CENTERS)]

    def model_factory(pretrained: bool = True, seed: int = seed):
        from benchmark.models.torch_image_model import FedIsicModel

        return FedIsicModel(pretrained=pretrained, seed=seed)

    return {
        "client_datasets": client_datasets,  # [(Dataset, client_idx), ...] one per natural center
        "test": test_pooled,
        "calib_datasets": calib_datasets,  # [(Dataset, calib_idx), ...] per center, pool for calibration
        "server_ref": server_ref_dataset,
        "model_factory": model_factory,
        "n_classes": 8,
        "n_centers": N_CENTERS,
        "center_sizes": center_sizes,
        "_split": split,
    }
