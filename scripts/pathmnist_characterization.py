"""PathMNIST honest-gradient characterization study. Locked design:
docs/BENCHMARK_PROTOCOL.md Addendum 2026-10-06 (e) — read that before this
code. NO attacks are injected and NO threshold (tau/rho/kappa) is selected
anywhere in this script.

Also produces the explicit content-hash disjointness proof for calibration
pseudo-clients (built from the official val split) against server_ref,
client-training data, and test — Addendum 2026-10-06 (d).4.

Usage: python scripts/pathmnist_characterization.py
Writes under results/pathmnist/characterization/:
  client_scores.csv, pairwise_cosine.csv, gref_size_sensitivity.csv,
  calibration_disjointness_proof.csv, summary.json, plots/*.png
"""

from __future__ import annotations

import csv
import itertools
import json
import os
import sys
import time
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from benchmark.datasets.pathmnist import (  # noqa: E402
    _hash_images,
    _medmnist_split,
    build_or_load_partitions,
    partition_dirichlet_indices,
    partition_iid_indices,
)
from benchmark.defenses.cosine import cosine_similarity  # noqa: E402
from benchmark.defenses.sign_consensus import sign_consensus_scores  # noqa: E402
from benchmark.models import local_training_seeded  # noqa: E402
from benchmark.models.torch_cnn import TorchCNNFlat  # noqa: E402

OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "pathmnist", "characterization"
)
PLOTS_DIR = os.path.join(OUT_DIR, "plots")

N_CLIENTS = 5
SEEDS = [42, 43, 44]
PARTITIONS = [("iid", None), ("dirichlet", 1.0), ("dirichlet", 0.5), ("dirichlet", 0.1)]
N_ROUNDS = 10
LOCAL_EPOCHS = 1
LOCAL_LR = 0.01
BATCH_SIZE = 128
SERVER_REF_FRACTION = 0.10
DEVICE = os.environ.get("ZKFL_DEVICE", "cuda")

GREF_SIZE_SETTINGS = [("iid", None), ("dirichlet", 0.1)]
GREF_SIZE_FRACTIONS_OF_POOL = [0.1, 0.25, 0.5, 1.0]
CALIB_SEED = 1000
N_CALIB_CLIENTS = max(3, N_CLIENTS)


def _tag(partition: str, alpha) -> str:
    return "iid" if partition == "iid" else f"dirichlet_a{alpha}"


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        open(path, "w", encoding="utf-8").close()
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _stats(values) -> Dict[str, float]:
    v = np.asarray(values, dtype=np.float64)
    return {
        "n": int(v.size),
        "mean": float(np.mean(v)),
        "std": float(np.std(v)),
        "median": float(np.median(v)),
        "p5": float(np.percentile(v, 5)),
        "p25": float(np.percentile(v, 25)),
        "p75": float(np.percentile(v, 75)),
        "p95": float(np.percentile(v, 95)),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
    }


def _model_factory(n_classes: int, seed: int) -> TorchCNNFlat:
    return TorchCNNFlat(n_classes=n_classes, seed=seed, device=DEVICE)


def _compute_delta(n_classes, seed, global_weights, X, y, rng) -> np.ndarray:
    m = _model_factory(n_classes, seed)
    m.set_weights(global_weights.copy())
    return local_training_seeded(m, X, y, LOCAL_EPOCHS, LOCAL_LR, BATCH_SIZE, rng)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("Loading official PathMNIST splits ...")
    X_train, y_train, info = _medmnist_split("train")
    X_val, y_val, _ = _medmnist_split("val")
    X_test, y_test, _ = _medmnist_split("test")
    n_classes = len(info["label"])
    print(f"train={len(y_train)} val={len(y_val)} test={len(y_test)} n_classes={n_classes} device={DEVICE}")

    train_hashes_all = _hash_images(X_train)
    test_hashes = set(_hash_images(X_test))

    client_rows: List[Dict[str, Any]] = []
    pairwise_rows: List[Dict[str, Any]] = []
    gref_size_rows: List[Dict[str, Any]] = []
    calib_disjoint_rows: List[Dict[str, Any]] = []

    for partition, alpha in PARTITIONS:
        tag = _tag(partition, alpha)
        for seed in SEEDS:
            t0 = time.perf_counter()
            record = build_or_load_partitions(
                n_train=len(X_train),
                y_train=y_train,
                partition=partition,
                alpha=alpha if alpha is not None else 0.5,
                n_clients=N_CLIENTS,
                server_ref_fraction=SERVER_REF_FRACTION,
                seed=seed,
            )
            server_ref_idx = np.array(record["server_ref_idx"], dtype=np.int64)
            client_idx = [np.array(c, dtype=np.int64) for c in record["client_idx"]]
            X_ref, y_ref = X_train[server_ref_idx], y_train[server_ref_idx]
            partitions_xy = [(X_train[c], y_train[c]) for c in client_idx]

            # --- calibration pseudo-client disjointness proof (Addendum (d).4) ---
            if partition == "iid":
                calib_parts_idx = partition_iid_indices(len(y_val), N_CALIB_CLIENTS, seed=CALIB_SEED)
            else:
                calib_parts_idx = partition_dirichlet_indices(len(y_val), y_val, N_CALIB_CLIENTS, alpha, seed=CALIB_SEED)
            calib_all_idx = sorted(set(int(i) for c in calib_parts_idx for i in c))
            calib_hashes = set(_hash_images(X_val[calib_all_idx]))
            server_ref_hashes = {train_hashes_all[i] for i in server_ref_idx.tolist()}
            client_all_idx = sorted(set(int(i) for c in client_idx for i in c))
            client_train_hashes = {train_hashes_all[i] for i in client_all_idx}
            ov_ref = len(calib_hashes & server_ref_hashes)
            ov_client = len(calib_hashes & client_train_hashes)
            ov_test = len(calib_hashes & test_hashes)
            calib_disjoint_rows.append(
                {
                    "partition": tag,
                    "seed": seed,
                    "n_calib_samples_used": len(calib_all_idx),
                    "calib_vs_server_ref": ov_ref,
                    "calib_vs_client_training": ov_client,
                    "calib_vs_test": ov_test,
                    "all_zero": (ov_ref == 0 and ov_client == 0 and ov_test == 0),
                }
            )

            # --- honest-only FL trajectory ---
            model = _model_factory(n_classes, seed)
            rng = np.random.default_rng(seed + 500)

            do_gref_sweep = (partition, alpha) in GREF_SIZE_SETTINGS
            pool_perm = np.random.default_rng(seed + 900).permutation(len(server_ref_idx)) if do_gref_sweep else None

            for r in range(N_ROUNDS):
                gw = model.get_weights().copy()
                g_ref = _compute_delta(n_classes, seed, gw, X_ref, y_ref, rng)

                deltas = [
                    _compute_delta(n_classes, seed, gw, Xc, yc, rng) for Xc, yc in partitions_xy
                ]

                norms = [float(np.linalg.norm(d)) for d in deltas]
                cos_refs = [cosine_similarity(d, g_ref) for d in deltas]
                sign_scores = sign_consensus_scores(deltas).tolist()

                for cid in range(N_CLIENTS):
                    client_rows.append(
                        {
                            "partition": tag,
                            "seed": seed,
                            "round": r,
                            "client_id": cid,
                            "norm": norms[cid],
                            "cos_ref": cos_refs[cid],
                            "sign_score": sign_scores[cid],
                        }
                    )
                for i, j in itertools.combinations(range(N_CLIENTS), 2):
                    pairwise_rows.append(
                        {
                            "partition": tag,
                            "seed": seed,
                            "round": r,
                            "client_i": i,
                            "client_j": j,
                            "cos_ij": cosine_similarity(deltas[i], deltas[j]),
                        }
                    )

                if do_gref_sweep:
                    for frac in GREF_SIZE_FRACTIONS_OF_POOL:
                        k = max(1, int(frac * len(server_ref_idx)))
                        sub_idx = server_ref_idx[pool_perm[:k]]
                        Xr, yr = X_train[sub_idx], y_train[sub_idx]
                        g_ref_k = _compute_delta(n_classes, seed, gw, Xr, yr, rng)
                        class_counts = np.bincount(yr, minlength=n_classes).tolist()
                        for cid in range(N_CLIENTS):
                            gref_size_rows.append(
                                {
                                    "partition": tag,
                                    "seed": seed,
                                    "round": r,
                                    "ref_fraction_of_pool": frac,
                                    "ref_n_samples": int(k),
                                    "ref_class_counts": json.dumps(class_counts),
                                    "client_id": cid,
                                    "cos_ref_at_size": cosine_similarity(deltas[cid], g_ref_k),
                                }
                            )

                update = np.mean(deltas, axis=0)
                model.set_weights(gw + update)

            dt = time.perf_counter() - t0
            print(f"{tag} seed={seed}: {N_ROUNDS} rounds done in {dt:.1f}s")

    _write_csv(os.path.join(OUT_DIR, "client_scores.csv"), client_rows)
    _write_csv(os.path.join(OUT_DIR, "pairwise_cosine.csv"), pairwise_rows)
    _write_csv(os.path.join(OUT_DIR, "gref_size_sensitivity.csv"), gref_size_rows)
    _write_csv(os.path.join(OUT_DIR, "calibration_disjointness_proof.csv"), calib_disjoint_rows)

    summary = build_summary(client_rows, pairwise_rows, gref_size_rows)
    with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    make_plots(client_rows, pairwise_rows, gref_size_rows)

    all_disjoint = all(r["all_zero"] for r in calib_disjoint_rows)
    print(f"calibration disjointness proof all_zero across all settings: {all_disjoint}")
    print(f"Saved characterization outputs to {OUT_DIR}")


def build_summary(client_rows, pairwise_rows, gref_size_rows) -> Dict[str, Any]:
    settings = sorted(set(r["partition"] for r in client_rows))
    summary: Dict[str, Any] = {"per_setting": {}, "gref_size_sensitivity": {}}
    for s in settings:
        rows = [r for r in client_rows if r["partition"] == s]
        prows = [r for r in pairwise_rows if r["partition"] == s]
        rounds = sorted(set(r["round"] for r in rows))
        summary["per_setting"][s] = {
            "norm": _stats([r["norm"] for r in rows]),
            "cos_ref": _stats([r["cos_ref"] for r in rows]),
            "sign_score": _stats([r["sign_score"] for r in rows]),
            "pairwise_cos": _stats([r["cos_ij"] for r in prows]),
            "cos_ref_by_round_mean": {
                str(rnd): float(np.mean([r["cos_ref"] for r in rows if r["round"] == rnd])) for rnd in rounds
            },
        }

    for s in sorted(set(r["partition"] for r in gref_size_rows)):
        rows = [r for r in gref_size_rows if r["partition"] == s]
        fracs = sorted(set(r["ref_fraction_of_pool"] for r in rows))
        summary["gref_size_sensitivity"][s] = {
            str(f): _stats([r["cos_ref_at_size"] for r in rows if r["ref_fraction_of_pool"] == f]) for f in fracs
        }
    return summary


def make_plots(client_rows, pairwise_rows, gref_size_rows) -> None:
    settings = sorted(set(r["partition"] for r in client_rows), key=lambda s: (s != "iid", s))

    def boxplot_metric(key, rows, title, ylabel, fname):
        data = [[r[key] for r in rows if r["partition"] == s] for s in settings]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.boxplot(data, tick_labels=settings, showfliers=True)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, fname), dpi=150)
        plt.close(fig)

    boxplot_metric("norm", client_rows, "Honest gradient L2 norm by partition", "||g_i||_2", "norm_by_partition.png")
    boxplot_metric(
        "cos_ref", client_rows, "cos(honest client, g_ref) by partition", "cos(g_i, g_ref)", "cos_ref_by_partition.png"
    )
    boxplot_metric(
        "sign_score",
        client_rows,
        "Sign Consensus score (honest-only) by partition",
        "sign consensus score",
        "sign_score_by_partition.png",
    )
    boxplot_metric(
        "cos_ij",
        pairwise_rows,
        "Pairwise honest-client cosine similarity by partition",
        "cos(g_i, g_j)",
        "pairwise_cosine_by_partition.png",
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for s in settings:
        rows = [r for r in client_rows if r["partition"] == s]
        rounds = sorted(set(r["round"] for r in rows))
        means = [float(np.mean([r["cos_ref"] for r in rows if r["round"] == rnd])) for rnd in rounds]
        ax.plot(rounds, means, marker="o", label=s)
    ax.set_xlabel("round")
    ax.set_ylabel("mean cos(honest, g_ref)")
    ax.set_title("Honest-client alignment with g_ref over training")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "cos_ref_vs_round.png"), dpi=150)
    plt.close(fig)

    gsettings = sorted(set(r["partition"] for r in gref_size_rows))
    if gsettings:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for s in gsettings:
            rows = [r for r in gref_size_rows if r["partition"] == s]
            fracs = sorted(set(r["ref_fraction_of_pool"] for r in rows))
            means = [float(np.mean([r["cos_ref_at_size"] for r in rows if r["ref_fraction_of_pool"] == f])) for f in fracs]
            stds = [float(np.std([r["cos_ref_at_size"] for r in rows if r["ref_fraction_of_pool"] == f])) for f in fracs]
            ax.errorbar(fracs, means, yerr=stds, marker="o", label=s, capsize=3)
        ax.set_xlabel("server_ref fraction of the reserved 10% pool")
        ax.set_ylabel("cos(honest, g_ref) mean +/- std")
        ax.set_title("g_ref size sensitivity")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, "gref_size_sensitivity.png"), dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
