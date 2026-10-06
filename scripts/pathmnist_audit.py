"""STEP 5 dataset audit: sample counts, per-client class distribution (both
IID and Dirichlet), server-reference and calibration(=val) class
distribution, and a programmatic zero-overlap proof across
server-ref/calibration/client-training/test. See
docs/BENCHMARK_PROTOCOL.md Sec 3 and Addendum 2026-10-05 (c).

Usage: python scripts/pathmnist_audit.py [alpha]
Writes results/pathmnist/dataset_audit.json and a companion .md summary.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from benchmark.config import DatasetConfig, RunConfig, ModelConfig, OptimConfig, AttackConfig, DefenseConfig, CalibrationConfig  # noqa: E402
from benchmark.datasets.pathmnist import assert_four_way_disjoint, load_pathmnist_problem  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "pathmnist")


def _class_counts(y: np.ndarray, n_classes: int) -> list:
    counts = np.bincount(y, minlength=n_classes)
    return counts.tolist()


def _audit_one(partition: str, alpha: float, n_clients: int, seed: int) -> dict:
    ds = DatasetConfig(
        name="pathmnist",
        n_clients=n_clients,
        partition=partition,
        dirichlet_alpha=alpha,
        server_ref_fraction=0.10,
        calib_fraction=0.15,  # unused for pathmnist (val split used as-is); kept for schema consistency
        seed=seed,
    )
    cfg = RunConfig(
        dataset=ds,
        model=ModelConfig(name="torch_cnn"),
        optim=OptimConfig(),
        attack=AttackConfig(),
        defense=DefenseConfig(),
        calibration=CalibrationConfig(),
        seed=seed,
        device="cpu",  # audit is pure indexing/counting, no need for GPU
    )
    problem = load_pathmnist_problem(cfg)
    audit = problem["_audit"]
    n_classes = problem["n_classes"]

    disjointness = assert_four_way_disjoint(
        audit["server_ref_idx"], audit["client_idx"], audit["X_train"], audit["X_val"], audit["X_test"]
    )

    client_stats = []
    for cid, (Xc, yc) in enumerate(problem["partitions"]):
        client_stats.append({"client_id": cid, "n_samples": int(len(yc)), "class_counts": _class_counts(yc, n_classes)})

    return {
        "partition": partition,
        "alpha": alpha if partition == "dirichlet" else None,
        "n_clients": n_clients,
        "seed": seed,
        "label_names": audit["info"]["label"],
        "n_classes": n_classes,
        "official_split_sizes": {
            "train": int(len(audit["y_train"])),
            "val": int(len(audit["y_val"])),
            "test": int(len(audit["y_test"])),
        },
        "server_ref": {
            "n_samples": int(len(audit["server_ref_idx"])),
            "class_counts": _class_counts(problem["server_ref"][1], n_classes),
        },
        "calibration_(val)": {
            "n_samples": int(len(audit["y_val"])),
            "class_counts": _class_counts(audit["y_val"], n_classes),
        },
        "test": {
            "n_samples": int(len(audit["y_test"])),
            "class_counts": _class_counts(audit["y_test"], n_classes),
        },
        "client_training": client_stats,
        "disjointness_proof": disjointness,
    }


def main() -> None:
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    os.makedirs(OUT_DIR, exist_ok=True)

    report = {
        "iid": _audit_one("iid", alpha=None, n_clients=5, seed=42),
        "dirichlet": _audit_one("dirichlet", alpha=alpha, n_clients=5, seed=42),
    }

    with open(os.path.join(OUT_DIR, "dataset_audit.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    lines = ["# PathMNIST dataset audit\n"]
    for key in ("iid", "dirichlet"):
        r = report[key]
        lines.append(f"## Partition: {key}" + (f" (alpha={r['alpha']})" if r["alpha"] is not None else ""))
        lines.append(f"- official split sizes: {r['official_split_sizes']}")
        lines.append(f"- server_ref: {r['server_ref']['n_samples']} samples, class_counts={r['server_ref']['class_counts']}")
        lines.append(f"- calibration (=val, untouched): {r['calibration_(val)']['n_samples']} samples, class_counts={r['calibration_(val)']['class_counts']}")
        lines.append(f"- test (untouched): {r['test']['n_samples']} samples, class_counts={r['test']['class_counts']}")
        lines.append("- per-client:")
        for c in r["client_training"]:
            lines.append(f"  - client {c['client_id']}: n={c['n_samples']}, class_counts={c['class_counts']}")
        lines.append(f"- disjointness proof (all overlap counts must be 0): {r['disjointness_proof']}")
        lines.append("")

    with open(os.path.join(OUT_DIR, "dataset_audit.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(json.dumps({k: v["disjointness_proof"]["all_zero"] for k, v in report.items()}, indent=2))
    print(f"Saved {OUT_DIR}/dataset_audit.json and dataset_audit.md")


if __name__ == "__main__":
    main()
