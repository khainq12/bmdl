"""Dataset loaders. `synthetic` (debug role) and `pathmnist` (main controlled
benchmark, STEP 5) are wired up. Fed-ISIC2019 is STEP 6 — see
docs/BENCHMARK_PROTOCOL.md Sec 1.
"""

from .pathmnist import load_pathmnist_problem
from .synthetic import load_synthetic_problem

DATASET_LOADERS = {
    "synthetic": load_synthetic_problem,
    "pathmnist": load_pathmnist_problem,
}

__all__ = ["DATASET_LOADERS", "load_synthetic_problem", "load_pathmnist_problem"]
