"""CLI: run one benchmark config end-to-end and save standard outputs.

Usage: python scripts/run_one.py configs/<name>.yaml results/synthetic/<run_name>/
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.config import RunConfig  # noqa: E402
from benchmark.metrics import save_run  # noqa: E402
from benchmark.runner import run  # noqa: E402


def main() -> None:
    if len(sys.argv) != 3:
        print("usage: python scripts/run_one.py <config.yaml> <out_dir>")
        raise SystemExit(1)
    cfg = RunConfig.from_yaml(sys.argv[1])
    out = run(cfg)
    save_run(out, sys.argv[2])
    print(f"final_accuracy={out['final_accuracy']:.4f}")
    if out["calibration"]:
        print("calibration:", out["calibration"])


if __name__ == "__main__":
    main()
