"""Standard output-file writer for one benchmark run. See
docs/BENCHMARK_PROTOCOL.md Sec 9 for the locked file schema. Column names
here are the source of truth any later aggregation script must match.
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        open(path, "w", encoding="utf-8").close()
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_run(out: Dict[str, Any], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(out["config"], f, indent=2, default=str)

    metrics = {"final_accuracy": out["final_accuracy"], "calibration": out["calibration"]}
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)

    _write_csv(os.path.join(out_dir, "per_round.csv"), out["per_round"])
    _write_csv(os.path.join(out_dir, "client_scores.csv"), out["client_scores"])
    _write_csv(os.path.join(out_dir, "timing.csv"), out["timing"])
