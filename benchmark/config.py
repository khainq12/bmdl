"""Shared experiment configuration schema for the gradient-defense benchmark.

See docs/BENCHMARK_PROTOCOL.md for the locked protocol this schema encodes.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclasses.dataclass
class DatasetConfig:
    name: str = "synthetic"  # "synthetic" | "pathmnist" | "fed_isic2019"
    n_clients: int = 5
    partition: str = "dirichlet"  # "iid" | "dirichlet"
    dirichlet_alpha: float = 0.5
    server_ref_fraction: float = 0.10
    calib_fraction: float = 0.15
    seed: int = 42


@dataclasses.dataclass
class ModelConfig:
    name: str = "simple_mlp"  # "simple_mlp" | "torch_cnn" (STEP 5/6)
    hidden: tuple = (64, 32)


@dataclasses.dataclass
class OptimConfig:
    local_epochs: int = 2
    local_lr: float = 0.05
    batch_size: int = 32
    n_rounds: int = 10


@dataclasses.dataclass
class AttackConfig:
    name: str = "no_attack"
    malicious_ratio: float = 0.0
    attack_from_round: int = 3
    params: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class DefenseConfig:
    name: str = "fedavg"
    params: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class CalibrationConfig:
    calib_rounds: int = 5
    fpr_target: float = 0.05
    calib_seed: int = 1000


@dataclasses.dataclass
class RunConfig:
    dataset: DatasetConfig
    model: ModelConfig
    optim: OptimConfig
    attack: AttackConfig
    defense: DefenseConfig
    calibration: CalibrationConfig
    seed: int = 42
    device: str = "cpu"

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @staticmethod
    def from_yaml(path: str) -> "RunConfig":
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return RunConfig(
            dataset=DatasetConfig(**raw.get("dataset", {})),
            model=ModelConfig(**raw.get("model", {})),
            optim=OptimConfig(**raw.get("optim", {})),
            attack=AttackConfig(**raw.get("attack", {})),
            defense=DefenseConfig(**raw.get("defense", {})),
            calibration=CalibrationConfig(**raw.get("calibration", {})),
            seed=raw.get("seed", 42),
            device=raw.get("device", "cpu"),
        )

    def save_json(self, path: str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, default=str))
