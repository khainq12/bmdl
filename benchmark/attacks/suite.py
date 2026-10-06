"""The six required attack scenarios (plus no_attack), exactly as locked in
docs/BENCHMARK_PROTOCOL.md Sec 4. Attack strength is always explicit in
`params` and recorded in the run config — never hidden inside a function
default that silently varies between call sites.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np


def no_attack(delta: np.ndarray, rng: np.random.Generator, params: Dict[str, Any]) -> np.ndarray:
    return delta


def large_norm(delta: np.ndarray, rng: np.random.Generator, params: Dict[str, Any]) -> np.ndarray:
    scale = float(params.get("scale", 50.0))
    return rng.normal(0.0, scale, size=delta.shape)


def low_norm(delta: np.ndarray, rng: np.random.Generator, params: Dict[str, Any]) -> np.ndarray:
    """Free-rider / stealth attack: scale the honest update toward zero."""
    eta = float(params.get("eta", 0.01))
    return delta * eta


def full_sign_flip(delta: np.ndarray, rng: np.random.Generator, params: Dict[str, Any]) -> np.ndarray:
    """Negates every coordinate — distinct from directional_poisoning below."""
    return -delta


def directional_poisoning(delta: np.ndarray, rng: np.random.Generator, params: Dict[str, Any]) -> np.ndarray:
    """Opposite direction, magnitude pinned to a target norm (default: the
    honest update's own norm, or the caller-supplied calibrated tau)."""
    target_norm = params.get("target_norm")
    direction = -delta
    n = float(np.linalg.norm(direction))
    if n < 1e-12:
        direction = rng.normal(0.0, 1.0, size=delta.shape)
        n = float(np.linalg.norm(direction))
    if target_norm is None:
        target_norm = float(np.linalg.norm(delta))
    return direction * (float(target_norm) / n)


def sparse_coordinate_attack(delta: np.ndarray, rng: np.random.Generator, params: Dict[str, Any]) -> np.ndarray:
    """Zero all but k coordinates, spike those to +-spike."""
    d = delta.shape[0]
    k = int(params.get("k", max(1, d // 100)))
    k = min(k, d)
    selection = params.get("selection", "topk")  # "topk" | "random"
    spike = params.get("spike")
    if spike is None:
        spike = 10.0 * float(np.max(np.abs(delta)) + 1e-8)
    spike = float(spike)

    out = np.zeros_like(delta)
    if selection == "random":
        idx = rng.choice(d, size=k, replace=False)
    else:
        idx = np.argsort(-np.abs(delta))[:k]
    signs = rng.choice([-1.0, 1.0], size=k)
    out[idx] = spike * signs
    return out


ATTACKS: Dict[str, Callable[[np.ndarray, np.random.Generator, Dict[str, Any]], np.ndarray]] = {
    "no_attack": no_attack,
    "large_norm": large_norm,
    "low_norm": low_norm,
    "full_sign_flip": full_sign_flip,
    "directional_poisoning": directional_poisoning,
    "sparse_coordinate_attack": sparse_coordinate_attack,
}


def apply_attack(
    name: str, delta: np.ndarray, rng: np.random.Generator, params: Dict[str, Any] | None = None
) -> np.ndarray:
    if name not in ATTACKS:
        raise ValueError(f"unknown attack '{name}'; registered: {sorted(ATTACKS)}")
    return ATTACKS[name](delta, rng, params or {})
