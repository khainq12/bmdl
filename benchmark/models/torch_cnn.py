"""Small CUDA-capable CNN for 28x28 RGB PathMNIST, wrapped in the
SimpleMLP-compatible flat-vector API (get_weights/set_weights/train_step/
evaluate/n_params) so it is a drop-in model for
benchmark.models.local_training_seeded and benchmark.runner without
changing either. Used only for the gradient-defense arms (FedAvg, Norm,
Cosine, Sign Consensus, Median, Multi-Krum) — per
docs/BENCHMARK_PROTOCOL.md Sec 1/2, the crypto stack is not run against
this model by default.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PathMNISTCNN(nn.Module):
    """Conv(3->16)->Pool->Conv(16->32)->Pool->FC(1568->128)->FC(128->n_classes).
    28x28 -> 14x14 -> 7x7, matches PathMNIST's native resolution exactly
    (no projection, consistent with docs/IMPLEMENTATION_PLAN.md's "no
    compact head / no projection" preference for the new real-dataset path).
    """

    def __init__(self, n_classes: int = 9):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def resolve_device(device: Optional[str]) -> str:
    if device in (None, "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but torch.cuda.is_available() is False")
    return device


class TorchCNNFlat:
    """Flat-vector adapter: matches fl_core.model.SimpleMLP's contract
    (get_weights/set_weights/train_step/evaluate/n_params) exactly, so
    benchmark.models.local_training_seeded works on this model unchanged.

    train_step returns the raw backward-pass gradient as a flat vector; it
    does NOT apply the update itself (local_training_seeded does
    `weights - lr * grad_vec` externally) — same division of responsibility
    as SimpleMLP.train_step.

    Inputs to train_step/evaluate are expected as uint8 arrays shaped
    (N, 3, 28, 28); normalization to [0,1] happens internally.
    """

    def __init__(self, n_classes: int = 9, seed: int = 42, device: Optional[str] = None):
        self.device = resolve_device(device)
        torch.manual_seed(seed)
        self.model = PathMNISTCNN(n_classes).to(self.device)
        self.n_classes = n_classes
        self._shapes = [tuple(p.shape) for p in self.model.parameters()]
        self._numel = [p.numel() for p in self.model.parameters()]

    def n_params(self) -> int:
        return int(sum(self._numel))

    def get_weights(self) -> np.ndarray:
        with torch.no_grad():
            return np.concatenate(
                [p.detach().cpu().numpy().ravel().astype(np.float64) for p in self.model.parameters()]
            )

    def set_weights(self, flat: np.ndarray) -> None:
        idx = 0
        with torch.no_grad():
            for p, n, shape in zip(self.model.parameters(), self._numel, self._shapes):
                chunk = np.asarray(flat[idx : idx + n], dtype=np.float32).reshape(shape)
                p.copy_(torch.from_numpy(chunk).to(self.device))
                idx += n

    def _to_tensor(self, X: np.ndarray) -> torch.Tensor:
        xb = torch.from_numpy(np.asarray(X)).to(self.device)
        if xb.dtype != torch.float32:
            xb = xb.float()
        if float(xb.max()) > 1.5:  # raw uint8-range pixels, not yet normalized
            xb = xb / 255.0
        return xb

    def train_step(self, X: np.ndarray, y: np.ndarray, lr: float) -> Tuple[np.ndarray, float]:
        self.model.train()
        xb = self._to_tensor(X)
        yb = torch.from_numpy(np.asarray(y, dtype=np.int64)).to(self.device)
        self.model.zero_grad(set_to_none=True)
        logits = self.model(xb)
        loss = F.cross_entropy(logits, yb)
        loss.backward()
        grads = []
        for p, n in zip(self.model.parameters(), self._numel):
            g = p.grad
            grads.append(g.detach().cpu().numpy().ravel().astype(np.float64) if g is not None else np.zeros(n))
        grad_vec = np.concatenate(grads)
        return grad_vec, float(loss.item())

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        self.model.eval()
        with torch.no_grad():
            xb = self._to_tensor(X)
            yb = torch.from_numpy(np.asarray(y, dtype=np.int64)).to(self.device)
            logits = self.model(xb)
            loss = float(F.cross_entropy(logits, yb).item())
            acc = float((logits.argmax(dim=1) == yb).float().mean().item())
        return acc, loss
