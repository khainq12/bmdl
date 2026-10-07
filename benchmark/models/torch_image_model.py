"""Flat-vector adapter around FLamby's native Fed-ISIC2019 Baseline model
(EfficientNet-b0, ImageNet-pretrained, 8-class head) -- same
get_weights/set_weights/n_params contract as TorchCNNFlat
(benchmark/models/torch_cnn.py), so every existing defense in
benchmark/defenses/ (including the locked C4-DA v1,
benchmark/defenses/combined.py::C4DriftAware, unmodified) works unchanged:
defenses only ever consume flat delta vectors + ctx, never touch the model.

train_step/evaluate differ from TorchCNNFlat's signature because
Fed-ISIC2019 images live on disk at variable aspect ratios (FLamby's own
preprocessing conserves aspect ratio -- see FLamby README) and cannot be
loaded into one fixed-shape in-memory array the way PathMNIST's 28x28
images were; they take a torch Dataset (FLamby's FedIsic2019 instance, or
a torch.utils.data.Subset of one) plus explicit index lists instead of
raw (X, y) arrays. See local_training_image.py for the matching
fixed-step-count training primitive.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

_IO_POOL = ThreadPoolExecutor(max_workers=8)  # per-item disk/PIL/albumentations fetch is I/O+CPU bound and
# was the actual bottleneck (not the GPU step) -- parallelizing it cuts wall-clock substantially.
# Order is preserved (results collected in submission order), so batch<->label pairing is unaffected.

from flamby.datasets.fed_isic2019.model import Baseline

from .torch_cnn import resolve_device


class FedIsicModel:
    """Flat-vector wrapper around FLamby's Baseline (EfficientNet-b0)."""

    def __init__(self, pretrained: bool = True, seed: int = 42, device: Optional[str] = None):
        self.device = resolve_device(device)
        torch.manual_seed(seed)
        self.model = Baseline(pretrained=pretrained).to(self.device)
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

    def _collate(self, dataset, indices: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        items = list(_IO_POOL.map(lambda i: dataset[int(i)], indices))
        xs = [it[0] for it in items]
        ys = [it[1] for it in items]
        return torch.stack(xs).to(self.device), torch.stack(ys).to(self.device)

    def train_step_batch(self, dataset, indices: np.ndarray, lr: float) -> Tuple[np.ndarray, float]:
        self.model.train()
        xb, yb = self._collate(dataset, indices)
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

    def evaluate(self, dataset, indices: Optional[List[int]] = None, batch_size: int = 32) -> Tuple[float, float]:
        self.model.eval()
        if indices is None:
            indices = np.arange(len(dataset))
        indices = np.asarray(indices)
        n_correct, total_loss, n_total = 0, 0.0, 0
        with torch.no_grad():
            for start in range(0, len(indices), batch_size):
                batch_idx = indices[start : start + batch_size]
                xb, yb = self._collate(dataset, batch_idx)
                logits = self.model(xb)
                loss = F.cross_entropy(logits, yb, reduction="sum")
                total_loss += float(loss.item())
                n_correct += int((logits.argmax(dim=1) == yb).sum().item())
                n_total += len(batch_idx)
        return n_correct / n_total, total_loss / n_total
