"""STEP 5 sanity check: centralized (non-federated) training of the new
PathMNISTCNN on the full official train split, evaluated on official
val/test. Purpose: verify the model/data pipeline can learn at all before
trusting any federated number. No attacks, no defenses, no calibration.

Usage: python scripts/pathmnist_centralized_sanity.py [n_epochs] [device]
Writes results/pathmnist/centralized_sanity.json
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from benchmark.datasets.pathmnist import _medmnist_split  # noqa: E402
from benchmark.models.torch_cnn import TorchCNNFlat  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "pathmnist")


def main() -> None:
    n_epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    device = sys.argv[2] if len(sys.argv) > 2 else "cuda"
    batch_size = 256
    lr = 0.01
    seed = 42

    print(f"Loading PathMNIST official splits (device={device}) ...")
    X_train, y_train, info = _medmnist_split("train")
    X_val, y_val, _ = _medmnist_split("val")
    X_test, y_test, _ = _medmnist_split("test")
    n_classes = len(info["label"])
    print(f"train={len(y_train)} val={len(y_val)} test={len(y_test)} n_classes={n_classes}")

    model = TorchCNNFlat(n_classes=n_classes, seed=seed, device=device)
    print(f"model n_params={model.n_params()} device={model.device}")

    rng = np.random.default_rng(seed)
    n = len(X_train)

    history = []
    for epoch in range(n_epochs):
        t0 = time.perf_counter()
        perm = rng.permutation(n)
        running_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = perm[start:end]
            grad_vec, loss = model.train_step(X_train[idx], y_train[idx], lr)
            new_weights = model.get_weights() - lr * grad_vec
            model.set_weights(new_weights)
            running_loss += loss
            n_batches += 1

        val_acc, val_loss = model.evaluate(X_val, y_val)
        test_acc, test_loss = model.evaluate(X_test, y_test)
        dt = time.perf_counter() - t0
        row = {
            "epoch": epoch,
            "train_loss_mean": running_loss / n_batches,
            "val_accuracy": val_acc,
            "val_loss": val_loss,
            "test_accuracy": test_acc,
            "test_loss": test_loss,
            "epoch_time_s": dt,
        }
        history.append(row)
        print(
            f"epoch {epoch+1}/{n_epochs}: train_loss={row['train_loss_mean']:.4f} "
            f"val_acc={val_acc:.4f} test_acc={test_acc:.4f} time={dt:.1f}s"
        )

    os.makedirs(OUT_DIR, exist_ok=True)
    out = {
        "device": model.device,
        "n_params": model.n_params(),
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
        "history": history,
        "chance_accuracy": 1.0 / n_classes,
    }
    path = os.path.join(OUT_DIR, "centralized_sanity.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {path}")
    print(f"chance accuracy = {out['chance_accuracy']:.4f}; final test_acc = {history[-1]['test_accuracy']:.4f}")


if __name__ == "__main__":
    main()
