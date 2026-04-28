import json
import os
import numpy as np
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.models.lstm_autoencoder import GRUAutoencoder


def compute_threshold(
    model: "GRUAutoencoder",
    val_windows: np.ndarray,
    percentile: float = 99.0,
    batch_size: int = 256,
    device: str = "cpu",
) -> float:
    """Compute anomaly threshold from Normal validation windows.

    Returns the `percentile`-th percentile of reconstruction errors on
    Normal data. Samples exceeding this value at inference are flagged ATTACK.
    """
    model.eval()
    errors = []
    with torch.no_grad():
        for i in range(0, len(val_windows), batch_size):
            batch = torch.tensor(val_windows[i : i + batch_size], dtype=torch.float32).to(device)
            err = model.reconstruction_error(batch).cpu().numpy()
            errors.append(err)
    errors = np.concatenate(errors)
    return float(np.percentile(errors, percentile))


def save_threshold(value: float, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"threshold": value}, f)


def load_threshold(path: str) -> float:
    with open(path) as f:
        return float(json.load(f)["threshold"])
