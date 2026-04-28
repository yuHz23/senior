import os
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.preprocessor import fit_scaler, load_scaler, WindowBuffer

FEATURES = [
    "flow_duration",
    "Rate",
    "Number",
    "IAT",
    "Tot size",
]
LABEL_COL = "label"
NORMAL_LABEL = "BenignTraffic"


def load_dataset(
    csv_path: str,
    scaler_path: str,
    chunksize: int = 10_000,
    val_split: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load CIC IoT CSV and return (X_train, X_val, X_test, y_test, attack_labels_test).

    Training and validation sets contain only Normal samples.
    Test set contains all samples (Normal + Attack) for evaluation.
    attack_labels_test: string attack type per test sample ("Normal", "DDoS-...", etc.)
    Scaler is fit on training split only.
    """
    normal_chunks, attack_chunks, attack_label_chunks = [], [], []

    for chunk in pd.read_csv(csv_path, usecols=FEATURES + [LABEL_COL], chunksize=chunksize):
        chunk = chunk.dropna(subset=FEATURES)
        normal_mask = chunk[LABEL_COL] == NORMAL_LABEL
        normal_chunks.append(chunk[normal_mask][FEATURES].values.astype(np.float32))
        attack_chunk = chunk[~normal_mask]
        attack_chunks.append(attack_chunk[FEATURES].values.astype(np.float32))
        attack_label_chunks.append(attack_chunk[LABEL_COL].values)

    normal = np.concatenate(normal_chunks, axis=0)
    attack = np.concatenate(attack_chunks, axis=0)
    attack_labels = np.concatenate(attack_label_chunks, axis=0)

    X_train, X_val = train_test_split(normal, test_size=val_split, random_state=42)

    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    scaler = fit_scaler(X_train, scaler_path)

    X_train = scaler.transform(X_train)
    X_val   = scaler.transform(X_val)

    X_test = scaler.transform(np.concatenate([normal, attack], axis=0))
    y_test = np.array([0] * len(normal) + [1] * len(attack), dtype=np.int32)
    attack_labels_test = np.concatenate([
        np.array(["Normal"] * len(normal)),
        attack_labels,
    ])

    return X_train, X_val, X_test, y_test, attack_labels_test


def make_windows(data: np.ndarray, window_size: int = 64) -> np.ndarray:
    """Convert a (N, F) array into (M, window_size, F) sliding windows."""
    n = max(0, len(data) - window_size + 1)
    if n == 0:
        return np.empty((0, window_size, data.shape[1]), dtype=np.float32)
    out = np.empty((n, window_size, data.shape[1]), dtype=np.float32)
    for i in range(n):
        out[i] = data[i : i + window_size]
    return out
