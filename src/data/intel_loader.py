import os
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.preprocessor import fit_scaler, load_scaler, WindowBuffer

FEATURES = ["temperature", "humidity", "light", "voltage"]

# Intel Berkeley data has no explicit attack labels.
# Faulty readings are identified by extreme outliers (e.g., temp > 100 or < -20).
FAULT_RULES = {
    "temperature": (-20.0, 100.0),
    "humidity": (0.0, 100.0),
    "light": (0.0, 3000.0),
    "voltage": (2.0, 3.5),
}


def _is_faulty(row: pd.Series) -> bool:
    for col, (lo, hi) in FAULT_RULES.items():
        if col in row and (row[col] < lo or row[col] > hi):
            return True
    return False


def load_dataset(
    csv_path: str,
    scaler_path: str,
    val_split: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load Intel Berkeley CSV and return (X_train, X_val, X_test, y_test, attack_labels_test).

    y_test: 0 = clean reading, 1 = faulty sensor reading (anomaly proxy).
    attack_labels_test: "Normal" or "SensorFault" per sample.
    """
    # File is whitespace-separated with no header row
    df = pd.read_csv(
        csv_path,
        sep=r"\s+",
        header=None,
        names=["date", "time", "epoch", "moteid", "temperature", "humidity", "light", "voltage"],
        engine="python",
    )
    df = df[FEATURES].apply(pd.to_numeric, errors="coerce").ffill().bfill().dropna()

    fault_mask = df.apply(_is_faulty, axis=1)
    normal = df[~fault_mask].values.astype(np.float32)
    faulty = df[fault_mask].values.astype(np.float32)

    X_train, X_val = train_test_split(normal, test_size=val_split, random_state=42)

    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    scaler = fit_scaler(X_train, scaler_path)

    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)

    X_test = scaler.transform(np.concatenate([normal, faulty], axis=0))
    y_test = np.array([0] * len(normal) + [1] * len(faulty), dtype=np.int32)
    attack_labels_test = np.array(
        ["Normal"] * len(normal) + ["SensorFault"] * len(faulty)
    )

    return X_train, X_val, X_test, y_test, attack_labels_test


def make_windows(data: np.ndarray, window_size: int = 64) -> np.ndarray:
    n = max(0, len(data) - window_size + 1)
    if n == 0:
        return np.empty((0, window_size, data.shape[1]), dtype=np.float32)
    out = np.empty((n, window_size, data.shape[1]), dtype=np.float32)
    for i in range(n):
        out[i] = data[i : i + window_size]
    return out
