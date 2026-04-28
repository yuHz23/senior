from collections import deque
from typing import Optional
import numpy as np
import joblib
from sklearn.preprocessing import MaxAbsScaler


def fit_scaler(normal_data: np.ndarray, save_path: str) -> MaxAbsScaler:
    scaler = MaxAbsScaler()
    scaler.fit(normal_data)
    joblib.dump(scaler, save_path)
    return scaler


def load_scaler(path: str) -> MaxAbsScaler:
    return joblib.load(path)


class WindowBuffer:
    """Accumulates samples and yields a full window once filled (sliding window)."""

    def __init__(self, window_size: int = 64):
        self.buffer = deque(maxlen=window_size)
        self.window_size = window_size

    def push(self, sample: np.ndarray) -> Optional[np.ndarray]:
        self.buffer.append(sample)
        if len(self.buffer) == self.window_size:
            return np.array(self.buffer, dtype=np.float32)  # (window_size, n_features)
        return None

    def reset(self):
        self.buffer.clear()
