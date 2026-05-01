from collections import deque
from typing import Optional
import numpy as np
import json
import os


class MaxAbsScaler:
    """Lightweight MaxAbsScaler without sklearn dependency."""

    def __init__(self, max_abs_: np.ndarray = None):
        self.max_abs_ = max_abs_
        self.scale_   = None

    def _ensure_scale(self):
        if self.scale_ is None and self.max_abs_ is not None:
            self.scale_ = np.where(self.max_abs_ == 0, 1.0, self.max_abs_)

    def fit(self, X: np.ndarray):
        self.max_abs_ = np.max(np.abs(X), axis=0).astype(np.float64)
        self.scale_   = None
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        self._ensure_scale()
        return X / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


def fit_scaler(normal_data: np.ndarray, save_path: str) -> MaxAbsScaler:
    scaler = MaxAbsScaler()
    scaler.fit(normal_data)
    _save_scaler(scaler, save_path)
    return scaler


def _save_scaler(scaler: MaxAbsScaler, save_path: str) -> None:
    if save_path.endswith(".json"):
        with open(save_path, "w") as f:
            json.dump({"type": "MaxAbsScaler", "max_abs_": scaler.max_abs_.tolist()}, f)
    else:
        try:
            import joblib
            joblib.dump(scaler, save_path)
        except ImportError:
            json_path = os.path.splitext(save_path)[0] + ".json"
            _save_scaler(scaler, json_path)


def load_scaler(path: str) -> MaxAbsScaler:
    """Load from JSON (preferred) or pickle fallback."""
    if path.endswith(".json") or os.path.exists(path + ".json"):
        json_path = path if path.endswith(".json") else path + ".json"
        with open(json_path) as f:
            d = json.load(f)
        return MaxAbsScaler(np.array(d["max_abs_"], dtype=np.float64))
    else:
        try:
            import joblib
            scaler = joblib.load(path)
            # Wrap sklearn scaler in our class if needed
            if not hasattr(scaler, "scale_"):
                scaler.scale_ = np.where(scaler.max_abs_ == 0, 1.0, scaler.max_abs_)
            return scaler
        except ImportError:
            raise RuntimeError(
                f"scaler load failed: neither JSON nor joblib available for {path}"
            )


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
