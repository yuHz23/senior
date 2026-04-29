import numpy as np
import onnxruntime as ort

from src.models.threshold import load_threshold
from src.data.preprocessor import load_scaler


class OnnxAnomalyDetector:
    """ONNX Runtime–based anomaly detector. Drop-in replacement for AnomalyDetector."""

    def __init__(self, model_path: str, scaler_path: str, threshold_path: str, **_):
        self.scaler    = load_scaler(scaler_path)
        self.threshold = load_threshold(threshold_path)

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

    def run_batch(self, x_np: np.ndarray):
        """x_np: (batch, seq_len, n_features) float32 → returns (z, x_hat) as numpy arrays."""
        return self.session.run(None, {"x": x_np})

    def detect(self, window: np.ndarray) -> dict:
        """window: (seq_len, n_features) raw (unscaled) numpy array."""
        scaled = self.scaler.transform(window).astype(np.float32)
        x = scaled[np.newaxis]
        z, x_hat = self.run_batch(x)
        error = float(np.mean((x - x_hat) ** 2))
        return {
            "label":                "ATTACK" if error > self.threshold else "NORMAL",
            "reconstruction_error": round(error, 6),
            "threshold":            self.threshold,
            "semantic_vector":      z[0].tolist(),
        }

    def detect_scaled(self, window_scaled: np.ndarray) -> dict:
        """Like detect() but input is already scaled."""
        x = window_scaled.astype(np.float32)[np.newaxis]
        z, x_hat = self.run_batch(x)
        error = float(np.mean((x - x_hat) ** 2))
        return {
            "label":                "ATTACK" if error > self.threshold else "NORMAL",
            "reconstruction_error": round(error, 6),
            "threshold":            self.threshold,
            "semantic_vector":      z[0].tolist(),
        }

    def reconstruct(self, window: np.ndarray) -> dict:
        scaled = self.scaler.transform(window).astype(np.float32)
        x = scaled[np.newaxis]
        z, x_hat = self.run_batch(x)
        error = float(np.mean((x - x_hat) ** 2))
        return {
            "label":                "ATTACK" if error > self.threshold else "NORMAL",
            "reconstruction_error": round(error, 6),
            "threshold":            self.threshold,
            "semantic_vector":      z[0].tolist(),
            "input_scaled":         x[0].tolist(),
            "reconstructed_scaled": x_hat[0].tolist(),
        }
