import numpy as np
import torch

from src.models.lstm_autoencoder import GRUAutoencoder
from src.models.threshold import load_threshold
from src.data.preprocessor import load_scaler


class AnomalyDetector:
    """Loads a trained model, scaler, and threshold; infers one window at a time."""

    def __init__(
        self,
        model_path: str,
        scaler_path: str,
        threshold_path: str,
        input_size: int,
        hidden_size: int = 128,
        bottleneck_size: int = 32,
        seq_len: int = 64,
        num_layers: int = 1,
        device: str = "cpu",
    ):
        self.device = device
        self.scaler = load_scaler(scaler_path)
        self.threshold = load_threshold(threshold_path)

        self.model = GRUAutoencoder(
            input_size=input_size,
            hidden_size=hidden_size,
            bottleneck_size=bottleneck_size,
            seq_len=seq_len,
            num_layers=num_layers,
        ).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

    def detect(self, window: np.ndarray) -> dict:
        """Infer one window.

        Args:
            window: (seq_len, n_features) raw (unscaled) numpy array

        Returns:
            dict with keys: label, reconstruction_error, semantic_vector
        """
        scaled = self.scaler.transform(window)  # (seq_len, n_features)
        x = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0).to(self.device)  # (1, seq_len, n_features)

        with torch.no_grad():
            z = self.model.encode(x)
            x_hat = self.model.decode(z)
            error = float(((x - x_hat) ** 2).mean().item())

        return {
            "label": "ATTACK" if error > self.threshold else "NORMAL",
            "reconstruction_error": round(error, 6),
            "threshold": self.threshold,
            "semantic_vector": z.squeeze(0).cpu().numpy().tolist(),
        }

    def detect_scaled(self, window_scaled: np.ndarray) -> dict:
        """Like detect() but skips scaling — input is already scaled."""
        x = torch.tensor(window_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            z = self.model.encode(x)
            x_hat = self.model.decode(z)
            error = float(((x - x_hat) ** 2).mean().item())
        return {
            "label": "ATTACK" if error > self.threshold else "NORMAL",
            "reconstruction_error": round(error, 6),
            "threshold": self.threshold,
            "semantic_vector": z.squeeze(0).cpu().numpy().tolist(),
        }

    def reconstruct(self, window: np.ndarray) -> dict:
        """Return input vs reconstructed signal for visualization.

        Args:
            window: (seq_len, n_features) raw (unscaled) numpy array

        Returns:
            dict with input_scaled, reconstructed_scaled (both seq_len×n_features lists),
            reconstruction_error, label, semantic_vector
        """
        scaled = self.scaler.transform(window)
        x = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            z = self.model.encode(x)
            x_hat = self.model.decode(z)
            error = float(((x - x_hat) ** 2).mean().item())
        return {
            "label": "ATTACK" if error > self.threshold else "NORMAL",
            "reconstruction_error": round(error, 6),
            "threshold": self.threshold,
            "semantic_vector": z.squeeze(0).cpu().numpy().tolist(),
            "input_scaled": x.squeeze(0).cpu().numpy().tolist(),
            "reconstructed_scaled": x_hat.squeeze(0).cpu().numpy().tolist(),
        }
