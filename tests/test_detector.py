import numpy as np
import pytest
import tempfile
import os
import torch

from src.models.lstm_autoencoder import GRUAutoencoder
from src.models.threshold import save_threshold, compute_threshold
from src.data.preprocessor import fit_scaler


def _make_temp_model(input_size=5, seq_len=64):
    model = GRUAutoencoder(input_size=input_size, hidden_size=32, bottleneck_size=8, seq_len=seq_len)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    torch.save(model.state_dict(), path)
    return model, path


def test_model_forward_shape():
    model = GRUAutoencoder(input_size=5, hidden_size=32, bottleneck_size=8, seq_len=64)
    x = torch.randn(4, 64, 5)
    out = model(x)
    assert out.shape == x.shape


def test_model_encode_shape():
    model = GRUAutoencoder(input_size=5, hidden_size=32, bottleneck_size=8, seq_len=64)
    x = torch.randn(4, 64, 5)
    z = model.encode(x)
    assert z.shape == (4, 8)


def test_reconstruction_error_shape():
    model = GRUAutoencoder(input_size=5, hidden_size=32, bottleneck_size=8, seq_len=64)
    x = torch.randn(8, 64, 5)
    err = model.reconstruction_error(x)
    assert err.shape == (8,)
    assert (err >= 0).all()


def test_detector_output_keys():
    from src.inference.detector import AnomalyDetector

    model, model_path = _make_temp_model(input_size=5, seq_len=64)

    data = np.random.randn(100, 5).astype(np.float32)
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        scaler_path = f.name
    fit_scaler(data, scaler_path)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        threshold_path = f.name
    save_threshold(0.5, threshold_path)

    try:
        detector = AnomalyDetector(
            model_path=model_path,
            scaler_path=scaler_path,
            threshold_path=threshold_path,
            input_size=5,
            hidden_size=32,
            bottleneck_size=8,
            seq_len=64,
        )
        window = np.random.randn(64, 5).astype(np.float32)
        result = detector.detect(window)
        assert "label" in result
        assert result["label"] in ("NORMAL", "ATTACK")
        assert "reconstruction_error" in result
        assert "semantic_vector" in result
        assert len(result["semantic_vector"]) == 8
    finally:
        os.unlink(model_path)
        os.unlink(scaler_path)
        os.unlink(threshold_path)
