import numpy as np
import pytest
import tempfile
import os

from src.data.preprocessor import fit_scaler, load_scaler, WindowBuffer


def test_window_buffer_fills():
    buf = WindowBuffer(window_size=4)
    for i in range(3):
        assert buf.push(np.array([float(i)] * 5)) is None
    result = buf.push(np.array([3.0] * 5))
    assert result is not None
    assert result.shape == (4, 5)


def test_window_buffer_slides():
    buf = WindowBuffer(window_size=3)
    for i in range(5):
        buf.push(np.array([float(i)] * 2))
    w = buf.push(np.array([5.0] * 2))
    assert w is not None
    assert w[0, 0] == 3.0  # oldest sample in the window


def test_scaler_normalizes():
    data = np.array([[10.0, -5.0], [20.0, 15.0], [-30.0, 0.0]], dtype=np.float32)
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        path = f.name
    try:
        scaler = fit_scaler(data, path)
        transformed = scaler.transform(data)
        assert np.all(np.abs(transformed) <= 1.0 + 1e-6), "MaxAbsScaler should produce values in [-1, 1]"
    finally:
        os.unlink(path)


def test_scaler_roundtrip():
    data = np.random.randn(100, 5).astype(np.float32)
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        path = f.name
    try:
        fit_scaler(data, path)
        loaded = load_scaler(path)
        result = loaded.transform(data)
        assert result.shape == data.shape
    finally:
        os.unlink(path)
