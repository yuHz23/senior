"""Export trained GRUAutoencoder weights to ONNX format.

Run once locally (requires torch + onnx installed):
    python scripts/export_onnx.py

Outputs:
    saved_models/ciciot_model.onnx
    saved_models/intel_model.onnx
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.lstm_autoencoder import GRUAutoencoder

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _AutoencoderExportWrapper(nn.Module):
    """Wraps GRUAutoencoder to export both z and x_hat as named outputs."""

    def __init__(self, model: GRUAutoencoder):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        z = self.model.encode(x)
        x_hat = self.model.decode(z)
        return z, x_hat


def export(dataset: str, input_size: int, hidden_size: int = 128,
           bottleneck_size: int = 32, seq_len: int = 64):
    pt_path   = os.path.join(ROOT, "saved_models", f"{dataset}_model.pt")
    onnx_path = os.path.join(ROOT, "saved_models", f"{dataset}_model.onnx")

    model = GRUAutoencoder(
        input_size=input_size,
        hidden_size=hidden_size,
        bottleneck_size=bottleneck_size,
        seq_len=seq_len,
        num_layers=1,
    )
    model.load_state_dict(torch.load(pt_path, map_location="cpu"))
    model.eval()

    wrapper = _AutoencoderExportWrapper(model)
    wrapper.eval()

    dummy = torch.zeros(1, seq_len, input_size)

    torch.onnx.export(
        wrapper,
        dummy,
        onnx_path,
        input_names=["x"],
        output_names=["z", "x_hat"],
        dynamic_axes={"x": {0: "batch"}, "z": {0: "batch"}, "x_hat": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    try:
        import onnx
        onnx.checker.check_model(onnx.load(onnx_path))
        print(f"[OK] {dataset}: {onnx_path}  ({os.path.getsize(onnx_path)/1024:.0f} KB)")
    except ImportError:
        print(f"[OK] {dataset}: {onnx_path}  (install 'onnx' to verify)")


if __name__ == "__main__":
    export("ciciot", input_size=5)
    export("intel",  input_size=4)
    print("\nDone. Commit the .onnx files and push to GitHub.")
