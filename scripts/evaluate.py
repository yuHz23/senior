"""Evaluate a trained model on the full test set.

Usage:
    python scripts/evaluate.py --dataset ciciot
    python scripts/evaluate.py --dataset intel
    python scripts/evaluate.py --dataset ciciot --per-attack
    python scripts/evaluate.py --dataset ciciot --limit 0        # full dataset
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import numpy as np
import torch
from tqdm import tqdm

from src.models.lstm_autoencoder import GRUAutoencoder
from src.models.threshold import load_threshold
from src.evaluation.metrics import print_evaluation


def compute_errors(model, X_scaled: np.ndarray, window_size: int, batch_size: int = 256, device: str = "cpu") -> np.ndarray:
    """Stream reconstruction errors without materializing the full windows array."""
    model.eval()
    N, F = X_scaled.shape
    n = max(0, N - window_size + 1)
    if n == 0:
        return np.empty(0, dtype=np.float32)

    data = np.ascontiguousarray(X_scaled)
    shape   = (n, window_size, F)
    strides = (data.strides[0], data.strides[0], data.strides[1])
    windows = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)

    errors = np.empty(n, dtype=np.float32)
    with torch.no_grad():
        for i in tqdm(range(0, n, batch_size), desc="Inference", unit="batch"):
            batch = torch.tensor(windows[i : i + batch_size].copy(), dtype=torch.float32).to(device)
            errors[i : i + batch_size] = model.reconstruction_error(batch).cpu().numpy()
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    choices=["ciciot", "intel"], required=True)
    parser.add_argument("--config",     default="configs/config.yaml")
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit",      type=int, default=200_000,
                        help="Max test samples (balanced). 0 = no limit.")
    parser.add_argument("--per-attack", action="store_true",
                        help="Print per-attack-type breakdown (CICIoT only).")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dcfg = cfg["data"][args.dataset]
    mcfg = cfg["model"]
    tcfg = cfg["training"]

    if args.dataset == "ciciot":
        from src.data.ciciot_loader import load_dataset
    else:
        from src.data.intel_loader import load_dataset

    print(f"Loading {args.dataset} dataset...")
    _, _, X_test, y_test_raw, attack_labels_raw = load_dataset(
        csv_path=dcfg["raw_path"],
        scaler_path=dcfg["scaler_path"],
        val_split=tcfg["val_split"],
    )

    # Balanced subsample
    if args.limit > 0 and len(X_test) > args.limit:
        rng = np.random.default_rng(42)
        normal_idx = np.where(y_test_raw == 0)[0]
        attack_idx = np.where(y_test_raw == 1)[0]
        half = args.limit // 2
        sel = np.concatenate([
            rng.choice(normal_idx, size=min(half, len(normal_idx)), replace=False),
            rng.choice(attack_idx, size=min(half, len(attack_idx)), replace=False),
        ])
        sel.sort()
        X_test          = X_test[sel]
        y_test_raw      = y_test_raw[sel]
        attack_labels_raw = attack_labels_raw[sel]
        print(f"Subsampled to {len(X_test):,} rows "
              f"({(y_test_raw==0).sum():,} normal / {(y_test_raw==1).sum():,} attack)")

    W         = mcfg["window_size"]
    n_windows = max(0, len(X_test) - W + 1)
    y_test          = y_test_raw[W - 1 : W - 1 + n_windows]
    attack_labels   = attack_labels_raw[W - 1 : W - 1 + n_windows]

    n_features = X_test.shape[1]
    model = GRUAutoencoder(
        input_size=n_features,
        hidden_size=mcfg["hidden_size"],
        bottleneck_size=mcfg["bottleneck_size"],
        seq_len=W,
        num_layers=mcfg["num_layers"],
    ).to(args.device)
    model.load_state_dict(torch.load(
        cfg["saved_models"][f"{args.dataset}_model"], map_location=args.device
    ))

    threshold = load_threshold(cfg["saved_models"][f"{args.dataset}_threshold"])
    print(f"Threshold: {threshold:.6f}")

    print("Computing reconstruction errors...")
    errors = compute_errors(model, X_test, W, device=args.device)

    print_evaluation(
        errors, y_test, threshold,
        attack_labels=attack_labels if args.per_attack else None,
    )


if __name__ == "__main__":
    main()
