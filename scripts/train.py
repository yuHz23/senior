"""Train the GRU Autoencoder for a given dataset.

Usage:
    python scripts/train.py --dataset ciciot --config configs/config.yaml
    python scripts/train.py --dataset intel   --config configs/config.yaml
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch

from src.models.lstm_autoencoder import GRUAutoencoder
from src.models.threshold import compute_threshold, save_threshold
from src.training.trainer import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ciciot", "intel"], required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dcfg = cfg["data"][args.dataset]
    mcfg = cfg["model"]
    tcfg = cfg["training"]

    if args.dataset == "ciciot":
        from src.data.ciciot_loader import load_dataset, make_windows
    else:
        from src.data.intel_loader import load_dataset, make_windows

    print(f"Loading {args.dataset} dataset...")
    X_train_raw, X_val_raw, X_test_raw, y_test, _ = load_dataset(
        csv_path=dcfg["raw_path"],
        scaler_path=dcfg["scaler_path"],
        val_split=tcfg["val_split"],
    )

    print("Building windows...")
    X_train = make_windows(X_train_raw, mcfg["window_size"])
    X_val = make_windows(X_val_raw, mcfg["window_size"])
    print(f"  train windows: {X_train.shape}, val windows: {X_val.shape}")

    n_features = X_train.shape[2]
    model = GRUAutoencoder(
        input_size=n_features,
        hidden_size=mcfg["hidden_size"],
        bottleneck_size=mcfg["bottleneck_size"],
        seq_len=mcfg["window_size"],
        num_layers=mcfg["num_layers"],
        dropout=mcfg["dropout"],
    )

    model_path = cfg["saved_models"][f"{args.dataset}_model"]
    history = train(
        model=model,
        X_train=X_train,
        X_val=X_val,
        save_path=model_path,
        epochs=tcfg["epochs"],
        batch_size=tcfg["batch_size"],
        lr=tcfg["learning_rate"],
        patience=tcfg["early_stopping_patience"],
        device=args.device,
    )

    print("Computing anomaly threshold from validation set...")
    model.load_state_dict(torch.load(model_path, map_location=args.device))
    pct_key = f"{args.dataset}_percentile"
    percentile = cfg["threshold"].get(pct_key, cfg["threshold"].get("percentile", 99))
    threshold_val = compute_threshold(
        model, X_val, percentile=percentile, device=args.device
    )
    threshold_path = cfg["saved_models"][f"{args.dataset}_threshold"]
    save_threshold(threshold_val, threshold_path)
    print(f"Threshold ({percentile}th pct): {threshold_val:.6f} → {threshold_path}")


if __name__ == "__main__":
    main()
