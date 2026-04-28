"""Ablation study: compare GRU, LSTM, and MLP autoencoders on CICIoT.

Usage:
    python scripts/compare_models.py --dataset ciciot
    python scripts/compare_models.py --dataset intel
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import yaml
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score

from src.models.lstm_autoencoder import GRUAutoencoder
from src.models.variants import LSTMAutoencoder, MLPAutoencoder
from src.models.threshold import compute_threshold
from src.training.trainer import train


MODELS = {
    "GRU  Autoencoder": GRUAutoencoder,
    "LSTM Autoencoder": LSTMAutoencoder,
    "MLP  Autoencoder": MLPAutoencoder,
}


def compute_errors_batch(model, X_scaled, window_size, batch_size=256, device="cpu"):
    model.eval()
    N, F = X_scaled.shape
    n = max(0, N - window_size + 1)
    if n == 0:
        return np.empty(0, dtype=np.float32)
    data    = np.ascontiguousarray(X_scaled)
    shape   = (n, window_size, F)
    strides = (data.strides[0], data.strides[0], data.strides[1])
    windows = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
    errors  = np.empty(n, dtype=np.float32)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = torch.tensor(windows[i : i + batch_size].copy(), dtype=torch.float32).to(device)
            errors[i : i + batch_size] = model.reconstruction_error(batch).cpu().numpy()
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ciciot", "intel"], default="ciciot")
    parser.add_argument("--config",  default="configs/config.yaml")
    parser.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit",   type=int, default=200_000)
    parser.add_argument("--epochs",  type=int, default=None, help="Override max epochs")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dcfg = cfg["data"][args.dataset]
    mcfg = cfg["model"]
    tcfg = cfg["training"]
    epochs = args.epochs or tcfg["epochs"]

    if args.dataset == "ciciot":
        from src.data.ciciot_loader import load_dataset, make_windows
    else:
        from src.data.intel_loader import load_dataset, make_windows

    print(f"Loading {args.dataset}...")
    X_train_raw, X_val_raw, X_test, y_test_raw, _ = load_dataset(
        csv_path=dcfg["raw_path"],
        scaler_path=dcfg["scaler_path"],
        val_split=tcfg["val_split"],
    )

    W = mcfg["window_size"]
    X_train = make_windows(X_train_raw, W)
    X_val   = make_windows(X_val_raw,   W)
    print(f"Train windows: {X_train.shape}, Val windows: {X_val.shape}")

    # Balanced subsample test
    if args.limit > 0 and len(X_test) > args.limit:
        rng = np.random.default_rng(42)
        half = args.limit // 2
        ni = rng.choice(np.where(y_test_raw == 0)[0], min(half, (y_test_raw==0).sum()), replace=False)
        ai = rng.choice(np.where(y_test_raw == 1)[0], min(half, (y_test_raw==1).sum()), replace=False)
        sel = np.sort(np.concatenate([ni, ai]))
        X_test     = X_test[sel]
        y_test_raw = y_test_raw[sel]

    n_windows_test = max(0, len(X_test) - W + 1)
    y_test = y_test_raw[W - 1 : W - 1 + n_windows_test]
    n_features = X_train.shape[2]

    results = {}
    pct_key  = f"{args.dataset}_percentile"
    percentile = cfg["threshold"].get(pct_key, cfg["threshold"].get("percentile", 99))

    for name, ModelClass in MODELS.items():
        print(f"\n{'='*50}")
        print(f"Training: {name}")
        print(f"{'='*50}")

        model = ModelClass(
            input_size=n_features,
            hidden_size=mcfg["hidden_size"],
            bottleneck_size=mcfg["bottleneck_size"],
            seq_len=W,
            num_layers=mcfg["num_layers"],
            dropout=mcfg["dropout"],
        )
        n_params = sum(p.numel() for p in model.parameters())

        tmp_path = f"saved_models/_ablation_{name.strip()}.pt"
        t0 = time.time()
        train(
            model=model, X_train=X_train, X_val=X_val,
            save_path=tmp_path, epochs=epochs,
            batch_size=tcfg["batch_size"], lr=tcfg["learning_rate"],
            patience=tcfg["early_stopping_patience"], device=args.device,
        )
        train_time = time.time() - t0

        model.load_state_dict(torch.load(tmp_path, map_location=args.device))
        threshold = compute_threshold(model, X_val, percentile=percentile, device=args.device)

        t1 = time.time()
        errors = compute_errors_batch(model, X_test, W, device=args.device)
        infer_time = time.time() - t1

        y_pred = (errors > threshold).astype(int)
        f1  = f1_score(y_test, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_test, errors)
        except Exception:
            auc = float("nan")
        fp = int(((y_pred == 1) & (y_test == 0)).sum())
        fn = int(((y_pred == 0) & (y_test == 1)).sum())

        results[name] = {
            "params":      n_params,
            "train_sec":   round(train_time, 1),
            "infer_sec":   round(infer_time, 2),
            "threshold":   round(threshold, 6),
            "f1":          round(f1, 4),
            "auc":         round(auc, 4),
            "fp":          fp,
            "fn":          fn,
        }
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # ── Print comparison table ────────────────────────���─────────────
    print(f"\n{'='*80}")
    print(f"ABLATION STUDY RESULTS — {args.dataset.upper()}")
    print(f"{'='*80}")
    header = f"{'Model':<22} {'Params':>8} {'Train(s)':>10} {'Infer(s)':>10} {'F1':>7} {'AUC':>7} {'FP':>7} {'FN':>5}"
    print(header)
    print("-" * 80)
    for name, r in results.items():
        print(
            f"{name:<22} {r['params']:>8,} {r['train_sec']:>10.1f} "
            f"{r['infer_sec']:>10.2f} {r['f1']:>7.4f} {r['auc']:>7.4f} "
            f"{r['fp']:>7} {r['fn']:>5}"
        )
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
