"""Pre-compute evaluation metrics and save to JSON.

Run once after training:
    python scripts/save_metrics.py

Outputs:
    saved_models/ciciot_metrics.json
    saved_models/intel_metrics.json
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score,
    f1_score, accuracy_score, roc_auc_score,
)

from src.models.lstm_autoencoder import GRUAutoencoder
from src.models.threshold import load_threshold
from src.data.preprocessor import load_scaler

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _ap(p):
    return p if os.path.isabs(p) else os.path.join(_root, p)

with open(os.path.join(_root, "configs", "config.yaml")) as f:
    cfg = yaml.safe_load(f)
mcfg = cfg["model"]
W    = mcfg["window_size"]


def _build_errors(data_scaled: np.ndarray, model, device: str = "cpu",
                  batch: int = 512) -> np.ndarray:
    """Sliding-window batch inference. Uses evaluate.py stride pattern (strides from contiguous)."""
    data_c = np.ascontiguousarray(data_scaled)
    n, f   = data_c.shape
    n_win  = max(0, n - W + 1)
    if n_win == 0:
        return np.array([], np.float32)
    shape   = (n_win, W, f)
    strides = (data_c.strides[0], data_c.strides[0], data_c.strides[1])
    wins    = np.lib.stride_tricks.as_strided(data_c, shape=shape, strides=strides)

    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, n_win, batch):
            b   = torch.tensor(wins[s:s+batch].copy(), dtype=torch.float32).to(device)
            z   = model.encode(b)
            xh  = model.decode(z)
            out.append(((b - xh) ** 2).mean(dim=(1, 2)).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def _load_model(dataset: str):
    dcfg  = cfg["data"][dataset]
    scaler    = load_scaler(_ap(dcfg["scaler_path"]))
    threshold = load_threshold(_ap(cfg["saved_models"][f"{dataset}_threshold"]))
    n_feat    = len(dcfg["features"])
    model     = GRUAutoencoder(
        input_size=n_feat,
        hidden_size=mcfg["hidden_size"],
        bottleneck_size=mcfg["bottleneck_size"],
        seq_len=W,
        num_layers=mcfg["num_layers"],
    )
    model.load_state_dict(torch.load(_ap(cfg["saved_models"][f"{dataset}_model"]),
                                     map_location="cpu"))
    return scaler, threshold, model


# ── CICIoT ─────────────────────────────────────────────────────────────────────

def _ciciot():
    t0 = time.time()
    dcfg     = cfg["data"]["ciciot"]
    features = dcfg["features"]
    demo_dir = os.path.join(_root, "data", "demo")
    scaler, threshold, model = _load_model("ciciot")

    # Use demo CSVs (pre-extracted balanced subsets)
    df_norm = pd.read_csv(os.path.join(demo_dir, "ciciot_normal.csv"))
    df_atk  = pd.read_csv(os.path.join(demo_dir, "ciciot_attack.csv"))

    n_scaled = scaler.transform(df_norm[features].ffill().fillna(0).values.astype(np.float32))
    a_scaled = scaler.transform(df_atk[features].ffill().fillna(0).values.astype(np.float32))

    print("  CICIoT normal errors …")
    norm_err = _build_errors(n_scaled, model)
    print(f"    {len(norm_err)} windows, mean={norm_err.mean():.5f}, "
          f"above_thresh={int((norm_err > threshold).sum())}")

    print("  CICIoT attack errors …")
    atk_err = _build_errors(a_scaled, model)
    print(f"    {len(atk_err)} windows, mean={atk_err.mean():.5f}, "
          f"above_thresh={int((atk_err > threshold).sum())}")

    atk_types = df_atk["label"].values[W - 1 : W - 1 + len(atk_err)]
    per_attack = []
    for atype in np.unique(atk_types):
        mask = atk_types == atype
        e    = atk_err[mask]
        hits = int((e > threshold).sum())
        per_attack.append({
            "label":      str(atype),
            "recall":     round(float(hits / len(e)), 4),
            "n":          int(len(e)),
            "mean_error": round(float(e.mean()), 6),
        })
    per_attack.sort(key=lambda x: x["recall"])

    y_true  = np.concatenate([np.zeros(len(norm_err)), np.ones(len(atk_err))])
    y_score = np.concatenate([norm_err, atk_err])
    y_pred  = (y_score > threshold).astype(int)
    cm      = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    result = {
        "dataset":           "ciciot",
        "threshold":         float(threshold),
        "n_normal_windows":  int(len(norm_err)),
        "n_attack_windows":  int(len(atk_err)),
        "auc_roc":           round(float(roc_auc_score(y_true, y_score)), 4),
        "accuracy":          round(float(accuracy_score(y_true, y_pred)), 4),
        "precision":         round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall":            round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1":                round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "confusion_matrix":  {"TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)},
        "error_samples":     {
            "normal": norm_err[:200].tolist(),
            "attack": atk_err[:200].tolist(),
        },
        "per_attack":        per_attack,
        "elapsed_sec":       round(time.time() - t0, 2),
    }
    out = os.path.join(_root, "saved_models", "ciciot_metrics.json")
    with open(out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"  Saved: {out}")
    print(f"    AUC={result['auc_roc']}  Acc={result['accuracy']}  "
          f"P={result['precision']}  R={result['recall']}  F1={result['f1']}")
    print(f"    TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    return result


# ── Intel Berkeley ──────────────────────────────────────────────────────────────

def _intel():
    t0 = time.time()
    dcfg     = cfg["data"]["intel"]
    features = dcfg["features"]
    demo_dir = os.path.join(_root, "data", "demo")
    scaler, threshold, model = _load_model("intel")

    # Use intel_mixed.csv (150 normal then 50 fault — sequential).
    # Sliding windows over the full sequence; assign label from last row of each window.
    df_mix = pd.read_csv(os.path.join(demo_dir, "intel_mixed.csv"))
    scaled = scaler.transform(df_mix[features].ffill().fillna(0).values.astype(np.float32))
    all_labels = df_mix["label"].values

    print("  Intel mixed errors ...")
    all_err = _build_errors(scaled, model)
    n_win   = len(all_err)
    win_lbl = all_labels[W - 1 : W - 1 + n_win]
    norm_err = all_err[win_lbl == "Normal"]
    atk_err  = all_err[win_lbl != "Normal"]
    print(f"    {len(norm_err)} normal windows, mean={norm_err.mean():.5f}, "
          f"above_thresh={int((norm_err > threshold).sum())}")
    print(f"    {len(atk_err)} fault windows, mean={atk_err.mean():.5f}, "
          f"above_thresh={int((atk_err > threshold).sum())}")

    if len(norm_err) == 0 or len(atk_err) == 0:
        raise RuntimeError(
            "Not enough windows — re-run scripts/make_demo_csvs.py with larger counts."
        )

    y_true  = np.concatenate([np.zeros(len(norm_err)), np.ones(len(atk_err))])
    y_score = np.concatenate([norm_err, atk_err])
    y_pred  = (y_score > threshold).astype(int)
    cm      = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    result = {
        "dataset":           "intel",
        "threshold":         float(threshold),
        "n_normal_windows":  int(len(norm_err)),
        "n_attack_windows":  int(len(atk_err)),
        "auc_roc":           round(float(roc_auc_score(y_true, y_score)), 4),
        "accuracy":          round(float(accuracy_score(y_true, y_pred)), 4),
        "precision":         round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall":            round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1":                round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "confusion_matrix":  {"TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)},
        "error_samples":     {
            "normal": norm_err[:200].tolist(),
            "attack": atk_err[:200].tolist(),
        },
        "per_attack":        [],
        "elapsed_sec":       round(time.time() - t0, 2),
    }
    out = os.path.join(_root, "saved_models", "intel_metrics.json")
    with open(out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"  Saved: {out}")
    print(f"    AUC={result['auc_roc']}  Acc={result['accuracy']}  "
          f"P={result['precision']}  R={result['recall']}  F1={result['f1']}")
    print(f"    TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    return result


if __name__ == "__main__":
    print("=== CICIoT metrics ===")
    _ciciot()
    print("\n=== Intel metrics ===")
    _intel()
    print("\nDone.")
