"""Run the live gateway inference loop over a CSV stream.

Usage:
    python scripts/run_gateway.py --dataset ciciot --config configs/config.yaml
    python scripts/run_gateway.py --dataset ciciot --limit 1000
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch
from datetime import datetime, UTC

from src.inference.detector import AnomalyDetector
from src.inference.gateway import run_gateway
from src.data.stream_simulator import CICIOT_FEATURES, INTEL_FEATURES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ciciot", "intel"], required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--limit", type=int, default=None, help="Max windows to process")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dcfg = cfg["data"][args.dataset]
    mcfg = cfg["model"]
    features = CICIOT_FEATURES if args.dataset == "ciciot" else INTEL_FEATURES
    n_features = len(features)

    detector = AnomalyDetector(
        model_path=cfg["saved_models"][f"{args.dataset}_model"],
        scaler_path=dcfg["scaler_path"],
        threshold_path=cfg["saved_models"][f"{args.dataset}_threshold"],
        input_size=n_features,
        hidden_size=mcfg["hidden_size"],
        bottleneck_size=mcfg["bottleneck_size"],
        seq_len=mcfg["window_size"],
        num_layers=mcfg["num_layers"],
        device=args.device,
    )

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    log_path = os.path.join(cfg["gateway"]["log_dir"], f"{args.dataset}_{timestamp}.jsonl")

    print(f"Starting gateway — dataset={args.dataset}, limit={args.limit}, log={log_path}")
    run_gateway(
        detector=detector,
        csv_path=dcfg["raw_path"],
        features=features,
        log_path=log_path,
        window_size=mcfg["window_size"],
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
