"""Gateway event loop: consumes a CSV stream, detects anomalies, writes JSONL logs."""
import json
import os
import time
from datetime import datetime, UTC
from typing import Optional

import numpy as np

from src.data.preprocessor import WindowBuffer
from src.data.stream_simulator import stream_csv
from src.inference.detector import AnomalyDetector


def run_gateway(
    detector: AnomalyDetector,
    csv_path: str,
    features: list,
    log_path: str,
    window_size: int = 64,
    limit: Optional[int] = None,
    label_col: Optional[str] = None,
    labels: Optional[list] = None,
) -> dict:
    """Run the gateway inference loop over a CSV file.

    Args:
        detector:    Loaded AnomalyDetector instance
        csv_path:    Path to the raw CSV (unscaled)
        features:    Feature column names to stream
        log_path:    Output JSONL log file path
        window_size: Sliding window size
        limit:       Max number of windows to process (None = all)
        label_col:   Ground truth label column name (optional, for evaluation)
        labels:      Pre-loaded label array aligned with rows (alternative to label_col)

    Returns:
        Summary dict: total_windows, n_attack, n_normal, elapsed_seconds
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    buf = WindowBuffer(window_size)
    window_id = 0
    n_attack = n_normal = 0
    start = time.time()

    raw_buf = []  # stores unscaled rows to pass to detector

    with open(log_path, "w") as log_file:
        for i, sample in enumerate(stream_csv(csv_path, features)):
            raw_buf.append(sample)
            # Keep raw_buf aligned with window
            if len(raw_buf) > window_size:
                raw_buf.pop(0)

            # Use preprocessor WindowBuffer as trigger
            trigger = buf.push(sample)
            if trigger is None:
                continue

            raw_window = np.array(raw_buf, dtype=np.float32)
            result = detector.detect(raw_window)
            result["window_id"] = window_id
            result["timestamp"] = datetime.now(UTC).isoformat()

            if labels is not None and window_id < len(labels):
                result["ground_truth"] = int(labels[window_id])

            log_file.write(json.dumps(result) + "\n")

            if result["label"] == "ATTACK":
                n_attack += 1
            else:
                n_normal += 1

            window_id += 1
            if limit and window_id >= limit:
                break

    elapsed = time.time() - start
    summary = {
        "total_windows": window_id,
        "n_normal": n_normal,
        "n_attack": n_attack,
        "elapsed_seconds": round(elapsed, 2),
        "windows_per_second": round(window_id / elapsed, 1) if elapsed > 0 else 0,
    }
    print(json.dumps(summary, indent=2))
    return summary
