from typing import Iterator, List
import csv
import numpy as np

CICIOT_FEATURES: List[str] = [
    "flow_duration",
    "Rate",
    "Number",
    "IAT",
    "Tot size",
]

INTEL_FEATURES: List[str] = ["temperature", "humidity", "light", "voltage"]


def stream_csv(
    path: str,
    features: List[str],
    chunksize: int = 512,
) -> Iterator[np.ndarray]:
    """Yield one scaled row at a time from a CSV, simulating real-time ingestion.

    The caller is responsible for passing pre-scaled data paths or applying
    the scaler to each yielded sample before passing to WindowBuffer.
    """
    with open(path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        batch = []
        prev_row = None
        for row in reader:
            try:
                vals = []
                for f in features:
                    v = row.get(f)
                    if v is None or v == "":
                        # forward-fill with previous row if available
                        if prev_row:
                            v = prev_row.get(f, "0")
                        else:
                            v = "0"
                    vals.append(float(v))
                arr = np.array(vals, dtype=np.float32)
                prev_row = row
                batch.append(arr)
                if len(batch) >= chunksize:
                    for b in batch:
                        yield b
                    batch.clear()
            except (ValueError, TypeError):
                continue
        for b in batch:
            yield b
