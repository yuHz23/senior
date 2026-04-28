from typing import Iterator, List
import numpy as np
import pandas as pd

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
    for chunk in pd.read_csv(path, usecols=features, chunksize=chunksize):
        chunk = chunk[features].ffill().fillna(0.0)
        for row in chunk.itertuples(index=False):
            yield np.array(row, dtype=np.float32)
