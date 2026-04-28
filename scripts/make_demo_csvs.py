"""Extract real demo CSV snippets from raw datasets.

Usage:
    python scripts/make_demo_csvs.py

Outputs to data/demo/:
    ciciot_normal.csv  — 300 BenignTraffic rows
    ciciot_attack.csv  — 300 attack rows (various types)
    ciciot_mixed.csv   — 200 normal + 100 attack (shuffled)
    intel_normal.csv   — 200 clean sensor rows
    intel_mixed.csv    — 150 normal + 50 faulty rows
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT   = os.path.join(_root, "data", "demo")
os.makedirs(OUT, exist_ok=True)

CICIOT_RAW = os.path.join(_root, "data", "raw", "ciciot.csv")
INTEL_RAW  = os.path.join(_root, "data", "raw", "intel_berkeley.csv")

CICIOT_FEATURES = ["flow_duration", "Rate", "Number", "IAT", "Tot size"]
INTEL_FEATURES  = ["temperature", "humidity", "light", "voltage"]

INTEL_FAULT_RULES = {
    "temperature": (-20.0, 100.0),
    "humidity":    (0.0,   100.0),
    "light":       (0.0,  3000.0),
    "voltage":     (2.0,    3.5),
}


def _faulty_intel(row: pd.Series) -> bool:
    for col, (lo, hi) in INTEL_FAULT_RULES.items():
        if col in row.index and (row[col] < lo or row[col] > hi):
            return True
    return False


# ── CICIoT ─────────────────────────────────────────────────────────────────────
print("=== CICIoT ===")
N_NORM, N_ATK = 300, 300
normal_parts: list = []
attack_parts: list = []

for chunk in pd.read_csv(CICIOT_RAW, chunksize=5000):
    n_have = sum(len(p) for p in normal_parts)
    a_have = sum(len(p) for p in attack_parts)
    if n_have >= N_NORM and a_have >= N_ATK:
        break
    if n_have < N_NORM:
        nb = chunk[chunk["label"] == "BenignTraffic"]
        if len(nb):
            normal_parts.append(nb[CICIOT_FEATURES + ["label"]])
    if a_have < N_ATK:
        ab = chunk[chunk["label"] != "BenignTraffic"]
        if len(ab):
            attack_parts.append(ab[CICIOT_FEATURES + ["label"]])

normal_df = pd.concat(normal_parts).head(N_NORM).reset_index(drop=True)
attack_df = pd.concat(attack_parts).head(N_ATK).reset_index(drop=True)

# ciciot_normal.csv
path = os.path.join(OUT, "ciciot_normal.csv")
normal_df.to_csv(path, index=False)
print(f"  {path}: {len(normal_df)} rows, label={normal_df['label'].iloc[0]}")

# ciciot_attack.csv
path = os.path.join(OUT, "ciciot_attack.csv")
attack_df.to_csv(path, index=False)
atk_types = list(attack_df["label"].unique()[:4])
print(f"  {path}: {len(attack_df)} rows, types={atk_types}")

# ciciot_mixed.csv — 200 normal THEN 100 attack (sequential, NOT shuffled)
# Keeps temporal structure intact: GRU sees clean normal block then clean attack block
mixed_df = pd.concat([
    normal_df.head(200),
    attack_df.head(100),
], ignore_index=True)
path = os.path.join(OUT, "ciciot_mixed.csv")
mixed_df.to_csv(path, index=False)
print(f"  {path}: {len(mixed_df)} rows (200 normal then 100 attack, sequential)")


# ── Intel Berkeley ──────────────────────────────────────────────────────────────
print("\n=== Intel Berkeley ===")
df_intel = pd.read_csv(
    INTEL_RAW,
    sep=r"\s+",
    header=None,
    names=["date", "time", "epoch", "moteid",
           "temperature", "humidity", "light", "voltage"],
    engine="python",
)
df_intel = (df_intel[INTEL_FEATURES]
            .apply(pd.to_numeric, errors="coerce")
            .ffill().bfill().dropna())

fault_mask = df_intel.apply(_faulty_intel, axis=1)
normal_intel = df_intel[~fault_mask].copy()
faulty_intel = df_intel[fault_mask].copy()
normal_intel["label"] = "Normal"
faulty_intel["label"] = "SensorFault"

# intel_normal.csv
intel_normal_df = normal_intel.head(200).reset_index(drop=True)
path = os.path.join(OUT, "intel_normal.csv")
intel_normal_df.to_csv(path, index=False)
print(f"  {path}: {len(intel_normal_df)} rows")

# Ensure we have 50 fault rows — synthesize if necessary
n_real = len(faulty_intel)
if n_real < 50:
    rng = np.random.default_rng(99)
    extra = normal_intel.head(100).copy()
    extra = extra.sample(50 - n_real, random_state=42)
    extra["temperature"] = rng.uniform(110, 160, len(extra))
    extra["humidity"]    = rng.uniform(105, 200, len(extra))
    extra["label"] = "SensorFault"
    faulty_intel = pd.concat([faulty_intel, extra])

# intel_mixed.csv — 150 normal THEN 50 fault (sequential, NOT shuffled)
intel_mixed = pd.concat([
    normal_intel.head(150)[INTEL_FEATURES + ["label"]],
    faulty_intel.head(50)[INTEL_FEATURES + ["label"]],
], ignore_index=True)
path = os.path.join(OUT, "intel_mixed.csv")
intel_mixed.to_csv(path, index=False)
print(f"  {path}: {len(intel_mixed)} rows "
      f"(~{(intel_mixed['label']=='Normal').sum()} normal, "
      f"~{(intel_mixed['label']=='SensorFault').sum()} fault)")

print(f"\nAll demo CSVs saved to {OUT}/")
