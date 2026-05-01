"""FastAPI REST endpoint for real-time anomaly detection.

Endpoints:
    POST /detect         — detect anomaly in a single raw window
    POST /detect/batch   — detect anomalies in multiple windows
    GET  /health         — liveness check
    GET  /info           — model/threshold metadata
"""
from __future__ import annotations
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import List, Optional
import io
import asyncio
import json as _json
import numpy as np
import csv
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.inference.onnx_detector import OnnxAnomalyDetector

app = FastAPI(
    title="IoT Anomaly Detection API",
    description="GRU Autoencoder–based anomaly detection for IoT gateway",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount docs/ as static files so /demo serves the demo page
_docs_dir = os.path.join(os.path.dirname(__file__), "../../docs")
if os.path.isdir(_docs_dir):
    app.mount("/static", StaticFiles(directory=_docs_dir), name="static")

# Detector instances — loaded at startup
_detectors: dict[str, OnnxAnomalyDetector] = {}


# ── Request / Response schemas ────────────────────────────────────────────────

class WindowRequest(BaseModel):
    dataset: str = Field("ciciot", description="'ciciot' or 'intel'")
    window: List[List[float]] = Field(
        ...,
        description="Raw (unscaled) window: list of seq_len rows, each row = n_features values",
    )

class BatchWindowRequest(BaseModel):
    dataset: str = Field("ciciot")
    windows: List[List[List[float]]] = Field(
        ..., description="List of windows (each is seq_len × n_features)"
    )

class DetectionResult(BaseModel):
    label: str
    reconstruction_error: float
    threshold: float
    semantic_vector: List[float]
    window_id: Optional[int] = None

class BatchDetectionResult(BaseModel):
    results: List[DetectionResult]
    n_normal: int
    n_attack: int

class HealthResponse(BaseModel):
    status: str
    loaded_datasets: List[str]

class InfoResponse(BaseModel):
    dataset: str
    threshold: float
    window_size: int
    bottleneck_size: int


# ── Lifecycle ─────────────────────────────────────────────────────────────────

@app.on_event("startup")
def load_models():
    import yaml
    # Resolve project root so all relative paths work regardless of CWD
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    cfg_path = os.path.join(_root, "configs", "config.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    def abs_path(p):
        return p if os.path.isabs(p) else os.path.join(_root, p)

    mcfg = cfg["model"]
    from src.data.stream_simulator import CICIOT_FEATURES, INTEL_FEATURES

    for dataset, features in [("ciciot", CICIOT_FEATURES), ("intel", INTEL_FEATURES)]:
        model_path     = abs_path(cfg["saved_models"][f"{dataset}_model"])
        threshold_path = abs_path(cfg["saved_models"][f"{dataset}_threshold"])
        scaler_path    = abs_path(cfg["data"][dataset]["scaler_path"])

        if not (os.path.exists(model_path) and os.path.exists(threshold_path)):
            print(f"  [skip] {dataset}: model or threshold not found at {model_path}")
            continue  # skip if not trained yet

        _detectors[dataset] = OnnxAnomalyDetector(
            model_path=model_path,
            scaler_path=scaler_path,
            threshold_path=threshold_path,
        )
    print(f"Loaded detectors: {list(_detectors.keys())}")
    # Pre-warm stream cache at startup so /batch and /stream don't time out
    for ds in list(_detectors.keys()):
        try:
            _ensure_stream_data(ds)
        except Exception as e:
            print(f"  [startup] stream pre-warm failed for {ds}: {e}")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok", "loaded_datasets": list(_detectors.keys())}


@app.get("/info/{dataset}", response_model=InfoResponse)
def info(dataset: str):
    det = _get_detector(dataset)
    import yaml
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    with open(os.path.join(_root, "configs", "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    return {
        "dataset":        dataset,
        "threshold":      det.threshold,
        "window_size":    cfg["model"]["window_size"],
        "bottleneck_size": cfg["model"]["bottleneck_size"],
    }


@app.post("/detect", response_model=DetectionResult)
def detect(req: WindowRequest):
    det    = _get_detector(req.dataset)
    window = np.array(req.window, dtype=np.float32)
    _validate_window(window, req.dataset)
    return det.detect(window)


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    # Inline 1x1 shield emoji as SVG favicon
    svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><text y=".9em" font-size="90">🛡️</text></svg>'
    return StreamingResponse(iter([svg.encode()]), media_type="image/svg+xml")


@app.get("/demo", response_class=HTMLResponse, include_in_schema=False)
def demo():
    demo_path = os.path.join(os.path.dirname(__file__), "../../docs/demo.html")
    with open(demo_path, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/sample/{dataset}")
def sample_data(dataset: str, n: int = 200):
    """Return a small synthetic CSV for demo purposes."""
    from src.data.stream_simulator import CICIOT_FEATURES, INTEL_FEATURES
    rng = np.random.default_rng(42)

    if dataset == "ciciot":
        cols = CICIOT_FEATURES
        # 80% normal, 20% attack
        n_normal = int(n * 0.8)
        n_attack = n - n_normal
        normal = rng.uniform(0, 1, (n_normal, len(cols)))
        attack = rng.uniform(5, 20, (n_attack, len(cols)))  # high values = attacks
        data = np.vstack([normal, attack])
        label = ["BenignTraffic"] * n_normal + ["DDoS-ICMP_Flood"] * n_attack
    elif dataset == "intel":
        cols = INTEL_FEATURES
        n_normal = int(n * 0.8)
        n_attack = n - n_normal
        normal = rng.normal([22, 55, 400, 2.7], [1, 3, 30, 0.1], (n_normal, len(cols)))
        attack = rng.normal([45, 80, 900, 1.0], [2, 5, 50, 0.2], (n_attack, len(cols)))
        data = np.vstack([normal, attack])
        label = ["Normal"] * n_normal + ["SensorFault"] * n_attack
    else:
        raise HTTPException(status_code=404, detail=f"Unknown dataset: {dataset}")

    idx = rng.permutation(n)
    rows_shuffled = data[idx]
    labels_shuffled = np.array(label)[idx]
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(cols + ["label"])
    for row, lbl in zip(rows_shuffled, labels_shuffled):
        writer.writerow(list(row) + [lbl])
    return StreamingResponse(
        io.BytesIO(buf.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{dataset}_sample.csv"'},
    )


@app.post("/detect/batch", response_model=BatchDetectionResult)
def detect_batch(req: BatchWindowRequest):
    det     = _get_detector(req.dataset)
    results = []
    n_normal = n_attack = 0
    for idx, w in enumerate(req.windows):
        window = np.array(w, dtype=np.float32)
        _validate_window(window, req.dataset)
        r = det.detect(window)
        r["window_id"] = idx
        results.append(r)
        if r["label"] == "ATTACK":
            n_attack += 1
        else:
            n_normal += 1
    return {"results": results, "n_normal": n_normal, "n_attack": n_attack}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_detector(dataset: str) -> OnnxAnomalyDetector:
    if dataset not in _detectors:
        raise HTTPException(
            status_code=404,
            detail=f"Detector for '{dataset}' not loaded. Run train.py first.",
        )
    return _detectors[dataset]


@app.post("/detect/csv")
async def detect_csv(
    file: UploadFile = File(...),
    dataset: str = Form("ciciot"),
):
    """Upload a CSV file and run anomaly detection on all sliding windows."""
    from src.data.stream_simulator import CICIOT_FEATURES, INTEL_FEATURES
    det = _get_detector(dataset)

    import yaml
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    with open(os.path.join(_root, "configs", "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    W = cfg["model"]["window_size"]
    features = CICIOT_FEATURES if dataset == "ciciot" else INTEL_FEATURES

    content = await file.read()
    try:
        reader = csv.DictReader(io.StringIO(content.decode("utf-8", errors="replace")))
        rows = list(reader)
        if not rows:
            raise ValueError("Empty CSV")
        header = list(rows[0].keys())
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Cannot parse CSV: {e}")

    missing = [c for c in features if c not in header]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing columns: {missing}. Expected: {features}",
        )

    label_col = cfg["data"][dataset].get("label_col")
    if not label_col and "label" in header:
        label_col = "label"
    has_labels = bool(label_col and label_col in header)

    data = []
    labels_raw = []
    for row in rows:
        try:
            data.append([float(row.get(f, 0) or 0) for f in features])
            if has_labels:
                labels_raw.append(row[label_col])
        except (ValueError, TypeError):
            continue
    data = np.array(data, dtype=np.float32)
    labels = labels_raw if has_labels else None
    n = len(data)
    if n < W:
        raise HTTPException(status_code=422, detail=f"Need at least {W} rows, got {n}")

    n_windows = n - W + 1

    # ── Batch inference (single forward pass for all windows) ─────────────
    data_scaled = det.scaler.transform(data)                   # (n, F)
    # Sliding windows via stride tricks — no data copy until batch
    shape   = (n_windows, W, data_scaled.shape[1])
    strides = (data_scaled.strides[0],) * 2 + (data_scaled.strides[1],)
    wins_scaled = np.lib.stride_tricks.as_strided(
        np.ascontiguousarray(data_scaled), shape=shape, strides=strides,
    )

    BATCH = 512
    all_errors: list = []
    all_z:      list = []
    all_x_hat:  list = []  # keep only for recon examples

    for s in range(0, n_windows, BATCH):
        batch_np = wins_scaled[s:s+BATCH].copy().astype(np.float32)
        z, xhat  = det.run_batch(batch_np)
        errs     = ((batch_np - xhat) ** 2).mean(axis=(1, 2))
        all_errors.append(errs)
        all_z.append(z)
        # store reconstructions only for the best/worst candidates
        all_x_hat.append(xhat)

    errors_np = np.concatenate(all_errors)        # (n_windows,)
    z_np      = np.concatenate(all_z)             # (n_windows, bottleneck)
    x_hat_np  = np.concatenate(all_x_hat)         # (n_windows, W, F)

    # ── Build results list (no model calls here) ──────────────────────────
    results = []
    n_normal = n_attack = 0
    best_normal_idx  = -1
    worst_attack_idx = -1
    best_err  = float("inf")
    worst_err = 0.0

    for i in range(n_windows):
        err   = float(round(float(errors_np[i]), 6))
        label = "ATTACK" if err > det.threshold else "NORMAL"
        r: dict = {
            "label":                label,
            "reconstruction_error": err,
            "threshold":            det.threshold,
            "semantic_vector":      z_np[i].tolist(),
            "window_id":            i,
        }
        if has_labels:
            r["true_label"] = str(labels[i + W - 1])
        results.append(r)
        if label == "ATTACK":
            n_attack += 1
            if err > worst_err:
                worst_err = err;  worst_attack_idx = i
        else:
            n_normal += 1
            if err < best_err:
                best_err = err;   best_normal_idx = i

    # ── Reconstruction examples using already-computed tensors ───────────
    recon_examples: dict = {}
    if best_normal_idx >= 0:
        recon_examples["normal"] = {
            "window_id":          best_normal_idx,
            "error":              best_err,
            "input_scaled":       wins_scaled[best_normal_idx].tolist(),
            "reconstructed_scaled": x_hat_np[best_normal_idx].tolist(),
        }
    if worst_attack_idx >= 0:
        recon_examples["attack"] = {
            "window_id":          worst_attack_idx,
            "error":              worst_err,
            "input_scaled":       wins_scaled[worst_attack_idx].tolist(),
            "reconstructed_scaled": x_hat_np[worst_attack_idx].tolist(),
        }

    return {
        "results": results,
        "n_normal": n_normal,
        "n_attack": n_attack,
        "total_windows": len(results),
        "dataset": dataset,
        "threshold": det.threshold,
        "recon_examples": recon_examples,
    }


_metrics_cache: dict = {}  # dataset -> cached metrics dict


@app.get("/metrics/{dataset}")
def get_metrics(dataset: str):
    """Return evaluation metrics. Reads pre-saved JSON (run scripts/save_metrics.py first)."""
    if dataset in _metrics_cache:
        return _metrics_cache[dataset]

    if dataset not in ("ciciot", "intel"):
        raise HTTPException(404, detail=f"Unknown dataset: {dataset}")

    _root2 = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    json_path = os.path.join(_root2, "saved_models", f"{dataset}_metrics.json")

    if not os.path.exists(json_path):
        raise HTTPException(
            404,
            detail=(
                f"Metrics file not found: {json_path}. "
                "Run: python scripts/save_metrics.py"
            ),
        )

    import json as _json_mod
    with open(json_path, "r") as fh:
        result = _json_mod.load(fh)

    _metrics_cache[dataset] = result
    return result


@app.get("/demo-csv/{dataset}/{scenario}")
def get_demo_csv(dataset: str, scenario: str):
    """Serve a pre-extracted demo CSV for scenario button quick-load."""
    _valid: dict[str, list] = {
        "ciciot": ["normal", "attack", "mixed"],
        "intel":  ["normal", "mixed"],
    }
    if dataset not in _valid:
        raise HTTPException(404, detail=f"Unknown dataset: {dataset}")
    if scenario not in _valid[dataset]:
        raise HTTPException(
            404,
            detail=f"Scenario '{scenario}' not available for {dataset}. "
                   f"Run: python scripts/make_demo_csvs.py",
        )
    _root2 = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    csv_path = os.path.join(_root2, "data", "demo", f"{dataset}_{scenario}.csv")
    if not os.path.exists(csv_path):
        raise HTTPException(
            404,
            detail="Demo CSV not found. Run: python scripts/make_demo_csvs.py",
        )
    with open(csv_path, "rb") as fh:
        content = fh.read()
    return StreamingResponse(
        io.BytesIO(content),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{dataset}_{scenario}.csv"'},
    )


# ── Streaming (SSE) ───────────────────────────────────────────────────────────

_stream_cache: dict = {}       # dataset -> {windows: np.ndarray, labels: list}
_stream_counters: dict = {}    # dataset -> next batch offset (for /batch)


def _ensure_stream_data(dataset: str):
    """Pre-load/cache windows for live streaming.

    Strategy:
    - NORMAL : load real rows (label==normal) from CSV → scaler → model reconstructs well
    - ATTACK : take normal rows + huge perturbation → out-of-distribution → high MSE
    Interleaved 4:1 ratio so both appear clearly in the live log.
    """
    if dataset in _stream_cache:
        return
    import yaml
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    with open(os.path.join(_root, "configs", "config.yaml")) as f:
        cfg = yaml.safe_load(f)

    def absp(p): return p if os.path.isabs(p) else os.path.join(_root, p)

    W          = cfg["model"]["window_size"]
    dcfg       = cfg["data"][dataset]
    features   = dcfg["features"]
    n_feat     = len(features)
    label_col  = dcfg.get("label_col")
    normal_lbl = dcfg.get("normal_label", "Normal")

    from src.data.preprocessor import load_scaler, WindowBuffer
    scaler = load_scaler(absp(dcfg["scaler_path"]))

    rng           = np.random.default_rng(42)
    n_normal_want = 800
    n_attack_want = 200

    # ── 1. Load real normal rows from CSV (try demo CSV first, then raw) ──
    normal_scaled = None
    demo_paths = [
        os.path.join(_root, "data", "demo", f"{dataset}_normal.csv"),
        os.path.join(_root, "data", "demo", f"{dataset}_mixed.csv"),
        absp(dcfg["raw_path"]),
    ]

    def _load_csv_rows(csv_path, filter_normal=True):
        """Load rows from CSV using standard csv (no pandas dependency)."""
        import csv
        rows, seen_cols = [], []
        with open(csv_path, "r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if len(rows) >= (n_normal_want if filter_normal else n_attack_want):
                    break
                # Check label
                if label_col and label_col in row:
                    lbl = row[label_col]
                    if filter_normal and lbl != normal_lbl:
                        continue
                    if not filter_normal and lbl == normal_lbl:
                        continue
                # Parse features
                try:
                    vals = [float(row.get(f, "0") or "0") for f in features]
                except (ValueError, TypeError):
                    continue
                rows.append(vals)
        return np.array(rows, dtype=np.float32) if rows else None

    for csv_path in demo_paths:
        if not os.path.exists(csv_path):
            continue
        try:
            arr = _load_csv_rows(csv_path, filter_normal=True)
            if arr is not None and len(arr) >= W + 10:
                normal_scaled = scaler.transform(arr[:n_normal_want])
                print(f"  [stream:{dataset}] {len(arr)} real normal rows loaded from {os.path.basename(csv_path)}")
                break
        except Exception as e:
            print(f"  [stream:{dataset}] CSV load failed for {os.path.basename(csv_path)}: {e}")
            continue

    # ── 2. Smooth-sine fallback (low amplitude to match scaled distribution) ──
    if normal_scaled is None or len(normal_scaled) < W + 10:
        t    = np.linspace(0, 6 * np.pi, n_normal_want)
        # Small amplitude ~0.001 to match training reconstruction error 0.0006-0.003
        base = np.column_stack([0.001 * np.sin(t + i * 0.7) for i in range(n_feat)])
        base = (base + rng.uniform(-0.0003, 0.0003, base.shape)).astype(np.float32)
        normal_scaled = base
        print(f"  [stream:{dataset}] WARNING: using sine-wave fallback for normals")

    if len(normal_scaled) < n_normal_want:
        reps = (n_normal_want // len(normal_scaled)) + 1
        normal_scaled = np.tile(normal_scaled, (reps, 1))[:n_normal_want]

    # ── 3. Attack = load real attack rows from CSV ────────────
    attack_scaled = None
    for csv_path in demo_paths:
        if not os.path.exists(csv_path):
            continue
        try:
            arr = _load_csv_rows(csv_path, filter_normal=False)
            if arr is not None and len(arr) >= W + 10:
                attack_scaled = scaler.transform(arr[:n_attack_want])
                print(f"  [stream:{dataset}] {len(arr)} real attack rows loaded from {os.path.basename(csv_path)}")
                break
        except Exception:
            continue

    # Fallback: small perturbation (enough to push error above threshold)
    if attack_scaled is None:
        base_atk = normal_scaled[:n_attack_want].copy()
        perturb  = rng.uniform(0.5, 1.5, (n_attack_want, n_feat)).astype(np.float32)
        sign     = rng.choice([-1.0, 1.0], (n_attack_want, n_feat)).astype(np.float32)
        attack_scaled = base_atk + perturb * sign
        print(f"  [stream:{dataset}] WARNING: using synthetic perturbation for attacks")

    # Tile arrays if we don't have enough rows
    if len(normal_scaled) < n_normal_want:
        reps = (n_normal_want // len(normal_scaled)) + 1
        normal_scaled = np.tile(normal_scaled, (reps, 1))[:n_normal_want]
    if len(attack_scaled) < n_attack_want:
        reps = (n_attack_want // len(attack_scaled)) + 1
        attack_scaled = np.tile(attack_scaled, (reps, 1))[:n_attack_want]

    # ── 4. Recalibrate threshold using ONNX model on real normal data ────
    # PyTorch threshold ≠ ONNX threshold → need to compute from ONNX errors
    if dataset in _detectors and normal_scaled is not None:
        det = _detectors[dataset]
        buf   = WindowBuffer(window_size=W)
        errs  = []
        # Use 50 windows max for fast recalibration on serverless
        for i in range(min(50, len(normal_scaled))):
            w = buf.push(normal_scaled[i])
            if w is not None:
                r = det.detect_scaled(w)
                errs.append(r["reconstruction_error"])
        if errs:
            new_threshold = float(np.percentile(errs, 99.0))
            det.threshold = new_threshold
            print(f"  [stream:{dataset}] ONNX threshold recalibrated: {new_threshold:.6f} (n={len(errs)}, max={max(errs):.6f})")

    # ── 5. Interleave 4 normals : 1 attack ────────────────────────────────
    rows, labs = [], []
    ni = ai = 0
    while ni < n_normal_want or ai < n_attack_want:
        for _ in range(4):
            if ni < n_normal_want:
                rows.append(normal_scaled[ni]); labs.append("Normal"); ni += 1
        if ai < n_attack_want:
            rows.append(attack_scaled[ai]); labs.append("Attack"); ai += 1

    all_scaled = np.array(rows, dtype=np.float32)
    all_labels = np.array(labs)

    # ── 5. Sliding windows ────────────────────────────────────────────────
    n = len(all_scaled)
    windows, win_labels = [], []
    for i in range(n - W + 1):
        windows.append(all_scaled[i : i + W])
        win_labels.append(str(all_labels[i + W - 1]))

    # Store recalibrated threshold in cache (survives request boundary)
    # Vercel cold-starts reset _detectors → threshold reverts to JSON value
    cached_threshold = _detectors[dataset].threshold if dataset in _detectors else None
    _stream_cache[dataset] = {
        "windows":   np.array(windows, dtype=np.float32),
        "labels":    win_labels,
        "threshold": cached_threshold,
    }
    print(f"  [stream:{dataset}] {len(windows)} windows ready "
          f"(~{labs.count('Normal')} normal rows, ~{labs.count('Attack')} attack rows)")


@app.get("/stream/mixed")
async def stream_mixed(speed: float = 5.0):
    """Interleave CICIoT + Intel detection results as SSE."""
    for ds in ["ciciot", "intel"]:
        if ds not in _detectors:
            raise HTTPException(503, detail=f"Detector '{ds}' not loaded")
        _ensure_stream_data(ds)

    delay = max(1.0 / speed, 0.02)

    async def generate():
        idxs  = {"ciciot": 0, "intel": 0}
        loops = {"ciciot": 0, "intel": 0}
        turn  = 0
        try:
            while True:
                ds  = "ciciot" if turn % 2 == 0 else "intel"
                turn += 1
                det  = _detectors[ds]
                data = _stream_cache[ds]
                idx  = idxs[ds]
                r    = det.detect_scaled(data["windows"][idx])
                r["window_id"]  = idx
                r["loop"]       = loops[ds]
                r["dataset"]    = ds
                r["true_label"] = data["labels"][idx]
                yield f"data: {_json.dumps(r)}\n\n"
                await asyncio.sleep(delay)
                idxs[ds] += 1
                if idxs[ds] >= len(data["windows"]):
                    idxs[ds] = 0
                    loops[ds] += 1
        except asyncio.CancelledError:
            pass

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                 "Connection": "keep-alive"},
    )


@app.get("/stream/{dataset}")
async def stream_dataset(dataset: str, speed: float = 5.0):
    """Stream detection results for a single dataset as SSE.

    On Vercel Serverless, SSE is not supported — falls back to
    a 5-event batch that the client polls via /batch/{dataset}.
    """
    det = _get_detector(dataset)
    _ensure_stream_data(dataset)
    data  = _stream_cache[dataset]
    delay = max(1.0 / float(speed), 0.02)

    # Vercel kills SSE connections immediately — use batch polling instead
    # Return a small batch of events as SSE so EventSource gets at least something
    async def generate():
        idx = _stream_counters.get(dataset, 0)
        total = len(data["windows"])
        batch_size = min(10, total)
        for _ in range(batch_size):
            r = det.detect_scaled(data["windows"][idx % total])
            r["window_id"]  = idx % total
            r["loop"]       = idx // total
            r["dataset"]    = dataset
            r["true_label"] = data["labels"][idx % total]
            yield f"data: {_json.dumps(r)}\n\n"
            idx += 1
        _stream_counters[dataset] = idx

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                 "Connection": "keep-alive"},
    )


@app.get("/batch/{dataset}")
async def batch_dataset(dataset: str, n: int = 10, offset: int = -1):
    """Return a batch of detection results as JSON (use for polling on Vercel).

    Client polls this every second instead of using EventSource.
    """
    try:
        det = _get_detector(dataset)
        _ensure_stream_data(dataset)
        data  = _stream_cache[dataset]
        total = len(data["windows"])
        n     = min(n, 50)  # cap at 50

        # Use threshold stored in cache (recalibrated from ONNX model on startup)
        if data.get("threshold") is not None:
            det.threshold = data["threshold"]

        if offset < 0:
            offset = _stream_counters.get(dataset, 0)

        results = []
        for i in range(n):
            idx = (offset + i) % total
            r   = det.detect_scaled(data["windows"][idx])
            r["window_id"]  = idx
            r["loop"]       = (offset + i) // total
            r["dataset"]    = dataset
            r["true_label"] = data["labels"][idx]
            results.append(r)

        next_offset = offset + n
        _stream_counters[dataset] = next_offset

        return {"results": results, "next_offset": next_offset, "total": total}
    except Exception as exc:
        import traceback
        return {"error": str(exc), "traceback": traceback.format_exc()}


def _validate_window(window: np.ndarray, dataset: str):
    import yaml
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    with open(os.path.join(_root, "configs", "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    W = cfg["model"]["window_size"]
    if window.ndim != 2 or window.shape[0] != W:
        raise HTTPException(
            status_code=422,
            detail=f"Window must be shape ({W}, n_features), got {window.shape}",
        )
