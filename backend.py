from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd

app = FastAPI(title="Industrial Fault Diagnosis Backend", version="4.0")

BASE_DIR = Path(__file__).resolve().parent
LEGACY_DATASET = BASE_DIR / "datasets" / "Industrial_IoT_Predictive_Maintenance_250k.csv"
ANOMALY_DATASET = BASE_DIR / "datasets" / "equipment_anomaly_data.csv"
MODEL_CACHE = BASE_DIR / "model_cache.pkl"
METRICS_FILE = BASE_DIR / "model_metrics.json"

LEGACY_FEATURES = ["Vibration_mm_s", "Temperature_C", "Current_A", "Noise_dB", "RPM"]
LEGACY_UNITS = ["mm/s", "°C", "A", "dB", "RPM"]
LEGACY_TARGET = "Failure_Type"
FAILURE_TARGET = "Failure_Flag"
ANOMALY_FEATURES = ["temperature", "pressure", "vibration", "humidity"]
ANOMALY_UNITS = ["°C", "bar", "mm/s", "%"]

# The original machine controls are preserved.
MACHINE_FILTERS = {"M-01": None, "P-02": "Pump", "C-03": "Compressor", "F-04": None}
MACHINES = {
    "M-01": {"name": "Industrial Motor M-01", "type": "Motor"},
    "P-02": {"name": "Industrial Pump P-02", "type": "Pump"},
    "C-03": {"name": "Air Compressor C-03", "type": "Compressor"},
    "F-04": {"name": "Cooling Fan F-04", "type": "Fan"},
}

LEGACY_COLUMNS = ["Machine_ID", "Machine_Type", "Timestamp"] + LEGACY_FEATURES + [FAILURE_TARGET, LEGACY_TARGET]
DATA = None
BUNDLE = None
METRICS = None
active = {}
run_counts = {k: 0 for k in MACHINES}


def load_legacy_data():
    global DATA
    if DATA is None:
        if not LEGACY_DATASET.exists():
            raise FileNotFoundError(f"Dataset not found: {LEGACY_DATASET}")
        DATA = pd.read_csv(LEGACY_DATASET, usecols=LEGACY_COLUMNS)
        DATA[LEGACY_TARGET] = DATA[LEGACY_TARGET].fillna("Normal").astype(str)
        DATA[FAILURE_TARGET] = pd.to_numeric(DATA[FAILURE_TARGET], errors="coerce").fillna(0).astype(int)
        for col in LEGACY_FEATURES:
            DATA[col] = pd.to_numeric(DATA[col], errors="coerce")
        DATA = DATA.dropna(subset=LEGACY_FEATURES).reset_index(drop=True)
    return DATA


def load_bundle():
    global BUNDLE
    if BUNDLE is None:
        if not MODEL_CACHE.exists():
            raise FileNotFoundError("Pre-trained model cache is missing. Run train_models.py once.")
        with MODEL_CACHE.open("rb") as f:
            BUNDLE = pickle.load(f)
    return BUNDLE


def load_metrics():
    global METRICS
    if METRICS is None:
        if METRICS_FILE.exists():
            METRICS = json.loads(METRICS_FILE.read_text(encoding="utf-8"))
        else:
            METRICS = {"equipment_anomaly_models": [], "legacy_fault_type_models": []}
    return METRICS


def dataset_rows_for_machine(machine: str):
    data = load_legacy_data()
    machine_type = MACHINE_FILTERS.get(machine)
    if machine_type:
        subset = data[data["Machine_Type"].eq(machine_type)]
        if len(subset):
            return subset
    return data


def row_to_values(row):
    return [round(float(row[c]), 2) if c != "RPM" else round(float(row[c]), 1) for c in LEGACY_FEATURES]


def legacy_diagnosis(values):
    models = load_bundle()["legacy_models"]
    x = np.asarray(values, dtype=float).reshape(1, -1)
    predictions = [m.predict(x)[0] for m in models]
    probs = [m.predict_proba(x)[0] for m in models]
    classes = models[0].classes_
    mean_prob = np.mean(probs, axis=0)
    idx = int(np.argmax(mean_prob))
    fault = str(classes[idx])
    confidence = float(mean_prob[idx] * 100)

    # The legacy fault-type model includes a Normal class.  Use the ensemble
    # probability of every non-Normal class as the model-estimated probability
    # of a fault. This is different from model confidence: confidence is the
    # probability of the predicted class, while failure_probability is
    # 1 - P(Normal).
    normal_idx = np.where(classes.astype(str) == "Normal")[0]
    if len(normal_idx):
        normal_probability = float(mean_prob[int(normal_idx[0])])
        failure_probability = float((1.0 - normal_probability) * 100.0)
    else:
        normal_probability = None
        failure_probability = None

    details = []
    for m, pred, pr in zip(models, predictions, probs):
        pi = int(np.where(m.classes_ == pred)[0][0])
        details.append({
            "model": m.__class__.__name__,
            "prediction": str(pred),
            "confidence": round(float(pr[pi] * 100), 1),
        })
    return fault, confidence, failure_probability, normal_probability, details


class Reading(BaseModel):
    machine: str
    values: list[float]


@app.get("/health")
def health():
    metrics = load_metrics()
    return {
        "status": "ok",
        "service": "fault-diagnosis-backend",
        "data_source": LEGACY_DATASET.name,
        "anomaly_data_source": ANOMALY_DATASET.name,
        "rows": int(metrics.get("legacy_rows", 250000)),
        "anomaly_rows": int(metrics.get("dataset_rows", 7672)),
        "algorithms": ["DecisionTreeClassifier", "RandomForestClassifier"],
        "models_pretrained": True,
    }


@app.get("/machines")
def machines():
    return MACHINES


@app.get("/dataset/info")
def dataset_info():
    data = load_legacy_data()
    counts = data[LEGACY_TARGET].value_counts(dropna=False).to_dict()
    return {"file": LEGACY_DATASET.name, "rows": int(len(data)), "columns": list(pd.read_csv(LEGACY_DATASET, nrows=0).columns), "features_used_for_diagnosis": LEGACY_FEATURES, "target": LEGACY_TARGET, "class_counts": {str(k): int(v) for k, v in counts.items()}}


@app.get("/dataset/sample")
def dataset_sample(n: int = 10):
    data = load_legacy_data(); n = max(1, min(int(n), 100)); sample = data.sample(n=n, random_state=42)
    return sample[LEGACY_COLUMNS].to_dict(orient="records")


@app.post("/start/{machine}")
def start(machine: str):
    if machine not in MACHINES: raise HTTPException(404, "Unknown machine")
    rows = dataset_rows_for_machine(machine)
    run_counts[machine] += 1
    row = rows.iloc[(run_counts[machine] - 1) % len(rows)]
    active[machine] = row
    return {"machine": machine, "fault_profile": str(row[LEGACY_TARGET]), "run_number": run_counts[machine], "data_source": LEGACY_DATASET.name, "dataset_machine_id": str(row.get("Machine_ID", ""))}


@app.post("/read/{machine}")
def read(machine: str):
    if machine not in MACHINES: raise HTTPException(404, "Unknown machine")
    rows = dataset_rows_for_machine(machine)
    row = active.get(machine)
    if row is None:
        row = rows.iloc[run_counts[machine] % len(rows)]; active[machine] = row
    return {"machine": machine, "values": row_to_values(row), "units": LEGACY_UNITS, "timestamp": str(row.get("Timestamp", "")), "data_source": LEGACY_DATASET.name, "dataset_machine_id": str(row.get("Machine_ID", "")), "machine_type": str(row.get("Machine_Type", "")), "ground_truth": str(row[LEGACY_TARGET]), "failure_flag": int(row.get(FAILURE_TARGET, 0))}


@app.post("/diagnose")
def diagnosis(req: Reading):
    if req.machine not in MACHINES: raise HTTPException(404, "Unknown machine")
    if len(req.values) != len(LEGACY_FEATURES): raise HTTPException(400, f"Exactly {len(LEGACY_FEATURES)} sensor values required")
    fault, confidence, failure_probability, normal_probability, details = legacy_diagnosis(req.values)
    row = active.get(req.machine)
    result = {
        "machine": req.machine,
        "fault": fault,
        "status": "Analysis complete",
        "health": round(confidence, 1),
        "confidence": round(confidence, 1),
        "failure_probability": round(failure_probability, 1) if failure_probability is not None else None,
        "normal_probability": round(normal_probability * 100, 1) if normal_probability is not None else None,
        "agreement": round(sum(d["prediction"] == details[0]["prediction"] for d in details) / len(details) * 100, 1),
        "models": details,
    }
    if row is not None:
        result.update({"ground_truth": str(row[LEGACY_TARGET]), "dataset_machine_id": str(row.get("Machine_ID", "")), "data_source": LEGACY_DATASET.name})
    return result


@app.get("/model/accuracy")
def model_accuracy():
    """Fast endpoint: returns precomputed held-out metrics. No training occurs here."""
    m = load_metrics()
    return {"algorithms": m.get("equipment_anomaly_models", []), "fault_type_backend": m.get("legacy_fault_type_models", []), "evaluation_dataset": "equipment_anomaly_data.csv", "train_rows": m.get("train_rows", 6137), "test_rows": m.get("test_rows", 1535)}


@app.get("/model/evaluation")
def model_evaluation():
    """Backend-only evaluation data, including confusion matrices."""
    m = load_metrics()
    return {"dataset": ANOMALY_DATASET.name, "training_rows": m.get("train_rows", 6137), "test_rows": m.get("test_rows", 1535), "classes": m.get("classes", [0, 1]), "models": m.get("equipment_anomaly_models", []), "confusion_matrices": m.get("equipment_anomaly_confusion_matrices", {})}


@app.get("/model/anomaly/sample")
def anomaly_sample():
    data = pd.read_csv(ANOMALY_DATASET)
    return data.sample(n=1, random_state=42).iloc[0].to_dict()


@app.get("/history/{machine}")
def history(machine: str, samples: int = 30):
    if machine not in MACHINES: raise HTTPException(404, "Unknown machine")
    samples = max(10, min(int(samples), 120)); rows = dataset_rows_for_machine(machine); start = (run_counts[machine] * samples) % max(1, len(rows) - samples); selected = rows.iloc[start:start + samples]
    if len(selected) < samples: selected = pd.concat([selected, rows.iloc[:samples-len(selected)]], ignore_index=True)
    return {"machine": machine, "samples": [row_to_values(r) for _, r in selected.iterrows()], "units": LEGACY_UNITS, "data_source": LEGACY_DATASET.name}
