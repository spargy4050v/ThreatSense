"""Intentionally simple inference UI with no OOD or temperature scaling logic."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, abort, jsonify, render_template, request, send_from_directory


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "federated_global_model.keras"
SCALER_PATH = PROJECT_ROOT / "models" / "scaler.joblib"
FEATURES_PATH = PROJECT_ROOT / "models" / "features.json"
DATASET_PATH = PROJECT_ROOT / "data" / "Obfuscated-MalMem2022.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
FEDERATED_LOG_PATH = RESULTS_DIR / "federated_log.csv"
BASELINE_METRICS_PATH = RESULTS_DIR / "baseline_metrics.json"
LABEL_COLUMN = "Class"
PREDICTION_THRESHOLD = 0.5
METRIC_NAMES = ["accuracy", "precision", "recall", "f1", "roc_auc"]
RESULT_IMAGE_NAMES = {
    "client_distribution_v2.png",
    "confusion_matrix.png",
    "federated_vs_baseline.png",
    "leakage_scores.png",
    "roc_curve.png",
}


def _require_artifacts() -> None:
    missing = [
        str(path.relative_to(PROJECT_ROOT))
        for path in (MODEL_PATH, SCALER_PATH, FEATURES_PATH)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "ThreatSense cannot start because required inference artifacts are "
            f"missing: {', '.join(missing)}. Run run_federated_training.py first."
        )


_require_artifacts()
with FEATURES_PATH.open("r", encoding="utf-8") as file:
    FEATURE_NAMES = json.load(file)

if not isinstance(FEATURE_NAMES, list) or not FEATURE_NAMES or not all(
    isinstance(name, str) and name for name in FEATURE_NAMES
):
    raise ValueError("models/features.json must contain a non-empty list of names.")

SCALER = joblib.load(SCALER_PATH)
MODEL = tf.keras.models.load_model(MODEL_PATH, compile=False)

app = Flask(__name__)


@lru_cache(maxsize=1)
def _load_demo_rows() -> pd.DataFrame:
    """Load only the columns needed by the random-example routes."""
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(
            "Random examples are unavailable because "
            "data/Obfuscated-MalMem2022.csv is missing."
        )

    try:
        rows = pd.read_csv(
            DATASET_PATH,
            usecols=[LABEL_COLUMN, *FEATURE_NAMES],
        )
    except ValueError as exc:
        raise ValueError(
            "The demo dataset does not match the saved feature schema."
        ) from exc
    return rows


def _load_evaluation_summary() -> list[dict[str, float | str]]:
    """Load the final federated and centralized metrics for the dashboard."""
    if not FEDERATED_LOG_PATH.is_file() or not BASELINE_METRICS_PATH.is_file():
        return []

    federated_log = pd.read_csv(FEDERATED_LOG_PATH)
    if federated_log.empty or "round" not in federated_log.columns:
        return []
    with BASELINE_METRICS_PATH.open("r", encoding="utf-8") as file:
        baseline_metrics = json.load(file)

    final_round = federated_log.sort_values("round").iloc[-1]
    if any(name not in final_round or name not in baseline_metrics for name in METRIC_NAMES):
        return []

    display_labels = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
        "roc_auc": "ROC-AUC",
    }
    return [
        {
            "name": name,
            "label": display_labels[name],
            "federated": float(final_round[name]),
            "centralized": float(baseline_metrics[name]),
        }
        for name in METRIC_NAMES
    ]


@app.get("/")
def index() -> str:
    """Render the evaluation dashboard and schema-generated inference form."""
    available_images = {
        filename
        for filename in RESULT_IMAGE_NAMES
        if (RESULTS_DIR / filename).is_file()
    }
    return render_template(
        "index.html",
        feature_names=FEATURE_NAMES,
        metrics=_load_evaluation_summary(),
        result_images=available_images,
    )


@app.get("/results/<path:filename>")
def result_image(filename: str):
    """Serve only the known generated evaluation figures."""
    if filename not in RESULT_IMAGE_NAMES or not (RESULTS_DIR / filename).is_file():
        abort(404)
    return send_from_directory(RESULTS_DIR, filename)


@app.post("/predict")
def predict():
    """Scale one JSON feature record and return its binary prediction."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="Request body must be a JSON object."), 400

    if not any(feature_name in payload for feature_name in FEATURE_NAMES):
        return jsonify(
            error="No feature values were provided. Load an example or enter values first."
        ), 400

    warnings: list[str] = []
    values: list[float] = []
    for feature_name in FEATURE_NAMES:
        if feature_name not in payload:
            values.append(0.0)
            warnings.append(f"Missing {feature_name}; used 0.")
            continue

        raw_value = payload[feature_name]
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            return jsonify(error=f"Feature {feature_name} must be numeric."), 400
        if not np.isfinite(value):
            return jsonify(error=f"Feature {feature_name} must be finite."), 400
        values.append(value)

    feature_row = pd.DataFrame([values], columns=FEATURE_NAMES)
    scaled_row = SCALER.transform(feature_row).astype(np.float32)
    probability = float(MODEL.predict(scaled_row, verbose=0).reshape(-1)[0])
    label = "Malware" if probability >= PREDICTION_THRESHOLD else "Benign"
    return jsonify(label=label, probability=probability, warnings=warnings)


@app.get("/random/<label>")
def random_example(label: str):
    """Return one real benign or malware row for presentation use."""
    normalized_label = label.strip().casefold()
    if normalized_label not in {"benign", "malware"}:
        return jsonify(error="label must be 'benign' or 'malware'."), 404

    try:
        rows = _load_demo_rows()
    except (FileNotFoundError, ValueError) as exc:
        return jsonify(error=str(exc)), 503

    normalized_classes = rows[LABEL_COLUMN].astype(str).str.strip().str.casefold()
    matches = (
        normalized_classes == "benign"
        if normalized_label == "benign"
        else normalized_classes != "benign"
    )
    candidates = rows.loc[matches, FEATURE_NAMES]
    if candidates.empty:
        return jsonify(error=f"No {normalized_label} rows found in the dataset."), 404

    sampled = candidates.sample(n=1)
    sampled = sampled.fillna(0.0)
    features = {
        feature_name: float(sampled.iloc[0][feature_name])
        for feature_name in FEATURE_NAMES
    }
    return jsonify(features)


if __name__ == "__main__":
    app.run(debug=False)
