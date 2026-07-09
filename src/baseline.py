"""Train the centralized ThreatSense comparison model."""

from __future__ import annotations

import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

if __package__:
    from .model import create_mlp
    from .preprocessing import load_dataset, prepare_labels, select_features
else:
    from model import create_mlp
    from preprocessing import load_dataset, prepare_labels, select_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "Obfuscated-MalMem2022.csv"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

RANDOM_STATE = 42
TEST_SIZE = 0.20
EPOCHS = 8
BATCH_SIZE = 32
LABEL_COL = "Class"
METRIC_NAMES = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def set_reproducible_seeds(seed: int = RANDOM_STATE) -> None:
    """Seed Python, NumPy, and TensorFlow for repeatable baseline training."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def compute_metrics(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Calculate metrics using the same threshold as the federated script."""
    probabilities = model.predict(X_test, verbose=0).reshape(-1)
    predictions = (probabilities >= 0.5).astype(np.int32)
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }


def load_comparable_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reuse the federated feature schema and scaler on the identical split."""
    feature_path = MODELS_DIR / "features.json"
    scaler_path = MODELS_DIR / "scaler.joblib"
    if not feature_path.is_file() or not scaler_path.is_file():
        raise FileNotFoundError(
            "Run run_federated_training.py first to create features.json "
            "and scaler.joblib."
        )

    with feature_path.open("r", encoding="utf-8") as file:
        feature_names = json.load(file)
    scaler = joblib.load(scaler_path)

    raw_data = load_dataset([str(DATASET_PATH)])
    labeled_data = prepare_labels(raw_data, label_col=LABEL_COL)
    train_raw, test_raw = train_test_split(
        labeled_data,
        test_size=TEST_SIZE,
        stratify=labeled_data[LABEL_COL],
        random_state=RANDOM_STATE,
    )
    train_selected = select_features(train_raw, label_col=LABEL_COL)
    test_selected = select_features(test_raw, label_col=LABEL_COL)

    missing_train = [name for name in feature_names if name not in train_selected]
    missing_test = [name for name in feature_names if name not in test_selected]
    if missing_train or missing_test:
        raise ValueError(
            f"Saved feature schema does not match the dataset. "
            f"Missing from train={missing_train}, test={missing_test}."
        )

    X_train = train_selected.loc[:, feature_names].copy()
    X_test = test_selected.loc[:, feature_names].copy()
    y_train = train_selected[LABEL_COL].to_numpy(dtype=np.int32)
    y_test = test_selected[LABEL_COL].to_numpy(dtype=np.int32)

    training_means = X_train.mean(numeric_only=True)
    X_train = X_train.fillna(training_means)
    X_test = X_test.fillna(training_means)
    if X_train.isna().any().any() or X_test.isna().any().any():
        raise ValueError("Missing values remain after training-mean imputation.")

    X_train_scaled = scaler.transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)
    return X_train_scaled, y_train, X_test_scaled, y_test


def train_baseline() -> dict[str, float]:
    """Train, evaluate, and persist the centralized comparison model."""
    set_reproducible_seeds()
    X_train, y_train, X_test, y_test = load_comparable_data()

    model = create_mlp(X_train.shape[1])
    model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
    )
    metrics = compute_metrics(model, X_test, y_test)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with (RESULTS_DIR / "baseline_metrics.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=2)
    model.save(MODELS_DIR / "centralized_baseline.keras")

    federated_log_path = RESULTS_DIR / "federated_log.csv"
    if not federated_log_path.is_file():
        raise FileNotFoundError(
            "Federated log is missing; run federated training before baseline."
        )
    federated_metrics = (
        pd.read_csv(federated_log_path)
        .sort_values("round")
        .iloc[-1]
        .to_dict()
    )

    print("\nFinal comparison on the identical held-out test set:")
    print(f"{'Metric':<12}{'Federated':>14}{'Centralized':>14}")
    print("-" * 40)
    for metric_name in METRIC_NAMES:
        print(
            f"{metric_name:<12}"
            f"{float(federated_metrics[metric_name]):>14.6f}"
            f"{metrics[metric_name]:>14.6f}"
        )
    return metrics


if __name__ == "__main__":
    train_baseline()
