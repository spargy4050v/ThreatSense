"""Run the complete ThreatSense federated-training experiment."""

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
from sklearn.preprocessing import StandardScaler

from src.client import local_train
from src.leakage_check import screen_leaky_features
from src.model import create_mlp
from src.partition import partition_non_iid_v2
from src.preprocessing import load_dataset, prepare_labels, select_features
from src.server import fed_avg_weighted


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_ROOT / "data" / "Obfuscated-MalMem2022.csv"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

RANDOM_STATE = 42
TEST_SIZE = 0.20
LEAKAGE_THRESHOLD = 0.85
N_CLIENTS = 4
ROUNDS = 8
LOCAL_EPOCHS = 1
BATCH_SIZE = 32
LABEL_COL = "Class"
CATEGORY_COL = "Category"
SCALER_PATH = MODELS_DIR / "scaler_v2.joblib"
FEATURES_PATH = MODELS_DIR / "features_v2.json"
MODEL_PATH = MODELS_DIR / "federated_global_model_v2.keras"
FEDERATED_LOG_PATH = RESULTS_DIR / "federated_log_v2.csv"


def set_reproducible_seeds(seed: int = RANDOM_STATE) -> None:
    """Seed Python, NumPy, and TensorFlow for repeatable experiment runs."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def compute_metrics(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Calculate the five classification metrics used by the project."""
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


def prepare_federated_data() -> tuple[
    list[pd.DataFrame],
    np.ndarray,
    np.ndarray,
    list[str],
]:
    """Prepare clients and test arrays without using test data for fitting."""
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

    X_train_candidates = train_selected.drop(columns=[LABEL_COL])
    y_train = train_selected[LABEL_COL].astype(np.int32)
    y_test = test_selected[LABEL_COL].astype(np.int32)

    suspect_features, feature_scores = screen_leaky_features(
        X_train_candidates,
        y_train,
        threshold=LEAKAGE_THRESHOLD,
    )
    feature_names = [
        column
        for column in X_train_candidates.columns
        if column not in suspect_features
    ]
    if not feature_names:
        raise RuntimeError("Leakage screening removed every candidate feature.")

    X_train = X_train_candidates.loc[:, feature_names].copy()
    X_test = test_selected.loc[:, feature_names].copy()

    training_means = X_train.mean(numeric_only=True)
    X_train = X_train.fillna(training_means)
    X_test = X_test.fillna(training_means)
    if X_train.isna().any().any() or X_test.isna().any().any():
        raise ValueError("Missing values remain after training-mean imputation.")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    with FEATURES_PATH.open("w", encoding="utf-8") as file:
        json.dump(feature_names, file, indent=2)

    scaled_training = pd.DataFrame(
        X_train_scaled,
        columns=feature_names,
        index=X_train.index,
    )
    scaled_training[LABEL_COL] = y_train.to_numpy()
    scaled_training[CATEGORY_COL] = train_raw.loc[
        scaled_training.index, CATEGORY_COL
    ]
    client_dfs = partition_non_iid_v2(
        scaled_training,
        n_clients=N_CLIENTS,
        label_col=LABEL_COL,
        category_col=CATEGORY_COL,
        random_state=RANDOM_STATE,
    )

    print(
        f"Training rows: {len(X_train):,} | Test rows: {len(X_test):,}",
        flush=True,
    )
    print(
        f"Candidate features: {len(X_train_candidates.columns)} | "
        f"Flagged: {len(suspect_features)} | Retained: {len(feature_names)}",
        flush=True,
    )
    print("Flagged feature candidates:", flush=True)
    for feature_name in suspect_features:
        score = dict(feature_scores)[feature_name]
        print(f"  {feature_name}: {score:.4f}", flush=True)

    return (
        client_dfs,
        X_test_scaled,
        y_test.to_numpy(dtype=np.int32),
        feature_names,
    )


def run_federated_training() -> dict[str, float]:
    """Train for eight FedAvg rounds and persist the global artifacts."""
    set_reproducible_seeds()
    client_dfs, X_test, y_test, feature_names = prepare_federated_data()

    input_dim = len(feature_names)
    global_model = create_mlp(input_dim)
    global_weights = global_model.get_weights()
    round_history: list[dict[str, float | int]] = []

    for round_number in range(1, ROUNDS + 1):
        local_weight_sets: list[list[np.ndarray]] = []
        client_sizes: list[int] = []
        update_norms: list[float] = []

        for client_df in client_dfs:
            X_client = client_df.drop(columns=[LABEL_COL, CATEGORY_COL]).to_numpy(
                dtype=np.float32
            )
            y_client = client_df[LABEL_COL].to_numpy(dtype=np.int32)
            local_weights = local_train(
                global_weights,
                X_client,
                y_client,
                input_dim=input_dim,
                epochs=LOCAL_EPOCHS,
                batch_size=BATCH_SIZE,
                verbose=0,
            )
            update_norm = float(
                np.sqrt(
                    sum(
                        np.sum((local - global_weight) ** 2)
                        for local, global_weight in zip(
                            local_weights,
                            global_weights,
                        )
                    )
                )
            )
            local_weight_sets.append(local_weights)
            client_sizes.append(len(client_df))
            update_norms.append(update_norm)

        global_weights = fed_avg_weighted(local_weight_sets, client_sizes)
        global_model.set_weights(global_weights)
        metrics = compute_metrics(global_model, X_test, y_test)
        round_history.append({"round": round_number, **metrics})

        metric_text = " | ".join(
            f"{name}={value:.4f}" for name, value in metrics.items()
        )
        norm_text = ", ".join(f"{value:.3f}" for value in update_norms)
        print(
            f"Round {round_number}/{ROUNDS} | {metric_text} | "
            f"client_update_l2=[{norm_text}]",
            flush=True,
        )

    pd.DataFrame(round_history).to_csv(
        FEDERATED_LOG_PATH,
        index=False,
    )
    global_model.save(MODEL_PATH)

    final_metrics = {
        key: float(value)
        for key, value in round_history[-1].items()
        if key != "round"
    }
    print("\nFinal federated test metrics:", flush=True)
    for metric_name, metric_value in final_metrics.items():
        print(f"  {metric_name:9s}: {metric_value:.6f}", flush=True)
    return final_metrics


if __name__ == "__main__":
    run_federated_training()
