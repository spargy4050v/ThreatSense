"""Measure federated and centralized stability across three random seeds."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from run_federated_training import (
    BATCH_SIZE,
    LABEL_COL,
    LEAKAGE_THRESHOLD,
    LOCAL_EPOCHS,
    N_CLIENTS,
    ROUNDS,
    TEST_SIZE,
    compute_metrics,
)
from src.baseline import EPOCHS as BASELINE_EPOCHS
from src.client import local_train
from src.leakage_check import screen_leaky_features
from src.model import create_mlp
from src.partition import partition_non_iid
from src.preprocessing import load_dataset, prepare_labels, select_features
from src.server import fed_avg_weighted


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_ROOT / "data" / "Obfuscated-MalMem2022.csv"
RESULTS_PATH = PROJECT_ROOT / "results" / "multiseed_results.json"
SEEDS = [42, 7, 123]
METRIC_NAMES = ["accuracy", "precision", "recall", "f1", "roc_auc"]
METRIC_LABELS = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]


def reset_training_seed(seed: int) -> None:
    """Reset Python, NumPy, and TensorFlow state for one model run."""
    tf.keras.backend.clear_session()
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def prepare_seed_data(
    raw_data: pd.DataFrame,
    seed: int,
) -> tuple[
    list[pd.DataFrame],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
    list[str],
]:
    """Rebuild the screened, scaled, and partitioned pipeline for one seed."""
    labeled_data = prepare_labels(raw_data, label_col=LABEL_COL)
    train_raw, test_raw = train_test_split(
        labeled_data,
        test_size=TEST_SIZE,
        stratify=labeled_data[LABEL_COL],
        random_state=seed,
    )
    train_selected = select_features(train_raw, label_col=LABEL_COL)
    test_selected = select_features(test_raw, label_col=LABEL_COL)

    candidate_features = train_selected.drop(columns=[LABEL_COL])
    y_train_series = train_selected[LABEL_COL].astype(np.int32)
    flagged_features, _ = screen_leaky_features(
        candidate_features,
        y_train_series,
        threshold=LEAKAGE_THRESHOLD,
    )
    feature_names = [
        name
        for name in candidate_features.columns
        if name not in flagged_features
    ]
    if not feature_names:
        raise RuntimeError(f"Seed {seed}: screening removed every feature.")

    X_train = train_selected.loc[:, feature_names].copy()
    X_test = test_selected.loc[:, feature_names].copy()
    y_train = y_train_series.to_numpy(dtype=np.int32)
    y_test = test_selected[LABEL_COL].to_numpy(dtype=np.int32)

    training_means = X_train.mean(numeric_only=True)
    X_train = X_train.fillna(training_means)
    X_test = X_test.fillna(training_means)
    if X_train.isna().any().any() or X_test.isna().any().any():
        raise ValueError(f"Seed {seed}: missing values remain after imputation.")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    scaled_training = pd.DataFrame(
        X_train_scaled,
        columns=feature_names,
        index=train_selected.index,
    )
    scaled_training[LABEL_COL] = y_train
    client_dfs = partition_non_iid(
        scaled_training,
        n_clients=N_CLIENTS,
        label_col=LABEL_COL,
        random_state=seed,
    )
    return (
        client_dfs,
        X_train_scaled,
        y_train,
        X_test_scaled,
        y_test,
        feature_names,
        flagged_features,
    )


def train_federated(
    client_dfs: list[pd.DataFrame],
    X_test: np.ndarray,
    y_test: np.ndarray,
    input_dim: int,
    seed: int,
) -> dict[str, float]:
    """Run the existing eight-round federated procedure for one seed."""
    reset_training_seed(seed)
    global_model = create_mlp(input_dim)
    global_weights = global_model.get_weights()

    for round_number in range(1, ROUNDS + 1):
        local_weight_sets: list[list[np.ndarray]] = []
        client_sizes: list[int] = []
        for client_df in client_dfs:
            X_client = client_df.drop(columns=[LABEL_COL]).to_numpy(
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
            local_weight_sets.append(local_weights)
            client_sizes.append(len(client_df))

        global_weights = fed_avg_weighted(local_weight_sets, client_sizes)
        global_model.set_weights(global_weights)
        metrics = compute_metrics(global_model, X_test, y_test)
        print(
            f"    Federated round {round_number}/{ROUNDS}: "
            f"accuracy={metrics['accuracy']:.4f}",
            flush=True,
        )

    return metrics


def train_centralized(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> dict[str, float]:
    """Run the existing centralized architecture and schedule for one seed."""
    reset_training_seed(seed)
    model = create_mlp(X_train.shape[1])
    model.fit(
        X_train,
        y_train,
        epochs=BASELINE_EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
    )
    return compute_metrics(model, X_test, y_test)


def summarize(
    seed_results: dict[int, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Return sample mean and sample standard deviation for every metric."""
    summary: dict[str, dict[str, float]] = {}
    for metric_name in METRIC_NAMES:
        values = np.array(
            [seed_results[seed][metric_name] for seed in SEEDS],
            dtype=float,
        )
        summary[metric_name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
        }
    return summary


def print_summary_table(
    title: str,
    seed_results: dict[int, dict[str, float]],
    summary: dict[str, dict[str, float]],
) -> None:
    """Print mean, sample standard deviation, and individual seed values."""
    print(f"\n{title}")
    print(
        f"{'Metric':<12}{'Mean':>12}{'Std':>12}"
        + "".join(f"{f'seed {seed}':>14}" for seed in SEEDS)
    )
    print("-" * (36 + 14 * len(SEEDS)))
    for metric_name, metric_label in zip(METRIC_NAMES, METRIC_LABELS):
        individual = "".join(
            f"{seed_results[seed][metric_name]:>14.6f}" for seed in SEEDS
        )
        print(
            f"{metric_label:<12}"
            f"{summary[metric_name]['mean']:>12.6f}"
            f"{summary[metric_name]['std']:>12.6f}"
            f"{individual}"
        )


def main() -> None:
    """Train both approaches for all seeds and persist stability results."""
    raw_data = load_dataset([str(DATASET_PATH)])
    federated_results: dict[int, dict[str, float]] = {}
    centralized_results: dict[int, dict[str, float]] = {}
    feature_audit: dict[int, dict[str, object]] = {}

    for seed in SEEDS:
        print(f"\nSeed {seed}: preparing split and leakage screen", flush=True)
        (
            client_dfs,
            X_train,
            y_train,
            X_test,
            y_test,
            feature_names,
            flagged_features,
        ) = prepare_seed_data(raw_data, seed)
        feature_audit[seed] = {
            "retained_feature_count": len(feature_names),
            "flagged_feature_count": len(flagged_features),
            "retained_features": feature_names,
            "flagged_features": flagged_features,
        }
        print(
            f"  Retained {len(feature_names)} features; "
            f"flagged {len(flagged_features)}",
            flush=True,
        )

        print(f"  Training federated model for seed {seed}", flush=True)
        federated_results[seed] = train_federated(
            client_dfs,
            X_test,
            y_test,
            input_dim=len(feature_names),
            seed=seed,
        )
        print(
            f"  Federated final accuracy: "
            f"{federated_results[seed]['accuracy']:.6f}",
            flush=True,
        )

        print(f"  Training centralized model for seed {seed}", flush=True)
        centralized_results[seed] = train_centralized(
            X_train,
            y_train,
            X_test,
            y_test,
            seed=seed,
        )
        print(
            f"  Centralized final accuracy: "
            f"{centralized_results[seed]['accuracy']:.6f}",
            flush=True,
        )

    federated_summary = summarize(federated_results)
    centralized_summary = summarize(centralized_results)
    print_summary_table(
        "Federated multi-seed stability (sample standard deviation)",
        federated_results,
        federated_summary,
    )
    print_summary_table(
        "Centralized multi-seed stability (sample standard deviation)",
        centralized_results,
        centralized_summary,
    )

    output = {
        "seeds": SEEDS,
        "standard_deviation": "sample (ddof=1)",
        "federated": {
            "per_seed": {
                str(seed): federated_results[seed] for seed in SEEDS
            },
            "summary": federated_summary,
        },
        "centralized": {
            "per_seed": {
                str(seed): centralized_results[seed] for seed in SEEDS
            },
            "summary": centralized_summary,
        },
        "feature_audit": {
            str(seed): feature_audit[seed] for seed in SEEDS
        },
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)
    print(f"\nSaved {RESULTS_PATH}")


if __name__ == "__main__":
    main()
