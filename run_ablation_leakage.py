"""Compare federated training with leakage screening enabled and disabled."""

from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from run_federated_training import (
    BATCH_SIZE,
    CATEGORY_COL,
    LABEL_COL,
    LEAKAGE_THRESHOLD,
    LOCAL_EPOCHS,
    N_CLIENTS,
    RANDOM_STATE,
    ROUNDS,
    TEST_SIZE,
    compute_metrics,
)
from src.client import local_train
from src.leakage_check import screen_leaky_features
from src.model import create_mlp
from src.partition import partition_non_iid_v2
from src.preprocessing import load_dataset, prepare_labels, select_features
from src.server import fed_avg_weighted


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_ROOT / "data" / "Obfuscated-MalMem2022.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
JSON_PATH = RESULTS_DIR / "leakage_ablation_v2.json"
PLOT_PATH = RESULTS_DIR / "leakage_ablation_plot_v2.png"
METRIC_NAMES = ["accuracy", "precision", "recall", "f1", "roc_auc"]
METRIC_LABELS = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
NEARLY_IDENTICAL_TOLERANCE = 0.001


def set_condition_seed(seed: int = RANDOM_STATE) -> None:
    """Reset all training randomness before an ablation condition."""
    tf.keras.backend.clear_session()
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def prepare_condition(
    train_selected: pd.DataFrame,
    test_selected: pd.DataFrame,
    train_categories: pd.Series,
    feature_names: list[str],
) -> tuple[list[pd.DataFrame], np.ndarray, np.ndarray]:
    """Scale and partition one feature condition without sharing fitted state."""
    X_train = train_selected.loc[:, feature_names].copy()
    X_test = test_selected.loc[:, feature_names].copy()
    y_train = train_selected[LABEL_COL].to_numpy(dtype=np.int32)
    y_test = test_selected[LABEL_COL].to_numpy(dtype=np.int32)

    training_means = X_train.mean(numeric_only=True)
    X_train = X_train.fillna(training_means)
    X_test = X_test.fillna(training_means)
    if X_train.isna().any().any() or X_test.isna().any().any():
        raise ValueError("Missing values remain after training-mean imputation.")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    scaled_training = pd.DataFrame(
        X_train_scaled,
        columns=feature_names,
        index=train_selected.index,
    )
    scaled_training[LABEL_COL] = y_train
    scaled_training[CATEGORY_COL] = train_categories.loc[scaled_training.index]
    clients = partition_non_iid_v2(
        scaled_training,
        n_clients=N_CLIENTS,
        label_col=LABEL_COL,
        category_col=CATEGORY_COL,
        random_state=RANDOM_STATE,
    )
    return clients, X_test_scaled, y_test


def train_condition(
    condition_name: str,
    client_dfs: list[pd.DataFrame],
    X_test: np.ndarray,
    y_test: np.ndarray,
    input_dim: int,
) -> dict[str, float]:
    """Run the unchanged eight-round federated loop for one condition."""
    set_condition_seed()
    global_model = create_mlp(input_dim)
    global_weights = global_model.get_weights()

    print(f"\n{condition_name}: {input_dim} features")
    for round_number in range(1, ROUNDS + 1):
        local_weight_sets: list[list[np.ndarray]] = []
        client_sizes: list[int] = []

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
            local_weight_sets.append(local_weights)
            client_sizes.append(len(client_df))

        global_weights = fed_avg_weighted(local_weight_sets, client_sizes)
        global_model.set_weights(global_weights)
        round_metrics = compute_metrics(global_model, X_test, y_test)
        print(
            f"  Round {round_number}/{ROUNDS}: "
            + " | ".join(
                f"{name}={round_metrics[name]:.4f}" for name in METRIC_NAMES
            ),
            flush=True,
        )

    return round_metrics


def save_comparison_plot(
    with_screening: dict[str, float],
    without_screening: dict[str, float],
) -> None:
    """Save a grouped metric chart for the two ablation conditions."""
    positions = np.arange(len(METRIC_NAMES))
    bar_width = 0.36
    figure, axis = plt.subplots(figsize=(9, 5.5))
    screened_bars = axis.bar(
        positions - bar_width / 2,
        [with_screening[name] for name in METRIC_NAMES],
        bar_width,
        label="With leakage screening",
        color="#176b87",
    )
    unscreened_bars = axis.bar(
        positions + bar_width / 2,
        [without_screening[name] for name in METRIC_NAMES],
        bar_width,
        label="Without leakage screening",
        color="#e58e26",
    )
    axis.set(
        title="Leakage-Screening Ablation",
        xlabel="Metric",
        ylabel="Score",
        xticks=positions,
        xticklabels=METRIC_LABELS,
        ylim=(0, 1.08),
    )
    axis.bar_label(screened_bars, fmt="%.3f", padding=3, fontsize=8)
    axis.bar_label(unscreened_bars, fmt="%.3f", padding=3, fontsize=8)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(PLOT_PATH, dpi=200, bbox_inches="tight")
    plt.close(figure)


def interpretation_note(
    with_screening_accuracy: float,
    without_screening_accuracy: float,
) -> str:
    """Describe the observed accuracy difference without hiding its direction."""
    difference = without_screening_accuracy - with_screening_accuracy
    if abs(difference) <= NEARLY_IDENTICAL_TOLERANCE:
        return (
            "Leakage screening did not inflate reported accuracy — flagged "
            "features were redundant/artifactual, not load-bearing."
        )
    if difference > 0:
        return (
            "Accuracy was higher without leakage screening by "
            f"{difference:.6f}; screening removed predictive signal, so this "
            "trade-off should be reported explicitly."
        )
    return (
        "Accuracy was higher with leakage screening by "
        f"{-difference:.6f}, indicating improved held-out generalization after "
        "the flagged features were removed."
    )


def main() -> None:
    """Run both ablation conditions on one identical held-out split."""
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

    candidate_features = train_selected.drop(columns=[LABEL_COL])
    y_train = train_selected[LABEL_COL].astype(np.int32)
    flagged_features, _ = screen_leaky_features(
        candidate_features,
        y_train,
        threshold=LEAKAGE_THRESHOLD,
    )
    features_full = candidate_features.columns.tolist()
    features_screened = [
        name for name in features_full if name not in flagged_features
    ]
    if not features_screened:
        raise RuntimeError("Leakage screening removed every candidate feature.")

    print("Features included without screening but excluded with screening:")
    for feature_name in flagged_features:
        print(f"  - {feature_name}")

    clients_full, X_test_full, y_test_full = prepare_condition(
        train_selected,
        test_selected,
        train_raw[CATEGORY_COL],
        features_full,
    )
    without_screening = train_condition(
        "Condition A — leakage screening OFF",
        clients_full,
        X_test_full,
        y_test_full,
        input_dim=len(features_full),
    )

    clients_screened, X_test_screened, y_test_screened = prepare_condition(
        train_selected,
        test_selected,
        train_raw[CATEGORY_COL],
        features_screened,
    )
    with_screening = train_condition(
        "Condition B — leakage screening ON",
        clients_screened,
        X_test_screened,
        y_test_screened,
        input_dim=len(features_screened),
    )

    print("\nFinal leakage-screening ablation:")
    print(f"{'Metric':<12}{'With screening':>18}{'Without screening':>21}")
    print("-" * 51)
    comparison_table: list[dict[str, float | str]] = []
    for metric_name, metric_label in zip(METRIC_NAMES, METRIC_LABELS):
        print(
            f"{metric_label:<12}"
            f"{with_screening[metric_name]:>18.6f}"
            f"{without_screening[metric_name]:>21.6f}"
        )
        comparison_table.append(
            {
                "metric": metric_name,
                "with_leakage_screening": with_screening[metric_name],
                "without_leakage_screening": without_screening[metric_name],
            }
        )

    note = interpretation_note(
        with_screening["accuracy"],
        without_screening["accuracy"],
    )
    print(f"\n{note}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "partitioner": "partition_non_iid_v2",
        "flagged_features": flagged_features,
        "feature_counts": {
            "with_leakage_screening": len(features_screened),
            "without_leakage_screening": len(features_full),
        },
        "comparison_table": comparison_table,
        "with_leakage_screening": with_screening,
        "without_leakage_screening": without_screening,
        "accuracy_delta_without_minus_with": (
            without_screening["accuracy"] - with_screening["accuracy"]
        ),
        "interpretation": note,
    }
    with JSON_PATH.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)
    save_comparison_plot(with_screening, without_screening)
    print(f"Saved {JSON_PATH}")
    print(f"Saved {PLOT_PATH}")


if __name__ == "__main__":
    main()
