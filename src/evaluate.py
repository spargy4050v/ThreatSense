"""Generate final ThreatSense evaluation metrics and result visualizations."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

if __package__:
    from .baseline import load_comparable_data
else:
    from baseline import load_comparable_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FEDERATED_MODEL_PATH = MODELS_DIR / "federated_global_model.keras"
FEDERATED_LOG_PATH = RESULTS_DIR / "federated_log.csv"
BASELINE_METRICS_PATH = RESULTS_DIR / "baseline_metrics.json"
METRIC_NAMES = ["accuracy", "precision", "recall", "f1", "roc_auc"]
METRIC_LABELS = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
PREDICTION_THRESHOLD = 0.5


def _prepare_output_path(save_path: str | Path) -> Path:
    """Create the output directory and return a normalized path."""
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def _predict_probabilities(
    model: tf.keras.Model,
    X_test: np.ndarray,
) -> np.ndarray:
    """Return one finite probability for every test row."""
    probabilities = np.asarray(
        model.predict(X_test, verbose=0),
        dtype=float,
    ).reshape(-1)
    if len(probabilities) != len(X_test):
        raise ValueError("Model returned a different number of predictions than rows.")
    if not np.isfinite(probabilities).all():
        raise ValueError("Model predictions contain non-finite values.")
    return probabilities


def plot_confusion_matrix(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_path: str | Path,
) -> np.ndarray:
    """Plot the binary confusion matrix using a fixed 0.5 threshold."""
    probabilities = _predict_probabilities(model, X_test)
    predictions = (probabilities >= PREDICTION_THRESHOLD).astype(np.int32)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])

    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        title="Federated Model Confusion Matrix",
        xlabel="Predicted label",
        ylabel="True label",
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Benign", "Malware"],
        yticklabels=["Benign", "Malware"],
    )

    threshold = matrix.max() / 2 if matrix.size else 0
    for row_id in range(matrix.shape[0]):
        for column_id in range(matrix.shape[1]):
            value = int(matrix[row_id, column_id])
            axis.text(
                column_id,
                row_id,
                f"{value:,}",
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=12,
            )

    figure.tight_layout()
    figure.savefig(_prepare_output_path(save_path), dpi=200, bbox_inches="tight")
    plt.close(figure)
    return matrix


def plot_roc_curve(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_path: str | Path,
) -> float:
    """Plot the federated model ROC curve and return its ROC-AUC."""
    probabilities = _predict_probabilities(model, X_test)
    false_positive_rate, true_positive_rate, _ = roc_curve(
        y_test,
        probabilities,
    )
    auc_value = float(roc_auc_score(y_test, probabilities))

    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot(
        false_positive_rate,
        true_positive_rate,
        linewidth=2,
        label=f"Federated model (AUC = {auc_value:.4f})",
    )
    axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        label="Random classifier",
    )
    axis.set(
        title="Federated Model ROC Curve",
        xlabel="False positive rate",
        ylabel="True positive rate",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(_prepare_output_path(save_path), dpi=200, bbox_inches="tight")
    plt.close(figure)
    return auc_value


def plot_federated_vs_baseline(
    federated_log_csv: str | Path,
    baseline_metrics_json: str | Path,
    save_path: str | Path,
) -> tuple[dict[str, float], dict[str, float]]:
    """Plot final federated metrics beside the centralized baseline."""
    federated_log = pd.read_csv(federated_log_csv)
    if federated_log.empty or "round" not in federated_log.columns:
        raise ValueError("Federated log must contain at least one numbered round.")

    with Path(baseline_metrics_json).open("r", encoding="utf-8") as file:
        baseline_data = json.load(file)

    missing_federated = [
        name for name in METRIC_NAMES if name not in federated_log.columns
    ]
    missing_baseline = [name for name in METRIC_NAMES if name not in baseline_data]
    if missing_federated or missing_baseline:
        raise ValueError(
            "Metric files are incomplete. "
            f"Federated missing={missing_federated}; "
            f"baseline missing={missing_baseline}."
        )

    final_round = federated_log.sort_values("round").iloc[-1]
    federated_metrics = {
        name: float(final_round[name]) for name in METRIC_NAMES
    }
    baseline_metrics = {
        name: float(baseline_data[name]) for name in METRIC_NAMES
    }

    positions = np.arange(len(METRIC_NAMES))
    bar_width = 0.36
    figure, axis = plt.subplots(figsize=(9, 5.5))
    federated_bars = axis.bar(
        positions - bar_width / 2,
        [federated_metrics[name] for name in METRIC_NAMES],
        bar_width,
        label="Federated",
        color="#176b87",
    )
    baseline_bars = axis.bar(
        positions + bar_width / 2,
        [baseline_metrics[name] for name in METRIC_NAMES],
        bar_width,
        label="Centralized",
        color="#e58e26",
    )
    axis.set(
        title="Federated vs Centralized Performance",
        xlabel="Metric",
        ylabel="Score",
        xticks=positions,
        xticklabels=METRIC_LABELS,
        ylim=(0, 1.08),
    )
    axis.bar_label(federated_bars, fmt="%.3f", padding=3, fontsize=8)
    axis.bar_label(baseline_bars, fmt="%.3f", padding=3, fontsize=8)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(_prepare_output_path(save_path), dpi=200, bbox_inches="tight")
    plt.close(figure)
    return federated_metrics, baseline_metrics


def _require_files(paths: list[Path]) -> None:
    """Fail with one clear message when saved evaluation inputs are missing."""
    missing = [str(path.relative_to(PROJECT_ROOT)) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Evaluation inputs are missing: "
            f"{', '.join(missing)}. Run federated training and the baseline first."
        )


def main() -> None:
    """Load saved artifacts, rebuild the held-out split, and create plots."""
    _require_files(
        [FEDERATED_MODEL_PATH, FEDERATED_LOG_PATH, BASELINE_METRICS_PATH]
    )
    model = tf.keras.models.load_model(FEDERATED_MODEL_PATH, compile=False)
    _, _, X_test, y_test = load_comparable_data()

    matrix = plot_confusion_matrix(
        model,
        X_test,
        y_test,
        RESULTS_DIR / "confusion_matrix.png",
    )
    auc_value = plot_roc_curve(
        model,
        X_test,
        y_test,
        RESULTS_DIR / "roc_curve.png",
    )
    federated_metrics, baseline_metrics = plot_federated_vs_baseline(
        FEDERATED_LOG_PATH,
        BASELINE_METRICS_PATH,
        RESULTS_DIR / "federated_vs_baseline.png",
    )

    print("Final comparison on the identical held-out test set:")
    print(f"{'Metric':<12}{'Federated':>14}{'Centralized':>14}")
    print("-" * 40)
    for metric_name, label in zip(METRIC_NAMES, METRIC_LABELS):
        print(
            f"{label:<12}"
            f"{federated_metrics[metric_name]:>14.6f}"
            f"{baseline_metrics[metric_name]:>14.6f}"
        )
    print("\nFederated confusion matrix [[TN, FP], [FN, TP]]:")
    print(matrix)
    print(f"Federated ROC-AUC from saved model: {auc_value:.6f}")
    print(f"Saved plots to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
