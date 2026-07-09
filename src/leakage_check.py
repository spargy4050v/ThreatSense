"""Rank features by how accurately each one predicts the label on its own.

This is a single-feature predictiveness screen, not a leakage detector. A
feature scoring high alone could be a genuine strong indicator or a dataset
artifact — this function only flags candidates; a human decides what to drop.

The screen cannot detect leakage that appears only through feature
combinations, duplicated observations, group overlap, or other split-design
problems. It does not remove columns; screening and removal remain separate
pipeline decisions.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


def screen_leaky_features(
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.85,
    cv_folds: int = 5,
) -> tuple[list[str], list[tuple[str, float]]]:
    """Score every feature with a standalone cross-validated classifier.

    A logistic-regression model is fitted separately for each feature using
    only that one column. The returned scores are mean validation accuracies,
    sorted from highest to lowest. Features meeting ``threshold`` are reported
    as candidates for human review, not automatically removed.
    """
    if X.empty or X.shape[1] == 0:
        raise ValueError("X must contain at least one row and one feature.")
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of observations.")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")
    if cv_folds < 2:
        raise ValueError("cv_folds must be at least 2.")
    if y.nunique(dropna=False) < 2:
        raise ValueError("y must contain at least two classes.")

    scores: list[tuple[str, float]] = []

    for feature_name in X.columns:
        classifier = LogisticRegression(max_iter=300, solver="liblinear")
        fold_scores = cross_val_score(
            classifier,
            X[[feature_name]],
            y,
            cv=cv_folds,
            scoring="accuracy",
        )
        scores.append((feature_name, float(fold_scores.mean())))

    scores.sort(key=lambda item: item[1], reverse=True)
    suspects = [
        feature_name
        for feature_name, mean_accuracy in scores
        if mean_accuracy >= threshold
    ]
    return suspects, scores


def plot_leakage_scores(
    scores: list[tuple[str, float]],
    save_path: str | Path,
    threshold: float = 0.85,
) -> None:
    """Save a descending bar chart of standalone feature accuracies."""
    if not scores:
        raise ValueError("scores must contain at least one feature result.")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")

    ordered_scores = sorted(scores, key=lambda item: item[1], reverse=True)
    feature_names = [feature for feature, _ in ordered_scores]
    accuracies = [accuracy for _, accuracy in ordered_scores]

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(18, 8))
    axis.bar(range(len(feature_names)), accuracies, color="#176b87")
    axis.axhline(
        threshold,
        color="#c43d3d",
        linestyle="--",
        linewidth=2,
        label=f"Review threshold ({threshold:.2f})",
    )
    axis.set_title("Standalone Feature Predictiveness")
    axis.set_xlabel("Features ranked by mean cross-validation accuracy")
    axis.set_ylabel("Mean accuracy")
    axis.set_ylim(0.0, 1.05)
    axis.set_xticks(range(len(feature_names)))
    axis.set_xticklabels(feature_names, rotation=90, fontsize=8)
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    from preprocessing import load_dataset, prepare_labels, select_features

    project_root = Path(__file__).resolve().parents[1]
    dataset_path = project_root / "data" / "Obfuscated-MalMem2022.csv"
    plot_path = project_root / "results" / "leakage_scores.png"

    raw_data = load_dataset([str(dataset_path)])
    labeled_data = prepare_labels(raw_data)
    cleaned_data = select_features(labeled_data)

    features = cleaned_data.drop(columns=["Class"])
    labels = cleaned_data["Class"]
    flagged_features, feature_scores = screen_leaky_features(
        features,
        labels,
        threshold=0.85,
    )
    plot_leakage_scores(feature_scores, plot_path, threshold=0.85)

    print(f"Suspect features ({len(flagged_features)}):")
    for feature in flagged_features:
        print(f"- {feature}")
    print(f"Saved leakage score plot to {plot_path}")
