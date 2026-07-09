"""Deterministic dataset loading and cleaning for ThreatSense.

This module does not perform leakage screening, feature scaling, or train/test
splitting. Those operations belong to later pipeline stages so that every
data-dependent decision can be fitted on training data without contaminating
the held-out test set.
"""

from pathlib import Path

import pandas as pd


def load_dataset(paths: list[str]) -> pd.DataFrame:
    """Load and return the first readable CSV from an ordered path list.

    Paths that do not exist or cannot be parsed as CSV files are skipped so a
    caller can provide fallback locations. If none can be loaded, the raised
    error reports every attempted path.
    """
    attempted_paths: list[str] = []
    load_errors: list[str] = []

    for candidate in paths:
        path = Path(candidate)
        attempted_paths.append(str(path))

        if not path.is_file():
            continue

        try:
            return pd.read_csv(path)
        except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            load_errors.append(f"{path}: {exc}")

    message = f"No readable CSV dataset found. Attempted: {attempted_paths}"
    if load_errors:
        message += f". Load errors: {load_errors}"
    raise FileNotFoundError(message)


def prepare_labels(
    df: pd.DataFrame,
    label_col: str = "Class",
) -> pd.DataFrame:
    """Return a copy whose label column uses binary benign/malware values.

    ``Benign`` values, matched after stripping whitespace and ignoring case,
    become 0; every other value becomes 1. This deliberately collapses all
    malware families (including ransomware, spyware, trojans, and others) into
    class 1. The resulting scope is therefore general malware detection rather
    than ransomware-specific detection, which is a known reporting limitation
    rather than an implementation bug.
    """
    if label_col not in df.columns:
        raise KeyError(f"Label column {label_col!r} is missing from the dataset.")

    labeled = df.copy()
    normalized = labeled[label_col].astype(str).str.strip().str.casefold()
    labeled[label_col] = (normalized != "benign").astype(int)
    return labeled


def select_features(
    df: pd.DataFrame,
    drop_cols: list[str] | None = None,
    label_col: str = "Class",
) -> pd.DataFrame:
    """Return numeric feature columns and the label from a copied dataframe."""
    if label_col not in df.columns:
        raise KeyError(f"Label column {label_col!r} is missing from the dataset.")

    selected = df.copy()
    requested_drops = drop_cols or []
    removable = [
        column
        for column in requested_drops
        if column != label_col and column in selected.columns
    ]
    if removable:
        selected = selected.drop(columns=removable)

    numeric_columns = selected.select_dtypes(include="number").columns.tolist()
    retained_columns = [
        column
        for column in selected.columns
        if column in numeric_columns or column == label_col
    ]
    return selected.loc[:, retained_columns]


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    dataset_path = project_root / "data" / "Obfuscated-MalMem2022.csv"

    if not dataset_path.is_file():
        print(f"Sanity check skipped: dataset not found at {dataset_path}")
    else:
        raw_data = load_dataset([str(dataset_path)])
        labeled_data = prepare_labels(raw_data)
        cleaned_data = select_features(labeled_data)

        print(f"Shape before cleaning: {raw_data.shape}")
        print(f"Shape after cleaning:  {cleaned_data.shape}")
        print("Binary label distribution:")
        print(cleaned_data["Class"].value_counts().sort_index().to_string())
