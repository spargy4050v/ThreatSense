"""Create and visualize heterogeneous client datasets for ThreatSense.

Non-IID partitioning matters because real organizations do not observe
identical benign-traffic and malware mixtures. Giving simulated clients
different class balances stress-tests whether federated averaging can still
converge to a useful shared model under realistic data heterogeneity.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def partition_non_iid(
    df: pd.DataFrame,
    n_clients: int = 4,
    label_col: str = "Class",
    random_state: int = 42,
) -> list[pd.DataFrame]:
    """Partition binary-labeled rows across clients with a class-ratio gradient.

    Client target sizes differ by at most one row. Benign target ratios descend
    evenly from 0.8 for the first client to 0.2 for the last. Rows are consumed
    with sequential class-specific pointers, so no source row is assigned more
    than once. Rounding or availability leftovers are distributed round-robin.
    """
    if n_clients < 2:
        raise ValueError("n_clients must be at least 2 for a non-IID partition.")
    if label_col not in df.columns:
        raise KeyError(f"Label column {label_col!r} is missing from the dataset.")
    if df.empty:
        raise ValueError("df must contain at least one row.")
    if df[label_col].isna().any():
        raise ValueError(f"Label column {label_col!r} must not contain missing values.")

    observed_labels = set(df[label_col].unique().tolist())
    if not observed_labels.issubset({0, 1}):
        raise ValueError(
            f"Label column {label_col!r} must be binary 0/1; "
            f"found {sorted(observed_labels, key=str)}."
        )

    benign_rows = (
        df[df[label_col] == 0]
        .sample(frac=1.0, random_state=random_state)
        .copy()
    )
    malware_rows = (
        df[df[label_col] == 1]
        .sample(frac=1.0, random_state=random_state)
        .copy()
    )

    base_size, remainder = divmod(len(df), n_clients)
    target_sizes = [
        base_size + (1 if client_id < remainder else 0)
        for client_id in range(n_clients)
    ]
    benign_ratios = np.linspace(0.8, 0.2, n_clients)

    client_dfs: list[pd.DataFrame] = []
    benign_pointer = 0
    malware_pointer = 0

    for client_id, (target_size, benign_ratio) in enumerate(
        zip(target_sizes, benign_ratios)
    ):
        requested_benign = int(round(target_size * benign_ratio))
        requested_malware = target_size - requested_benign

        benign_count = min(
            requested_benign,
            len(benign_rows) - benign_pointer,
        )
        malware_count = min(
            requested_malware,
            len(malware_rows) - malware_pointer,
        )

        benign_part = benign_rows.iloc[
            benign_pointer : benign_pointer + benign_count
        ]
        malware_part = malware_rows.iloc[
            malware_pointer : malware_pointer + malware_count
        ]
        benign_pointer += benign_count
        malware_pointer += malware_count

        client_df = pd.concat([benign_part, malware_part], axis=0)
        client_dfs.append(client_df)

    leftovers = pd.concat(
        [
            benign_rows.iloc[benign_pointer:],
            malware_rows.iloc[malware_pointer:],
        ],
        axis=0,
    ).sample(frac=1.0, random_state=random_state)

    for client_id in range(n_clients):
        round_robin_rows = leftovers.iloc[client_id::n_clients]
        if not round_robin_rows.empty:
            client_dfs[client_id] = pd.concat(
                [client_dfs[client_id], round_robin_rows],
                axis=0,
            )

    return [
        client_df.sample(
            frac=1.0,
            random_state=random_state + client_id,
        )
        for client_id, client_df in enumerate(client_dfs)
    ]


def plot_client_distribution(
    client_dfs: list[pd.DataFrame],
    label_col: str,
    save_path: str,
) -> None:
    """Save a stacked bar chart of benign and malware counts per client."""
    if not client_dfs:
        raise ValueError("client_dfs must contain at least one client dataframe.")
    if any(label_col not in client_df.columns for client_df in client_dfs):
        raise KeyError(f"Every client dataframe must contain {label_col!r}.")

    benign_counts = [
        int((client_df[label_col] == 0).sum())
        for client_df in client_dfs
    ]
    malware_counts = [
        int((client_df[label_col] == 1).sum())
        for client_df in client_dfs
    ]
    client_labels = [
        f"Client {client_id + 1}"
        for client_id in range(len(client_dfs))
    ]

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(9, 6))
    axis.bar(client_labels, benign_counts, label="Benign", color="#4c9be8")
    axis.bar(
        client_labels,
        malware_counts,
        bottom=benign_counts,
        label="Malware",
        color="#d85852",
    )
    axis.set_title("Non-IID Class Distribution Across Simulated Clients")
    axis.set_xlabel("Simulated organization")
    axis.set_ylabel("Number of samples")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    from preprocessing import load_dataset, prepare_labels, select_features

    project_root = Path(__file__).resolve().parents[1]
    dataset_path = project_root / "data" / "Obfuscated-MalMem2022.csv"
    plot_path = project_root / "results" / "client_distribution.png"

    raw_data = load_dataset([str(dataset_path)])
    cleaned_data = select_features(prepare_labels(raw_data))
    clients = partition_non_iid(cleaned_data, n_clients=4)

    for client_id, client_df in enumerate(clients, start=1):
        benign_count = int((client_df["Class"] == 0).sum())
        malware_count = int((client_df["Class"] == 1).sum())
        print(
            f"Client {client_id}: total={len(client_df)}, "
            f"benign={benign_count}, malware={malware_count}"
        )

    plot_client_distribution(clients, "Class", str(plot_path))
    print(f"Saved client distribution plot to {plot_path}")
