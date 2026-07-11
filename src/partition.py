"""Create and visualize heterogeneous client datasets for ThreatSense.

Non-IID partitioning matters because real organizations do not observe
identical benign-traffic and malware mixtures. Giving simulated clients
different class balances stress-tests whether federated averaging can still
converge to a useful shared model under realistic data heterogeneity.
"""

from pathlib import Path
from typing import Callable

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


def _category_group(value: object) -> str:
    """Return the stable category prefix from a CIC-MalMem sample label."""
    text = str(value).strip()
    return text.split("-", maxsplit=1)[0] if text else "Unknown"


def _fit_counts_to_total(
    desired: np.ndarray,
    capacities: np.ndarray,
    total: int,
) -> np.ndarray:
    """Round desired counts while preserving capacities and an exact total."""
    counts = np.clip(np.rint(desired), 0, capacities).astype(int)
    difference = int(total - counts.sum())

    while difference != 0:
        if difference > 0:
            candidates = np.flatnonzero(counts < capacities)
            if not len(candidates):
                raise RuntimeError("No capacity remains for the requested rows.")
            marginal_costs = 2 * (counts[candidates] - desired[candidates]) + 1
            chosen = int(candidates[np.argmin(marginal_costs)])
            counts[chosen] += 1
            difference -= 1
        else:
            candidates = np.flatnonzero(counts > 0)
            if not len(candidates):
                raise RuntimeError("No assigned rows can be removed.")
            marginal_costs = -2 * (counts[candidates] - desired[candidates]) + 1
            chosen = int(candidates[np.argmin(marginal_costs)])
            counts[chosen] -= 1
            difference += 1

    return counts


def _dirichlet_client_sizes(
    total_rows: int,
    n_clients: int,
    minimum_rows: int,
    rng: np.random.Generator,
    score_function: Callable[[np.ndarray], float] | None = None,
) -> np.ndarray:
    """Draw reproducible unequal client sizes with an enforced floor."""
    required_minimum = minimum_rows * n_clients
    if total_rows < required_minimum:
        raise ValueError(
            f"At least {required_minimum} rows are required to give each of "
            f"{n_clients} clients {minimum_rows} rows; found {total_rows}."
        )

    lower_bound = max(minimum_rows / total_rows, 0.4 / n_clients)
    upper_bound = min(1.0, 1.6 / n_clients)
    distributable = total_rows - required_minimum
    best_sizes: np.ndarray | None = None
    best_score = np.inf

    for _ in range(20_000):
        proportions = rng.dirichlet(np.ones(n_clients))
        if (
            proportions.min() < lower_bound
            or proportions.max() > upper_bound
        ):
            continue

        raw_sizes = proportions * distributable
        sizes = np.floor(raw_sizes).astype(int) + minimum_rows
        remainder = int(total_rows - sizes.sum())
        fractional_order = np.argsort(-(raw_sizes - np.floor(raw_sizes)))
        sizes[fractional_order[:remainder]] += 1

        score = score_function(sizes) if score_function else 0.0
        if score < best_score:
            best_sizes = sizes
            best_score = score
            if score == 0.0:
                break

    if best_sizes is None:
        raise RuntimeError("Could not draw a valid Dirichlet client-size vector.")
    return best_sizes


def _allocate_malware_categories(
    malware_counts: np.ndarray,
    category_supplies: np.ndarray,
    dominant_indices: np.ndarray,
) -> np.ndarray:
    """Fit 70/30 category targets to the rows that actually exist."""
    n_clients = len(malware_counts)
    n_categories = len(category_supplies)
    weights = np.full(
        (n_clients, n_categories),
        0.3 / (n_categories - 1) if n_categories > 1 else 0.0,
    )
    weights[np.arange(n_clients), dominant_indices] = 1.0 if n_categories == 1 else 0.7
    targets = malware_counts[:, None] * weights

    allocations = np.zeros_like(targets, dtype=int)
    remaining_rows = malware_counts.astype(int).copy()
    remaining_supplies = category_supplies.astype(int).copy()

    while remaining_rows.sum() > 0:
        valid = (remaining_rows[:, None] > 0) & (remaining_supplies[None, :] > 0)
        if not valid.any():
            raise RuntimeError("Malware category allocation exhausted valid cells.")

        marginal_costs = (
            2 * (allocations - targets) + 1
        ) / np.maximum(malware_counts[:, None], 1)
        marginal_costs[~valid] = np.inf
        client_id, category_id = np.unravel_index(
            np.argmin(marginal_costs),
            marginal_costs.shape,
        )
        allocations[client_id, category_id] += 1
        remaining_rows[client_id] -= 1
        remaining_supplies[category_id] -= 1

    if remaining_supplies.sum() != 0:
        raise RuntimeError("Malware category allocation left rows unassigned.")
    return allocations


def partition_non_iid_v2(
    df: pd.DataFrame,
    n_clients: int = 4,
    label_col: str = "Class",
    category_col: str = "Category",
    random_state: int = 42,
) -> list[pd.DataFrame]:
    """Implemented and validated standalone; not currently wired into the training pipeline — see README future scope.

    Partition rows with label, malware-category, and quantity skew.

    Client sizes come from a seeded Dirichlet(1, ..., 1) draw, with at least
    500 rows per client. Benign target ratios descend from 80% to 20%. Within
    each client's malware allocation, 70% targets its dominant category and
    the remaining 30% is divided evenly among other categories.

    CIC-MalMem's ``Category`` values embed hashes and sample suffixes, such as
    ``Ransomware-Ako-<hash>-1.raw``. The stable leading category is therefore
    used here (Ransomware, Spyware, or Trojan), rather than treating every
    sample identifier as a separate family.

    For four clients and three observed malware categories, the target is:

    ========  ===============  =========================
    Client    Dominant group   Target within malware rows
    ========  ===============  =========================
    1         Ransomware       70%
    2         Spyware          70%
    3         Trojan           70%
    4         Ransomware       70% (cycle wraps)
    ========  ===============  =========================

    Exact percentages can move slightly when global category supplies and
    integer row counts constrain the requested 70/30 composition. Every input
    row is nevertheless assigned exactly once.
    """
    if n_clients < 2:
        raise ValueError("n_clients must be at least 2 for a non-IID partition.")
    missing_columns = [
        column
        for column in (label_col, category_col)
        if column not in df.columns
    ]
    if missing_columns:
        raise KeyError(f"Required columns are missing: {missing_columns}.")
    if df.empty:
        raise ValueError("df must contain at least one row.")
    if not df.index.is_unique:
        raise ValueError("df must have a unique index to verify zero overlap.")
    if df[[label_col, category_col]].isna().any().any():
        raise ValueError("Label and category columns must not contain missing values.")

    observed_labels = set(df[label_col].unique().tolist())
    if not observed_labels.issubset({0, 1}):
        raise ValueError(
            f"Label column {label_col!r} must be binary 0/1; "
            f"found {sorted(observed_labels, key=str)}."
        )
    if observed_labels != {0, 1}:
        raise ValueError("Both benign (0) and malware (1) rows are required.")

    benign_ratios = np.linspace(0.8, 0.2, n_clients)
    total_malware = int((df[label_col] == 1).sum())
    malware_rows = df[df[label_col] == 1].copy()
    malware_rows["_partition_category"] = malware_rows[category_col].map(
        _category_group
    )
    malware_categories = sorted(
        malware_rows["_partition_category"].unique().tolist()
    )
    if not malware_categories:
        raise ValueError("No non-benign categories were found.")

    category_supplies = np.array(
        [
            int((malware_rows["_partition_category"] == category).sum())
            for category in malware_categories
        ],
        dtype=int,
    )
    dominant_indices = np.arange(n_clients) % len(malware_categories)

    target_weights = np.full(
        (n_clients, len(malware_categories)),
        0.3 / (len(malware_categories) - 1)
        if len(malware_categories) > 1
        else 0.0,
    )
    target_weights[
        np.arange(n_clients),
        dominant_indices,
    ] = 0.7 if len(malware_categories) > 1 else 1.0

    def composition_mismatch(candidate_sizes: np.ndarray) -> float:
        """Prefer Dirichlet draws compatible with global category supplies."""
        estimated = candidate_sizes * (1.0 - benign_ratios)
        target_supplies = (estimated[:, None] * target_weights).sum(axis=0)
        return float(np.abs(target_supplies - category_supplies).sum())

    rng = np.random.default_rng(random_state)
    client_sizes = _dirichlet_client_sizes(
        len(df),
        n_clients,
        minimum_rows=500,
        rng=rng,
        score_function=composition_mismatch,
    )
    desired_malware = client_sizes * (1.0 - benign_ratios)
    malware_counts = _fit_counts_to_total(
        desired_malware,
        client_sizes,
        total_malware,
    )
    benign_counts = client_sizes - malware_counts

    benign_rows = df[df[label_col] == 0].sample(
        frac=1.0,
        random_state=random_state,
    )
    malware_allocations = _allocate_malware_categories(
        malware_counts,
        category_supplies,
        dominant_indices,
    )

    category_pools = {
        category: malware_rows[
            malware_rows["_partition_category"] == category
        ].sample(
            frac=1.0,
            random_state=random_state + category_id + 1,
        )
        for category_id, category in enumerate(malware_categories)
    }
    category_pointers = {category: 0 for category in malware_categories}
    benign_pointer = 0
    client_dfs: list[pd.DataFrame] = []

    for client_id in range(n_clients):
        pieces = [
            benign_rows.iloc[
                benign_pointer : benign_pointer + benign_counts[client_id]
            ]
        ]
        benign_pointer += int(benign_counts[client_id])

        for category_id, category in enumerate(malware_categories):
            count = int(malware_allocations[client_id, category_id])
            start = category_pointers[category]
            pieces.append(category_pools[category].iloc[start : start + count])
            category_pointers[category] += count

        client_df = pd.concat(pieces, axis=0).drop(
            columns=["_partition_category"],
            errors="ignore",
        )
        client_dfs.append(
            client_df.sample(
                frac=1.0,
                random_state=random_state + client_id,
            )
        )

    # Conservation guarantees: every source row appears in exactly one client.
    assigned_indices = pd.Index(
        np.concatenate([client_df.index.to_numpy() for client_df in client_dfs])
    )
    assert sum(len(client_df) for client_df in client_dfs) == len(df)
    assert assigned_indices.is_unique
    assert set(assigned_indices) == set(df.index)
    assert all(len(client_df) >= 500 for client_df in client_dfs)

    return client_dfs


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


def plot_client_distribution_v2(
    client_dfs: list[pd.DataFrame],
    category_col: str,
    save_path: str,
) -> None:
    """Plot each client's row count as stable stacked category groups."""
    if not client_dfs:
        raise ValueError("client_dfs must contain at least one client dataframe.")
    if any(category_col not in client_df.columns for client_df in client_dfs):
        raise KeyError(f"Every client dataframe must contain {category_col!r}.")

    grouped_categories = [
        client_df[category_col].map(_category_group)
        for client_df in client_dfs
    ]
    category_names = sorted(
        set().union(*(set(categories.unique()) for categories in grouped_categories))
    )
    client_labels = [
        f"Client {client_id + 1}"
        for client_id in range(len(client_dfs))
    ]

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(10, 6))
    bottoms = np.zeros(len(client_dfs), dtype=int)
    colors = plt.get_cmap("tab10").colors
    for category_id, category in enumerate(category_names):
        counts = np.array(
            [int((groups == category).sum()) for groups in grouped_categories]
        )
        axis.bar(
            client_labels,
            counts,
            bottom=bottoms,
            label=category,
            color=colors[category_id % len(colors)],
        )
        bottoms += counts

    axis.set_title("Multi-Axis Non-IID Distribution Across Simulated Clients")
    axis.set_xlabel("Simulated organization")
    axis.set_ylabel("Number of samples")
    axis.legend(title="Category")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    from preprocessing import load_dataset, prepare_labels

    project_root = Path(__file__).resolve().parents[1]
    dataset_path = project_root / "data" / "Obfuscated-MalMem2022.csv"
    plot_path = project_root / "results" / "client_distribution_v2.png"

    raw_data = load_dataset([str(dataset_path)])
    labeled_data = prepare_labels(raw_data)
    clients = partition_non_iid_v2(labeled_data, n_clients=4)

    for client_id, client_df in enumerate(clients, start=1):
        benign_count = int((client_df["Class"] == 0).sum())
        malware_count = int((client_df["Class"] == 1).sum())
        total_count = len(client_df)
        print(
            f"Client {client_id}: total={total_count:,}, "
            f"benign={benign_count:,} ({benign_count / total_count:.1%}), "
            f"malware={malware_count:,} ({malware_count / total_count:.1%})"
        )
        category_counts = (
            client_df["Category"]
            .map(_category_group)
            .value_counts()
            .sort_index()
        )
        for category, count in category_counts.items():
            malware_share = count / malware_count if category != "Benign" else None
            suffix = (
                f" ({malware_share:.1%} of malware)"
                if malware_share is not None
                else ""
            )
            print(f"  {category}: {count:,}{suffix}")

    plot_client_distribution_v2(clients, "Category", str(plot_path))
    print(f"Saved multi-axis distribution plot to {plot_path}")
