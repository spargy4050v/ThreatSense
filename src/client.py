"""Train a ThreatSense client model from the current global state."""

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

if __package__:
    from .model import create_mlp
else:
    from model import create_mlp


def local_train(
    global_weights: list,
    X_client: np.ndarray,
    y_client: np.ndarray,
    input_dim: int,
    epochs: int = 1,
    batch_size: int = 32,
    verbose: int = 0,
) -> list:
    """Train one client locally and return its updated model weights.

    Every client must begin a federated round from the exact same global
    weights. If clients start from divergent models, their updates no longer
    describe changes from a shared reference point, so averaging those models
    stops being meaningful.

    Class weighting is important because the non-IID partitions can be as
    imbalanced as 80/20. Without it, a client could achieve deceptively high
    local accuracy by favoring its majority class, producing an update that
    looks good locally but is practically poor for shared malware detection.
    """
    X_array = np.asarray(X_client)
    y_array = np.asarray(y_client)

    if X_array.ndim != 2:
        raise ValueError("X_client must be a two-dimensional array.")
    if y_array.ndim != 1:
        raise ValueError("y_client must be a one-dimensional array.")
    if len(X_array) != len(y_array):
        raise ValueError("X_client and y_client must contain the same row count.")
    if len(X_array) == 0:
        raise ValueError("Client training data must not be empty.")
    if X_array.shape[1] != input_dim:
        raise ValueError(
            f"X_client has {X_array.shape[1]} features, expected {input_dim}."
        )
    if epochs < 1:
        raise ValueError("epochs must be at least 1.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    classes = np.unique(y_array)
    class_weight_values = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_array,
    )
    class_weights = {
        class_label: float(weight)
        for class_label, weight in zip(classes, class_weight_values)
    }

    model = create_mlp(input_dim)
    model.set_weights(global_weights)
    model.fit(
        X_array,
        y_array,
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        verbose=verbose,
    )
    return model.get_weights()


if __name__ == "__main__":
    from pathlib import Path

    from sklearn.preprocessing import StandardScaler

    from leakage_check import screen_leaky_features
    from partition import partition_non_iid
    from preprocessing import load_dataset, prepare_labels, select_features

    project_root = Path(__file__).resolve().parents[1]
    dataset_path = project_root / "data" / "Obfuscated-MalMem2022.csv"

    raw_data = load_dataset([str(dataset_path)])
    cleaned_data = select_features(prepare_labels(raw_data))

    all_features = cleaned_data.drop(columns=["Class"])
    all_labels = cleaned_data["Class"]
    suspect_features, _ = screen_leaky_features(
        all_features,
        all_labels,
        threshold=0.85,
    )
    filtered_data = cleaned_data.drop(columns=suspect_features)
    client_dfs = partition_non_iid(filtered_data, n_clients=4)

    first_client = client_dfs[0]
    X_first = first_client.drop(columns=["Class"]).to_numpy(dtype=np.float32)
    y_first = first_client["Class"].to_numpy(dtype=np.int32)
    X_first_scaled = StandardScaler().fit_transform(X_first)

    feature_count = X_first_scaled.shape[1]
    initial_model = create_mlp(feature_count)
    initial_weights = initial_model.get_weights()
    updated_weights = local_train(
        initial_weights,
        X_first_scaled,
        y_first,
        input_dim=feature_count,
        epochs=1,
        batch_size=32,
    )

    initial_shapes = [weight.shape for weight in initial_weights]
    updated_shapes = [weight.shape for weight in updated_weights]
    print(f"Input features: {feature_count}")
    print(f"Initial weight shapes: {initial_shapes}")
    print(f"Updated weight shapes: {updated_shapes}")
    print(f"Shape contract matches: {initial_shapes == updated_shapes}")
