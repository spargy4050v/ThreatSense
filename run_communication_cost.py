"""Compare federated weight traffic with one-time data centralization."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from run_federated_training import (
    N_CLIENTS,
    RANDOM_STATE,
    ROUNDS,
    TEST_SIZE,
)
from src.preprocessing import load_dataset, prepare_labels


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "federated_global_model.keras"
DATASET_PATH = PROJECT_ROOT / "data" / "Obfuscated-MalMem2022.csv"
RESULT_PATH = PROJECT_ROOT / "results" / "communication_cost.json"
DIRECTIONS_PER_ROUND = 2


def human_bytes(byte_count: int) -> str:
    """Format a byte count using binary KB, MB, and GB units."""
    value = float(byte_count)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024.0 or unit == "GB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    raise AssertionError("Unreachable unit conversion state.")


def serialized_weight_size(model: tf.keras.Model) -> tuple[int, int]:
    """Return parameter count and uncompressed pickle payload size."""
    weights = model.get_weights()
    parameter_count = int(sum(weight.size for weight in weights))
    payload = pickle.dumps(weights, protocol=pickle.HIGHEST_PROTOCOL)
    return parameter_count, len(payload)


def serialized_training_split_size() -> tuple[int, int, int]:
    """Serialize the actual seed-42 training rows as an in-memory CSV."""
    raw_data = load_dataset([str(DATASET_PATH)])
    labeled_data = prepare_labels(raw_data)
    train_indices, _ = train_test_split(
        raw_data.index,
        test_size=TEST_SIZE,
        stratify=labeled_data["Class"],
        random_state=RANDOM_STATE,
    )
    training_rows = raw_data.loc[train_indices]
    csv_payload = training_rows.to_csv(index=False).encode("utf-8")
    return len(csv_payload), len(training_rows), len(raw_data)


def main() -> None:
    """Measure both byte totals and save a transparent comparison."""
    missing = [
        str(path.relative_to(PROJECT_ROOT))
        for path in (MODEL_PATH, DATASET_PATH)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Required artifacts are missing: {missing}.")

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    parameter_count, one_weight_payload_bytes = serialized_weight_size(model)
    centralized_training_bytes, training_rows, total_rows = (
        serialized_training_split_size()
    )

    transmissions = N_CLIENTS * ROUNDS * DIRECTIONS_PER_ROUND
    federated_total_bytes = one_weight_payload_bytes * transmissions
    ratio = federated_total_bytes / centralized_training_bytes
    percentage = ratio * 100.0

    print("Communication-cost assumptions:")
    print(
        "  Weight serialization: pickle.dumps(weights, "
        "protocol=pickle.HIGHEST_PROTOCOL), without compression."
    )
    print(
        f"  Federated traffic: {N_CLIENTS} clients × {ROUNDS} rounds × "
        f"{DIRECTIONS_PER_ROUND} directions = {transmissions} payloads."
    )
    print(
        f"  Centralized traffic: the actual stratified seed-{RANDOM_STATE} "
        f"training split ({training_rows:,}/{total_rows:,} rows) serialized "
        "once as UTF-8 CSV."
    )
    print(f"\nModel parameters: {parameter_count:,}")
    print(
        "One serialized weight payload: "
        f"{human_bytes(one_weight_payload_bytes)} "
        f"({one_weight_payload_bytes:,} bytes)"
    )
    print(
        "Federated full-training traffic: "
        f"{human_bytes(federated_total_bytes)} "
        f"({federated_total_bytes:,} bytes)"
    )
    print(
        "One-time centralized training-data traffic: "
        f"{human_bytes(centralized_training_bytes)} "
        f"({centralized_training_bytes:,} bytes)"
    )
    print(
        f"Ratio: {ratio:.6f} — federated traffic is {percentage:.2f}% "
        "of the one-time centralized training-data volume."
    )

    caveats = [
        "This comparison measures total bytes moved, not network security, "
        "privacy guarantees, latency, bandwidth contention, or round-trip count.",
        "Raw training data is counted as a one-time centralized upload, while "
        "federated weights are counted in both directions during every round.",
        "The weight estimate uses uncompressed Python pickle serialization; "
        "production protocols, compression, encryption, headers, and secure "
        "aggregation would change the on-wire total.",
        "The centralized estimate uses UTF-8 CSV serialization of the exact "
        "training rows; a binary or compressed dataset representation would "
        "change that total.",
    ]
    print("\nWhat this does not prove:")
    for caveat in caveats:
        print(f"  - {caveat}")

    summary = (
        f"Headline: Over the full training process, federated learning moved "
        f"{human_bytes(federated_total_bytes)}, or {percentage:.2f}% of the "
        f"{human_bytes(centralized_training_bytes)} required to centralize the "
        "training split once."
    )
    print(f"\n{summary}")

    output = {
        "parameter_count": parameter_count,
        "serialization_method": (
            "pickle.dumps(model.get_weights(), "
            "protocol=pickle.HIGHEST_PROTOCOL), uncompressed"
        ),
        "one_weight_payload_bytes": one_weight_payload_bytes,
        "federated": {
            "clients": N_CLIENTS,
            "rounds": ROUNDS,
            "directions_per_round": DIRECTIONS_PER_ROUND,
            "payload_transmissions": transmissions,
            "total_bytes": federated_total_bytes,
            "human_readable": human_bytes(federated_total_bytes),
        },
        "centralized": {
            "representation": "UTF-8 CSV, uncompressed",
            "upload_count": 1,
            "training_rows": training_rows,
            "total_dataset_rows": total_rows,
            "training_fraction": training_rows / total_rows,
            "total_bytes": centralized_training_bytes,
            "human_readable": human_bytes(centralized_training_bytes),
        },
        "federated_to_centralized_ratio": ratio,
        "federated_as_percent_of_centralized": percentage,
        "caveats": caveats,
        "summary": summary,
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULT_PATH.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)
    print(f"Saved {RESULT_PATH}")


if __name__ == "__main__":
    main()
