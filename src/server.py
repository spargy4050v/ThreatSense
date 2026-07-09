"""Aggregate compatible ThreatSense client models on the federated server."""

import numpy as np


def fed_avg_weighted(
    weights_list: list[list[np.ndarray]],
    client_sizes: list[int],
) -> list[np.ndarray]:
    """Return the sample-count-weighted average of client model parameters.

    For client ``k`` with ``n_k`` local rows and weights ``W_k``, federated
    averaging computes:

        W_global = Σ (n_k / Σ n_j) * W_k

    The calculation is applied independently to every tensor position while
    preserving the input tensor order, shape, and original dtype.
    """
    if not weights_list:
        raise ValueError("weights_list must contain at least one client model.")
    if len(weights_list) != len(client_sizes):
        raise ValueError(
            "weights_list and client_sizes must describe the same clients."
        )
    if any(
        not isinstance(size, (int, np.integer))
        or isinstance(size, (bool, np.bool_))
        or size <= 0
        for size in client_sizes
    ):
        raise ValueError("Every client size must be a positive integer.")

    tensor_count = len(weights_list[0])
    if tensor_count == 0:
        raise ValueError("Each client model must contain at least one tensor.")
    if any(len(client_weights) != tensor_count for client_weights in weights_list):
        raise ValueError("Every client model must contain the same tensor count.")

    for tensor_index in range(tensor_count):
        expected_shape = np.asarray(weights_list[0][tensor_index]).shape
        for client_index, client_weights in enumerate(weights_list[1:], start=1):
            actual_shape = np.asarray(client_weights[tensor_index]).shape
            if actual_shape != expected_shape:
                raise ValueError(
                    f"Tensor {tensor_index} for client {client_index} has shape "
                    f"{actual_shape}, expected {expected_shape}."
                )

    total_samples = sum(int(size) for size in client_sizes)
    averaged_weights: list[np.ndarray] = []

    for tensors_at_position in zip(*weights_list):
        reference = np.asarray(tensors_at_position[0])
        weighted_tensor = np.zeros(reference.shape, dtype=np.float64)

        for client_tensor, client_size in zip(
            tensors_at_position,
            client_sizes,
        ):
            contribution = np.asarray(client_tensor, dtype=np.float64)
            weighted_tensor += contribution * (client_size / total_samples)

        averaged_weights.append(weighted_tensor.astype(reference.dtype))

    return averaged_weights


if __name__ == "__main__":
    if __package__:
        from .model import create_mlp
    else:
        from model import create_mlp

    first_weights = create_mlp(31).get_weights()
    second_weights = create_mlp(31).get_weights()

    weighted = fed_avg_weighted(
        [first_weights, second_weights],
        client_sizes=[100, 300],
    )
    unweighted = fed_avg_weighted(
        [first_weights, second_weights],
        client_sizes=[1, 1],
    )

    # Example: values 2 and 8 with sizes 100 and 300 average to 6.5, not 5.
    expected_weighted = [
        (first * 0.25 + second * 0.75).astype(first.dtype)
        for first, second in zip(first_weights, second_weights)
    ]
    assert all(
        np.allclose(actual, expected)
        for actual, expected in zip(weighted, expected_weighted)
    )
    assert any(
        not np.allclose(weighted_tensor, unweighted_tensor)
        for weighted_tensor, unweighted_tensor in zip(weighted, unweighted)
    )

    sample_tensor_index = 4
    first_mean = float(first_weights[sample_tensor_index].mean())
    second_mean = float(second_weights[sample_tensor_index].mean())
    weighted_mean = float(weighted[sample_tensor_index].mean())
    midpoint_mean = float(unweighted[sample_tensor_index].mean())

    assert abs(weighted_mean - second_mean) < abs(weighted_mean - first_mean)

    print(f"First client tensor mean:  {first_mean:.8f}")
    print(f"Second client tensor mean: {second_mean:.8f}")
    print(f"Weighted tensor mean:      {weighted_mean:.8f}")
    print(f"Unweighted midpoint mean:  {midpoint_mean:.8f}")
    print("Weighted FedAvg assertions passed.")
