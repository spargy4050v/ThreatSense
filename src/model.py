"""Define the shared ThreatSense neural-network architecture."""

import tensorflow as tf


def create_mlp(input_dim: int) -> tf.keras.Model:
    """Build and compile the regularized binary-classification MLP.

    L2 regularization penalizes large weights, discouraging the network from
    relying too heavily on any single input feature. Dropout randomly zeroes
    neurons during training so the network cannot over-depend on specific
    neuron co-adaptations. Label smoothing slightly softens the binary 0/1
    targets, reducing overconfidence and supporting later probability
    calibration work, including the temperature-scaling step used by the
    inference application.
    """
    if not isinstance(input_dim, int) or isinstance(input_dim, bool):
        raise TypeError("input_dim must be an integer.")
    if input_dim <= 0:
        raise ValueError("input_dim must be greater than zero.")

    inputs = tf.keras.Input(shape=(input_dim,), name="behavioral_features")
    hidden = tf.keras.layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(0.01),
        name="dense_128",
    )(inputs)
    hidden = tf.keras.layers.Dropout(0.5, name="dropout_128")(hidden)
    hidden = tf.keras.layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(0.01),
        name="dense_64",
    )(hidden)
    hidden = tf.keras.layers.Dropout(0.5, name="dropout_64")(hidden)
    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="malware_probability",
    )(hidden)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="threatsense_mlp",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    sanity_model = create_mlp(input_dim=31)
    sanity_model.summary()
    print("Model compiled successfully.")
