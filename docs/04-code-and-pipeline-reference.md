# Code and pipeline reference

This document explains every source module at the function and constant-group
level. It is a maintainable alternative to comments that repeat every Python
statement without explaining design intent.

## `src/preprocessing.py`

### `load_dataset(paths)`

- Accepts ordered candidate CSV paths.
- Tries each existing file in order.
- Returns the first successfully parsed pandas DataFrame.
- Records attempted paths and parse errors.
- Raises `FileNotFoundError` with useful context if none can be loaded.

### `prepare_labels(df, label_col="Class")`

- Copies the input instead of mutating it.
- Verifies the label exists.
- Converts values to strings, trims whitespace, and compares case-insensitively.
- Maps `Benign` to integer `0` and every other value to integer `1`.
- This is the exact reason the model is general malware detection rather than
  ransomware-specific detection.

### `select_features(df, drop_cols=None, label_col="Class")`

- Copies the input.
- Optionally drops requested non-label columns that exist.
- Retains numeric columns plus the label.
- Drops string `Category`, hashes, names, and other nonnumeric metadata.
- Does not scale, impute, split, or screen; those decisions happen later.

### Script block

When invoked directly, it loads the local dataset, prepares labels, selects
numeric columns, and prints shape and class-distribution sanity checks.

## `src/leakage_check.py`

### `screen_leaky_features(X, y, threshold=0.85, cv_folds=5)`

- Validates dimensions, threshold, fold count, and class diversity.
- Trains a separate one-feature logistic-regression classifier for each column.
- Uses five-fold cross-validation by default.
- Scores mean accuracy for each feature.
- Sorts results from highest to lowest.
- Returns columns meeting or exceeding the review threshold.

It is a **predictiveness screen**, not proof of leakage. A high score can be a
legitimate malware indicator, a collection artifact, or true target leakage.

### `plot_leakage_scores(scores, save_path, threshold=0.85)`

Creates the descending feature-score bar chart and threshold line.

### Script block

Runs the screen on the complete locally loaded dataset for a visualization
sanity check. The primary training pipeline correctly screens only training
rows after the split.

## `src/partition.py`

### `partition_non_iid(...)`

The primary training partitioner:

- requires at least two clients and binary `0/1` labels;
- separates benign and malware rows;
- shuffles each class reproducibly;
- gives clients nearly equal total target sizes;
- assigns target benign ratios from 0.8 to 0.2 using `np.linspace`;
- assigns unconsumed rows round-robin;
- returns reproducibly shuffled client DataFrames.

It preserves every row and avoids reusing source slices. Its purpose is label
skew, not quantity skew.

### Plot function

`plot_client_distribution` writes the stacked Benign/Malware count plot used
to inspect the partition.

### Script block

Runs `partition_non_iid` against the real dataset, prints each client's class
counts, then writes `client_distribution.png`.

## `src/model.py`

### `create_mlp(input_dim)`

Validation rejects booleans, non-integers, and nonpositive dimensions.

Architecture:

| Layer | Output | Purpose |
| --- | --- | --- |
| Input | `input_dim` | One value per retained feature. |
| Dense | 128 ReLU units | Learns nonlinear combinations. |
| Dropout | 50% | Reduces co-adaptation during training. |
| Dense | 64 ReLU units | Learns a smaller hidden representation. |
| Dropout | 50% | Additional regularization. |
| Dense | 1 sigmoid unit | Produces malware probability. |

Both hidden Dense layers use L2 coefficient `0.01`. Compilation uses Adam at
`5e-4`, binary cross-entropy with label smoothing `0.05`, and accuracy as the
Keras training metric.

For 31 inputs the parameter calculation is:

```text
(31 * 128 + 128) + (128 * 64 + 64) + (64 * 1 + 1)
= 4,096 + 8,256 + 65
= 12,417 parameters
```

## `src/client.py`

### `local_train(...)`

- converts inputs to arrays and validates shape, row count, dimension, epoch,
  and batch constraints;
- calculates balanced class weights from the client's local labels;
- creates the shared architecture;
- replaces its random initial weights with the supplied global weights;
- trains locally;
- returns only the new weight list.

Class weighting reduces the temptation for an 80/20 client to predict only its
majority class.

## `src/server.py`

### `fed_avg_weighted(weights_list, client_sizes)`

- rejects empty inputs, mismatched client counts, invalid sizes, empty weight
  lists, and incompatible tensor structures;
- accumulates each tensor in float64 for the averaging calculation;
- multiplies each client tensor by its row-count fraction;
- casts the result back to the original tensor dtype;
- returns one list with the same tensor order and shapes.

The direct-execution assertions compare weighted versus unweighted averaging
and verify that a larger client has more influence.

## `run_federated_training.py`

### Configuration constants

`RANDOM_STATE=42`, `TEST_SIZE=0.20`, `LEAKAGE_THRESHOLD=0.85`,
`N_CLIENTS=4`, `ROUNDS=8`, `LOCAL_EPOCHS=1`, `BATCH_SIZE=32`, and
`LABEL_COL="Class"` define the primary experiment.

### `set_reproducible_seeds`

Seeds Python, NumPy, and TensorFlow.

### `compute_metrics`

Gets sigmoid probabilities, thresholds at 0.5, and returns accuracy, precision,
recall, F1, and ROC-AUC.

### `prepare_federated_data`

Performs the full safe preprocessing order, saves the fitted scaler and feature
schema, builds the non-IID clients, and prints the flagged features.

### `run_federated_training`

Creates the global model, runs 8 x 4 local training calls, measures update L2
norms, aggregates, evaluates each round, saves the round log and global model,
and returns final metrics.

## `src/baseline.py`

### `load_comparable_data`

Reloads the dataset, reproduces the seed-42 split, uses saved feature names and
scaler, applies training means, and returns train/test arrays and labels. It
ensures the centralized experiment is directly comparable.

### `train_baseline`

Resets seeds, trains the same MLP for eight epochs on all centralized training
rows, saves metrics/model, and prints the final comparison.

## `src/evaluate.py`

- `_predict_probabilities` validates prediction length and finiteness.
- `plot_confusion_matrix` creates a labeled 2x2 matrix at threshold 0.5.
- `plot_roc_curve` creates the ROC and diagonal random reference.
- `plot_federated_vs_baseline` reads final metrics and plots paired bars.
- `main` loads the saved federated model and identical test split, generates
  all three figures, and prints a clean table.

## Standalone study scripts

### `run_ablation_leakage.py`

Builds screened and unscreened feature conditions independently, including a
fresh scaler and full eight-round federated loop for each. It saves metrics,
flagged names, the accuracy difference, interpretation, and a bar chart.

### `run_multiseed_eval.py`

For seeds 42, 7, and 123 it varies split, feature screen, partition, and model
randomness; trains both federated and centralized models; computes sample mean
and sample standard deviation; and saves the feature selection audit.

### `run_communication_cost.py`

Loads the saved model without retraining, pickles its weight list, multiplies by
64 transmissions, serializes the exact seed-42 training split once as UTF-8
CSV, and reports the byte ratio with explicit limitations.
