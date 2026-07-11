# Technical glossary

## Federated learning terms

### Federated learning (FL)

A training arrangement in which clients train locally and share model updates
rather than raw rows. Used in `run_federated_training.py`, `src/client.py`, and
`src/server.py`. In this prototype the clients are simulated in one process.

### Client

One local data owner/partition. ThreatSense uses four DataFrames as simulated
organizations. A client trains one local model per round.

### Server

The logical coordinator that combines client models. `src/server.py` implements
aggregation; it is not an HTTP or physical server.

### Local model

A temporary MLP trained on one client's rows starting from the current global
weights. Created inside `local_train`.

### Global model

The shared MLP after aggregation. Saved as
`models/federated_global_model.keras` and used by the app.

### Federated round

One cycle of global distribution, local training, upload, aggregation, and
evaluation. The primary experiment has eight rounds.

### Federated Averaging (FedAvg)

Element-wise averaging of corresponding client model tensors. ThreatSense uses
sample-count weights in `fed_avg_weighted`.

### Sample-weighted aggregation

A client with `n_k` rows receives weight `n_k / total_rows`. This prevents a
small client from influencing the global model as much as a much larger one.
The current clients are nearly equal in size, while sample weighting keeps the
aggregation correct if client sizes change.

### Model update, weights, parameters, and gradients

- **Parameters/weights** are learned numeric tensors inside Dense layers.
- A **gradient** is a derivative used by the optimizer during training.
- An **update** means the changed model state after local training.

ThreatSense returns complete trained weight lists, not raw gradients. Report
language should say **weights/model parameters**, not gradients.

### Non-IID

Non-independent and identically distributed. Client datasets have different
class mixtures or feature distributions. Implemented by `partition_non_iid`.

### Label skew

Clients have different proportions of class `0` and class `1`. ThreatSense targets
benign ratios of 80%, 60%, 40%, and 20%.

### Privacy-preserving

In the limited FL sense, raw rows are not exchanged during the simulated
training loop. FL alone does not guarantee privacy: weights can leak
information, and this project has no secure aggregation or differential
privacy. Prefer **raw-data-local collaborative training** when precision matters.

### Secure aggregation

A protocol that prevents the coordinator from seeing individual client
updates. Not implemented.

### Differential privacy (DP)

A formal privacy framework that adds calibrated randomness and tracks a privacy
budget. Not implemented.

## Machine-learning terms

### Feature

One input measurement, such as `malfind.ninjections`. The model uses 31.

### Label / target

The value to predict. `Class=0` means Benign; `Class=1` means Malware.

### Binary classification

Prediction between exactly two classes. ThreatSense predicts Benign/Malware.

### Multiclass classification

Prediction among more than two classes. ThreatSense does not currently do this.

### MLP / multilayer perceptron

A feed-forward neural network of Dense layers. Defined in `src/model.py`.

### Dense layer

Every output unit connects to every input from the prior layer. Keras stores a
weight matrix and bias vector for each Dense layer.

### ReLU

Rectified Linear Unit, `max(0, x)`. Used in both hidden layers.

### Sigmoid

Maps a real number into `(0, 1)`. The final sigmoid value is interpreted as
malware probability.

### Decision threshold

The probability boundary used to convert probability to a label. Threshold
`0.5` means values at least 0.5 are Malware.

### Loss function

The quantity minimized during training. ThreatSense uses binary cross-entropy.

### Binary cross-entropy

A probability-sensitive binary classification loss. It penalizes confident
wrong predictions more strongly.

### Label smoothing

Slightly softens hard `0/1` training targets. ThreatSense uses `0.05` inside the
loss to reduce extreme confidence during training. This is not probability
calibration.

### Optimizer / Adam

The algorithm that changes weights from gradients. Adam is used at learning
rate `0.0005`.

### Learning rate

Controls optimizer step size. Too high can destabilize training; too low can
slow convergence.

### Epoch

One pass over a training dataset. Each client performs one epoch per round;
the centralized baseline performs eight epochs.

### Batch

A subset processed before one optimizer update. Batch size is 32.

### Class imbalance

Unequal label counts. The overall dataset is balanced, but clients are skewed.

### Class weighting

Changes loss contribution so minority-class examples matter more. Calculated
per client with `compute_class_weight(class_weight="balanced")`.

### Regularization

Techniques that discourage overfitting. ThreatSense uses L2, dropout, and label
smoothing.

### L2 regularization

Adds a penalty related to squared weight magnitude. Both hidden layers use
coefficient `0.01`.

### Dropout

Randomly disables a fraction of hidden activations during training. Both
dropout layers use rate `0.5`; dropout is inactive during prediction.

### Overfitting

Learning training-specific patterns that fail on unseen data. Held-out testing,
regularization, ablation, and multi-seed runs help assess it but do not prove
cross-environment generalization.

### Generalization

Performance on data not used for fitting. Current evidence is within one
dataset and collection regime.

## Data preparation terms

### Train/test split

Training rows fit the model; test rows evaluate it. ThreatSense uses 80/20.

### Stratified split

Preserves class proportions in train and test. Implemented with
`train_test_split(..., stratify=labels)`.

### Random seed

A fixed starting state for pseudorandom operations. Seed 42 is primary; 7 and
123 are used in stability testing.

### Reproducibility

Ability to repeat a result under documented code, data, dependencies, and
seeds. Hardware/library differences can still cause small numeric changes.

### Imputation

Replacing missing values. ThreatSense fills with training-column means, though
the current CSV summary contains no missing values.

### Standardization / `StandardScaler`

Transforms each feature approximately as `(x - training_mean) / training_std`.
The scaler is fitted only on training rows and saved with joblib.

### Schema

The ordered feature contract in `features.json`. It connects training,
evaluation, and inference.

## Leakage and experimental-design terms

### Data leakage

Information unavailable at genuine prediction time improperly influences model
training or evaluation. Examples include fitting preprocessing on test data,
using the label as a feature, or splitting duplicate captures across sets.

### Predictiveness screen

The implemented one-feature cross-validation audit. It flags strong columns but
cannot determine why they are strong.

### Logistic regression

A linear probabilistic classifier used only for the feature screen, not the
final MLP.

### Cross-validation (CV)

Repeated train/validation folds used to estimate single-feature accuracy. The
screen uses five folds.

### Ablation study

A controlled comparison that removes or changes one component. The leakage
ablation compares all 55 numeric features against the screened 31-feature set.

### Baseline

A comparison system. The centralized MLP is the primary baseline.

### Collection artifact

A pattern caused by the lab, VM, tooling, timing, or data-generation procedure
rather than malware behavior. Broad service/handle counts warrant review.

### Sample standard deviation

Spread estimate using `ddof=1`. Used across the three seeds.

## Evaluation terms

### Confusion matrix

Counts true negatives, false positives, false negatives, and true positives.
The saved federated matrix is `[[5752, 108], [25, 5835]]`.

### True positive (TP)

Malware correctly predicted as Malware: 5,835 in the seed-42 run.

### True negative (TN)

Benign correctly predicted as Benign: 5,752.

### False positive (FP)

Benign incorrectly predicted as Malware: 108.

### False negative (FN)

Malware incorrectly predicted as Benign: 25.

### Accuracy

`(TP + TN) / all rows`. Useful here because the test set is balanced.

### Precision

`TP / (TP + FP)`. Of Malware predictions, how many were correct.

### Recall / sensitivity / true-positive rate

`TP / (TP + FN)`. Of actual Malware rows, how many were found.

### F1 score

Harmonic mean of precision and recall.

### ROC curve

True-positive rate versus false-positive rate across all thresholds.

### ROC-AUC

Area under the ROC curve. Measures ranking quality across thresholds; it is not
the same as accuracy at threshold 0.5.

## Communication terms

### Serialization

Turning objects into bytes. The communication study uses uncompressed pickle
for weights and UTF-8 CSV for raw training rows.

### Pickle

Python object serialization. It provides a reproducible estimate here, not a
production network protocol.

### On-wire size

Actual network bytes including serialization, framing, encryption, and other
overheads. The script estimates payload bytes and explicitly excludes several
production overheads.

### Round trip

Messages sent in both directions. The communication calculation counts one
upload and one download per client per round, but does not model latency.

## Web terms

### Flask

Python web framework used in `app/app.py`.

### Route / endpoint

A URL and HTTP method handled by a function, such as `POST /predict`.

### GET and POST

GET retrieves a page/resource; POST sends prediction input.

### JSON

Text data format used for feature inputs and prediction responses.

### HTTP status code

`200` success, `400` invalid input, `404` missing/disallowed resource, and
`503` unavailable optional demo data.

### OOD detection

Out-of-distribution detection identifies inputs unlike training data. Not
implemented.

### Temperature scaling

A post-training probability calibration technique. Not implemented.
