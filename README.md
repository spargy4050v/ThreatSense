# ThreatSense

ThreatSense is a modular prototype for federated-learning-based ransomware
detection using memory-forensics features.

This initial commit contains only the project structure. The implementation
will be developed and reviewed one module at a time before training begins.

## Planned workflow

1. Load the dataset and create a stratified train/test split.
2. Screen individual features for suspicious predictive shortcuts using
   five-fold logistic-regression validation.
3. Partition training data into non-IID simulated organizations.
4. Train class-weighted local multilayer perceptrons.
5. aggregate client updates with sample-count-weighted FedAvg.
6. Compare the federated model with a centralized baseline.
7. Evaluate both approaches and expose the global model through Flask.

## Repository layout

- `data/`: local datasets, excluded from Git.
- `src/`: preprocessing, partitioning, modeling, training, and evaluation.
- `app/`: Flask inference interface.
- `models/`: generated model artifacts.
- `results/`: verified experiment outputs and visualizations.
- `run_federated_training.py`: future federated-training entry point.

No training is performed by the current scaffold.

