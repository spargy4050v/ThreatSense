# ThreatSense

ThreatSense is a modular prototype for federated-learning-based obfuscated
malware detection using memory-forensics features.

Complete project documentation is available in [`docs/`](docs/README.md).

ThreatSense is our independently implemented prototype combining leakage-aware
feature screening, simulated non-IID organizational clients, class-weighted MLP
training, sample-weighted FedAvg, centralized comparison, and Flask-based
inference on memory-forensics data.

## Scope and limitations

The current binary label mapping collapses ransomware, spyware, trojans, and
other non-benign families into one malware class. The trained model is therefore
a general obfuscated-malware detector; ransomware-specific detection and family
attribution are future work.

The project builds on prior federated-malware research rather than claiming the
idea of federated ransomware detection as novel. In particular, FEDetect (2025)
applies federated learning to CIC-MalMem-2022 for malware classification.
ThreatSense differs in its non-IID client simulation with a benign/malware ratio
gradient, pre-training single-feature predictiveness screen, sample-weighted
FedAvg, and direct centralized comparison on the same split.

## Workflow

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
- `run_federated_training.py`: federated-training entry point.
- `app/app.py`: Flask inference interface and real-row demo endpoints.
- `run_ablation_leakage.py`: leakage-screening ablation.
- `run_multiseed_eval.py`: three-seed stability evaluation.
- `run_communication_cost.py`: byte-volume communication comparison.
- `docs/`: full architecture, code, terminology, results, API, and reproduction
  handbook.
