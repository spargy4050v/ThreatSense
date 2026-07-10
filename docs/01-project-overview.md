# Project overview

## What ThreatSense is

ThreatSense is a research and demonstration prototype for training a shared
binary malware detector from simulated organizational datasets without pooling
their raw training rows during the federated training loop. The model consumes
tabular features extracted from Windows memory-forensics analysis.

The project combines:

- deterministic data loading and label preparation;
- a training-only single-feature predictiveness screen;
- simulated non-identically distributed client partitions;
- a regularized multilayer perceptron;
- class-weighted local training;
- sample-count-weighted Federated Averaging;
- a centralized comparison on an identical held-out split;
- evaluation plots and three supporting studies;
- a Flask dashboard for results and single-row inference.

## What ThreatSense is not

The current prototype is not:

- a ransomware-family classifier;
- a ransomware-versus-benign-only experiment;
- a live endpoint agent or continuous memory monitor;
- a networked multi-organization deployment;
- a cryptographically secure aggregation system;
- a differential-privacy implementation;
- a guarantee that model updates reveal no information;
- an out-of-distribution detector;
- a calibrated probability system using temperature scaling;
- a zero-day guarantee;
- a production antivirus product.

## Dataset scope

`data/Obfuscated-MalMem2022.csv` contains:

| Property | Value |
| --- | ---: |
| Rows | 58,596 |
| Columns | 57 |
| Benign rows | 29,298 |
| Malware rows | 29,298 |
| Coarse ransomware rows | 9,791 |
| Coarse spyware rows | 10,020 |
| Coarse trojan rows | 9,487 |
| Missing values observed | 0 |

The `Category` strings embed type, family, hash, and sample information, for
example `Ransomware-Ako-<hash>-1.raw`. The binary `Class` column contains
`Benign` or `Malware`.

## Binary label convention

`src/preprocessing.py::prepare_labels` applies:

```text
normalized Class == "benign"  -> 0
every other Class value       -> 1
```

The sigmoid model output is therefore **malware probability**, meaning the
estimated probability of class `1`. A benign prediction should have a value
near zero; a malware prediction should have a value near one.

## Primary research question

Can a simulated set of organizations collaboratively train a useful shared
memory-forensics malware detector using parameter exchange, while achieving
performance close to a model trained on centralized data?

## Supporting questions

1. Does the federated model approach the centralized baseline on the same test
   rows?
2. Are results stable across different splits and initializations?
3. How much performance changes when highly predictive features are removed?
4. How many bytes move under the project's simplified federated communication
   assumptions compared with centralizing the raw training split once?
5. How does stronger client heterogeneity look before it is adopted for a new
   training run?

## Current main configuration

| Setting | Value |
| --- | --- |
| Split | Stratified 80% train / 20% test |
| Primary seed | 42 |
| Candidate numeric features | 55 |
| Screening threshold | Mean single-feature CV accuracy >= 0.85 |
| Retained features | 31 |
| Simulated clients | 4 |
| Main partition | `partition_non_iid` (equal quantity, label-ratio skew) |
| Rounds | 8 |
| Local epochs per round | 1 |
| Batch size | 32 |
| Aggregation | Sample-count-weighted FedAvg |
| Decision threshold | 0.5 |
| Centralized epochs | 8 |

## Current model configuration

The model is a 12,417-parameter MLP:

```text
31 inputs
  -> Dense(128, ReLU, L2)
  -> Dropout(0.5)
  -> Dense(64, ReLU, L2)
  -> Dropout(0.5)
  -> Dense(1, sigmoid)
```

It uses Adam with learning rate `0.0005` and binary cross-entropy with label
smoothing `0.05`.

## Current state versus experimental v2 partition

The saved model and headline metrics use the original `partition_non_iid`
function. `partition_non_iid_v2` has been implemented and independently
verified, but `run_federated_training.py` has not been switched to it. The v2
plot must not be presented as the distribution used to produce the current
saved model.
