# Experiments and results

## Reading the numbers correctly

All metric values are fractions in saved JSON/CSV files. For example,
`0.988652` means `98.8652%`. Unless stated otherwise, the main figures use the
seed-42 80/20 stratified split and threshold 0.5.

## Primary seed-42 federated run

| Round | Accuracy | Precision | Recall | F1 | ROC-AUC |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.969966 | 0.950589 | 0.991468 | 0.970598 | 0.985319 |
| 2 | 0.981314 | 0.968599 | 0.994881 | 0.981564 | 0.992850 |
| 3 | 0.985666 | 0.975761 | 0.996075 | 0.985813 | 0.994467 |
| 4 | 0.986519 | 0.976437 | 0.997099 | 0.986660 | 0.996278 |
| 5 | 0.987713 | 0.979852 | 0.995904 | 0.987813 | 0.996842 |
| 6 | 0.987969 | 0.980185 | 0.996075 | 0.988066 | 0.997822 |
| 7 | 0.988567 | 0.980853 | 0.996587 | 0.988658 | 0.998175 |
| 8 | 0.988652 | 0.981827 | 0.995734 | 0.988732 | 0.998442 |

Interpretation: performance improves across rounds and largely stabilizes by
rounds 6-8. Recall is consistently higher than precision, meaning the detector
misses few malware rows but produces more benign false alarms than malware
misses.

## Seed-42 centralized comparison

| Metric | Federated | Centralized | Centralized minus federated |
| --- | ---: | ---: | ---: |
| Accuracy | 0.988652 | 0.994539 | 0.005887 |
| Precision | 0.981827 | 0.994371 | 0.012543 |
| Recall | 0.995734 | 0.994710 | -0.001024 |
| F1 | 0.988732 | 0.994540 | 0.005809 |
| ROC-AUC | 0.998442 | 0.999790 | 0.001348 |

The centralized model is better on most metrics; the federated model has
slightly higher recall in this split.

## Confusion matrix

At threshold 0.5:

```text
                 Predicted Benign  Predicted Malware
Actual Benign             5,752                108
Actual Malware               25              5,835
```

This corresponds to 108 false alarms and 25 missed malware rows on 11,720 test
rows.

## Leakage-screening ablation

The comparison reruns a complete eight-round federated pipeline for each
condition with independent scalers:

| Metric | With screening (31) | Without screening (55) |
| --- | ---: | ---: |
| Accuracy | 0.988652 | 0.996672 |
| Precision | 0.981827 | 0.995742 |
| Recall | 0.995734 | 0.997611 |
| F1 | 0.988732 | 0.996675 |
| ROC-AUC | 0.998442 | 0.999634 |

Removing the 24 flagged features cost 0.008020 accuracy, or 0.802 percentage
points. This proves the columns carry predictive signal. It does not prove
whether that signal is desirable behavior, collection artifact, or leakage.

## Multi-seed stability

Seeds 42, 7, and 123 vary the split, feature screen, partition, and model
randomness. The same 24 features were flagged in all three runs.

### Federated

| Metric | Mean | Sample std |
| --- | ---: | ---: |
| Accuracy | 0.988965 | 0.000542 |
| Precision | 0.982379 | 0.000587 |
| Recall | 0.995791 | 0.000599 |
| F1 | 0.989040 | 0.000538 |
| ROC-AUC | 0.998574 | 0.000405 |

### Centralized

| Metric | Mean | Sample std |
| --- | ---: | ---: |
| Accuracy | 0.994738 | 0.000345 |
| Precision | 0.993756 | 0.000680 |
| Recall | 0.995734 | 0.000903 |
| F1 | 0.994744 | 0.000346 |
| ROC-AUC | 0.999710 | 0.000077 |

Report-ready interpretation:

> Across three random seeds, the federated model achieved 98.90% +/- 0.05%
> accuracy, compared with 99.47% +/- 0.03% for centralized training. Federated
> training therefore cost approximately 0.58 percentage points of accuracy on
> average in this experiment, while keeping raw client rows out of the
> aggregation loop.

The variability is small within this dataset. This does not demonstrate
stability across different datasets, time periods, organizations, or collection
tools.

## Communication-cost comparison

| Quantity | Value |
| --- | ---: |
| Model parameters | 12,417 |
| Pickled weight payload | 49,930 bytes / 48.76 KB |
| Payload count | 64 |
| Total federated payload | 3,195,520 bytes / 3.05 MB |
| One raw training-split CSV | 15,439,241 bytes / 14.72 MB |
| Federated / centralized | 20.70% |

The 64 payloads come from `4 clients * 8 rounds * 2 directions`.

This result is representation-dependent. It excludes protocol headers,
encryption, secure aggregation, retransmission, and compression. It measures
byte volume, not security or latency. Raw data is counted once; weights are
counted every round in both directions.

## v2 distribution verification

The independent v2 check produced:

| Client | Rows | Benign share | Dominant malware type |
| --- | ---: | ---: | ---: |
| 1 | 9,854 | 78.8% | Ransomware 68.1% |
| 2 | 22,749 | 59.5% | Spyware 71.9% |
| 3 | 15,246 | 39.2% | Trojan 70.0% |
| 4 | 10,747 | 18.9% | Ransomware 68.1% |

Conservation, unique indices, deterministic reruns, and minimum client size all
passed. No headline model metric currently comes from this partition.

## Generated figures

- `leakage_scores.png`: which individual features crossed 0.85.
- `client_distribution.png`: v1 class skew.
- `client_distribution_v2.png`: experimental v2 quantity/type skew.
- `confusion_matrix.png`: threshold-specific errors.
- `roc_curve.png`: threshold-independent ranking curve.
- `federated_vs_baseline.png`: main metric comparison.
- `leakage_ablation_plot.png`: screening trade-off.
