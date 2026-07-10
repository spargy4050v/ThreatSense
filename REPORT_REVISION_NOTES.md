# ThreatSense report and poster revisions

## Contribution statement

Use this wording verbatim in the report's contribution statement:

> ThreatSense is our independently implemented prototype combining leakage-aware feature screening, simulated non-IID organizational clients, class-weighted MLP training, sample-weighted FedAvg, centralized comparison, and Flask-based inference on memory-forensics data.

## Scope and limitations

Add this sentence to the report's Scope and Limitations sections and to the
poster:

> Trained as a general obfuscated-malware detector (ransomware, spyware, trojan collapsed to one class); ransomware-specific attribution is future work.

The current binary experiment must not be described as distinguishing
ransomware from benign software. It distinguishes benign records from all
non-benign CIC-MalMem-2022 records combined.

## Closest prior work

Add this paragraph to the literature review or related-work section:

> FEDetect (2025) applies federated learning to CIC-MalMem-2022 for multiclass malware classification; ThreatSense differs in using non-IID client simulation with a benign/malware ratio gradient, a single-feature predictiveness-screening step before training, and sample-weighted FedAvg with a direct centralized baseline comparison on the same split.

Reference: Z. Ciplak, K. Yildiz, and S. Altinkaya, "FEDetect: A Federated
Learning-Based Malware Detection and Classification Using Deep Neural Network
Algorithms," *Arabian Journal for Science and Engineering*, vol. 50, no. 19,
pp. 16107-16134, 2025. <https://doi.org/10.1007/s13369-025-10043-x>

## Verified poster results

Replace the placeholder-results note with the following values from the saved
held-out test results:

| Metric | Federated | Centralized |
| --- | ---: | ---: |
| Accuracy | 98.865% | 99.454% |
| Precision | 98.183% | 99.437% |
| Recall | 99.573% | 99.471% |
| F1 | 98.873% | 99.454% |
| ROC-AUC | 99.844% | 99.979% |

## Multi-seed result

Across seeds 42, 7, and 123, federated accuracy was 98.8965% +/- 0.0542
percentage points and centralized accuracy was 99.4738% +/- 0.0345 percentage
points (sample standard deviation). The centralized mean advantage was about
0.58 percentage points.

## Leakage-screening ablation

The screened 31-feature condition achieved 98.8652% accuracy; retaining all 55
numeric candidates achieved 99.6672%. Do not describe the 24 removed features
as confirmed leakage. They carry predictive signal and require domain and
cross-environment review.

## Communication estimate

The uncompressed full-training federated payload estimate was 3.05 MB versus
14.72 MB for one UTF-8 CSV serialization of the seed-42 training split. The
federated estimate was 20.70% as large under these representation choices. This
is a byte-volume estimate, not proof of security, latency, or formal privacy.

Also replace `Dirichlet partitioning` on the poster with `class-ratio-gradient
partitioning`; the implemented partitioner assigns target benign ratios from
0.8 to 0.2 rather than sampling from a Dirichlet distribution.

## Claims to remove from the existing compiled report

The available PDF describes capabilities that this prototype does not
implement. Remove or rewrite claims of real-time monitoring, continuous
learning, zero-day identification, dynamic client addition/removal, model
compression, network-behavior collection, and ransomware-family-specific
detection. The current system is an offline simulation over tabular
memory-forensics features with a Flask single-record inference demo.
