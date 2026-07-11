# Limitations and report language

## Claim-safe project description

> ThreatSense is an independently implemented prototype for federated binary
> obfuscated-malware detection using CIC-MalMem-2022 memory-forensics features.
> It simulates non-IID organizational clients, performs class-weighted local
> MLP training and sample-weighted FedAvg, and compares the resulting global
> model with an equivalent centralized baseline on identical held-out splits.

## Scope statement that must appear prominently

> The current label mapping combines ransomware, spyware, and trojan records
> into one Malware class. Results therefore describe general obfuscated-malware
> detection, not ransomware-specific detection or family attribution.

## Supported findings

The current evidence supports these statements:

- Seed-42 federated accuracy is 98.8652% on the held-out split.
- Seed-42 centralized accuracy is 99.4539% on the identical split.
- Across three seeds, federated accuracy is 98.8965% +/- 0.0542 percentage
  points and centralized accuracy is 99.4738% +/- 0.0345 percentage points.
- The centralized model's mean accuracy advantage is about 0.58 percentage
  points in this experiment.
- Removing all 24 highly predictive features reduced federated accuracy by
  0.802 percentage points.
- Under the script's uncompressed serialization assumptions, complete
  federated weight traffic was 20.70% of one raw training-split CSV upload.

## Findings that require careful wording

### "Leakage-aware"

Acceptable: the pipeline screens training features for unusually strong
standalone predictiveness and performs an ablation.

Not supported: all removed features are proven leakage. The ablation shows
they carry useful signal, and the names are forensically plausible.

### "Privacy-preserving"

Acceptable: raw client rows remain outside the aggregation step in the
simulation.

Not supported: formal privacy protection, resistance to model inversion, secure
aggregation, or differential privacy.

### "Ransomware detection"

Acceptable only as broad project motivation or as one malware type contained in
the positive class.

Not supported as the experimental target: the positive class also contains
spyware and trojans.

### "Real-time"

The Flask endpoint returns predictions interactively after receiving a prepared
feature vector. It does not monitor memory or extract features continuously.
Call it **interactive single-record inference**, not real-time endpoint
detection.

### "Communication savings"

The measured 79.30% reduction applies only to the documented uncompressed
pickle-versus-CSV representations and traffic formula. It does not include real
protocol overhead or compression.

## Unsupported claims to remove

Do not claim that ThreatSense currently:

- is the first federated ransomware detector;
- is completely novel as an idea;
- identifies zero-day attacks;
- detects malware families;
- performs live memory acquisition;
- supports dynamic client arrival/removal;
- performs network-traffic analysis;
- uses model compression;
- uses encrypted updates;
- implements secure aggregation or differential privacy;
- defends against poisoning or inference attacks;
- has been deployed across real organizations;
- has been validated outside this dataset;
- guarantees that 98-99% performance will transfer to production.

## Closest prior work wording

> FEDetect (2025) applies federated learning to CIC-MalMem-2022 for malware
> classification. ThreatSense differs in its explicit training-only
> single-feature predictiveness audit, reproducible non-IID simulation,
> sample-weighted FedAvg implementation, identical-split centralized comparison,
> leakage-screening ablation, multi-seed stability study, and transparent
> communication-volume estimate.

Reference: Z. Ciplak, K. Yildiz, and S. Altinkaya, "FEDetect: A Federated
Learning-Based Malware Detection and Classification Using Deep Neural Network
Algorithms," *Arabian Journal for Science and Engineering*, 50(19),
16107-16134, 2025. DOI: 10.1007/s13369-025-10043-x.

## Report-ready results paragraph

> On the seed-42 held-out split, the federated model achieved 98.87% accuracy,
> 98.18% precision, 99.57% recall, 98.87% F1, and 99.84% ROC-AUC. Across seeds
> 42, 7, and 123, federated accuracy was 98.90% +/- 0.05 percentage points,
> compared with 99.47% +/- 0.03 for the centralized baseline. The resulting
> mean accuracy gap of approximately 0.58 percentage points was small and
> consistent within this dataset. This comparison does not establish
> cross-dataset or production generalization.

## Report-ready feature paragraph

> The training-only screen flagged 24 features with standalone five-fold
> accuracy of at least 0.85. An ablation found 98.87% accuracy after removing
> them and 99.67% when retaining them, showing that the flagged set carried
> predictive signal. Because standalone predictiveness cannot distinguish
> legitimate forensic behavior from collection artifact, the flagged names
> were reviewed rather than treated as confirmed leakage. Loader-list
> discrepancies were considered behaviorally plausible, while broad service
> and handle totals were identified as priorities for cross-environment
> validation.

## Report-ready communication paragraph

> The saved 12,417-parameter model produced a 49,930-byte uncompressed pickle
> weight payload. Counting four clients, eight rounds, and upload/download in
> every round yielded 3.05 MB of total federated payload traffic. Serializing
> the actual seed-42 training split once as uncompressed UTF-8 CSV required
> 14.72 MB, so the federated estimate was 20.70% as large. This is a
> representation-specific byte-volume comparison, not a measurement of
> security, privacy guarantees, latency, or production protocol overhead.

## Next experiments that would strengthen the work

1. Test selected flagged feature groups instead of removing all 24 together.
2. Use group-aware splits by capture/sample/family if suitable identifiers are
   available.
3. Validate on a different collection environment or memory-malware dataset.
4. Add FedProx or another heterogeneity-aware baseline.
5. Measure real serialized transport with protocol and security overhead.
6. Add probability calibration and out-of-distribution checks.
7. Add formal privacy mechanisms if privacy guarantees are claimed.
