# ThreatSense documentation handbook

This folder explains the complete ThreatSense prototype: what it does, how data
moves through it, what every source file is responsible for, what the technical
terms mean, how to reproduce every result, and which claims are and are not
supported by the current implementation.

## Read in this order

1. [Project overview](01-project-overview.md) - purpose, scope, objectives, and
   current state.
2. [Architecture and data flow](02-architecture-and-data-flow.md) - the full
   path from CSV rows to the Flask prediction screen.
3. [Repository and artifact reference](03-repository-reference.md) - every
   folder and file in ThreatSense.
4. [Code and pipeline reference](04-code-and-pipeline-reference.md) - every
   Python module, function, route, constant group, and algorithm.
5. [Feature dictionary](05-feature-dictionary.md) - memory-forensics feature
   families, all retained features, all flagged features, and their meaning.
6. [Technical glossary](06-technical-glossary.md) - buzzwords and technical
   terms, with where each one appears in the project.
7. [Experiments and results](07-experiments-and-results.md) - the primary run,
   ablation, multi-seed stability, communication cost, figures, and caveats.
8. [Web API and interface](08-web-api-and-interface.md) - Flask routes, request
   formats, UI behavior, and prediction semantics.
9. [Running and reproduction](09-running-and-reproduction.md) - installation,
   command order, expected outputs, and troubleshooting.
10. [Limitations and report language](10-limitations-and-report-language.md) -
    safe claims, unsupported claims, and report-ready wording.

## The most important scope statement

ThreatSense currently performs **binary obfuscated-malware detection**. It maps
`Benign` to class `0` and every non-benign record to class `1`. Ransomware,
spyware, and trojan records are therefore combined into one Malware class. It
does not currently perform ransomware-only detection or family attribution.

## Current evidence at a glance

- Dataset: CIC-MalMem-2022-derived CSV, 58,596 rows, balanced 50/50.
- Saved model inputs: 31 screened numeric memory-forensics features.
- Federated configuration: four simulated clients, eight rounds, one local
  epoch per round, class-weighted local MLP training, sample-weighted FedAvg.
- Seed-42 federated accuracy: 98.8652%.
- Seed-42 centralized accuracy: 99.4539%.
- Three-seed federated accuracy: 98.8965% +/- 0.0542 percentage points.
- Three-seed centralized accuracy: 99.4738% +/- 0.0345 percentage points.
- Leakage ablation: 98.8652% with screening versus 99.6672% without it.
- Communication estimate: 3.05 MB federated weight traffic versus 14.72 MB for
  one raw training-split CSV upload under the documented serialization choices.

## Documentation convention

Words such as **privacy-preserving**, **leakage**, **family**, and **real-time**
are used carefully. The glossary and limitations document explain where these
terms are shorthand and where stronger claims would require additional work.
