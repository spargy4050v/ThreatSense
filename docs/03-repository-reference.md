# Repository and artifact reference

## Root files

| Path | Responsibility |
| --- | --- |
| `.gitignore` | Keeps the dataset, trained models, joblib files, virtual environment, caches, and generated PNGs out of normal commits. |
| `README.md` | Short public introduction, scope warning, workflow, and folder map. |
| `REPORT_REVISION_NOTES.md` | Report/poster corrections, prior-work wording, verified single-seed metrics, and claims to remove. |
| `requirements.txt` | Python dependencies. `scikit-learn==1.6.1` is pinned for compatibility with the saved scaler. |
| `run_federated_training.py` | Primary eight-round federated experiment and artifact writer. |
| `run_ablation_leakage.py` | Standalone screening-on versus screening-off federated ablation. |
| `run_multiseed_eval.py` | Standalone three-seed federated and centralized stability study. |
| `run_communication_cost.py` | Standalone byte-volume estimate with no retraining. |

## `src/` package

| Path | Responsibility |
| --- | --- |
| `src/__init__.py` | Marks `src` as a Python package. |
| `src/preprocessing.py` | Dataset fallback loading, binary label mapping, and numeric-column selection. |
| `src/leakage_check.py` | Single-feature logistic-regression cross-validation and score plot. |
| `src/partition.py` | Original label-skew partition, experimental multi-axis v2 partition, conservation checks, and distribution plots. |
| `src/model.py` | Defines and compiles the shared regularized Keras MLP. |
| `src/client.py` | Trains one local client from the current global weights using balanced class weights. |
| `src/server.py` | Validates client weight structures and computes sample-weighted FedAvg. |
| `src/baseline.py` | Rebuilds the same split from saved preprocessing artifacts and trains the centralized model. |
| `src/evaluate.py` | Creates confusion matrix, ROC curve, and federated-versus-centralized chart. |

## `app/`

| Path | Responsibility |
| --- | --- |
| `app/app.py` | Loads inference artifacts at startup, exposes Flask routes, loads metrics, and serves an allowlist of result images. |
| `app/templates/index.html` | Responsive dashboard, plots, metrics, feature form, fetch requests, and label-first result display. |
| `app/static/.gitkeep` | Keeps the otherwise empty static folder in Git. The current UI uses inline CSS/JavaScript. |

## `data/`

| Path | Responsibility |
| --- | --- |
| `data/.gitkeep` | Keeps the data directory in Git. |
| `data/Obfuscated-MalMem2022.csv` | Local 58,596-row dataset. Ignored because it is large and should be sourced separately. |

## `models/`

| Path | Produced by | Consumed by |
| --- | --- | --- |
| `models/.gitkeep` | Repository scaffold | Git only |
| `models/features.json` | Federated preprocessing | baseline, evaluation, Flask app |
| `models/scaler.joblib` | Federated preprocessing | baseline, evaluation, Flask app |
| `models/federated_global_model.keras` | Federated training | evaluation, communication study, Flask app |
| `models/centralized_baseline.keras` | Baseline training | optional inspection/comparison |

The model and scaler artifacts are ignored because they are generated binaries.
`models/features.json` is locally excluded through `.git/info/exclude`.

## `results/`

| Path | Meaning |
| --- | --- |
| `results/federated_log.csv` | One row of test metrics per federated round. |
| `results/baseline_metrics.json` | Final seed-42 centralized metrics. |
| `results/client_distribution.png` | Original binary label distribution across v1 clients. |
| `results/client_distribution_v2.png` | Experimental v2 stacked coarse-category distribution. |
| `results/leakage_scores.png` | Standalone feature accuracy scores and the 0.85 review threshold. |
| `results/confusion_matrix.png` | Seed-42 federated confusion matrix at threshold 0.5. |
| `results/roc_curve.png` | Seed-42 federated ROC curve and AUC. |
| `results/federated_vs_baseline.png` | Five side-by-side seed-42 metrics. |
| `results/leakage_ablation.json` | Full screening ablation numbers and flagged feature list. |
| `results/leakage_ablation_plot.png` | Screening-on/off metric bars. |
| `results/multiseed_results.json` | Three-seed metrics, means, sample standard deviations, and feature audit. |
| `results/communication_cost.json` | Parameter count, payload sizes, ratio, assumptions, and caveats. |

Generated result images are ignored by `.gitignore`; result CSV/JSON files are
locally excluded through `.git/info/exclude`. They remain available to the
Flask dashboard on the machine where experiments were run.

## Directories that should not be documented as project logic

| Directory | Meaning |
| --- | --- |
| `.git/` | Git history and local repository configuration. |
| `.venv/` | Local Python interpreter and installed third-party packages. |
| `__pycache__/` and `src/__pycache__/` | Generated Python bytecode caches. |

These directories are implementation environment, not ThreatSense source.

## Dependency reference

| Package | Use |
| --- | --- |
| TensorFlow/Keras | MLP definition, training, model saving/loading, prediction. |
| scikit-learn | split, scaler, logistic regression, cross-validation, class weights, and metrics. |
| pandas | CSV loading, labeled tables, feature selection, partitions, logs. |
| NumPy | arrays, weight arithmetic, random draws, numeric checks. |
| Flask | dashboard routes and JSON inference API. |
| Matplotlib | all generated plots. |
| joblib | scaler persistence. |

## Git history milestones

The repository history separates major capabilities:

- preprocessing;
- leakage screening;
- non-IID partitioning;
- model architecture;
- local client training;
- FedAvg;
- federated and centralized runs;
- evaluation visualizations;
- app dashboard;
- leakage ablation;
- multi-seed study;
- communication-cost study.
