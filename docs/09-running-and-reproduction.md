# Running and reproduction

All commands below assume PowerShell and the ThreatSense root directory.

## 1. Enter the project

```powershell
cd "C:\Users\sparg\Documents\Federated Learning Based Ransomware Detection\ThreatSense"
```

## 2. Create or activate the environment

Create if needed:

```powershell
python -m venv .venv
```

Activate:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use the interpreter directly in every command:

```powershell
.\.venv\Scripts\python.exe --version
```

## 3. Install dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 4. Place the dataset

Expected path:

```text
data/Obfuscated-MalMem2022.csv
```

Expected current shape is 58,596 rows and 57 columns. The CSV is intentionally
not committed.

## 5. Optional module sanity checks

```powershell
.\.venv\Scripts\python.exe src\preprocessing.py
.\.venv\Scripts\python.exe src\leakage_check.py
.\.venv\Scripts\python.exe src\partition.py
.\.venv\Scripts\python.exe src\model.py
.\.venv\Scripts\python.exe src\client.py
.\.venv\Scripts\python.exe src\server.py
```

Note: running `src/partition.py` exercises v2 and writes the v2 plot. The main
trainer still uses v1.

## 6. Run primary federated training

```powershell
.\.venv\Scripts\python.exe run_federated_training.py
```

Creates or replaces:

- `models/features.json`;
- `models/scaler.joblib`;
- `models/federated_global_model.keras`;
- `results/federated_log.csv`.

This is compute-intensive and runs 32 local training jobs: 4 clients x 8 rounds.

## 7. Run centralized baseline

```powershell
.\.venv\Scripts\python.exe src\baseline.py
```

Requires the saved feature schema/scaler from federated preprocessing. Creates:

- `models/centralized_baseline.keras`;
- `results/baseline_metrics.json`.

## 8. Generate final evaluation plots

```powershell
.\.venv\Scripts\python.exe src\evaluate.py
```

Creates:

- `results/confusion_matrix.png`;
- `results/roc_curve.png`;
- `results/federated_vs_baseline.png`.

## 9. Run supporting studies

Leakage ablation, two complete federated runs:

```powershell
.\.venv\Scripts\python.exe run_ablation_leakage.py
```

Multi-seed stability, three federated and three centralized runs:

```powershell
.\.venv\Scripts\python.exe run_multiseed_eval.py
```

Communication volume, no retraining:

```powershell
.\.venv\Scripts\python.exe run_communication_cost.py
```

## 10. Run the dashboard

```powershell
.\.venv\Scripts\python.exe app\app.py
```

Open:

```text
http://127.0.0.1:5000
```

Stop with `Ctrl+C`.

## Recommended clean reproduction order

```text
install dependencies
  -> add dataset
  -> run_federated_training.py
  -> src/baseline.py
  -> src/evaluate.py
  -> optional studies
  -> app/app.py
```

## Quick verification expectations

- The primary split should have 46,876 training rows and 11,720 test rows.
- Seed 42 should flag 24 of 55 numeric features and retain 31.
- The app should render 31 inputs.
- A real benign example should usually return a low malware probability.
- A real malware example should usually return a high malware probability.
- Exact random sample probabilities vary because `/random` chooses a row each
  time.

## Troubleshooting

### Required artifact missing

Run federated training before the app. Run the baseline before evaluation.

### `ModuleNotFoundError`

Install requirements using the same Python executable that runs the script.

### Scaler compatibility warning

Use the pinned `scikit-learn==1.6.1`, matching the saved scaler version.

### Native Windows GPU warning

TensorFlow >=2.11 generally uses CPU on native Windows. The warning does not
mean CPU training failed.

### Results differ slightly

Confirm dataset contents, dependencies, seed, hardware, and source commit.
TensorFlow numerical kernels can produce small platform-dependent variation.

### App starts slowly

TensorFlow model loading can take several seconds on first startup.

### Result plots absent from the dashboard

Run `src/evaluate.py`, `src/leakage_check.py`, and `src/partition.py` as needed.
The dashboard shows only allowlisted files that exist.
