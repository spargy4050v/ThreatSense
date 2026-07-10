# Web API and interface

## Startup behavior

`app/app.py` resolves paths relative to the project root and requires:

- `models/federated_global_model.keras`;
- `models/scaler.joblib`;
- `models/features.json`.

If any are absent, startup raises a clear `FileNotFoundError`. The app does not
silently use an untrained or default model.

At startup it:

1. validates the feature-schema JSON;
2. loads the scaler with joblib;
3. loads the Keras model without compiling it for training;
4. creates the Flask application.

## Routes

### `GET /`

Renders the dashboard with:

- scope boundary;
- five federated and centralized metrics;
- generated evidence plots that exist locally;
- 31 dynamically generated feature inputs;
- random benign/malware buttons;
- prediction result box.

### `GET /results/<filename>`

Serves only these allowlisted generated images:

- `client_distribution_v2.png`;
- `confusion_matrix.png`;
- `federated_vs_baseline.png`;
- `leakage_scores.png`;
- `roc_curve.png`.

Arbitrary filesystem paths and non-allowlisted files return 404.

### `POST /predict`

Request body:

```json
{
  "pslist.nproc": 40,
  "malfind.ninjections": 2
}
```

The request may omit features. Every missing feature becomes `0.0` and creates
a warning. Present values must be numeric and finite. Extra keys are ignored.

Processing order:

1. read JSON object;
2. build values in saved feature-schema order;
3. fill missing values and record warnings;
4. convert to a one-row DataFrame;
5. apply saved scaler;
6. call model prediction;
7. threshold at 0.5;
8. return JSON.

Response:

```json
{
  "label": "Benign",
  "probability": 0.018299,
  "warnings": []
}
```

`probability` always means Malware/class-1 probability. It is not confidence in
whichever label was returned.

### `GET /random/benign`

Loads/caches only the label plus saved feature columns, samples a real benign
row, fills any missing values with zero, and returns a flat feature object.

### `GET /random/malware`

Uses every non-benign row as eligible Malware, consistent with label mapping.

## Status codes

| Code | Meaning in this app |
| --- | --- |
| 200 | Successful page, image, example, or prediction. |
| 400 | Prediction body is not an object or a feature is invalid. |
| 404 | Invalid random label or disallowed/missing result image. |
| 503 | Optional demo dataset is unavailable or incompatible. |

## UI result semantics

The UI deliberately displays:

```text
Prediction: Benign
Malware probability: 1.83% (raw: 0.018299)
```

The label is visually primary. The class-1 probability is supporting detail so
a low benign probability is not misread as low confidence.

## Dashboard truthfulness

The dashboard labels v2 client heterogeneity as validated but not yet used for
the saved model. Main metrics are explicitly tied to the current v1 model.

## Security and deployment notes

The Flask development server is for local demonstration. Production use would
need a production WSGI server, authentication, input size limits, TLS, request
logging policy, dependency hardening, and model/data governance.

The endpoint accepts manually supplied feature-space values; it does not
extract memory features from live systems.
