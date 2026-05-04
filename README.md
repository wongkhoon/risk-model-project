# Risk Model — End-to-End Credit Risk Pipeline

> Production-style ML pipeline that predicts loan default risk using CatBoost,
> with a registry system, inference API, and drift monitoring. Originally a
> take-home assessment, extended into a full reproducible pipeline.

![CI](https://github.com/wongkhoon/risk-model-project/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 🎯 Highlights
- **Reproducible training** with dataset / hyperparameter / model registries (`metadata/*.json`)
- **CatBoost classifier** with hyperparameter trial tracking
- **Inference API** (`app.py`) servicing simulated production batches
- **Drift monitoring** (`monitoring.py` → `monitoring_log.json`) tracking score-distribution stability
- **One-command pipeline** via `.bat` / `.ps1` launchers

---

## 🏗️ Architecture

```
raw_data/  →  feature_engineering.py  →  data/clean_df.parquet
                                              ↓
                          register_dataset.py  → metadata/dataset_registry.json
                                              ↓
                          register_hyperparams.py + train.py  → models/risk_model_v1.cbm
                                              ↓                    ↓
                          inference_pipeline.py            models/train_metrics.json
                                              ↓
                              simulation/predictions.parquet
                                              ↓
                                    monitoring.py  → monitoring_log.json
```

| Layer | Module |
|---|---|
| Configuration | `src/config.py`, `src/logging_config.py` |
| Registries | `src/register_dataset.py`, `register_hyperparams.py`, `register_model.py`, `register_baseline.py` |
| Feature engineering | `src/feature_engineering.py` |
| Training | `src/train.py` |
| Inference | `src/predict.py`, `src/inference_pipeline.py` |
| Simulation | `src/build_simulated_batch.py` |
| Monitoring | `src/monitoring.py` |
| API | `app.py`, `test_api.py` |

---

## 🚀 Quickstart

```bash
# 1. Clone
git clone https://github.com/wongkhoon/risk-model-project.git
cd risk-model-project

# 2. Create environment
python -m venv venv
.\venv\Scripts\Activate.ps1     # Windows
# source venv/bin/activate      # macOS/Linux
pip install -r requirements.txt

# 3. Place raw CSVs into raw_data/
#    (clarity_underwriting_variables.csv, loan.csv, payment.csv)

# 4. Run the full pipeline
.\01_run_pipeline.bat

# 5. Launch the API
.\02_run_api.bat

# 6. Smoke-test the API
.\03_test_api.bat
```

---

## 📊 Results

Performance on the **held-out test set** (never seen during training or hyperparameter tuning):

| Metric | Value |
|---|---|
| ROC-AUC | 0.9725252939828957 |
| Log Loss | 0.19387453663052417 |

(Full report: `models/train_metrics.json`)

> **Note on production metrics:** ROC-AUC / Log Loss require ground-truth labels,
> which only become available once loans mature. Production health is tracked via
> **score-distribution drift** in `monitoring.py` as a leading indicator until
> outcome labels materialize.

---

## 🧪 Testing

```bash
pytest
```

---

## 🔐 Data Privacy
Raw data files are **not included** in this repository.
The pipeline expects three CSVs in `raw_data/` matching the schemas
described in the original take-home brief.

---

## 📄 License
MIT
