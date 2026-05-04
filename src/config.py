"""
src/config.py

Centralized project configuration and safe path management.

Purpose
-------
This module defines all filesystem paths and lightweight constants used across
the project so that the rest of the codebase does not rely on fragile relative
paths or duplicated hardcoded values.

Design Principles
-----------------
- Use absolute paths derived from the project root.
- Create required directories automatically.
- Keep environment-specific values in one place.
- Make data lineage and artifact locations easy to audit.
- Keep configuration simple and laptop-friendly.

Notes
-----
Update the schema-related constants below to match actual datasets,
especially:
- EXCLUDED_COLUMNS
- BATCH_KEY_COLUMN
- APPLICATION_DATE_COLUMN
"""

from pathlib import Path

# =============================================================================
# Project Root and Core Directories
# =============================================================================

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = PROJECT_ROOT / "data"
MODELS_DIR: Path = PROJECT_ROOT / "models"
METADATA_DIR: Path = PROJECT_ROOT / "metadata"
RAW_DATA_DIR: Path = PROJECT_ROOT / "raw_data"
SIMULATION_DIR: Path = PROJECT_ROOT / "simulation"
SIM_RAW_BATCH_DIR: Path = SIMULATION_DIR / "raw_batch"
SIM_PROCESSED_BATCH_DIR: Path = SIMULATION_DIR / "processed_batch"

for directory in (
    DATA_DIR,
    MODELS_DIR,
    METADATA_DIR,
    RAW_DATA_DIR,
    SIMULATION_DIR,
    SIM_RAW_BATCH_DIR,
    SIM_PROCESSED_BATCH_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Registered Historical Training Dataset
# =============================================================================

CLEAN_DATA_PATH: Path = DATA_DIR / "clean_df.parquet"
DATASET_REGISTRY_PATH: Path = METADATA_DIR / "dataset_registry.json"

# =============================================================================
# Raw Source Data for Simulation
# =============================================================================

RAW_LOAN_DATA_PATH: Path = RAW_DATA_DIR / "loan.csv"
RAW_ACH_DATA_PATH: Path = RAW_DATA_DIR / "payment.csv"
RAW_UNDERWRITING_DATA_PATH: Path = RAW_DATA_DIR / "clarity_underwriting_variables.csv"

# =============================================================================
# Hyperparameter Artifacts and Registries
# =============================================================================

REGISTERED_BEST_TRIALS_PATH: Path = MODELS_DIR / "best_trials_CatBoostClassifier.json"
HYPERPARAM_REGISTRY_PATH: Path = METADATA_DIR / "hyperparameter_registry.json"
PRODUCTION_PARAMS_PATH: Path = MODELS_DIR / "best_params_catboost_v1.json"

# =============================================================================
# Model Artifacts and Registries
# =============================================================================

MODEL_PATH: Path = MODELS_DIR / "risk_model_v1.cbm"
BASELINE_PRED_PATH: Path = MODELS_DIR / "baseline_predictions.npy"
MONITORING_LOG_PATH: Path = MODELS_DIR / "monitoring_log.json"
TRAIN_METRICS_PATH: Path = MODELS_DIR / "train_metrics.json"

MODEL_REGISTRY_PATH: Path = METADATA_DIR / "model_registry.json"
BASELINE_REGISTRY_PATH: Path = METADATA_DIR / "baseline_registry.json"

# =============================================================================
# Simulation Batch Paths
# =============================================================================

SIM_LOAN_BATCH_PATH: Path = SIM_RAW_BATCH_DIR / "loan_data_batch.csv"
SIM_ACH_BATCH_PATH: Path = SIM_RAW_BATCH_DIR / "ach_payment_data_batch.csv"
SIM_UNDERWRITING_BATCH_PATH: Path = SIM_RAW_BATCH_DIR / "underwriting_data_batch.csv"

SIM_FEATURES_PATH: Path = SIM_PROCESSED_BATCH_DIR / "new_features.parquet"
SIM_PREDICTIONS_PATH: Path = SIM_PROCESSED_BATCH_DIR / "predictions.parquet"

SIMULATION_REGISTRY_PATH: Path = METADATA_DIR / "simulation_registry.json"

# =============================================================================
# Common Schema Constants
# =============================================================================

TARGET_COLUMN: str = "target"
GROUP_COLUMN: str = "anon_ssn"

BATCH_KEY_COLUMN: str = "loanId"

# applicationDate is the preferred batch boundary.
APPLICATION_DATE_COLUMN: str = "applicationDate"

# =============================================================================
# Simulation Controls
# =============================================================================

SIMULATION_LOOKBACK_DAYS: int = 30
SEED: int = 42

# =============================================================================
# Schema Governance
# =============================================================================
# 16 excluded columns.

EXCLUDED_COLUMNS: set[str] = {
    TARGET_COLUMN,
    GROUP_COLUMN,
    "underwritingid",
    "loanId",
    "clarityFraudId",
    "applicationDate",
    "originatedDate",
    "loanStatus",
    "fpymtDate",
    "fpymtAmt",
    "fpymtStatus",
    "yr_mth",
    "mth",
    "principal_tot",
    "fees_tot",
    "paymentAmount_tot",
}
