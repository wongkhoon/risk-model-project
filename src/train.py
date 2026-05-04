"""
train.py

Train the final governed CatBoost risk model using the registered historical
engineered dataset and the frozen production configuration.

Purpose
-------
This script is the controlled training entry point for the project. It ensures
that model training is performed only against:
- the governed historical dataset
- the frozen production feature schema
- the selected approved hyperparameters

Workflow
--------
1. Load frozen production parameters and feature schema
2. Load the governed historical engineered dataset
3. Validate required schema components
4. Enforce exact feature selection and feature ordering
5. Apply group-aware train/test splitting
6. Train the final CatBoost classifier
7. Evaluate holdout performance
8. Save the trained model artifact
9. Save baseline holdout prediction probabilities for monitoring
10. Save training metrics for reproducibility and auditability

Execution
---------
python -m src.train

Notes
-----
- Group-aware splitting uses `anon_ssn` to reduce leakage across train/test.
- The output baseline predictions are the positive-class probabilities from the
  holdout set and are later used for drift monitoring.
- This script assumes `prepare_production_params.py` has already been run.
"""

from __future__ import annotations

import gc
import io
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from src.config import (
    BASELINE_PRED_PATH,
    CLEAN_DATA_PATH,
    EXCLUDED_COLUMNS,
    GROUP_COLUMN,
    MODEL_PATH,
    PRODUCTION_PARAMS_PATH,
    SEED,
    TARGET_COLUMN,
    TRAIN_METRICS_PATH,
)

# logging.basicConfig(level = logging.INFO, format = "%(asctime)s - %(levelname)s - %(message)s",)
# Import the centralized logging configuration shared by all entry points
from src.logging_config import setup_logging

# Create a named logger for this script.
# "train" appears in every log line so you can identify the source.
logger = setup_logging("train")


def _load_production_config() -> dict:
    """
    Load the frozen production configuration.

    Returns
    -------
    dict
        Dictionary containing selected parameters and frozen feature schema.

    Raises
    ------
    FileNotFoundError
        If the production configuration file does not exist.
    ValueError
        If the configuration is malformed or incomplete.
    """
    if not PRODUCTION_PARAMS_PATH.exists():
        raise FileNotFoundError(
            f"Production parameters not found at {PRODUCTION_PARAMS_PATH}. "
            "Run prepare_production_params.py first."
        )

    with PRODUCTION_PARAMS_PATH.open("r", encoding="utf-8") as file_obj:
        config = json.load(file_obj)

    if not isinstance(config, dict):
        raise ValueError("Production configuration must be a dictionary.")

    if "parameters" not in config or "feature_columns" not in config:
        raise ValueError(
            "Production configuration must contain 'parameters' and 'feature_columns'."
        )

    if not isinstance(config["parameters"], dict) or not config["parameters"]:
        raise ValueError("Production 'parameters' must be a non-empty dictionary.")

    if not isinstance(config["feature_columns"], list) or not config["feature_columns"]:
        raise ValueError("Production 'feature_columns' must be a non-empty list.")

    return config


def _load_training_dataset() -> pd.DataFrame:
    """
    Load the governed historical engineered dataset i.e. clean_df.parquet

    Returns
    -------
    pd.DataFrame
        Historical modeling dataset used for final training.

    Raises
    ------
    FileNotFoundError
        If the governed dataset file does not exist.
    """
    if not CLEAN_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Historical training dataset not found at {CLEAN_DATA_PATH}. "
            "Register the governed dataset first."
        )

    return pd.read_parquet(CLEAN_DATA_PATH, engine="pyarrow")


def _validate_dataset_schema(df: pd.DataFrame, expected_features: list[str]) -> None:
    """
    Validate that the training dataset contains the required schema elements.

    Parameters
    ----------
    df : pd.DataFrame
        Historical training dataset.
    expected_features : list[str]
        Frozen approved model feature columns.

    Raises
    ------
    ValueError
        If required excluded columns, target, group column, or expected features
        are missing.
    """
    missing_excluded = EXCLUDED_COLUMNS - set(df.columns)
    if missing_excluded:
        raise ValueError(f"Dataset missing required excluded columns: {sorted(missing_excluded)}")

    required_core_columns = {TARGET_COLUMN, GROUP_COLUMN}
    missing_core = required_core_columns - set(df.columns)
    if missing_core:
        raise ValueError(f"Dataset missing required core columns: {sorted(missing_core)}")

    actual_features = set(df.columns) - EXCLUDED_COLUMNS
    missing_features = set(expected_features) - actual_features
    extra_features = actual_features - set(expected_features)

    if missing_features:
        raise ValueError(f"Missing expected features: {sorted(missing_features)}")

    if extra_features:
        logger.warning("Extra columns detected and ignored: %s", sorted(extra_features))


def _prepare_feature_matrix(
    df: pd.DataFrame, expected_features: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    """
    Prepare the model feature matrix in exact frozen schema order.

    Parameters
    ----------
    df : pd.DataFrame
        Historical training dataset.
    expected_features : list[str]
        Frozen approved feature schema.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        Prepared feature matrix and categorical column names.
    """
    X = df[expected_features].copy()

    # Normalize missing values for CatBoost compatibility where possible.
    # X = X.fillna(np.nan)

    # Handle categoricals safely
    cat_cols = X.select_dtypes(include="category").columns.tolist()
    """
    # Convert missing NaN in Dtype categorical to "nan" as Dtype object
    for col in cat_cols:
        X[col] = X[col].astype(str)
    """
    """
    if cat_cols:
        # Convert categorical columns to string representation for CatBoost.
        X[cat_cols] = X[cat_cols].astype(str)
    """
    # Standardize missing values in categorical features by explicitly introducing a sentinel category ("__MISSING__")
    # and replacing NaNs with it. This ensures categorical dtype integrity, prevents invalid category assignment errors,
    # and maintains compatibility with downstream machine learning models requiring explicit category membership.
    for col in cat_cols:
        X[col] = X[col].cat.add_categories(["__MISSING__"]).fillna("__MISSING__")

    # Handle Pandas Nullable Extension Types (Ints and Booleans)
    # CatBoost/NumPy requires missing values in numeric columns to be floats.
    ext_cols = X.select_dtypes(
        include=["Int8", "Int16", "Int32", "Int64", "boolean"]
    ).columns.tolist()
    if ext_cols:
        # Vectorized cast instantly handles pd.NA -> np.nan conversion i.e. <NA> to NaN
        X[ext_cols] = X[ext_cols].astype("float32")  # float32 will suffice in this case

    return X, cat_cols


def _build_training_params(base_params: dict) -> dict:
    """
    Build the final CatBoost training parameter dictionary.

    Parameters
    ----------
    base_params : dict
        Frozen approved hyperparameters.

    Returns
    -------
    dict
        Final CatBoost training parameter dictionary with production-safe
        overrides applied.
    """
    params = dict(base_params)

    params.update(
        {
            "objective": "Logloss",
            "eval_metric": "Logloss",
            "custom_metric": ["AUC", "PRAUC"],
            "random_seed": SEED,
            "thread_count": -1,
            "verbose": True,
            "allow_writing_files": False,
        }
    )

    return params


def _save_training_metrics(
    trial_number: str,
    params: dict,
    n_train_rows: int,
    n_holdout_rows: int,
    n_features: int,
    n_categorical_features: int,
    holdout_logloss: float,
    holdout_roc_auc: float,
    holdout_pr_auc: float,
) -> None:
    """
    Save final training metrics and metadata to a governed JSON artifact.

    Parameters
    ----------
    trial_number : str
        Approved trial identifier used for final training.
    params : dict
        Final CatBoost parameter dictionary used for training.
    n_train_rows : int
        Number of training rows.
    n_holdout_rows : int
        Number of holdout rows.
    n_features : int
        Number of model features.
    n_categorical_features : int
        Number of categorical model features.
    holdout_logloss : float
        Holdout log loss.
    holdout_roc_auc : float
        Holdout ROC-AUC.
    holdout_pr_auc : float
        Holdout PR-AUC.
    """
    metrics_payload = {
        "trained_at_malaysia": datetime.now(ZoneInfo("Asia/Kuala_Lumpur")).isoformat(),
        "trial_number": trial_number,
        "random_seed": SEED,
        "train_rows": n_train_rows,
        "holdout_rows": n_holdout_rows,
        "feature_count": n_features,
        "categorical_feature_count": n_categorical_features,
        "holdout_metrics": {
            "log_loss": holdout_logloss,
            "roc_auc": holdout_roc_auc,
            "pr_auc": holdout_pr_auc,
        },
        "model_artifact_path": str(MODEL_PATH),
        "baseline_predictions_path": str(BASELINE_PRED_PATH),
        "training_parameters": params,
    }

    with TRAIN_METRICS_PATH.open("w", encoding="utf-8") as file_obj:
        json.dump(metrics_payload, file_obj, indent=4)

    logger.info("Training metrics saved to %s", TRAIN_METRICS_PATH)


def main() -> None:
    """Train the final governed CatBoost model and persist its artifacts."""
    logger.info("Loading frozen production configuration...")
    config = _load_production_config()

    trial_number = str(config.get("trial_number", "unknown"))
    params = _build_training_params(config["parameters"])
    expected_features = config["feature_columns"]

    logger.info("Loading historical engineered dataset...")
    df = _load_training_dataset()

    logger.info("Validating training dataset schema...")
    _validate_dataset_schema(df, expected_features)

    """
    code_cols = [c for c in df.columns if c.endswith("code")]
    if code_cols:
        logging.info("Columns ending with 'code': %s", code_cols)
        logging.info("Dtypes of 'code' columns:\n%s", df[code_cols].dtypes)
    else:
        logging.info("No columns ending with 'code' found.")
    buffer = io.StringIO()
    df[expected_features].info(buf = buffer)
    logging.info("Feature schema summary:\n%s", buffer.getvalue())
    """
    buffer = io.StringIO()
    cat_code_cols = list(
        set(df.select_dtypes(include=["category"]).columns)
        | set(c for c in df.columns if c.endswith("code"))
    )
    df[cat_code_cols].info(buf=buffer)
    logger.info("Columns with category Dtypes and end with 'code':\n%s", buffer.getvalue())

    y = df[TARGET_COLUMN]
    groups = df[GROUP_COLUMN]

    if y.isna().any():
        raise ValueError("Target column contains missing values.")

    if groups.isna().any():
        raise ValueError(f'Group column "{GROUP_COLUMN}" contains missing values.')

    logger.info("Preparing feature matrix...")
    X, cat_cols = _prepare_feature_matrix(df, expected_features)

    logger.info("Performing group-aware train/test split...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    logger.info(
        "Training rows: %d | Holdout rows: %d | Num features: %d | Cat features: %d",
        len(X_train),
        len(X_test),
        X.shape[1],
        len(cat_cols),
    )

    logger.info(
        "Categorical features (%d): %s",
        len(cat_cols),
        ", ".join(cat_cols),
    )

    logger.info("Training final CatBoost model...")
    model = CatBoostClassifier(**params)

    bad_cat_cols = []

    for col in cat_cols:
        series = X_train[col]

        # Check missing values
        na_mask = series.isna()
        if na_mask.any():
            bad_cat_cols.append(col)
            logger.error(
                "Categorical column '%s' contains missing values. Sample bad values: %s",
                col,
                series[na_mask].head(10).tolist(),
            )
            continue

        # Check invalid types for CatBoost categorical features
        invalid_mask = ~series.map(lambda v: isinstance(v, (str, int, np.integer)))
        if invalid_mask.any():
            bad_cat_cols.append(col)
            logger.error(
                "Categorical column '%s' contains invalid non-string/non-integer values. "
                "Sample bad values: %s | Sample types: %s",
                col,
                series[invalid_mask].head(10).tolist(),
                series[invalid_mask].head(10).map(lambda v: type(v).__name__).tolist(),
            )

    if bad_cat_cols:
        raise ValueError(
            f"Invalid categorical columns detected before Pool creation: {sorted(set(bad_cat_cols))}"
        )

    train_pool = Pool(X_train, y_train, cat_features=cat_cols)
    model.fit(train_pool)

    logger.info("Evaluating holdout performance...")
    y_prob = model.predict_proba(X_test)[:, 1]

    holdout_logloss = log_loss(y_test, y_prob)
    holdout_roc_auc = roc_auc_score(y_test, y_prob)
    holdout_pr_auc = average_precision_score(y_test, y_prob)

    logger.info("Holdout LogLoss: %.6f", holdout_logloss)
    logger.info("Holdout ROC-AUC: %.6f", holdout_roc_auc)
    logger.info("Holdout PR-AUC: %.6f", holdout_pr_auc)

    logger.info("Saving trained model to %s", MODEL_PATH)
    model.save_model(str(MODEL_PATH))

    logger.info("Saving baseline holdout predictions to %s", BASELINE_PRED_PATH)
    np.save(BASELINE_PRED_PATH, y_prob.astype(np.float32))

    _save_training_metrics(
        trial_number=trial_number,
        params=params,
        n_train_rows=len(X_train),
        n_holdout_rows=len(X_test),
        n_features=X.shape[1],
        n_categorical_features=len(cat_cols),
        holdout_logloss=holdout_logloss,
        holdout_roc_auc=holdout_roc_auc,
        holdout_pr_auc=holdout_pr_auc,
    )

    gc.collect()
    logger.info("Training complete. Model, baseline, and metrics artifacts saved successfully.")


if __name__ == "__main__":
    main()
