"""
Production inference module.

This module:
- lazy-loads the trained CatBoost model on first use
- enforces strict feature schema
- supports batch prediction
- rejects partial feature input

This is an internal model-serving layer that expects model-ready features.
"""

from __future__ import annotations

import pandas as pd
from catboost import CatBoostClassifier

from src.config import MODEL_PATH

# logging.basicConfig(level = logging.INFO, format = "%(asctime)s - %(levelname)s - %(message)s",)
# Import the centralized logging configuration shared by all entry points
from src.logging_config import setup_logging

# Create a named logger for this script.
# "predict" appears in every log line so you can identify the source.
logger = setup_logging("predict")


class RiskModel:
    """Wrapper around CatBoost model for safe, schema-controlled inference."""

    def __init__(self) -> None:
        """Load trained model artifact and capture expected feature names."""
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}")

        self.model = CatBoostClassifier()
        self.model.load_model(str(MODEL_PATH))
        self.expected_features = list(self.model.feature_names_)

        logger.info(
            "Model loaded successfully with %d expected features.",
            len(self.expected_features),
        )

    def _validate_and_align(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate incoming schema and enforce training feature order."""
        incoming_columns = set(df.columns)
        expected_columns = set(self.expected_features)

        missing = expected_columns - incoming_columns
        extra = incoming_columns - expected_columns

        if missing:
            raise ValueError(f"Missing required features: {sorted(missing)}")

        if extra:
            logger.warning("Extra columns detected and ignored: %s", sorted(extra))

        return df[self.expected_features].copy()

    # def predict_batch(self, df: pd.DataFrame) -> pd.Series:
    # """Predict positive-class probabilities for a batch of model-ready records."""
    # if not isinstance(df, pd.DataFrame):
    # raise TypeError("Input must be a pandas DataFrame.")

    # if df.empty:
    # raise ValueError("Input DataFrame is empty.")

    # df_aligned = self._validate_and_align(df)
    # probabilities = self.model.predict_proba(df_aligned)[:, 1]
    # return pd.Series(probabilities, index = df.index, name = "risk_probability")

    def predict_batch(self, df: pd.DataFrame) -> pd.Series:
        df_aligned = df.copy()

        # Align columns to expected training schema
        df_aligned = df_aligned.reindex(columns=self.expected_features)

        # If your model wrapper knows categorical features, use them.
        # Otherwise infer from model metadata if available.
        cat_feature_indices = self.model.get_cat_feature_indices()
        cat_feature_names = [df_aligned.columns[i] for i in cat_feature_indices]

        # logging.info("CatBoost categorical feature names: %s", cat_feature_names)
        logger.info(
            "Categorical features (%d): %s",
            len(cat_feature_names),
            ", ".join(cat_feature_names),
        )

        # Convert categorical columns to string and replace nulls with sentinel
        for col in cat_feature_names:
            df_aligned[col] = df_aligned[col].astype("object")
            df_aligned[col] = df_aligned[col].where(df_aligned[col].notna(), "__MISSING__")
            df_aligned[col] = df_aligned[col].astype(str)

        # Optional debug check
        bad_cat_cols = [col for col in cat_feature_names if df_aligned[col].isna().any()]
        if bad_cat_cols:
            logger.warning("Categorical columns still containing NaN: %s", bad_cat_cols)

        logger.info("Column at feature_idx=32: %s", df_aligned.columns[32])
        logger.info("Dtype at feature_idx=32: %s", df_aligned.dtypes.iloc[32])
        logger.info("Null count at feature_idx=32: %s", df_aligned.iloc[:, 32].isna().sum())
        logger.info("Sample values at feature_idx=32:\n%s", df_aligned.iloc[:10, 32])

        probabilities = self.model.predict_proba(df_aligned)[:, 1]
        return pd.Series(probabilities, index=df.index, name="risk_probability")

    def predict_single(self, features: dict) -> float:
        """Predict positive-class probability for a single fully-specified record."""
        if not isinstance(features, dict):
            raise TypeError("Single-record inference input must be a dictionary.")

        df = pd.DataFrame([features])
        df_aligned = self._validate_and_align(df)
        return float(self.model.predict_proba(df_aligned)[0, 1])


_model_instance: RiskModel | None = None


def get_model_instance() -> RiskModel:
    """
    Return a singleton-like RiskModel instance, loading the model lazily on first use.

    This avoids import-time failures in modules that depend on prediction utilities
    before the model artifact is present.
    """
    global _model_instance

    if _model_instance is None:
        _model_instance = RiskModel()

    return _model_instance
