"""
End-to-end batch inference pipeline for simulated production scoring.

This script:
1. loads simulated raw batch files
2. builds model-ready features
3. saves engineered features
4. preserves identifying columns for prediction traceability
5. generates prediction probabilities
6. saves scored output with identifying columns

Execution
---------
python -m src.inference_pipeline
"""

from __future__ import annotations

import pandas as pd

from src.config import (
    SIM_ACH_BATCH_PATH,
    SIM_FEATURES_PATH,
    SIM_LOAN_BATCH_PATH,
    SIM_PREDICTIONS_PATH,
    SIM_PROCESSED_BATCH_DIR,
    SIM_UNDERWRITING_BATCH_PATH,
)
from src.feature_engineering import build_features

# Import the centralized logging configuration shared by all entry points
from src.logging_config import setup_logging
from src.predict import get_model_instance

# Create a named logger for this script.
logger = setup_logging("inference_pipeline")


def main() -> None:
    """Run the simulated batch inference pipeline."""
    if not SIM_LOAN_BATCH_PATH.exists():
        raise FileNotFoundError(
            f"Simulated loan batch not found at {SIM_LOAN_BATCH_PATH}. "
            "Run python -m src.build_simulated_batch first."
        )
    if not SIM_ACH_BATCH_PATH.exists():
        raise FileNotFoundError(
            f"Simulated ACH batch not found at {SIM_ACH_BATCH_PATH}. "
            "Run python -m src.build_simulated_batch first."
        )
    if not SIM_UNDERWRITING_BATCH_PATH.exists():
        raise FileNotFoundError(
            f"Simulated underwriting batch not found at {SIM_UNDERWRITING_BATCH_PATH}. "
            "Run python -m src.build_simulated_batch first."
        )

    logger.info("Loading simulated raw batch files...")
    loan_batch_df = pd.read_csv(
        SIM_LOAN_BATCH_PATH,
        parse_dates=["applicationDate", "originatedDate"],
        date_format="ISO8601",
    )
    ach_batch_df = pd.read_csv(
        SIM_ACH_BATCH_PATH,
        parse_dates=["paymentDate"],
        date_format="ISO8601",
        low_memory=False,
    )
    underwriting_batch_df = pd.read_csv(
        SIM_UNDERWRITING_BATCH_PATH,
        low_memory=False,
    )

    logger.info("Building model-ready features...")

    # -------------------------------------------------------------------
    # build_features() returns two dataframes:
    # - features_df: the 268 model-ready features (used for prediction)
    # - id_df: identifying columns for traceability:
    #     anon_ssn, loanId, applicationDate, underwritingid,
    #     clarityFraudId, originatedDate, clearfraudscore, loanStatus
    #   These are NOT used for prediction.
    # -------------------------------------------------------------------
    features_df, id_df = build_features(
        loan_df=loan_batch_df,
        ach_df=ach_batch_df,
        underwriting_df=underwriting_batch_df,
    )

    if features_df.empty:
        raise ValueError("Feature engineering produced an empty batch.")

    # -------------------------------------------------------------------
    # Log identifying columns status
    # -------------------------------------------------------------------
    if not id_df.empty:
        logger.info(
            "Identifying columns preserved: %s (%d rows)",
            id_df.columns.tolist(),
            len(id_df),
        )
    else:
        logger.warning("No identifying columns available for traceability.")

    SIM_PROCESSED_BATCH_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------
    # Save the model-ready features (268 columns, no identifying columns).
    # This is the exact input the model expects.
    # -------------------------------------------------------------------
    features_df.to_parquet(SIM_FEATURES_PATH, index=False)
    logger.info("Engineered features saved to %s", SIM_FEATURES_PATH)
    logger.info("Feature batch row count: %d", len(features_df))

    logger.info("Generating predictions...")
    predictions = get_model_instance().predict_batch(features_df)

    # -------------------------------------------------------------------
    # Build the predictions output.
    #
    # The identifying columns (id_df) may share some column names with
    # features_df (e.g., clearfraudscore exists in both). To avoid
    # duplicate column errors when saving to parquet:
    # 1. Remove any overlapping columns from id_df
    # 2. Concatenate the remaining id columns with features
    # 3. Add the risk_probability column
    # 4. Use .copy() to defragment the DataFrame for performance
    #
    # Output column order:
    #   [id_columns, feature_1, ..., feature_268, risk_probability]
    # -------------------------------------------------------------------
    if not id_df.empty:
        # Find columns that exist in BOTH id_df and features_df
        overlap = set(id_df.columns) & set(features_df.columns)
        if overlap:
            logger.info(
                "Removing overlapping columns from id_df to prevent "
                "duplicates (these already exist in features_df): %s",
                sorted(overlap),
            )
            id_df_clean = id_df.drop(columns=list(overlap))
        else:
            id_df_clean = id_df

        # Only concatenate if id_df still has columns after removing overlaps
        if not id_df_clean.empty and len(id_df_clean.columns) > 0:
            predictions_df = pd.concat(
                [id_df_clean.reset_index(drop=True), features_df.reset_index(drop=True)],
                axis=1,
            ).copy()
        else:
            predictions_df = features_df.copy()
    else:
        predictions_df = features_df.copy()

    predictions_df["risk_probability"] = predictions.values
    predictions_df.to_parquet(SIM_PREDICTIONS_PATH, index=False)

    logger.info("Predictions saved to %s", SIM_PREDICTIONS_PATH)
    logger.info("Inference pipeline completed successfully.")


if __name__ == "__main__":
    main()
