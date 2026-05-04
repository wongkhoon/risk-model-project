"""
build_simulated_batch.py

Create a time-based simulated incoming raw batch from the original raw source
data using the most recent application window from the loan dataset.

Why this exists
---------------
This script simulates how a new production batch might arrive. It does not
create model features directly. Instead, it creates raw batch snapshots that
are later passed into the feature engineering pipeline.

Primary time field
------------------
`applicationDate` from the loan raw data is used as the batch boundary because
ACH payment dates may contain preset future dates and are therefore less
trustworthy for defining an incoming batch window.

Execution
---------
python -m src.build_simulated_batch
"""

from __future__ import annotations

import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from src.config import (
    APPLICATION_DATE_COLUMN,
    BATCH_KEY_COLUMN,
    RAW_ACH_DATA_PATH,
    RAW_LOAN_DATA_PATH,
    RAW_UNDERWRITING_DATA_PATH,
    SIM_ACH_BATCH_PATH,
    SIM_LOAN_BATCH_PATH,
    SIM_RAW_BATCH_DIR,
    SIM_UNDERWRITING_BATCH_PATH,
    SIMULATION_LOOKBACK_DAYS,
    SIMULATION_REGISTRY_PATH,
)

# logging.basicConfig(level = logging.INFO, format = "%(asctime)s - %(levelname)s - %(message)s",)
# Import the centralized logging configuration shared by all entry points
from src.logging_config import setup_logging

# Create a named logger for this script.
# "build_simulated_batch" appears in every log line so you can identify the source.
logger = setup_logging("build_simulated_batch")


def _validate_required_columns(df: pd.DataFrame, required_columns: set[str], label: str) -> None:
    """Validate that a DataFrame contains all required columns."""
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"{label} is missing required columns: {sorted(missing)}")


def _append_registry(entry: dict) -> None:
    """Append simulation metadata to the simulation registry JSON file."""
    if SIMULATION_REGISTRY_PATH.exists():
        registry = json.loads(SIMULATION_REGISTRY_PATH.read_text(encoding="utf-8"))
    else:
        registry = []

    registry.append(entry)
    SIMULATION_REGISTRY_PATH.write_text(json.dumps(registry, indent=4), encoding="utf-8")


def main() -> None:
    """Build a time-based simulated raw batch from the original raw data."""
    if not RAW_LOAN_DATA_PATH.exists():
        raise FileNotFoundError(f"Loan raw data not found at {RAW_LOAN_DATA_PATH}")
    if not RAW_ACH_DATA_PATH.exists():
        raise FileNotFoundError(f"ACH raw data not found at {RAW_ACH_DATA_PATH}")
    if not RAW_UNDERWRITING_DATA_PATH.exists():
        raise FileNotFoundError(f"Underwriting raw data not found at {RAW_UNDERWRITING_DATA_PATH}")

    logger.info("Loading raw source datasets...")
    loan_df = pd.read_csv(
        RAW_LOAN_DATA_PATH,
        parse_dates=["applicationDate", "originatedDate"],
        date_format="ISO8601",  # Up to millisecond precision -> yyyy-mm-dd hh:mm:ss.sss
    )
    ach_df = pd.read_csv(
        RAW_ACH_DATA_PATH,
        parse_dates=["paymentDate"],
        date_format="ISO8601",  # Up to millisecond precision -> yyyy-mm-dd hh:mm:ss.sss
    )
    underwriting_df = pd.read_csv(RAW_UNDERWRITING_DATA_PATH, low_memory=False)

    _validate_required_columns(
        loan_df,
        {BATCH_KEY_COLUMN, APPLICATION_DATE_COLUMN, "clarityFraudId"},
        "Loan raw data",
    )
    _validate_required_columns(ach_df, {BATCH_KEY_COLUMN}, "ACH raw data")
    _validate_required_columns(underwriting_df, {"underwritingid"}, "Underwriting raw data")

    logger.info("Parsing application date column...")
    loan_df[APPLICATION_DATE_COLUMN] = pd.to_datetime(
        loan_df[APPLICATION_DATE_COLUMN],
        errors="coerce",
        utc=False,
    )

    null_app_dates = loan_df[APPLICATION_DATE_COLUMN].isna().sum()
    if null_app_dates > 0:
        logger.warning(
            "Dropping %d loan rows with unparseable %s.",
            null_app_dates,
            APPLICATION_DATE_COLUMN,
        )
        loan_df = loan_df.loc[loan_df[APPLICATION_DATE_COLUMN].notna()].copy()

    if loan_df.empty:
        raise ValueError(f'No valid loan rows remain after parsing "{APPLICATION_DATE_COLUMN}".')

    max_date = loan_df[APPLICATION_DATE_COLUMN].max()
    cutoff_date = max_date - pd.Timedelta(days=SIMULATION_LOOKBACK_DAYS)

    logger.info(
        "Selecting simulated batch using %s >= %s",
        APPLICATION_DATE_COLUMN,
        cutoff_date,
    )

    loan_batch_df = loan_df.loc[loan_df[APPLICATION_DATE_COLUMN] >= cutoff_date].copy()

    if loan_batch_df.empty:
        raise ValueError(
            "Simulated batch is empty. Adjust SIMULATION_LOOKBACK_DAYS or inspect raw dates."
        )

    batch_loan_ids = set(loan_batch_df[BATCH_KEY_COLUMN].dropna().unique())
    ach_batch_df = ach_df.loc[ach_df[BATCH_KEY_COLUMN].isin(batch_loan_ids)].copy()

    batch_underwriting_ids = set(loan_batch_df["clarityFraudId"].dropna().unique())
    underwriting_batch_df = underwriting_df.loc[
        underwriting_df["underwritingid"].isin(batch_underwriting_ids)
    ].copy()

    SIM_RAW_BATCH_DIR.mkdir(parents=True, exist_ok=True)
    loan_batch_df.to_csv(SIM_LOAN_BATCH_PATH, index=False)
    ach_batch_df.to_csv(SIM_ACH_BATCH_PATH, index=False)
    underwriting_batch_df.to_csv(SIM_UNDERWRITING_BATCH_PATH, index=False)

    _append_registry(
        {
            "created_at_malaysia": datetime.now(ZoneInfo("Asia/Kuala_Lumpur")).isoformat(),
            "simulation_type": "time_based_batch",
            "primary_date_source": APPLICATION_DATE_COLUMN,
            "lookback_days": SIMULATION_LOOKBACK_DAYS,
            "cutoff_date": cutoff_date.isoformat(),
            "max_application_date": max_date.isoformat(),
            "loan_batch_rows": int(len(loan_batch_df)),
            "ach_batch_rows": int(len(ach_batch_df)),
            "underwriting_batch_rows": int(len(underwriting_batch_df)),
            "loan_batch_path": str(SIM_LOAN_BATCH_PATH),
            "ach_batch_path": str(SIM_ACH_BATCH_PATH),
            "underwriting_batch_path": str(SIM_UNDERWRITING_BATCH_PATH),
        }
    )

    logger.info("Simulated raw batch created successfully.")


if __name__ == "__main__":
    main()
