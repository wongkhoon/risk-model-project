"""
feature_engineering.py

Build a model-ready feature table from simulated raw batch data.

Design goals
------------
This module mirrors the historical logic used to create `clean_df.parquet`
closely enough to support a credible production-style inference simulation.

Key behaviors
-------------
- Renames underwriting columns using the original prefix mapping
- Aggregates ACH/payment data to loan level
- Merges underwriting + loan + payment-derived features
- Keeps only rows matched across all three sources
- Drops redundant columns previously removed in historical preprocessing
- Aligns output to the frozen training feature schema
- Preserves identifying columns (anon_ssn, loanId, loanStatus,
  applicationDate) separately for prediction traceability

Important note
--------------
This is a production-style reconstruction intended for portfolio use.
It focuses on the essential historical logic that defines the modeling
population and feature structure.
"""

from __future__ import annotations

import io
import json
from functools import reduce

import numpy as np
import pandas as pd

from src.config import (
    BATCH_KEY_COLUMN,
    GROUP_COLUMN,
    PRODUCTION_PARAMS_PATH,
    TARGET_COLUMN,
)

# Import the centralized logging configuration shared by all entry points
from src.logging_config import setup_logging

# Create a named logger for this script.
logger = setup_logging("feature_engineering")

# ---------------------------------------------------------------------------
# Identifying columns to preserve for prediction traceability.
#
# These columns come from different source datasets:
#   - From loan data:         anon_ssn, loanId, applicationDate,
#                             clarityFraudId, originatedDate, loanStatus
#   - From underwriting data: underwritingid, clearfraudscore
#
# Linkage keys:
#   - underwritingid (underwriting) = clarityFraudId (loan)
#   - loanId is present in both loan and ACH payment datasets
#
# These columns are dropped during feature engineering but are returned
# separately so every prediction can be traced back to its source record.
# ---------------------------------------------------------------------------
ID_COLUMNS = [
    "anon_ssn",
    "loanId",
    "applicationDate",
    "underwritingid",
    "clarityFraudId",
    "originatedDate",
    "clearfraudscore",
    "loanStatus",
]
# ID_COLUMNS = ["anon_ssn", "loanId", "loanStatus", "applicationDate"]

# ---------------------------------------------------------------------------
# Historical prefix mapping for underwriting column names.
# ---------------------------------------------------------------------------
PREFIX_MAP = {
    ".underwritingdataclarity.clearfraud.clearfraudinquiry.": "cfinq.",
    ".underwritingdataclarity.clearfraud.clearfraudindicator.": "cfind.",
    ".underwritingdataclarity.clearfraud.clearfraudidentityverification.": "cfindvrfy.",
}


def _load_expected_features() -> list[str]:
    """Load frozen feature schema from production configuration."""
    if not PRODUCTION_PARAMS_PATH.exists():
        raise FileNotFoundError(
            f"Production params not found at {PRODUCTION_PARAMS_PATH}. "
            "Run prepare_production_params.py first."
        )

    with PRODUCTION_PARAMS_PATH.open("r", encoding="utf-8") as file_obj:
        config = json.load(file_obj)

    return config["feature_columns"]


def _strip_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace from object/string columns to improve key consistency."""
    df = df.copy()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].map(lambda value: value.strip() if isinstance(value, str) else value)
    return df


def _rename_underwriting_columns(cuv_df: pd.DataFrame) -> pd.DataFrame:
    """Apply the historical underwriting prefix shortening logic."""
    cuv_df = cuv_df.copy()
    cuv_df.rename(
        columns=lambda col: next(
            (
                col.replace(original, new, 1)
                for original, new in PREFIX_MAP.items()
                if col.startswith(original)
            ),
            col,
        ),
        inplace=True,
    )
    return cuv_df


def _is_bool_nan_col(col: pd.Series) -> bool:
    """Check whether a column contains only True/False/NA values."""
    uniq_vals = set(col.dropna().unique())
    return uniq_vals <= {True, False}


def _aggregate_payments(payment_df: pd.DataFrame) -> pd.DataFrame:
    """
    Recreate the core historical loan-level payment feature engineering logic.

    Returns
    -------
    pd.DataFrame
        Loan-level payment aggregate features keyed by loanId.
    """
    payment_df = payment_df.copy()

    required_payment_columns = {
        BATCH_KEY_COLUMN,
        "paymentDate",
        "paymentAmount",
        "paymentStatus",
        "principal",
        "fees",
        "isCollection",
        "paymentReturnCode",
    }
    missing_payment_cols = required_payment_columns - set(payment_df.columns)
    if missing_payment_cols:
        raise ValueError(
            f"payment_df missing required columns for aggregation: {sorted(missing_payment_cols)}"
        )

    # Historical logic: treat missing paymentStatus as "None".
    payment_df["paymentStatus_recode"] = payment_df["paymentStatus"].fillna("None")

    # Historical logic: recode collection indicator.
    payment_df["isCollection_recode"] = payment_df["isCollection"].map(
        {True: "custom", False: "non custom"}
    )

    # Historical logic: missing return code treated as visible category.
    payment_df["paymentReturnCode"] = payment_df["paymentReturnCode"].fillna("NaN").astype(str)

    # Loan-specific totals using Checked/Complete statuses.
    sum_df = (
        payment_df[payment_df["paymentStatus"].isin(["Checked", "Complete"])]
        .groupby(BATCH_KEY_COLUMN)[["principal", "fees", "paymentAmount"]]
        .sum()
        .rename(columns=lambda col: f"{col}_tot")
        .reset_index()
    )

    # Status-specific numerical summaries.
    melted_df = payment_df.melt(
        id_vars=[BATCH_KEY_COLUMN, "paymentStatus_recode"],
        value_vars=["principal", "fees", "paymentAmount"],
        var_name="type",
        value_name="amount",
    ).replace({"type": {"paymentAmount": "pymtAmt"}})

    num_agg = melted_df.pivot_table(
        index=BATCH_KEY_COLUMN,
        columns=["type", "paymentStatus_recode"],
        values="amount",
        aggfunc=["sum", "mean", "median", "std", "count", "min", "max"],
        fill_value=0,
    )

    num_agg.columns = [
        "_".join(col).replace("median", "med").replace("count", "cnt").strip()
        for col in num_agg.columns.values
    ]
    num_agg.reset_index(inplace=True)

    # Days between payments.
    payment_df.sort_values(by=[BATCH_KEY_COLUMN, "paymentDate"], inplace=True)
    payment_df["days_btw_pymts"] = (
        payment_df.groupby(BATCH_KEY_COLUMN)["paymentDate"].diff().dt.days
    )
    payment_df["days_btw_pymts"] = payment_df["days_btw_pymts"].fillna(0)

    days_btw_pymts = (
        payment_df.groupby(BATCH_KEY_COLUMN)["days_btw_pymts"]
        .agg(
            sum_days_btw_pymts="sum",
            mean_days_btw_pymts="mean",
            med_days_btw_pymts="median",
            std_days_btw_pymts="std",
            cnt_days_btw_pymts="count",
            min_days_btw_pymts="min",
            max_days_btw_pymts="max",
        )
        .reset_index()
    )

    # Categorical counts by loanId.
    cat_features = ["isCollection_recode", "paymentStatus_recode", "paymentReturnCode"]
    cat_count_dfs: list[pd.DataFrame] = []

    for feat in cat_features:
        cat_counts = payment_df.groupby(BATCH_KEY_COLUMN)[feat].value_counts().unstack(fill_value=0)

        if feat == "isCollection_recode":
            cat_counts.columns = [f"cnt_{col}" for col in cat_counts.columns]
        elif feat == "paymentStatus_recode":
            cat_counts.columns = [f"cnt_pymtStatus_{col}" for col in cat_counts.columns]
        elif feat == "paymentReturnCode":
            cat_counts.columns = [f"cnt_pymtRCode_{col}" for col in cat_counts.columns]
            cat_counts = cat_counts.reindex(payment_df[BATCH_KEY_COLUMN].unique(), fill_value=0)

        cat_count_dfs.append(cat_counts)

    cat_agg = pd.concat(cat_count_dfs, axis=1).reset_index()

    # First payment with paymentAmount > 0.
    positive_payment_df = payment_df[payment_df["paymentAmount"] > 0]
    earliest_df = positive_payment_df.loc[
        positive_payment_df.groupby(BATCH_KEY_COLUMN)["paymentDate"].idxmin(),
        [BATCH_KEY_COLUMN, "paymentDate", "paymentAmount", "paymentStatus_recode"],
    ].rename(
        columns={
            "paymentDate": "fpymtDate",
            "paymentAmount": "fpymtAmt",
            "paymentStatus_recode": "fpymtStatus",
        }
    )

    # Merge all payment feature tables.
    dfs_to_merge = [sum_df, days_btw_pymts, num_agg, cat_agg]
    merged_payment = reduce(
        lambda left, right: pd.merge(left, right, on=BATCH_KEY_COLUMN, how="inner"),
        dfs_to_merge,
    )

    agg_payment_df = pd.merge(
        merged_payment,
        earliest_df,
        on=BATCH_KEY_COLUMN,
        how="outer",
    )

    return agg_payment_df


def _apply_historical_type_logic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the most important historical dtype normalization rules.

    This improves consistency with the modeling table while keeping the code
    right-sized for a portfolio project.
    """
    df = df.copy()

    # Convert object columns that are logically boolean to nullable boolean.
    bool_obj_cols = [
        col for col in df.select_dtypes(include=["object"]).columns if _is_bool_nan_col(df[col])
    ]
    for col in bool_obj_cols:
        df[col] = df[col].astype("boolean")

    # Historical conversion for selected binary fields.
    for col in ["isFunded", "hasCF"]:
        if col in df.columns:
            df[col] = df[col].astype("boolean")

    int32_candidates = [
        "cfinq.thirtydaysago",
        "cfinq.twentyfourhoursago",
        "cfinq.oneminuteago",
        "cfinq.onehourago",
        "cfinq.ninetydaysago",
        "cfinq.sevendaysago",
        "cfinq.tenminutesago",
        "cfinq.fifteendaysago",
        "cfinq.threesixtyfivedaysago",
        "cfind.totalnumberoffraudindicators",
        "cfind.maxnumberofssnswithanybankaccount",
        "nPaidOff",
    ] + [col for col in df.columns if col.startswith("cnt_")]

    for col in int32_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int32")

    # Historical specific conversion.
    if "cfindvrfy.overallmatchreasoncode" in df.columns:
        df["cfindvrfy.overallmatchreasoncode"] = pd.to_numeric(
            df["cfindvrfy.overallmatchreasoncode"],
            errors="coerce",
        ).astype("Int32")

    # Convert selected object/code columns to categorical, excluding identifiers.
    categorical_candidates = [col for col in df.select_dtypes(include=["object"]).columns] + [
        col for col in df.select_dtypes(exclude=["object"]).columns if col.endswith("code")
    ]

    excluded_ids = {"underwritingid", "loanId", "anon_ssn", "clarityFraudId"}
    for col in categorical_candidates:
        if col not in excluded_ids and col in df.columns:
            df[col] = df[col].astype("category")

    return df


def build_features(
    loan_df: pd.DataFrame,
    ach_df: pd.DataFrame,
    underwriting_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a model-ready feature dataframe from simulated raw batch data.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        - First: model-ready features aligned to the frozen schema.
        - Second: identifying columns (anon_ssn, loanId, loanStatus,
          applicationDate) for prediction traceability. Not used as
          model inputs.

    Important
    ---------
    This function keeps only rows matched across all three datasets:
    - underwriting
    - loan
    - payment/ACH-derived aggregates

    This is intended to match how the historical clean_df.parquet
    modeling table was built.
    """

    expected_features = _load_expected_features()

    # Defensive copies and string cleanup.
    loan_df = _strip_object_columns(loan_df.copy())
    ach_df = _strip_object_columns(ach_df.copy())
    underwriting_df = _strip_object_columns(underwriting_df.copy())

    # Parse known date fields where present.
    if "applicationDate" in loan_df.columns:
        loan_df["applicationDate"] = pd.to_datetime(loan_df["applicationDate"], errors="coerce")
    if "originatedDate" in loan_df.columns:
        loan_df["originatedDate"] = pd.to_datetime(loan_df["originatedDate"], errors="coerce")
    if "paymentDate" in ach_df.columns:
        ach_df["paymentDate"] = pd.to_datetime(ach_df["paymentDate"], errors="coerce")

    # Underwriting historical column normalization.
    underwriting_df = _rename_underwriting_columns(underwriting_df)

    required_loan_cols = {BATCH_KEY_COLUMN, "clarityFraudId"}
    required_underwriting_cols = {"underwritingid"}
    required_payment_cols = {
        BATCH_KEY_COLUMN,
        "paymentDate",
        "paymentAmount",
        "paymentStatus",
        "principal",
        "fees",
        "isCollection",
        "paymentReturnCode",
    }

    missing_loan = required_loan_cols - set(loan_df.columns)
    missing_underwriting = required_underwriting_cols - set(underwriting_df.columns)
    missing_payment = required_payment_cols - set(ach_df.columns)

    if missing_loan:
        raise ValueError(f"loan_df missing required columns: {sorted(missing_loan)}")
    if missing_underwriting:
        raise ValueError(
            f"underwriting_df missing required columns: {sorted(missing_underwriting)}"
        )
    if missing_payment:
        raise ValueError(f"ach_df missing required columns: {sorted(missing_payment)}")

    logger.info("Aggregating ACH/payment data to loan level...")
    agg_payment_df = _aggregate_payments(ach_df)

    logger.info("Merging underwriting and loan data...")
    cuv_loan_df = pd.merge(
        underwriting_df,
        loan_df,
        left_on="underwritingid",
        right_on="clarityFraudId",
        how="inner",
    )

    logger.info("Merging matched underwriting/loan with payment aggregates...")
    combined_df = pd.merge(
        cuv_loan_df,
        agg_payment_df,
        on=BATCH_KEY_COLUMN,
        how="inner",
    )

    # -------------------------------------------------------------------
    # Preserve identifying columns from the merged dataset BEFORE any
    # column drops. These are needed so every prediction can be traced
    # back to its source loan record. They are returned separately so
    # they are never accidentally used as model inputs.
    # -------------------------------------------------------------------
    available_ids = [col for col in ID_COLUMNS if col in combined_df.columns]
    if available_ids:
        logger.info("Preserving identifying columns: %s", available_ids)
        id_df = combined_df[available_ids].reset_index(drop=True)
    else:
        logger.warning("No identifying columns found in merged dataset.")
        id_df = pd.DataFrame()

    # Historical redundant column drops.
    redundant_description_cols = [
        "cfindvrfy.phonematchtypedescription",
        "cfindvrfy.ssnnamereasoncodedescription",
        "cfindvrfy.nameaddressreasoncodedescription",
    ]

    redundant_count_cols = (
        [col for col in combined_df.columns if col.startswith("cnt_fees_")]
        + [col for col in combined_df.columns if col.startswith("cnt_principal_")]
        + [col for col in combined_df.columns if col.startswith("cnt_pymtAmt_")]
    )

    cols_to_drop = [
        col
        for col in (redundant_description_cols + redundant_count_cols)
        if col in combined_df.columns
    ]
    feature_df = combined_df.drop(columns=cols_to_drop, errors="ignore")

    # Ensure target/group are not used for inference features if present.
    feature_df = feature_df.drop(
        columns=[col for col in [TARGET_COLUMN, GROUP_COLUMN] if col in feature_df.columns],
        errors="ignore",
    )

    # Apply historical dtype shaping where practical.
    feature_df = _apply_historical_type_logic(feature_df)

    # Strict alignment to frozen training schema for historical parity.
    actual_columns = set(feature_df.columns)
    expected_columns = set(expected_features)

    missing_features = expected_columns - actual_columns
    extra_columns = actual_columns - expected_columns

    buffer = io.StringIO()
    feature_df.info(buf=buffer, verbose=True)
    logger.info("feature_df schema summary:\n%s", buffer.getvalue())

    cols_to_check = [BATCH_KEY_COLUMN, "underwritingid", "clarityFraudId"]
    available_cols = [c for c in cols_to_check if c in feature_df.columns]
    total_rows = len(feature_df)
    missing_counts = feature_df[available_cols].isna().sum()
    rows_any_missing = feature_df[available_cols].isna().any(axis=1).sum()
    logger.info(
        "Missing values per column (total rows: %d):\n%s\n\nRows with any missing: %d (%.2f%%)",
        total_rows,
        missing_counts.to_string(),
        rows_any_missing,
        rows_any_missing / total_rows * 100,
    )

    if missing_features:
        logger.warning(
            "Adding missing expected features with default values: %s",
            sorted(missing_features),
        )
        for col in missing_features:
            feature_df[col] = 0

    if extra_columns:
        logger.info("Dropping extra non-model columns: %s", sorted(extra_columns))

    feature_df = feature_df.reindex(columns=expected_features)

    # Final CatBoost-safe normalization.
    feature_df = feature_df.fillna(np.nan)

    # Handle Pandas Nullable Extension Types (Ints and Booleans)
    ext_cols = feature_df.select_dtypes(
        include=["Int8", "Int16", "Int32", "Int64", "boolean"]
    ).columns.tolist()
    if ext_cols:
        feature_df[ext_cols] = feature_df[ext_cols].astype("float64")

    # Identify categorical features explicitly marked as pandas "category" dtype.
    cat_cols = feature_df.select_dtypes(include=["category"]).columns.tolist()
    if cat_cols:
        logger.info("Categorical columns detected: %s", sorted(cat_cols))

        buffer = io.StringIO()
        feature_df[cat_cols].info(buf=buffer, verbose=True)
        logger.info("Categorical columns schema summary:\n%s", buffer.getvalue())

        logger.info(
            "overallmatchreasoncode value counts (including NaNs):\n%s",
            feature_df["cfindvrfy.overallmatchreasoncode"].value_counts(dropna=False),
        )

    # Convert categorical columns to string for CatBoost compatibility
    for col in cat_cols:
        feature_df[col] = feature_df[col].astype(str)

    return feature_df, id_df
