"""
Freeze the selected Optuna parameters and approved feature schema for production.

This script:
- loads the registered best-trials JSON
- validates that exactly one best trial is present
- loads the governed historical dataset
- freezes the approved feature whitelist
- writes a production config JSON used by training and inference

Execution
---------
python prepare_production_params.py
"""

from __future__ import annotations

import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from src.config import (
    CLEAN_DATA_PATH,
    EXCLUDED_COLUMNS,
    PRODUCTION_PARAMS_PATH,
    REGISTERED_BEST_TRIALS_PATH,
)


def main() -> None:
    """Freeze production parameters and feature schema."""
    if not REGISTERED_BEST_TRIALS_PATH.exists():
        raise FileNotFoundError(
            f"Registered best-trials file not found at {REGISTERED_BEST_TRIALS_PATH}"
        )

    if not CLEAN_DATA_PATH.exists():
        raise FileNotFoundError(f"Registered historical dataset not found at {CLEAN_DATA_PATH}")

    with REGISTERED_BEST_TRIALS_PATH.open("r", encoding="utf-8") as file_obj:
        trials = json.load(file_obj)

    if not isinstance(trials, dict) or not trials:
        raise ValueError(
            "Registered best-trials JSON must be a non-empty dictionary "
            "mapping trial identifiers to parameter dictionaries."
        )

    if len(trials) != 1:
        raise ValueError(
            "Expected exactly one best trial in the registered best-trials JSON, "
            f"but found {len(trials)}. "
            "Please explicitly curate the approved production trial before freezing production params."
        )

    trial_number = next(iter(trials))
    params = trials[trial_number]

    if not isinstance(params, dict) or not params:
        raise ValueError(
            f'Selected trial "{trial_number}" does not contain a valid parameter dictionary.'
        )

    df = pd.read_parquet(CLEAN_DATA_PATH, engine="pyarrow")

    missing_excluded = EXCLUDED_COLUMNS - set(df.columns)
    if missing_excluded:
        raise ValueError(
            f"Expected excluded columns not found in dataset: {sorted(missing_excluded)}"
        )

    feature_columns = sorted(list(set(df.columns) - EXCLUDED_COLUMNS))
    if not feature_columns:
        raise ValueError("No feature columns detected after exclusions.")

    production_config = {
        "frozen_at_malaysia": datetime.now(ZoneInfo("Asia/Kuala_Lumpur")).isoformat(),
        "trial_number": trial_number,
        "parameters": params,
        "feature_columns": feature_columns,
        "excluded_columns": sorted(list(EXCLUDED_COLUMNS)),
    }

    with PRODUCTION_PARAMS_PATH.open("w", encoding="utf-8") as file_obj:
        json.dump(production_config, file_obj, indent=4)

    print(
        f"Production parameters and feature schema frozen successfully at {PRODUCTION_PARAMS_PATH}."
    )


if __name__ == "__main__":
    main()
