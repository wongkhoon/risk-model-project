"""
Run prediction drift monitoring on the simulated batch output.

Execution
---------
python run_monitoring.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import BASELINE_PRED_PATH, SIM_PREDICTIONS_PATH
from src.monitoring import check_prediction_drift, save_monitoring_result


def main() -> None:
    """Compare baseline predictions with simulated batch predictions."""
    if not BASELINE_PRED_PATH.exists():
        raise FileNotFoundError("Baseline predictions not found. Run training first.")

    if not SIM_PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            "Simulated predictions not found. Run the inference pipeline first."
        )

    baseline = np.load(BASELINE_PRED_PATH)
    simulated_predictions_df = pd.read_parquet(SIM_PREDICTIONS_PATH, engine="pyarrow")

    if "risk_probability" not in simulated_predictions_df.columns:
        raise ValueError("Predictions file must contain 'risk_probability' column.")

    new_predictions = simulated_predictions_df["risk_probability"].to_numpy()
    result = check_prediction_drift(baseline, new_predictions)
    save_monitoring_result(result)

    print("Monitoring result:", result)


if __name__ == "__main__":
    main()
