"""
Prediction drift monitoring utilities.
"""

from __future__ import annotations

import json
from datetime import datetime
from zoneinfo import ZoneInfo

from scipy.stats import ks_2samp

from src.config import MONITORING_LOG_PATH


def check_prediction_drift(baseline_predictions, new_predictions) -> dict:
    """Compare baseline and new prediction distributions using a KS test."""
    if len(baseline_predictions) == 0 or len(new_predictions) == 0:
        raise ValueError("Both baseline and new predictions must be non-empty.")

    stat, p_value = ks_2samp(baseline_predictions, new_predictions)

    return {
        "timestamp_malaysia": datetime.now(ZoneInfo("Asia/Kuala_Lumpur")).isoformat(),
        "ks_statistic": float(stat),
        "p_value": float(p_value),
        "drift_detected": bool(p_value < 0.05),
    }


def save_monitoring_result(result: dict) -> None:
    """Append a monitoring result to the monitoring log JSON."""
    if MONITORING_LOG_PATH.exists():
        history = json.loads(MONITORING_LOG_PATH.read_text(encoding="utf-8"))
    else:
        history = []

    history.append(result)
    MONITORING_LOG_PATH.write_text(json.dumps(history, indent=4), encoding="utf-8")
