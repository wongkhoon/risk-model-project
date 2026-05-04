"""
test_api.py

Send a sample batch to the running API server and print the predictions
alongside key identifying fields from each record.

Purpose
-------
Verify that the FastAPI prediction endpoint is working correctly by
sending real engineered features from the processed inference batch.

Usage
-----
1. Start the API server:  double-click 02_run_api.bat
2. In a separate terminal: double-click 03_test_api.bat

Output
------
- Screen: formatted tables and interpretation for human review.
- logs/test_YYYYMMDD_HHMMSS.log: structured log entries for auditability.
- logs/test_YYYYMMDD_HHMMSS_full.txt: complete console transcript.

Notes
-----
- The API server must be running before executing this script.
- This script loads real processed features so the request format
  exactly matches what the model expects.
- Identifying columns are loaded from predictions.parquet for display
  only. They are NOT sent to the API.

Identifying Column Sources
--------------------------
- From loan data:         anon_ssn, loanId, applicationDate,
                          clarityFraudId, originatedDate, loanStatus
- From underwriting data: underwritingid, clearfraudscore
- Linkage: underwritingid = clarityFraudId, loanId links loan and ACH
"""

from __future__ import annotations

import io
import json

import pandas as pd
import requests

# -----------------------------------------------------------------------
# Centralized logging — structured entries go to test_YYYYMMDD_HHMMSS.log
# -----------------------------------------------------------------------
from src.logging_config import setup_logging

logger = setup_logging("test_api")

# -----------------------------------------------------------------------
# API endpoint URL (must match the running uvicorn server)
# -----------------------------------------------------------------------
API_URL = "http://127.0.0.1:8000/predict-batch"

# -----------------------------------------------------------------------
# Identifying columns that are NOT model features.
# These exist in predictions.parquet for traceability only.
#
# Sources:
#   - From loan data:         anon_ssn, loanId, applicationDate,
#                             clarityFraudId, originatedDate, loanStatus
#   - From underwriting data: underwritingid, clearfraudscore
# -----------------------------------------------------------------------
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

# -----------------------------------------------------------------------
# Load the model-ready features (268 columns, no identifying columns)
# -----------------------------------------------------------------------
logger.info("Loading model-ready features from new_features.parquet...")
df_features = pd.read_parquet("simulation/processed_batch/new_features.parquet", engine="pyarrow")
logger.info(
    "Loaded %d rows x %d columns from new_features.parquet.",
    len(df_features),
    len(df_features.columns),
)

# -----------------------------------------------------------------------
# Load the predictions file which includes identifying columns
# -----------------------------------------------------------------------
logger.info("Loading predictions file from predictions.parquet...")
df_predictions = pd.read_parquet("simulation/processed_batch/predictions.parquet", engine="pyarrow")
logger.info(
    "Loaded %d rows x %d columns from predictions.parquet.",
    len(df_predictions),
    len(df_predictions.columns),
)

# -----------------------------------------------------------------------
# Display the full schema of both files
# -----------------------------------------------------------------------
print("=" * 60)
print("FEATURE SCHEMA (new_features.parquet)")
print("=" * 60)
buffer = io.StringIO()
df_features.info(verbose=True, show_counts=True, buf=buffer)
print(buffer.getvalue())

print("=" * 60)
print("PREDICTIONS SCHEMA (predictions.parquet)")
print("=" * 60)
buffer = io.StringIO()
df_predictions.info(verbose=True, show_counts=True, buf=buffer)
print(buffer.getvalue())

# -----------------------------------------------------------------------
# Take a random sample of 5 rows for testing.
# random_state ensures the same 5 rows are selected every time you run
# the script, which makes results reproducible and easier to debug.
# -----------------------------------------------------------------------
sample_idx = df_features.sample(n=5, random_state=42).index
logger.info("Sampled %d records at indices: %s", len(sample_idx), list(sample_idx))

# Features to send to the API (268 model features only)
sample_features = df_features.loc[sample_idx]

# -----------------------------------------------------------------------
# Identifying columns from predictions file (for display only).
# Note: clearfraudscore may overlap with features_df and might not be
# in predictions.parquet. We only display what is actually available.
# -----------------------------------------------------------------------
available_ids = [col for col in ID_COLUMNS if col in df_predictions.columns]
logger.info("Available identifying columns: %s", available_ids)

# Also check if clearfraudscore exists in features (it may have been
# kept as a model feature rather than an ID column)
if "clearfraudscore" not in available_ids and "clearfraudscore" in df_features.columns:
    available_ids.append("clearfraudscore")

sample_ids = (
    df_predictions.loc[sample_idx][[col for col in available_ids if col in df_predictions.columns]]
    if available_ids
    else pd.DataFrame()
)

# If clearfraudscore comes from features instead of predictions, add it
if "clearfraudscore" in df_features.columns and "clearfraudscore" not in sample_ids.columns:
    sample_ids = sample_ids.copy()
    sample_ids["clearfraudscore"] = df_features.loc[sample_idx]["clearfraudscore"].values

# -----------------------------------------------------------------------
# Show the records being sent to the API
# -----------------------------------------------------------------------
print("=" * 60)
print("RECORDS BEING SENT TO THE API")
print("=" * 60)

if not sample_ids.empty:
    print(sample_ids.to_string())
else:
    print("No identifying columns available.")

print("=" * 60)

# -----------------------------------------------------------------------
# Convert features to the list-of-dicts format the API expects.
# Only the 268 model features are sent — no identifying columns.
# -----------------------------------------------------------------------
payload = json.loads(sample_features.to_json(orient="records"))
logger.info(
    "Payload prepared: %d records, %d features each.",
    len(payload),
    len(payload[0]) if payload else 0,
)

# -----------------------------------------------------------------------
# Send the request to the running API server
# -----------------------------------------------------------------------
print(f"\nSending {len(payload)} records to {API_URL}...")
logger.info("Sending POST request to %s with %d records...", API_URL, len(payload))

try:
    response = requests.post(API_URL, json=payload)
    logger.info("Response received: HTTP %d", response.status_code)
except requests.exceptions.ConnectionError as exc:
    logger.error("Connection failed: %s", exc)
    print(f"\nERROR: Could not connect to {API_URL}")
    print("Make sure the API server is running (02_run_api.bat).")
    raise SystemExit(1) from exc

# -----------------------------------------------------------------------
# Print the results alongside identifying information
# -----------------------------------------------------------------------
if response.status_code == 200:
    predictions = response.json()["predictions"]
    logger.info("Received %d predictions successfully.", len(predictions))

    print(f"\nReceived {len(predictions)} predictions:\n")

    # Build a results table: identifying columns + risk probability
    if not sample_ids.empty:
        results = sample_ids.reset_index(drop=True).copy()
    else:
        results = pd.DataFrame()

    results["risk_probability"] = predictions

    print(results.to_string())

    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    for i, pred in enumerate(predictions):
        risk = "HIGH RISK" if pred >= 0.5 else "LOW RISK"
        print(f"  Record {i + 1}: {pred:.6f} ({pred*100:.1f}%) -> {risk}")
        logger.info("Record %d: probability=%.6f (%s)", i + 1, pred, risk)

    logger.info("API test completed successfully.")

else:
    logger.error("API returned HTTP %d: %s", response.status_code, response.text)
    print(f"Error {response.status_code}: {response.text}")
