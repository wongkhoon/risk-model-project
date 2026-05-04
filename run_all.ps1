# ===========================================================================
# run_all.ps1
#
# Master pipeline execution script.
#
# Purpose:
#   Run every step of the governed ML pipeline in sequence and capture
#   ALL output (Python logs + PowerShell verification results) into
#   timestamped log files.
#
# Usage:
#   Called by 01_run_pipeline.bat, or manually:
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\run_all.ps1"
#
# Output:
#   - logs/full_session_YYYYMMDD_HHMMSS.txt : Complete session record
#   - logs/session_YYYYMMDD_HHMMSS.log      : Python-only log (clean)
# ===========================================================================

# -------------------------------------------------------------------
# Set project root and navigate there FIRST (before anything else)
# -------------------------------------------------------------------
$projectRoot = "C:\Users\grace\OneDrive\Documents\Learning\risk_model_project"
Set-Location $projectRoot

# -------------------------------------------------------------------
# Kill any lingering transcript from a crashed previous run
# -------------------------------------------------------------------
try { Stop-Transcript } catch { }

# -------------------------------------------------------------------
# Generate ONE timestamp for the entire run
# -------------------------------------------------------------------
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

# -------------------------------------------------------------------
# Ensure logs folder exists
# -------------------------------------------------------------------
if (-not (Test-Path "$projectRoot\logs")) {
    New-Item -ItemType Directory -Path "$projectRoot\logs" | Out-Null
}

# -------------------------------------------------------------------
# Define the full session log path using ABSOLUTE path
# -------------------------------------------------------------------
$fullLog = "$projectRoot\logs\full_session_$timestamp.txt"

# -------------------------------------------------------------------
# Start recording everything (screen + file simultaneously)
# -------------------------------------------------------------------
Start-Transcript -Path $fullLog

# -------------------------------------------------------------------
# List project root contents to confirm correct working directory
# -------------------------------------------------------------------
Get-ChildItem

# -------------------------------------------------------------------
# Create a local Python virtual environment for this project
# -------------------------------------------------------------------
python -m venv venv

# -------------------------------------------------------------------
# Allow activation of the virtual environment for the current
# PowerShell session only
# -------------------------------------------------------------------
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# -------------------------------------------------------------------
# Activate the virtual environment
# -------------------------------------------------------------------
.\venv\Scripts\Activate.ps1

# -------------------------------------------------------------------
# Set SESSION_LOG_FILENAME **AFTER** activation so it survives
# -------------------------------------------------------------------
$env:SESSION_LOG_FILENAME = "session_$timestamp.log"

Write-Output "Python session log : $projectRoot\logs\$env:SESSION_LOG_FILENAME"
Write-Output "Full session log   : $fullLog"
Write-Output ""

# -------------------------------------------------------------------
# Upgrade pip to the latest version inside the virtual environment
# -------------------------------------------------------------------
python -m pip install --upgrade pip

# -------------------------------------------------------------------
# Install all required Python dependencies from the project
# requirements file
# -------------------------------------------------------------------
python -m pip install -r requirements.txt

# -------------------------------------------------------------------
# Register the approved historical engineered dataset
# -------------------------------------------------------------------
python -m src.register_dataset --source "C:\Users\grace\OneDrive\Documents\JobOpportunities\Money Lion\DS Assessment\temp\Loan-level\clean_df.parquet"

# Verify dataset registration artifacts exist
Write-Output "data\clean_df.parquet exists:            $(Test-Path .\data\clean_df.parquet)"
Write-Output "metadata\dataset_registry.json exists:   $(Test-Path .\metadata\dataset_registry.json)"
Write-Output ""
Write-Output "--- dataset_registry.json contents ---"
Get-Content .\metadata\dataset_registry.json

# -------------------------------------------------------------------
# Register the approved Optuna best-trials artifact
# -------------------------------------------------------------------
python -m src.register_hyperparams --source "C:\Users\grace\OneDrive\Documents\JobOpportunities\Money Lion\DS Assessment\temp\Loan-level\best_trials_CatBoostClassifier.json"

# Verify hyperparameter registration artifacts exist
Write-Output "models\best_trials_CatBoostClassifier.json exists:  $(Test-Path .\models\best_trials_CatBoostClassifier.json)"
Write-Output "metadata\hyperparameter_registry.json exists:       $(Test-Path .\metadata\hyperparameter_registry.json)"

# -------------------------------------------------------------------
# Freeze the production parameter file and approved feature schema
# -------------------------------------------------------------------
python prepare_production_params.py

# Verify frozen production parameters were created
Write-Output "models\best_params_catboost_v1.json exists:  $(Test-Path .\models\best_params_catboost_v1.json)"
Write-Output ""
Write-Output "--- best_params_catboost_v1.json contents ---"
Get-Content .\models\best_params_catboost_v1.json

# -------------------------------------------------------------------
# Train the final governed CatBoost model
# -------------------------------------------------------------------
python -m src.train

# Verify training artifacts were created
Write-Output "models\risk_model_v1.cbm exists:          $(Test-Path .\models\risk_model_v1.cbm)"
Write-Output "models\baseline_predictions.npy exists:   $(Test-Path .\models\baseline_predictions.npy)"
Write-Output "models\train_metrics.json exists:         $(Test-Path .\models\train_metrics.json)"

# Inspect the saved training metrics JSON
Write-Output ""
Write-Output "--- train_metrics.json contents ---"
Get-Content .\models\train_metrics.json | ConvertFrom-Json

# Inspect file sizes and timestamps of model artifacts
Write-Output ""
Write-Output "--- Model artifact details ---"
Get-Item .\models\risk_model_v1.cbm           | Select-Object Name, Length, LastWriteTime
Get-Item .\models\baseline_predictions.npy    | Select-Object Name, Length, LastWriteTime
Get-Item .\models\train_metrics.json          | Select-Object Name, Length, LastWriteTime

# -------------------------------------------------------------------
# Build a simulated incoming raw batch
# -------------------------------------------------------------------
python -m src.build_simulated_batch

# Verify simulated batch files were created
Write-Output "simulation\raw_batch\loan_data_batch.csv exists:           $(Test-Path .\simulation\raw_batch\loan_data_batch.csv)"
Write-Output "simulation\raw_batch\ach_payment_data_batch.csv exists:    $(Test-Path .\simulation\raw_batch\ach_payment_data_batch.csv)"
Write-Output "simulation\raw_batch\underwriting_data_batch.csv exists:   $(Test-Path .\simulation\raw_batch\underwriting_data_batch.csv)"
Write-Output ""
Write-Output "--- simulation_registry.json contents ---"
Get-Content .\metadata\simulation_registry.json

# -------------------------------------------------------------------
# Run the end-to-end inference pipeline
# -------------------------------------------------------------------
python -m src.inference_pipeline

# Verify inference output artifacts were created
Write-Output "simulation\processed_batch\new_features.parquet exists:   $(Test-Path .\simulation\processed_batch\new_features.parquet)"
Write-Output "simulation\processed_batch\predictions.parquet exists:    $(Test-Path .\simulation\processed_batch\predictions.parquet)"

# -------------------------------------------------------------------
# Run lightweight prediction drift monitoring
# -------------------------------------------------------------------
python run_monitoring.py

# Verify monitoring output was created
Write-Output "models\monitoring_log.json exists:  $(Test-Path .\models\monitoring_log.json)"
Write-Output ""
Write-Output "--- monitoring_log.json contents ---"
Get-Content .\models\monitoring_log.json

# -------------------------------------------------------------------
# Stop recording
# -------------------------------------------------------------------
Stop-Transcript
