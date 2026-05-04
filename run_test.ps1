# ===========================================================================
# run_test.ps1
#
# Test the running API server.
#
# Purpose:
#   Activate the virtual environment, set the session log filename,
#   and run test_api.py to send sample records to the API.
#
# Usage:
#   Called by 03_test_api.bat, or manually:
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\run_test.ps1"
#
# Output:
#   - logs/test_YYYYMMDD_HHMMSS.log      : Python-only log (structured)
#   - logs/test_YYYYMMDD_HHMMSS_full.txt  : Full console transcript
#
# Prerequisites:
#   - 01_run_pipeline.bat must have been run at least once.
#   - The API server must already be running (02_run_api.bat).
# ===========================================================================

# -------------------------------------------------------------------
# Define project root as an absolute path
# -------------------------------------------------------------------
$projectRoot = "C:\Users\grace\OneDrive\Documents\Learning\risk_model_project"

# -------------------------------------------------------------------
# Navigate to the project root directory
# -------------------------------------------------------------------
Set-Location $projectRoot

# -------------------------------------------------------------------
# Kill any lingering transcript from a crashed previous run
# -------------------------------------------------------------------
try { Stop-Transcript } catch { }

# -------------------------------------------------------------------
# Allow script execution for this PowerShell session only
# -------------------------------------------------------------------
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# -------------------------------------------------------------------
# Verify the virtual environment exists before proceeding
# -------------------------------------------------------------------
$activateScript = "$projectRoot\venv\Scripts\Activate.ps1"

if (-not (Test-Path $activateScript)) {
    Write-Host ""
    Write-Host "ERROR: Virtual environment not found at:" -ForegroundColor Red
    Write-Host "  $activateScript" -ForegroundColor Red
    Write-Host ""
    Write-Host "Run 01_run_pipeline.bat first to create the virtual environment." -ForegroundColor Yellow
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

# -------------------------------------------------------------------
# Activate the virtual environment using ABSOLUTE path
# -------------------------------------------------------------------
& $activateScript

# -------------------------------------------------------------------
# Generate ONE timestamp for this test session
# -------------------------------------------------------------------
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

# -------------------------------------------------------------------
# Ensure logs folder exists
# -------------------------------------------------------------------
if (-not (Test-Path "$projectRoot\logs")) {
    New-Item -ItemType Directory -Path "$projectRoot\logs" | Out-Null
}

# -------------------------------------------------------------------
# Set SESSION_LOG_FILENAME AFTER activation so Python's
# logging_config.py uses this instead of generating a new timestamp
# -------------------------------------------------------------------
$env:SESSION_LOG_FILENAME = "test_$timestamp.log"

# -------------------------------------------------------------------
# Start full console transcript
# -------------------------------------------------------------------
$fullLog = "$projectRoot\logs\test_${timestamp}_full.txt"
Start-Transcript -Path $fullLog

Write-Output "Test Python log     : $projectRoot\logs\$env:SESSION_LOG_FILENAME"
Write-Output "Test full transcript: $fullLog"
Write-Output ""

# -------------------------------------------------------------------
# Run the test script
# -------------------------------------------------------------------
python test_api.py

# -------------------------------------------------------------------
# Stop recording
# -------------------------------------------------------------------
Stop-Transcript
