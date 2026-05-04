# ===========================================================================
# run_api.ps1
#
# Start the FastAPI prediction API server.
#
# Purpose:
#   Activate the virtual environment, set the session log filename,
#   install API dependencies if needed, and launch the Uvicorn server
#   hosting the prediction endpoint.
#
# Usage:
#   Called by 02_run_api.bat, or manually:
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\run_api.ps1"
#
# Output:
#   - logs/api_YYYYMMDD_HHMMSS.log      : Python-only log (structured)
#   - logs/api_YYYYMMDD_HHMMSS_full.txt  : Full console transcript
#
# To stop:
#   Press Ctrl+C in the PowerShell window.
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
# Generate ONE timestamp for this API session
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
$env:SESSION_LOG_FILENAME = "api_$timestamp.log"

# -------------------------------------------------------------------
# Start full console transcript (captures uvicorn + Python output)
# -------------------------------------------------------------------
$fullLog = "$projectRoot\logs\api_${timestamp}_full.txt"
Start-Transcript -Path $fullLog

Write-Output "API Python log     : $projectRoot\logs\$env:SESSION_LOG_FILENAME"
Write-Output "API full transcript: $fullLog"
Write-Output ""

# -------------------------------------------------------------------
# Install API server dependencies.
# pip skips these automatically if they are already installed, so it
# is safe to run every time.
# -------------------------------------------------------------------
python -m pip install uvicorn
python -m pip install fastapi

# -------------------------------------------------------------------
# Confirm we are in the project root before launching uvicorn
# -------------------------------------------------------------------
Set-Location $projectRoot

Write-Output ""
Write-Output "Starting API server from: $(Get-Location)"
Write-Output ""

# -------------------------------------------------------------------
# Start the API server with access logging enabled.
# --reload enables auto-restart when source files change (development
# mode only — remove --reload for production deployment).
# --access-log logs every HTTP request.
# The server runs continuously until you press Ctrl+C.
# -------------------------------------------------------------------
python -m uvicorn app:app --reload --access-log --host 127.0.0.1 --port 8000
