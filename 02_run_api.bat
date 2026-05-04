@echo off
REM ======================================================================
REM 02_run_api.bat
REM
REM Step 2 of 3: Start the API server.
REM
REM Purpose:
REM   This batch file launches PowerShell and runs run_api.ps1 which
REM   activates the virtual environment and starts the Uvicorn server.
REM
REM Prerequisites:
REM   - 01_run_pipeline.bat must have been run at least once.
REM
REM Usage:
REM   Double-click this file after 01_run_pipeline.bat has completed.
REM
REM Next step:
REM   After this shows "Application startup complete", double-click
REM   03_test_api.bat.
REM ======================================================================

powershell -NoProfile -ExecutionPolicy Bypass -NoExit -File "C:\Users\grace\OneDrive\Documents\Learning\risk_model_project\run_api.ps1"
