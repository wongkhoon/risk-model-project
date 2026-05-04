@echo off
REM ======================================================================
REM 03_test_api.bat
REM
REM Step 3 of 3: Test the running API server.
REM
REM Prerequisites:
REM   - 01_run_pipeline.bat must have been run at least once.
REM   - The API server must already be running (02_run_api.bat).
REM
REM Usage:
REM   Double-click this file while the API server is running.
REM ======================================================================

powershell -NoProfile -ExecutionPolicy Bypass -NoExit -File "C:\Users\grace\OneDrive\Documents\Learning\risk_model_project\run_test.ps1"
