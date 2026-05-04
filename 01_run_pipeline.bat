@echo off
REM ======================================================================
REM 01_run_pipeline.bat
REM
REM Step 1 of 3: Run the full ML pipeline.
REM
REM Purpose:
REM   This batch file launches PowerShell with the execution policy
REM   already set to Bypass, then runs run_all.ps1 which executes the
REM   full governed ML pipeline: training, scoring, and monitoring.
REM
REM Usage:
REM   Double-click this file.
REM
REM Next step:
REM   After this finishes, double-click 02_run_api.bat.
REM ======================================================================

powershell -NoProfile -ExecutionPolicy Bypass -File ".\run_all.ps1"
