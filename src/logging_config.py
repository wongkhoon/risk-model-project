"""
logging_config.py

Centralized logging configuration for all project entry points.

Purpose
-------
Ensure that every Python script in this project writes log output to:
1. The console (stdout) — so the user sees output in real time.
2. A single shared log file (logs/session_YYYYMMDD_HHMMSS.log) — so all
   output is captured in one governed, timestamped artifact for
   auditability and reproducibility.

Why Timestamp in the Filename
-----------------------------
Including a timestamp in the log filename ensures that:
- Each pipeline run produces a distinct, self-contained log file.
- Previous session logs are never overwritten or lost.
- Logs across different runs side by side can be compared.
- The exact time of each pipeline run can be traced.

Why stdout Instead of stderr
-----------------------------
Python's logging module defaults to stderr. However, PowerShell's Tee-Object
only reliably captures stdout. Writing to stdout ensures that all Python log
output is captured in the PowerShell session log file without
NativeCommandError noise or stream-merging workarounds.

Usage
-----
In every Python script that needs logging, replace:

    logging.basicConfig(level = logging.INFO, format = "...")

with:

    from src.logging_config import setup_logging
    logger = setup_logging("<script_name>")

Then use logger.info(), logger.warning(), logger.error() instead of
logging.info(), logging.warning(), logging.error().

Notes
-----
- All scripts within the same Python process append to the same timestamped
  session log file.
- Separate Python processes (separate python commands in run_all.ps1) each
  create their own session log file unless SESSION_LOG_FILENAME is set.
- The _DONE flag prevents duplicate handlers when multiple modules import
  this configuration within the same Python process.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

# ---------------------------------------------------------------------------
# Governed log directory — created automatically if it does not exist.
# Lives at the project root alongside data/, models/, metadata/, etc.
# ---------------------------------------------------------------------------
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Generate a timestamped log filename using Malaysia timezone.
#
# The environment variable SESSION_LOG_FILENAME allows run_all.ps1 to force
# all Python scripts in the same pipeline run to write to one single log
# file. If the variable is not set, each Python process generates its own
# timestamped filename as a safe fallback.
#
# Example filenames:
#   session_20260425_012754.log  (from environment variable)
#   session_20260425_013012.log  (auto-generated fallback)
# ---------------------------------------------------------------------------
_ENV_LOG_FILENAME = os.environ.get("SESSION_LOG_FILENAME")

if _ENV_LOG_FILENAME:
    # Use the shared filename set by run_all.ps1 so every script in the
    # pipeline run appends to the same file.
    LOG_FILE = LOG_DIR / _ENV_LOG_FILENAME
else:
    # Fallback: generate a unique timestamped filename for this process.
    _TIMESTAMP = datetime.now(ZoneInfo("Asia/Kuala_Lumpur")).strftime("%Y%m%d_%H%M%S")
    LOG_FILE = LOG_DIR / f"session_{_TIMESTAMP}.log"

# ---------------------------------------------------------------------------
# Module-level flag to prevent attaching duplicate handlers when multiple
# scripts or modules call setup_logging() within the same Python process.
# ---------------------------------------------------------------------------
_DONE = False


def setup_logging(name: str) -> logging.Logger:
    """
    Configure and return a named logger for a project entry point.

    On the first call, this function attaches two handlers to the root logger:
    - A StreamHandler writing to stdout (visible on screen).
    - A FileHandler appending to the timestamped session log file.

    Subsequent calls within the same process skip handler setup and simply
    return a new named logger that inherits the root logger's handlers.

    Parameters
    ----------
    name : str
        Identifier for the calling script (e.g., "train", "monitoring").
        Appears in every log line so you can trace which script produced it.

    Returns
    -------
    logging.Logger
        A named logger ready for use with .info(), .warning(), .error(), etc.
    """
    # Reference the module-level flag
    global _DONE

    # Create a named logger unique to each entry point.
    # Named loggers inherit handlers from the root logger.
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Only configure handlers once per Python process
    if not _DONE:
        # Consistent format: timestamp, severity, source script, message
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")

        # Handler 1: Console output via stdout.
        # Using stdout (not stderr) so PowerShell's Tee-Object captures it
        # cleanly without NativeCommandError formatting noise.
        screen_handler = logging.StreamHandler(sys.stdout)
        screen_handler.setFormatter(formatter)

        # Handler 2: File output appending to the timestamped session log.
        # mode="a" ensures every script adds to the same file rather than
        # overwriting previous output.
        file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)

        # Attach both handlers to the root logger.
        # All named loggers inherit from root, so every logger.info() call
        # automatically writes to both console and file.
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)

        # Clear any pre-existing handlers to prevent duplicates from
        # prior basicConfig() calls or interactive sessions.
        root_logger.handlers.clear()

        root_logger.addHandler(screen_handler)
        root_logger.addHandler(file_handler)

        # Mark setup as complete for this process
        _DONE = True

    return logger
