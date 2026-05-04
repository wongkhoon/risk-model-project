"""
Register an externally generated historical training dataset into the project.

This script:
- copies clean_df.parquet into the governed data directory
- computes a SHA256 fingerprint
- records registration metadata for auditability

Execution
---------
python -m src.register_dataset --source "/full/path/to/clean_df.parquet"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from src.config import CLEAN_DATA_PATH, DATASET_REGISTRY_PATH

# logging.basicConfig(level = logging.INFO, format = "%(asctime)s - %(levelname)s - %(message)s",)
# Import the centralized logging configuration shared by all entry points
from src.logging_config import setup_logging

# Create a named logger for this script.
# "register_dataset" appears in every log line so you can identify the source.
logger = setup_logging("register_dataset")


def compute_hash(path: Path) -> str:
    """Compute the SHA256 hash of a file."""
    sha = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def main() -> None:
    """Register the governed historical training dataset."""
    parser = argparse.ArgumentParser(description="Register historical training dataset.")
    parser.add_argument("--source", required=True, help="Path to clean_df.parquet")
    args = parser.parse_args()

    source_path = Path(args.source)
    if not source_path.exists():
        raise FileNotFoundError(f"Dataset not found at {source_path}")

    logger.info("Copying dataset into governed data directory...")
    shutil.copy2(source_path, CLEAN_DATA_PATH)

    df = pd.read_parquet(CLEAN_DATA_PATH, engine="pyarrow")

    metadata_entry = {
        "registered_at_malaysia": datetime.now(ZoneInfo("Asia/Kuala_Lumpur")).isoformat(),
        "original_source_path": str(source_path.resolve()),
        "project_dataset_path": str(CLEAN_DATA_PATH.resolve()),
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "sha256_hash": compute_hash(CLEAN_DATA_PATH),
        "file_size_bytes": CLEAN_DATA_PATH.stat().st_size,
    }

    if DATASET_REGISTRY_PATH.exists():
        registry = json.loads(DATASET_REGISTRY_PATH.read_text(encoding="utf-8"))
    else:
        registry = []

    registry.append(metadata_entry)
    DATASET_REGISTRY_PATH.write_text(json.dumps(registry, indent=4), encoding="utf-8")

    logger.info("Dataset registered successfully.")


if __name__ == "__main__":
    main()
