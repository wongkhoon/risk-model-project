"""
Register an externally generated baseline prediction artifact into the project.

This script:
- copies baseline_predictions.npy into the governed models directory
- computes a SHA256 fingerprint
- records metadata for auditability

Execution
---------
python -m src.register_baseline --source "/full/path/to/baseline_predictions.npy"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from src.config import BASELINE_PRED_PATH, BASELINE_REGISTRY_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def compute_hash(path: Path) -> str:
    """Compute the SHA256 hash of a file."""
    sha = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def main() -> None:
    """Register approved baseline prediction artifact."""
    parser = argparse.ArgumentParser(description="Register baseline prediction artifact.")
    parser.add_argument(
        "--source",
        required=True,
        help="Path to baseline_predictions.npy",
    )
    args = parser.parse_args()

    source_path = Path(args.source)
    if not source_path.exists():
        raise FileNotFoundError(f"Baseline artifact not found at {source_path}")

    logging.info("Copying baseline prediction artifact into governed models directory...")
    shutil.copy2(source_path, BASELINE_PRED_PATH)

    baseline_array = np.load(BASELINE_PRED_PATH)

    metadata_entry = {
        "registered_at_malaysia": datetime.now(ZoneInfo("Asia/Kuala_Lumpur")).isoformat(),
        "original_source_path": str(source_path.resolve()),
        "project_baseline_path": str(BASELINE_PRED_PATH.resolve()),
        "sha256_hash": compute_hash(BASELINE_PRED_PATH),
        "file_size_bytes": BASELINE_PRED_PATH.stat().st_size,
        "num_predictions": int(len(baseline_array)),
    }

    if BASELINE_REGISTRY_PATH.exists():
        registry = json.loads(BASELINE_REGISTRY_PATH.read_text(encoding="utf-8"))
    else:
        registry = []

    registry.append(metadata_entry)
    BASELINE_REGISTRY_PATH.write_text(json.dumps(registry, indent=4), encoding="utf-8")

    logging.info("Baseline prediction artifact registered successfully.")


if __name__ == "__main__":
    main()
