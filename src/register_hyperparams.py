"""
Register externally generated Optuna best-trials JSON into the project.

This script:
- copies best_trials_CatBoostClassifier.json into the governed models directory
- computes a SHA256 fingerprint
- records metadata for auditability

Execution
---------
python -m src.register_hyperparams --source "/full/path/to/best_trials_CatBoostClassifier.json"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from src.config import HYPERPARAM_REGISTRY_PATH, REGISTERED_BEST_TRIALS_PATH

# logging.basicConfig(level = logging.INFO, format = "%(asctime)s - %(levelname)s - %(message)s",)
# Import the centralized logging configuration shared by all entry points
from src.logging_config import setup_logging

# Create a named logger for this script.
# "register_hyperparams" appears in every log line so you can identify the source.
logger = setup_logging("register_hyperparams")


def compute_hash(path: Path) -> str:
    """Compute the SHA256 hash of a file."""
    sha = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def main() -> None:
    """Register governed hyperparameter tuning artifact."""
    parser = argparse.ArgumentParser(description="Register Optuna best-trials JSON.")
    parser.add_argument(
        "--source",
        required=True,
        help="Path to best_trials_CatBoostClassifier.json",
    )
    args = parser.parse_args()

    source_path = Path(args.source)
    if not source_path.exists():
        raise FileNotFoundError(f"Hyperparameter artifact not found at {source_path}")

    logger.info("Copying hyperparameter artifact into governed models directory...")
    shutil.copy2(source_path, REGISTERED_BEST_TRIALS_PATH)

    metadata_entry = {
        "registered_at_malaysia": datetime.now(ZoneInfo("Asia/Kuala_Lumpur")).isoformat(),
        "original_source_path": str(source_path.resolve()),
        "project_hyperparam_path": str(REGISTERED_BEST_TRIALS_PATH.resolve()),
        "sha256_hash": compute_hash(REGISTERED_BEST_TRIALS_PATH),
        "file_size_bytes": REGISTERED_BEST_TRIALS_PATH.stat().st_size,
    }

    if HYPERPARAM_REGISTRY_PATH.exists():
        registry = json.loads(HYPERPARAM_REGISTRY_PATH.read_text(encoding="utf-8"))
    else:
        registry = []

    registry.append(metadata_entry)
    HYPERPARAM_REGISTRY_PATH.write_text(json.dumps(registry, indent=4), encoding="utf-8")

    logger.info("Hyperparameter artifact registered successfully.")


if __name__ == "__main__":
    main()
