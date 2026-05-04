"""
Register an externally generated approved trained model artifact into the project.

This script:
- copies the trained CatBoost model artifact into the governed models directory
- computes a SHA256 fingerprint
- records model artifact metadata for auditability

Execution
---------
python -m src.register_model --source "/full/path/to/risk_model_v1.cbm"
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

from src.config import MODEL_PATH, MODEL_REGISTRY_PATH

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
    """Register approved trained model artifact."""
    parser = argparse.ArgumentParser(description="Register trained model artifact.")
    parser.add_argument("--source", required=True, help="Path to risk_model_v1.cbm")
    args = parser.parse_args()

    source_path = Path(args.source)
    if not source_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {source_path}")

    logging.info("Copying model artifact into governed models directory...")
    shutil.copy2(source_path, MODEL_PATH)

    metadata_entry = {
        "registered_at_malaysia": datetime.now(ZoneInfo("Asia/Kuala_Lumpur")).isoformat(),
        "original_source_path": str(source_path.resolve()),
        "project_model_path": str(MODEL_PATH.resolve()),
        "sha256_hash": compute_hash(MODEL_PATH),
        "file_size_bytes": MODEL_PATH.stat().st_size,
    }

    if MODEL_REGISTRY_PATH.exists():
        registry = json.loads(MODEL_REGISTRY_PATH.read_text(encoding="utf-8"))
    else:
        registry = []

    registry.append(metadata_entry)
    MODEL_REGISTRY_PATH.write_text(json.dumps(registry, indent=4), encoding="utf-8")

    logging.info("Model artifact registered successfully.")


if __name__ == "__main__":
    main()
