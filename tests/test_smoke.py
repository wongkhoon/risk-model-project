"""
Smoke tests: verify all source modules can be imported cleanly.
Catches syntax errors and missing dependencies in CI before they reach prod.
"""

import importlib

import pytest

# Modules expected to import without error
MODULES = [
    "src.config",
    "src.logging_config",
    "src.feature_engineering",
    "src.predict",
    "src.inference_pipeline",
    "src.monitoring",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name):
    """Each module under src/ must import without raising."""
    importlib.import_module(module_name)
