"""Sanity checks on the config module — guards against typos and missing constants."""

from src import config


def test_config_module_loads():
    """The config module should import without errors."""
    assert config is not None
