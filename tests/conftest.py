"""
Pytest configuration for Options Probability tests

Critical: TensorFlow must be imported BEFORE other ML libraries on Windows
to avoid DLL initialization errors. This conftest.py ensures proper import order.
"""

# CRITICAL: Import TensorFlow first (Windows DLL fix)
try:
    import tensorflow as tf
    # Suppress TensorFlow warnings during tests
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
except ImportError:
    pass

# Now safe to import other test fixtures
import pytest


def pytest_configure(config):
    """Register custom markers for model accuracy tests."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "regression: marks tests as regression tests for accuracy baselines"
    )
    config.addinivalue_line(
        "markers", "accuracy_baseline: marks tests that establish model accuracy baselines"
    )
