"""
Example architecture tests for drl-trading-training service.

Copy this file to drl-trading-training/tests/unit/architecture/architecture_test.py
and it will work immediately with zero code duplication!
"""

from pathlib import Path

import pytest

from drl_trading_common.testing.architecture_test_base import (
    BaseArchitectureDocumentationTest,
    BaseArchitectureTest,
)


class TestTrainingArchitecture(BaseArchitectureTest):
    """Architecture tests for the training service."""

    service_name = "training"
    service_package = "drl_trading_training"

    @pytest.fixture(scope="class")
    def project_root(self) -> Path:
        """Get the training service root directory."""
        return Path(__file__).parent.parent.parent.parent


class TestTrainingArchitectureDocumentation(BaseArchitectureDocumentationTest):
    """Documentation tests for training architecture rules."""

    service_name = "training"
    service_package = "drl_trading_training"
