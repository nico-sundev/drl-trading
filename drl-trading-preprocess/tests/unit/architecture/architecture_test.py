"""
Architecture Tests for drl-trading-preprocess Service.

This module uses the shared architecture test framework from drl-trading-common
to enforce hexagonal architecture constraints with minimal code duplication.
"""

from pathlib import Path

import pytest

from drl_trading_common.testing.architecture_test_base import (
    BaseArchitectureDocumentationTest,
    BaseArchitectureTest,
)


class TestPreprocessArchitecture(BaseArchitectureTest):
    """Architecture tests for the preprocess service."""

    service_name = "preprocess"
    service_package = "drl_trading_preprocess"

    @pytest.fixture(scope="class")
    def project_root(self) -> Path:
        """Get the preprocess service root directory."""
        # Navigate from tests/unit/architecture to service root
        return Path(__file__).parent.parent.parent.parent


class TestPreprocessArchitectureDocumentation(BaseArchitectureDocumentationTest):
    """Documentation tests for preprocess architecture rules."""

    service_name = "preprocess"
    service_package = "drl_trading_preprocess"
