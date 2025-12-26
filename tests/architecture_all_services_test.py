"""
Workspace-level architecture tests for all services.

This test file can run from the workspace root and test all services at once.
It discovers all services dynamically and runs architecture tests for each.

Usage:
    # From workspace root
    pytest tests/architecture_all_services_test.py -v

    # Test specific service
    pytest tests/architecture_all_services_test.py -v -k "preprocess"
"""

from pathlib import Path

import pytest

from drl_trading_common.testing.architecture_test_base import (
    BaseArchitectureTest,
)

# Define all services to test
SERVICES = [
    {"name": "preprocess", "package": "drl_trading_preprocess", "path": "drl-trading-preprocess"},
    {"name": "training", "package": "drl_trading_training", "path": "drl-trading-training"},
    {"name": "inference", "package": "drl_trading_inference", "path": "drl-trading-inference"},
    {"name": "execution", "package": "drl_trading_execution", "path": "drl-trading-execution"},
    {"name": "ingest", "package": "drl_trading_ingest", "path": "drl-trading-ingest"},
]


@pytest.mark.parametrize("service_config", SERVICES, ids=[s["name"] for s in SERVICES])
class TestAllServicesArchitecture(BaseArchitectureTest):
    """Parameterized architecture tests for all services."""

    @pytest.fixture(scope="class", autouse=True)
    def setup_service(self, request, service_config: dict) -> None:
        """Set service-specific configuration before running tests."""
        # Set class attributes dynamically based on the parameter
        request.cls.service_name = service_config["name"]
        request.cls.service_package = service_config["package"]
        request.cls._service_path = service_config["path"]

    @pytest.fixture(scope="class")
    def project_root(self) -> Path:
        """Get the service root directory from workspace root."""
        workspace_root = Path(__file__).parent.parent
        return workspace_root / self._service_path

    # All test methods are inherited from BaseArchitectureTest
    # They will run automatically for each service!
