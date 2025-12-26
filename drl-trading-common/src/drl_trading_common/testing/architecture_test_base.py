"""
Base architecture test class for hexagonal architecture enforcement.

This module provides a reusable test base that can be used by any service
to enforce hexagonal architecture constraints with minimal code duplication.
"""

import subprocess
from pathlib import Path

import pytest

from drl_trading_common.testing.architecture_rules import (
    ArchitectureRule,
    HexagonalArchitectureRules,
)


class BaseArchitectureTest:
    """
    Base test class for hexagonal architecture enforcement.

    Services should inherit from this class and provide the service-specific
    configuration via class attributes.

    Example:
        class TestPreprocessArchitecture(BaseArchitectureTest):
            service_name = "preprocess"
            service_package = "drl_trading_preprocess"
    """

    # Override these in subclasses
    service_name: str = ""
    service_package: str = ""

    @pytest.fixture(scope="class")
    def project_root(self) -> Path:
        """Get the project root directory (service directory)."""
        # Navigate from tests/unit/architecture to service root
        return Path(__file__).parent.parent.parent.parent

    @pytest.fixture(scope="class")
    def rules(self) -> HexagonalArchitectureRules:
        """Get the architecture rules instance."""
        return HexagonalArchitectureRules()

    def test_import_linter_configuration_exists(self, project_root: Path) -> None:
        """Test that import-linter configuration file exists."""
        # Given
        config_file = project_root / ".importlinter"

        # When / Then
        assert config_file.exists(), (
            f"Missing .importlinter configuration file for {self.service_name} service"
        )

    def test_core_layer_independence(self, project_root: Path, rules: HexagonalArchitectureRules) -> None:
        """
        Test that core layer does not import from adapter or infrastructure.

        This is the most critical rule of hexagonal architecture.
        """
        # Given
        rule = rules.get_core_independence_rule(self.service_name)
        layers = rules.get_layer_packages(self.service_package)

        # When
        result = self._run_import_linter(project_root, rule.name)

        # Then
        assert result.returncode == 0, (
            f"Core layer independence violation in {self.service_name}!\n"
            f"Rule: {rule.description}\n"
            f"Core package: {layers['core']}\n"
            f"Forbidden imports: {layers['adapter']}, {layers['infrastructure']}\n\n"
            f"Output:\n{result.stdout.decode()}"
        )

    def test_adapter_layer_dependencies(self, project_root: Path, rules: HexagonalArchitectureRules) -> None:
        """Test that adapter layer can use core but follows layering rules."""
        # Given
        rule = rules.get_adapter_dependency_rule(self.service_name)
        layers = rules.get_layer_packages(self.service_package)

        # When
        result = self._run_import_linter(project_root, rule.name)

        # Then
        assert result.returncode == 0, (
            f"Adapter layer dependency violation in {self.service_name}!\n"
            f"Rule: {rule.description}\n"
            f"Adapter package: {layers['adapter']}\n\n"
            f"Output:\n{result.stdout.decode()}"
        )

    def test_infrastructure_layer_integration(self, project_root: Path, rules: HexagonalArchitectureRules) -> None:
        """Test that infrastructure layer properly integrates core and adapters."""
        # Given
        rule = rules.get_infrastructure_integration_rule(self.service_name)
        layers = rules.get_layer_packages(self.service_package)

        # When
        result = self._run_import_linter(project_root, rule.name)

        # Then
        assert result.returncode == 0, (
            f"Infrastructure layer integration violation in {self.service_name}!\n"
            f"Rule: {rule.description}\n"
            f"Infrastructure package: {layers['infrastructure']}\n\n"
            f"Output:\n{result.stdout.decode()}"
        )

    def test_all_architecture_rules_pass(self, project_root: Path, rules: HexagonalArchitectureRules) -> None:
        """Test that all architecture rules pass together."""
        # Given
        all_rules = rules.get_all_rules(self.service_name)

        # When
        result = self._run_import_linter(project_root)

        # Then
        assert result.returncode == 0, (
            f"Architecture violations in {self.service_name}! "
            f"{len(all_rules)} rules defined.\n\n"
            f"Output:\n{result.stdout.decode()}"
        )

    @staticmethod
    def _run_import_linter(
        project_root: Path, contract_name: str | None = None
    ) -> subprocess.CompletedProcess:
        """
        Run import-linter to check architecture rules.

        Args:
            project_root: Root directory of the service
            contract_name: Optional specific contract to check

        Returns:
            CompletedProcess with return code and output
        """
        cmd = ["lint-imports", "--config", str(project_root / ".importlinter")]

        if contract_name:
            cmd.extend(["--contract", contract_name])

        return subprocess.run(
            cmd,
            cwd=project_root,
            capture_output=True,
            check=False,
        )


class BaseArchitectureDocumentationTest:
    """Base test class for architecture documentation verification."""

    service_name: str = ""
    service_package: str = ""

    def test_architecture_rules_are_documented(self) -> None:
        """Test that all architecture rules have proper documentation."""
        # Given
        rules_instance = HexagonalArchitectureRules()
        rules = rules_instance.get_all_rules(self.service_name)

        # When / Then
        for rule in rules:
            assert rule.name, f"Rule is missing name: {rule}"
            assert rule.description, f"Rule {rule.name} is missing description"
            assert rule.contract_type in [
                "forbidden",
                "layers",
                "independence",
            ], f"Rule {rule.name} has invalid contract type: {rule.contract_type}"

    def test_layer_packages_are_defined(self) -> None:
        """Test that all layer package names are properly defined."""
        # Given / When
        layers = HexagonalArchitectureRules.get_layer_packages(self.service_package)

        # Then
        assert "core" in layers
        assert "adapter" in layers
        assert "infrastructure" in layers
        assert all(self.service_package in pkg for pkg in layers.values())

    def test_all_rules_are_retrievable(self) -> None:
        """Test that all rules can be retrieved as a collection."""
        # Given / When
        rules = HexagonalArchitectureRules.get_all_rules(self.service_name)

        # Then
        assert len(rules) == 3, "Expected exactly 3 architecture rules"
        assert all(isinstance(rule, ArchitectureRule) for rule in rules)
