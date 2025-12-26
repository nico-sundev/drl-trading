"""
Architecture rule definitions for hexagonal architecture enforcement.

This module provides reusable architecture rules that can be applied
to any service in the drl-trading system.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ArchitectureRule:
    """Represents a single architecture rule to be enforced."""

    name: str
    description: str
    contract_type: str


class HexagonalArchitectureRules:
    """
    Defines hexagonal architecture rules for drl-trading services.

    These rules are service-agnostic and can be applied to any service
    by substituting the service name.
    """

    @classmethod
    def get_core_independence_rule(cls, service_name: str) -> ArchitectureRule:
        """
        Core layer must not depend on adapter or infrastructure layers.

        Args:
            service_name: Name of the service (e.g., 'preprocess', 'training')
        """
        return ArchitectureRule(
            name="core-independence",
            description=(
                f"{service_name} core layer must not depend on adapter or infrastructure layers "
                "(including all common adapter/infrastructure/application packages). "
                "Core defines ports (interfaces) that adapters implement."
            ),
            contract_type="forbidden",
        )

    @classmethod
    def get_adapter_dependency_rule(cls, service_name: str) -> ArchitectureRule:
        """
        Adapter layer can use core but not infrastructure.

        Args:
            service_name: Name of the service (e.g., 'preprocess', 'training')
        """
        return ArchitectureRule(
            name="adapter-can-use-core",
            description=(
                f"{service_name} adapter layer can depend on core layers "
                "to implement ports, but not on infrastructure."
            ),
            contract_type="layers",
        )

    @classmethod
    def get_infrastructure_integration_rule(cls, service_name: str) -> ArchitectureRule:
        """
        Infrastructure layer can use both core and adapter.

        Args:
            service_name: Name of the service (e.g., 'preprocess', 'training')
        """
        return ArchitectureRule(
            name="infrastructure-can-use-all",
            description=(
                f"{service_name} infrastructure layer can depend on all layers "
                "(adapter, core, and common packages) to wire the application together."
            ),
            contract_type="layers",
        )

    @classmethod
    def get_all_rules(cls, service_name: str) -> list[ArchitectureRule]:
        """
        Get all defined architecture rules for a service.

        Args:
            service_name: Name of the service (e.g., 'preprocess', 'training')
        """
        return [
            cls.get_core_independence_rule(service_name),
            cls.get_adapter_dependency_rule(service_name),
            cls.get_infrastructure_integration_rule(service_name),
        ]

    @staticmethod
    def get_layer_packages(service_package: str) -> dict[str, str]:
        """
        Get layer package names for a service.

        Args:
            service_package: Full package name (e.g., 'drl_trading_preprocess')

        Returns:
            Dictionary with layer names as keys and package names as values
        """
        return {
            "core": f"{service_package}.core",
            "adapter": f"{service_package}.adapter",
            "infrastructure": f"{service_package}.infrastructure",
        }
