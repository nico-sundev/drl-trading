"""
Mapper for converting between adapter and core FeatureDefinition models.
"""

from drl_trading_common.adapter.model.feature_definition import FeatureDefinition as AdapterFeatureDefinition
from drl_trading_core.core.model.feature_definition import FeatureDefinition as CoreFeatureDefinition
from .base_parameter_set_config_mapper import BaseParameterSetConfigMapper


class FeatureDefinitionMapper:
    """
    Mapper for FeatureDefinition between adapter and core layers.
    """

    @staticmethod
    def dto_to_domain(dto: AdapterFeatureDefinition) -> CoreFeatureDefinition:
        """
        Convert adapter FeatureDefinition (DTO) to core FeatureDefinition (domain).

        Args:
            dto: Adapter FeatureDefinition DTO

        Returns:
            Core FeatureDefinition domain model
        """
        # Map parameter sets from adapter to core
        parsed_parameter_sets = {
            key: BaseParameterSetConfigMapper.dto_to_domain(param_set)
            for key, param_set in dto.parsed_parameter_sets.items()
        }

        return CoreFeatureDefinition(
            name=dto.name,
            enabled=dto.enabled,
            derivatives=dto.derivatives,
            parsed_parameter_sets=parsed_parameter_sets
        )

    @staticmethod
    def domain_to_dto(domain: CoreFeatureDefinition) -> AdapterFeatureDefinition:
        """
        Convert core FeatureDefinition (domain) to adapter FeatureDefinition (DTO).

        Args:
            domain: Core FeatureDefinition domain model

        Returns:
            Adapter FeatureDefinition DTO
        """
        # Map parameter sets from core to adapter
        parsed_parameter_sets = {
            key: BaseParameterSetConfigMapper.domain_to_dto(param_set)
            for key, param_set in domain.parsed_parameter_sets.items()
        }

        return AdapterFeatureDefinition(
            name=domain.name,
            enabled=domain.enabled,
            derivatives=domain.derivatives,
            parsed_parameter_sets=parsed_parameter_sets
        )
