"""
Mapper for converting between adapter and core FeatureConfigVersionInfo models.
"""

from drl_trading_common.adapter.model.feature_config_version_info import FeatureConfigVersionInfo as AdapterFeatureConfigVersionInfo
from drl_trading_core.core.model.feature_config_version_info import FeatureConfigVersionInfo as CoreFeatureConfigVersionInfo
from .feature_definition_mapper import FeatureDefinitionMapper


class FeatureConfigVersionInfoMapper:
    """
    Mapper for FeatureConfigVersionInfo between adapter and core layers.
    """

    @staticmethod
    def dto_to_domain(dto: AdapterFeatureConfigVersionInfo) -> CoreFeatureConfigVersionInfo:
        """
        Convert adapter FeatureConfigVersionInfo (DTO) to core FeatureConfigVersionInfo (domain).

        Args:
            dto: Adapter FeatureConfigVersionInfo DTO

        Returns:
            Core FeatureConfigVersionInfo domain model
        """
        return CoreFeatureConfigVersionInfo(
            semver=dto.semver,
            hash=dto.hash,
            created_at=dto.created_at,
            feature_definitions=[FeatureDefinitionMapper.dto_to_domain(fd) for fd in dto.feature_definitions],
            description=dto.description
        )

    @staticmethod
    def domain_to_dto(domain: CoreFeatureConfigVersionInfo) -> AdapterFeatureConfigVersionInfo:
        """
        Convert core FeatureConfigVersionInfo (domain) to adapter FeatureConfigVersionInfo (DTO).

        Args:
            domain: Core FeatureConfigVersionInfo domain model

        Returns:
            Adapter FeatureConfigVersionInfo DTO
        """
        return AdapterFeatureConfigVersionInfo(
            semver=domain.semver,
            hash=domain.hash,
            created_at=domain.created_at,
            feature_definitions=[FeatureDefinitionMapper.domain_to_dto(fd) for fd in domain.feature_definitions],
            description=domain.description
        )
