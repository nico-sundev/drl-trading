"""
Mapper for converting between adapter and core DatasetIdentifier models.
"""

from drl_trading_common.adapter.model.dataset_identifier import DatasetIdentifier as AdapterDatasetIdentifier
from drl_trading_common.core.model.dataset_identifier import DatasetIdentifier as CoreDatasetIdentifier


class DatasetIdentifierMapper:
    """
    Mapper for DatasetIdentifier between adapter and core layers.
    """

    @staticmethod
    def dto_to_domain(dto: AdapterDatasetIdentifier) -> CoreDatasetIdentifier:
        """
        Convert adapter DatasetIdentifier (DTO) to core DatasetIdentifier (domain).

        Args:
            dto: Adapter DatasetIdentifier DTO

        Returns:
            Core DatasetIdentifier domain model
        """
        return CoreDatasetIdentifier(
            symbol=dto.symbol,
            timeframe=dto.timeframe
        )

    @staticmethod
    def domain_to_dto(domain: CoreDatasetIdentifier) -> AdapterDatasetIdentifier:
        """
        Convert core DatasetIdentifier (domain) to adapter DatasetIdentifier (DTO).

        Args:
            domain: Core DatasetIdentifier domain model

        Returns:
            Adapter DatasetIdentifier DTO
        """
        return AdapterDatasetIdentifier(
            symbol=domain.symbol,
            timeframe=domain.timeframe
        )
