"""
Mapper for Timeframe enum between adapter and core layers.
"""

from drl_trading_common.adapter.model.timeframe import Timeframe as AdapterTimeframe
from drl_trading_common.core.model.timeframe import Timeframe as CoreTimeframe


class TimeframeMapper:
    """
    Mapper for Timeframe enum between adapter and core layers.

    Since Timeframe is an enum with identical values in both layers,
    the mapper simply returns the same enum value.
    """

    @staticmethod
    def dto_to_domain(dto_timeframe: AdapterTimeframe) -> CoreTimeframe:
        """
        Convert adapter Timeframe (DTO) to core Timeframe (domain).

        Args:
            dto_timeframe: Timeframe from adapter layer (DTO)

        Returns:
            Corresponding core Timeframe domain model
        """
        # Since both enums have identical values, we can map by value
        return CoreTimeframe(dto_timeframe.value)

    @staticmethod
    def domain_to_dto(domain_timeframe: CoreTimeframe) -> AdapterTimeframe:
        """
        Convert core Timeframe (domain) to adapter Timeframe (DTO).

        Args:
            domain_timeframe: Timeframe from core layer (domain)

        Returns:
            Corresponding adapter Timeframe DTO
        """
        # Since both enums have identical values, we can map by value
        return AdapterTimeframe(domain_timeframe.value)
