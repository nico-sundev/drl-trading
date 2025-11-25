"""
Mapper for converting between adapter and core TradingContext models.
"""

from drl_trading_common.adapter.model.trading_context import TradingContext as AdapterTradingContext
from drl_trading_core.core.model.trading_context import TradingContext as CoreTradingContext


class TradingContextMapper:
    """
    Mapper for TradingContext between adapter and core layers.
    """

    @staticmethod
    def dto_to_domain(dto: AdapterTradingContext) -> CoreTradingContext:
        """
        Convert adapter TradingContext (DTO) to core TradingContext (domain).

        Args:
            dto: Adapter TradingContext DTO

        Returns:
            Core TradingContext domain model
        """
        return CoreTradingContext(
            correlation_id=dto.correlation_id,
            event_id=dto.event_id,
            symbol=dto.symbol,
            timestamp=dto.timestamp,
            strategy_id=dto.strategy_id,
            timeframe=dto.timeframe,
            model_version=dto.model_version,
            prediction_confidence=dto.prediction_confidence,
            trade_id=dto.trade_id,
            execution_price=dto.execution_price,
            execution_quantity=dto.execution_quantity,
            metadata=dto.metadata
        )

    @staticmethod
    def domain_to_dto(domain: CoreTradingContext) -> AdapterTradingContext:
        """
        Convert core TradingContext (domain) to adapter TradingContext (DTO).

        Args:
            domain: Core TradingContext domain model

        Returns:
            Adapter TradingContext DTO
        """
        return AdapterTradingContext(
            correlation_id=domain.correlation_id,
            event_id=domain.event_id,
            symbol=domain.symbol,
            timestamp=domain.timestamp,
            strategy_id=domain.strategy_id,
            timeframe=domain.timeframe,
            model_version=domain.model_version,
            prediction_confidence=domain.prediction_confidence,
            trade_id=domain.trade_id,
            execution_price=domain.execution_price,
            execution_quantity=domain.execution_quantity,
            metadata=domain.metadata
        )
