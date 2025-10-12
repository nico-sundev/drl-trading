"""Factory for creating data import strategy instances."""

import logging
from typing import Dict, Type

from injector import inject

from drl_trading_common.config.local_data_import_config import LocalDataImportConfig

from drl_trading_core.common.data_import.base_data_import_service import (
    BaseDataImportService,
)
from drl_trading_core.common.data_import.local.csv_data_import_service import (
    CsvDataImportService,
)
from drl_trading_core.common.data_import.web.yahoo_data_import_service import (
    YahooDataImportService,
)

logger = logging.getLogger(__name__)


@inject
class DataImportStrategyFactory:
    """Factory for creating data import strategy instances based on configuration."""

    def __init__(self):
        """Initialize the factory with available strategies."""
        self._strategies: Dict[str, Type[BaseDataImportService]] = {
            "csv": CsvDataImportService,
            "yahoo": YahooDataImportService,
        }

    def register_strategy(
        self, strategy_name: str, strategy_class: Type[BaseDataImportService]
    ) -> None:
        """
        Register a new data import strategy.

        Args:
            strategy_name: Name of the strategy
            strategy_class: Class implementing BaseDataImportService
        """
        if not issubclass(strategy_class, BaseDataImportService):
            raise TypeError(
                f"Strategy class {strategy_class.__name__} must extend BaseDataImportService"
            )

        if strategy_name in self._strategies:
            logger.warning(
                f"Overriding existing strategy '{strategy_name}': "
                f"{self._strategies[strategy_name].__name__} -> {strategy_class.__name__}"
            )

        self._strategies[strategy_name] = strategy_class
        logger.debug(
            f"Registered strategy '{strategy_name}' with class {strategy_class.__name__}"
        )

    def create_import_service(
        self, config: LocalDataImportConfig
    ) -> BaseDataImportService:
        """
        Create a data import service based on the strategy specified in config.

        Args:
            config: Local data import configuration containing strategy

        Returns:
            Instance of BaseDataImportService

        Raises:
            ValueError: If the strategy is not registered
            TypeError: If the strategy class cannot be instantiated with the given config
        """
        strategy_name = config.strategy.lower()

        if strategy_name not in self._strategies:
            available_strategies = ", ".join(self._strategies.keys())
            raise ValueError(
                f"Unknown import strategy '{strategy_name}'. "
                f"Available strategies: {available_strategies}"
            )

        strategy_class = self._strategies[strategy_name]
        logger.info(f"Creating import service using strategy: {strategy_name}")

        try:
            # For CSV strategy, pass the config directly
            if strategy_name == "csv":
                return strategy_class(config)

            # For Yahoo strategy, we need to handle the different constructor
            # For now, create a basic instance with placeholder values
            # In a real implementation, you'd extract these from config
            elif strategy_name == "yahoo":
                # Note: YahooDataImportService has a different constructor
                # This is a simplified implementation
                return strategy_class(
                    ticker="AAPL",  # This should come from config
                    start_date=None,
                    end_date=None,
                )

            # Default case for future strategies
            return strategy_class(config)

        except Exception as e:
            raise TypeError(
                f"Failed to instantiate strategy '{strategy_name}' "
                f"with class {strategy_class.__name__}: {e}"
            ) from e

    def get_available_strategies(self) -> list[str]:
        """
        Get list of available strategy names.

        Returns:
            List of registered strategy names
        """
        return list(self._strategies.keys())
