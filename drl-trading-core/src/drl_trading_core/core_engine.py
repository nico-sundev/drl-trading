import logging
import os
from typing import List, Optional

from drl_trading_common.base.base_strategy_module import BaseStrategyModule
from injector import Injector

from drl_trading_core.common.data_import.data_import_manager import DataImportManager
from drl_trading_core.common.di.core_module import CoreModule
from drl_trading_core.common.model.preprocessing_result import PreprocessingResult
from drl_trading_core.preprocess.data_set_utils.strip_service import StripService
from drl_trading_preprocess.core.service.preprocess_service import PreprocessService

logger = logging.getLogger(__name__)

class CoreEngine:
    def __init__(self, strategy_module: BaseStrategyModule):
        self.injector = Injector([
            CoreModule(),
            strategy_module.as_injector_module()
        ])

    def run_batch_preprocessing(self, config_path: Optional[str] = None) -> List[PreprocessingResult]:
        """
        Run batch preprocessing on the provided data.

        Args:
            data: The input data to preprocess.

        Returns:
            Preprocessed data.
        """
        return self._preprocess(config_path)

    def run_incremental_preprocessing(self, data):
        """
        Run incremental preprocessing on the provided data.

        Args:
            data: The input data to preprocess.

        Returns:
            Preprocessed data.
        """
        raise NotImplementedError(
            "Incremental preprocessing is not implemented yet. Please use batch preprocessing."
        )


    def get_service(self, service_class):
        """Convenience function to get a service from the injector."""
        return self.injector.get(service_class)


    def _preprocess(
        self,
        config_path: Optional[str] = None,
    ) -> List[PreprocessingResult]:
        """Bootstrap the application using modern dependency injection.

        This function initializes the injector container, configures logging,
        and processes data through the pipeline using @inject decorators.

        Args:
            config_path: Optional path to config file. If provided, it overrides any
                        path set by DRL_TRADING_CONFIG_PATH environment variable or default.

        Returns:
            Tuple containing:
            - None (no longer returning container, services are resolved via injector)
            - List of preprocessed dataframes
        """

        # Log the effective config path being used
        env_var_value = os.getenv("DRL_TRADING_CONFIG_PATH")
        if config_path:
            logger.info(f"Using explicit config_path parameter: {config_path}")
        elif env_var_value:
            logger.info(
                f"Using config path from DRL_TRADING_CONFIG_PATH environment variable: {env_var_value}"
            )
        else:
            logger.info("Using default config path from injector configuration")

        logger.info("Modern DI injector configured")

        data_import_manager = self.get_service(DataImportManager)
        strip_service = self.get_service(StripService)
        preprocess_service = self.get_service(PreprocessService)

        # Import data for all symbols
        symbol_containers = data_import_manager.get_data()
        logger.info(f"Imported data for {len(symbol_containers)} symbols")

        final_datasets = []
        for symbol_container in symbol_containers:
            logger.info(
                f"Processing {len(symbol_container.datasets)} datasets for symbol: {symbol_container.symbol}"
            )

            # Transform and strip other timeframes
            symbol_container.datasets = strip_service.strip_asset_price_datasets(
                symbol_container.datasets
            )
            logger.info("Timeframe stripping completed")

            # Preprocess data using injected service
            preprocessed_dataset = preprocess_service.preprocess_data(symbol_container)
            final_datasets.append(preprocessed_dataset)
            logger.info(
                f"Feature preprocessing completed for symbol: {symbol_container.symbol}"
            )

        logger.info("Application bootstrap completed successfully")
        return final_datasets
