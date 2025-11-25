import logging
import os
from typing import Dict, List, Optional, Tuple
from warnings import deprecated

from drl_trading_common.config.feature_config import FeaturesConfig
# Legacy import - this class is deprecated, use FeatureConfigReader/Writer adapters instead
from drl_trading_training.adapter.feature_config.feature_config_writer import FeatureConfigWriter
from drl_trading_common.adapter.model.feature_config_version_info import (
    FeatureConfigVersionInfo,
)
from drl_trading_training.core.agents.base_agent import BaseAgent
from drl_trading_core.common.model.preprocessing_result import PreprocessingResult
from drl_trading_core.core_engine import CoreEngine
from drl_trading_core.preprocess.data_set_utils.split_service import SplitService
from drl_trading_core.training.services.agent_training_service import (
    AgentTrainingService,
)
from drl_trading_strategy_example.module.example_strategy_module import ExampleStrategyModule

logger = logging.getLogger(__name__)

@deprecated
class TrainingApp:

    def run(self) -> None:
        """
        Run the training application.
        """
        self.start_training()

    @staticmethod
    @deprecated("This method is obsolete. Use FeatureConfigWriter.save_config() and FeatureConfigReader.get_config() directly")
    def register_feature_config(writer: FeatureConfigWriter, config: FeaturesConfig, semver: str) -> FeatureConfigVersionInfo:
        """
        DEPRECATED: This method is no longer functional.

        Instead, use the new adapter pattern:
        - FeatureConfigReader.get_config() to read configurations
        - FeatureConfigWriter.save_config() to save configurations (has UPSERT semantics)
        """
        raise NotImplementedError(
            "This legacy method is deprecated. Use FeatureConfigWriter.save_config() and "
            "FeatureConfigReader.get_config() from the new adapter architecture instead."
        )

    def _create_environments_and_train(
        self,
        core_engine: CoreEngine,
        final_datasets: List[PreprocessingResult],
    ) -> Tuple[Dict[str, BaseAgent]]:
        """
        Create environments and train agents using injected services.

        Args:
            injector: The injector instance with resolved services
            final_datasets: List of preprocessing results containing final DataFrames
            env_class: The class of the trading environment to be used

        Returns:
            Dictionary of trained agents mapped by name
        """
        # Resolve required services from injector
        split_service = core_engine.get_service(SplitService)
        agent_training_service = core_engine.get_service(AgentTrainingService)

        # Extract final DataFrames from preprocessing results and split datasets
        split_datasets = []
        for preprocessing_result in final_datasets:
            # Extract the final DataFrame from the preprocessing result
            final_dataframe = preprocessing_result.final_result
            split_datasets.append(split_service.split_dataset(final_dataframe))

        # Create environments and train agents
        logger.info(
            f"Creating environments and training agents with {len(split_datasets)} datasets"
        )
        train_env, val_env, agents = agent_training_service.create_env_and_train_agents(
            split_datasets
        )
        return agents

    def start_training(
        self,
        config_path: Optional[str] = None,
    ) -> None:
        """
        Bootstraps the agent training process.

        Initializes the application, preprocesses data, creates training
        and validation environments, and trains agents.

        The configuration path is determined in the following order of precedence:
        1. The `config_path` parameter, if provided.
        2. The `DRL_TRADING_CONFIG_PATH` environment variable, if set.
        3. The default path hardcoded in the modern injector container.

        Args:
            env_class: The class of the trading environment to be used.
                    Must be a subclass of BaseTradingEnv.
            config_path: Optional path to the configuration file. If None, the system
                        will attempt to use DRL_TRADING_CONFIG_PATH or the default.
        """
        # Log the intention based on parameters and environment variables
        env_var_value = os.getenv("DRL_TRADING_CONFIG_PATH")
        if config_path:
            logger.info(
                f"bootstrap_agent_training called with explicit config_path: {config_path}"
            )
        elif env_var_value:
            logger.info(
                f"bootstrap_agent_training: config_path is None, DRL_TRADING_CONFIG_PATH is set to: {env_var_value}"
            )
        else:
            logger.info(
                "bootstrap_agent_training: config_path is None and DRL_TRADING_CONFIG_PATH is not set. Container will use its default path."
            )

        # Bootstrap application with DI - _preprocess returns (None, final_datasets)

        core_app = CoreEngine(strategy_module=ExampleStrategyModule())
        final_datasets = core_app._preprocess()

        # Create environments and train agents
        agents = self._create_environments_and_train(core_app, final_datasets)

        logger.info(f"Training completed for {len(agents)} agents")
        logger.info("Agent training bootstrap completed successfully")
