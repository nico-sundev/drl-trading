"""RL training environment DI module - provides gym environment for training services only."""

import logging
from typing import Type

from injector import Module, provider, singleton

from drl_trading_common.base.base_trading_env import BaseTradingEnv
from drl_trading_strategy_example.gym_env.custom_env import MyCustomTradingEnv

logger = logging.getLogger(__name__)


class RLEnvironmentModule(Module):
    """DI module for RL training environment (training services only)."""

    @provider
    @singleton
    def provide_trading_environment_class(self) -> Type[BaseTradingEnv]:
        """Provide custom trading environment class for RL training."""
        logger.info("RLEnvironmentModule: Providing MyCustomTradingEnv")
        return MyCustomTradingEnv
