"""
Configuration for DRL Trading Execution Service.

Provides configuration settings for order execution, risk management,
and broker connectivity.
"""

from dataclasses import dataclass

from drl_trading_common.base.base_application_config import BaseApplicationConfig


@dataclass
class ExecutionConfig(BaseApplicationConfig):
    """
    Configuration for the execution service.

    Extends BaseApplicationConfig with execution-specific settings.
    """

    # Order Management Configuration
    max_orders_per_second: int = 10
    order_timeout_seconds: int = 30

    # Risk Management Configuration
    max_position_size: float = 100000.0
    daily_loss_limit: float = 10000.0

    # Broker Configuration
    broker_api_timeout: int = 5
    reconnect_attempts: int = 3

    # Performance Configuration
    latency_threshold_ms: int = 100

    def __post_init__(self) -> None:
        """Post-initialization validation."""
        super().__post_init__()

        # Validate execution-specific settings
        if self.max_orders_per_second <= 0:
            raise ValueError("max_orders_per_second must be positive")

        if self.order_timeout_seconds <= 0:
            raise ValueError("order_timeout_seconds must be positive")

        if self.max_position_size <= 0:
            raise ValueError("max_position_size must be positive")

        if self.daily_loss_limit <= 0:
            raise ValueError("daily_loss_limit must be positive")
