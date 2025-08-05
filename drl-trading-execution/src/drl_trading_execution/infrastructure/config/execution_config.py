"""Service-specific configuration for execution service."""
from pydantic import Field
from drl_trading_common.base.base_application_config import BaseApplicationConfig
from drl_trading_common.base.base_schema import BaseSchema
from drl_trading_common.config.infrastructure_config import InfrastructureConfig


class BrokerConfig(BaseSchema):
    """Broker connection and API configuration."""
    provider: str = "paper_trading"  # paper_trading | ftmo | mt5 | binance
    account_id: str | None = None
    api_key_env: str = "BROKER_API_KEY"
    api_secret_env: str = "BROKER_API_SECRET"
    base_url: str | None = None
    connection_timeout: int = 10  # seconds
    retry_attempts: int = 3


class RiskManagementConfig(BaseSchema):
    """Risk management and compliance configuration."""
    # FTMO compliance settings
    max_daily_loss_percent: float = 5.0
    max_total_loss_percent: float = 10.0
    max_positions: int = 5
    require_stop_loss: bool = True

    # Position sizing
    position_sizing_method: str = "fixed_risk"  # fixed_risk | fixed_lot | kelly
    risk_per_trade_percent: float = 1.0
    max_risk_per_trade_percent: float = 2.0

    # Stop-loss and take-profit configuration
    stop_loss_atr_multiplier: float = 1.5
    take_profit_atr_multiplier: float = 2.0
    use_trailing_stop: bool = True
    trailing_stop_activation_percent: float = 1.0
    trailing_stop_step_percent: float = 0.5


class ExecutionParametersConfig(BaseSchema):
    """Trade execution parameters configuration."""
    execution_mode: str = "market"  # market | limit
    slippage_tolerance_pips: int = 2
    execution_timeout_seconds: int = 5
    cancel_failed_orders: bool = True
    max_spread_pips: int = 10
    retry_execution_attempts: int = 2


class MessageRoutingConfig(BaseSchema):
    """Message routing configuration for execution service."""
    signal_topic: str = "trading_signals"
    execution_topic: str = "trade_executions"
    status_topic: str = "execution_status"
    error_topic: str = "execution_errors"


class ExecutionConfig(BaseApplicationConfig):
    """Configuration for trade execution service - focused on trade execution."""
    app_name: str = "drl-trading-execution"
    infrastructure: InfrastructureConfig = Field(default_factory=InfrastructureConfig)
    broker: BrokerConfig = Field(default_factory=BrokerConfig)
    risk_management: RiskManagementConfig = Field(default_factory=RiskManagementConfig)
    execution_params: ExecutionParametersConfig = Field(default_factory=ExecutionParametersConfig)
    message_routing: MessageRoutingConfig = Field(default_factory=MessageRoutingConfig)
