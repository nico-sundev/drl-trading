"""Service-specific configuration for execution service."""
from typing import Any, Dict

from drl_trading_common.base.base_application_config import BaseApplicationConfig


class ExecutionConfig(BaseApplicationConfig):
    """Configuration for trade execution service."""
    # Broker configuration
    broker_config: Dict[str, Any] = {
        "provider": "paper_trading",  # paper_trading, ftmo, mt5, binance
        "account_id": None,
        "api_key_env": "BROKER_API_KEY",
        "api_secret_env": "BROKER_API_SECRET",
        "base_url": None,  # Used for custom broker APIs
        "connection_timeout": 10,  # seconds
        "retry_attempts": 3
    }

    # Risk management configuration
    risk_management: Dict[str, Any] = {
        # FTMO compliance settings
        "max_daily_loss_percent": 5.0,
        "max_total_loss_percent": 10.0,
        "max_positions": 5,
        "require_stop_loss": True,

        # Position sizing
        "position_sizing_method": "fixed_risk",  # fixed_risk, fixed_lot, kelly
        "risk_per_trade_percent": 1.0,
        "max_risk_per_trade_percent": 2.0,

        # Stop-loss and take-profit configuration
        "stop_loss_atr_multiplier": 1.5,
        "take_profit_atr_multiplier": 2.0,
        "use_trailing_stop": True,
        "trailing_stop_activation_percent": 1.0,
        "trailing_stop_step_percent": 0.5,
    }

    # Execution parameters
    execution_params: Dict[str, Any] = {
        "execution_mode": "market",  # market, limit
        "slippage_tolerance_pips": 2,
        "execution_timeout_seconds": 5,
        "cancel_failed_orders": True,
        "max_spread_pips": 10,  # Maximum spread to execute trades
        "retry_execution_attempts": 2
    }

    # Messaging configuration
    message_routing: Dict[str, str] = {
        "signal_topic": "trading_signals",
        "execution_topic": "trade_executions",
        "status_topic": "execution_status",
        "error_topic": "execution_errors"
    }
