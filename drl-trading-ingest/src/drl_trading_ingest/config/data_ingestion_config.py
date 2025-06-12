"""Service-specific configuration for data ingestion service."""
from drl_trading_common.base.base_application_config import BaseApplicationConfig
from drl_trading_common.config.infrastructure_config import InfrastructureConfig


class DataIngestionConfig(BaseApplicationConfig):
    """Configuration for data ingestion service - only what it needs."""
    infrastructure: InfrastructureConfig

    # Data source configuration
    data_sources: dict = {
        "mt5": {
            "enabled": True,
            "symbols": ["EURUSD", "GBPUSD"],
            "timeframes": ["M1", "M5", "H1"],
            "max_bars": 10000
        },
        "binance": {
            "enabled": False,
            "api_key_env": "BINANCE_API_KEY",
            "secret_key_env": "BINANCE_SECRET_KEY"
        }
    }

    # Message bus configuration (only what ingestion needs)
    message_routing: dict = {
        "market_data_topic": "market_data",
        "heartbeat_interval": 30,
        "batch_size": 100
    }
