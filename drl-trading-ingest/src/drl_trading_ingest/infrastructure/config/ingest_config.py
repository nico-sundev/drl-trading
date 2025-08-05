"""T004-compliant configuration for ingest service."""
from typing import List
from pydantic import BaseModel, Field

from drl_trading_common.base.base_application_config import BaseApplicationConfig
from drl_trading_common.config.infrastructure_config import InfrastructureConfig


class DataSourceConfig(BaseModel):
    """Configuration for data source providers."""

    # MT5 configuration
    mt5_enabled: bool = Field(default=True)
    mt5_symbols: List[str] = Field(default_factory=lambda: ["EURUSD", "GBPUSD", "USDJPY"])
    mt5_timeframes: List[str] = Field(default_factory=lambda: ["M1", "M5", "H1", "D1"])
    mt5_max_bars: int = Field(default=10000)

    # Binance configuration
    binance_enabled: bool = Field(default=False)
    binance_api_key_env: str = Field(default="BINANCE_API_KEY")
    binance_secret_key_env: str = Field(default="BINANCE_SECRET_KEY")
    binance_symbols: List[str] = Field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])

    # Yahoo Finance configuration
    yahoo_enabled: bool = Field(default=False)
    yahoo_symbols: List[str] = Field(default_factory=lambda: ["SPY", "QQQ", "IWM"])


class MessageRoutingConfig(BaseModel):
    """Configuration for message routing and publishing."""

    market_data_topic: str = Field(default="market_data")
    heartbeat_topic: str = Field(default="heartbeat")
    error_topic: str = Field(default="ingest_errors")

    heartbeat_interval: int = Field(default=30)  # seconds
    batch_size: int = Field(default=100)
    max_retry_attempts: int = Field(default=3)
    retry_delay: int = Field(default=5)  # seconds


class DataValidationConfig(BaseModel):
    """Configuration for data validation and quality checks."""

    enable_validation: bool = Field(default=True)
    max_price_deviation: float = Field(default=0.1)  # 10% max price change
    min_volume_threshold: int = Field(default=0)
    max_age_seconds: int = Field(default=300)  # 5 minutes max age

    # Outlier detection
    enable_outlier_detection: bool = Field(default=True)
    outlier_threshold: float = Field(default=3.0)  # standard deviations


class IngestConfig(BaseApplicationConfig):
    """T004-compliant configuration for ingest service."""

    infrastructure: InfrastructureConfig
    data_source: DataSourceConfig = Field(default_factory=DataSourceConfig)
    message_routing: MessageRoutingConfig = Field(default_factory=MessageRoutingConfig)
    data_validation: DataValidationConfig = Field(default_factory=DataValidationConfig)
