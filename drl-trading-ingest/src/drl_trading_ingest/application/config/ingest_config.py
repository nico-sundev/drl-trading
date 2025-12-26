"""T004-compliant configuration for ingest service."""
from typing import List
from pydantic import BaseModel, Field

from drl_trading_common.base.base_application_config import BaseApplicationConfig
from drl_trading_common.config.infrastructure_config import InfrastructureConfig
from drl_trading_common.config.service_logging_config import ServiceLoggingConfig


class BinanceProviderConfig(BaseModel):
    """Configuration for Binance data provider."""

    enabled: bool = Field(default=False, description="Enable Binance provider")
    api_key_env: str = Field(default="BINANCE_API_KEY", description="Environment variable for API key")
    secret_key_env: str = Field(default="BINANCE_SECRET_KEY", description="Environment variable for secret key")
    base_url: str = Field(default="https://api.binance.com", description="Base URL for Binance API")
    testnet: bool = Field(default=False, description="Use testnet instead of production")

    # Data configuration
    symbols: List[str] = Field(
        default_factory=lambda: ["BTCUSDT", "ETHUSDT", "BNBUSDT"],
        description="Symbols to fetch data for"
    )
    timeframes: List[str] = Field(
        default_factory=lambda: ["1m", "5m", "1h", "1d"],
        description="Timeframes/intervals for data"
    )
    max_bars: int = Field(default=1000, description="Maximum bars per request")

    # Streaming configuration
    websocket_url: str = Field(default="wss://stream.binance.com:9443", description="WebSocket URL for streaming")
    reconnect_attempts: int = Field(default=5, description="Max reconnection attempts")
    ping_interval: int = Field(default=30, description="WebSocket ping interval in seconds")


class TwelveDataProviderConfig(BaseModel):
    """Configuration for Twelve Data provider."""

    enabled: bool = Field(default=False, description="Enable Twelve Data provider")
    api_key_env: str = Field(default="TWELVE_DATA_API_KEY", description="Environment variable for API key")
    base_url: str = Field(default="https://api.twelvedata.com", description="Base URL for Twelve Data API")

    # Data configuration
    symbols: List[str] = Field(
        default_factory=lambda: ["AAPL", "GOOGL", "MSFT", "SPY"],
        description="Symbols to fetch data for"
    )
    intervals: List[str] = Field(
        default_factory=lambda: ["1min", "5min", "1h", "1day"],
        description="Intervals for data"
    )
    outputsize: int = Field(default=5000, description="Number of data points to return")

    # Data types
    data_types: List[str] = Field(
        default_factory=lambda: ["stocks", "forex", "crypto"],
        description="Types of market data to fetch"
    )

    # Streaming configuration
    websocket_url: str = Field(default="wss://ws.twelvedata.com/v1", description="WebSocket URL for streaming")
    price_stream: bool = Field(default=True, description="Enable price streaming")
    quote_stream: bool = Field(default=False, description="Enable quote streaming")


class CsvProviderConfig(BaseModel):
    """Configuration for CSV file data provider."""

    enabled: bool = Field(default=True, description="Enable CSV provider")
    base_path: str = Field(default="data/csv", description="Base path for CSV files")
    symbols: List[str] = Field(
        default_factory=lambda: ["EURUSD", "GBPUSD"],
        description="Symbols to load from CSV"
    )
    file_pattern: str = Field(default="{symbol}_{timeframe}.csv", description="File naming pattern")


class YahooProviderConfig(BaseModel):
    """Configuration for Yahoo Finance data provider."""

    enabled: bool = Field(default=False, description="Enable Yahoo Finance provider")
    symbols: List[str] = Field(
        default_factory=lambda: ["SPY", "QQQ", "IWM", "^GSPC"],
        description="Symbols to fetch data for"
    )
    intervals: List[str] = Field(
        default_factory=lambda: ["1m", "1h", "1d"],
        description="Intervals for data"
    )
    max_period: str = Field(default="1y", description="Maximum time period to fetch")


class DataSourceConfig(BaseModel):
    """Configuration for all data source providers."""

    # Provider configurations
    binance: BinanceProviderConfig = Field(default_factory=BinanceProviderConfig)
    twelve_data: TwelveDataProviderConfig = Field(default_factory=TwelveDataProviderConfig)
    csv: CsvProviderConfig = Field(default_factory=CsvProviderConfig)
    yahoo: YahooProviderConfig = Field(default_factory=YahooProviderConfig)

    # Global settings
    default_provider: str = Field(default="csv", description="Default provider to use")
    enable_multiple_providers: bool = Field(default=False, description="Allow using multiple providers simultaneously")


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
    logging: ServiceLoggingConfig = Field(default_factory=ServiceLoggingConfig)
    data_source: DataSourceConfig = Field(default_factory=DataSourceConfig)
    message_routing: MessageRoutingConfig = Field(default_factory=MessageRoutingConfig)
    data_validation: DataValidationConfig = Field(default_factory=DataValidationConfig)
