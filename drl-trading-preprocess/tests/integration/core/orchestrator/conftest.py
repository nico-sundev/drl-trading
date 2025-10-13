from typing import Generator, Optional
from datetime import datetime, timedelta

import pandas as pd
import pytest
from drl_trading_common.base.base_feature import BaseFeature
from drl_trading_common.base.base_parameter_set_config import BaseParameterSetConfig
from drl_trading_common.decorator.feature_role_decorator import feature_role
from drl_trading_common.enum.feature_role_enum import FeatureRoleEnum
from drl_trading_common.interface.indicator.technical_indicator_facade_interface import (
    ITechnicalIndicatorFacade,
)
from drl_trading_common.interface.feature.feature_factory_interface import IFeatureFactory
from drl_trading_common.model.dataset_identifier import DatasetIdentifier
from injector import Injector
from pandas import DataFrame

from drl_trading_adapter.adapter.database.entity.market_data_entity import Base, MarketDataEntity
from drl_trading_adapter.adapter.database.session_factory import SQLAlchemySessionFactory
from drl_trading_common.config.infrastructure_config import DatabaseConfig
from drl_trading_common.model.timeframe import Timeframe
from drl_trading_preprocess.adapter.messaging.publisher.stub_preprocess_message_publisher import (
    StubPreprocessingMessagePublisher,
)


# Test Feature Implementations for Core Integration Testing
class MockTechnicalIndicatorFacade(ITechnicalIndicatorFacade):
    """Mock technical indicator facade that returns controlled test data for integration testing."""

    def __init__(self) -> None:
        self._indicators: dict[str, DataFrame] = {}

    def register_instance(self, name: str, indicator_type: str, **params: int) -> None:
        """Register a mock indicator that returns predictable test data."""
        # Create predictable test data based on indicator type with UTC timezone
        dates = pd.date_range(start="2024-01-01", periods=100, freq="1h", tz="UTC")  # Use lowercase 'h' as 'H' is deprecated

        if "rsi" in name.lower():
            # Generate RSI-like values between 30-70
            values = [50.0 + (i % 20) for i in range(100)]
            self._indicators[name] = pd.DataFrame(
                {"event_timestamp": dates, name: values}
            )
        elif "close_price" in name.lower():
            # Generate price-like values
            values = [1.1000 + (i % 50) * 0.0001 for i in range(100)]
            self._indicators[name] = pd.DataFrame(
                {"event_timestamp": dates, name: values}
            )
        else:
            # Default values
            values = [float(i) for i in range(100)]
            self._indicators[name] = pd.DataFrame(
                {"event_timestamp": dates, name: values}
            )

    def add(self, name: str, value: DataFrame) -> None:
        """Mock incremental computation - not needed for current tests."""
        pass

    def get_all(self, name: str) -> Optional[DataFrame]:
        """Return all mock indicator data."""
        return self._indicators.get(name)

    def get_latest(self, name: str) -> Optional[DataFrame]:
        """Return latest mock indicator data."""
        data = self._indicators.get(name)
        return data.tail(1) if data is not None else None


class TestRsiConfig(BaseParameterSetConfig):
    """Test configuration for RSI feature that follows the existing pattern."""

    type: str = "rsi"
    enabled: bool = True
    period: int = 14

    def hash_id(self) -> str:
        return "A1b2c3"


@feature_role(FeatureRoleEnum.OBSERVATION_SPACE)
class TestRsiFeature(BaseFeature):
    """Test RSI feature implementation adapted from existing MockFeature pattern."""

    def __init__(
        self,
        dataset_id: DatasetIdentifier,
        indicator_service: MockTechnicalIndicatorFacade,
        config: TestRsiConfig,
        postfix: str = "",
    ):
        super().__init__(dataset_id, indicator_service, config, postfix)
        self._feature_name = "rsi"
        # Register the indicator when feature is created
        self.indicator_service.register_instance(
            f"rsi_{config.period}", "rsi", period=config.period
        )

    def get_feature_name(self) -> str:
        return self._feature_name

    def get_sub_features_names(self) -> list[str]:
        return []

    def compute_all(self) -> Optional[DataFrame]:
        """Compute RSI using the mock indicator service."""
        # Assert config is TestRsiConfig to satisfy type checker
        assert isinstance(self.config, TestRsiConfig), "Config must be TestRsiConfig"

        indicator_name = f"rsi_{self.config.period}"
        indicator_data = self.indicator_service.get_all(indicator_name)

        if indicator_data is None:
            return None

        # Add the required symbol column for Feast compatibility
        result = indicator_data.copy()
        result[self.dataset_id.symbol] = self.dataset_id.symbol
        return result

    def update(self, df: DataFrame) -> None:
        """Mock incremental computation - not implemented for current testing."""
        pass

    def compute_latest(self) -> Optional[DataFrame]:
        """Get latest RSI values."""
        # Assert config is TestRsiConfig to satisfy type checker
        assert isinstance(self.config, TestRsiConfig), "Config must be TestRsiConfig"

        indicator_name = f"rsi_{self.config.period}"
        indicator_data = self.indicator_service.get_latest(indicator_name)

        if indicator_data is None:
            return None

        result = indicator_data.copy()
        result[self.dataset_id.symbol] = self.dataset_id.symbol
        return result

    def get_config_to_string(self) -> str:
        # Assert config is TestRsiConfig to satisfy type checker
        assert isinstance(self.config, TestRsiConfig), "Config must be TestRsiConfig"
        return f"{self.config.period}"


@feature_role(FeatureRoleEnum.OBSERVATION_SPACE)
class TestClosePriceFeature(BaseFeature):
    """Test close price feature implementation adapted from existing MockFeature pattern."""

    def __init__(
        self,
        dataset_id: DatasetIdentifier,
        indicator_service: MockTechnicalIndicatorFacade,
        config: Optional[BaseParameterSetConfig] = None,
        postfix: str = "",
    ):
        super().__init__(dataset_id, indicator_service, config, postfix)
        self._feature_name = "close_price"

    def get_feature_name(self) -> str:
        return self._feature_name

    def get_sub_features_names(self) -> list[str]:
        return []

    def compute_all(self) -> Optional[DataFrame]:
        """Compute close prices using the mock indicator service."""
        indicator_name = "close_price"
        indicator_data = self.indicator_service.get_all(indicator_name)

        if indicator_data is None:
            return None

        # Add the required symbol column for Feast compatibility
        result = indicator_data.copy()
        result[self.dataset_id.symbol] = self.dataset_id.symbol
        return result

    def update(self, df: DataFrame) -> None:
        """Mock incremental computation - not implemented for current testing."""
        pass

    def compute_latest(self) -> Optional[DataFrame]:
        """Get latest close price values."""
        indicator_name = "close_price"
        indicator_data = self.indicator_service.get_latest(indicator_name)

        if indicator_data is None:
            return None

        result = indicator_data.copy()
        result[self.dataset_id.symbol] = self.dataset_id.symbol
        return result

    def get_config_to_string(self) -> Optional[str]:
        return None

class TestFeatureFactory(IFeatureFactory):
    """Test feature factory that creates test feature instances."""

    def __init__(self, indicator_facade: MockTechnicalIndicatorFacade):
        """Initialize factory with indicator facade.

        Args:
            indicator_facade: Mock indicator facade for features
        """
        self.indicator_facade = indicator_facade

    def create_feature(
        self,
        feature_name: str,
        dataset_id: DatasetIdentifier,
        config: Optional[BaseParameterSetConfig] = None,
        postfix: str = "",
    ) -> Optional[BaseFeature]:
        """Create a test feature instance.

        Args:
            feature_name: Name of feature to create ("rsi" or "close_price")
            dataset_id: Dataset identifier
            config: Feature configuration
            postfix: Optional postfix

        Returns:
            Created feature instance or None if unknown feature
        """
        if feature_name == "rsi":
            if config is None or not isinstance(config, TestRsiConfig):
                # Create default config if none provided
                config = TestRsiConfig(type="rsi", enabled=True, period=14)
            return TestRsiFeature(dataset_id, self.indicator_facade, config, postfix)
        elif feature_name == "close_price":
            return TestClosePriceFeature(dataset_id, self.indicator_facade, config, postfix)
        else:
            return None

    def create_config_instance(
        self, feature_name: str, config_data: dict
    ) -> Optional[BaseParameterSetConfig]:
        """Create a configuration instance for a feature.

        Args:
            feature_name: Name of feature
            config_data: Configuration data dictionary

        Returns:
            Configuration instance or None if unknown feature
        """
        if feature_name == "rsi":
            return TestRsiConfig(**config_data)
        elif feature_name == "close_price":
            # ClosePrice has no config
            return None
        else:
            return None

    def is_feature_supported(self, feature_name: str) -> bool:
        """Check if a feature is supported by this factory.

        Args:
            feature_name: Name of feature to check

        Returns:
            True if feature is supported, False otherwise
        """
        return feature_name in ("rsi", "close_price")


@pytest.fixture(scope="function")
def integration_container(
    real_feast_container: Injector,
    clean_integration_environment: None
) -> Generator[Injector, None, None]:
    """Provide container with clean integration environment for each test.

    This ensures each test gets a fresh, clean state while using real
    services created through dependency injection.

    Args:
        real_feast_container: DI container with real services
        clean_integration_environment: Clean environment setup

    Returns:
        Injector: Ready-to-use DI container for integration testing
    """
    yield real_feast_container

@pytest.fixture(scope="function")
def sample_trading_features_df() -> DataFrame:
    """Create comprehensive sample trading features DataFrame for integration testing.

    This fixture provides a realistic set of trading features with proper column names
    and data types that match what the feature store repositories expect.
    """
    # Create realistic time series data with UTC timezone to avoid Feast timezone issues
    timestamps = pd.date_range(
        start="2024-01-01 09:00:00",
        periods=50,
        freq="h",  # Use lowercase 'h' as 'H' is deprecated in newer pandas versions
        tz="UTC",  # Add UTC timezone to match Feast expectations
    )

    # Generate realistic trading feature data
    return DataFrame(
        {
            "event_timestamp": timestamps,
            "symbol": ["EURUSD"] * len(timestamps),
            # Technical indicators - observation space features (match sub-feature names)
            "rsi_14_A1b2c3": [30.0 + (i % 40) + (i * 0.5) for i in range(len(timestamps))],
            # OHLCV data - reward space features (match sub-feature names)
            "close_price": [
                1.0850 + (i % 20) * 0.0001 for i in range(len(timestamps))
            ],
        }
    )


@pytest.fixture(scope="function")
def session_factory(database_config: DatabaseConfig) -> SQLAlchemySessionFactory:
    """Create SQLAlchemy session factory connected to test database.

    Args:
        database_config: Database configuration from testcontainer

    Returns:
        SQLAlchemySessionFactory: Factory for creating database sessions
    """
    return SQLAlchemySessionFactory(database_config)


@pytest.fixture(scope="function", autouse=True)
def setup_database_schema(session_factory: SQLAlchemySessionFactory) -> Generator[None, None, None]:
    """Create database schema before each test.

    This fixture:
    1. Creates all tables defined in Base metadata
    2. Yields control to the test
    3. Cleanup is handled by container disposal

    Args:
        session_factory: Session factory connected to test database
    """
    Base.metadata.create_all(session_factory._engine)
    yield
    # No explicit cleanup needed - container disposal handles it


@pytest.fixture(scope="function")
def populated_market_data(
    session_factory: SQLAlchemySessionFactory,
) -> list[MarketDataEntity]:
    """Populate test database with realistic market data.

    Creates 50 hourly EURUSD bars starting from 2024-01-01 09:00:00 UTC.
    This matches the sample_trading_features_df fixture for consistency.

    Args:
        session_factory: Session factory for database access

    Returns:
        List of created MarketDataEntity objects for test validation
    """
    entities = []
    base_time = datetime(2024, 1, 1, 9, 0, 0)
    base_price = 1.0850

    # Create 50 hourly bars for EURUSD
    for i in range(50):
        timestamp = base_time + timedelta(hours=i)
        price_variation = (i % 20) * 0.0001

        entity = MarketDataEntity(
            symbol="EURUSD",
            timeframe=Timeframe.HOUR_1.value,
            timestamp=timestamp,
            open_price=base_price + price_variation,
            high_price=base_price + price_variation + 0.0005,
            low_price=base_price + price_variation - 0.0003,
            close_price=base_price + price_variation,
            volume=1000 + (i * 10),
        )
        entities.append(entity)

    # Insert into database
    with session_factory.get_session() as session:
        session.add_all(entities)
        session.commit()

    return entities


@pytest.fixture(scope="function")
def spy_message_publisher() -> StubPreprocessingMessagePublisher:
    """Create a spy message publisher to verify notification calls.

    Returns a real StubPreprocessingMessagePublisher instance that we can
    inspect to verify the orchestrator publishes correct notifications.

    Returns:
        StubPreprocessingMessagePublisher: Publisher with inspection capabilities
    """
    return StubPreprocessingMessagePublisher(log_level="DEBUG")


@pytest.fixture(scope="function")
def mock_indicator_facade() -> MockTechnicalIndicatorFacade:
    """Create mock technical indicator facade.

    Returns:
        MockTechnicalIndicatorFacade: Mock facade for test features
    """
    facade = MockTechnicalIndicatorFacade()
    # Pre-register indicators that test features will use
    facade.register_instance("rsi_14", "rsi", period=14)
    facade.register_instance("close_price", "close_price")
    return facade


@pytest.fixture(scope="function")
def test_feature_factory(mock_indicator_facade: MockTechnicalIndicatorFacade) -> TestFeatureFactory:
    """Create test feature factory.

    Args:
        mock_indicator_facade: Mock indicator facade

    Returns:
        TestFeatureFactory: Factory for creating test features
    """
    return TestFeatureFactory(mock_indicator_facade)
