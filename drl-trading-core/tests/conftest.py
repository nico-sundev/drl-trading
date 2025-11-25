import os
from typing import Literal, Optional, Type
from unittest.mock import MagicMock

from drl_trading_common.interface.feature.feature_factory_interface import IFeatureFactory
import pytest
from drl_trading_common import BaseParameterSetConfig
from drl_trading_common.core.model.base_feature import BaseFeature
from drl_trading_common.interface.indicator.technical_indicator_facade_interface import (
    ITechnicalIndicatorFacade,
)
from injector import Injector
from pandas import DataFrame

from drl_trading_core.infrastructure.di import CoreModule
import numpy as np

class RsiConfig(BaseParameterSetConfig):
    """RSI configuration for testing."""

    type: Literal["rsi"] = "rsi"
    length: int = 14

    def __init__(self, enabled=True, length=14, **kwargs):
        super().__init__(type="rsi", enabled=enabled)
        self.length = length
        self._param_string = str(length)

    @property
    def param_string(self) -> str:
        return self._param_string

    def hash_id(self) -> str:
        return f"rsi_{self.length}"


class RsiFeature(BaseFeature):
    """Mock RSI feature implementation for testing."""

    def __init__(
        self,
        config: BaseParameterSetConfig,
        indicator_service: ITechnicalIndicatorFacade,
        postfix: str = "",
    ) -> None:
        super().__init__(config, indicator_service, postfix)
        self.config: RsiConfig = self.config
        self.feature_name = f"rsi_{self.config.length}{self.postfix}"
        # Mock the indicator service registration for testing

    def update(self, df: DataFrame) -> None:
        """Add data to the feature (mock implementation)."""
        pass

    def compute_latest(self) -> Optional[DataFrame]:
        """Compute latest RSI value (mock implementation)."""
        if hasattr(self, 'df_source') and not self.df_source.empty:
            latest_rsi = np.random.uniform(low=0, high=100, size=1)
            result_df = DataFrame(index=self.df_source.index[-1:])
            result_df[f"rsi_{self.config.length}{self.postfix}"] = latest_rsi
            return result_df
        return None

    def compute_all(self) -> Optional[DataFrame]:
        """Compute all RSI values (mock implementation)."""
        if hasattr(self, 'df_source') and not self.df_source.empty:
            rsi_values = np.random.uniform(low=0, high=100, size=len(self.df_source))
            result_df = DataFrame(index=self.df_source.index)
            result_df[f"rsi_{self.config.length}{self.postfix}"] = rsi_values
            return result_df
        return None

    def _get_sub_features_names(self) -> list[str]:
        """Get sub-feature names."""
        return [f"rsi_{self.config.length}{self.postfix}"]

    def _get_feature_name(self) -> str:
        """Get feature name."""
        return "rsi"

    def _get_config_to_string(self) -> str:
        return f"{self.config.length}"

@pytest.fixture(scope="session")
def mock_rsi_config_class() -> Type[RsiConfig]:
    """Mock RSI configuration class for testing."""
    return RsiConfig

@pytest.fixture(scope="session")
def feature_factory():
    """Create a feature factory mock instance for testing."""
    mock_factory = MagicMock(spec=IFeatureFactory)
    mock_factory.create_feature = MagicMock(
        side_effect=lambda feature_type, source_data, config, indicator_service, postfix="": {
            "rsi": RsiFeature(config, indicator_service, postfix),
        }.get(feature_type.lower(), None)
    )
    mock_factory.create_config_instance = MagicMock(
        side_effect=lambda feature_type, config_data: {
            "rsi": RsiConfig(**config_data) if config_data else RsiConfig(),
        }.get(feature_type.lower(), None)
    )
    return mock_factory


@pytest.fixture(scope="session")
def mocked_container(feature_factory):
    """Create a mocked dependency injection container using the injector library.

    This fixture provides a configured injector instance with test dependencies
    for integration testing, replacing the legacy dependency-injector approach.
    """
    # Use the test config path
    test_config_path = os.path.join(
        os.path.dirname(__file__), "resources/applicationConfig-test.json"
    )

    app_module = CoreModule(config_path=test_config_path)
    injector = Injector([app_module])

    # Override the feature factory with our test fixture
    injector.binder.bind(IFeatureFactory, to=feature_factory)

    return injector
