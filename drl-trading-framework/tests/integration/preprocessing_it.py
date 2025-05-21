import os
from typing import Literal, Optional

import pandas_ta as ta
import pytest
from pandas import DataFrame, DatetimeIndex

from drl_trading_framework.common.config.base_parameter_set_config import (
    BaseParameterSetConfig,
)
from drl_trading_framework.common.config.config_loader import ConfigLoader
from drl_trading_framework.common.config.feature_config_factory import (
    FeatureConfigFactory,
)
from drl_trading_framework.common.data_import.data_import_manager import (
    DataImportManager,
)
from drl_trading_framework.common.data_import.local.csv_data_import_service import (
    CsvDataImportService,
)
from drl_trading_framework.common.model.preprocessing_result import PreprocessingResult
from drl_trading_framework.common.model.symbol_import_container import (
    SymbolImportContainer,
)
from drl_trading_framework.preprocess.data_set_utils.context_feature_service import (
    ContextFeatureService,
)
from drl_trading_framework.preprocess.data_set_utils.merge_service import MergeService
from drl_trading_framework.preprocess.feature.collection.base_feature import BaseFeature
from drl_trading_framework.preprocess.feature.feature_aggregator import (
    FeatureAggregator,
)
from drl_trading_framework.preprocess.feature.feature_class_registry import (
    FeatureClassRegistry,
)
from drl_trading_framework.preprocess.metrics.technical_metrics_service import (
    TechnicalMetricsServiceInterface,
)
from drl_trading_framework.preprocess.preprocess_service import PreprocessService


class RsiConfig(BaseParameterSetConfig):
    type: Literal["rsi"]
    length: int


class RsiFeature(BaseFeature):

    def __init__(
        self,
        source: DataFrame,
        config: BaseParameterSetConfig,
        postfix: str = "",
        metrics_service: Optional[TechnicalMetricsServiceInterface] = None,
    ) -> None:
        super().__init__(source, config, postfix, metrics_service)
        self.config: RsiConfig = self.config

    def compute(self) -> DataFrame:
        # Get source DataFrame with ensured DatetimeIndex using the base class method
        source_df = self._prepare_source_df()

        # Create a DataFrame with the same index as the source
        rsi_values = ta.rsi(source_df["Close"], length=self.config.length)

        # Create result DataFrame with both Time column and feature values
        df = DataFrame(index=source_df.index)
        df[f"rsi_{self.config.length}{self.postfix}"] = rsi_values

        return df

    def get_sub_features_names(self) -> list[str]:
        return [f"rsi_{self.config.length}{self.postfix}"]


@pytest.fixture
def feature_config_factory():
    """Create a fresh feature config factory instance for testing."""
    factory = FeatureConfigFactory()
    factory.discover_config_classes(package_name="tests.integration")
    return factory


@pytest.fixture
def config(feature_config_factory):
    config = ConfigLoader.get_config(
        os.path.join(
            os.path.dirname(__file__), "../resources/applicationConfig-test.json"
        )
    )
    # To speed up the test, only use RSI
    config.features_config.feature_definitions = [
        f for f in config.features_config.feature_definitions if f.name == "rsi"
    ]

    # Parse parameters using the factory
    config.features_config.parse_all_parameters(feature_config_factory)

    return config


@pytest.fixture
def symbol_container(config) -> SymbolImportContainer:
    # Create a service with the complete config
    repository = CsvDataImportService(config.local_data_import_config)
    importer = DataImportManager(repository)

    # Get all symbol containers
    return importer.get_data()[0]


@pytest.fixture
def class_registry():
    reg = FeatureClassRegistry(package_name="tests.integration")
    return reg


@pytest.fixture
def feast_service(config):
    """Create a real FeastService for testing."""
    from drl_trading_framework.preprocess.feast.feast_service import FeastService

    return FeastService(config=config.feature_store_config)


@pytest.fixture
def feature_aggregator(config, class_registry, feast_service) -> FeatureAggregator:
    return FeatureAggregator(config.features_config, class_registry, feast_service)


@pytest.fixture
def merge_service():
    return MergeService()


@pytest.fixture
def context_service(config):
    return ContextFeatureService(config.context_feature_config)


@pytest.fixture
def preprocess_service(
    config, class_registry, feature_aggregator, merge_service, context_service
):
    return PreprocessService(
        features_config=config.features_config,
        feature_class_registry=class_registry,
        feature_aggregator=feature_aggregator,
        merge_service=merge_service,
        context_feature_service=context_service,
    )


def test_preprocessing(
    preprocess_service: PreprocessService, symbol_container: SymbolImportContainer
):
    """Test that preprocessing creates the expected feature columns."""
    # Given
    # Time is now in the index, not a column
    expected_context_related_columns = ["High", "Low", "Close", "Atr"]

    expected_feature_columns = [
        "rsi_7",
        "HTF-240_rsi_7",
    ]

    all_expected_columns = sorted(
        expected_context_related_columns + expected_feature_columns
    )

    # When
    preproc_result = preprocess_service.preprocess_data(symbol_container)
    # Should return a PreprocessingResult object
    assert isinstance(
        preproc_result, PreprocessingResult
    ), f"Expected PreprocessingResult, got {type(preproc_result)}"
    df = preproc_result.final_result
    actual_columns = sorted(set(df.columns))

    # Then
    assert (
        actual_columns == all_expected_columns
    ), f"Column mismatch! Expected: {all_expected_columns}, but got: {actual_columns}"

    # Verify that we have a DatetimeIndex on the final result
    assert isinstance(df.index, DatetimeIndex), "Result should have a DatetimeIndex"

    # print(feature_df_merged.head())
