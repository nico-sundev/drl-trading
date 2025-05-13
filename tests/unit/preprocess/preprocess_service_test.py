from typing import Dict, List
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from pandas import DataFrame

from drl_trading_framework.common.config.feature_config import FeaturesConfig
from drl_trading_framework.common.model.asset_price_dataset import AssetPriceDataSet
from drl_trading_framework.common.model.symbol_import_container import (
    SymbolImportContainer,
)
from drl_trading_framework.preprocess.data_set_utils.merge_service import MergeService
from drl_trading_framework.preprocess.feast.feast_service import FeastService
from drl_trading_framework.preprocess.feature.feature_aggregator import (
    FeatureAggregator,
)
from drl_trading_framework.preprocess.feature.feature_class_registry import (
    FeatureClassRegistry,
)
from drl_trading_framework.preprocess.preprocess_service import PreprocessService


@pytest.fixture
def mock_feature_config() -> FeaturesConfig:
    """Create a mock feature config."""
    mock_config = MagicMock(spec=FeaturesConfig)
    # Add the feature_definitions attribute that is used in FeatureAggregator.compute()
    mock_config.feature_definitions = []  # Empty list to avoid iterations
    return mock_config


@pytest.fixture
def mock_feature_class_registry() -> FeatureClassRegistry:
    """Create a mock feature class registry."""
    return MagicMock(spec=FeatureClassRegistry)


@pytest.fixture
def mock_feast_service() -> FeastService:
    """Create a mock FeastService."""
    return MagicMock(spec=FeastService)


@pytest.fixture
def mock_feature_aggregator() -> MagicMock:
    """Create a mock FeatureAggregator."""
    mock = MagicMock(spec=FeatureAggregator)
    # Mock the compute method to return a list of delayed tasks
    mock.compute.return_value = [MagicMock()]
    return mock


@pytest.fixture
def mock_base_dataset() -> AssetPriceDataSet:
    """Create a mock base dataset."""
    df = pd.DataFrame(
        {
            "Time": pd.date_range(start="2023-01-01", periods=10, freq="H"),
            "Open": range(10, 20),
            "High": range(20, 30),
            "Low": range(5, 15),
            "Close": range(15, 25),
            "Volume": range(1000, 1010),
        }
    )

    dataset = MagicMock(spec=AssetPriceDataSet)
    dataset.timeframe = "H1"
    dataset.base_dataset = True
    dataset.symbol = "EURUSD"
    dataset.asset_price_dataset = df
    return dataset


@pytest.fixture
def mock_other_dataset() -> AssetPriceDataSet:
    """Create a mock higher timeframe dataset."""
    df = pd.DataFrame(
        {
            "Time": pd.date_range(start="2023-01-01", periods=5, freq="4H"),
            "Open": range(10, 15),
            "High": range(20, 25),
            "Low": range(5, 10),
            "Close": range(15, 20),
            "Volume": range(1000, 1005),
        }
    )

    dataset = MagicMock(spec=AssetPriceDataSet)
    dataset.timeframe = "H4"
    dataset.base_dataset = False
    dataset.symbol = "EURUSD"
    dataset.asset_price_dataset = df
    return dataset


@pytest.fixture
def mock_datasets(
    mock_base_dataset: AssetPriceDataSet, mock_other_dataset: AssetPriceDataSet
) -> List[AssetPriceDataSet]:
    """Create a list of mock datasets."""
    return [mock_base_dataset, mock_other_dataset]


@pytest.fixture
def mock_symbol_container(
    mock_base_dataset: AssetPriceDataSet, mock_other_dataset: AssetPriceDataSet
) -> SymbolImportContainer:
    """Create a mock SymbolImportContainer with datasets."""
    container = MagicMock(spec=SymbolImportContainer)
    container.symbol = "EURUSD"

    # Create a dictionary of datasets using timeframe as key
    datasets_dict: Dict[str, AssetPriceDataSet] = {
        mock_base_dataset.timeframe: mock_base_dataset,
        mock_other_dataset.timeframe: mock_other_dataset,
    }
    container.asset_price_data_sets = datasets_dict

    return container


@pytest.fixture
def mock_merge_service() -> MagicMock:
    """Create a mock MergeService."""
    mock = MagicMock(spec=MergeService)
    mock.merge_timeframes.return_value = pd.DataFrame(
        {
            "Time": pd.date_range(start="2023-01-01", periods=10, freq="H"),
            "Open": range(10, 20),
            "High": range(20, 30),
            "Low": range(5, 15),
            "Close": range(15, 25),
            "Volume": range(1000, 1010),
            "HTF240_Volume": [
                1000,
                1000,
                1000,
                1000,
                1001,
                1001,
                1001,
                1001,
                1002,
                1002,
            ],
        }
    )
    return mock


class TestPreprocessService:
    """Test suite for PreprocessService."""

    def test_preprocess_data_successful_execution(
        self,
        mock_symbol_container: SymbolImportContainer,
        mock_feature_config: FeaturesConfig,
        mock_feature_class_registry: FeatureClassRegistry,
        mock_feast_service: FeastService,
        mock_merge_service: MagicMock,
    ) -> None:
        """Test the happy path of the preprocess_data method."""
        # Given
        # Set up test preconditions
        service = PreprocessService(
            features_config=mock_feature_config,
            feature_class_registry=mock_feature_class_registry,
            feast_service=mock_feast_service,
        )

        # Create expected result DataFrame with HTF240_Volume
        expected_result = pd.DataFrame(
            {
                "Time": pd.date_range(start="2023-01-01", periods=10, freq="H"),
                "Open": range(10, 20),
                "High": range(20, 30),
                "Low": range(5, 15),
                "Close": range(15, 25),
                "Volume": range(1000, 1010),
                "HTF240_Volume": [
                    1000,
                    1000,
                    1000,
                    1000,
                    1001,
                    1001,
                    1001,
                    1001,
                    1002,
                    1002,
                ],
            }
        )

        # Mock FeatureAggregator creation and usage
        with patch(
            "drl_trading_framework.preprocess.feature.feature_aggregator.FeatureAggregator"
        ) as mock_feature_agg_cls:
            mock_feature_agg = MagicMock()
            mock_feature_agg.compute.return_value = [MagicMock()]
            mock_feature_agg_cls.return_value = mock_feature_agg

            # Mock dask.compute to return our test data
            with patch("dask.compute", return_value=([expected_result],)):

                # Mock MergeService
                with patch(
                    "drl_trading_framework.preprocess.preprocess_service.MergeService"
                ) as mock_merge_service_cls:
                    mock_merge_service_cls.return_value = mock_merge_service

                    # When
                    # Execute the method being tested
                    result = service.preprocess_data(mock_symbol_container)

                    # Then
                    # Assert the expected outcomes
                    assert result is not None
                    assert isinstance(result, DataFrame)
                    assert "HTF240_Volume" in result.columns

                    # Verify FeatureAggregator was created with correct arguments
                    mock_feature_agg_cls.assert_called()

                    # Verify merge_timeframes was called
                    mock_merge_service.merge_timeframes.assert_called()

    def test_preprocess_data_no_features(
        self,
        mock_symbol_container: SymbolImportContainer,
        mock_feature_config: FeaturesConfig,
        mock_feature_class_registry: FeatureClassRegistry,
        mock_feast_service: FeastService,
        mock_merge_service: MagicMock,
    ) -> None:
        """Test preprocessing when no feature tasks are generated."""
        # Given
        # Set up test preconditions
        service = PreprocessService(
            features_config=mock_feature_config,
            feature_class_registry=mock_feature_class_registry,
            feast_service=mock_feast_service,
        )

        # Set up expected result DataFrame
        expected_result = pd.DataFrame(
            {
                "Time": pd.date_range(start="2023-01-01", periods=10, freq="H"),
                "Open": range(10, 20),
                "High": range(20, 30),
                "Low": range(5, 15),
                "Close": range(15, 25),
                "Volume": range(1000, 1010),
                "HTF240_Volume": [
                    1000,
                    1000,
                    1000,
                    1000,
                    1001,
                    1001,
                    1001,
                    1001,
                    1002,
                    1002,
                ],
            }
        )

        # Mock FeatureAggregator to return empty list of tasks
        with patch(
            "drl_trading_framework.preprocess.feature.feature_aggregator.FeatureAggregator"
        ) as mock_feature_agg_cls:
            mock_feature_agg = MagicMock()
            mock_feature_agg.compute.return_value = []  # No tasks!
            mock_feature_agg_cls.return_value = mock_feature_agg

            # Mock MergeService
            with patch(
                "drl_trading_framework.preprocess.preprocess_service.MergeService"
            ) as mock_merge_service_cls:
                mock_merge_service_cls.return_value = mock_merge_service

                # When
                # Execute the method
                result = service.preprocess_data(mock_symbol_container)

                # Then
                # Assert expected outcomes
                assert result is not None
                assert isinstance(result, DataFrame)

                # Verify FeatureAggregator was created
                mock_feature_agg_cls.assert_called()

                # Verify merge_timeframes was called
                mock_merge_service.merge_timeframes.assert_called()

    def test_preprocess_data_no_base_dataset(
        self,
        mock_symbol_container: SymbolImportContainer,
        mock_feature_config: FeaturesConfig,
        mock_feature_class_registry: FeatureClassRegistry,
        mock_feast_service: FeastService,
    ) -> None:
        """Test handling when no base dataset is found after feature computation."""
        # Given
        # Set up test preconditions
        service = PreprocessService(
            features_config=mock_feature_config,
            feature_class_registry=mock_feature_class_registry,
            feast_service=mock_feast_service,
        )

        # Mock dask.compute to return feature dataframes
        with patch(
            "dask.compute",
            return_value=(
                [
                    pd.DataFrame(
                        {"Time": pd.date_range(start="2023-01-01", periods=5)}
                    ),
                ],
            ),
        ):
            # Mock FeatureAggregator
            with patch(
                "drl_trading_framework.preprocess.feature.feature_aggregator.FeatureAggregator"
            ) as mock_feature_agg_cls:
                mock_feature_agg = MagicMock()
                mock_feature_agg.compute.return_value = [MagicMock()]
                mock_feature_agg_cls.return_value = mock_feature_agg

                # Mock separate_computed_datasets to return (None, [])
                with patch(
                    "drl_trading_framework.preprocess.preprocess_service.separate_computed_datasets",
                    return_value=(None, []),
                ):

                    # When
                    # Execute the method being tested
                    result = service.preprocess_data(mock_symbol_container)

                    # Then
                    # Assert the expected outcomes
                    assert result.empty
                    assert isinstance(result, DataFrame)

                    # Verify FeatureAggregator was created
                    mock_feature_agg_cls.assert_called()

    def test_preprocess_data_no_datasets_processed(
        self,
        mock_symbol_container: SymbolImportContainer,
        mock_feature_config: FeaturesConfig,
        mock_feature_class_registry: FeatureClassRegistry,
        mock_feast_service: FeastService,
    ) -> None:
        """Test handling when no datasets could be processed successfully."""
        # Given
        # Set up test preconditions
        service = PreprocessService(
            features_config=mock_feature_config,
            feature_class_registry=mock_feature_class_registry,
            feast_service=mock_feast_service,
        )

        # Mock dask.compute to return invalid results forcing empty computed_dataset_containers
        with patch("dask.compute", return_value=([None, None],)):
            # Mock FeatureAggregator
            with patch(
                "drl_trading_framework.preprocess.feature.feature_aggregator.FeatureAggregator"
            ) as mock_feature_agg_cls:
                mock_feature_agg = MagicMock()
                mock_feature_agg.compute.return_value = [MagicMock(), MagicMock()]
                mock_feature_agg_cls.return_value = mock_feature_agg

                # Force _prepare_dataframe_for_join to return None to simulate failed dataset preparation
                with patch.object(
                    service, "_prepare_dataframe_for_join", return_value=None
                ):

                    # When
                    # Execute the method being tested
                    result = service.preprocess_data(mock_symbol_container)

                    # Then
                    # Assert the expected outcomes
                    assert result.empty
                    assert isinstance(result, DataFrame)

                    # Verify FeatureAggregator was created
                    mock_feature_agg_cls.assert_called()

    def test_prepare_dataframe_for_join_success(
        self,
        mock_feature_config: FeaturesConfig,
        mock_feature_class_registry: FeatureClassRegistry,
        mock_feast_service: FeastService,
    ) -> None:
        """Test that _prepare_dataframe_for_join correctly prepares a DataFrame for joining."""
        # Given
        # Set up test preconditions
        service = PreprocessService(
            features_config=mock_feature_config,
            feature_class_registry=mock_feature_class_registry,
            feast_service=mock_feast_service,
        )
        df = pd.DataFrame(
            {
                "Time": pd.date_range(start="2023-01-01", periods=5, freq="H"),
                "Value": range(5),
            }
        )
        dataset_info = "test dataset"

        # When
        # Execute the function being tested
        result = service._prepare_dataframe_for_join(df, dataset_info)

        # Then
        # Assert the expected outcomes
        assert result is not None
        assert result.index.name == "Time"
        assert "Value" in result.columns

    def test_prepare_dataframe_for_join_empty_df(
        self,
        mock_feature_config: FeaturesConfig,
        mock_feature_class_registry: FeatureClassRegistry,
        mock_feast_service: FeastService,
    ) -> None:
        """Test that _prepare_dataframe_for_join handles empty DataFrames correctly."""
        # Given
        # Set up test preconditions
        service = PreprocessService(
            features_config=mock_feature_config,
            feature_class_registry=mock_feature_class_registry,
            feast_service=mock_feast_service,
        )
        df = pd.DataFrame()
        dataset_info = "test dataset"

        # When
        # Execute the function being tested
        result = service._prepare_dataframe_for_join(df, dataset_info)

        # Then
        # Assert the expected outcomes
        assert result is None
