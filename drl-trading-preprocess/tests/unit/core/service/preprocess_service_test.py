# from unittest.mock import MagicMock, patch

# import pandas as pd
# import pytest
# from dask import delayed
# from drl_trading_common.config.feature_config import FeaturesConfig

# from drl_trading_core.common.model.asset_price_dataset import AssetPriceDataSet
# from drl_trading_core.common.model.computed_dataset_container import (
#     ComputedDataSetContainer,
# )
# from drl_trading_core.common.model.preprocessing_result import PreprocessingResult
# from drl_trading_core.common.model.symbol_import_container import (
#     SymbolImportContainer,
# )
# from drl_trading_adapter.adapter.feature_store.feature_store_fetch_repository import (
#     IFeatureStoreFetchRepository,
# )
# from drl_trading_preprocess.core.service.preprocess_service import (
#     PreprocessService,
#     PreprocessServiceInterface,
# )


# @pytest.fixture
# def mock_feature_config() -> FeaturesConfig:
#     """Create a mock feature config."""
#     mock_config = MagicMock(spec=FeaturesConfig)
#     # Add the feature_definitions attribute that is used in FeatureAggregator.compute()
#     mock_config.feature_definitions = []  # Empty list to avoid iterations
#     return mock_config


# @pytest.fixture
# def mock_feature_class_registry() -> IFeatureFactory:
#     """Create a mock feature class registry."""
#     return MagicMock(spec=IFeatureFactory)


# @pytest.fixture
# def mock_feast_fetch_repo() -> IFeatureStoreFetchRepository:
#     """Create a mock FeastService."""
#     return MagicMock(spec=IFeatureStoreFetchRepository)


# @pytest.fixture
# def mock_feature_aggregator() -> MagicMock:
#     """Create a mock FeatureAggregator."""
#     mock = MagicMock(spec=IFeatureAggregator)

#     # Create a real delayed object that returns a dataframe
#     def create_sample_df(i=0):
#         dates = pd.date_range(start="2023-01-01", periods=10, freq="H")
#         return pd.DataFrame({f"feature_{i}": range(10, 20)}, index=dates)

#     # Mock the compute method to return a list of real delayed tasks
#     mock.compute.return_value = [
#         delayed(create_sample_df)(0),
#         delayed(create_sample_df)(1),
#     ]
#     return mock


# @pytest.fixture
# def mock_base_dataset() -> AssetPriceDataSet:
#     """Create a mock base dataset."""
#     dates = pd.to_datetime(pd.date_range(start="2023-01-01", periods=10, freq="H"))
#     df = pd.DataFrame(
#         {
#             "Open": range(10, 20),
#             "High": range(20, 30),
#             "Low": range(5, 15),
#             "Close": range(15, 25),
#             "Volume": range(1000, 1010),
#         },
#         index=dates,
#     )

#     dataset = MagicMock(spec=AssetPriceDataSet)
#     dataset.timeframe = "H1"
#     dataset.base_dataset = True
#     dataset.symbol = "EURUSD"
#     dataset.asset_price_dataset = df
#     return dataset


# @pytest.fixture
# def mock_base_dataset_computed_container(
#     mock_base_dataset: AssetPriceDataSet,
# ) -> ComputedDataSetContainer:
#     """Create a mock ComputedDataSetContainer."""
#     container = MagicMock()
#     container.source_dataset = mock_base_dataset
#     container.computed_dataframe = pd.DataFrame(
#         {"rsi_7": range(10, 20)}, mock_base_dataset.asset_price_dataset.index
#     )
#     return container


# @pytest.fixture
# def mock_other_dataset() -> AssetPriceDataSet:
#     """Create a mock higher timeframe dataset."""
#     dates = pd.to_datetime(pd.date_range(start="2023-01-01", periods=10, freq="4H"))
#     df = pd.DataFrame(
#         {
#             "Open": range(10, 20),
#             "High": range(15, 25),
#             "Low": range(0, 10),
#             "Close": range(10, 20),
#             "Volume": range(995, 1005),
#         },
#         index=dates,
#     )

#     dataset = MagicMock(spec=AssetPriceDataSet)
#     dataset.timeframe = "H4"
#     dataset.base_dataset = False
#     dataset.asset_price_dataset = df
#     return dataset


# @pytest.fixture
# def mock_other_dataset_computed_container(
#     mock_other_dataset: AssetPriceDataSet,
# ) -> ComputedDataSetContainer:
#     """Create a mock ComputedDataSetContainer."""
#     container = MagicMock()
#     container.source_dataset = mock_other_dataset
#     container.computed_dataframe = pd.DataFrame(
#         {"rsi_7": range(10, 20)}, mock_other_dataset.asset_price_dataset.index
#     )
#     return container


# @pytest.fixture
# def mock_computes_containers(
#     mock_base_dataset_computed_container: ComputedDataSetContainer,
#     mock_other_dataset_computed_container: ComputedDataSetContainer,
# ) -> list[ComputedDataSetContainer]:
#     """Create a list of mock ComputedDataSetContainers."""
#     return [mock_base_dataset_computed_container, mock_other_dataset_computed_container]


# @pytest.fixture
# def mock_symbol_container(
#     mock_base_dataset: AssetPriceDataSet, mock_other_dataset: AssetPriceDataSet
# ) -> SymbolImportContainer:
#     """Create a mock SymbolImportContainer with datasets."""
#     container = MagicMock(spec=SymbolImportContainer)
#     container.symbol = "EURUSD"

#     container.datasets = [mock_base_dataset, mock_other_dataset]

#     return container


# @pytest.fixture
# def mock_merge_service() -> MagicMock:
#     """Create a mock MergeService."""
#     mock = MagicMock(spec=MergeServiceInterface)

#     # Create sample merged dataframe with HTF240_Volume
#     dates = pd.date_range(start="2023-01-01", periods=10, freq="H")
#     merged_df = pd.DataFrame(
#         {
#             "Open": range(10, 20),
#             "High": range(20, 30),
#             "Low": range(5, 15),
#             "Close": range(15, 25),
#             "Volume": range(1000, 1010),
#             "HTF240_Volume": [
#                 1000,
#                 1000,
#                 1000,
#                 1000,
#                 1001,
#                 1001,
#                 1001,
#                 1001,
#                 1002,
#                 1002,
#             ],
#         },
#         index=dates,
#     )

#     # Make merge_timeframes return this dataframe
#     mock.merge_timeframes.return_value = merged_df
#     return mock


# @pytest.fixture
# def mock_context_feature_service() -> ContextFeatureService:
#     """Create a mock ContextFeatureService.

#     This mock simulates the behavior of ContextFeatureService in PreprocessService:
#     1. prepare_context_features - Returns context features DataFrame
#     2. merge_context_features - Merges context features with computed features
#     """
#     mock = MagicMock(spec=ContextFeatureService)

#     # Create sample context features DataFrame
#     context_features = pd.DataFrame(
#         {
#             "Time": pd.date_range(start="2023-01-01", periods=10, freq="H"),
#             "Open": range(10, 20),
#             "High": range(20, 30),
#             "Low": range(5, 15),
#             "Close": range(15, 25),
#             "Volume": range(1000, 1010),
#             "Atr": [0.5] * 10,  # Mocked ATR values
#         }
#     )
#     context_features = context_features.set_index("Time")

#     # Mock the prepare_context_features method
#     mock.prepare_context_features.return_value = context_features

#     # Mock the merge_context_features method to simulate joining dataframes
#     mock.merge_context_features.side_effect = lambda df, context_df: df.join(
#         context_df.iloc[:, ~context_df.columns.isin(df.columns)], how="left"
#     )

#     return mock


# @pytest.fixture
# def preprocess_service(
#     mock_feature_config: FeaturesConfig,
#     mock_feature_class_registry: IFeatureFactory,
#     mock_feature_aggregator: MagicMock,
#     mock_merge_service: MagicMock,
#     mock_context_feature_service: ContextFeatureService,
#     mock_base_dataset_computed_container: ComputedDataSetContainer,
# ) -> PreprocessServiceInterface:
#     """Create a PreprocessService instance with patched compute_features_for_dataset method."""
#     service = PreprocessService(
#         features_config=mock_feature_config,
#         feature_factory=mock_feature_class_registry,
#         feature_aggregator=mock_feature_aggregator,
#         merge_service=mock_merge_service,
#         context_feature_service=mock_context_feature_service,
#     )

#     # Patch the internal _compute_features_for_dataset method to return our mock container
#     # This avoids the need to patch dask.compute
#     with patch.object(
#         service,
#         "_compute_features_for_dataset",
#         return_value=mock_base_dataset_computed_container,
#     ):
#         yield service


# def test_preprocess_data_successful_execution(
#     mock_symbol_container: SymbolImportContainer,
#     preprocess_service: PreprocessServiceInterface,
#     mock_merge_service: MergeServiceInterface,
#     mock_base_dataset_computed_container: ComputedDataSetContainer,
#     mock_context_feature_service: ContextFeatureService,
# ) -> None:
#     """Test the happy path of the preprocess_data method."""
#     # Given
#     # Mock separation of base and other computed datasets
#     with patch(
#         "drl_trading_core.preprocess.preprocess_service.separate_computed_datasets",
#         return_value=(
#             mock_base_dataset_computed_container,
#             [mock_base_dataset_computed_container],
#         ),
#     ), patch.object(
#         preprocess_service,
#         "_merge_all_timeframes_features_together",
#         return_value=pd.DataFrame({"feature_1": range(10), "feature_2": range(10, 20)}),
#     ):
#         # When
#         # Execute the method being tested
#         result = preprocess_service.preprocess_data(mock_symbol_container)

#         # Then
#         # Assert the expected outcomes
#         assert result is not None
#         assert isinstance(result, PreprocessingResult)

#         # Verify context feature service methods were called
#         mock_context_feature_service.prepare_context_features.assert_called_once()
#         mock_context_feature_service.merge_context_features.assert_called_once()


# def test_preprocess_data_no_features(
#     mock_symbol_container: SymbolImportContainer,
#     preprocess_service: PreprocessServiceInterface,
#     mock_context_feature_service: ContextFeatureService,
# ) -> None:
#     """Test preprocessing when no feature tasks are generated."""
#     # Given
#     # Using preprocess_service fixture with patched _compute_features_for_dataset

#     # Override the patched _compute_features_for_dataset to return None
#     with patch.object(
#         preprocess_service, "_compute_features_for_dataset", return_value=None
#     ):
#         # When/Then
#         # Execute the method - should raise ValueError due to no computed datasets
#         with pytest.raises(
#             ValueError, match="No valid computed datasets were produced"
#         ):
#             preprocess_service.preprocess_data(mock_symbol_container)


# def test_preprocess_data_no_base_dataset(
#     mock_symbol_container: SymbolImportContainer,
#     preprocess_service: PreprocessServiceInterface,
#     mock_other_dataset_computed_container: ComputedDataSetContainer,
# ) -> None:
#     """Test handling when no base dataset is found after feature computation."""
#     # Given
#     # Using preprocess_service fixture with patched _compute_features_for_dataset
#     # Override to return a non-none value to avoid the first validation check

#     # Mock separate_computed_datasets to return (None, [])
#     with patch(
#         "drl_trading_core.preprocess.preprocess_service.separate_computed_datasets",
#         return_value=(None, [mock_other_dataset_computed_container]),
#     ):
#         # This should raise a ValueError because base_computed_container is None
#         with pytest.raises(ValueError, match="No base dataset found"):
#             preprocess_service.preprocess_data(mock_symbol_container)


# def test_compute_features_for_dataset_no_tasks(
#     preprocess_service: PreprocessServiceInterface,
#     mock_base_dataset: AssetPriceDataSet,
#     mock_feature_aggregator: MagicMock,
# ) -> None:
#     """Test _compute_features_for_dataset when no feature tasks are returned."""
#     # Given
#     # Override patched _compute_features_for_dataset to use original implementation
#     with patch.object(
#         preprocess_service,
#         "_compute_features_for_dataset",
#         wraps=PreprocessService._compute_features_for_dataset.__get__(
#             preprocess_service
#         ),
#     ):
#         # And mock feature_aggregator to return no tasks
#         mock_feature_aggregator.compute.return_value = []

#         # When
#         result = preprocess_service._compute_features_for_dataset(
#             mock_base_dataset, "EURUSD"
#         )

#         # Then
#         assert result is None
#         mock_feature_aggregator.compute.assert_called_once_with(
#             asset_data=mock_base_dataset, symbol="EURUSD"
#         )


# def test_prepare_dataframe_for_join_success(
#     preprocess_service: PreprocessServiceInterface,
#     mock_context_feature_service: ContextFeatureService,
# ) -> None:
#     """Test that _prepare_dataframe_for_join correctly prepares a DataFrame for joining."""
#     # Given
#     # Using preprocess_service fixture which already has all the mocks set up
#     df = pd.DataFrame(
#         {
#             "Time": pd.date_range(start="2023-01-01", periods=5, freq="H"),
#             "Value": range(5),
#         }
#     )
#     dataset_info = "test dataset"
#     # When
#     # Execute the function being tested
#     result = preprocess_service._prepare_dataframe_for_join(df, dataset_info)

#     # Then
#     # Assert the expected outcomes
#     assert result is not None
#     assert result.index.name == "Time"
#     assert "Value" in result.columns


# def test_prepare_dataframe_for_join_empty_df(
#     preprocess_service: PreprocessServiceInterface,
#     mock_context_feature_service: ContextFeatureService,
# ) -> None:
#     """Test that _prepare_dataframe_for_join handles empty DataFrames correctly."""
#     # Given
#     # Using preprocess_service fixture which already has all the mocks set up
#     df = pd.DataFrame()
#     dataset_info = "test dataset"
#     # When
#     # Execute the function being tested
#     result = preprocess_service._prepare_dataframe_for_join(df, dataset_info)

#     # Then
#     # Assert the expected outcomes
#     assert result is None


# def test_merge_all_timeframes_features_together_length_mismatch(
#     preprocess_service: PreprocessServiceInterface,
#     mock_base_dataset_computed_container: ComputedDataSetContainer,
#     mock_other_dataset_computed_container: ComputedDataSetContainer,
#     mock_merge_service: MagicMock,
# ) -> None:
#     """Test handling of length mismatch in merge_all_timeframes_features_together."""
#     # Given
#     # Mock the merge_service to return a dataframe with different length
#     mock_merge_service.merge_timeframes.return_value = pd.DataFrame(
#         {"mismatched_column": range(5)}  # Different length from base frame
#     )

#     # When/Then
#     with pytest.raises(
#         ValueError, match="One or more DataFrames have a different length"
#     ):
#         preprocess_service._merge_all_timeframes_features_together(
#             mock_base_dataset_computed_container,
#             [mock_other_dataset_computed_container],
#         )


# def test_compute_features_for_dataset_with_valid_tasks(
#     preprocess_service: PreprocessServiceInterface,
#     mock_base_dataset: AssetPriceDataSet,
#     mock_feature_aggregator: MagicMock,
# ) -> None:
#     """Test _compute_features_for_dataset with valid tasks."""
#     # Given
#     # Override patched _compute_features_for_dataset to use original implementation
#     with patch.object(
#         preprocess_service,
#         "_compute_features_for_dataset",
#         wraps=PreprocessService._compute_features_for_dataset.__get__(
#             preprocess_service
#         ),
#     ):
#         # Setup a real dataframe that will be returned by our compute function
#         dates = pd.date_range(start="2023-01-01", periods=10, freq="H")
#         feature_df = pd.DataFrame({"feature1": range(10)}, index=dates)

#         # Instead of using a lambda, create a real value directly
#         # This is important because dask will try to call .copy() on the result
#         mock_feature_aggregator.compute.return_value = [delayed(feature_df)]

#         # When
#         result = preprocess_service._compute_features_for_dataset(
#             mock_base_dataset, "EURUSD"
#         )

#         # Then
#         assert result is not None
#         assert isinstance(result, ComputedDataSetContainer)
#         assert result.source_dataset == mock_base_dataset
#         pd.testing.assert_frame_equal(result.computed_dataframe, feature_df)
