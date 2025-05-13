"""Unit tests for the ContextFeatureService class."""

import pandas as pd
import pytest
from pandas import DataFrame

from drl_trading_framework.common.model.asset_price_dataset import AssetPriceDataSet
from drl_trading_framework.common.trading_constants import (
    ALL_CONTEXT_COLUMNS,
    DERIVED_CONTEXT_COLUMNS,
    PRIMARY_CONTEXT_COLUMNS,
)
from drl_trading_framework.preprocess.data_set_utils.context_feature_service import (
    ContextFeatureService,
)
from tests.unit.fixture.sample_data import mock_ohlcv_data_1h


@pytest.fixture
def mock_ohlcv_dataframe() -> DataFrame:
    """Create a mock OHLCV DataFrame for testing using sample_data."""
    return mock_ohlcv_data_1h().asset_price_dataset


@pytest.fixture
def mock_ohlcv_with_atr_dataframe(mock_ohlcv_dataframe) -> DataFrame:
    """Create a mock OHLCV DataFrame with ATR for testing."""
    df = mock_ohlcv_dataframe.copy()
    df["Atr"] = [0.1] * len(df)
    return df


@pytest.fixture
def mock_asset_price_dataset(mock_ohlcv_dataframe) -> AssetPriceDataSet:
    """Create a mock AssetPriceDataSet for testing."""
    return AssetPriceDataSet(
        timeframe="H1",
        base_dataset=True,
        asset_price_dataset=mock_ohlcv_dataframe,
    )


@pytest.fixture
def mock_asset_price_dataset_with_atr(
    mock_ohlcv_with_atr_dataframe,
) -> AssetPriceDataSet:
    """Create a mock AssetPriceDataSet with ATR for testing."""
    return AssetPriceDataSet(
        timeframe="H1",
        base_dataset=True,
        asset_price_dataset=mock_ohlcv_with_atr_dataframe,
    )


@pytest.fixture
def context_feature_service() -> ContextFeatureService:
    """Create a ContextFeatureService instance for testing."""
    return ContextFeatureService(atr_period=14)


def test_prepare_context_features_computes_atr_when_missing(
    context_feature_service, mock_asset_price_dataset
):
    """Test that prepare_context_features computes ATR when it's missing."""
    # Given
    # Asset price dataset without ATR column
    assert "Atr" not in mock_asset_price_dataset.asset_price_dataset.columns

    # When
    result = context_feature_service.prepare_context_features(mock_asset_price_dataset)

    # Then
    assert "Atr" in result.columns
    assert len(result) == len(mock_asset_price_dataset.asset_price_dataset)

    # Verify all required columns are present
    for col in PRIMARY_CONTEXT_COLUMNS:
        assert col in result.columns


def test_prepare_context_features_uses_existing_atr(
    context_feature_service, mock_asset_price_dataset_with_atr
):
    """Test that prepare_context_features uses existing ATR when available."""
    # Given
    # Asset price dataset with pre-computed ATR
    assert "Atr" in mock_asset_price_dataset_with_atr.asset_price_dataset.columns
    original_atr_values = mock_asset_price_dataset_with_atr.asset_price_dataset[
        "Atr"
    ].copy()

    # When
    result = context_feature_service.prepare_context_features(
        mock_asset_price_dataset_with_atr
    )

    # Then
    # Verify ATR values were preserved
    pd.testing.assert_series_equal(result["Atr"], original_atr_values)

    # Verify all primary columns are present
    for col in PRIMARY_CONTEXT_COLUMNS:
        assert col in result.columns


def test_validate_primary_columns(context_feature_service, mock_ohlcv_dataframe):
    """Test that _validate_primary_columns correctly validates primary columns."""
    # Given
    # Complete DataFrame with all primary columns

    # When
    # This should not raise an exception
    context_feature_service._validate_primary_columns(mock_ohlcv_dataframe)

    # Then
    # No exception means validation passed


def test_validate_primary_columns_missing_columns(context_feature_service):
    """Test that _validate_primary_columns raises error for missing primary columns."""
    # Given
    # DataFrame missing some primary columns
    incomplete_df = DataFrame(
        {
            "Time": pd.date_range(start="2022-01-01", periods=10, freq="H"),
            # Missing High, Low
            "Open": [1.0] * 10,
            "Close": [1.5] * 10,
        }
    )

    # When/Then
    # This should raise a ValueError
    with pytest.raises(ValueError) as excinfo:
        context_feature_service._validate_primary_columns(incomplete_df)

    # Verify error message mentions missing columns
    assert "missing" in str(excinfo.value).lower()


def test_compute_derived_columns(context_feature_service, mock_ohlcv_dataframe):
    """Test that _compute_derived_columns adds missing derived columns."""
    # Given
    # DataFrame without derived columns
    assert "Atr" not in mock_ohlcv_dataframe.columns

    # When
    result = context_feature_service._compute_derived_columns(mock_ohlcv_dataframe)

    # Then
    # Should have computed ATR
    assert "Atr" in result.columns

    # First 13 values should be NaN due to ATR lookback period of 14
    assert result["Atr"].iloc[:13].isna().all()

    # Values starting from index 14 should not be NaN
    assert not result["Atr"].iloc[14:].isna().any()


def test_compute_derived_columns_preserves_existing(
    context_feature_service, mock_ohlcv_with_atr_dataframe
):
    """Test that _compute_derived_columns preserves existing derived columns."""
    # Given
    # DataFrame with pre-computed ATR
    original_atr = mock_ohlcv_with_atr_dataframe["Atr"].copy()

    # When
    result = context_feature_service._compute_derived_columns(
        mock_ohlcv_with_atr_dataframe
    )

    # Then
    # Should have preserved original ATR
    pd.testing.assert_series_equal(result["Atr"], original_atr)


def test_prepare_context_features_raises_error_for_missing_required_columns(
    context_feature_service,
):
    """Test that prepare_context_features raises error when required columns are missing."""
    # Given
    # Create a dataset missing required columns
    incomplete_df = DataFrame(
        {
            "Time": pd.date_range(start="2022-01-01", periods=10, freq="H"),
            # Missing High, Low columns
            "Open": [1.0] * 10,
            "Close": [1.5] * 10,
            "Volume": [1000.0] * 10,
        }
    )

    incomplete_dataset = AssetPriceDataSet(
        timeframe="H1",
        base_dataset=True,
        asset_price_dataset=incomplete_df,
    )

    # When/Then
    with pytest.raises(ValueError) as excinfo:
        context_feature_service.prepare_context_features(incomplete_dataset)

    # Verify error message mentions the missing columns
    assert "missing" in str(excinfo.value)
    assert "High" in str(excinfo.value)
    assert "Low" in str(excinfo.value)


def test_prepare_context_features_handles_empty_dataset(context_feature_service):
    """Test that prepare_context_features handles empty datasets gracefully."""
    # Given
    # Empty DataFrame with correct columns
    empty_df = DataFrame(columns=["Time", "Open", "High", "Low", "Close", "Volume"])
    empty_dataset = AssetPriceDataSet(
        timeframe="H1",
        base_dataset=True,
        asset_price_dataset=empty_df,
    )

    # When/Then
    # Should not raise error but return empty DataFrame with ATR
    result = context_feature_service.prepare_context_features(empty_dataset)

    # Then
    assert len(result) == 0
    assert "Atr" in result.columns


def test_is_context_column_identifies_correctly(context_feature_service):
    """Test that is_context_column correctly identifies context columns."""
    # Given
    context_columns = ALL_CONTEXT_COLUMNS
    non_context_columns = ["feature1", "feature2", "indicator3"]

    # When/Then
    # Should identify context columns
    for col in context_columns:
        assert context_feature_service.is_context_column(col) is True

    # Should not identify non-context columns
    for col in non_context_columns:
        assert context_feature_service.is_context_column(col) is False


def test_is_primary_column_identifies_correctly(context_feature_service):
    """Test that is_primary_column correctly identifies primary columns."""
    # Given
    primary_columns = PRIMARY_CONTEXT_COLUMNS
    derived_columns = DERIVED_CONTEXT_COLUMNS

    # When/Then
    # Should identify primary columns
    for col in primary_columns:
        assert context_feature_service.is_primary_column(col) is True

    # Should not identify derived columns as primary
    for col in derived_columns:
        assert context_feature_service.is_primary_column(col) is False


def test_is_derived_column_identifies_correctly(context_feature_service):
    """Test that is_derived_column correctly identifies derived columns."""
    # Given
    primary_columns = PRIMARY_CONTEXT_COLUMNS
    derived_columns = DERIVED_CONTEXT_COLUMNS

    # When/Then
    # Should identify derived columns
    for col in derived_columns:
        assert context_feature_service.is_derived_column(col) is True

    # Should not identify primary columns as derived
    for col in primary_columns:
        assert context_feature_service.is_derived_column(col) is False


def test_get_context_columns_with_dataframe(
    context_feature_service, mock_ohlcv_dataframe
):
    """Test that get_context_columns returns correct columns when DataFrame is provided."""
    # Given
    # Add a non-context column
    df = mock_ohlcv_dataframe.copy()
    df["feature1"] = [1.0] * len(df)

    # When
    result = context_feature_service.get_context_columns(df)

    # Then
    # Should return only context columns that exist in the DataFrame
    assert set(result).issubset(set(ALL_CONTEXT_COLUMNS))
    assert "feature1" not in result
    assert all(col in df.columns for col in result)


def test_get_context_columns_without_dataframe(context_feature_service):
    """Test that get_context_columns returns all context columns when no DataFrame is provided."""
    # Given
    # No DataFrame provided

    # When
    result = context_feature_service.get_context_columns()

    # Then
    # Should return all context columns
    assert set(result) == set(ALL_CONTEXT_COLUMNS)


def test_get_feature_columns(context_feature_service):
    """Test that get_feature_columns returns non-context columns."""
    # Given
    # DataFrame with both context and feature columns
    df = DataFrame(
        {
            "Time": pd.date_range(start="2022-01-01", periods=10, freq="H"),
            "Open": [1.0] * 10,
            "High": [2.0] * 10,
            "Low": [0.5] * 10,
            "Close": [1.5] * 10,
            "Volume": [1000.0] * 10,
            "feature1": [1.0] * 10,
            "feature2": [2.0] * 10,
            "indicator3": [3.0] * 10,
        }
    )

    # When
    result = context_feature_service.get_feature_columns(df)

    # Then
    # Should return only non-context columns
    assert "feature1" in result
    assert "feature2" in result
    assert "indicator3" in result
    assert "Time" not in result
    assert "Open" not in result


def test_merge_context_features(context_feature_service, mock_ohlcv_dataframe):
    """Test that merge_context_features correctly merges context features into computed features."""
    # Given
    # Create a mock computed features DataFrame using the same time range as mock_ohlcv_dataframe
    computed_df = DataFrame(
        {
            "Time": mock_ohlcv_dataframe["Time"].copy(),
            "feature1": [1.0] * len(mock_ohlcv_dataframe),
            "feature2": [2.0] * len(mock_ohlcv_dataframe),
        }
    )

    context_df = mock_ohlcv_dataframe.copy()

    # When
    result = context_feature_service.merge_context_features(computed_df, context_df)

    # Then
    # Result should have both computed features and context columns
    assert "feature1" in result.columns
    assert "feature2" in result.columns

    for col in mock_ohlcv_dataframe.columns:
        assert col in result.columns

    # Check that the merge was performed correctly on Time
    assert len(result) == len(computed_df)

    # Verify values from both DataFrames are preserved
    pd.testing.assert_series_equal(result["feature1"], computed_df["feature1"])
    pd.testing.assert_series_equal(result["Open"], context_df["Open"])


def test_merge_context_features_with_non_matching_indices(
    context_feature_service, mock_ohlcv_dataframe
):
    """Test merge_context_features with non-matching timestamps."""
    # Given
    # Create DataFrames with some non-matching timestamps
    computed_df = DataFrame(
        {
            "Time": pd.date_range(start="2008-10-03 13:00:00", periods=30, freq="H"),
            "feature1": list(range(30)),
        }
    )

    context_df = mock_ohlcv_dataframe.copy()

    # Shift context data to create non-matching timestamps
    context_df_shifted = context_df.copy()
    context_df_shifted["Time"] = pd.date_range(
        start="2008-10-03 15:00:00", periods=30, freq="H"
    )

    # When
    result = context_feature_service.merge_context_features(
        computed_df, context_df_shifted
    )

    # Then
    # Should have all columns
    assert "feature1" in result.columns
    assert "Open" in result.columns

    # Should have all rows from computed_df
    assert len(result) == len(computed_df)

    # Timestamps from computed_df should be preserved
    pd.testing.assert_series_equal(result["Time"], computed_df["Time"])

    # NaN values should exist for timestamps not in context_df
    assert result["Open"].isna().sum() == 2  # First two hours don't match


def test_handles_nan_values_in_price_data(
    context_feature_service, mock_ohlcv_dataframe
):
    """Test that context feature service handles NaN values in price data."""
    # Given
    # DataFrame with some NaN values
    df = mock_ohlcv_dataframe.copy()
    df.loc[1, "Open"] = None
    df.loc[2, "Close"] = None

    dataset = AssetPriceDataSet(
        timeframe="H1",
        base_dataset=True,
        asset_price_dataset=df,
    )

    # When
    result = context_feature_service.prepare_context_features(dataset)

    # Then
    # Should compute ATR despite NaN values
    assert "Atr" in result.columns

    # ATR should contain some NaN values due to NaN inputs and lookback period
    assert result["Atr"].isna().sum() > 0
