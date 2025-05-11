from unittest.mock import MagicMock, patch

import pytest
from pandas import DataFrame, DatetimeIndex, Series, to_datetime

from ai_trading.preprocess.feature.collection.rvi_feature import RviFeature


@pytest.fixture
def mock_data() -> DataFrame:
    # Create sample dates
    dates = to_datetime(
        [
            "2023-01-01",
            "2023-01-02",
            "2023-01-03",
            "2023-01-04",
            "2023-01-05",
            "2023-01-06",
        ]
    )

    # Create the DataFrame with sample data and datetime index
    data = {
        "Open": [100, 101, 102, 103, 104, 105],
        "High": [105, 106, 107, 108, 109, 110],
        "Low": [95, 96, 97, 98, 99, 100],
        "Close": [103, 102, 104, 105, 107, 106],
    }

    df = DataFrame(data, index=dates)
    df.index.name = "Time"
    return df


@pytest.fixture
def config() -> MagicMock:
    mock_config = MagicMock()
    mock_config.length = 14
    return mock_config


@pytest.fixture
def feature(mock_data: DataFrame, config: MagicMock) -> RviFeature:
    return RviFeature(mock_data, config, "test")


@pytest.fixture
def prepared_source_df(feature: RviFeature):
    """Fixture that mocks the _prepare_source_df method and returns a controlled DataFrame."""
    with patch.object(feature, "_prepare_source_df") as mock_prepare:
        mock_df = feature.df_source.copy()
        mock_prepare.return_value = mock_df
        yield mock_df


@patch("pandas_ta.rvi")
def test_compute_rvi(
    patched_rvi: MagicMock,
    feature: RviFeature,
    config: MagicMock,
    prepared_source_df: DataFrame,
) -> None:
    # Given
    expected_values = [50, 55, 60, 65, 70, 75]

    # Create a Series with the same index as prepared_source_df for the RVI values
    rvi_series = Series(expected_values, index=prepared_source_df.index)
    patched_rvi.return_value = rvi_series

    # When
    result = feature.compute()

    # Then
    patched_rvi.assert_called_once_with(
        prepared_source_df["Close"],
        prepared_source_df["High"],
        prepared_source_df["Low"],
        length=config.length,
    )

    assert isinstance(result.index, DatetimeIndex)
    assert "rvi_14test" in result.columns
    assert result["rvi_14test"].head(6).tolist() == expected_values
