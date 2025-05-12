import unittest
from unittest.mock import Mock, patch

import pandas as pd
from pandas import DataFrame

from ai_trading.model.asset_price_dataset import AssetPriceDataSet
from ai_trading.preprocess.metrics.technical_metrics_service import (
    TechnicalMetricsService,
    TechnicalMetricsServiceFactory,
)


class TestTechnicalMetricsService(unittest.TestCase):
    """Tests for the TechnicalMetricsService class."""

    def test_get_atr(self):
        """Test that ATR is calculated correctly with the specified period."""
        # Given
        # Create sample OHLC data with price movements for testing ATR calculation
        dates = pd.date_range(start="2023-01-01", periods=20, freq="D")
        data = {
            "Open": [100.0, 102.0, 104.0, 103.0, 105.0] * 4,
            "High": [105.0, 107.0, 106.0, 105.0, 110.0] * 4,
            "Low": [98.0, 100.0, 102.0, 101.0, 103.0] * 4,
            "Close": [102.0, 104.0, 103.0, 105.0, 108.0] * 4,
            "Volume": [1000] * 20,
        }
        df = pd.DataFrame(data, index=dates)
        df.index.name = "Time"

        mock_asset_data = Mock(spec=AssetPriceDataSet)
        mock_asset_data.timeframe = "D1"
        mock_asset_data.asset_price_dataset = df

        # When
        # Create service and request ATR with period=5
        service = TechnicalMetricsService(mock_asset_data)
        atr_df = service.get_atr(period=5)

        # Then
        # Verify ATR calculation exists and has correct format
        self.assertIsNotNone(atr_df)
        self.assertIn("ATR", atr_df.columns)
        self.assertEqual(atr_df.index.name, "Time")
        self.assertTrue(isinstance(atr_df.index, pd.DatetimeIndex))

        # ATR should be positive since it measures volatility
        self.assertTrue((atr_df["ATR"].dropna() > 0).all())

        # Check that first period-1 rows have NaN (due to rolling window)
        self.assertTrue(atr_df["ATR"].iloc[:4].isna().all())

        # Check that remaining values are not NaN
        self.assertFalse(atr_df["ATR"].iloc[4:].isna().any())

    def test_get_atr_from_cache(self):
        """Test that ATR is returned from cache on subsequent calls."""
        # Given
        # Create sample data and metrics service
        dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
        df = pd.DataFrame(
            {
                "Open": [100.0] * 10,
                "High": [105.0] * 10,
                "Low": [95.0] * 10,
                "Close": [102.0] * 10,
            },
            index=dates,
        )
        df.index.name = "Time"

        mock_asset_data = Mock(spec=AssetPriceDataSet)
        mock_asset_data.timeframe = "D1"
        mock_asset_data.asset_price_dataset = df

        service = TechnicalMetricsService(mock_asset_data)

        # When
        # Call get_atr twice with the same period
        with patch.object(
            TechnicalMetricsService, "_calculate_atr", wraps=service._calculate_atr
        ) as mock_calc:
            first_call = service.get_atr(period=14)
            second_call = service.get_atr(period=14)

        # Then
        # Verify the calculation method was called only once
        mock_calc.assert_called_once()
        # Verify both calls return the same DataFrame object
        self.assertIs(first_call, second_call)


class TestTechnicalMetricsServiceFactory(unittest.TestCase):
    """Tests for the TechnicalMetricsServiceFactory class."""

    def test_create(self):
        """Test that factory creates a service instance with correct timeframe."""
        # Given
        # Create mock asset data with H1 timeframe
        mock_asset_data = Mock(spec=AssetPriceDataSet)
        mock_asset_data.timeframe = "H1"
        mock_asset_data.asset_price_dataset = DataFrame(
            {"Open": [100.0], "High": [105.0], "Low": [98.0], "Close": [102.0]}
        )

        # When
        # Create service using factory
        service = TechnicalMetricsServiceFactory.create(mock_asset_data)

        # Then
        # Verify service has correct timeframe
        self.assertEqual(service.timeframe, "H1")


if __name__ == "__main__":
    unittest.main()
