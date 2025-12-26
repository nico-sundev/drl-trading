"""Unit tests for FeatureComputationCoordinator."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock

import pandas as pd
import pytest
from pandas import DataFrame

from drl_trading_common.core.model.timeframe import Timeframe
from drl_trading_core.core.dto.feature_preprocessing_request import (
    FeaturePreprocessingRequest,
)
from drl_trading_core.core.model.feature_definition import FeatureDefinition
from drl_trading_core.core.model.market_data_model import MarketDataModel
from drl_trading_core.core.port.market_data_reader_port import MarketDataReaderPort
from drl_trading_preprocess.core.service.compute.computing_service import (
    FeatureComputingService,
)
from drl_trading_preprocess.core.service.compute.feature_computation_coordinator import (
    FeatureComputationCoordinator,
)
from drl_trading_preprocess.application.config.preprocess_config import (
    FeatureComputationCoordinatorConfig,
)


class TestFeatureComputationCoordinator:
    """Test suite for FeatureComputationCoordinator."""

    @pytest.fixture
    def mock_market_data_reader(self) -> MagicMock:
        """Create mock MarketDataReaderPort."""
        return MagicMock(spec=MarketDataReaderPort)

    @pytest.fixture
    def mock_feature_computer(self) -> MagicMock:
        """Create mock FeatureComputingService."""
        return MagicMock(spec=FeatureComputingService)

    @pytest.fixture
    def coordinator_config(self) -> FeatureComputationCoordinatorConfig:
        """Create FeatureComputationCoordinatorConfig with test values."""
        return FeatureComputationCoordinatorConfig(pagination_chunk_size=1000)

    @pytest.fixture
    def coordinator(
        self,
        mock_market_data_reader: MagicMock,
        mock_feature_computer: MagicMock,
        coordinator_config: FeatureComputationCoordinatorConfig,
    ) -> FeatureComputationCoordinator:
        """Create FeatureComputationCoordinator instance."""
        return FeatureComputationCoordinator(
            market_data_reader=mock_market_data_reader,
            feature_computer=mock_feature_computer,
            config=coordinator_config,
        )

    @pytest.fixture
    def sample_request(self) -> FeaturePreprocessingRequest:
        """Create sample FeaturePreprocessingRequest."""
        return Mock(
            symbol="BTCUSDT",
            start_time=datetime(2025, 1, 1, 0, 0),
            end_time=datetime(2025, 1, 1, 23, 59),
            target_timeframes=[Timeframe.MINUTE_5],
        )

    @pytest.fixture
    def sample_features(self) -> list[FeatureDefinition]:
        """Create sample feature definitions."""
        return [
            Mock(spec=FeatureDefinition, name="rsi_14"),
            Mock(spec=FeatureDefinition, name="sma_20"),
        ]

    @pytest.fixture
    def sample_dataframe(self) -> DataFrame:
        """Create sample market data DataFrame."""
        base_time = datetime(2025, 1, 1, 0, 0)
        data = {
            "Open": [100.0 + i for i in range(10)],
            "High": [105.0 + i for i in range(10)],
            "Low": [95.0 + i for i in range(10)],
            "Close": [102.0 + i for i in range(10)],
            "Volume": [1000.0 + i * 10 for i in range(10)],
        }
        df = DataFrame(data, index=[base_time + timedelta(minutes=i * 5) for i in range(10)])
        df.index.name = "timestamp"
        return df

    @pytest.fixture
    def sample_market_data_models(self) -> list[MarketDataModel]:
        """Create sample MarketDataModel list."""
        base_time = datetime(2025, 1, 1, 0, 0)
        return [
            MarketDataModel(
                symbol="BTCUSDT",
                timeframe=Timeframe.MINUTE_5,
                timestamp=base_time + timedelta(minutes=i * 5),
                open_price=100.0 + i,
                high_price=105.0 + i,
                low_price=95.0 + i,
                close_price=102.0 + i,
                volume=1000.0 + i * 10,
            )
            for i in range(10)
        ]

    def test_compute_features_with_all_resampled_data(
        self,
        coordinator: FeatureComputationCoordinator,
        mock_feature_computer: MagicMock,
        sample_request: Mock,
        sample_features: list[FeatureDefinition],
        sample_dataframe: DataFrame,
    ) -> None:
        """Test computing features when all timeframes have resampled data."""
        # Given
        timeframe = Timeframe.MINUTE_5
        resampled_data = {timeframe: sample_dataframe}

        # Mock feature computation result
        features_df = sample_dataframe.copy()
        features_df["rsi_14"] = [50.0] * len(sample_dataframe)
        features_df["sma_20"] = [100.0] * len(sample_dataframe)
        mock_feature_computer.compute_batch.return_value = features_df

        # When
        result = coordinator.compute_features_for_timeframes(
            request=sample_request,
            features_to_compute=sample_features,
            resampled_data=resampled_data,
        )

        # Then
        assert timeframe in result
        assert not result[timeframe].empty
        assert "event_timestamp" in result[timeframe].columns
        assert "symbol" in result[timeframe].columns
        assert result[timeframe]["symbol"].iloc[0] == "BTCUSDT"
        mock_feature_computer.compute_batch.assert_called_once()

    def test_compute_features_with_db_fetch_when_no_resampled_data(
        self,
        coordinator: FeatureComputationCoordinator,
        mock_market_data_reader: MagicMock,
        mock_feature_computer: MagicMock,
        sample_request: Mock,
        sample_features: list[FeatureDefinition],
        sample_market_data_models: list[MarketDataModel],
    ) -> None:
        """Test computing features fetches from DB when timeframe not in resampled_data."""
        # Given
        timeframe = Timeframe.MINUTE_5
        resampled_data = {}  # Empty - no resampled data

        # Mock DB fetch
        mock_market_data_reader.get_symbol_data_range_paginated.return_value = (
            sample_market_data_models
        )

        # Mock feature computation result
        features_df = pd.DataFrame(
            {"rsi_14": [50.0] * 10, "sma_20": [100.0] * 10},
            index=[m.timestamp for m in sample_market_data_models],
        )
        features_df.index.name = "timestamp"
        mock_feature_computer.compute_batch.return_value = features_df

        # When
        result = coordinator.compute_features_for_timeframes(
            request=sample_request,
            features_to_compute=sample_features,
            resampled_data=resampled_data,
        )

        # Then
        assert timeframe in result
        assert not result[timeframe].empty
        mock_market_data_reader.get_symbol_data_range_paginated.assert_called()
        mock_feature_computer.compute_batch.assert_called_once()

    def test_compute_features_mixed_scenario(
        self,
        coordinator: FeatureComputationCoordinator,
        mock_market_data_reader: MagicMock,
        mock_feature_computer: MagicMock,
        sample_features: list[FeatureDefinition],
        sample_dataframe: DataFrame,
        sample_market_data_models: list[MarketDataModel],
    ) -> None:
        """Test computing features with mixed resampled and DB-fetched data."""
        # Given
        tf_5m = Timeframe.MINUTE_5
        tf_15m = Timeframe.MINUTE_15

        request = Mock(
            symbol="BTCUSDT",
            start_time=datetime(2025, 1, 1, 0, 0),
            end_time=datetime(2025, 1, 1, 23, 59),
            target_timeframes=[tf_5m, tf_15m],
        )

        # 5m has resampled data, 15m needs DB fetch
        resampled_data = {tf_5m: sample_dataframe}

        # Mock DB fetch for 15m
        mock_market_data_reader.get_symbol_data_range_paginated.return_value = (
            sample_market_data_models
        )

        # Mock feature computation
        mock_feature_computer.compute_batch.return_value = pd.DataFrame(
            {"rsi_14": [50.0] * 10},
            index=sample_dataframe.index,
        )

        # When
        result = coordinator.compute_features_for_timeframes(
            request=request,
            features_to_compute=sample_features,
            resampled_data=resampled_data,
        )

        # Then
        assert tf_5m in result
        assert tf_15m in result
        assert mock_feature_computer.compute_batch.call_count == 2
        # Verify DB was called only for 15m (not in resampled_data)
        mock_market_data_reader.get_symbol_data_range_paginated.assert_called()

    def test_fetch_market_data_as_dataframe_with_pagination(
        self,
        mock_market_data_reader: MagicMock,
        sample_market_data_models: list[MarketDataModel],
    ) -> None:
        """Test fetching market data with pagination."""
        # Given
        # Use smaller chunk size for testing pagination
        config = FeatureComputationCoordinatorConfig(pagination_chunk_size=5)
        coordinator = FeatureComputationCoordinator(
            market_data_reader=mock_market_data_reader,
            feature_computer=MagicMock(),
            config=config,
        )

        symbol = "BTCUSDT"
        timeframe = Timeframe.MINUTE_5
        start_time = datetime(2025, 1, 1, 0, 0)
        end_time = datetime(2025, 1, 1, 1, 0)

        # Mock paginated responses - each chunk has 5 records (equals chunk_size)
        chunk1 = sample_market_data_models[:5]  # 5 records (equals chunk_size, continues)
        chunk2 = sample_market_data_models[5:]  # 5 records (equals chunk_size, continues)
        chunk3 = []  # Empty chunk signals end
        mock_market_data_reader.get_symbol_data_range_paginated.side_effect = [
            chunk1,
            chunk2,
            chunk3,
        ]

        # When
        result_df = coordinator.fetch_market_data_as_dataframe(
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time,
        )

        # Then
        assert not result_df.empty
        assert len(result_df) == 10
        assert list(result_df.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert result_df.index.name == "timestamp"

    def test_fetch_market_data_filters_invalid_ohlcv(
        self,
        coordinator: FeatureComputationCoordinator,
        mock_market_data_reader: MagicMock,
    ) -> None:
        """Test fetching market data filters invalid OHLCV records."""
        # Given
        symbol = "BTCUSDT"
        timeframe = Timeframe.MINUTE_5
        start_time = datetime(2025, 1, 1, 0, 0)
        end_time = datetime(2025, 1, 1, 1, 0)

        valid_record = MarketDataModel(
            symbol="BTCUSDT",
            timeframe=Timeframe.MINUTE_5,
            timestamp=datetime(2025, 1, 1, 0, 0),
            open_price=100.0,
            high_price=105.0,
            low_price=95.0,
            close_price=102.0,
            volume=1000.0,
        )

        # Invalid: high < low
        invalid_record = MarketDataModel(
            symbol="BTCUSDT",
            timeframe=Timeframe.MINUTE_5,
            timestamp=datetime(2025, 1, 1, 0, 5),
            open_price=100.0,
            high_price=90.0,  # Invalid
            low_price=95.0,
            close_price=102.0,
            volume=1000.0,
        )

        mock_market_data_reader.get_symbol_data_range_paginated.return_value = [
            valid_record,
            invalid_record,
        ]

        # When
        result_df = coordinator.fetch_market_data_as_dataframe(
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time,
        )

        # Then
        assert len(result_df) == 1  # Only valid record

    def test_fetch_market_data_fallback_to_non_paginated(
        self,
        coordinator: FeatureComputationCoordinator,
        mock_market_data_reader: MagicMock,
        sample_market_data_models: list[MarketDataModel],
    ) -> None:
        """Test fallback to non-paginated method when pagination unavailable."""
        # Given
        symbol = "BTCUSDT"
        timeframe = Timeframe.MINUTE_5
        start_time = datetime(2025, 1, 1, 0, 0)
        end_time = datetime(2025, 1, 1, 1, 0)

        mock_market_data_reader.get_symbol_data_range_paginated.side_effect = (
            AttributeError("Not implemented")
        )
        mock_market_data_reader.get_symbol_data_range.return_value = (
            sample_market_data_models
        )

        # When
        result_df = coordinator.fetch_market_data_as_dataframe(
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time,
        )

        # Then
        assert not result_df.empty
        assert len(result_df) == 10
        mock_market_data_reader.get_symbol_data_range.assert_called_once()

    def test_convert_to_dataframe(
        self,
        coordinator: FeatureComputationCoordinator,
        sample_market_data_models: list[MarketDataModel],
    ) -> None:
        """Test converting MarketDataModel list to DataFrame."""
        # Given
        # When
        result_df = coordinator._convert_to_dataframe(sample_market_data_models)

        # Then
        assert not result_df.empty
        assert len(result_df) == 10
        assert list(result_df.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert result_df.index.name == "timestamp"
        assert result_df["Open"].iloc[0] == 100.0
        assert result_df["Close"].iloc[-1] == 111.0

    def test_convert_empty_list_to_dataframe(
        self, coordinator: FeatureComputationCoordinator
    ) -> None:
        """Test converting empty list returns empty DataFrame with correct structure."""
        # Given
        empty_list: list[MarketDataModel] = []

        # When
        result_df = coordinator._convert_to_dataframe(empty_list)

        # Then
        assert result_df.empty
        assert list(result_df.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert result_df.index.name == "timestamp"

    def test_is_invalid_ohlcv_with_high_less_than_low(
        self, coordinator: FeatureComputationCoordinator
    ) -> None:
        """Test invalid OHLCV detection for high < low."""
        # Given
        record = MarketDataModel(
            symbol="BTCUSDT",
            timeframe=Timeframe.MINUTE_5,
            timestamp=datetime(2025, 1, 1, 0, 0),
            open_price=100.0,
            high_price=90.0,  # Invalid
            low_price=95.0,
            close_price=102.0,
            volume=1000.0,
        )

        # When
        result = coordinator._is_invalid_ohlcv(record)

        # Then
        assert result is True

    def test_is_invalid_ohlcv_with_zero_price(
        self, coordinator: FeatureComputationCoordinator
    ) -> None:
        """Test invalid OHLCV detection for zero prices."""
        # Given
        record = MarketDataModel(
            symbol="BTCUSDT",
            timeframe=Timeframe.MINUTE_5,
            timestamp=datetime(2025, 1, 1, 0, 0),
            open_price=0.0,  # Invalid
            high_price=105.0,
            low_price=95.0,
            close_price=102.0,
            volume=1000.0,
        )

        # When
        result = coordinator._is_invalid_ohlcv(record)

        # Then
        assert result is True

    def test_is_invalid_ohlcv_with_negative_volume(
        self, coordinator: FeatureComputationCoordinator
    ) -> None:
        """Test invalid OHLCV detection for negative volume."""
        # Given
        record = MarketDataModel(
            symbol="BTCUSDT",
            timeframe=Timeframe.MINUTE_5,
            timestamp=datetime(2025, 1, 1, 0, 0),
            open_price=100.0,
            high_price=105.0,
            low_price=95.0,
            close_price=102.0,
            volume=-1000.0,  # Invalid
        )

        # When
        result = coordinator._is_invalid_ohlcv(record)

        # Then
        assert result is True

    def test_is_invalid_ohlcv_with_valid_data(
        self, coordinator: FeatureComputationCoordinator
    ) -> None:
        """Test invalid OHLCV detection returns False for valid data."""
        # Given
        record = MarketDataModel(
            symbol="BTCUSDT",
            timeframe=Timeframe.MINUTE_5,
            timestamp=datetime(2025, 1, 1, 0, 0),
            open_price=100.0,
            high_price=105.0,
            low_price=95.0,
            close_price=102.0,
            volume=1000.0,
        )

        # When
        result = coordinator._is_invalid_ohlcv(record)

        # Then
        assert result is False

    def test_compute_features_returns_empty_dict_for_empty_input(
        self,
        coordinator: FeatureComputationCoordinator,
        sample_request: Mock,
        sample_features: list[FeatureDefinition],
    ) -> None:
        """Test computing features returns empty dict when no timeframes provided."""
        # Given
        sample_request.target_timeframes = []
        resampled_data = {}

        # When
        result = coordinator.compute_features_for_timeframes(
            request=sample_request,
            features_to_compute=sample_features,
            resampled_data=resampled_data,
        )

        # Then
        assert result == {}
