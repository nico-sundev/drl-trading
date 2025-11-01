"""Unit tests for KafkaMessagePublisher."""

from datetime import datetime
from typing import Dict, List
from unittest.mock import MagicMock, Mock

import pytest

from drl_trading_common.adapter.messaging.kafka_producer_adapter import (
    KafkaProducerAdapter,
)
from drl_trading_common.model.timeframe import Timeframe
from drl_trading_core.common.model.market_data_model import MarketDataModel
from drl_trading_preprocess.adapter.messaging.publisher.kafka_message_publisher import (
    KafkaMessagePublisher,
)


class TestKafkaMessagePublisherInitialization:
    """Test suite for KafkaMessagePublisher initialization."""

    def test_initialization_with_valid_producers(self) -> None:
        """Test successful initialization with valid producer configurations."""
        # Given
        resampled_data_producer = Mock(spec=KafkaProducerAdapter)
        error_producer = Mock(spec=KafkaProducerAdapter)
        resampled_data_topic = "requested.store-resampled-data"
        error_topic = "error.preprocess-data"

        # When
        publisher = KafkaMessagePublisher(
            resampled_data_producer=resampled_data_producer,
            error_producer=error_producer,
            resampled_data_topic=resampled_data_topic,
            error_topic=error_topic,
        )

        # Then
        assert publisher._resampled_data_producer == resampled_data_producer
        assert publisher._error_producer == error_producer
        assert publisher._resampled_data_topic == resampled_data_topic
        assert publisher._error_topic == error_topic


class TestKafkaMessagePublisherPublishResampledData:
    """Test suite for publishing resampled data."""

    @pytest.fixture
    def mock_resampled_producer(self) -> Mock:
        """Create a mock resampled data producer."""
        return MagicMock(spec=KafkaProducerAdapter)

    @pytest.fixture
    def mock_error_producer(self) -> Mock:
        """Create a mock error producer."""
        return MagicMock(spec=KafkaProducerAdapter)

    @pytest.fixture
    def publisher(
        self, mock_resampled_producer: Mock, mock_error_producer: Mock
    ) -> KafkaMessagePublisher:
        """Create a KafkaMessagePublisher with mocked producers."""
        return KafkaMessagePublisher(
            resampled_data_producer=mock_resampled_producer,
            error_producer=mock_error_producer,
            resampled_data_topic="requested.store-resampled-data",
            error_topic="error.preprocess-data",
        )

    @pytest.fixture
    def sample_market_data(self) -> List[MarketDataModel]:
        """Create sample market data for testing."""
        return [
            MarketDataModel(
                symbol="AAPL",
                timeframe=Timeframe.MINUTE_1,
                timestamp=datetime(2024, 1, 1, 10, 0),
                open_price=150.0,
                high_price=151.0,
                low_price=149.5,
                close_price=150.5,
                volume=1000,
            ),
            MarketDataModel(
                symbol="AAPL",
                timeframe=Timeframe.MINUTE_1,
                timestamp=datetime(2024, 1, 1, 10, 1),
                open_price=150.5,
                high_price=151.5,
                low_price=150.0,
                close_price=151.0,
                volume=1200,
            ),
        ]

    def test_publish_resampled_data_success(
        self,
        publisher: KafkaMessagePublisher,
        mock_resampled_producer: Mock,
        sample_market_data: List[MarketDataModel],
    ) -> None:
        """Test successful publication of resampled data."""
        # Given
        symbol = "AAPL"
        base_timeframe = Timeframe.MINUTE_1
        resampled_data: Dict[Timeframe, List[MarketDataModel]] = {
            Timeframe.MINUTE_5: sample_market_data
        }
        new_candles_count: Dict[Timeframe, int] = {Timeframe.MINUTE_5: 2}

        # When
        publisher.publish_resampled_data(
            symbol=symbol,
            base_timeframe=base_timeframe,
            resampled_data=resampled_data,
            new_candles_count=new_candles_count,
        )

        # Then
        mock_resampled_producer.publish.assert_called_once()
        call_kwargs = mock_resampled_producer.publish.call_args.kwargs
        assert call_kwargs["topic"] == "requested.store-resampled-data"
        assert call_kwargs["key"] == symbol
        assert call_kwargs["headers"]["event_type"] == "resampled_data"
        assert call_kwargs["headers"]["symbol"] == symbol
        
        # Verify payload structure
        payload = call_kwargs["value"]
        assert payload["symbol"] == symbol
        assert payload["base_timeframe"] == base_timeframe.value
        assert payload["total_records"] == 2
        assert payload["total_new_candles"] == 2
        assert Timeframe.MINUTE_5.value in payload["resampled_data"]

    def test_publish_resampled_data_with_multiple_timeframes(
        self,
        publisher: KafkaMessagePublisher,
        mock_resampled_producer: Mock,
        sample_market_data: List[MarketDataModel],
    ) -> None:
        """Test publication with multiple target timeframes."""
        # Given
        symbol = "AAPL"
        base_timeframe = Timeframe.MINUTE_1
        resampled_data: Dict[Timeframe, List[MarketDataModel]] = {
            Timeframe.MINUTE_5: sample_market_data,
            Timeframe.MINUTE_15: sample_market_data[:1],
        }
        new_candles_count: Dict[Timeframe, int] = {
            Timeframe.MINUTE_5: 2,
            Timeframe.MINUTE_15: 1,
        }

        # When
        publisher.publish_resampled_data(
            symbol=symbol,
            base_timeframe=base_timeframe,
            resampled_data=resampled_data,
            new_candles_count=new_candles_count,
        )

        # Then
        mock_resampled_producer.publish.assert_called_once()
        call_kwargs = mock_resampled_producer.publish.call_args.kwargs
        payload = call_kwargs["value"]
        
        assert payload["total_records"] == 3  # 2 + 1
        assert payload["total_new_candles"] == 3  # 2 + 1
        assert len(payload["resampled_data"]) == 2
        assert Timeframe.MINUTE_5.value in payload["resampled_data"]
        assert Timeframe.MINUTE_15.value in payload["resampled_data"]


class TestKafkaMessagePublisherPublishResamplingError:
    """Test suite for publishing resampling errors."""

    @pytest.fixture
    def publisher(self) -> KafkaMessagePublisher:
        """Create a KafkaMessagePublisher with mocked producers."""
        return KafkaMessagePublisher(
            resampled_data_producer=MagicMock(spec=KafkaProducerAdapter),
            error_producer=MagicMock(spec=KafkaProducerAdapter),
            resampled_data_topic="requested.store-resampled-data",
            error_topic="error.preprocess-data",
        )

    def test_publish_resampling_error_success(self, publisher: KafkaMessagePublisher) -> None:
        """Test successful publication of resampling error."""
        # Given
        symbol = "AAPL"
        base_timeframe = Timeframe.MINUTE_1
        target_timeframes = [Timeframe.MINUTE_5, Timeframe.MINUTE_15]
        error_message = "Insufficient data for resampling"
        error_details = {"min_required": "100", "actual": "50"}

        # When
        publisher.publish_resampling_error(
            symbol=symbol,
            base_timeframe=base_timeframe,
            target_timeframes=target_timeframes,
            error_message=error_message,
            error_details=error_details,
        )

        # Then
        publisher._error_producer.publish.assert_called_once()
        call_kwargs = publisher._error_producer.publish.call_args.kwargs
        
        assert call_kwargs["topic"] == "error.preprocess-data"
        assert call_kwargs["key"] == symbol
        assert call_kwargs["headers"]["event_type"] == "resampling_error"
        
        # Verify payload
        payload = call_kwargs["value"]
        assert payload["symbol"] == symbol
        assert payload["base_timeframe"] == base_timeframe.value
        assert payload["error_message"] == error_message
        assert payload["error_details"] == error_details
        assert payload["target_timeframes"] == [tf.value for tf in target_timeframes]


class TestKafkaMessagePublisherHealthCheck:
    """Test suite for health check functionality."""

    def test_health_check_both_producers_healthy(self) -> None:
        """Test health check returns True when both producers are healthy."""
        # Given
        resampled_producer = MagicMock(spec=KafkaProducerAdapter)
        error_producer = MagicMock(spec=KafkaProducerAdapter)
        resampled_producer.health_check.return_value = True
        error_producer.health_check.return_value = True

        publisher = KafkaMessagePublisher(
            resampled_data_producer=resampled_producer,
            error_producer=error_producer,
            resampled_data_topic="test-topic",
            error_topic="error-topic",
        )

        # When
        result = publisher.health_check()

        # Then
        assert result is True
        resampled_producer.health_check.assert_called_once()
        error_producer.health_check.assert_called_once()

    def test_health_check_resampled_producer_unhealthy(self) -> None:
        """Test health check returns False when resampled producer is unhealthy."""
        # Given
        resampled_producer = MagicMock(spec=KafkaProducerAdapter)
        error_producer = MagicMock(spec=KafkaProducerAdapter)
        resampled_producer.health_check.return_value = False
        error_producer.health_check.return_value = True

        publisher = KafkaMessagePublisher(
            resampled_data_producer=resampled_producer,
            error_producer=error_producer,
            resampled_data_topic="test-topic",
            error_topic="error-topic",
        )

        # When
        result = publisher.health_check()

        # Then
        assert result is False

    def test_health_check_error_producer_unhealthy(self) -> None:
        """Test health check returns False when error producer is unhealthy."""
        # Given
        resampled_producer = MagicMock(spec=KafkaProducerAdapter)
        error_producer = MagicMock(spec=KafkaProducerAdapter)
        resampled_producer.health_check.return_value = True
        error_producer.health_check.return_value = False

        publisher = KafkaMessagePublisher(
            resampled_data_producer=resampled_producer,
            error_producer=error_producer,
            resampled_data_topic="test-topic",
            error_topic="error-topic",
        )

        # When
        result = publisher.health_check()

        # Then
        assert result is False

    def test_health_check_both_producers_unhealthy(self) -> None:
        """Test health check returns False when both producers are unhealthy."""
        # Given
        resampled_producer = MagicMock(spec=KafkaProducerAdapter)
        error_producer = MagicMock(spec=KafkaProducerAdapter)
        resampled_producer.health_check.return_value = False
        error_producer.health_check.return_value = False

        publisher = KafkaMessagePublisher(
            resampled_data_producer=resampled_producer,
            error_producer=error_producer,
            resampled_data_topic="test-topic",
            error_topic="error-topic",
        )

        # When
        result = publisher.health_check()

        # Then
        assert result is False


class TestKafkaMessagePublisherClose:
    """Test suite for close functionality."""

    def test_close_calls_both_producers(self) -> None:
        """Test close method calls close on both producers."""
        # Given
        resampled_producer = MagicMock(spec=KafkaProducerAdapter)
        error_producer = MagicMock(spec=KafkaProducerAdapter)

        publisher = KafkaMessagePublisher(
            resampled_data_producer=resampled_producer,
            error_producer=error_producer,
            resampled_data_topic="test-topic",
            error_topic="error-topic",
        )

        # When
        publisher.close()

        # Then
        resampled_producer.close.assert_called_once()
        error_producer.close.assert_called_once()
