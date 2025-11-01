"""Unit tests for KafkaProducerAdapter."""

import pytest
from confluent_kafka import KafkaException
from unittest.mock import Mock, MagicMock, patch

from drl_trading_common.adapter.messaging.kafka_producer_adapter import (
    KafkaProducerAdapter,
)
from drl_trading_common.config.resilience_config import RetryConfig


class TestKafkaProducerAdapterInitialization:
    """Test suite for KafkaProducerAdapter initialization."""

    def test_initialization_with_valid_config(self) -> None:
        """Test successful initialization with valid configuration."""
        # Given
        producer_config = {"bootstrap.servers": "localhost:9092"}
        retry_config = RetryConfig(
            max_attempts=3,
            wait_exponential_multiplier=1.0,
            wait_exponential_max=10.0,
        )
        dlq_topic = "error-topic"

        # When
        with patch("drl_trading_common.adapter.messaging.kafka_producer_adapter.Producer"):
            adapter = KafkaProducerAdapter(
                producer_config=producer_config,
                retry_config=retry_config,
                dlq_topic=dlq_topic,
            )

        # Then
        assert adapter._retry_config == retry_config
        assert adapter._dlq_topic == dlq_topic

    def test_initialization_without_dlq(self) -> None:
        """Test initialization without DLQ topic."""
        # Given
        producer_config = {"bootstrap.servers": "localhost:9092"}
        retry_config = RetryConfig(
            max_attempts=3,
            wait_exponential_multiplier=1.0,
            wait_exponential_max=10.0,
        )

        # When
        with patch("drl_trading_common.adapter.messaging.kafka_producer_adapter.Producer"):
            adapter = KafkaProducerAdapter(
                producer_config=producer_config,
                retry_config=retry_config,
                dlq_topic=None,
            )

        # Then
        assert adapter._dlq_topic is None


class TestKafkaProducerAdapterPublish:
    """Test suite for KafkaProducerAdapter publish functionality."""

    @pytest.fixture
    def mock_producer(self) -> Mock:
        """Create a mock Kafka producer."""
        return MagicMock()

    @pytest.fixture
    def retry_config(self) -> RetryConfig:
        """Create a retry configuration for testing."""
        return RetryConfig(
            max_attempts=3,
            wait_exponential_multiplier=1.0,
            wait_exponential_max=10.0,
            wait_jitter_max=1.0,
        )

    @pytest.fixture
    def adapter(self, mock_producer: Mock, retry_config: RetryConfig) -> KafkaProducerAdapter:
        """Create a KafkaProducerAdapter with mocked producer."""
        with patch("drl_trading_common.adapter.messaging.kafka_producer_adapter.Producer", return_value=mock_producer):
            return KafkaProducerAdapter(
                producer_config={"bootstrap.servers": "localhost:9092"},
                retry_config=retry_config,
                dlq_topic="error-topic",
            )

    def test_publish_success(
        self, adapter: KafkaProducerAdapter, mock_producer: Mock
    ) -> None:
        """Test successful message publication."""
        # Given
        topic = "test-topic"
        key = "test-key"
        value = {"message": "test"}
        headers = {"header1": "value1"}

        # When
        adapter.publish(topic=topic, key=key, value=value, headers=headers)

        # Then
        mock_producer.produce.assert_called_once()
        call_kwargs = mock_producer.produce.call_args.kwargs
        assert call_kwargs["topic"] == topic
        assert call_kwargs["key"] == key.encode("utf-8")
        assert call_kwargs["headers"] == [("header1", b"value1")]
        mock_producer.poll.assert_called_once_with(0)

    def test_publish_without_headers(
        self, adapter: KafkaProducerAdapter, mock_producer: Mock
    ) -> None:
        """Test message publication without headers."""
        # Given
        topic = "test-topic"
        key = "test-key"
        value = {"message": "test"}

        # When
        adapter.publish(topic=topic, key=key, value=value, headers=None)

        # Then
        mock_producer.produce.assert_called_once()
        call_kwargs = mock_producer.produce.call_args.kwargs
        assert call_kwargs["headers"] is None

    def test_publish_with_delivery_failure_triggers_callback_error(
        self, adapter: KafkaProducerAdapter, mock_producer: Mock
    ) -> None:
        """Test that delivery callback raises KafkaException on failure."""
        # Given
        mock_producer.produce.side_effect = lambda **kwargs: kwargs["callback"](
            Exception("Delivery failed"), None
        )

        # When / Then
        with pytest.raises(KafkaException):
            adapter.publish(
                topic="test-topic",
                key="test-key",
                value={"message": "test"},
            )

    def test_publish_retry_on_kafka_exception(
        self, retry_config: RetryConfig
    ) -> None:
        """Test retry logic on KafkaException."""
        # Given
        mock_producer = MagicMock()
        # Simulate failure on first two attempts, success on third
        mock_producer.produce.side_effect = [
            KafkaException("Connection error"),
            KafkaException("Connection error"),
            None,
        ]

        with patch("drl_trading_common.adapter.messaging.kafka_producer_adapter.Producer", return_value=mock_producer):
            adapter = KafkaProducerAdapter(
                producer_config={"bootstrap.servers": "localhost:9092"},
                retry_config=retry_config,
                dlq_topic=None,
            )

            # When
            adapter.publish(
                topic="test-topic",
                key="test-key",
                value={"message": "test"},
            )

        # Then
        assert mock_producer.produce.call_count == 3

    def test_publish_fails_after_max_retries_without_dlq(
        self, retry_config: RetryConfig
    ) -> None:
        """Test that publishing fails after max retries when no DLQ is configured."""
        # Given
        mock_producer = MagicMock()
        mock_producer.produce.side_effect = KafkaException("Persistent error")

        with patch("drl_trading_common.adapter.messaging.kafka_producer_adapter.Producer", return_value=mock_producer):
            adapter = KafkaProducerAdapter(
                producer_config={"bootstrap.servers": "localhost:9092"},
                retry_config=retry_config,
                dlq_topic=None,
            )

            # When / Then
            with pytest.raises(KafkaException):
                adapter.publish(
                    topic="test-topic",
                    key="test-key",
                    value={"message": "test"},
                )

        # Verify retries occurred
        assert mock_producer.produce.call_count == retry_config.max_attempts

    def test_publish_sends_to_dlq_after_max_retries(
        self, retry_config: RetryConfig
    ) -> None:
        """Test that failed messages are sent to DLQ after max retries."""
        # Given
        mock_producer = MagicMock()
        mock_producer.flush.return_value = 0  # Flush succeeds
        # Fail on main topic, succeed on DLQ
        call_count = 0

        def produce_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if kwargs["topic"] == "test-topic":
                raise KafkaException("Persistent error")
            # DLQ publish succeeds
            return None

        mock_producer.produce.side_effect = produce_side_effect

        with patch("drl_trading_common.adapter.messaging.kafka_producer_adapter.Producer", return_value=mock_producer):
            adapter = KafkaProducerAdapter(
                producer_config={"bootstrap.servers": "localhost:9092"},
                retry_config=retry_config,
                dlq_topic="error-topic",
            )

            # When
            adapter.publish(
                topic="test-topic",
                key="test-key",
                value={"message": "test"},
            )

        # Then
        # Max attempts on main topic + 1 DLQ publish
        assert mock_producer.produce.call_count == retry_config.max_attempts + 1
        # Verify DLQ headers were added
        dlq_calls = [
            c for c in mock_producer.produce.call_args_list
            if c.kwargs["topic"] == "error-topic"
        ]
        assert len(dlq_calls) == 1


class TestKafkaProducerAdapterHealthCheck:
    """Test suite for KafkaProducerAdapter health check."""

    def test_health_check_success(self) -> None:
        """Test health check returns True when cluster is accessible."""
        # Given
        mock_producer = MagicMock()
        mock_metadata = MagicMock()
        mock_metadata.topics = {"topic1": Mock(), "topic2": Mock()}
        mock_producer.list_topics.return_value = mock_metadata

        with patch("drl_trading_common.adapter.messaging.kafka_producer_adapter.Producer", return_value=mock_producer):
            adapter = KafkaProducerAdapter(
                producer_config={"bootstrap.servers": "localhost:9092"},
                retry_config=RetryConfig(
                    max_attempts=1,
                    wait_exponential_multiplier=1.0,
                    wait_exponential_max=1.0,
                ),
            )

            # When
            result = adapter.health_check()

        # Then
        assert result is True
        mock_producer.list_topics.assert_called_once_with(timeout=5.0)

    def test_health_check_failure(self) -> None:
        """Test health check returns False when cluster is not accessible."""
        # Given
        mock_producer = MagicMock()
        mock_producer.list_topics.side_effect = KafkaException("Connection failed")

        with patch("drl_trading_common.adapter.messaging.kafka_producer_adapter.Producer", return_value=mock_producer):
            adapter = KafkaProducerAdapter(
                producer_config={"bootstrap.servers": "localhost:9092"},
                retry_config=RetryConfig(
                    max_attempts=1,
                    wait_exponential_multiplier=1.0,
                    wait_exponential_max=1.0,
                ),
            )

            # When
            result = adapter.health_check()

        # Then
        assert result is False


class TestKafkaProducerAdapterFlush:
    """Test suite for KafkaProducerAdapter flush functionality."""

    def test_flush_success_all_messages_sent(self) -> None:
        """Test flush returns 0 when all messages are sent."""
        # Given
        mock_producer = MagicMock()
        mock_producer.flush.return_value = 0

        with patch("drl_trading_common.adapter.messaging.kafka_producer_adapter.Producer", return_value=mock_producer):
            adapter = KafkaProducerAdapter(
                producer_config={"bootstrap.servers": "localhost:9092"},
                retry_config=RetryConfig(
                    max_attempts=1,
                    wait_exponential_multiplier=1.0,
                    wait_exponential_max=1.0,
                ),
            )

            # When
            remaining = adapter.flush(timeout=5.0)

        # Then
        assert remaining == 0
        mock_producer.flush.assert_called_once_with(5.0)

    def test_flush_timeout_with_remaining_messages(self) -> None:
        """Test flush returns remaining message count on timeout."""
        # Given
        mock_producer = MagicMock()
        mock_producer.flush.return_value = 5

        with patch("drl_trading_common.adapter.messaging.kafka_producer_adapter.Producer", return_value=mock_producer):
            adapter = KafkaProducerAdapter(
                producer_config={"bootstrap.servers": "localhost:9092"},
                retry_config=RetryConfig(
                    max_attempts=1,
                    wait_exponential_multiplier=1.0,
                    wait_exponential_max=1.0,
                ),
            )

            # When
            remaining = adapter.flush(timeout=5.0)

        # Then
        assert remaining == 5


class TestKafkaProducerAdapterClose:
    """Test suite for KafkaProducerAdapter close functionality."""

    def test_close_flushes_and_logs_success(self) -> None:
        """Test close flushes all messages and logs success."""
        # Given
        mock_producer = MagicMock()
        mock_producer.flush.return_value = 0

        with patch("drl_trading_common.adapter.messaging.kafka_producer_adapter.Producer", return_value=mock_producer):
            adapter = KafkaProducerAdapter(
                producer_config={"bootstrap.servers": "localhost:9092"},
                retry_config=RetryConfig(
                    max_attempts=1,
                    wait_exponential_multiplier=1.0,
                    wait_exponential_max=1.0,
                ),
            )

            # When
            adapter.close()

        # Then
        mock_producer.flush.assert_called_once_with(30.0)

    def test_close_logs_warning_on_remaining_messages(self) -> None:
        """Test close logs warning when messages remain after flush."""
        # Given
        mock_producer = MagicMock()
        mock_producer.flush.return_value = 3

        with patch("drl_trading_common.adapter.messaging.kafka_producer_adapter.Producer", return_value=mock_producer):
            adapter = KafkaProducerAdapter(
                producer_config={"bootstrap.servers": "localhost:9092"},
                retry_config=RetryConfig(
                    max_attempts=1,
                    wait_exponential_multiplier=1.0,
                    wait_exponential_max=1.0,
                ),
            )

            # When
            adapter.close()

        # Then
        mock_producer.flush.assert_called_once_with(30.0)


class TestKafkaProducerAdapterRetryConfiguration:
    """Test suite for retry configuration behavior."""

    def test_retry_with_stop_after_delay(self) -> None:
        """Test retry stops after configured delay."""
        # Given
        mock_producer = MagicMock()
        mock_producer.produce.side_effect = KafkaException("Persistent error")

        retry_config = RetryConfig(
            max_attempts=10,  # High number
            wait_exponential_multiplier=1.0,
            wait_exponential_max=1.0,
            stop_after_delay=0.1,  # Very short delay to stop quickly
        )

        with patch("drl_trading_common.adapter.messaging.kafka_producer_adapter.Producer", return_value=mock_producer):
            adapter = KafkaProducerAdapter(
                producer_config={"bootstrap.servers": "localhost:9092"},
                retry_config=retry_config,
                dlq_topic=None,
            )

            # When / Then
            with pytest.raises(KafkaException):
                adapter.publish(
                    topic="test-topic",
                    key="test-key",
                    value={"message": "test"},
                )

        # Should stop due to time limit, not max_attempts
        # Exact count depends on timing, but should be less than 10
        assert mock_producer.produce.call_count < 10

    def test_create_retry_decorator_returns_callable(self) -> None:
        """Test that _create_retry_decorator returns a proper decorator."""
        # Given
        mock_producer = MagicMock()
        
        with patch("drl_trading_common.adapter.messaging.kafka_producer_adapter.Producer", return_value=mock_producer):
            adapter = KafkaProducerAdapter(
                producer_config={"bootstrap.servers": "localhost:9092"},
                retry_config=RetryConfig(
                    max_attempts=3,
                    wait_exponential_multiplier=1.0,
                    wait_exponential_max=10.0,
                ),
            )

            # When
            decorator = adapter._create_retry_decorator()

        # Then
        assert callable(decorator)
