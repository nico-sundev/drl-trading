"""Integration tests for Kafka consumer with real Kafka broker and topic-based routing."""

import json
import logging
import threading
import time
from typing import Dict, Generator
from unittest.mock import Mock

import pytest
from confluent_kafka import Producer
from testcontainers.kafka import KafkaContainer

from drl_trading_common.adapter.messaging.kafka_consumer_topic_adapter import KafkaConsumerTopicAdapter
from drl_trading_common.messaging.kafka_message_handler import KafkaMessageHandler
from drl_trading_preprocess.core.orchestrator.preprocessing_orchestrator import PreprocessingOrchestrator


logger = logging.getLogger(__name__)

TEST_TOPIC = "test.preprocessing.requests"
TEST_CONSUMER_GROUP = "test-consumer-group"


@pytest.fixture(scope="module")
def kafka_container() -> Generator[KafkaContainer, None, None]:
    """Start a Kafka container for integration tests."""
    # Given
    container = KafkaContainer(image="confluentinc/cp-kafka:7.5.0")
    container.start()

    # Wait for Kafka to be ready
    time.sleep(5)

    yield container

    # When
    container.stop()


@pytest.fixture
def kafka_bootstrap_servers(kafka_container: KafkaContainer) -> str:
    """Get Kafka bootstrap servers from container."""
    # Given
    return kafka_container.get_bootstrap_server()


@pytest.fixture
def kafka_producer(kafka_bootstrap_servers: str) -> Producer:
    """Create a Kafka producer for sending test messages."""
    # Given
    config = {
        "bootstrap.servers": kafka_bootstrap_servers,
        "client.id": "test-producer",
    }
    return Producer(config)


@pytest.fixture
def mock_orchestrator() -> Mock:
    """Create a mock orchestrator to verify handler invocations."""
    # Given
    return Mock(spec=PreprocessingOrchestrator)


@pytest.fixture
def topic_handlers(mock_orchestrator: Mock) -> Dict[str, KafkaMessageHandler]:
    """Create topic->handler mapping for topic-based routing."""
    # Given
    from drl_trading_preprocess.adapter.messaging.kafka_message_handler_factory import KafkaMessageHandlerFactory
    from drl_trading_common.adapter.mappers.feature_preprocessing_request_mapper import FeaturePreprocessingRequestMapper

    factory = KafkaMessageHandlerFactory()
    request_mapper = FeaturePreprocessingRequestMapper()
    preprocessing_handler = factory.create_preprocessing_request_handler(mock_orchestrator, request_mapper)

    return {
        TEST_TOPIC: preprocessing_handler,
    }


class TestKafkaConsumerIntegration:
    """Integration tests for Kafka topic-based message routing with real broker."""

    def test_handler_processes_valid_message(
        self,
        kafka_producer: Producer,
        kafka_bootstrap_servers: str,
        mock_orchestrator: Mock,
        topic_handlers: Dict[str, KafkaMessageHandler],
    ) -> None:
        """Test that a valid message is consumed and handler is invoked."""
        # Given
        test_message = {
            "request_id": "test-request-123",
            "symbol": "BTCUSD",
            "start_time": "2024-01-01T00:00:00Z",
            "end_time": "2024-01-31T23:59:59Z",
            "base_timeframe": "1m",
            "target_timeframes": ["1h", "4h"],
            "processing_context": "backtest",
            "skip_existing_features": False,
            "force_recompute": False,
            "materialize_online": False,
            "feature_config_version_info": {
                "semver": "1.0.0",
                "hash": "abc123",
                "created_at": "2024-01-01T00:00:00Z",
                "feature_definitions": [
                    {
                        "name": "rsi",
                        "enabled": True,
                        "derivatives": [14],
                        "raw_parameter_sets": [{"period": 14}]
                    }
                ]
            }
        }

        consumer_config = {
            "bootstrap.servers": kafka_bootstrap_servers,
            "group.id": TEST_CONSUMER_GROUP,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }

        consumer_adapter = KafkaConsumerTopicAdapter(
            consumer_config=consumer_config,
            topics=[TEST_TOPIC],
            topic_handlers=topic_handlers,
        )

        # When
        # Produce message to Kafka (NO headers needed for topic-based routing)
        kafka_producer.produce(
            topic=TEST_TOPIC,
            value=json.dumps(test_message).encode("utf-8"),
        )
        kafka_producer.flush()

        # Consume and process message (with timeout) - run in background thread
        consumer_thread = threading.Thread(target=consumer_adapter.start, daemon=True)
        consumer_thread.start()

        time.sleep(3)  # Give consumer time to process message
        consumer_adapter.stop()
        consumer_thread.join(timeout=2)

        # Then
        assert mock_orchestrator.process_feature_computation_request.called
        call_args = mock_orchestrator.process_feature_computation_request.call_args
        assert call_args is not None

        request = call_args[0][0]
        assert request.request_id == "test-request-123"
        assert request.symbol == "BTCUSD"
        assert request.base_timeframe.value == "1m"  # Enum comparison
        assert len(request.target_timeframes) == 2
        assert request.target_timeframes[0].value == "1h"
        assert request.target_timeframes[1].value == "4h"

    def test_handler_logs_invalid_json(
        self,
        kafka_producer: Producer,
        kafka_bootstrap_servers: str,
        topic_handlers: Dict[str, KafkaMessageHandler],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that invalid JSON is logged and doesn't crash consumer."""
        # Given
        invalid_message = b"not-valid-json{{"

        consumer_config = {
            "bootstrap.servers": kafka_bootstrap_servers,
            "group.id": f"{TEST_CONSUMER_GROUP}-invalid-json",
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }

        consumer_adapter = KafkaConsumerTopicAdapter(
            consumer_config=consumer_config,
            topics=[TEST_TOPIC],
            topic_handlers=topic_handlers,
        )

        # When
        kafka_producer.produce(
            topic=TEST_TOPIC,
            value=invalid_message,
        )
        kafka_producer.flush()

        with caplog.at_level(logging.ERROR):
            consumer_thread = threading.Thread(target=consumer_adapter.start, daemon=True)
            consumer_thread.start()
            time.sleep(3)
            consumer_adapter.stop()
            consumer_thread.join(timeout=2)

        # Then
        assert any("Invalid JSON" in record.message for record in caplog.records)

    def test_consumer_graceful_shutdown(
        self,
        kafka_bootstrap_servers: str,
        topic_handlers: Dict[str, KafkaMessageHandler],
    ) -> None:
        """Test that consumer shuts down gracefully without errors."""
        # Given
        consumer_config = {
            "bootstrap.servers": kafka_bootstrap_servers,
            "group.id": f"{TEST_CONSUMER_GROUP}-shutdown",
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }

        consumer_adapter = KafkaConsumerTopicAdapter(
            consumer_config=consumer_config,
            topics=[TEST_TOPIC],
            topic_handlers=topic_handlers,
        )

        # When
        consumer_thread = threading.Thread(target=consumer_adapter.start, daemon=True)
        consumer_thread.start()
        time.sleep(1)
        consumer_adapter.stop()
        consumer_thread.join(timeout=2)

        # Then
        # No assertion needed - test passes if no exception raised

    def test_handler_exception_without_failure_policy_warns_and_commits(
        self,
        kafka_producer: Producer,
        kafka_bootstrap_servers: str,
        mock_orchestrator: Mock,
        topic_handlers: Dict[str, KafkaMessageHandler],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that when handler throws exception without failure policy, consumer logs warning and commits offset."""
        # Given
        mock_orchestrator.process_feature_computation_request.side_effect = ValueError("Processing failed")

        test_message = {
            "request_id": "test-error-handling",
            "symbol": "ETHUSD",
            "start_time": "2024-01-01T00:00:00Z",
            "end_time": "2024-01-01T01:00:00Z",
            "base_timeframe": "1m",
            "target_timeframes": ["5m"],
            "processing_context": "backtest",
            "skip_existing_features": False,
            "force_recompute": False,
            "materialize_online": False,
            "feature_config_version_info": {
                "semver": "1.0.0",
                "hash": "abc123",
                "created_at": "2024-01-01T00:00:00Z",
                "feature_definitions": [
                    {
                        "name": "rsi",
                        "enabled": True,
                        "derivatives": [14],
                        "raw_parameter_sets": [{"period": 14}]
                    }
                ]
            }
        }

        consumer_config = {
            "bootstrap.servers": kafka_bootstrap_servers,
            "group.id": f"{TEST_CONSUMER_GROUP}-error",
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }

        consumer_adapter = KafkaConsumerTopicAdapter(
            consumer_config=consumer_config,
            topics=[TEST_TOPIC],
            topic_handlers=topic_handlers,
        )

        # When
        kafka_producer.produce(
            topic=TEST_TOPIC,
            value=json.dumps(test_message).encode("utf-8"),
        )
        kafka_producer.flush()

        with caplog.at_level(logging.ERROR):
            consumer_thread = threading.Thread(target=consumer_adapter.start, daemon=True)
            consumer_thread.start()
            time.sleep(3)
            consumer_adapter.stop()
            consumer_thread.join(timeout=2)

        # Then
        assert mock_orchestrator.process_feature_computation_request.called
        # With new retry topic pattern: no failure policy = log error and commit offset (message lost)
        assert any("Handler failed, no failure policy configured - committing offset (message lost)" in record.message for record in caplog.records)
        assert any("Processing failed" in str(record.exc_info) for record in caplog.records if record.exc_info)

    def test_message_with_key_is_logged(
        self,
        kafka_producer: Producer,
        kafka_bootstrap_servers: str,
        mock_orchestrator: Mock,
        topic_handlers: Dict[str, KafkaMessageHandler],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that messages with keys are processed and key is logged."""
        # Given
        test_message = {
            "request_id": "test-with-key",
            "symbol": "BTCUSD",
            "start_time": "2024-01-01T00:00:00Z",
            "end_time": "2024-01-01T01:00:00Z",
            "base_timeframe": "1m",
            "target_timeframes": ["5m"],
            "processing_context": "backtest",
            "skip_existing_features": False,
            "force_recompute": False,
            "materialize_online": False,
            "feature_config_version_info": {
                "semver": "1.0.0",
                "hash": "abc123",
                "created_at": "2024-01-01T00:00:00Z",
                "feature_definitions": [
                    {
                        "name": "rsi",
                        "enabled": True,
                        "derivatives": [14],
                        "raw_parameter_sets": [{"period": 14}]
                    }
                ]
            }
        }

        consumer_config = {
            "bootstrap.servers": kafka_bootstrap_servers,
            "group.id": f"{TEST_CONSUMER_GROUP}-with-key",
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }

        consumer_adapter = KafkaConsumerTopicAdapter(
            consumer_config=consumer_config,
            topics=[TEST_TOPIC],
            topic_handlers=topic_handlers,
        )

        # When
        # Use symbol as message key for partitioning
        kafka_producer.produce(
            topic=TEST_TOPIC,
            key="BTCUSD".encode("utf-8"),
            value=json.dumps(test_message).encode("utf-8"),
        )
        kafka_producer.flush()

        with caplog.at_level(logging.DEBUG):
            consumer_thread = threading.Thread(target=consumer_adapter.start, daemon=True)
            consumer_thread.start()
            time.sleep(3)
            consumer_adapter.stop()
            consumer_thread.join(timeout=2)

        # Then
        assert mock_orchestrator.process_feature_computation_request.called
        # Verify message was processed successfully
        debug_records = [r for r in caplog.records if r.levelname == "DEBUG" and "Message processed successfully" in r.message]
        assert len(debug_records) > 0, "No debug log found for successful message processing"

    def test_initialization_fails_without_topic_handlers(
        self,
        kafka_bootstrap_servers: str,
    ) -> None:
        """Test that consumer fails fast if no topic_handlers are provided."""
        # Given
        consumer_config = {
            "bootstrap.servers": kafka_bootstrap_servers,
            "group.id": f"{TEST_CONSUMER_GROUP}-no-handlers",
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }

        # When / Then
        with pytest.raises(ValueError, match="topic_handlers cannot be empty"):
            KafkaConsumerTopicAdapter(
                consumer_config=consumer_config,
                topics=[TEST_TOPIC],
                topic_handlers={},  # Empty dict should fail
            )

    def test_retry_topic_publishing_on_failure(
        self,
        kafka_producer: Producer,
        kafka_bootstrap_servers: str,
        mock_orchestrator: Mock,
        topic_handlers: Dict[str, KafkaMessageHandler],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that failed messages are published to retry topic with correct headers."""
        # Given
        from drl_trading_common.adapter.messaging.kafka_producer_adapter import KafkaProducerAdapter
        from drl_trading_common.config.kafka_config import ConsumerFailurePolicy
        from drl_trading_common.config.resilience_config import RetryConfig

        RETRY_TOPIC = "retry.test-topic"

        # Mock the handler to fail
        mock_orchestrator.process_feature_computation_request.side_effect = ValueError("Simulated processing failure")

        # Create retry producer
        retry_config = RetryConfig(
            max_attempts=1,
            wait_exponential_multiplier=1.0,
            wait_exponential_max=1.0,
        )

        producer_config = {
            "bootstrap.servers": kafka_bootstrap_servers,
        }

        retry_producer = KafkaProducerAdapter(
            producer_config=producer_config,
            retry_config=retry_config,
            dlq_topic=None,
        )

        failure_policy = ConsumerFailurePolicy(
            max_retries=3,
            retry_topic=RETRY_TOPIC,
            dlq_topic="dlq.test-topic",
            track_retry_in_headers=True,
            retry_backoff_base_seconds=1.0,
            retry_backoff_multiplier=2.0,
        )

        consumer_config = {
            "bootstrap.servers": kafka_bootstrap_servers,
            "group.id": f"{TEST_CONSUMER_GROUP}-retry-topic-test",
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }

        consumer_adapter = KafkaConsumerTopicAdapter(
            consumer_config=consumer_config,
            topics=[TEST_TOPIC],
            topic_handlers=topic_handlers,
            failure_policies={TEST_TOPIC: failure_policy},
            retry_producer=retry_producer,
        )

        test_message = {
            "request_id": "retry-test",
            "symbol": "BTCUSD",
            "start_time": "2024-01-01T00:00:00Z",
            "end_time": "2024-01-01T01:00:00Z",
            "base_timeframe": "1m",
            "target_timeframes": ["5m"],
            "processing_context": "backtest",
            "skip_existing_features": False,
            "force_recompute": False,
            "materialize_online": False,
            "feature_config_version_info": {
                "semver": "1.0.0",
                "hash": "abc123",
                "created_at": "2024-01-01T00:00:00Z",
                "feature_definitions": [
                    {
                        "name": "rsi",
                        "enabled": True,
                        "derivatives": [14],
                        "raw_parameter_sets": [{"period": 14}]
                    }
                ]
            }
        }

        # When
        kafka_producer.produce(
            topic=TEST_TOPIC,
            value=json.dumps(test_message).encode("utf-8"),
        )
        kafka_producer.flush()

        with caplog.at_level(logging.INFO):
            consumer_thread = threading.Thread(target=consumer_adapter.start, daemon=True)
            consumer_thread.start()
            time.sleep(3)
            consumer_adapter.stop()
            consumer_thread.join(timeout=2)

        # Then
        # Verify retry topic publish logged
        retry_logs = [r for r in caplog.records if "Published message to retry topic" in r.message]
        assert len(retry_logs) > 0, "Expected retry topic publish success log"

        # Verify handler failure logged
        error_logs = [r for r in caplog.records if "Handler failed (attempt 1/" in r.message]
        assert len(error_logs) > 0, "Expected handler failure log"

        retry_producer.close()

    def test_dlq_publishing_after_max_retries(
        self,
        kafka_producer: Producer,
        kafka_bootstrap_servers: str,
        mock_orchestrator: Mock,
        topic_handlers: Dict[str, KafkaMessageHandler],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that message is sent to DLQ after exceeding max retries."""
        # Given
        from drl_trading_common.adapter.messaging.kafka_producer_adapter import KafkaProducerAdapter
        from drl_trading_common.config.kafka_config import ConsumerFailurePolicy
        from drl_trading_common.config.resilience_config import RetryConfig
        from confluent_kafka import Consumer, OFFSET_BEGINNING

        DLQ_TOPIC = "dlq.test-topic"

        # Mock the handler to always fail
        mock_orchestrator.process_feature_computation_request.side_effect = ValueError("Persistent failure")

        # Create DLQ producer with minimal retry config
        dlq_retry_config = RetryConfig(
            max_attempts=1,
            wait_exponential_multiplier=1.0,
            wait_exponential_max=1.0,
        )

        producer_config = {
            "bootstrap.servers": kafka_bootstrap_servers,
        }

        dlq_producer = KafkaProducerAdapter(
            producer_config=producer_config,
            retry_config=dlq_retry_config,
            dlq_topic=None,
        )

        failure_policy = ConsumerFailurePolicy(
            max_retries=0,  # Send to DLQ immediately after first failure
            dlq_topic=DLQ_TOPIC,
            track_retry_in_headers=True,
        )

        consumer_config = {
            "bootstrap.servers": kafka_bootstrap_servers,
            "group.id": f"{TEST_CONSUMER_GROUP}-dlq-test",
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }

        consumer_adapter = KafkaConsumerTopicAdapter(
            consumer_config=consumer_config,
            topics=[TEST_TOPIC],
            topic_handlers=topic_handlers,
            failure_policies={TEST_TOPIC: failure_policy},
            dlq_producer=dlq_producer,
        )

        test_message = {
            "request_id": "dlq-test",
            "symbol": "ETHUSD",
            "start_time": "2024-01-01T00:00:00Z",
            "end_time": "2024-01-01T01:00:00Z",
            "base_timeframe": "1m",
            "target_timeframes": ["5m"],
            "processing_context": "backtest",
            "skip_existing_features": False,
            "force_recompute": False,
            "materialize_online": False,
            "feature_config_version_info": {
                "semver": "1.0.0",
                "hash": "abc123",
                "created_at": "2024-01-01T00:00:00Z",
                "feature_definitions": [
                    {
                        "name": "rsi",
                        "enabled": True,
                        "derivatives": [14],
                        "raw_parameter_sets": [{"period": 14}]
                    }
                ]
            }
        }

        # When - Produce test message
        kafka_producer.produce(
            topic=TEST_TOPIC,
            value=json.dumps(test_message).encode("utf-8"),
        )
        kafka_producer.flush()

        # Start consumer (will fail multiple times, then send to DLQ)
        with caplog.at_level(logging.INFO):  # Capture INFO for DLQ success logs
            consumer_thread = threading.Thread(target=consumer_adapter.start, daemon=True)
            consumer_thread.start()
            time.sleep(5)  # Give time for retries + DLQ publish
            consumer_adapter.stop()
            consumer_thread.join(timeout=2)

        # Then - Verify DLQ publish was logged
        dlq_logs = [r for r in caplog.records if "Successfully published message to DLQ" in r.message]
        assert len(dlq_logs) > 0, "Expected DLQ publish success log"

        # Verify max retries exceeded message
        max_retry_logs = [r for r in caplog.records if f"Max retries ({failure_policy.max_retries}) exceeded" in r.message]
        assert len(max_retry_logs) > 0, "Expected max retries exceeded log"

        # Verify message actually arrived in DLQ topic
        from confluent_kafka import TopicPartition

        dlq_consumer_config = {
            "bootstrap.servers": kafka_bootstrap_servers,
            "group.id": f"{TEST_CONSUMER_GROUP}-dlq-verifier",
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }
        dlq_consumer = Consumer(dlq_consumer_config)
        dlq_consumer.assign([TopicPartition(DLQ_TOPIC, 0, OFFSET_BEGINNING)])

        dlq_message = dlq_consumer.poll(timeout=5.0)
        assert dlq_message is not None, "Expected message in DLQ topic"
        assert not dlq_message.error(), f"DLQ consumer error: {dlq_message.error()}"

        # Verify DLQ payload structure
        dlq_payload = json.loads(dlq_message.value().decode("utf-8"))
        assert dlq_payload["original_topic"] == TEST_TOPIC
        assert dlq_payload["error_type"] == "ValueError"
        assert dlq_payload["retry_attempts"] == failure_policy.max_retries + 1  # Should be 1 now
        assert "original_value" in dlq_payload
        assert dlq_payload["consumer_group"] == f"{TEST_CONSUMER_GROUP}-dlq-test"

        dlq_consumer.close()
        dlq_producer.close()
