"""Integration tests for Kafka handler with real Kafka broker."""

import json
import logging
import threading
import time
from typing import Any, Dict, Generator
from unittest.mock import Mock

import pytest
from confluent_kafka import Producer
from testcontainers.kafka import KafkaContainer

from drl_trading_common.adapter.messaging.kafka_consumer_header_adapter import KafkaConsumerHeaderAdapter
from drl_trading_preprocess.adapter.messaging.kafka_handler_constants import HANDLER_ID_PREPROCESSING_REQUEST
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
def handler_registry(mock_orchestrator: Mock) -> Dict[str, Any]:
    """Create handler registry with mocked orchestrator."""
    # Given
    from drl_trading_preprocess.adapter.messaging.kafka_message_handler_factory import KafkaMessageHandlerFactory
    
    factory = KafkaMessageHandlerFactory()
    return {
        HANDLER_ID_PREPROCESSING_REQUEST: factory.create_preprocessing_request_handler(mock_orchestrator),
    }


class TestKafkaHandlerIntegration:
    """Integration tests for Kafka message handling with real broker."""

    def test_handler_processes_valid_message(
        self,
        kafka_producer: Producer,
        kafka_bootstrap_servers: str,
        mock_orchestrator: Mock,
        handler_registry: Dict[str, Any],
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
            "feature_definitions": [
                {
                    "name": "rsi",
                    "feature_type": "technical_indicator",
                    "enabled": True,
                    "parameters": {"period": 14}
                }
            ],
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
                        "feature_type": "technical_indicator",
                        "enabled": True,
                        "parameters": {"period": 14}
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
        
        consumer_adapter = KafkaConsumerHeaderAdapter(
            consumer_config=consumer_config,
            topics=[TEST_TOPIC],
            handler_registry=handler_registry,
        )

        # When
        # Produce message to Kafka
        kafka_producer.produce(
            topic=TEST_TOPIC,
            value=json.dumps(test_message).encode("utf-8"),
            headers=[("handler_id", HANDLER_ID_PREPROCESSING_REQUEST.encode("utf-8"))],
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
        handler_registry: Dict[str, Any],
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
        
        consumer_adapter = KafkaConsumerHeaderAdapter(
            consumer_config=consumer_config,
            topics=[TEST_TOPIC],
            handler_registry=handler_registry,
        )

        # When
        kafka_producer.produce(
            topic=TEST_TOPIC,
            value=invalid_message,
            headers=[("handler_id", HANDLER_ID_PREPROCESSING_REQUEST.encode("utf-8"))],
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
        handler_registry: Dict[str, Any],
    ) -> None:
        """Test that consumer shuts down gracefully without errors."""
        # Given
        consumer_config = {
            "bootstrap.servers": kafka_bootstrap_servers,
            "group.id": f"{TEST_CONSUMER_GROUP}-shutdown",
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }
        
        consumer_adapter = KafkaConsumerHeaderAdapter(
            consumer_config=consumer_config,
            topics=[TEST_TOPIC],
            handler_registry=handler_registry,
        )

        # When
        consumer_thread = threading.Thread(target=consumer_adapter.start, daemon=True)
        consumer_thread.start()
        time.sleep(1)
        consumer_adapter.stop()
        consumer_thread.join(timeout=2)

        # Then
        # No assertion needed - test passes if no exception raised
