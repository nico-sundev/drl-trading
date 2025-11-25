"""Unit tests for Kafka message handler factory."""

import json
from unittest.mock import Mock

import pytest
from confluent_kafka import Message

from drl_trading_common.adapter.model.feature_preprocessing_request import FeaturePreprocessingRequest
from drl_trading_preprocess.adapter.messaging.kafka_message_handler_factory import KafkaMessageHandlerFactory
from drl_trading_preprocess.core.orchestrator.preprocessing_orchestrator import PreprocessingOrchestrator


@pytest.fixture
def valid_request_data() -> dict:
    """Fixture providing valid FeaturePreprocessingRequest data."""
    return {
        "request_id": "req-123",
        "symbol": "EURUSD",
        "start_time": "2024-01-01T00:00:00Z",
        "end_time": "2024-01-31T23:59:59Z",
        "base_timeframe": "1m",
        "target_timeframes": ["5m", "15m"],
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


class TestKafkaMessageHandlerFactory:
    """Test suite for KafkaMessageHandlerFactory."""

    def test_create_preprocessing_request_handler_returns_callable(self) -> None:
        """Test factory returns a callable handler."""
        # Given
        mock_orchestrator = Mock(spec=PreprocessingOrchestrator)
        factory = KafkaMessageHandlerFactory()

        # When
        handler = factory.create_preprocessing_request_handler(mock_orchestrator)

        # Then
        assert callable(handler), "Handler should be callable"

    def test_handler_invokes_orchestrator_with_parsed_request(self, valid_request_data: dict) -> None:
        """Test handler parses message and invokes orchestrator."""
        # Given
        mock_orchestrator = Mock(spec=PreprocessingOrchestrator)
        factory = KafkaMessageHandlerFactory()
        handler = factory.create_preprocessing_request_handler(mock_orchestrator)

        mock_message = Mock(spec=Message)
        mock_message.value.return_value = json.dumps(valid_request_data).encode('utf-8')
        mock_message.topic.return_value = "requested.preprocess-data"
        mock_message.partition.return_value = 0
        mock_message.offset.return_value = 100

        # When
        handler(mock_message)

        # Then
        mock_orchestrator.process_feature_computation_request.assert_called_once()
        call_args = mock_orchestrator.process_feature_computation_request.call_args[0]
        assert isinstance(call_args[0], FeaturePreprocessingRequest)
        assert call_args[0].request_id == "req-123"
        assert call_args[0].symbol == "EURUSD"

    def test_handler_raises_on_invalid_json(self) -> None:
        """Test handler raises JSONDecodeError on invalid JSON."""
        # Given
        mock_orchestrator = Mock(spec=PreprocessingOrchestrator)
        factory = KafkaMessageHandlerFactory()
        handler = factory.create_preprocessing_request_handler(mock_orchestrator)

        mock_message = Mock(spec=Message)
        mock_message.value.return_value = b"invalid json {{"
        mock_message.topic.return_value = "requested.preprocess-data"
        mock_message.partition.return_value = 0
        mock_message.offset.return_value = 100

        # When / Then
        with pytest.raises(json.JSONDecodeError):
            handler(mock_message)

        mock_orchestrator.process_feature_computation_request.assert_not_called()

    def test_handler_raises_on_validation_error(self) -> None:
        """Test handler raises ValidationError on invalid request data."""
        # Given
        mock_orchestrator = Mock(spec=PreprocessingOrchestrator)
        factory = KafkaMessageHandlerFactory()
        handler = factory.create_preprocessing_request_handler(mock_orchestrator)

        mock_message = Mock(spec=Message)
        invalid_data = {
            "request_id": "req-123"
            # Missing required fields
        }
        mock_message.value.return_value = json.dumps(invalid_data).encode('utf-8')
        mock_message.topic.return_value = "requested.preprocess-data"
        mock_message.partition.return_value = 0
        mock_message.offset.return_value = 100

        # When / Then
        with pytest.raises(Exception):  # Pydantic ValidationError
            handler(mock_message)

        mock_orchestrator.process_feature_computation_request.assert_not_called()

    def test_handler_propagates_orchestrator_exceptions(self, valid_request_data: dict) -> None:
        """Test handler propagates exceptions raised by orchestrator."""
        # Given
        mock_orchestrator = Mock(spec=PreprocessingOrchestrator)
        mock_orchestrator.process_feature_computation_request.side_effect = RuntimeError("Orchestrator error")

        factory = KafkaMessageHandlerFactory()
        handler = factory.create_preprocessing_request_handler(mock_orchestrator)

        mock_message = Mock(spec=Message)
        mock_message.value.return_value = json.dumps(valid_request_data).encode('utf-8')
        mock_message.topic.return_value = "requested.preprocess-data"
        mock_message.partition.return_value = 0
        mock_message.offset.return_value = 100

        # When / Then
        with pytest.raises(RuntimeError, match="Orchestrator error"):
            handler(mock_message)

    def test_multiple_handlers_have_independent_orchestrators(self, valid_request_data: dict) -> None:
        """Test each handler instance has its own orchestrator reference."""
        # Given
        mock_orchestrator_1 = Mock(spec=PreprocessingOrchestrator)
        mock_orchestrator_2 = Mock(spec=PreprocessingOrchestrator)
        factory = KafkaMessageHandlerFactory()

        # When
        handler_1 = factory.create_preprocessing_request_handler(mock_orchestrator_1)
        handler_2 = factory.create_preprocessing_request_handler(mock_orchestrator_2)

        mock_message = Mock(spec=Message)
        mock_message.value.return_value = json.dumps(valid_request_data).encode('utf-8')
        mock_message.topic.return_value = "requested.preprocess-data"
        mock_message.partition.return_value = 0
        mock_message.offset.return_value = 100

        handler_1(mock_message)
        handler_2(mock_message)

        # Then
        mock_orchestrator_1.process_feature_computation_request.assert_called_once()
        mock_orchestrator_2.process_feature_computation_request.assert_called_once()
