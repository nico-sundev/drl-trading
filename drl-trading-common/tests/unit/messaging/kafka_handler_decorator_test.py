"""Unit tests for Kafka handler decorator."""

import json
from unittest.mock import Mock

import pytest
from confluent_kafka import Message
from pydantic import BaseModel, ValidationError

from drl_trading_common.messaging.kafka_handler_decorator import kafka_handler


class SampleRequest(BaseModel):
    """Sample request model for decorator testing."""
    request_id: str
    value: int


class TestKafkaHandlerDecorator:
    """Test suite for @kafka_handler decorator."""

    def test_decorator_parses_valid_json_and_invokes_handler(self) -> None:
        """Test decorator successfully parses valid JSON and invokes handler."""
        # Given
        handler_id = "test_handler"
        mock_message = Mock(spec=Message)
        test_data = {"request_id": "req-123", "value": 42}
        mock_message.value.return_value = json.dumps(test_data).encode('utf-8')
        mock_message.topic.return_value = "test-topic"
        mock_message.partition.return_value = 0
        mock_message.offset.return_value = 100

        handler_invoked = False
        received_request = None

        @kafka_handler(SampleRequest, handler_id)
        def test_handler(request: SampleRequest, message: Message) -> None:
            nonlocal handler_invoked, received_request
            handler_invoked = True
            received_request = request

        # When
        test_handler(mock_message)

        # Then
        assert handler_invoked, "Handler should have been invoked"
        assert received_request is not None, "Request should have been parsed"
        assert received_request.request_id == "req-123"
        assert received_request.value == 42

    def test_decorator_raises_on_invalid_json(self) -> None:
        """Test decorator raises JSONDecodeError on invalid JSON."""
        # Given
        handler_id = "test_handler"
        mock_message = Mock(spec=Message)
        mock_message.value.return_value = b"invalid json {{"
        mock_message.topic.return_value = "test-topic"
        mock_message.partition.return_value = 0
        mock_message.offset.return_value = 100

        @kafka_handler(SampleRequest, handler_id)
        def test_handler(request: SampleRequest, message: Message) -> None:
            pass

        # When / Then
        with pytest.raises(json.JSONDecodeError):
            test_handler(mock_message)

    def test_decorator_raises_on_validation_error(self) -> None:
        """Test decorator raises ValidationError when Pydantic validation fails."""
        # Given
        handler_id = "test_handler"
        mock_message = Mock(spec=Message)
        invalid_data = {"request_id": "req-123"}  # Missing required 'value' field
        mock_message.value.return_value = json.dumps(invalid_data).encode('utf-8')
        mock_message.topic.return_value = "test-topic"
        mock_message.partition.return_value = 0
        mock_message.offset.return_value = 100

        @kafka_handler(SampleRequest, handler_id)
        def test_handler(request: SampleRequest, message: Message) -> None:
            pass

        # When / Then
        with pytest.raises(ValidationError):
            test_handler(mock_message)

    def test_decorator_propagates_handler_exceptions(self) -> None:
        """Test decorator propagates exceptions raised by handler."""
        # Given
        handler_id = "test_handler"
        mock_message = Mock(spec=Message)
        test_data = {"request_id": "req-123", "value": 42}
        mock_message.value.return_value = json.dumps(test_data).encode('utf-8')
        mock_message.topic.return_value = "test-topic"
        mock_message.partition.return_value = 0
        mock_message.offset.return_value = 100

        @kafka_handler(SampleRequest, handler_id)
        def test_handler(request: SampleRequest, message: Message) -> None:
            raise RuntimeError("Business logic error")

        # When / Then
        with pytest.raises(RuntimeError, match="Business logic error"):
            test_handler(mock_message)

    def test_decorator_extracts_request_id_for_logging(self) -> None:
        """Test decorator extracts request_id from message for logging context."""
        # Given
        handler_id = "test_handler"
        mock_message = Mock(spec=Message)
        test_data = {"request_id": "req-999", "value": 42}
        mock_message.value.return_value = json.dumps(test_data).encode('utf-8')
        mock_message.topic.return_value = "test-topic"
        mock_message.partition.return_value = 0
        mock_message.offset.return_value = 100

        captured_request_id = None

        @kafka_handler(SampleRequest, handler_id)
        def test_handler(request: SampleRequest, message: Message) -> None:
            nonlocal captured_request_id
            captured_request_id = request.request_id

        # When
        test_handler(mock_message)

        # Then
        assert captured_request_id == "req-999"

    def test_decorator_provides_message_to_handler(self) -> None:
        """Test decorator provides original Message object to handler."""
        # Given
        handler_id = "test_handler"
        mock_message = Mock(spec=Message)
        test_data = {"request_id": "req-123", "value": 42}
        mock_message.value.return_value = json.dumps(test_data).encode('utf-8')
        mock_message.topic.return_value = "test-topic"
        mock_message.partition.return_value = 0
        mock_message.offset.return_value = 100

        received_message = None

        @kafka_handler(SampleRequest, handler_id)
        def test_handler(request: SampleRequest, message: Message) -> None:
            nonlocal received_message
            received_message = message

        # When
        test_handler(mock_message)

        # Then
        assert received_message is mock_message
        assert received_message.topic() == "test-topic"
        assert received_message.partition() == 0
        assert received_message.offset() == 100
