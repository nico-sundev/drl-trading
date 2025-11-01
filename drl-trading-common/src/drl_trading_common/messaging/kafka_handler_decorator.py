"""Decorator for Kafka message handlers with automatic request parsing."""

import json
import logging
from functools import wraps
from typing import Callable, Type, TypeVar

from confluent_kafka import Message
from pydantic import BaseModel, ValidationError

from drl_trading_common.messaging.kafka_message_handler import KafkaMessageHandler


logger = logging.getLogger(__name__)

TRequest = TypeVar('TRequest', bound=BaseModel)


def kafka_handler(
    request_type: Type[TRequest],
    handler_id: str
) -> Callable[[Callable[[TRequest, Message], None]], KafkaMessageHandler]:
    """
    Decorator for Kafka message handlers with automatic request parsing.

    Handles:
    - JSON decoding from message.value()
    - Pydantic model validation
    - Structured logging with handler_id, topic, partition, offset
    - Exception handling and re-raising for consumer retry logic

    Args:
        request_type: Pydantic model class to parse message into
        handler_id: Unique identifier for this handler (for logging)

    Returns:
        Decorator that transforms handler function signature from
        (request: TRequest, message: Message) -> None
        to
        (message: Message) -> None

    Example:
        @kafka_handler(FeaturePreprocessingRequest, HANDLER_ID_PREPROCESSING_REQUEST)
        def handle_preprocessing_request(request: FeaturePreprocessingRequest, message: Message) -> None:
            # Business logic here - request is already parsed and validated
            orchestrator.process(request)
    """
    def decorator(handler_func: Callable[[TRequest, Message], None]) -> KafkaMessageHandler:
        @wraps(handler_func)
        def wrapper(message: Message) -> None:
            request_id = None
            try:
                raw_json = message.value().decode('utf-8')
                request_dict = json.loads(raw_json)
                request_id = request_dict.get("request_id", "unknown")

                logger.info(
                    f"Processing {request_type.__name__}",
                    extra={
                        "handler_id": handler_id,
                        "topic": message.topic(),
                        "partition": message.partition(),
                        "offset": message.offset(),
                        "request_id": request_id,
                        "request_type": request_type.__name__,
                    }
                )

                request = request_type.model_validate(request_dict)
                handler_func(request, message)

                logger.debug(
                    f"{request_type.__name__} handled successfully: {request_id}",
                    extra={"handler_id": handler_id, "request_id": request_id}
                )

            except json.JSONDecodeError as e:
                logger.error(
                    f"Invalid JSON in {request_type.__name__}: {e}",
                    extra={
                        "handler_id": handler_id,
                        "topic": message.topic(),
                        "partition": message.partition(),
                        "offset": message.offset(),
                    },
                    exc_info=True
                )
                raise
            except ValidationError as e:
                logger.error(
                    f"Validation failed for {request_type.__name__}: {e}",
                    extra={
                        "handler_id": handler_id,
                        "topic": message.topic(),
                        "partition": message.partition(),
                        "offset": message.offset(),
                        "request_id": request_id,
                        "validation_errors": e.errors(),
                    },
                    exc_info=True
                )
                raise
            except Exception as e:
                logger.error(
                    f"Failed to process {request_type.__name__}: {e}",
                    extra={
                        "handler_id": handler_id,
                        "topic": message.topic(),
                        "partition": message.partition(),
                        "offset": message.offset(),
                        "request_id": request_id,
                    },
                    exc_info=True
                )
                raise

        return wrapper
    return decorator
