"""
End-to-end test fixtures for testing services with real infrastructure.

This module provides fixtures for testing services that are already running
with their infrastructure dependencies (Kafka, databases) started via docker-compose.

Usage pattern:
1. Start docker-compose (Kafka, DBs, etc.)
2. Start the service manually (e.g., `STAGE=ci python main.py`)
3. Run E2E tests that interact with the real service via Kafka

This tests the complete integration without mocking.
"""

import json
import time
from collections.abc import Generator
from typing import Any

import pytest
from confluent_kafka import Consumer, KafkaError, Producer


@pytest.fixture
def kafka_bootstrap_servers() -> str:
    """
    Kafka bootstrap servers for E2E tests.

    Assumes Kafka is running on localhost:9092 (started via docker-compose).
    Override with KAFKA_BOOTSTRAP_SERVERS env var if needed.

    Returns:
        Kafka bootstrap servers string
    """
    import os
    return os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")


@pytest.fixture
def kafka_producer(kafka_bootstrap_servers: str) -> Generator[Producer, None, None]:
    """
    Kafka producer for publishing test messages to service input topics.

    Use this to simulate upstream services or trigger service workflows.

    Args:
        kafka_bootstrap_servers: Kafka connection string

    Yields:
        Configured Kafka producer

    Example:
        def test_service(kafka_producer):
            kafka_producer.produce(
                topic='input.topic',
                key=b'test-key',
                value=json.dumps({'data': 'test'}).encode()
            )
            kafka_producer.flush()
    """
    producer = Producer({
        'bootstrap.servers': kafka_bootstrap_servers,
        'client.id': 'e2e-test-producer'
    })

    yield producer

    # Ensure all messages are sent before cleanup
    producer.flush(timeout=10)


@pytest.fixture
def kafka_consumer_factory(kafka_bootstrap_servers: str) -> Generator[Any, None, None]:
    """
    Factory for creating Kafka consumers for different topics.

    Creates a consumer that reads from the latest offset (only new messages).
    Useful for verifying service output without reading historical data.

    Args:
        kafka_bootstrap_servers: Kafka connection string

    Returns:
        Function that creates consumers subscribed to specified topics

    Example:
        def test_service(kafka_consumer_factory):
            consumer = kafka_consumer_factory(['output.topic'])
            # Trigger service...
            message = wait_for_message(consumer, timeout=10)
            assert message['status'] == 'completed'
    """
    consumers: list[Consumer] = []

    def create_consumer(topics: list[str]) -> Consumer:
        """
        Create a consumer subscribed to specified topics.

        Args:
            topics: List of topic names to subscribe to

        Returns:
            Configured Kafka consumer
        """
        import uuid
        consumer = Consumer({
            'bootstrap.servers': kafka_bootstrap_servers,
            'group.id': f'e2e-test-{uuid.uuid4().hex[:8]}',
            'auto.offset.reset': 'latest',  # Only read new messages
            'enable.auto.commit': False
        })
        consumer.subscribe(topics)
        consumers.append(consumer)
        return consumer

    yield create_consumer

    # Cleanup all consumers
    for consumer in consumers:
        consumer.close()


@pytest.fixture
def wait_for_kafka_message() -> Any:
    """
    Helper function to wait for a message on a Kafka consumer.

    Polls the consumer until a message arrives or timeout is reached.
    Automatically deserializes JSON messages.

    Returns:
        Function that waits for and returns the next message

    Example:
        def test_service(kafka_consumer_factory, wait_for_kafka_message):
            consumer = kafka_consumer_factory(['output.topic'])
            # Trigger service...
            message = wait_for_kafka_message(consumer, timeout=10)
            assert message['result'] == 'success'
    """
    def wait_for_message(
        consumer: Consumer,
        timeout: int = 30,
        expected_key: str | None = None
    ) -> dict[str, Any]:
        """
        Wait for a message from Kafka consumer.

        Args:
            consumer: Kafka consumer to poll
            timeout: Maximum seconds to wait for message
            expected_key: Optional key to filter messages (waits for specific key)

        Returns:
            Deserialized message as dictionary

        Raises:
            TimeoutError: If no message received within timeout
            AssertionError: If message has error
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            msg = consumer.poll(timeout=1.0)

            if msg is None:
                continue

            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                raise AssertionError(f"Kafka error: {msg.error()}")

            # Check if key matches (if filtering)
            if expected_key is not None:
                msg_key = msg.key().decode('utf-8') if msg.key() else None
                if msg_key != expected_key:
                    continue

            # Deserialize message
            try:
                value: dict[str, Any] = json.loads(msg.value().decode('utf-8'))
                return value
            except json.JSONDecodeError as e:
                raise AssertionError(f"Failed to deserialize message: {e}") from e

        raise TimeoutError(
            f"No message received within {timeout} seconds "
            f"(expected_key={expected_key})"
        )

    return wait_for_message


@pytest.fixture
def publish_kafka_message(kafka_producer: Producer) -> Any:
    """
    Helper function to publish messages to Kafka with automatic flushing.

    Simplifies publishing test messages with proper error handling.

    Args:
        kafka_producer: Configured Kafka producer

    Returns:
        Function that publishes messages

    Example:
        def test_service(publish_kafka_message):
            publish_kafka_message(
                topic='input.topic',
                key='AAPL',
                value={'symbol': 'AAPL', 'price': 150.0}
            )
    """
    def publish(
        topic: str,
        key: str,
        value: dict[str, Any],
        headers: dict[str, str] | None = None
    ) -> None:
        """
        Publish a message to Kafka topic.

        Args:
            topic: Kafka topic name
            key: Message key (for partitioning)
            value: Message payload (will be JSON-serialized)
            headers: Optional message headers

        Raises:
            AssertionError: If publishing fails
        """
        kafka_headers = None
        if headers:
            kafka_headers = [(k, v.encode('utf-8')) for k, v in headers.items()]

        try:
            kafka_producer.produce(
                topic=topic,
                key=key.encode('utf-8'),
                value=json.dumps(value).encode('utf-8'),
                headers=kafka_headers,
            )

            # Poll to trigger callbacks
            kafka_producer.poll(0)

            # Flush and wait for delivery
            remaining = kafka_producer.flush(timeout=10)

            if remaining > 0:
                raise AssertionError(
                    f"Failed to flush all messages to Kafka. {remaining} messages remaining in queue"
                )
        except Exception as e:
            raise AssertionError(f"Failed to publish message to topic '{topic}': {e}") from e

    return publish
