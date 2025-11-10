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


@pytest.fixture(scope="session")
def seed_market_data() -> Generator[None, None, None]:
    """
    Seed TimescaleDB with test market data for E2E tests.

    Creates sample OHLCV data for BTCUSD, ETHUSD, and SOLUSD symbols
    covering the period 2024-01-01 00:00:00 to 2024-01-01 01:00:00.

    Also clears resampling state to ensure clean test runs without stale timestamps.

    This is a session-scoped fixture that runs once before all tests
    and cleans up after all tests complete.

    Yields:
        None (data is seeded in database)
    """
    import os
    from datetime import UTC, datetime, timedelta
    from pathlib import Path

    import psycopg2

    # Clear resampling state file to prevent using stale timestamps from previous test runs
    # This ensures the service fetches all data, not just incremental updates
    state_dir = Path("state")
    state_file = state_dir / "resampling_context.json"
    backup_file = state_dir / "resampling_context.json.backup"

    if state_file.exists():
        state_file.unlink()
        print(f"Cleared resampling state file: {state_file}")
    if backup_file.exists():
        backup_file.unlink()
        print(f"Cleared resampling backup file: {backup_file}")

    # Database connection parameters
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = int(os.getenv("DB_PORT", "5432"))
    db_name = os.getenv("DB_NAME", "marketdata")
    db_user = os.getenv("DB_USER", "postgres")
    db_password = os.getenv("DB_PASSWORD", "postgres")

    # Connect to database
    conn = psycopg2.connect(
        host=db_host,
        port=db_port,
        database=db_name,
        user=db_user,
        password=db_password,
    )
    conn.autocommit = True
    cursor = conn.cursor()

    try:
        # Define test symbols and time range
        symbols = ["BTCUSD", "ETHUSD", "SOLUSD"]
        # Start earlier to provide warmup data for indicators (RSI needs ~70 min for period=14 on 5m)
        start_time = datetime(2023, 12, 31, 22, 0, 0, tzinfo=UTC)  # 2 hours before test period
        base_timeframe = "1m"
        target_timeframe = "5m"

        # Generate 180 minutes of 1-minute data (2 hours warmup + 1 hour test period)
        # Also pre-generate corresponding 5m candles so warmup can find them
        for symbol in symbols:
            print(f"Seeding market data for {symbol}...")

            # Seed 1m data
            for minute in range(180):
                timestamp = start_time + timedelta(minutes=minute)

                # Generate sample OHLCV data (realistic-looking values)
                if symbol == "BTCUSD":
                    base_price = 42000.0 + (minute * 10)  # Trending up slightly
                elif symbol == "ETHUSD":
                    base_price = 2200.0 + (minute * 2)
                else:  # SOLUSD
                    base_price = 100.0 + (minute * 0.5)

                open_price = base_price
                high_price = base_price * 1.001  # 0.1% higher
                low_price = base_price * 0.999   # 0.1% lower
                close_price = base_price + (minute % 2 * 5)  # Slight variation
                volume = 1000 + (minute * 100)

                # Insert into market_data table (1m)
                cursor.execute(
                    """
                    INSERT INTO market_data (timestamp, symbol, timeframe, open_price, high_price, low_price, close_price, volume)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (timestamp, symbol, timeframe) DO NOTHING
                    """,
                    (timestamp, symbol, base_timeframe, open_price, high_price, low_price, close_price, volume),
                )

            # Seed 5m data (36 candles from 180 minutes)
            # This ensures warmup can find data in the target timeframe
            for candle_index in range(36):
                # 5m candles start at 22:00, 22:05, 22:10, etc.
                timestamp = start_time + timedelta(minutes=candle_index * 5)

                # Aggregate 5 minutes of 1m data into one 5m candle
                minute_offset = candle_index * 5
                if symbol == "BTCUSD":
                    base_price = 42000.0 + (minute_offset * 10)
                elif symbol == "ETHUSD":
                    base_price = 2200.0 + (minute_offset * 2)
                else:  # SOLUSD
                    base_price = 100.0 + (minute_offset * 0.5)

                open_price = base_price
                high_price = base_price * 1.005  # 0.5% higher over 5min
                low_price = base_price * 0.995   # 0.5% lower over 5min
                close_price = base_price + 25  # Price movement over 5min
                volume = 5000 + (candle_index * 500)  # Aggregate of 5 minutes

                # Insert into market_data table (5m)
                cursor.execute(
                    """
                    INSERT INTO market_data (timestamp, symbol, timeframe, open_price, high_price, low_price, close_price, volume)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (timestamp, symbol, timeframe) DO NOTHING
                    """,
                    (timestamp, symbol, target_timeframe, open_price, high_price, low_price, close_price, volume),
                )

            print(f"Seeded 180 rows (1m) + 36 rows (5m) for {symbol}")

        print("Market data seeding completed successfully")

        yield

    finally:
        # Cleanup: Delete test data (warmup + test period, both timeframes)
        print("Cleaning up test market data...")
        for symbol in symbols:
            # Delete 1m data
            cursor.execute(
                "DELETE FROM market_data WHERE symbol = %s AND timeframe = %s AND timestamp >= %s AND timestamp < %s",
                (symbol, "1m", start_time, start_time + timedelta(hours=3)),  # 3 hours total
            )
            # Delete 5m data
            cursor.execute(
                "DELETE FROM market_data WHERE symbol = %s AND timeframe = %s AND timestamp >= %s AND timestamp < %s",
                (symbol, "5m", start_time, start_time + timedelta(hours=3)),  # 3 hours total
            )
        cursor.close()
        conn.close()
        print("Test data cleanup completed")
