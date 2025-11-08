# E2E Testing for AI Trading Services

## Overview

End-to-end tests validate complete service workflows with real infrastructure:
- Real Kafka message passing
- Real database interactions
- Real service processing logic

**Philosophy**: Test the service as it runs in production, not mocked simulations.

## Prerequisites

- Docker & Docker Compose installed
- Service dependencies built: `uv sync --group dev-all`
- Kafka & databases configured in `docker_compose/docker-compose.yml`

## Quick Start

### Option 1: Automated (Recommended)

```bash
# Run complete E2E test suite
./scripts/run_e2e_tests.sh

# Keep services running for debugging
./scripts/run_e2e_tests.sh --keep-running
```

### Option 2: Manual (For Development)

```bash
# 1. Start infrastructure
docker-compose -f docker_compose/docker-compose.yml up -d

# 2. Start service (in separate terminal)
cd drl-trading-preprocess
STAGE=local uv run python main.py

# 3. Run tests (in another terminal)
pytest tests/e2e/ -v

# 4. Cleanup
kill <service_pid>
docker-compose -f docker_compose/docker-compose.yml down
```

## Test Structure

```
tests/e2e/
├── conftest.py                      # Shared fixtures (Kafka helpers)
├── test_preprocess_service.py       # Preprocess service E2E tests
├── test_ingest_service.py           # Ingest service E2E tests (future)
└── test_end_to_end_workflow.py      # Multi-service workflows (future)
```

## Writing E2E Tests

### Basic Pattern

```python
def test_service_workflow(
    publish_kafka_message,           # Fixture: Publish to input topic
    kafka_consumer_factory,           # Fixture: Create consumer for output topic
    wait_for_kafka_message,          # Fixture: Wait for service output
):
    """Test complete service workflow."""
    # Given - Create consumer for output BEFORE triggering service
    consumer = kafka_consumer_factory(['output.topic'])
    time.sleep(2)  # Let consumer subscribe

    # When - Publish message to service input topic
    publish_kafka_message(
        topic='input.topic',
        key='test-key',
        value={'data': 'test'}
    )

    # Then - Wait for and verify service output
    result = wait_for_kafka_message(consumer, timeout=30)
    assert result['status'] == 'completed'
```

### Testing Multiple Messages

```python
def test_batch_processing(publish_kafka_message, kafka_consumer_factory, wait_for_kafka_message):
    """Test service handles multiple messages."""
    # Given
    consumer = kafka_consumer_factory(['output.topic'])
    time.sleep(2)

    test_items = ['item1', 'item2', 'item3']

    # When - Publish multiple messages
    for item in test_items:
        publish_kafka_message(
            topic='input.topic',
            key=item,
            value={'id': item, 'data': 'test'}
        )

    # Then - Receive all results
    results = []
    for _ in test_items:
        msg = wait_for_kafka_message(consumer, timeout=10)
        results.append(msg['id'])

    assert set(results) == set(test_items)
```

### Testing Error Handling

```python
def test_invalid_input_handling(publish_kafka_message, kafka_consumer_factory, wait_for_kafka_message):
    """Test service publishes errors to DLQ."""
    # Given - Listen to error topic
    consumer = kafka_consumer_factory(['error.topic'])
    time.sleep(2)

    # When - Send invalid message
    publish_kafka_message(
        topic='input.topic',
        key='invalid',
        value={'incomplete': 'data'}  # Missing required fields
    )

    # Then - Expect error message
    error_msg = wait_for_kafka_message(consumer, timeout=10)
    assert 'error' in error_msg
```

## Available Fixtures

### `kafka_bootstrap_servers`
Returns Kafka connection string (default: `localhost:9092`).

### `kafka_producer`
Kafka producer for publishing test messages.

**Usage:**
```python
def test_something(kafka_producer):
    kafka_producer.produce(
        topic='test-topic',
        key=b'key',
        value=json.dumps({'data': 'test'}).encode()
    )
    kafka_producer.flush()
```

### `kafka_consumer_factory`
Factory for creating consumers subscribed to topics.

**Usage:**
```python
def test_something(kafka_consumer_factory):
    consumer = kafka_consumer_factory(['output-topic-1', 'output-topic-2'])
    # Consumer is automatically cleaned up
```

### `publish_kafka_message`
Helper for easy message publishing with JSON serialization.

**Usage:**
```python
def test_something(publish_kafka_message):
    publish_kafka_message(
        topic='input.topic',
        key='test-key',
        value={'field': 'value'},
        headers={'correlation-id': '123'}  # Optional
    )
```

### `wait_for_kafka_message`
Wait for message from consumer with timeout and optional key filtering.

**Usage:**
```python
def test_something(kafka_consumer_factory, wait_for_kafka_message):
    consumer = kafka_consumer_factory(['output.topic'])

    # Wait for any message
    msg = wait_for_kafka_message(consumer, timeout=10)

    # Wait for specific key
    msg = wait_for_kafka_message(consumer, timeout=10, expected_key='AAPL')
```

## Configuration

### Environment Variables

- `KAFKA_BOOTSTRAP_SERVERS`: Override Kafka connection (default: `localhost:9092`)
- `STAGE`: Service configuration stage (use `ci` for E2E tests)

### Service Configuration

Create `config/application-ci.yaml` for E2E test configuration:

```yaml
infrastructure:
  kafka:
    bootstrap_servers: "localhost:9092"
  database:
    host: "localhost"
    port: 5432
    database: "test_db"

logging:
  level: "DEBUG"
  console_enabled: true
```

## Debugging

### View Service Logs

```bash
# While service is running
tail -f drl-trading-preprocess/logs/e2e-test.log

# After test run
cat drl-trading-preprocess/logs/e2e-test.log
```

### Check Kafka Messages

```bash
# List topics
docker-compose -f docker_compose/docker-compose.yml exec kafka \
    kafka-topics.sh --bootstrap-server localhost:9092 --list

# Consume messages from topic
docker-compose -f docker_compose/docker-compose.yml exec kafka \
    kafka-console-consumer.sh \
    --bootstrap-server localhost:9092 \
    --topic requested.preprocess-data \
    --from-beginning
```

### Keep Services Running

```bash
# Run with --keep-running flag
./scripts/run_e2e_tests.sh --keep-running

# Services remain running after tests
# Manually stop when done:
kill <service_pid>
docker-compose down
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: E2E Tests

on: [pull_request, push]

jobs:
  e2e-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'

      - name: Install uv
        run: pip install uv

      - name: Install dependencies
        run: uv sync --group dev-all

      - name: Run E2E tests
        run: ./scripts/run_e2e_tests.sh
```

## Best Practices

### ✅ DO

- **Test happy paths** - Most common scenarios
- **Use realistic data** - Actual market data formats
- **Test async behavior** - Services process in background
- **Clean up between tests** - Each test should be independent
- **Use meaningful timeouts** - Account for processing time

### ❌ DON'T

- **Mock Kafka** - Use real Kafka for E2E tests
- **Share state** - Tests should not depend on each other
- **Test every edge case** - Save that for unit tests
- **Ignore timing** - Give services time to process
- **Hard-code topic names** - Use configuration

## Troubleshooting

### "No message received within timeout"

- Check service is actually running: `ps aux | grep python`
- Verify Kafka is healthy: `docker-compose ps`
- Check service logs for errors
- Ensure topic names match configuration
- Increase timeout if processing is slow

### "Service failed to start"

- Check service logs: `cat logs/e2e-test.log`
- Verify configuration file exists: `ls config/application-ci.yaml`
- Check database connections
- Ensure ports are not already in use

### "Kafka connection refused"

- Ensure docker-compose is running
- Wait longer after docker-compose start (Kafka takes ~10s)
- Check Kafka logs: `docker-compose logs kafka`

## Performance

- **Single test**: ~5-10 seconds (including Kafka round-trip)
- **Full E2E suite**: ~2-5 minutes
- **With infrastructure startup**: ~15-20 minutes (first run)

## Extending

### Add New Service Tests

1. Create test file: `tests/e2e/test_<service>_service.py`
2. Use existing fixtures from `conftest.py`
3. Follow the Given/When/Then pattern
4. Update `run_e2e_tests.sh` to start your service

### Add Multi-Service Tests

```python
# tests/e2e/test_end_to_end_workflow.py
def test_ingestion_to_training_workflow(...):
    """Test complete data flow through multiple services."""
    # Trigger ingest service
    # Wait for preprocess output
    # Wait for training input
    # Verify end result
```

## Questions?

See existing tests in `tests/e2e/test_preprocess_service.py` for examples.
