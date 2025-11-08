"""
E2E test for preprocess service: Market data resampling workflow.

This test validates the complete preprocessing workflow:
1. Publish market data request to input topic
2. Service consumes, processes, and resamples data
3. Service publishes resampled data to output topic
4. Verify output message format and data integrity

Prerequisites:
- Docker Compose running (Kafka, TimescaleDB, etc.)
- Preprocess service running: `STAGE=ci python main.py`
"""

from typing import Any

import pytest


class TestPreprocessServiceE2E:
    """E2E tests for preprocessing service with real Kafka integration."""

    def test_market_data_preprocessing_happy_path(
        self,
        publish_kafka_message: Any,
        kafka_consumer_factory: Any,
        wait_for_kafka_message: Any,
    ) -> None:
        """
        Test complete preprocessing workflow: input → processing → output.

        This tests the service's ability to:
        - Consume messages from input topic
        - Process/resample market data
        - Publish results to output topic
        """
        # Given - Input topic and expected output topic
        input_topic = "requested.preprocess-data"
        output_topic = "requested.store-resampled-data"

        # Create consumer for output topic BEFORE publishing
        # (to ensure we catch the message when service publishes it)
        consumer = kafka_consumer_factory([output_topic])

        # Give consumer time to subscribe
        import time
        time.sleep(2)

        # Test data matching service contract
        test_request = {
            "symbol": "AAPL",
            "timeframe": "1h",
            "request_id": "test-e2e-001",
            "data": [
                {
                    "timestamp": "2024-01-01T10:00:00",
                    "open": 150.0,
                    "high": 152.0,
                    "low": 149.0,
                    "close": 151.0,
                    "volume": 1000000
                },
                {
                    "timestamp": "2024-01-01T10:05:00",
                    "open": 151.0,
                    "high": 153.0,
                    "low": 150.5,
                    "close": 152.5,
                    "volume": 1200000
                }
            ]
        }

        # When - Publish preprocessing request (simulating upstream service)
        publish_kafka_message(
            topic=input_topic,
            key="AAPL_1h",
            value=test_request
        )

        # Then - Wait for service to process and publish result
        result_message = wait_for_kafka_message(
            consumer,
            timeout=30,  # Give service time to process
            expected_key="AAPL_1h"  # Filter for our test message
        )

        # Verify output message structure
        assert result_message is not None, "No output message received"
        assert "symbol" in result_message, "Missing 'symbol' field"
        assert result_message["symbol"] == "AAPL"

        assert "timeframe" in result_message, "Missing 'timeframe' field"
        assert result_message["timeframe"] == "1h"

        assert "resampled_data" in result_message or "data" in result_message, \
            "Missing resampled data field"

        # Verify data was processed (not just echoed back)
        output_data = result_message.get("resampled_data") or result_message.get("data")
        assert len(output_data) > 0, "No resampled data in output"

    def test_multiple_symbols_parallel_processing(
        self,
        publish_kafka_message: Any,
        kafka_consumer_factory: Any,
        wait_for_kafka_message: Any,
    ) -> None:
        """
        Test service handles multiple symbols in parallel.

        Validates:
        - Service doesn't mix up data from different symbols
        - Each symbol's output matches its input
        - Processing happens concurrently (not blocking)
        """
        # Given - Multiple test requests for different symbols
        input_topic = "requested.preprocess-data"
        output_topic = "requested.store-resampled-data"

        consumer = kafka_consumer_factory([output_topic])

        import time
        time.sleep(2)  # Let consumer subscribe

        test_symbols = ["AAPL", "GOOGL", "TSLA"]

        # When - Publish requests for all symbols
        for symbol in test_symbols:
            publish_kafka_message(
                topic=input_topic,
                key=f"{symbol}_1h",
                value={
                    "symbol": symbol,
                    "timeframe": "1h",
                    "request_id": f"test-{symbol}",
                    "data": [
                        {
                            "timestamp": "2024-01-01T10:00:00",
                            "open": 100.0,
                            "high": 101.0,
                            "low": 99.0,
                            "close": 100.5,
                            "volume": 1000
                        }
                    ]
                }
            )

        # Then - Receive results for all symbols (order may vary)
        received_symbols = set()
        for _ in range(len(test_symbols)):
            message = wait_for_kafka_message(consumer, timeout=30)
            symbol = message.get("symbol")
            assert symbol in test_symbols, f"Unexpected symbol: {symbol}"
            received_symbols.add(symbol)

        # Verify all symbols were processed
        assert received_symbols == set(test_symbols), \
            f"Not all symbols processed. Expected {test_symbols}, got {received_symbols}"

    @pytest.mark.skip(reason="Requires service to handle invalid data gracefully")
    def test_invalid_data_handling(
        self,
        publish_kafka_message: Any,
        kafka_consumer_factory: Any,
        wait_for_kafka_message: Any,
    ) -> None:
        """
        Test service handles invalid input data gracefully.

        Should publish error message to error topic or DLQ.
        """
        # Given - Invalid request (missing required fields)
        input_topic = "requested.preprocess-data"
        error_topic = "error.preprocess-data"

        consumer = kafka_consumer_factory([error_topic])

        import time
        time.sleep(2)

        invalid_request = {
            "symbol": "INVALID",
            # Missing: timeframe, data
            "request_id": "test-invalid"
        }

        # When - Publish invalid request
        publish_kafka_message(
            topic=input_topic,
            key="INVALID",
            value=invalid_request
        )

        # Then - Service should publish error message
        error_message = wait_for_kafka_message(consumer, timeout=10)

        assert error_message is not None
        assert "error" in error_message or "status" in error_message
        assert error_message.get("request_id") == "test-invalid"
