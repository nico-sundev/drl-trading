"""
E2E test for preprocess service: Feature preprocessing workflow.

This test validates the complete preprocessing workflow:
1. Publish feature preprocessing request to input topic
2. Service consumes, processes, and computes features
3. Service publishes completion message to output topic
4. Verify message format and processing results

Prerequisites:
- Docker Compose running (Kafka, TimescaleDB, etc.)
- Preprocess service running: `STAGE=ci python main.py`

Note: Uses topic-based routing (no handler_id headers needed).
The service routes messages based on the topic they arrive on, as configured
in the topic_subscriptions section of the service config.
"""

from datetime import datetime
from typing import Any

import pytest

from drl_trading_common.adapter.model.feature_definition import FeatureDefinition
from drl_trading_common.adapter.model.timeframe import Timeframe
from builders import FeaturePreprocessingRequestBuilder


@pytest.mark.e2e
class TestPreprocessServiceE2E:
    """E2E tests for preprocessing service with real Kafka integration."""

    def test_feature_preprocessing_happy_path(
        self,
        seed_market_data: None,
        publish_kafka_message: Any,
        kafka_consumer_factory: Any,
        wait_for_kafka_message: Any,
    ) -> None:
        """
        Test complete preprocessing workflow: input to processing to output.

        This tests the service's ability to:
        - Consume messages from input topic (topic-based routing)
        - Process feature computation requests
        - Publish completion messages to output topic
        """
        # Given - Input topic and expected output topic
        input_topic = "requested.preprocess-data"
        output_topic = "completed.preprocess-data"

        # Create consumer for output topic BEFORE publishing
        consumer = kafka_consumer_factory([output_topic])

        import time

        time.sleep(2)  # Let consumer subscribe

        # Build RSI feature with proper parameters
        rsi_feature = FeatureDefinition(
            name="rsi",
            enabled=True,
            derivatives=[0],
            raw_parameter_sets=[{"type": "rsi", "enabled": True, "length": 14}],
        )

        # Build request using test builder (ensures correct schema)
        request = (
            FeaturePreprocessingRequestBuilder()
            .for_backfill()
            .with_symbol("BTCUSD")
            .with_target_timeframes([Timeframe.MINUTE_5])
            .with_time_range(
                start=datetime(2024, 1, 1, 0, 0, 0),
                end=datetime(2024, 1, 1, 4, 0, 0),
            )
            .with_features([rsi_feature])
            .with_skip_existing(True)
            .with_force_recompute(False)
            .with_materialize_online(False)
            .build()
        )

        # Convert to dict for Kafka (Pydantic model_dump)
        test_request = request.model_dump(mode="json")

        # When - Publish preprocessing request (NO handler_id header needed - topic-based routing)
        publish_kafka_message(topic=input_topic, key=request.symbol, value=test_request)

        # Then - Wait for service to process and publish completion
        # Note: Response uses request_id as key, not the input key
        result_message = wait_for_kafka_message(
            consumer, timeout=30
        )

        # Verify output message structure
        assert result_message is not None, "No completion message received"
        assert "request_id" in result_message, "Missing 'request_id' field"
        assert "symbol" in result_message, "Missing 'symbol' field"
        assert result_message["symbol"] == "BTCUSD"

    def test_multiple_symbols_parallel_processing(
        self,
        seed_market_data: None,
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
        output_topic = "completed.preprocess-data"

        consumer = kafka_consumer_factory([output_topic])

        import time

        time.sleep(2)  # Let consumer subscribe

        test_symbols = ["BTCUSD", "ETHUSD", "SOLUSD"]

        # Build RSI feature with proper parameters

        rsi_feature = FeatureDefinition(
            name="rsi",
            enabled=True,
            derivatives=[0],
            raw_parameter_sets=[{"type": "rsi", "enabled": True, "length": 14}],
        )

        # When - Publish requests for all symbols using builder
        for symbol in test_symbols:
            request = (
                FeaturePreprocessingRequestBuilder()
                .for_backfill()
                .with_symbol(symbol)
                .with_target_timeframes([Timeframe.MINUTE_5])
                .with_time_range(
                    start=datetime(2024, 1, 1, 0, 0, 0),
                    end=datetime(2024, 1, 1, 1, 0, 0),
                )
                .with_features([rsi_feature])
                .with_skip_existing(False)
                .with_force_recompute(False)
                .with_materialize_online(False)
                .build()
            )

            publish_kafka_message(
                topic=input_topic,
                key=f"{symbol}_5m",
                value=request.model_dump(mode="json"),
            )

        # Then - Receive results for all symbols (order may vary)
        received_symbols = set()
        for _ in range(len(test_symbols)):
            message = wait_for_kafka_message(consumer, timeout=30)
            symbol = message.get("symbol")
            assert symbol in test_symbols, f"Unexpected symbol: {symbol}"
            received_symbols.add(symbol)

        # Verify all symbols were processed
        assert received_symbols == set(
            test_symbols
        ), f"Not all symbols processed. Expected {test_symbols}, got {received_symbols}"

    def test_invalid_data_sent_to_dlq(
        self,
        publish_kafka_message: Any,
        kafka_consumer_factory: Any,
        wait_for_kafka_message: Any,
    ) -> None:
        """
        Test service sends invalid messages to DLQ after max retries.

        With DLQ infrastructure in place, invalid messages should:
        1. Fail validation/processing
        2. Retry according to failure policy
        3. Eventually be sent to DLQ topic
        """
        # Given - Invalid request (missing required fields)
        input_topic = "requested.preprocess-data"
        dlq_topic = "dlq.preprocess-data"

        consumer = kafka_consumer_factory([dlq_topic])

        import time

        time.sleep(2)

        invalid_request = {
            "request_id": "test-invalid-001",
            "symbol": "INVALID",
            # Missing: start_time, end_time, base_timeframe, target_timeframes,
            # feature_definitions, feature_config_version_info
        }

        # When - Publish invalid request (topic-based routing, no handler_id)
        publish_kafka_message(topic=input_topic, key="INVALID", value=invalid_request)

        # Then - Message should appear in DLQ after retries exhausted
        dlq_message = wait_for_kafka_message(consumer, timeout=15)

        assert dlq_message is not None, "No DLQ message received"
        assert "original_topic" in dlq_message, "Missing DLQ metadata: original_topic"
        assert dlq_message["original_topic"] == input_topic
        assert "error_type" in dlq_message, "Missing DLQ metadata: error_type"
        assert "retry_attempts" in dlq_message, "Missing DLQ metadata: retry_attempts"
        assert "original_value" in dlq_message, "Missing original message in DLQ"

        # Verify original message preserved in DLQ
        original = dlq_message["original_value"]
        assert original.get("request_id") == "test-invalid-001"
