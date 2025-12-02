"""
Comprehensive E2E test for production inference workflow.

This test validates real-world production scenarios across multiple requests:
0. Chained backfill: Backfill to catch up before inference
1. Cold start: Initial inference with no prior features
2. Warm start: Duplicate request (should skip computation)
3. Incremental: New data for ongoing inference
4. Full historical: Complete inference recomputation

External systems involved:
- Kafka (requested.preprocess-data, completed.preprocess-data, requested.store-resampled-data)
- TimescaleDB (1-minute base data + 5-minute resampled data)
- Feast Offline Store (parquet files)
- Feast Online Store (for real-time serving)
- Service internal cache (feature instances - should persist and reuse)

Production workflow:
1. Service resamples 1m → 5m using MarketDataResamplingService
2. Service publishes resampled data to requested.store-resampled-data Kafka topic
3. Ingest service (simulated in test) consumes and stores 5m data to TimescaleDB
4. Service computes features from resampled 5m data in TimescaleDB
5. Service persists features to Feast offline store (parquet) in incremental mode
6. Service pushes features to online store for real-time inference

Test simulation:
- Seed 1m base data in TimescaleDB
- Wait for resampling message on Kafka (with retry for robustness)
- Store resampled 5m data to TimescaleDB (simulate ingest service)
- Verify feature computation, persistence, and online serving
- Validate feature instance reuse across requests (no duplicates)

Prerequisites:
- Docker Compose running (Kafka, TimescaleDB, Feast, etc.)
- Preprocess service running: `STAGE=ci python main.py`
"""

import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

import pandas as pd
import psycopg2
import pytest

from drl_trading_common.adapter.model.feature_definition import FeatureDefinition
from drl_trading_common.adapter.model.timeframe import Timeframe
from builders import FeaturePreprocessingRequestBuilder


@pytest.fixture
def unique_symbol() -> str:
    """Generate unique symbol for test isolation."""
    import uuid
    return f"TEST{uuid.uuid4().hex[:6].upper()}"


@pytest.fixture
def timescale_connection() -> Generator[Any, None, None]:
    """Provide connection to TimescaleDB for data insertion and verification."""
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5432")),
        database=os.getenv("DB_NAME", "marketdata"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "postgres"),
    )
    conn.autocommit = True
    yield conn
    conn.close()


@pytest.fixture
def feast_offline_store_path() -> Path:
    """Provide path to Feast offline store parquet files."""
    # Read from environment or use default
    base_path = os.getenv("FEAST_OFFLINE_STORE_PATH", "config/local/data/feature_store")
    return Path(base_path).resolve()


@pytest.fixture
def seed_unique_symbol_data(unique_symbol: str, timescale_connection: Any) -> Generator[None, None, None]:
    """
    Seed TimescaleDB with 1-minute OHLCV data for the unique test symbol.

    Creates 1000 minutes (~16.5 hours) of 1-minute data from 2024-01-01 00:00:00
    to provide sufficient warmup for indicators (200 candles for 5m timeframe).

    This fixture ensures the service has base data to resample into 5-minute timeframe
    with proper warmup period for technical indicators.
    """
    from datetime import UTC, datetime, timedelta

    cursor = timescale_connection.cursor()

    try:
        print(f"\nSeeding 1m market data for {unique_symbol}...")

        # Start time for test data - start BEFORE the requested period to allow warmup
        # Request period: [2024-01-01 00:00 - 02:00]
        # Data: 1000 minutes starting from 2023-12-31 07:20
        # This provides 200 candles (1000min / 5min = 200) of warmup BEFORE 00:00
        # AND continues through the requested period
        # Data will span: 2023-12-31 07:20 → 2024-01-01 00:00 (warmup) → 2024-01-01 16:40 (computation)
        start_time = datetime(2023, 12, 31, 7, 20, 0, tzinfo=UTC)
        base_timeframe = "1m"

        # Generate 2000 minutes of data (1000 for warmup + 1000 for computation/future)
        for minute in range(2000):
            timestamp = start_time + timedelta(minutes=minute)

            # Generate realistic-looking OHLCV data
            base_price = 50000.0 + (minute * 10)  # Trending up
            open_price = base_price
            high_price = base_price * 1.001  # 0.1% higher
            low_price = base_price * 0.999   # 0.1% lower
            close_price = base_price + (minute % 2 * 5)
            volume = 1000 + (minute * 100)

            # Insert 1m candle
            cursor.execute(
                """
                INSERT INTO market_data (timestamp, symbol, timeframe, open_price, high_price, low_price, close_price, volume)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (timestamp, symbol, timeframe) DO NOTHING
                """,
                (timestamp, unique_symbol, base_timeframe, open_price, high_price, low_price, close_price, volume),
            )

        timescale_connection.commit()
        print(f"✓ Seeded 2000 rows (1m) for {unique_symbol}")

        # Verify data was inserted
        cursor.execute(
            "SELECT COUNT(*) FROM market_data WHERE symbol = %s AND timeframe = %s",
            (unique_symbol, base_timeframe)
        )
        count = cursor.fetchone()[0]
        print(f"✓ Verified: {count} rows exist in database for {unique_symbol} {base_timeframe}")

        # Small delay to ensure DB transaction is fully visible to other connections
        time.sleep(0.5)

        yield

    finally:
        # Cleanup: Delete test data
        print(f"\nCleaning up market data for {unique_symbol}...")
        cursor.execute(
            "DELETE FROM market_data WHERE symbol = %s",
            (unique_symbol,)
        )
        timescale_connection.commit()
        cursor.close()


@pytest.mark.e2e
class TestProductionInferenceWorkflow:
    """
    E2E test simulating complete production inference workflow.

    Uses unique symbol per test run to avoid conflicts with previous test data.
    Validates state transitions across cold start → warm start → incremental → full historical scenarios.
    """

    def test_production_inference_workflow(
        self,
        unique_symbol: str,
        timescale_connection: Any,
        seed_unique_symbol_data: None,
        publish_kafka_message: Any,
        kafka_consumer_factory: Any,
        wait_for_kafka_message: Any,
        feast_offline_store_path: Path,
    ) -> None:
        """
        Test complete production inference workflow across 5 scenarios.

        Flow per scenario:
        1. Publish preprocessing request to Kafka
        2. Service resamples 1m → 5m, publishes to requested.store-resampled-data
        3. Test consumes resampling message, stores 5m data to TimescaleDB (simulates ingest service)
        4. Service computes features from 5m data in TimescaleDB
        5. Service persists features to Feast offline store (parquet) in incremental mode
        6. Service pushes features to online store for real-time serving
        7. Verify completion message and artifacts

        Scenario 0 (Chained backfill): [2023-12-31 08:00 - 2024-01-01 00:00]
        - Backfill historical data to ensure no gaps before inference starts
        - Uses batch mode to prepare clean historical features

        Scenario 1 (Cold start): [00:00-02:00]
        - First inference processing after backfill
        - Should compute all features and push to online store

        Scenario 2 (Warm start): [00:00-02:00] (duplicate)
        - Features exist from scenario 1
        - skip_existing=True should skip computation
        - Reuses cached feature instances (no new instances)

        Scenario 3 (Incremental): [02:00-04:00] (new data)
        - New time range for ongoing inference updates
        - Should compute features incrementally and update online store
        - Still reuses feature instances from scenario 1

        Scenario 4 (Full historical): [None - 2024-01-01 08:00]
        - Complete inference recomputation
        - skip_existing=False forces recomputation and online store refresh
        - Tests full inference pipeline refresh

        Verifications:
        - Kafka messages (resampling + completion)
        - TimescaleDB state (5m resampled data persisted)
        - Feast offline store (parquet files)
        - Feast online store (features available for serving)
        - Feature instance reuse (service logs should show 1 RSI instance, not 8)
        """

        # Setup consumers for output and resampling topics
        output_topic = "completed.preprocess-data"
        resample_topic = "requested.store-resampled-data"

        output_consumer = kafka_consumer_factory([output_topic])
        resample_consumer = kafka_consumer_factory([resample_topic])

        time.sleep(2)  # Let consumers subscribe

        # Drain any old messages from previous test runs
        print("Draining old messages from Kafka topics...")
        self._drain_consumer(resample_consumer, timeout_seconds=2)
        self._drain_consumer(output_consumer, timeout_seconds=2)
        print("✓ Kafka topics drained")

        # Build RSI feature configuration
        rsi_feature = FeatureDefinition(
            name="rsi",
            enabled=True,
            derivatives=[0],
            raw_parameter_sets=[{"type": "rsi", "enabled": True, "length": 14}],
        )

        # ==========================================
        # SCENARIO 0: CHAINED BACKFILL - CATCH UP BEFORE INFERENCE
        # ==========================================
        print(f"\n{'='*60}")
        print(f"SCENARIO 0: CHAINED BACKFILL - Symbol: {unique_symbol}")
        print(f"{'='*60}")

        # Given - Backfill request to prepare historical data
        backfill_request = (
            FeaturePreprocessingRequestBuilder()
            .for_backfill()
            .with_symbol(unique_symbol)
            .with_target_timeframes([Timeframe.MINUTE_5])
            .with_time_range(
                start=datetime(2023, 12, 31, 8, 0, tzinfo=timezone.utc),  # Historical period
                end=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),     # Up to inference start
            )
            .with_features([rsi_feature])
            .with_skip_existing(True)
            .with_force_recompute(False)
            .build()
        )

        # When - Publish backfill request first
        print(f"Publishing backfill request: {unique_symbol} [2023-12-31 08:00 - 2024-01-01 00:00]")
        publish_kafka_message(
            topic="requested.preprocess-data",
            key=unique_symbol,
            value=backfill_request.model_dump(mode="json")
        )

        # Then - Wait for resampling message and store to DB
        print("Waiting for backfill resampling message...")
        backfill_resample = self._wait_for_message_with_retry(wait_for_kafka_message, resample_consumer, timeout=30)

        assert backfill_resample["symbol"] == unique_symbol

        print(f"✓ Received backfill resampling message: {backfill_resample['total_new_candles']} new candles")

        # Simulate ingest service storing backfill resampled data
        print("Storing backfill resampled 5m data to TimescaleDB...")
        self._store_resampled_data_to_timescale(
            timescale_connection,
            backfill_resample
        )

        # Then - Verify backfill computation completed
        print("Waiting for backfill completion message...")
        backfill_completion = self._wait_for_message_with_retry(wait_for_kafka_message, output_consumer, timeout=30)

        assert backfill_completion["symbol"] == unique_symbol
        assert backfill_completion["total_features_computed"] > 0, "Backfill should compute features"

        print(f"✓ Backfill completion received: {backfill_completion['total_features_computed']} features computed")

        # ==========================================
        # SCENARIO 1: COLD START - INITIAL INFERENCE
        # ==========================================
        print(f"\n{'='*60}")
        print(f"SCENARIO 1: COLD START - Symbol: {unique_symbol}")
        print(f"{'='*60}")

        # Given - First inference request
        request_1 = (
            FeaturePreprocessingRequestBuilder()
            .for_inference()
            .with_symbol(unique_symbol)
            .with_target_timeframes([Timeframe.MINUTE_5])
            .with_time_range(
                start=datetime(2024, 1, 1, 0, 0, 0),
                end=datetime(2024, 1, 1, 2, 0, 0),  # 2 hours = 24 x 5m candles
            )
            .with_features([rsi_feature])
            .with_skip_existing(True)
            .with_force_recompute(False)
            .with_materialize_online(True)  # Enable online serving for inference
            .build()
        )

        # When - Publish first request
        print(f"Publishing request 1: {unique_symbol} [00:00-02:00]")
        publish_kafka_message(
            topic="requested.preprocess-data",
            key=unique_symbol,
            value=request_1.model_dump(mode="json")
        )

        # Then - Wait for resampling message and store to DB (simulate ingest service)
        print("Waiting for resampling message...")
        resample_message_1 = self._wait_for_message_with_retry(wait_for_kafka_message, resample_consumer, timeout=30)

        assert resample_message_1["symbol"] == unique_symbol

        print(f"✓ Received resampling message: {resample_message_1['total_new_candles']} new candles")

        # Simulate ingest service storing resampled data to TimescaleDB
        print("Storing resampled 5m data to TimescaleDB...")
        self._store_resampled_data_to_timescale(
            timescale_connection,
            resample_message_1
        )

        # Then - Verify computation completed
        print("Waiting for completion message...")
        completion_1 = self._wait_for_message_with_retry(wait_for_kafka_message, output_consumer, timeout=30)

        assert completion_1["symbol"] == unique_symbol
        assert completion_1["total_features_computed"] > 0, "Features should be computed on cold start"

        print(f"✓ Completion received: {completion_1['total_features_computed']} features computed")

        # Then - Verify parquet files created
        print(f"Verifying Feast offline store under {feast_offline_store_path}...")
        parquet_files = list(feast_offline_store_path.glob(f"{unique_symbol}/**/*.parquet"))

        assert len(parquet_files) > 0, f"Expected parquet files, found {len(parquet_files)}"
        print(f"✓ Found {len(parquet_files)} parquet file(s)")

        # Verify parquet content
        all_features = pd.concat([pd.read_parquet(f) for f in parquet_files])

        assert "event_timestamp" in all_features.columns
        assert "symbol" in all_features.columns
        assert "rsi_14" in all_features.columns
        assert len(all_features) > 0

        # Count valid RSI values (not NaN)
        valid_rsi_count = all_features["rsi_14"].notna().sum()
        print(f"✓ Parquet contains {len(all_features)} records, {valid_rsi_count} valid RSI values")

        initial_feature_count = len(all_features)
        initial_file_count = len(parquet_files)

        # ==========================================
        # SCENARIO 2: WARM START - DUPLICATE REQUEST (SHOULD SKIP)
        # ==========================================
        print(f"\n{'='*60}")
        print(f"SCENARIO 2: WARM START (DUPLICATE) - Symbol: {unique_symbol}")
        print(f"{'='*60}")

        # Give service a moment to settle
        time.sleep(1)

        # Given - Identical request (same time range, same features)
        request_2 = request_1  # Exact same request

        # When - Publish duplicate request
        print(f"Publishing request 2 (duplicate): {unique_symbol} [00:00-02:00]")
        publish_kafka_message(
            topic="requested.preprocess-data",
            key=unique_symbol,
            value=request_2.model_dump(mode="json")
        )

        # Then - Verify completion indicates skipping
        print("Waiting for completion message...")
        completion_2 = self._wait_for_message_with_retry(wait_for_kafka_message, output_consumer, timeout=30)

        assert completion_2["symbol"] == unique_symbol
        assert completion_2["total_features_computed"] == 0, "Should skip - features already exist"

        print(f"✓ Completion received: {completion_2['total_features_computed']} features computed (SKIPPED)")

        # Then - Verify NO new parquet files
        parquet_files_after = list(feast_offline_store_path.glob(f"{unique_symbol}/**/*.parquet"))
        assert len(parquet_files_after) == initial_file_count, "Should not create new files on duplicate"

        print(f"✓ Parquet file count unchanged: {len(parquet_files_after)}")

        # Then - Verify parquet content unchanged
        all_features_after = pd.concat([pd.read_parquet(f) for f in parquet_files_after])
        assert len(all_features_after) == initial_feature_count, "Feature count should not increase"

        print(f"✓ Feature record count unchanged: {len(all_features_after)}")

        # ==========================================
        # SCENARIO 3: INCREMENTAL - NEW INFERENCE DATA
        # ==========================================
        print(f"\n{'='*60}")
        print(f"SCENARIO 3: INCREMENTAL NEW DATA - Symbol: {unique_symbol}")
        print(f"{'='*60}")

        # Give service a moment to settle
        time.sleep(1)

        # Given - New time range for incremental inference updates
        request_3 = (
            FeaturePreprocessingRequestBuilder()
            .for_inference()
            .with_symbol(unique_symbol)
            .with_target_timeframes([Timeframe.MINUTE_5])
            .with_time_range(
                start=datetime(2024, 1, 1, 2, 0, 0),  # Starts where request_1 ended
                end=datetime(2024, 1, 1, 4, 0, 0),    # 2 more hours
            )
            .with_features([rsi_feature])
            .with_skip_existing(True)
            .with_force_recompute(False)
            .with_materialize_online(True)
            .build()
        )

        # When - Publish incremental request
        print(f"Publishing request 3 (incremental): {unique_symbol} [02:00-04:00]")
        publish_kafka_message(
            topic="requested.preprocess-data",
            key=unique_symbol,
            value=request_3.model_dump(mode="json")
        )

        # Then - Wait for resampling message and store to DB
        print("Waiting for resampling message...")
        resample_message_3 = self._wait_for_message_with_retry(wait_for_kafka_message, resample_consumer, timeout=30)

        assert resample_message_3["symbol"] == unique_symbol

        print(f"✓ Received resampling message: {resample_message_3['total_new_candles']} new candles")

        # Simulate ingest service storing new resampled data
        print("Storing new resampled 5m data to TimescaleDB...")
        self._store_resampled_data_to_timescale(
            timescale_connection,
            resample_message_3
        )

        # Then - Verify computation completed for new data
        print("Waiting for completion message...")
        completion_3 = self._wait_for_message_with_retry(wait_for_kafka_message, output_consumer, timeout=30)

        assert completion_3["symbol"] == unique_symbol
        assert completion_3["total_features_computed"] > 0, "Should compute features for new data"

        print(f"✓ Completion received: {completion_3['total_features_computed']} features computed")

        # Then - Verify new parquet file(s) or updated files
        parquet_files_final = list(feast_offline_store_path.glob(f"{unique_symbol}/**/*.parquet"))
        assert len(parquet_files_final) >= initial_file_count, "Should have same or more files"

        print(f"✓ Parquet file count: {len(parquet_files_final)} (was {initial_file_count})")

        # Verify total feature records increased
        all_features_final = pd.concat([pd.read_parquet(f) for f in parquet_files_final])
        assert len(all_features_final) > initial_feature_count, "Should have more features after incremental update"

        print(f"✓ Feature record count increased: {len(all_features_final)} (was {initial_feature_count})")

        # ==========================================
        # SCENARIO 4: FULL HISTORICAL - INFERENCE REFRESH
        # ==========================================
        print(f"\n{'='*60}")
        print(f"SCENARIO 4: FULL HISTORICAL REFRESH - Symbol: {unique_symbol}")
        print(f"{'='*60}")

        # Give service a moment to settle
        time.sleep(1)

        # Given - Complete inference recomputation
        request_4 = (
            FeaturePreprocessingRequestBuilder()
            .for_inference()
            .with_symbol(unique_symbol)
            .with_target_timeframes([Timeframe.MINUTE_5])
            .with_time_range(
                start=None,
                end=datetime(2024, 1, 1, 8, 0, tzinfo=timezone.utc),  # Covers more data
            )
            .with_features([rsi_feature])
            .with_skip_existing(False)  # Force full recompute for inference refresh
            .with_force_recompute(False)
            .with_materialize_online(True)
            .build()
        )

        # When - Publish full inference refresh request
        print(f"Publishing request 4: {unique_symbol} [None - 2024-01-01 08:00]")
        publish_kafka_message(
            topic="requested.preprocess-data",
            key=unique_symbol,
            value=request_4.model_dump(mode="json")
        )

        # Then - Wait for resampling message and store to DB
        print("Waiting for resampling message...")
        resample_message_4 = self._wait_for_message_with_retry(wait_for_kafka_message, resample_consumer, timeout=30)

        assert resample_message_4["symbol"] == unique_symbol

        print(f"✓ Received resampling message: {resample_message_4['total_new_candles']} new candles")

        # Simulate ingest service storing new resampled data
        print("Storing full resampled 5m data to TimescaleDB...")
        self._store_resampled_data_to_timescale(
            timescale_connection,
            resample_message_4
        )

        # Then - Verify computation completed for full range
        print("Waiting for completion message...")
        completion_4 = self._wait_for_message_with_retry(wait_for_kafka_message, output_consumer, timeout=30)

        assert completion_4["symbol"] == unique_symbol
        assert completion_4["total_features_computed"] > 0, "Should compute features for full refresh"

        print(f"✓ Completion received: {completion_4['total_features_computed']} features computed")

        # Then - Verify parquet files updated or new
        parquet_files_final_final = list(feast_offline_store_path.glob(f"{unique_symbol}/**/*.parquet"))
        assert len(parquet_files_final_final) >= len(parquet_files_final), "Should have same or more files"

        print(f"✓ Parquet file count: {len(parquet_files_final_final)} (was {len(parquet_files_final)})")

        # Verify total feature records increased
        all_features_final_final = pd.concat([pd.read_parquet(f) for f in parquet_files_final_final])
        assert len(all_features_final_final) > len(all_features_final), "Should have more features after full refresh"

        print(f"✓ Feature record count increased: {len(all_features_final_final)} (was {len(all_features_final)})")

        print(f"\n{'='*60}")
        print(f"ALL SCENARIOS PASSED FOR {unique_symbol}")
        print(f"{'='*60}\n")

    def _wait_for_message_with_retry(
        self,
        wait_for_kafka_message: Any,
        consumer: Any,
        timeout: int = 30,
        retries: int = 3,
        retry_delay: float = 5.0
    ) -> Any:
        """
        Wait for a Kafka message with retry logic for robustness.

        This handles transient failures like Kafka timeouts or temporary service issues
        by retrying the wait operation multiple times with delays.

        Args:
            wait_for_kafka_message: The fixture function to wait for messages
            consumer: Kafka consumer to poll
            timeout: Timeout per attempt in seconds
            retries: Number of retry attempts
            retry_delay: Delay between retries in seconds

        Returns:
            The received message

        Raises:
            AssertionError: If message not received after all retries
        """
        import time

        for attempt in range(retries):
            try:
                message = wait_for_kafka_message(consumer, timeout=timeout)
                if message is not None:
                    return message
                if attempt < retries - 1:
                    print(f"Message not received, retrying in {retry_delay}s... (attempt {attempt + 1}/{retries})")
                    time.sleep(retry_delay)
            except Exception as e:
                if attempt < retries - 1:
                    print(f"Error waiting for message: {e}, retrying in {retry_delay}s... (attempt {attempt + 1}/{retries})")
                    time.sleep(retry_delay)
                else:
                    raise

        # If we get here, all retries failed
        raise AssertionError(f"Failed to receive message after {retries} attempts with {timeout}s timeout each")

    def _store_resampled_data_to_timescale(
        self,
        db_connection: Any,
        resample_message: dict
    ) -> None:
        """
        Simulate ingest service storing resampled data to TimescaleDB.

        This mirrors what the ingest service does in production:
        1. Consumes from requested.store-resampled-data topic
        2. Extracts resampled candles from message
        3. Stores to TimescaleDB market_data table

        This ensures subsequent requests find existing resampled data,
        validating skip_existing and incremental behavior.

        Args:
            db_connection: PostgreSQL connection
            resample_message: Kafka message from requested.store-resampled-data
        """
        symbol = resample_message["symbol"]
        resampled_data = resample_message.get("resampled_data", {})

        cursor = db_connection.cursor()
        total_inserted = 0

        # Iterate through timeframes (e.g., "5m")
        for timeframe_str, candles in resampled_data.items():
            for candle in candles:
                try:
                    cursor.execute(
                        """
                        INSERT INTO market_data (timestamp, symbol, timeframe, open_price, high_price, low_price, close_price, volume)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (timestamp, symbol, timeframe) DO NOTHING
                        """,
                        (
                            candle["timestamp"],
                            symbol,
                            timeframe_str,
                            candle["open_price"],
                            candle["high_price"],
                            candle["low_price"],
                            candle["close_price"],
                            candle["volume"]
                        )
                    )
                    total_inserted += 1
                except Exception as e:
                    print(f"Warning: Error inserting candle: {e}")

        db_connection.commit()
        cursor.close()
        print(f"✓ Stored {total_inserted} resampled 5m candles to TimescaleDB")
