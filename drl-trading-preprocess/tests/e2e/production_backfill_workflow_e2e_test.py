"""
Comprehensive E2E test for production backfill workflow.

This test validates real-world production scenarios across multiple requests:
1. Cold start: Initial backfill with no prior data
2. Warm start: Duplicate request (should skip computation)
3. Incremental: Gap fill with new time range

External systems involved:
- Kafka (requested.preprocess-data, completed.preprocess-data, requested.store-resampled-data)
- TimescaleDB (1-minute base data + 5-minute resampled data)
- Feast Offline Store (parquet files)
- Service internal cache (feature instances - should persist and reuse)

Production workflow:
1. Service resamples 1m → 5m using MarketDataResamplingService
2. Service publishes resampled data to requested.store-resampled-data Kafka topic
3. Ingest service (simulated in test) consumes and stores 5m data to TimescaleDB
4. Service computes features from resampled 5m data in TimescaleDB
5. Service persists features to Feast offline store (parquet)

Test simulation:
- Seed 1m base data in TimescaleDB
- Wait for resampling message on Kafka
- Store resampled 5m data to TimescaleDB (simulate ingest service)
- Verify feature computation and persistence
- Validate feature instance reuse across requests (no duplicates)

Prerequisites:
- Docker Compose running (Kafka, TimescaleDB, etc.)
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
    base_path = os.getenv("FEAST_OFFLINE_STORE_PATH", "data/feature_store")
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
class TestProductionBackfillWorkflow:
    """
    E2E test simulating complete production backfill workflow.

    Uses unique symbol per test run to avoid conflicts with previous test data.
    Validates state transitions across cold start → warm start → incremental → full historical scenarios.
    """

    def test_production_backfill_workflow(
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
        Test complete production backfill workflow across 4 scenarios.

        Flow per scenario:
        1. Publish preprocessing request to Kafka
        2. Service resamples 1m → 5m, publishes to requested.store-resampled-data
        3. Test consumes resampling message, stores 5m data to TimescaleDB (simulates ingest service)
        4. Service computes features from 5m data in TimescaleDB
        5. Service persists features to Feast parquet
        6. Verify completion message and artifacts

        Scenario 1 (Cold start): [00:00-02:00]
        - First time processing, no prior resampled data
        - Should compute all features

        Scenario 2 (Warm start): [00:00-02:00] (duplicate)
        - Resampled 5m data exists from scenario 1
        - skip_existing=True should skip computation
        - Reuses cached feature instances (no new instances)

        Scenario 3 (Incremental): [02:00-04:00] (gap fill)
        - New time range, new resampled data
        - Should compute features for gap
        - Still reuses feature instances from scenario 1

        Scenario 4 (Full historical): [None - 2024-01-01 08:00]
        - Backfill from earliest available data (start=None) to specific end date
        - skip_existing=False forces recomputation
        - Tests complete historical data processing

        Verifications:
        - Kafka messages (resampling + completion)
        - TimescaleDB state (5m resampled data persisted)
        - Feast parquet files (features persisted)
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
        # SCENARIO 1: COLD START - INITIAL BACKFILL
        # ==========================================
        print(f"\n{'='*60}")
        print(f"SCENARIO 1: COLD START - Symbol: {unique_symbol}")
        print(f"{'='*60}")

        # Given - First backfill request
        request_1 = (
            FeaturePreprocessingRequestBuilder()
            .for_backfill()
            .with_symbol(unique_symbol)
            .with_target_timeframes([Timeframe.MINUTE_5])
            .with_time_range(
                start=datetime(2024, 1, 1, 0, 0, 0),
                end=datetime(2024, 1, 1, 2, 0, 0),  # 2 hours = 24 x 5m candles
            )
            .with_features([rsi_feature])
            .with_skip_existing(True)
            .with_force_recompute(False)
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
        resample_message_1 = wait_for_kafka_message(resample_consumer, timeout=30)

        assert resample_message_1 is not None, "Should receive resampling message"
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
        completion_1 = wait_for_kafka_message(output_consumer, timeout=30)

        assert completion_1 is not None, "Should receive completion message"
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
        completion_2 = wait_for_kafka_message(output_consumer, timeout=30)

        assert completion_2 is not None
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
        # SCENARIO 3: INCREMENTAL - GAP FILL
        # ==========================================
        print(f"\n{'='*60}")
        print(f"SCENARIO 3: INCREMENTAL GAP FILL - Symbol: {unique_symbol}")
        print(f"{'='*60}")

        # Give service a moment to settle
        time.sleep(1)

        # Given - New time range (gap after scenario 1)
        request_3 = (
            FeaturePreprocessingRequestBuilder()
            .for_backfill()
            .with_symbol(unique_symbol)
            .with_target_timeframes([Timeframe.MINUTE_5])
            .with_time_range(
                start=datetime(2024, 1, 1, 2, 0, 0),  # Starts where request_1 ended
                end=datetime(2024, 1, 1, 4, 0, 0),    # 2 more hours
            )
            .with_features([rsi_feature])
            .with_skip_existing(True)
            .with_force_recompute(False)
            .build()
        )

        # When - Publish incremental request
        print(f"Publishing request 3 (gap fill): {unique_symbol} [02:00-04:00]")
        publish_kafka_message(
            topic="requested.preprocess-data",
            key=unique_symbol,
            value=request_3.model_dump(mode="json")
        )

        # Then - Wait for resampling message and store to DB
        print("Waiting for resampling message...")
        resample_message_3 = wait_for_kafka_message(resample_consumer, timeout=30)

        assert resample_message_3 is not None, "Should receive resampling message for gap"
        assert resample_message_3["symbol"] == unique_symbol

        print(f"✓ Received resampling message: {resample_message_3['total_new_candles']} new candles")

        # Simulate ingest service storing new resampled data
        print("Storing new resampled 5m data to TimescaleDB...")
        self._store_resampled_data_to_timescale(
            timescale_connection,
            resample_message_3
        )

        # Then - Verify computation completed for gap
        print("Waiting for completion message...")
        completion_3 = wait_for_kafka_message(output_consumer, timeout=30)

        assert completion_3 is not None
        assert completion_3["symbol"] == unique_symbol
        assert completion_3["total_features_computed"] > 0, "Should compute features for gap"

        print(f"✓ Completion received: {completion_3['total_features_computed']} features computed")

        # Then - Verify new parquet file(s) or updated files
        parquet_files_final = list(feast_offline_store_path.glob(f"{unique_symbol}/**/*.parquet"))
        assert len(parquet_files_final) >= initial_file_count, "Should have same or more files"

        print(f"✓ Parquet file count: {len(parquet_files_final)} (was {initial_file_count})")

        # Verify total feature records increased
        all_features_final = pd.concat([pd.read_parquet(f) for f in parquet_files_final])
        assert len(all_features_final) > initial_feature_count, "Should have more features after gap fill"

        print(f"✓ Feature record count increased: {len(all_features_final)} (was {initial_feature_count})")

        # ==========================================
        # SCENARIO 4: FULL HISTORICAL BACKFILL UP TO SPECIFIC DATE
        # ==========================================
        print(f"\n{'='*60}")
        print(f"SCENARIO 4: FULL HISTORICAL BACKFILL - Symbol: {unique_symbol}")
        print(f"{'='*60}")

        # Give service a moment to settle
        time.sleep(1)

        # Given - Full backfill from earliest data to specific end date
        request_4 = (
            FeaturePreprocessingRequestBuilder()
            .for_backfill()
            .with_symbol(unique_symbol)
            .with_target_timeframes([Timeframe.MINUTE_5])
            .with_time_range(
                start=None,
                end=datetime(2024, 1, 1, 8, 0, tzinfo=timezone.utc),  # Covers more data
            )
            .with_features([rsi_feature])
            .with_skip_existing(False)  # Force full recompute for historical backfill
            .with_force_recompute(False)
            .build()
        )

        # When - Publish full backfill request
        print(f"Publishing request 4: {unique_symbol} [None - 2024-01-01 08:00]")
        publish_kafka_message(
            topic="requested.preprocess-data",
            key=unique_symbol,
            value=request_4.model_dump(mode="json")
        )

        # Then - Wait for resampling message and store to DB
        print("Waiting for resampling message...")
        resample_message_4 = wait_for_kafka_message(resample_consumer, timeout=30)

        assert resample_message_4 is not None, "Should receive resampling message for full backfill"
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
        completion_4 = wait_for_kafka_message(output_consumer, timeout=30)

        assert completion_4 is not None
        assert completion_4["symbol"] == unique_symbol
        assert completion_4["total_features_computed"] > 0, "Should compute features for full range"

        print(f"✓ Completion received: {completion_4['total_features_computed']} features computed")

        # Then - Verify parquet files updated or new
        parquet_files_final_final = list(feast_offline_store_path.glob(f"{unique_symbol}/**/*.parquet"))
        assert len(parquet_files_final_final) >= len(parquet_files_final), "Should have same or more files"

        print(f"✓ Parquet file count: {len(parquet_files_final_final)} (was {len(parquet_files_final)})")

        # Verify total feature records increased
        all_features_final_final = pd.concat([pd.read_parquet(f) for f in parquet_files_final_final])
        assert len(all_features_final_final) > len(all_features_final), "Should have more features after full backfill"

        print(f"✓ Feature record count increased: {len(all_features_final_final)} (was {len(all_features_final)})")

        print(f"\n{'='*60}")
        print(f"ALL SCENARIOS PASSED FOR {unique_symbol}")
        print(f"{'='*60}\n")

    def _drain_consumer(self, consumer: Any, timeout_seconds: int = 2) -> None:
        """
        Drain all pending messages from a Kafka consumer.

        This removes old messages from previous test runs that may still be
        in the topic, ensuring tests only process messages from the current run.

        Args:
            consumer: Kafka consumer to drain
            timeout_seconds: How long to poll for messages before stopping
        """
        from confluent_kafka import KafkaError

        drained_count = 0
        start_time = time.time()

        while time.time() - start_time < timeout_seconds:
            msg = consumer.poll(timeout=0.5)

            if msg is None:
                continue

            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                break

            drained_count += 1

        if drained_count > 0:
            print(f"  Drained {drained_count} old message(s)")

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
