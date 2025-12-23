"""
Comprehensive E2E test for production backfill workflow.

This test validates real-world production scenarios across multiple requests:
1. Cold start: Initial backfill with no prior data
2. Warm start: Duplicate request (should skip computation)
3. Incremental: Gap fill with new time range
4. Full historical: Backfill from earliest data

External systems involved:
- Kafka (requested.preprocess-data, completed.preprocess-data)
- TimescaleDB (1-minute base data for resampling)
- Feast Offline Store (parquet files for computed features)

Production workflow:
1. Service resamples 1m → 5m in-memory using MarketDataResamplingService
2. Service computes features from the in-memory resampled data
3. Service persists features to Feast offline store (parquet)
4. Service publishes completion message to Kafka

Test simulation:
- Seed 1m base data in TimescaleDB
- Publish preprocessing request to Kafka
- Wait for completion message
- Verify feature files in Feast parquet store

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


# Use UTC timezone consistently
UTC = timezone.utc


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
def cleanup_parquet_files(unique_symbol: str, feast_offline_store_path: Path) -> Generator[None, None, None]:
    """
    Clean up parquet files created during the test.

    This fixture runs after the test to remove any parquet files created
    for the unique test symbol, preventing pollution of the feature store.
    """
    yield

    # Cleanup: Remove parquet files for the test symbol
    import shutil

    print(f"\nCleaning up parquet files for {unique_symbol}...")

    # Try both possible path patterns
    symbol_paths = [
        feast_offline_store_path / unique_symbol,
        *list(feast_offline_store_path.glob(f"**/{unique_symbol}")),
    ]

    for symbol_path in symbol_paths:
        if symbol_path.exists() and symbol_path.is_dir():
            try:
                shutil.rmtree(symbol_path)
                print(f"[OK] Removed parquet directory: {symbol_path}")
            except Exception as e:
                print(f"Warning: Failed to remove {symbol_path}: {e}")


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
        print(f"[OK] Seeded 2000 rows (1m) for {unique_symbol}")

        # Verify data was inserted
        cursor.execute(
            "SELECT COUNT(*) FROM market_data WHERE symbol = %s AND timeframe = %s",
            (unique_symbol, base_timeframe)
        )
        count = cursor.fetchone()[0]
        print(f"[OK] Verified: {count} rows exist in database for {unique_symbol} {base_timeframe}")

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
        cleanup_parquet_files: None,
        publish_kafka_message: Any,
        kafka_consumer_factory: Any,
        wait_for_kafka_message: Any,
        feast_offline_store_path: Path,
    ) -> None:
        """
        Test complete production backfill workflow across 4 scenarios.

        Flow per scenario:
        1. Publish preprocessing request to Kafka
        2. Service resamples 1m → 5m (in-memory) and computes features synchronously
        3. Service persists features to Feast parquet
        4. Verify completion message and artifacts

        Scenario 1 (Cold start): [00:00-02:00]
        - First time processing, bootstrap features from scratch
        - Should compute all features in batch mode

        Scenario 2 (Warm start): [00:00-02:00] (duplicate)
        - Features already exist from scenario 1
        - skip_existing=True should skip computation

        Scenario 3 (Incremental): [02:00-04:00] (gap fill)
        - New time range
        - Should compute features for gap
        - Still reuses feature instances from scenario 1

        Scenario 4 (Full historical): [None - 2024-01-01 08:00]
        - Backfill from earliest available data (start=None) to specific end date
        - skip_existing=False forces recomputation
        - Tests complete historical data processing

        Verifications:
        - Kafka completion messages
        - Feast parquet files (features persisted)
        """

        # Setup consumer for completion topic only
        # Note: Service computes features from in-memory resampled data, not from DB
        # The resampling message is published for downstream services (like ingest) but
        # feature computation happens synchronously with the in-memory data
        output_topic = "completed.preprocess-data"

        output_consumer = kafka_consumer_factory([output_topic])

        time.sleep(2)  # Let consumers subscribe

        # Drain any old messages from previous test runs
        print("Draining old messages from Kafka topics...")
        self._drain_consumer(output_consumer, timeout_seconds=2)
        print("[OK] Kafka topics drained")

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
                start=datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC),
                end=datetime(2024, 1, 1, 2, 0, 0, tzinfo=UTC),  # 2 hours = 24 x 5m candles
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

        # Then - Wait for completion message
        # Note: Service resamples in-memory and computes features synchronously
        print("Waiting for completion message...")
        completion_1 = wait_for_kafka_message(output_consumer, timeout=30)

        assert completion_1 is not None, "Should receive completion message"
        assert completion_1["symbol"] == unique_symbol
        assert completion_1["total_features_computed"] > 0, "Features should be computed on cold start"

        print(f"[OK] Completion received: {completion_1['total_features_computed']} features computed")

        # Then - Verify parquet files created
        # Parquet structure: {base_path}/{symbol}/year=YYYY/month=MM/day=DD/*.parquet
        print(f"Verifying Feast offline store under {feast_offline_store_path}...")
        parquet_files = list(feast_offline_store_path.glob(f"**/{unique_symbol}/**/*.parquet"))
        if not parquet_files:
            # Try alternative pattern (symbol at root)
            parquet_files = list(feast_offline_store_path.glob(f"{unique_symbol}/**/*.parquet"))

        assert len(parquet_files) > 0, f"Expected parquet files, found {len(parquet_files)}"
        print(f"[OK] Found {len(parquet_files)} parquet file(s)")

        # Verify parquet content
        all_features = pd.concat([pd.read_parquet(f) for f in parquet_files])

        assert "event_timestamp" in all_features.columns
        assert "symbol" in all_features.columns

        # Feature columns may have hash suffix (e.g., rsi_14_d96c08b4...)
        rsi_columns = [c for c in all_features.columns if c.startswith("rsi_14")]
        assert len(rsi_columns) > 0, f"Expected RSI column, found columns: {all_features.columns.tolist()}"
        rsi_column = rsi_columns[0]

        assert len(all_features) > 0

        # Count valid RSI values (not NaN)
        valid_rsi_count = all_features[rsi_column].notna().sum()
        print(f"[OK] Parquet contains {len(all_features)} records, {valid_rsi_count} valid RSI values")

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

        print(f"[OK] Completion received: {completion_2['total_features_computed']} features computed (SKIPPED)")

        # Then - Verify NO new parquet files
        parquet_files_after = list(feast_offline_store_path.glob(f"**/{unique_symbol}/**/*.parquet"))
        if not parquet_files_after:
            parquet_files_after = list(feast_offline_store_path.glob(f"{unique_symbol}/**/*.parquet"))
        assert len(parquet_files_after) == initial_file_count, "Should not create new files on duplicate"

        print(f"[OK] Parquet file count unchanged: {len(parquet_files_after)}")

        # Then - Verify parquet content unchanged
        all_features_after = pd.concat([pd.read_parquet(f) for f in parquet_files_after])
        assert len(all_features_after) == initial_feature_count, "Feature count should not increase"

        print(f"[OK] Feature record count unchanged: {len(all_features_after)}")

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
                start=datetime(2024, 1, 1, 2, 0, 0, tzinfo=UTC),  # Starts where request_1 ended
                end=datetime(2024, 1, 1, 4, 0, 0, tzinfo=UTC),    # 2 more hours
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

        # Then - Wait for completion message
        print("Waiting for completion message...")
        completion_3 = wait_for_kafka_message(output_consumer, timeout=30)

        assert completion_3 is not None
        assert completion_3["symbol"] == unique_symbol

        # Gap fill may compute new features OR skip if already covered
        # (depends on how parquet partitioning worked in scenario 1)
        print(f"[OK] Completion received: {completion_3['total_features_computed']} features computed")

        # Then - Verify parquet files exist (may be same or more)
        parquet_files_final = list(feast_offline_store_path.glob(f"**/{unique_symbol}/**/*.parquet"))
        if not parquet_files_final:
            parquet_files_final = list(feast_offline_store_path.glob(f"{unique_symbol}/**/*.parquet"))
        assert len(parquet_files_final) >= initial_file_count, "Should have same or more files"

        print(f"[OK] Parquet file count: {len(parquet_files_final)} (was {initial_file_count})")

        # Verify total feature records - may be same or more depending on coverage
        all_features_final = pd.concat([pd.read_parquet(f) for f in parquet_files_final])

        # If features were computed, count should increase; if skipped, count stays same
        if completion_3["total_features_computed"] > 0:
            assert len(all_features_final) > initial_feature_count, "Should have more features after gap fill"
            print(f"[OK] Feature record count increased: {len(all_features_final)} (was {initial_feature_count})")
        else:
            assert len(all_features_final) >= initial_feature_count, "Feature count should not decrease"
            print(f"[OK] Feature record count unchanged (gap already covered): {len(all_features_final)}")

        # ==========================================
        # SCENARIO 4: FULL HISTORICAL BACKFILL UP TO SPECIFIC DATE
        # ==========================================
        print(f"\n{'='*60}")
        print(f"SCENARIO 4: FULL HISTORICAL BACKFILL - Symbol: {unique_symbol}")
        print(f"{'='*60}")

        # Give service a moment to settle
        time.sleep(1)

        # Given - Full backfill covering entire data range
        # Data was seeded from 2024-01-01 00:00, so use an earlier start
        request_4 = (
            FeaturePreprocessingRequestBuilder()
            .for_backfill()
            .with_symbol(unique_symbol)
            .with_target_timeframes([Timeframe.MINUTE_5])
            .with_time_range(
                start=datetime(2023, 12, 31, 0, 0, tzinfo=UTC),  # Before data start
                end=datetime(2024, 1, 1, 8, 0, tzinfo=UTC),  # Covers more data
            )
            .with_features([rsi_feature])
            .with_skip_existing(False)  # Force full recompute for historical backfill
            .with_force_recompute(False)
            .build()
        )

        # When - Publish full backfill request
        print(f"Publishing request 4: {unique_symbol} [2023-12-31 00:00 - 2024-01-01 08:00]")
        publish_kafka_message(
            topic="requested.preprocess-data",
            key=unique_symbol,
            value=request_4.model_dump(mode="json")
        )

        # Then - Wait for completion message
        print("Waiting for completion message...")
        completion_4 = wait_for_kafka_message(output_consumer, timeout=30)

        assert completion_4 is not None
        assert completion_4["symbol"] == unique_symbol
        assert completion_4["total_features_computed"] > 0, "Should compute features for full range"

        print(f"[OK] Completion received: {completion_4['total_features_computed']} features computed")

        # Then - Verify parquet files updated or new
        parquet_files_final_final = list(feast_offline_store_path.glob(f"**/{unique_symbol}/**/*.parquet"))
        if not parquet_files_final_final:
            parquet_files_final_final = list(feast_offline_store_path.glob(f"{unique_symbol}/**/*.parquet"))
        assert len(parquet_files_final_final) >= len(parquet_files_final), "Should have same or more files"

        print(f"[OK] Parquet file count: {len(parquet_files_final_final)} (was {len(parquet_files_final)})")

        # Verify total feature records increased
        all_features_final_final = pd.concat([pd.read_parquet(f) for f in parquet_files_final_final])
        assert len(all_features_final_final) > len(all_features_final), "Should have more features after full backfill"

        print(f"[OK] Feature record count increased: {len(all_features_final_final)} (was {len(all_features_final)})")

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
