"""
Integration tests for PreprocessingOrchestrator with real services.

This test suite validates the complete preprocessing orchestration workflow
using real implementations:
- PostgreSQL/TimescaleDB via testcontainers for market data
- Local Feast feature store for feature storage
- Real resampling, computing, and coverage services
- Spy message publisher for verification

Test Philosophy:
- Each test gets a fresh database container and Feast repository
- Tests use realistic market data and feature definitions
- Validates both processing logic and data storage
- Focuses on end-to-end integration scenarios
"""

import pytest
from datetime import datetime
from injector import Injector

from drl_trading_adapter.adapter.database.entity.market_data_entity import MarketDataEntity
from drl_trading_adapter.adapter.feature_store.provider import FeatureStoreWrapper
from drl_trading_common.model.feature_preprocessing_request import FeaturePreprocessingRequest
from drl_trading_common.model.timeframe import Timeframe
from drl_trading_preprocess.adapter.messaging.publisher.stub_preprocess_message_publisher import (
    StubPreprocessingMessagePublisher,
)
from drl_trading_preprocess.core.orchestrator.preprocessing_orchestrator import (
    PreprocessingOrchestrator,
)


class TestPreprocessingOrchestratorIntegration:
    """Integration test suite for PreprocessingOrchestrator."""

    @pytest.fixture
    def orchestrator(self, integration_container: Injector) -> PreprocessingOrchestrator:
        """Get orchestrator instance from DI container.

        Args:
            integration_container: Configured DI container with all dependencies

        Returns:
            PreprocessingOrchestrator: Fully wired orchestrator instance
        """
        return integration_container.get(PreprocessingOrchestrator)

    @pytest.fixture
    def feature_store(self, integration_container: Injector) -> FeatureStoreWrapper:
        """Get feature store wrapper for verification.

        Args:
            integration_container: Configured DI container

        Returns:
            FeatureStoreWrapper: Feature store for fetching stored features
        """
        return integration_container.get(FeatureStoreWrapper)

    @pytest.fixture
    def spy_publisher(self, integration_container: Injector) -> StubPreprocessingMessagePublisher:
        """Get spy message publisher for verification.

        Args:
            integration_container: Configured DI container

        Returns:
            StubPreprocessingMessagePublisher: Publisher with inspection capabilities
        """
        from drl_trading_preprocess.core.port.preprocessing_message_publisher_port import (
            PreprocessingMessagePublisherPort,
        )
        # Retrieve via interface, cast to concrete type for test inspection
        return integration_container.get(PreprocessingMessagePublisherPort)  # type: ignore[return-value]

    def test_happy_path_fresh_environment(
        self,
        orchestrator: PreprocessingOrchestrator,
        feature_store: FeatureStoreWrapper,
        spy_publisher: StubPreprocessingMessagePublisher,
        populated_market_data: list[MarketDataEntity],
    ) -> None:
        """Test complete preprocessing workflow with fresh database and feature store.

        This test validates the entire orchestration pipeline:
        1. Request validation
        2. Coverage analysis (empty Feast)
        3. Market data resampling (1h → 4h)
        4. Feature computation (RSI + ClosePrice)
        5. Feature store persistence
        6. Success notification publishing

        Args:
            orchestrator: Orchestrator under test
            feature_store: Feature store for verification
            spy_publisher: Spy publisher for message verification
            populated_market_data: Pre-populated market data in TimescaleDB
        """
        # Given: Fresh environment with market data but no features yet
        assert len(populated_market_data) == 50, "Should have 50 hourly bars"

        # Create feature definitions as dictionaries (as expected by FeatureConfigVersionInfo)
        feature_definitions_dict = [
            {
                "name": "rsi",
                "enabled": True,
                "derivatives": [],
                "parameter_sets": [{"type": "rsi", "period": 14}],
            },
            {
                "name": "close_price",
                "enabled": True,
                "derivatives": [],
                "parameter_sets": [],
            },
        ]

        # Create feature config version info
        from drl_trading_common.model.feature_config_version_info import FeatureConfigVersionInfo

        feature_config_version = FeatureConfigVersionInfo(
            semver="1.0.0-test",
            hash="test-hash-001",
            created_at=datetime.now(),
            feature_definitions=feature_definitions_dict,
            description="Integration test feature configuration",
        )

        # Create preprocessing request
        request = FeaturePreprocessingRequest(
            request_id="test-request-001",
            symbol="EURUSD",
            base_timeframe=Timeframe.HOUR_1,
            target_timeframes=[Timeframe.HOUR_4],
            feature_config_version_info=feature_config_version,
            start_time=datetime(2024, 1, 1, 9, 0, 0),
            end_time=datetime(2024, 1, 3, 9, 0, 0),  # ~48 hours of data
            skip_existing_features=True,
            processing_context="training",
        )

        # When: Process the feature computation request
        orchestrator.process_feature_computation_request(request)

        # Then: Verify success notification was published
        assert len(spy_publisher._published_messages) == 1, "Should publish one success message"

        success_message = spy_publisher._published_messages[0]
        assert success_message["type"] == "preprocessing_completed"
        assert success_message["request_id"] == "test-request-001"
        assert success_message["symbol"] == "EURUSD"
        assert success_message["processing_context"] == "training"
        assert success_message["total_features_computed"] > 0, "Should have computed features"
        assert Timeframe.HOUR_4.value in success_message["timeframes_processed"]

        # Then: Verify no error messages were published
        assert len(spy_publisher._error_messages) == 0, "Should not publish any errors"

        # Then: Verify features are stored in Feast
        # Note: Full Feast verification would require fetching features
        # For now, we verify the workflow completed without errors
        # TODO: Add Feast feature retrieval verification in follow-up
