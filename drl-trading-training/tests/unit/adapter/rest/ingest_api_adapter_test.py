"""
Unit tests for the IngestApiAdapter.

Tests the REST API adapter for communicating with the ingest service.
"""

import pytest
from unittest.mock import MagicMock

from drl_trading_training.adapter.rest.ingest_api_adapter import IngestApiAdapter


@pytest.fixture
def mock_api_client():
    """Fixture for mocked API client."""
    return MagicMock()


@pytest.fixture
def mock_default_api():
    """Fixture for mocked DefaultApi."""
    return MagicMock()


@pytest.fixture
def adapter(mock_api_client, mock_default_api):
    """Fixture for IngestApiAdapter with mocked dependencies."""
    adapter = IngestApiAdapter(api_client=mock_api_client)
    adapter.api = mock_default_api
    return adapter


class TestIngestApiAdapter:
    """Test cases for IngestApiAdapter."""

    def test_submit_preprocessing_request_success(self, adapter, mock_default_api):
        """Test successful submission of preprocessing request."""
        # Given
        from drl_trading_core.core.dto.feature_preprocessing_request import FeaturePreprocessingRequest
        from drl_trading_common.core.model.timeframe import Timeframe
        from drl_trading_core.core.model.feature_definition import FeatureDefinition
        from drl_trading_core.core.model.feature_config_version_info import FeatureConfigVersionInfo
        from datetime import datetime, timezone

        request = FeaturePreprocessingRequest(
            symbol="AAPL",
            base_timeframe=Timeframe.HOUR_1,
            target_timeframes=[Timeframe.HOUR_4, Timeframe.DAY_1],
            feature_config_version_info=FeatureConfigVersionInfo(
                semver="1.0.0",
                hash="test_hash",
                created_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
                feature_definitions=[FeatureDefinition(name="sma_20", enabled=True, derivatives=[20])]
            ),
            start_time=datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            end_time=datetime(2024, 1, 2, 0, 0, 0, tzinfo=timezone.utc),
            request_id="test-request-123"
        )

        # Mock the API response
        mock_response = MagicMock()
        mock_response.request_id = "test-request-123"
        mock_response.status = "accepted"
        mock_default_api.submit_preprocessing_request.return_value = mock_response

        # When
        result = adapter.submit_preprocessing_request(request)

        # Then
        assert result["request_id"] == "test-request-123"
        assert result["status"] == "accepted"
        mock_default_api.submit_preprocessing_request.assert_called_once()
