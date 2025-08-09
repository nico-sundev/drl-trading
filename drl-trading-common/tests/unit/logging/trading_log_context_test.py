"""
Unit tests for TradingLogContext.

Tests the thread-local context management, trading context integration,
and context field management functionality.
"""

from typing import Any

from drl_trading_common.logging.trading_log_context import TradingLogContext


class TestTradingLogContextBasics:
    """Test basic TradingLogContext functionality."""

    def test_context_initialization(self, clean_log_context: Any) -> None:
        """Test TradingLogContext starts with clean state."""
        # Given
        # clean_log_context fixture ensures clean state

        # When
        context = TradingLogContext.get_available_context()

        # Then
        assert context == {}
        assert TradingLogContext.get_correlation_id() is None
        assert TradingLogContext.get_symbol() is None

    def test_correlation_id_management(self, clean_log_context: Any) -> None:
        """Test correlation ID setting and retrieval."""
        # Given
        correlation_id = "test-correlation-123"

        # When
        TradingLogContext.set_correlation_id(correlation_id)
        retrieved_id = TradingLogContext.get_correlation_id()

        # Then
        assert retrieved_id == correlation_id

    def test_context_clearing(self, clean_log_context: Any) -> None:
        """Test context clearing functionality."""
        # Given
        TradingLogContext.set_correlation_id("test-123")
        TradingLogContext.set_symbol("GBPUSD")
        TradingLogContext.set_strategy_id("test_strategy")

        # When
        TradingLogContext.clear()
        context = TradingLogContext.get_available_context()

        # Then
        assert context == {}
        assert TradingLogContext.get_correlation_id() is None
        assert TradingLogContext.get_symbol() is None
        assert TradingLogContext.get_strategy_id() is None


class TestTradingLogContextGeneration:
    """Test TradingLogContext ID generation functionality."""

    def test_correlation_id_generation(self, clean_log_context: Any) -> None:
        """Test automatic correlation ID generation."""
        # Given
        # Clean context

        # When
        correlation_id = TradingLogContext.generate_new_correlation_id()

        # Then
        assert correlation_id is not None
        assert correlation_id.startswith("trade-")
        assert len(correlation_id) > 10  # UUID should make it longer

    def test_event_id_generation(self, clean_log_context: Any) -> None:
        """Test event ID generation with service prefix."""
        # Given
        service_name = "drl-trading-ingest"

        # When
        event_id = TradingLogContext.generate_new_event_id(service_name)

        # Then
        assert event_id is not None
        assert event_id.startswith("drl-trading-ingest-")
        assert len(event_id) > 20  # Service name + UUID should make it longer

        # Context should be updated
        assert TradingLogContext.get_event_id() == event_id


class TestTradingLogContextIntegration:
    """Test TradingLogContext integration with TradingContext."""

    def test_from_trading_context(
        self, clean_log_context: Any, sample_trading_context: Any
    ) -> None:
        """Test context creation from TradingContext object."""
        # Given
        trading_context = sample_trading_context

        # When
        TradingLogContext.from_trading_context(trading_context)

        # Then
        assert TradingLogContext.get_correlation_id() == trading_context.correlation_id
        assert TradingLogContext.get_symbol() == trading_context.symbol
        assert TradingLogContext.get_strategy_id() == trading_context.strategy_id
        assert TradingLogContext.get_model_version() == trading_context.model_version

    def test_get_context_complete(self, clean_log_context: Any) -> None:
        """Test getting complete context dictionary."""
        # Given
        TradingLogContext.set_correlation_id("test-correlation-456")
        TradingLogContext.set_symbol("USDJPY")
        TradingLogContext.set_strategy_id("arbitrage_v2")
        TradingLogContext.set_model_version("model_v1.5")
        TradingLogContext.generate_new_event_id("test-service")

        # When
        context = TradingLogContext.get_available_context()

        # Then
        assert context["correlation_id"] == "test-correlation-456"
        assert context["symbol"] == "USDJPY"
        assert context["strategy_id"] == "arbitrage_v2"
        assert context["model_version"] == "model_v1.5"
        assert context["event_id"].startswith("test-service-")
