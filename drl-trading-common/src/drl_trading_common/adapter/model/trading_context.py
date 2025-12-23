"""
Trading context data structures for cross-service context propagation.

These DTOs enable consistent context tracking across the DRL trading microservice
ecosystem, supporting distributed tracing and structured logging.
"""

from datetime import datetime, UTC
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
import uuid


class TradingContext(BaseModel):
    """
    Standardized context that flows through all trading services.

    This context is progressively enriched as it flows through the trading pipeline:
    - Ingest: Creates initial context with market data identifiers
    - Preprocess: Adds strategy and timeframe information
    - Inference: Adds model metadata and prediction results
    - Execution: Adds trade execution details
    """

    # Core identifiers (always present)
    correlation_id: str = Field(description="Unique ID tracking business operation across services")
    event_id: str = Field(description="Unique ID for individual event/message")
    symbol: str = Field(description="Financial instrument symbol (e.g., BTCUSDT, EURUSD)")
    timestamp: datetime = Field(description="Event timestamp")

    # Preprocessing context (available after preprocessing)
    strategy_id: Optional[str] = Field(default=None, description="Trading strategy identifier")
    timeframe: Optional[str] = Field(default=None, description="Market data timeframe (e.g., 1m, 5m, 1h)")

    # Model context (available after inference)
    model_version: Optional[str] = Field(default=None, description="ML model version used for prediction")
    prediction_confidence: Optional[float] = Field(default=None, description="Model prediction confidence score (0.0-1.0)")

    # Execution context (available after execution)
    trade_id: Optional[str] = Field(default=None, description="Unique trade execution identifier")
    execution_price: Optional[float] = Field(default=None, description="Actual execution price")
    execution_quantity: Optional[float] = Field(default=None, description="Actual execution quantity")

    # Additional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context-specific metadata")

    @classmethod
    def create_initial_context(cls, symbol: str, **kwargs: Any) -> "TradingContext":
        """
        Create initial trading context for market data ingestion.

        Args:
            symbol: Financial instrument symbol
            **kwargs: Additional metadata

        Returns:
            TradingContext with initial fields populated
        """
        return cls(
            correlation_id=str(uuid.uuid4()),
            event_id=f"ingest-{uuid.uuid4()}",
            symbol=symbol,
            timestamp=datetime.now(UTC).replace(tzinfo=None),
            metadata=kwargs
        )

    def enrich_with_strategy(self, strategy_id: str, timeframe: str) -> "TradingContext":
        """
        Enrich context with strategy information (typically in preprocessing service).

        Args:
            strategy_id: Trading strategy identifier
            timeframe: Market data timeframe

        Returns:
            Updated context with strategy information
        """
        self.strategy_id = strategy_id
        self.timeframe = timeframe
        return self

    def enrich_with_model(self, model_version: str, prediction_confidence: float) -> "TradingContext":
        """
        Enrich context with model information (typically in inference service).

        Args:
            model_version: ML model version
            prediction_confidence: Prediction confidence score

        Returns:
            Updated context with model information
        """
        self.model_version = model_version
        self.prediction_confidence = prediction_confidence
        return self

    def enrich_with_execution(self, trade_id: str, execution_price: float,
                            execution_quantity: float) -> "TradingContext":
        """
        Enrich context with execution information (typically in execution service).

        Args:
            trade_id: Trade execution identifier
            execution_price: Actual execution price
            execution_quantity: Actual execution quantity

        Returns:
            Updated context with execution information
        """
        self.trade_id = trade_id
        self.execution_price = execution_price
        self.execution_quantity = execution_quantity
        return self

    def to_log_context(self) -> Dict[str, Any]:
        """
        Convert to dictionary suitable for logging context.
        Only includes non-None values to avoid cluttering logs.

        Returns:
            Dictionary with available context fields
        """
        context = {
            'correlation_id': self.correlation_id,
            'event_id': self.event_id,
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat()
        }

        # Add optional fields only if present
        optional_fields = [
            'strategy_id', 'timeframe', 'model_version',
            'prediction_confidence', 'trade_id', 'execution_price', 'execution_quantity'
        ]

        for field in optional_fields:
            value = getattr(self, field)
            if value is not None:
                context[field] = value

        # Add metadata if present
        if self.metadata:
            for key, value in self.metadata.items():
                context[f'meta_{key}'] = value

        return context
