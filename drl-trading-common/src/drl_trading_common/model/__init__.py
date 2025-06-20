"""Common data models for trading system."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, Optional
from .timeframe import Timeframe
from .dataset_identifier import DatasetIdentifier

# Import external model classes
from .asset_price_import_properties import AssetPriceImportProperties


class OrderAction(Enum):
    """Trading actions."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class OrderStatus(Enum):
    """Order execution status."""

    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class MarketData:
    """Market data structure."""

    symbol: str
    timeframe: str
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for messaging."""
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "timestamp": self.timestamp.isoformat(),
            "open": float(self.open),
            "high": float(self.high),
            "low": float(self.low),
            "close": float(self.close),
            "volume": self.volume,
        }


@dataclass
class TradingFeatures:
    """Feature vector for ML models."""

    symbol: str
    timeframe: str
    timestamp: datetime
    features: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for messaging."""
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "timestamp": self.timestamp.isoformat(),
            "features": self.features,
        }


@dataclass
class TradingSignal:
    """Trading signal from ML model."""

    signal_id: str
    symbol: str
    action: OrderAction
    confidence: float
    timestamp: datetime
    features_used: Optional[Dict[str, float]] = None
    model_version: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for messaging."""
        return {
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "action": self.action.value,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "features_used": self.features_used,
            "model_version": self.model_version,
        }


@dataclass
class ExecutionResult:
    """Trade execution result."""

    order_id: str
    signal_id: str
    symbol: str
    action: OrderAction
    status: OrderStatus
    requested_quantity: Optional[Decimal] = None
    executed_quantity: Optional[Decimal] = None
    executed_price: Optional[Decimal] = None
    timestamp: Optional[datetime] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for messaging."""
        return {
            "order_id": self.order_id,
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "action": self.action.value,
            "status": self.status.value,
            "requested_quantity": (
                float(self.requested_quantity) if self.requested_quantity else None
            ),
            "executed_quantity": (
                float(self.executed_quantity) if self.executed_quantity else None
            ),
            "executed_price": (
                float(self.executed_price) if self.executed_price else None
            ),
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "error_message": self.error_message,
        }


@dataclass
class InferenceRequest:
    """Request for ML model inference."""

    request_id: str
    symbol: str
    features: Dict[str, float]
    model_name: Optional[str] = None
    timestamp: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for messaging."""
        return {
            "request_id": self.request_id,
            "symbol": self.symbol,
            "features": self.features,
            "model_name": self.model_name,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


@dataclass
class InferenceResponse:
    """Response from ML model inference."""

    request_id: str
    signal: TradingSignal
    processing_time_ms: Optional[float] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for messaging."""
        return {
            "request_id": self.request_id,
            "signal": self.signal.to_dict() if self.signal else None,
            "processing_time_ms": self.processing_time_ms,
            "error": self.error,
        }


# Import external model classes

# Export all models
__all__ = [
    "OrderAction",
    "OrderStatus",
    "MarketData",
    "TradingFeatures",
    "TradingSignal",
    "ExecutionResult",
    "InferenceRequest",
    "InferenceResponse",
    "AssetPriceImportProperties",
    "Timeframe",
    "DatasetIdentifier",
]
