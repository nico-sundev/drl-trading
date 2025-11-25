from dataclasses import dataclass
from drl_trading_common.adapter.model.timeframe import Timeframe


@dataclass(frozen=True)
class DatasetIdentifier:
    """Identifier for a dataset representing financial market data."""

    symbol: str
    timeframe: Timeframe
