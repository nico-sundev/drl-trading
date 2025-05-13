from dataclasses import dataclass
from typing import List

from ai_trading.common.model.asset_price_dataset import AssetPriceDataSet


@dataclass
class SymbolImportContainer:
    """Container for symbol data import results."""

    symbol: str
    datasets: List[AssetPriceDataSet]
