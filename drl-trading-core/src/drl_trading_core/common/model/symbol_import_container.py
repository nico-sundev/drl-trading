from dataclasses import dataclass
from typing import List

from drl_trading_core.common.model.asset_price_dataset import AssetPriceDataSet


@dataclass
class SymbolImportContainer:
    """Container for symbol data import results."""

    symbol: str
    datasets: List[AssetPriceDataSet]
