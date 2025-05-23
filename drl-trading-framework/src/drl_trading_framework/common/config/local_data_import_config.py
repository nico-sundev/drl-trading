from typing import List, Optional

from drl_trading_framework.common.config.base_schema import BaseSchema
from drl_trading_framework.common.model.asset_price_import_properties import (
    AssetPriceImportProperties,
)


class SymbolConfig(BaseSchema):
    symbol: str
    datasets: List[AssetPriceImportProperties]


class LocalDataImportConfig(BaseSchema):
    symbols: List[SymbolConfig]
    limit: Optional[int] = None
    strategy: str = "csv"  # Default to csv strategy
