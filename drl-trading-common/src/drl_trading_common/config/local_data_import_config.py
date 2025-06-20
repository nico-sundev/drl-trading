from typing import List, Optional

from ..model import AssetPriceImportProperties
from ..base.base_schema import BaseSchema


class SymbolConfig(BaseSchema):
    symbol: str
    datasets: List[AssetPriceImportProperties]


class LocalDataImportConfig(BaseSchema):
    symbols: List[SymbolConfig]
    limit: Optional[int] = None
    strategy: str = "csv"  # Default to csv strategy
