from typing import List, Optional

from ai_trading.config.base_schema import BaseSchema
from ai_trading.model.asset_price_import_properties import AssetPriceImportProperties


class SymbolConfig(BaseSchema):
    symbol: str
    datasets: List[AssetPriceImportProperties]


class LocalDataImportConfig(BaseSchema):
    symbols: List[SymbolConfig]
    limit: Optional[int] = None
