from typing import List, Optional

from ai_trading.config.base_schema import BaseSchema
from ai_trading.model.asset_price_import_properties import AssetPriceImportProperties


class LocalDataImportConfig(BaseSchema):
    datasets: List[AssetPriceImportProperties]
    limit: Optional[int] = None
