"""Asset price import properties model."""


from drl_trading_common.base.base_schema import BaseSchema
from drl_trading_common.model.timeframe import Timeframe


class AssetPriceImportProperties(BaseSchema):
    """Properties for importing asset price data."""

    # Indicates timeframe of the data e.g. H1
    timeframe: Timeframe

    # During merge process, all computed datasets are merged into base dataset
    base_dataset: bool

    # Simple ohlcv data + timestamp of the asset
    file_path: str
