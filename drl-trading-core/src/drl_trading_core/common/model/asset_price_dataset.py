from dataclasses import dataclass

from pandas import DataFrame


@dataclass
class AssetPriceDataSet:
    # Indicates timeframe of the data e.g. H1
    timeframe: str

    # During merge process, all computed datasets are merged into base dataset
    base_dataset: bool

    # Simple ohlcv data + timestamp of the asset
    asset_price_dataset: DataFrame
