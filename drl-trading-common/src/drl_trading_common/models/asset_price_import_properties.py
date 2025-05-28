"""Asset price import properties model."""

from dataclasses import dataclass


@dataclass
class AssetPriceImportProperties:
    """Properties for importing asset price data."""

    # Indicates timeframe of the data e.g. H1
    timeframe: str

    # During merge process, all computed datasets are merged into base dataset
    base_dataset: bool

    # Simple ohlcv data + timestamp of the asset
    file_path: str
