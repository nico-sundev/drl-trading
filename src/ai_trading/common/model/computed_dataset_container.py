from dataclasses import dataclass

from pandas import DataFrame

from ai_trading.common.model.asset_price_dataset import AssetPriceDataSet


@dataclass
class ComputedDataSetContainer:
    source_dataset: AssetPriceDataSet
    computed_dataframe: DataFrame
