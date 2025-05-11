import pandas_ta as ta
from pandas import DataFrame

from ai_trading.config.base_parameter_set_config import BaseParameterSetConfig
from ai_trading.config.feature_config_collection import RviConfig
from ai_trading.preprocess.feature.collection.base_feature import BaseFeature


class RviFeature(BaseFeature):

    def __init__(
        self, source: DataFrame, config: BaseParameterSetConfig, postfix: str = ""
    ) -> None:
        super().__init__(source, config, postfix)
        self.config: RviConfig = self.config

    def compute(self) -> DataFrame:
        # Get source DataFrame with ensured DatetimeIndex using the base class method
        source_df = self._prepare_source_df()

        # Create result DataFrame with the same index
        df = DataFrame(index=source_df.index)
        df[f"rvi_{self.config.length}{self.postfix}"] = ta.rvi(
            source_df["Close"],
            source_df["High"],
            source_df["Low"],
            length=self.config.length,
        )

        return df

    def get_sub_features_names(self) -> list[str]:
        return [f"rvi_{self.config.length}{self.postfix}"]
