import numpy as np
import pandas as pd
import pandas_ta as ta


class FeatureFactory:
    def __init__(self, source: pd.DataFrame, timeframe_postfix: str = "") -> None:
        self.df_source = source
        self.postfix = timeframe_postfix

    def add_extreme_zones_from_ma(
        self, df_target: pd.DataFrame, atr_multiplier: float = 1.5
    ) -> pd.DataFrame:
        """Calculate near MA support/resistance zones based on ATR"""
        temp_df = pd.DataFrame()
        temp_df["atr"] = ta.atr(
            self.df_source["High"],
            self.df_source["Low"],
            self.df_source["Close"],
            length=14,
        )
        temp_df["ma50"] = ta.sma(self.df_source["Close"], length=50)
        temp_df["ma100"] = ta.sma(self.df_source["Close"], length=100)
        temp_df["ma200"] = ta.sma(self.df_source["Close"], length=200)

        for ma_length in [50, 100, 200]:
            ma_col = f"ma{ma_length}"
            extreme_zone = f"near_ma{ma_length}_zone{self.postfix}"
            low_inside_zone = (
                self.df_source["Low"]
                >= temp_df[ma_col] - temp_df["atr"] * atr_multiplier
            ) & (
                self.df_source["Low"]
                <= temp_df[ma_col] + temp_df["atr"] * atr_multiplier
            )
            high_inside_zone = (
                self.df_source["High"]
                >= temp_df[ma_col] - temp_df["atr"] * atr_multiplier
            ) & (
                self.df_source["High"]
                <= temp_df[ma_col] + temp_df["atr"] * atr_multiplier
            )
            df_target[extreme_zone] = np.where(low_inside_zone | high_inside_zone, 1, 0)

        return df_target

    def add_extreme_zones_from_bands(self, df_target: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands, ATR Bands, Ichimoku Cloud, and create support/resistance boxes"""
        temp_df = pd.DataFrame()
        temp_df["atr"] = ta.atr(
            self.df_source["High"],
            self.df_source["Low"],
            self.df_source["Close"],
            length=14,
        )

        # Bollinger Bands
        bbands_df = ta.bbands(self.df_source["Close"], length=20, std=2)
        temp_df["bb_upper"] = bbands_df["BBU_20_2.0"]
        temp_df["bb_lower"] = bbands_df["BBL_20_2.0"]

        # ATR Bands
        temp_df["atr_upper_band"] = self.df_source["Close"] + temp_df["atr"]
        temp_df["atr_lower_band"] = self.df_source["Close"] - temp_df["atr"]

        # Ichimoku Cloud
        donchian = ta.donchian(self.df_source["High"], self.df_source["Low"])
        temp_df["donchian_upper"] = donchian["DCU_20_20"]
        temp_df["donchian_lower"] = donchian["DCL_20_20"]

        # Create Resistance Box (from upper bounds)
        resistance_upper_bound = temp_df[
            ["bb_upper", "atr_upper_band", "donchian_upper"]
        ].max(axis=1)
        resistance_lower_bound = temp_df[
            ["bb_upper", "atr_upper_band", "donchian_upper"]
        ].min(axis=1)
        low_inside_resistance_zone = (
            self.df_source["Low"] >= resistance_lower_bound
        ) & (self.df_source["Low"] <= resistance_upper_bound)
        high_inside_resistance_zone = (
            self.df_source["High"] >= resistance_lower_bound
        ) & (self.df_source["High"] <= resistance_upper_bound)
        df_target["bands_resistance_touched" + self.postfix] = np.where(
            low_inside_resistance_zone | high_inside_resistance_zone, 1, 0
        )

        # Create Support Box (from lower bounds)
        support_upper_bound = temp_df[
            ["bb_lower", "atr_lower_band", "donchian_lower"]
        ].max(axis=1)
        support_lower_bound = temp_df[
            ["bb_lower", "atr_lower_band", "donchian_lower"]
        ].min(axis=1)
        low_inside_support_zone = (self.df_source["Low"] >= support_lower_bound) & (
            self.df_source["Low"] <= support_upper_bound
        )
        high_inside_support_zone = (self.df_source["High"] >= support_lower_bound) & (
            self.df_source["High"] <= support_upper_bound
        )
        df_target["bands_support_touched" + self.postfix] = np.where(
            low_inside_support_zone | high_inside_support_zone, 1, 0
        )

        return df_target
