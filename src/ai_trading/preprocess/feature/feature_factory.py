import pandas as pd
import pandas_ta as ta
import numpy as np

from ai_trading.preprocess.feature.custom.enum.wick_handle_strategy_enum import WICK_HANDLE_STRATEGY
from ai_trading.preprocess.feature.custom.range_indicator import SupportResistanceFinder


class FeatureFactory:

    def __init__(self, source: pd.DataFrame, timeframe_postfix: str = ""):
        self.df_source = source
        self.postfix = timeframe_postfix

    # Function to calculate near MA support/resistance zones based on ATR
    def add_extreme_zones_from_ma(df_source, df_target, atr_multiplier=1.5, postfix=""):
        # Create a temporary DataFrame to store MACD values
        temp_df = pd.DataFrame()
        temp_df["atr"] = ta.atr(
            df_source["High"], df_source["Low"], df_source["Close"], length=14
        )
        temp_df["ma50"] = ta.sma(df_source["Close"], length=50)
        temp_df["ma100"] = ta.sma(df_source["Close"], length=100)
        temp_df["ma200"] = ta.sma(df_source["Close"], length=200)

        for ma_length in [50, 100, 200]:
            ma_col = f"ma{ma_length}"
            extreme_zone = f"near_ma{ma_length}_zone{postfix}"
            low_inside_zone = (
                df_source["Low"] >= temp_df[ma_col] - temp_df["atr"] * atr_multiplier
            ) & (df_source["Low"] <= temp_df[ma_col] + temp_df["atr"] * atr_multiplier)
            high_inside_zone = (
                df_source["High"] >= temp_df[ma_col] - temp_df["atr"] * atr_multiplier
            ) & (df_source["High"] <= temp_df[ma_col] + temp_df["atr"] * atr_multiplier)
            df_target[extreme_zone] = np.where(low_inside_zone | high_inside_zone, 1, 0)

        return df_target

    # Function to calculate Bollinger Bands, ATR Bands, Ichimoku Cloud, and create support/resistance boxes
    def add_extreme_zones_from_bands(df_source, df_target, postfix=""):
        temp_df = pd.DataFrame()
        temp_df["atr"] = ta.atr(
            df_source["High"], df_source["Low"], df_source["Close"], length=14
        )

        # Bollinger Bands
        bbands_df = ta.bbands(df_source["Close"], length=20, std=2)
        temp_df["bb_upper"] = bbands_df["BBU_20_2.0"]
        temp_df["bb_lower"] = bbands_df["BBL_20_2.0"]

        # ATR Bands
        temp_df["atr_upper_band"] = df_source["Close"] + temp_df["atr"]
        temp_df["atr_lower_band"] = df_source["Close"] - temp_df["atr"]

        # Ichimoku Cloud
        donchian = ta.donchian(df_source["High"], df_source["Low"])

        temp_df["donchian_upper"] = donchian["DCU_20_20"]
        temp_df["donchian_lower"] = donchian["DCL_20_20"]

        # Create Resistance Box (from upper bounds)
        resistance_upper_bound = temp_df[
            ["bb_upper", "atr_upper_band", "donchian_upper"]
        ].max(axis=1)
        resistance_lower_bound = temp_df[
            ["bb_upper", "atr_upper_band", "donchian_upper"]
        ].min(axis=1)
        low_inside_resistance_zone = (df_source["Low"] >= resistance_lower_bound) & (
            df_source["Low"] <= resistance_upper_bound
        )
        high_inside_resistance_zone = (df_source["High"] >= resistance_lower_bound) & (
            df_source["High"] <= resistance_upper_bound
        )
        df_target["bands_resistance_touched" + postfix] = np.where(
            low_inside_resistance_zone | high_inside_resistance_zone, 1, 0
        )

        # Create Support Box (from lower bounds)
        support_upper_bound = temp_df[
            ["bb_lower", "atr_lower_band", "donchian_lower"]
        ].max(axis=1)
        support_lower_bound = temp_df[
            ["bb_lower", "atr_lower_band", "donchian_lower"]
        ].min(axis=1)
        low_inside_support_zone = (df_source["Low"] >= support_lower_bound) & (
            df_source["Low"] <= support_upper_bound
        )
        high_inside_support_zone = (df_source["High"] >= support_lower_bound) & (
            df_source["High"] <= support_upper_bound
        )
        df_target["bands_support_touched" + postfix] = np.where(
            low_inside_support_zone | high_inside_support_zone, 1, 0
        )

        return df_target
