import pandas as pd
import pandas_ta as ta
import numpy as np

from ai_trading.preprocess.feature.custom.enum.wick_handle_strategy_enum import WICK_HANDLE_STRATEGY
from ai_trading.preprocess.feature.custom.range_indicator import SupportResistanceFinder


class FeatureFactory:

    def __init__(self, source: pd.DataFrame, timeframe_postfix: str = ""):
        self.df_source = source
        self.postfix = timeframe_postfix

    def compute_macd_signals(
        self, fast_length: int, slow_length: int, signal_length: int
    ):
        """Calculate MACD using the given length parameters.

        Args:
            fast_length (int): Fast MA lookback
            slow_length (int): Slow MA lookback
            signal_length (int): MACD Signal MA lookback

        Returns:
            pd.DataFrame: A new dataframe consisting of MACD Indicator Signals timeseries data.
        """

        macd = ta.macd(
            self.df_source["Close"],
            fast=fast_length,
            slow=slow_length,
            signal=signal_length,
            fillna=np.nan,
            signal_indicators=True,
        )

        df = pd.DataFrame()
        df["Time"] = self.df_source["Time"]
        df["macd_cross_bullish" + self.postfix] = macd[
            f"MACDh_{fast_length}_{slow_length}_{signal_length}_XA_0"
        ]
        df["macd_cross_bearish" + self.postfix] = macd[
            f"MACDh_{fast_length}_{slow_length}_{signal_length}_XB_0"
        ]
        df["macd_trend" + self.postfix] = macd[
            f"MACD_{fast_length}_{slow_length}_{signal_length}_A_0"
        ]

        return df

    def compute_rsi(self, length: int) -> pd.DataFrame:
        """Calculate RSI using the given length parameter.

        Args:
            length (int): RSI lookback

        Returns:
            pd.DataFrame: A new dataframe consisting of RSI Indicator timeseries data.
        """

        df = pd.DataFrame()
        df["Time"] = self.df_source["Time"]
        df[f"rsi_{length}{self.postfix}"] = ta.rsi(
            self.df_source["Close"], length=length
        )
        return df

    # Function to calculate multiple Rate of change Series and append to origin DataFrame
    def compute_roc(self, length: int) -> pd.DataFrame:
        """Calculate Rate of change using given length parameter.

        Args:
            length (int): ROC lookback

        Returns:
            pd.DataFrame: A new dataframe consisting of ROC Indicator timeseries data.
        """
        df = pd.DataFrame()
        df["Time"] = self.df_source["Time"]
        df[f"roc_{length}{self.postfix}"] = ta.roc(
            self.df_source["Close"], length=length
        )

        return df
    
    def compute_ranges(self, lookback: int, wick_handle_strategy: WICK_HANDLE_STRATEGY) -> pd.DataFrame:
        """Compute S/R price ranges

        Args:
            lookback (int): Iteration limit for pivot points cache
            wick_handle_strategy (WICK_HANDLE_STRATEGY): Strategy to calculate pivot point zone bottom or top

        Returns:
            pd.DataFrame: A new dataframe consisting of Range Indicator timeseries data.
        """
        df = pd.DataFrame()
        df["Time"] = self.df_source["Time"]
        finder = SupportResistanceFinder(self.df_source, lookback, wick_handle_strategy)
        ranges = finder.find_support_resistance_zones()
        df[f"resistance_range{lookback}{self.postfix}"] = ranges["resistance_range"]
        df[f"support_range{lookback}{self.postfix}"] = ranges["support_range"]

        return df

    # function to calculate market dynamic indicators
    def add_market_dynamic_indicators(df_source, df_target, postfix=""):
        df_target["rvi_7" + postfix] = ta.rvi(
            df_source["Close"], df_source["High"], df_source["Low"], length=7
        )
        df_target["rvi_14" + postfix] = ta.rvi(
            df_source["Close"], df_source["High"], df_source["Low"], length=14
        )
        return df_target

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
