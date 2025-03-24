import pandas as pd
import numpy as np

from ai_trading.preprocess.feature.feature_config import FeatureConfig

class SupportResistanceFinder:
    """
    Finds support and resistance zones based on pivot points.
    Uses a lookback period and caches previous zones for efficiency.
    """

    def __init__(self, config: FeatureConfig):        
        self.config = config
        self.prev_support = None  # Cached support zone
        self.prev_resistance = None  # Cached resistance zone

    def validate_dataframe(self, df: pd.DataFrame):
        """Ensures required columns exist and data is valid."""
        required_cols = {"Close", "High", "Low"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        if df.empty:
            raise ValueError("DataFrame is empty.")

        if df.isnull().any().any():
            raise ValueError("DataFrame contains NaN values.")

    def find_pivot_points(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identifies pivot highs and lows (local maxima/minima) in the dataset.
        """
        df = df.copy()  # Avoid modifying original DataFrame
        df["pivot_high"] = df["Close"] < df["Close"].shift(1)
        df["pivot_low"] = df["Close"] > df["Close"].shift(1)
        return df

    def find_next_support_resistance(self, df: pd.DataFrame, last_close: float):
        """
        Iterates backward to find the nearest valid support and resistance within the lookback window.
        """
        support_zone, resistance_zone = {"low": np.nan, "high": np.nan}, {"low": np.nan, "high": np.nan}
        found_support, found_resistance = False, False

        for i in range(len(df) - 2, max(-1, len(df) - 2 - self.config.range.lookback), -1):
            row, prev_row = df.iloc[i], df.iloc[i - 1] if i > 0 else None

            if not found_resistance and row["pivot_high"] and prev_row is not None and prev_row["Close"] < row["Close"] and row["Close"] > last_close:
                found_resistance = True
                resistance_zone["low"] = prev_row["Close"]
                resistance_zone["high"] = self.calculate_wick_threshold(row["Close"], max(row["High"], prev_row["High"]))

            if not found_support and row["pivot_low"] and prev_row is not None and prev_row["Close"] > row["Close"] and row["Close"] < last_close:
                found_support = True
                support_zone["high"] = prev_row["Close"]
                support_zone["low"] = self.calculate_wick_threshold(row["Close"], min(row["Low"], prev_row["Low"]))

            if found_support and found_resistance:
                break

        return support_zone, resistance_zone

    @staticmethod
    def calculate_wick_threshold(close: float, extreme: float, ratio: float = 2 / 3) -> float:
        """
        Calculates a threshold within the wick range to define support/resistance zones.
        The default ratio is 2/3, meaning the zone extends into the wick by two-thirds.
        """
        return close + (extreme - close) * ratio

    def find_support_resistance_zones(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes support and resistance zones for the last candle in the dataset.
        Caches zones for efficiency and reuses them if they remain valid.
        """
        self.validate_dataframe(df)  # Validate input data
        df = self.find_pivot_points(df)

        last_close = df.iloc[-1]["Close"]

        # If previously found zones are still valid, reuse them
        support_valid = self.prev_support and last_close > self.prev_support["low"]
        resistance_valid = self.prev_resistance and last_close < self.prev_resistance["high"]

        if support_valid and resistance_valid:
            df.loc[df.index[-1], "support_zone_low"] = self.prev_support["low"]
            df.loc[df.index[-1], "support_zone_high"] = self.prev_support["high"]
            df.loc[df.index[-1], "resistance_zone_low"] = self.prev_resistance["low"]
            df.loc[df.index[-1], "resistance_zone_high"] = self.prev_resistance["high"]
            return df  # Return early to save CPU cycles

        # Compute new zones if needed
        support_zone, resistance_zone = self.find_next_support_resistance(df, last_close)

        # Cache the results
        self.prev_support = support_zone if not support_valid else self.prev_support
        self.prev_resistance = resistance_zone if not resistance_valid else self.prev_resistance

        # Assign results to last row
        df.loc[df.index[-1], "support_zone_low"] = self.prev_support["low"]
        df.loc[df.index[-1], "support_zone_high"] = self.prev_support["high"]
        df.loc[df.index[-1], "resistance_zone_low"] = self.prev_resistance["low"]
        df.loc[df.index[-1], "resistance_zone_high"] = self.prev_resistance["high"]

        return df
