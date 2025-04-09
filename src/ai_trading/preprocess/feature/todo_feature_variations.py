import numpy as np
import pandas as pd
import pandas_ta as talib

class FeatureEngineer:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def compute_roc(self, column: str, periods: list) -> pd.DataFrame:
        """Compute multiple ROC variations for a given column."""
        for period in periods:
            self.df[f"roc_{column}_{period}"] = np.log(self.df[column] / self.df[column].shift(period))
        return self.df

    def compute_moving_highest_signal(self, column: str, window: int) -> pd.Series:
        """Binary feature: was there a signal in the last N candles?"""
        return self.df[column].rolling(window).max() == self.df[column]

    def compute_volatility_agnostic_distance(self, indicator: str, atr_length: int) -> pd.Series:
        """Normalize price-based indicators using ATR."""
        atr = talib.ATR(self.df["High"], self.df["Low"], self.df["Close"], timeperiod=atr_length)
        return (self.df[indicator] - self.df["Close"]) / atr

    def compute_trend_persistence(self, column: str, window: int) -> pd.Series:
        """Measure how long price stays above/below a moving average."""
        return self.df[column] > self.df[column].rolling(window).mean()

    def compute_relative_volume(self, volume_window: int) -> pd.Series:
        """Compare current volume to past X candles."""
        return self.df["Volume"] / self.df["Volume"].rolling(volume_window).mean()

    def compute_percentile_rank(self, column: str, window: int) -> pd.Series:
        """Calculate the percentile rank of an indicator within a rolling window."""
        return self.df[column].rolling(window).apply(lambda x: np.argsort(x)[-1] / len(x), raw=True)

# Example Usage
df = pd.DataFrame({"Close": np.random.rand(100), "High": np.random.rand(100), "Low": np.random.rand(100), "Volume": np.random.rand(100)})
fe = FeatureEngineer(df)
df = fe.compute_roc("Close", [3, 7, 14])
df["trend_persistence"] = fe.compute_trend_persistence("Close", 20)
df["volatility_agnostic_macd"] = fe.compute_volatility_agnostic_distance("Close", 14)
