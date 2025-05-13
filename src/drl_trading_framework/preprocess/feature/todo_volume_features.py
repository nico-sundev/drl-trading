from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd
import pandas_ta as ta

try:
    from feast import FeatureStore  # Optional: Feature store integration
except ImportError:
    FeatureStore = None


class VolumeIndicators:

    def __init__(
        self,
        df: pd.DataFrame,
        use_feature_store: bool = False,
        feature_store_path: Optional[str] = None,
    ):
        """
        Initialize the class with a DataFrame containing:
        - 'Close': Closing price
        - 'High': Highest price
        - 'Low': Lowest price
        - 'Volume': Trading volume

        Params:
        - use_feature_store (bool): Whether to use a feature store
        - feature_store_path (str, optional): Path to the feature store repo (if applicable)
        """
        self.df = df.copy()
        self.use_feature_store = use_feature_store
        self.feature_store = (
            FeatureStore(feature_store_path)
            if use_feature_store and FeatureStore
            else None
        )

    @lru_cache(maxsize=32)
    def compute_atr(self, length: int = 14) -> pd.Series:
        """Compute the Average True Range (ATR)"""
        return ta.atr(self.df["High"], self.df["Low"], self.df["Close"], length=length)

    @lru_cache(maxsize=32)
    def compute_normalized_volume(self, atr_length: int = 14) -> pd.Series:
        """Compute normalized volume by dividing volume by ATR"""
        atr = self.compute_atr(atr_length)
        return self.df["Volume"] / atr

    @lru_cache(maxsize=32)
    def compute_vroc(self, length: int = 14) -> pd.Series:
        """Compute Volume Rate of Change (VROC) using logarithmic difference"""
        return np.log(self.df["Volume"] / self.df["Volume"].shift(length))

    @lru_cache(maxsize=32)
    def compute_relative_volume(self, length: int = 50) -> pd.Series:
        """Compute relative volume as a Z-score"""
        rolling_mean = self.df["Volume"].rolling(length).mean()
        rolling_std = self.df["Volume"].rolling(length).std()
        return (self.df["Volume"] - rolling_mean) / rolling_std

    @lru_cache(maxsize=32)
    def compute_obv(self) -> pd.Series:
        """Compute On-Balance Volume (OBV)"""
        return ta.obv(self.df["Close"], self.df["Volume"])

    @lru_cache(maxsize=32)
    def compute_normalized_obv(self, atr_length: int = 14) -> pd.Series:
        """Compute OBV normalized by ATR"""
        obv = self.compute_obv()
        atr = self.compute_atr(atr_length)
        return obv / atr

    @lru_cache(maxsize=32)
    def compute_vvr(self, atr_length: int = 14) -> pd.Series:
        """Compute Volume-to-Volatility Ratio (VVR)"""
        atr = self.compute_atr(atr_length)
        return self.df["Volume"] / atr

    def store_features(self, features_df: pd.DataFrame):
        """Store features in the feature store, if enabled"""
        if self.feature_store:
            self.feature_store.write("volume_features", features_df)

    def fetch_features(self) -> Optional[pd.DataFrame]:
        """Fetch features from the feature store, if available"""
        if self.feature_store:
            return self.feature_store.read("volume_features")
        return None

    def compute_all(self) -> pd.DataFrame:
        """Compute all indicators and return as a DataFrame, using caching and feature store where applicable"""
        cached_features = self.fetch_features()
        if cached_features is not None:
            return cached_features  # Return from feature store if available

        indicators = {
            "Normalized_Volume": self.compute_normalized_volume(),
            "VROC": self.compute_vroc(),
            "Relative_Volume": self.compute_relative_volume(),
            "OBV": self.compute_obv(),
            "Normalized_OBV": self.compute_normalized_obv(),
            "VVR": self.compute_vvr(),
        }
        result_df = pd.DataFrame(indicators, index=self.df.index)

        self.store_features(result_df)  # Store in feature store if enabled
        return result_df
