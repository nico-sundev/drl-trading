import numpy as np
import pandas as pd

from ai_trading.preprocess.feature.custom.enum.wick_handle_strategy_enum import (
    WICK_HANDLE_STRATEGY,
)
from ai_trading.preprocess.feature.custom.wick_handler import WickHandler

PIVOT_HIGH = "pivot_high"
PIVOT_LOW = "pivot_low"


class SupportResistanceFinder:
    """
    Finds support and resistance zones based on pivot points.
    Uses a lookback period and caches previous zones for efficiency.
    """

    def __init__(
        self,
        source_data_frame: pd.DataFrame,
        lookback: int,
        wick_handle_strategy: WICK_HANDLE_STRATEGY,
    ) -> None:
        self.lookback = lookback
        self.wick_handle_strategy = wick_handle_strategy
        self.source_data_frame = source_data_frame
        self.pivot_cache = pd.DataFrame(columns=["index", "top", "bottom", "type"])
        self.prev_support = None
        self.prev_resistance = None

    def validate_dataframe(self):
        """Ensures required columns exist and data is valid."""
        required_cols = {"Close", "High", "Low"}
        df = self.source_data_frame
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        if df.empty:
            raise ValueError("DataFrame is empty.")

        if df.isnull().any().any():
            raise ValueError("DataFrame contains NaN values.")

    def find_pivot_points(self, last_index: int) -> tuple:
        """
        Identifies pivot highs and lows (local maxima/minima) in the dataset.
        """
        if last_index < 1:
            return [False, False]

        df = self.source_data_frame
        found_pivot_high = (
            df["Close"].iloc[last_index] < df["Open"].iloc[last_index]
        ) & (df["Close"].iloc[last_index - 1] > df["Open"].iloc[last_index - 1])
        found_pivot_low = (
            df["Close"].iloc[last_index] > df["Open"].iloc[last_index]
        ) & (df["Close"].iloc[last_index - 1] < df["Open"].iloc[last_index - 1])
        return [found_pivot_high, found_pivot_low]

    def clean_pivot_cache(self, last_close: float) -> None:
        self.pivot_cache.drop(
            self.pivot_cache[
                (self.pivot_cache["type"] == PIVOT_HIGH)
                & (self.pivot_cache["top"] < last_close)
                | (self.pivot_cache["type"] == PIVOT_LOW)
                & (self.pivot_cache["bottom"] > last_close)
            ].index,
            inplace=True,
        )

    def find_next_zone(self, type: str, last_close: float) -> dict[str, float]:
        """
        Iterates backward to find the nearest valid support and resistance within the lookback window.
        """
        zone = {"bottom": np.nan, "top": np.nan}

        for i in range(
            len(self.pivot_cache) - 1,
            max(-1, len(self.pivot_cache) - 1 - self.lookback),
            -1,
        ):
            curr_pivot_point = self.pivot_cache.iloc[i]
            if (
                type == PIVOT_HIGH
                and curr_pivot_point["type"] == PIVOT_HIGH
                and curr_pivot_point["top"] > last_close
            ):
                zone["bottom"] = curr_pivot_point["bottom"]
                zone["top"] = curr_pivot_point["top"]
                break

            elif (
                type == PIVOT_LOW
                and curr_pivot_point["type"] == PIVOT_LOW
                and curr_pivot_point["bottom"] < last_close
            ):
                zone["bottom"] = curr_pivot_point["bottom"]
                zone["top"] = curr_pivot_point["top"]
                break

        return zone

    def find_support_resistance_zones(
        self,
    ) -> pd.DataFrame:
        """
        Computes support and resistance zones for the last candle in the dataset.
        Caches zones for efficiency and reuses them if they remain valid.
        """
        self.validate_dataframe()  # Validate input data
        df = self.source_data_frame.copy()
        clean_cache = False

        for index, row in df.iterrows():

            if index < 2:
                df.loc[index, "resistance_range"] = np.nan
                df.loc[index, "support_range"] = np.nan
                continue

            last_close = row["Close"]

            resistance_valid = (
                self.prev_resistance and last_close < self.prev_resistance["top"]
            )
            support_valid = (
                self.prev_support and last_close > self.prev_support["bottom"]
            )
            [found_pivot_high, found_pivot_low] = self.find_pivot_points(index)

            # check if new pivot high has been created
            if found_pivot_high:
                pivot_top = WickHandler.calculate_wick_threshold(
                    df, index, self.wick_handle_strategy
                )
                # update cache
                self.pivot_cache.loc[len(self.pivot_cache)] = [
                    index,
                    pivot_top,
                    row["Open"],
                    PIVOT_HIGH,
                ]
                # Calculate range
                df.loc[index, "resistance_range"] = row["Open"] - last_close
                # update prev zone references
                self.prev_resistance = {"bottom": row["Open"], "top": pivot_top}
            # check if former resistance is still valid
            elif resistance_valid:
                # Calculate range
                df.loc[index, "resistance_range"] = (
                    self.prev_resistance["bottom"] - last_close
                )
            # resistance must be recalculated
            else:
                next_resistance_zone = self.find_next_zone(PIVOT_HIGH, last_close)
                if not np.isnan(next_resistance_zone["bottom"]):
                    df.loc[index, "resistance_range"] = (
                        next_resistance_zone["bottom"] - last_close
                    )
                    self.prev_resistance = next_resistance_zone
                    clean_cache = True

            # check if new pivot low has been created
            if found_pivot_low:
                pivot_bottom = WickHandler.calculate_wick_threshold(
                    df, index, self.wick_handle_strategy
                )
                pivot_top = row["Open"]
                # update cache
                self.pivot_cache.loc[len(self.pivot_cache)] = [
                    index,
                    pivot_top,
                    pivot_bottom,
                    PIVOT_LOW,
                ]
                # Calculate range
                df.loc[index, "support_range"] = pivot_top - last_close
                # update prev zone references
                self.prev_support = {"bottom": pivot_bottom, "top": pivot_top}

            # check if former support is still valid
            elif support_valid:
                pivot_top = self.prev_support["top"]
                # Calculate range
                df.loc[index, "support_range"] = pivot_top - last_close
            # support must be recalculated
            else:
                next_support_zone = self.find_next_zone(PIVOT_LOW, last_close)
                if not np.isnan(next_support_zone["top"]):
                    pivot_top = next_support_zone["top"]
                    df.loc[index, "support_range"] = pivot_top - last_close
                    self.prev_support = next_support_zone
                    clean_cache = True

            if clean_cache:
                self.clean_pivot_cache(last_close)

        return df
