import pandas as pd

from ai_trading.data_set_utils.util import detect_timeframe


class MergeService:
    """Merges a higher timeframe dataset into a lower timeframe dataset."""

    def __init__(self) -> None:
        """Initializes the MergeService class."""
        pass

    def merge_timeframes(
        self, base_df: pd.DataFrame, higher_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Performs optimized two-pointer merge of OHLC data."""
        high_tf = detect_timeframe(higher_df)
        high_tf_label = int(high_tf.total_seconds() / 60)

        higher_df["Close_Time"] = higher_df["Time"] + high_tf

        base_df = base_df.sort_values("Time").reset_index(drop=True)
        higher_df = higher_df.sort_values("Time").reset_index(drop=True)

        higher_idx = 0
        last_closed_candle = None
        merged_data = []

        for _, row in base_df.iterrows():
            current_base_time = row["Time"]

            while (
                higher_idx < len(higher_df)
                and higher_df.iloc[higher_idx]["Close_Time"] <= current_base_time
            ):
                last_closed_candle = higher_df.iloc[higher_idx]
                higher_idx += 1

            merged_row: dict = {}
            merged_row["Time"] = current_base_time
            if last_closed_candle is not None:
                for col in higher_df.columns:
                    if col not in [
                        "Time",
                        "Open",
                        "High",
                        "Low",
                        "Close",
                        "Close_Time",
                        "Volume",
                    ]:
                        merged_row[f"HTF{high_tf_label}_{col}"] = last_closed_candle[
                            col
                        ]

            merged_data.append(merged_row)

        return pd.DataFrame(merged_data)
