import pandas as pd

from ai_trading.data_set_utils.util import detect_timeframe


class MergeService:
    """Merges a higher timeframe dataset into a lower timeframe dataset."""

    def __init__(self, base_df, higher_df):
        self.base_df = base_df
        self.higher_df = higher_df

    def merge_timeframes(self):
        """Performs optimized two-pointer merge of OHLC data."""
        #low_tf = self.detect_timeframe(self.base_df)
        high_tf = detect_timeframe(self.higher_df)
        high_tf_label = int(high_tf.total_seconds() / 60)


        #self.base_df["Close_Time"] = self.base_df["Time"] + low_tf
        self.higher_df["Close_Time"] = self.higher_df["Time"] + high_tf

        self.base_df = self.base_df.sort_values("Time").reset_index(drop=True)
        self.higher_df = self.higher_df.sort_values("Time").reset_index(drop=True)

        higher_idx = 0
        last_closed_candle = None
        merged_data = []

        for _, row in self.base_df.iterrows():
            current_time = row["Time"]

            while higher_idx < len(self.higher_df) and self.higher_df.iloc[higher_idx]["Close_Time"] <= current_time:
                last_closed_candle = self.higher_df.iloc[higher_idx]
                higher_idx += 1  

            merged_row = row.to_dict()
            if last_closed_candle is not None:
                for col in self.higher_df.columns:
                    if col not in ["Time", "Open", "High", "Low", "Close", "Close_Time"]:
                        merged_row[f"HTF{high_tf_label}_{col}"] = last_closed_candle[col]

            merged_data.append(merged_row)

        return pd.DataFrame(merged_data)