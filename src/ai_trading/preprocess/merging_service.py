import pandas as pd


class MergingService:
    """Merges a higher timeframe dataset into a lower timeframe dataset."""

    def __init__(self, base_df, higher_df):
        self.base_df = base_df
        self.higher_df = higher_df

    def detect_timeframe(self, df):
        """Auto-detects the timeframe of a dataset."""
        return df["Time"].diff().mode()[0]

    def merge_timeframes(self):
        """Performs optimized two-pointer merge of OHLC data."""
        low_tf = self.detect_timeframe(self.base_df)
        high_tf = self.detect_timeframe(self.higher_df)

        self.base_df["Close_Time"] = self.base_df["Time"] + low_tf
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
                    if col not in ["Time", "Close_Time"]:
                        merged_row[f"HTF_{col}"] = last_closed_candle[col]

            merged_data.append(merged_row)

        return pd.DataFrame(merged_data)