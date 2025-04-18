from typing import Optional

from numpy import mean
from pandas import DataFrame

from ai_trading.preprocess.feature.custom.enum.wick_handle_strategy_enum import (
    WICK_HANDLE_STRATEGY,
)


class WickHandler:
    @staticmethod
    def calculate_wick_threshold(
        df: DataFrame,
        index: int,
        strategy: WICK_HANDLE_STRATEGY,
        atr: Optional[float] = None,
    ) -> float:
        if df.empty:
            raise ValueError("DataFrame is empty")

        prev_row, pre_prev_row = df.iloc[index] if index >= 0 else None, (
            df.iloc[index - 1] if index > 0 else None
        )

        prev_candle_green = prev_row["Close"] > prev_row["Open"]

        def last_wick():
            if prev_row is None:
                raise ValueError("Not enough data for last wick calculation")
            return prev_row["Low"] if prev_candle_green else prev_row["High"]

        def previous_wick():
            if pre_prev_row is None:
                raise ValueError("Not enough data for previous wick calculation")
            return pre_prev_row["Low"] if prev_candle_green else pre_prev_row["High"]

        def mean_wick():
            return mean([last_wick(), previous_wick()])

        def max_below_atr():
            if atr is None:
                raise ValueError("ATR value is required for MAX_BELOW_ATR strategy")
            return (
                max([min([last_wick(), previous_wick()]), prev_row["Open"] - atr])
                if prev_candle_green
                else min(
                    [([last_wick(), previous_wick()]).max(), prev_row["Open"] + atr]
                )
            )

        strategy_map = {
            WICK_HANDLE_STRATEGY.LAST_WICK_ONLY: last_wick,
            WICK_HANDLE_STRATEGY.PREVIOUS_WICK_ONLY: previous_wick,
            WICK_HANDLE_STRATEGY.MEAN: mean_wick,
            WICK_HANDLE_STRATEGY.MAX_BELOW_ATR: max_below_atr,
        }

        if strategy not in strategy_map:
            raise ValueError(f"Unknown strategy: {strategy}")

        return strategy_map[strategy]()  # Calls the corresponding function
