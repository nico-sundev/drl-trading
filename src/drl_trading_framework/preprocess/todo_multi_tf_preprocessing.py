import os

import numpy as np
import pandas as pd
import pandas_ta as ta  # type: ignore


class TimeSeriesLengthError(Exception):
    pass


# Function to calculate MACD crossover
def add_macd_crossover(df_source, df_target, postfix=""):
    # Create a temporary DataFrame to store MACD values
    macd = ta.macd(
        df_source["Close"],
        fast=12,
        slow=26,
        signal=9,
        fillna=True,
        signal_indicators=True,
    )
    df_target["macd_cross_bullish" + postfix] = macd["MACDh_12_26_9_XA_0"]
    df_target["macd_cross_bearish" + postfix] = macd["MACDh_12_26_9_XB_0"]
    df_target["macd_trend" + postfix] = macd["MACD_12_26_9_A_0"]

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


# Function to calculate RSI and append to origin DateFrame
def add_rsi(df_source, df_target, postfix=""):
    df_target["rsi_14" + postfix] = ta.rsi(df_source["Close"], length=14)
    df_target["rsi_3" + postfix] = ta.rsi(df_source["Close"], length=3)
    return df_target


# Function to calculate multiple Rate of change Series and append to origin DataFrame
def add_roc(df_source, df_target, postfix=""):

    # Normalize the 'roc_3' column
    df_target["roc_3" + postfix] = ta.roc(df_source["Close"], length=3)

    # Normalize the 'roc_7' column
    df_target["roc_7" + postfix] = ta.roc(df_source["Close"], length=7)

    # Normalize the 'roc_14' column
    df_target["roc_14" + postfix] = ta.roc(df_source["Close"], length=14)

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
    support_upper_bound = temp_df[["bb_lower", "atr_lower_band", "donchian_lower"]].max(
        axis=1
    )
    support_lower_bound = temp_df[["bb_lower", "atr_lower_band", "donchian_lower"]].min(
        axis=1
    )
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


# Function to calculate entry and exit strategy indicators
def add_entry_exit_indicators(df_source, df_target, atr_multiplier=1.5, postfix=""):
    df_target["williams_r_14" + postfix] = ta.willr(
        df_source["High"], df_source["Low"], df_source["Close"], length=14
    )
    df_target["cci_14" + postfix] = ta.cci(
        df_source["High"], df_source["Low"], df_source["Close"], length=14
    )

    add_extreme_zones_from_ma(df_source, df_target, atr_multiplier, postfix)
    add_extreme_zones_from_bands(df_source, df_target, postfix)

    return df_target


# function to calculate market dynamic indicators
def add_market_dynamic_indicators(df_source, df_target, postfix=""):
    df_target["rvi_7" + postfix] = ta.rvi(
        df_source["Close"], df_source["High"], df_source["Low"], length=7
    )
    df_target["rvi_14" + postfix] = ta.rvi(
        df_source["Close"], df_source["High"], df_source["Low"], length=14
    )
    return df_target


# Function to add trend indicator features
def add_trend_related_indicators(df_source, df_target, postfix=""):
    add_rsi(df_source, df_target, postfix)
    add_roc(df_source, df_target, postfix)
    add_macd_crossover(df_source, df_target, postfix)
    return df_target


# Calculate all technical indicator features
def calculate_indicators(df_source, df_target, atr_multiplier=1.5, postfix=""):
    add_trend_related_indicators(df_source, df_target, postfix)
    add_market_dynamic_indicators(df_source, df_target, postfix)
    add_entry_exit_indicators(df_source, df_target, atr_multiplier, postfix)
    return df_target


# Helper function to get the most recent value from a higher timeframe
def get_past_value(higher_tf_df, timestamp, indicator_column):
    # Find the most recent entry from the higher timeframe that is before or equal to the given timestamp
    past_data = higher_tf_df[higher_tf_df["Time"] <= timestamp]
    if not past_data.empty:
        return past_data.iloc[-1][
            indicator_column
        ]  # Return the most recent row's value
    else:
        return None  # Return None if no past data exists


# Helper function to merge and forward-fill higher timeframe values onto the base 1-hour timeframe
def merge_higher_tf_data(
    base_df: pd.DataFrame,
    higher_tf_df,
    indicator_column,
    time_column="Time",
    postfix="",
):
    snippet_higher_tf = pd.DataFrame()
    snippet_higher_tf[time_column] = higher_tf_df[time_column]
    snippet_higher_tf["Value"] = higher_tf_df[indicator_column + postfix]

    # Create iterators for both datasets
    it_higher_tf = snippet_higher_tf.itertuples()

    # move higher TF pointer,
    # future one pointing to t + 2
    t_plus_2_iter_higher_tf = next(it_higher_tf, None)

    if t_plus_2_iter_higher_tf is None:
        raise TimeSeriesLengthError("Cursor future_iter_higher_tf is invalid")

    # Add a new column to df_1h to store the 4H data
    base_df[indicator_column + postfix] = None
    # Cache last recent finished candle record
    last_recent_finished_record = None

    # Position the past_iter_higher_tf cursor correctly to be able
    # to start the multi TF merge operation.
    # future_iter_higher_tf Cursor will be carried along the way to keep
    # holding t+1 state
    while (
        t_plus_2_iter_higher_tf
        and base_df.iloc[0]["Time"] > t_plus_2_iter_higher_tf.Time
    ):
        last_recent_finished_record = t_plus_2_iter_higher_tf.Value
        t_plus_2_iter_higher_tf = next(it_higher_tf, None)

    # This is t + 1 Cursor
    t_plus_1_iter_higher_tf = t_plus_2_iter_higher_tf
    # Position the t_plus_2_iter_higher_tf Cursor to be t+2
    t_plus_2_iter_higher_tf = next(it_higher_tf, None)

    # Iterate through the 1H dataset and update the 1H DataFrame
    for index, row in base_df.iterrows():
        # If 1H timestamp >= 4H timestamp, update the last_close_4h with the new 4H close
        if t_plus_2_iter_higher_tf and row.Time >= t_plus_2_iter_higher_tf.Time:
            last_recent_finished_record = t_plus_1_iter_higher_tf.Value
            t_plus_1_iter_higher_tf = t_plus_2_iter_higher_tf
            t_plus_2_iter_higher_tf = next(
                it_higher_tf, None
            )  # Move to the next 4H data point

        # Update the 1H DataFrame with the latest 4H close price
        base_df.at[index, indicator_column + postfix] = last_recent_finished_record

    return base_df


# Function to preprocess multi-timeframe OHLCV data and merge with higher timeframe indicators
def merge_timeframes_into_base(data_source: dict, write_results_to_disk=False):

    # Calculate indicators for each timeframe
    df_1h = calculate_indicators(data_source["H1"], pd.DataFrame(), 1.5)
    df_4h = calculate_indicators(data_source["H4"], pd.DataFrame(), 1.5, "_H4")
    # df_1d = calculate_indicators(data_source["D1"], pd.DataFrame(), 1.5, "_D1")

    # We will now merge indicator values from higher timeframes (4H, 12H, 1D) into the 1H data
    df_merged = df_1h.copy()
    # logging.debug(df_merged)
    df_merged["Time"] = data_source["H1"]["Time"]
    df_merged["Close"] = data_source["H1"]["Close"]
    # logging.debug(df_merged)

    df_4h["Time"] = data_source["H4"]["Time"]
    df_4h["Close"] = data_source["H4"]["Close"]

    if write_results_to_disk:
        df_merged.to_csv(
            os.path.join(
                os.path.dirname(__file__),
                "..\\..\\..\\data\\processed\\df_merged_with_indicators.csv",
            ),
            date_format="%Y-%m-%d %H:%M:%S",
            sep=";",
        )
        df_4h.to_csv(
            os.path.join(
                os.path.dirname(__file__),
                "../../../data/processed/df_4h_with_indicators.csv",
            ),
            date_format="%Y-%m-%d %H:%M:%S",
            sep=";",
        )

    # df_1d["Time"] = data_source["D1"]["Time"]
    # df_1d["Close"] = data_source["D1"]["Close"]

    # For each 1H candle, find the most recent 4H, 12H, and 1D data points to prevent lookahead bias
    for series_name, _series in df_1h.items():
        # Merge higher timeframe data (e.g., 6-hour close prices) onto the 1-hour base dataframe
        df_merged = merge_higher_tf_data(df_merged, df_4h, series_name, "Time", "_H4")

    if write_results_to_disk:
        df_merged.to_csv(
            os.path.join(
                os.path.dirname(__file__),
                "..\\..\\..\\data\\processed\\df_merged_combined_with_indicators.csv",
            ),
            date_format="%Y-%m-%d %H:%M:%S",
            sep=";",
        )

    return df_merged


# Function to generate mock OHLCV data for given frequency
def generate_mock_ohlcv(start_time, periods, freq):
    np.random.seed(42)
    time = pd.date_range(start=start_time, periods=periods, freq=freq)
    open_prices = np.random.uniform(low=100, high=200, size=periods)
    high_prices = open_prices + np.random.uniform(low=0, high=10, size=periods)
    low_prices = open_prices - np.random.uniform(low=0, high=10, size=periods)
    close_prices = np.random.uniform(low=100, high=200, size=periods)
    volume = np.random.uniform(low=1000, high=10000, size=periods)

    data = {
        "Time": time,
        "Open": open_prices,
        "High": high_prices,
        "Low": low_prices,
        "Close": close_prices,
        "Volume": volume,
    }

    return pd.DataFrame(data)


# Function to ensure alignment of OHLCV data across different timeframes
def align_datasets(df_1h, df_4h, df_1d):
    # Align 1H to 4H: Set the open of each 4H period to the close of the last corresponding 1H period
    for i in range(len(df_4h)):
        corresponding_1h_idx = (i + 1) * 4 - 1  # Index of the last 1H in the 4H period
        df_4h.at[i, "Open"] = df_1h.at[corresponding_1h_idx, "Close"]

    # Align 4H to 1D: Set the open of each 1D period to the close of the last corresponding 4H period
    for i in range(len(df_1d)):
        corresponding_4h_idx = (i + 1) * 6 - 1  # Index of the last 4H in the 1D period
        df_1d.at[i, "Open"] = df_4h.at[corresponding_4h_idx, "Close"]

    return df_1h, df_4h, df_1d
