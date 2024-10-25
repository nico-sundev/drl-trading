import logging
import time
import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from ai_trading.preprocess import multi_tf_preprocessing as pp


from ai_trading.data_import.local_data_import_service import LocalDataImportService


# Function to calculate MACD crossover
def add_macd_crossover(df_source, df_target, postfix=""):
    # Create a temporary DataFrame to store MACD values
    temp_df = pd.DataFrame()
    macd = ta.macd(
        df_source["Close"], fast=12, slow=26, signal=9, fillna = True
    )
    df_target["macd_crossover" + postfix] = 0
    df_target["macd_signal_line_bias" + postfix] = 0
    
    temp_df["macd"] = macd["MACD_12_26_9"]
    temp_df["signal"] = macd["MACDs_12_26_9"]
    temp_df["histogram"] = macd["MACDh_12_26_9"]
    for i in range(1, len(temp_df)):
        if (
            temp_df["macd"].iloc[i] > temp_df["signal"].iloc[i]
            and temp_df["macd"].iloc[i - 1] <= temp_df["signal"].iloc[i - 1]
        ):
            df_target.at[i, "macd_crossover" + postfix] = 1
        elif (
            temp_df["macd"].iloc[i] < temp_df["signal"].iloc[i]
            and temp_df["macd"].iloc[i - 1] >= temp_df["signal"].iloc[i - 1]
        ):
            df_target.at[i, "macd_crossover" + postfix] = -1

    # MACD: Using ta.trend.MACD for the signal line
    df_target["macd_signal_line_bias" + postfix] = temp_df["macd"].apply(
        lambda x: 1 if x > 0 else -1
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
    df_target["rvi_7" + postfix] = (
        ta.rvi(df_source["Close"], df_source["High"], df_source["Low"], length=7)
    )
    df_target["rvi_14" + postfix] = (
        ta.rvi(df_source["Close"], df_source["High"], df_source["Low"], length=14)
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

    # Initialize variables to track current 1H and 4H data points
    iter_higher_tf = next(it_higher_tf, None)
    subject_indicator_value = None

    # Add a new column to df_1h to store the 4H data
    base_df[indicator_column + postfix] = None

    # Iterate through the 1H dataset and update the 1H DataFrame
    for index, row in base_df.iterrows():
        # If 1H timestamp >= 4H timestamp, update the last_close_4h with the new 4H close
        if iter_higher_tf and row.Time >= iter_higher_tf.Time:
            subject_indicator_value = iter_higher_tf.Value
            iter_higher_tf = next(it_higher_tf, None)  # Move to the next 4H data point

        # Update the 1H DataFrame with the latest 4H close price
        base_df.at[index, indicator_column + postfix] = subject_indicator_value

    return base_df


# Function to preprocess multi-timeframe OHLCV data and merge with higher timeframe indicators
def merge_timeframes_into_base(data_source: dict):

    # Calculate indicators for each timeframe
    df_1h = calculate_indicators(data_source["H1"], pd.DataFrame(), 1.5)
    df_4h = calculate_indicators(data_source["H4"], pd.DataFrame(), 1.5, "_H4")
    # df_1d = calculate_indicators(data_source["D1"], pd.DataFrame(), 1.5, "_D1")

    # We will now merge indicator values from higher timeframes (4H, 12H, 1D) into the 1H data
    df_merged = df_1h.copy()
    df_merged["Time"] = data_source["H1"]["Time"]
    df_merged["Close"] = data_source["H1"]["Close"]

    df_4h["Time"] = data_source["H4"]["Time"]
    df_4h["Close"] = data_source["H4"]["Close"]

    # df_1d["Time"] = data_source["D1"]["Time"]
    # df_1d["Close"] = data_source["D1"]["Close"]

    # For each 1H candle, find the most recent 4H, 12H, and 1D data points to prevent lookahead bias
    for series_name, series in df_1h.items():
        # Merge 4H indicators
        # df_merged[series_name + '_4h'] = df_merged["Time"].apply(lambda ts: get_past_value(df_4h, ts, series_name))
        # Merge 1D indicators
        # df_merged[series_name + '_1d'] = df_merged["Time"].apply(lambda ts: get_past_value(df_1d, ts, series_name))
        # Merge higher timeframe data (e.g., 6-hour close prices) onto the 1-hour base dataframe
        df_merged = merge_higher_tf_data(df_merged, df_4h, series_name, "Time", "_H4")

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


import_svc = LocalDataImportService()

print(f"Running merge_timeframes")
tf_data_sets: dict

# Generate 1H, 4H, and 1D frequency data
# start_time = "2023-01-01"
# df_1h = generate_mock_ohlcv(start_time, periods=24 * 7, freq="1h")  # 1H for 1 week
# df_4h = generate_mock_ohlcv(start_time, periods=6 * 7, freq="4h")  # 4H for 1 week
# df_1d = generate_mock_ohlcv(start_time, periods=7, freq="1D")  # 1D for 1 week

# # Align datasets
# df_1h_aligned, df_4h_aligned, df_1d_aligned = align_datasets(df_1h, df_4h, df_1d)

# # Show the first few rows of each dataframe to confirm alignment
# # df_1h_aligned.head(12), df_4h_aligned.head(), df_1d_aligned.head()
# print(df_1h_aligned.head(12))
# print(df_4h_aligned.head(12))
# # print(df_1d_aligned.head(12))
# tf_data_sets = {"H1": df_1h_aligned, "H4": df_4h_aligned}

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# # Console handler
# console_handler = logging.StreamHandler()
# console_handler.setLevel(logging.DEBUG)

# # Formatter
# formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# console_handler.setFormatter(formatter)

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
# # Add the handler to the logger
# logger.addHandler(console_handler)

# # Test messages
# logger.debug("This is a debug message.")
# logger.info("This is an info message.")
# logger.warning("This is a warning message.")
# logger.error("This is an error message.")
# logger.critical("This is a critical message.")

start_time = time.time()
tf_data_sets = import_svc.import_data(300)
end_time = time.time()
execution_time = end_time - start_time
logging.debug(f"Import data Execution time: {execution_time} seconds")

start_time = time.time()
merged_dataframe: pd.DataFrame = pp.merge_timeframes_into_base(tf_data_sets)
end_time = time.time()
execution_time = end_time - start_time
logging.debug(f"Merge timeframes Execution time: {execution_time} seconds")

print(merged_dataframe.head(100))
