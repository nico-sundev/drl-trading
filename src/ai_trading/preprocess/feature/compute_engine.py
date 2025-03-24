import dask.dataframe as dd
import pandas_ta as ta  # for technical indicators

def compute_features(file_path: str):
    df = dd.read_parquet(file_path)

    # Compute indicators with Dask
    df["macd"] = df["Close"].map_partitions(lambda s: ta.macd(s).iloc[:, 0])
    df["rsi"] = df["Close"].map_partitions(lambda s: ta.rsi(s))
    df["roc"] = df["Close"].map_partitions(lambda s: ta.roc(s))

    df.to_parquet("data/features.parquet", engine="pyarrow")  # Save for Feast
