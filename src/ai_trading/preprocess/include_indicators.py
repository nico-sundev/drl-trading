import numpy as np
from IPython.display import display
import pandas as pd
from ai_trading.data_import.import_data import create_data_sets, fetch_stock_data


def add_technical_indicators(df) -> pd.DataFrame:

    df = df.copy()
    return df

def preprocess_dataset(tickers):
    # Call the function to get data
    stock_data = fetch_stock_data(tickers, '2009-01-01', '2020-05-08')
    data_sets = create_data_sets(stock_data)
    # add technical indicators to the training data for each stock
    for ticker, df in data_sets.training_data.items():
        data_sets.training_data[ticker] = add_technical_indicators(df)

    # add technical indicators to the validation data for each stock
    for ticker, df in data_sets.validation_data.items():
        data_sets.validation_data[ticker] = add_technical_indicators(df)

    # add technical indicators to the test data for each stock
    for ticker, df in data_sets.test_data.items():
        data_sets.test_data[ticker] = add_technical_indicators(df)
        
    return data_sets
