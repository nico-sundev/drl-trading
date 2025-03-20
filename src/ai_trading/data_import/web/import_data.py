import yfinance as yf
import matplotlib.pyplot as plt
from IPython.display import display

from ai_trading.model.dataset_container import DataSetContainer


# List of stocks in the Dow Jones 30
tickers = [
    'MMM', 'AXP', 'AAPL'
]

# Get historical data from Yahoo Finance and save it to dictionary
def fetch_stock_data(tickers, start_date, end_date):
    stock_data = {}
    for ticker in tickers:
        stock_data[ticker] = yf.download(ticker, start=start_date, end=end_date)
    return stock_data


# create data sets
def create_data_sets(stock_data):

    # split the data into training, validation and test sets
    training_data_time_range = ('2009-01-01', '2015-12-31')
    validation_data_time_range = ('2016-01-01', '2016-12-31')
    test_data_time_range = ('2017-01-01', '2020-05-08')

    # split the data into training, validation and test sets
    training_data = {}
    validation_data = {}
    test_data = {}

    for ticker, df in stock_data.items():
        training_data[ticker] = df.loc[training_data_time_range[0]:training_data_time_range[1]]
        validation_data[ticker] = df.loc[validation_data_time_range[0]:validation_data_time_range[1]]
        test_data[ticker] = df.loc[test_data_time_range[0]:test_data_time_range[1]]
        
    return DataSetContainer(training_data, validation_data, test_data)
    
def test():
    # Call the function to get data
    stock_data = fetch_stock_data(tickers, '2009-01-01', '2020-05-08')
    data_sets = create_data_sets(stock_data)
    # print shape of training, validation and test data
    ticker = 'AAPL'
    print(f'- Training data shape for {ticker}: {data_sets.training_data[ticker].shape}')
    print(f'- Validation data shape for {ticker}: {data_sets.validation_data[ticker].shape}')
    print(f'- Test data shape for {ticker}: {data_sets.test_data[ticker].shape}\n')


    # Display the first 5 rows of the data
    display(stock_data['AAPL'].head())
    print('\n')

    # Plot:
    plt.figure(figsize=(12, 4))
    plt.plot(data_sets.training_data[ticker].index, data_sets.training_data[ticker]['Open'], label='Training', color='blue')
    plt.plot(data_sets.validation_data[ticker].index, data_sets.validation_data[ticker]['Open'], label='Validation', color='red')
    plt.plot(data_sets.test_data[ticker].index, data_sets.test_data[ticker]['Open'], label='Test', color='green')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title(f'{ticker} Stock, Open Price')
    plt.legend()
    plt.show()
