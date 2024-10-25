

import logging
import pandas as pd

# Combines two dataframes into one, while scaling the open/high/low/close values of data_b
# based on the last value of data_a respectively.
def combine(data_a, data_b) -> pd.DataFrame:
    # 1. Get the last Close price from Dataset A (GBPUSD)
    last_close_a = data_a['Close'].iloc[-1]

    # 2. Calculate the percentage returns for Dataset B (BTCUSD)
    pct_returns_b = data_b['Close'].pct_change().fillna(0)  # Fill NaN with 0 for the first row

    # 3. Initialize the scaled DataFrame for BTCUSD
    data_b_scaled = data_b.copy()

    # 4. Scale the BTCUSD close prices dynamically based on percentage returns
    data_b_scaled['Scaled_Close'] = last_close_a * (1 + pct_returns_b).cumprod()

    # 5. Calculate the percentage difference from Close to High/Low
    high_diff_pct = data_b['High'] / data_b['Close'] - 1
    low_diff_pct = data_b['Low'] / data_b['Close'] - 1

    # 6. Apply the same percentage difference to scale the High and Low based on the Scaled_Close
    data_b_scaled['Scaled_High'] = data_b_scaled['Scaled_Close'] * (1 + high_diff_pct)
    data_b_scaled['Scaled_Low'] = data_b_scaled['Scaled_Close'] * (1 + low_diff_pct)

    # 7. Adjust the timestamps to follow Dataset A
    last_time_a = data_a['Time'].iloc[-1]
    data_b_scaled['Time'] = pd.date_range(start=last_time_a + pd.Timedelta(hours=1), periods=len(data_b), freq='H')

    # 8. Combine dataset A and scaled dataset B
    combined_data = pd.concat([data_a, data_b_scaled[['Time', 'Scaled_Close', 'Scaled_High', 'Scaled_Low']]], ignore_index=True)

    # Rename the scaled columns for clarity
    combined_data.rename(columns={
        'Scaled_Close': 'Close',
        'Scaled_High': 'High',
        'Scaled_Low': 'Low'
    }, inplace=True)

    return combined_data

# Sample data for GBPUSD (Dataset A) and BTCUSD (Dataset B)
data_a = pd.DataFrame({
    'Time': pd.date_range(start='2023-01-01', periods=5, freq='H'),
    'Open': [1.2000, 1.2010, 1.2020, 1.2030, 1.2040],
    'High': [1.2020, 1.2030, 1.2040, 1.2050, 1.2060],
    'Low': [1.1990, 1.2000, 1.2010, 1.2020, 1.2030],
    'Close': [1.2015, 1.2025, 1.2035, 1.2045, 1.2055]
})

data_b = pd.DataFrame({
    'Time': pd.date_range(start='2023-01-01', periods=5, freq='H'),
    'Open': [30000, 30500, 31000, 31500, 32000],
    'High': [30500, 31000, 31500, 32000, 32500],
    'Low': [29500, 30000, 30500, 31000, 31500],
    'Close': [30250, 30750, 31250, 31750, 32250]
})
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

combined_df: pd.DataFrame = combine(data_a, data_b)
logging.info(combined_df)