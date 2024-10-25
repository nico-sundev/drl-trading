import logging
import pandas as pd
import pytest
from ai_trading.preprocess import dataset_combiner as dc

@pytest.fixture 
def logger():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    yield logging
    logging.info("Finish test.")


def test_combination(logger):
    print(f"Running test_combination")
    
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

    combined_df: pd.DataFrame = dc.combine(data_a, data_b)
    logger.info(combined_df)
    assert len(combined_df) == 10
    
