import pandas as pd
from ai_trading.model.asset_price_dataset import AssetPriceDataSet


def mock_ohlcv_data_1h(base_dataset=True):
    """Fixture to provide mock OHLCV data wrapped in AssetPriceDataSet."""
    data = {
        "Time": [
            "2008-10-03 13:00:00",
            "2008-10-03 14:00:00",
            "2008-10-03 15:00:00",
            "2008-10-03 16:00:00",
            "2008-10-03 17:00:00",
        ],
        "Open": [1.37475, 1.3748, 1.3809, 1.3816, 1.38],
        "High": [1.3767, 1.38285, 1.38415, 1.3863, 1.3871],
        "Low": [1.37025, 1.37175, 1.3783, 1.37895, 1.3789],
        "Close": [1.3748, 1.3809, 1.38155, 1.37995, 1.38585],
        "Volume": [73566, 58872, 52659, 37992, 47956]
    }
    df = pd.DataFrame(data)
    df["Time"] = pd.to_datetime(df["Time"])

    return AssetPriceDataSet("H1", base_dataset, asset_price_dataset=df)


def mock_ohlcv_data_4h(base_dataset=True):
    """Fixture to provide mock OHLCV data for 4H timeframe wrapped in AssetPriceDataSet."""
    data = {
        "Time": [
            "2008-10-03 00:00:00",
            "2008-10-03 04:00:00",
            "2008-10-03 08:00:00",
            "2008-10-03 12:00:00",
            "2008-10-03 16:00:00",
        ],
        "Open": [1.3700, 1.3740, 1.3780, 1.3820, 1.3860],
        "High": [1.3750, 1.3790, 1.3830, 1.3870, 1.3910],
        "Low": [1.3650, 1.3690, 1.3730, 1.3770, 1.3810],
        "Close": [1.3740, 1.3780, 1.3820, 1.3860, 1.3900],
        "Volume": [100000, 95000, 90000, 85000, 80000]
    }
    df = pd.DataFrame(data)
    df["Time"] = pd.to_datetime(df["Time"])

    return AssetPriceDataSet("H4", base_dataset, asset_price_dataset=df)