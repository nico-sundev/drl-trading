import pandas as pd
from feast import FeatureStore

fs = FeatureStore(repo_path="feature_repo")


def query_stored_features():
    training_df = fs.get_historical_features(
        entity_df=pd.DataFrame(
            {"symbol": ["AAPL", "TSLA"], "Time": ["2025-03-01", "2025-03-02"]}
        ),
        features=[
            "technical_indicators:macd",
            "technical_indicators:rsi",
            "technical_indicators:roc",
        ],
    ).to_df()
    print(training_df)
    return training_df


def query_real_time_features():
    return fs.get_online_features(
        features=[
            "technical_indicators:macd",
            "technical_indicators:rsi",
            "technical_indicators:roc",
        ],
        entity_rows=[{"symbol": "AAPL"}],
    ).to_dict()
