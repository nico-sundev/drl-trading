from feast import Entity, Feature, FeatureView, ValueType
from feast.infra.offline_stores.file_source import FileSource

trading_data = FileSource(
    path="data/features.parquet",
    event_timestamp_column="Time",
)

trading_entity = Entity(name="symbol", value_type=ValueType.STRING, description="Trading Symbol")

technical_indicators_fv = FeatureView(
    name="technical_indicators",
    entities=["symbol"],
    ttl=None,
    batch_source=trading_data,
    features=[
        Feature(name="macd", dtype=ValueType.FLOAT),
        Feature(name="rsi", dtype=ValueType.FLOAT),
        Feature(name="roc", dtype=ValueType.FLOAT),
    ],
)
