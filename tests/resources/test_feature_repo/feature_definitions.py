from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32

# Define entity for trading symbols
symbol_entity = Entity(name="symbol", join_keys=["symbol"])

# Define source for technical indicators data
indicator_source = FileSource(
    path="../test_feature_store_data",  # Path is relative to the repository
    timestamp_field="event_timestamp",
)

# Define feature view for RSI
rsi_feature_view = FeatureView(
    name="rsi_default",
    entities=[symbol_entity],
    ttl=timedelta(days=1),
    schema=[
        Field(name="rsi_7", dtype=Float32),
        Field(name="rsi_14", dtype=Float32),
    ],
    source=indicator_source,
    online=False,
    tags={"feature_type": "rsi"},
)

# Define feature view for MACD
macd_feature_view = FeatureView(
    name="macd_default",
    entities=[symbol_entity],
    ttl=timedelta(days=1),
    schema=[
        Field(name="macd", dtype=Float32),
        Field(name="macd_signal", dtype=Float32),
        Field(name="macd_histogram", dtype=Float32),
    ],
    source=indicator_source,
    online=False,
    tags={"feature_type": "macd"},
)
