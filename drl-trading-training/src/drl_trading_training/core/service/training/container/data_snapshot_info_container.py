from dataclasses import dataclass
import datetime


@dataclass
class DataSnapshotInfo:
    """Track when data was available for training."""
    snapshot_date: datetime
    data_range_start: datetime
    data_range_end: datetime
    records_count: int
    feature_config_hash: str
