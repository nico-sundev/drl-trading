from dataclasses import dataclass
from pandas import DataFrame


@dataclass
class DataSetContainer:
    training_data: DataFrame
    validation_data: DataFrame
    test_data: DataFrame
