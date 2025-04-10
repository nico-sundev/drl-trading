from dataclasses import dataclass
from pandas import DataFrame


@dataclass
class PreprocessedDataSetContainer:
    training_data: DataFrame
    validation_data: DataFrame
    test_data: DataFrame
