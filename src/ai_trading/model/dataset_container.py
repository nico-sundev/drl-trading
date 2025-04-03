from pandas import DataFrame


class DataSetContainer:
    def __init__(
        self, training_data: DataFrame, validation_data: DataFrame, test_data: DataFrame
    ):
        self.training_data = training_data
        self.validation_data = validation_data
        self.test_data = test_data
