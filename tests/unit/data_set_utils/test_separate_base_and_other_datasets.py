from unittest.mock import MagicMock
from ai_trading.data_set_utils.util import separate_base_and_other_datasets
from ai_trading.model.computed_dataset_container import ComputedDataSetContainer
from ai_trading.model.asset_price_dataset import AssetPriceDataSet

def test_separate_base_and_other_datasets():
    # Mock datasets
    base_dataset = MagicMock(spec=AssetPriceDataSet)
    base_dataset.base_dataset = True

    other_dataset_1 = MagicMock(spec=AssetPriceDataSet)
    other_dataset_1.base_dataset = False

    other_dataset_2 = MagicMock(spec=AssetPriceDataSet)
    other_dataset_2.base_dataset = False

    base_dataset_container = ComputedDataSetContainer(
        source_dataset=base_dataset, computed_dataframe=MagicMock()
    )
    other_dataset_container_1 = ComputedDataSetContainer(
        source_dataset=other_dataset_1, computed_dataframe=MagicMock()
    )
    other_dataset_container_2 = ComputedDataSetContainer(
        source_dataset=other_dataset_2, computed_dataframe=MagicMock()
    )

    datasets = [base_dataset_container, other_dataset_container_1, other_dataset_container_2]

    # Call the function
    result_base, result_others = separate_base_and_other_datasets(datasets)

    # Assertions
    assert result_base == base_dataset_container
    assert other_dataset_container_1 in result_others
    assert other_dataset_container_2 in result_others
    assert len(result_others) == 2