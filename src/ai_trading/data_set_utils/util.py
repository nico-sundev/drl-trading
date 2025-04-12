from typing import List

from ai_trading.model.computed_dataset_container import ComputedDataSetContainer


def separate_base_and_other_datasets(datasets: List[ComputedDataSetContainer]) -> tuple:
    """
    Separates the datasets into base and other datasets.
    """
    base_dataset = None
    other_datasets = []
    
    for dataset in datasets:
        if dataset.source_dataset.base_dataset:
            base_dataset = dataset
        else:
            other_datasets.append(dataset)
    
    return base_dataset, other_datasets