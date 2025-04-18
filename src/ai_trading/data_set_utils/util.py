from typing import List

from ai_trading.model.asset_price_dataset import AssetPriceDataSet
from ai_trading.model.computed_dataset_container import ComputedDataSetContainer


def separate_asset_price_datasets(datasets: List[AssetPriceDataSet]) -> tuple:
    """
    Separates the AssetPriceDataSet datasets into base and other datasets.
    """
    base_dataset = None
    other_datasets = []

    for dataset in datasets:
        if dataset.base_dataset:
            base_dataset = dataset
        else:
            other_datasets.append(dataset)

    return base_dataset, other_datasets


def separate_computed_datasets(datasets: List[ComputedDataSetContainer]) -> tuple:
    """
    Separates the ComputedDataSetContainer datasets into base and other datasets.
    """
    base_dataset = None
    other_datasets = []

    for dataset in datasets:
        if dataset.source_dataset.base_dataset:
            base_dataset = dataset
        else:
            other_datasets.append(dataset)

    return base_dataset, other_datasets


def detect_timeframe(df):
    """Auto-detects the timeframe of a dataset."""
    return df["Time"].diff().mode()[0]
