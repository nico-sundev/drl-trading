from dataclasses import dataclass
from typing import List

from pandas import DataFrame

from drl_trading_core.common.model.computed_dataset_container import (
    ComputedDataSetContainer,
)
from drl_trading_core.common.model.symbol_import_container import (
    SymbolImportContainer,
)


@dataclass
class PreprocessingResult:
    # The input symbol container
    symbol_container: SymbolImportContainer
    # All computed dataset containers including base and others
    computed_dataset_containers: List[ComputedDataSetContainer]
    # The base timeframe computed dataset container
    base_computed_container: ComputedDataSetContainer
    # Other timeframe computed dataset containers
    other_computed_containers: List[ComputedDataSetContainer]
    # Merged result before context features
    merged_result: DataFrame
    # Context-related features DataFrame
    context_features: DataFrame
    # Final merged DataFrame including context features
    final_result: DataFrame
