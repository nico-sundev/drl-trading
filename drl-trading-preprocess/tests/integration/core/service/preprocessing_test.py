import pytest
from injector import Injector
from pandas import DatetimeIndex

from drl_trading_core.common.data_import.data_import_manager import (
    DataImportManager,
)
from drl_trading_core.common.model.preprocessing_result import PreprocessingResult
from drl_trading_core.common.model.symbol_import_container import (
    SymbolImportContainer,
)
from drl_trading_preprocess.core.service.preprocess_service import (
    PreprocessServiceInterface,
)


@pytest.fixture
def symbol_container(mocked_container: Injector) -> SymbolImportContainer:
    # Create a service with the complete config
    importer = mocked_container.get(DataImportManager)

    # Get all symbol containers
    return importer.get_data()[0]


@pytest.fixture
def preprocess_service(mocked_container: Injector) -> PreprocessServiceInterface:
    """Get preprocess service from the mocked container."""
    return mocked_container.get(PreprocessServiceInterface)


def test_preprocessing(
    preprocess_service: PreprocessServiceInterface,
    symbol_container: SymbolImportContainer,
):
    """Test that preprocessing creates the expected feature columns."""
    # Given
    # Time is now in the index, not a column
    expected_context_related_columns = ["High", "Low", "Close", "Atr"]

    expected_feature_columns = [
        "rsi_7",
        "HTF-240_rsi_7",
    ]

    all_expected_columns = sorted(
        expected_context_related_columns + expected_feature_columns
    )

    # When
    preproc_result = preprocess_service.preprocess_data(symbol_container)
    # Should return a PreprocessingResult object
    assert isinstance(
        preproc_result, PreprocessingResult
    ), f"Expected PreprocessingResult, got {type(preproc_result)}"
    df = preproc_result.final_result
    actual_columns = sorted(set(df.columns))

    # Then
    assert (
        actual_columns == all_expected_columns
    ), f"Column mismatch! Expected: {all_expected_columns}, but got: {actual_columns}"

    # Verify that we have a DatetimeIndex on the final result
    assert isinstance(df.index, DatetimeIndex), "Result should have a DatetimeIndex"

    # print(feature_df_merged.head())
