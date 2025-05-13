from unittest.mock import MagicMock

import pandas as pd
import pytest

from drl_trading_framework.common.model.split_dataset_container import (
    SplitDataSetContainer,
)
from drl_trading_framework.preprocess.data_set_utils.split_service import SplitService


@pytest.fixture
def rl_model_config() -> MagicMock:
    mocked_config = MagicMock()
    mocked_config.training_split_ratio = 0.7
    mocked_config.validating_split_ratio = 0.2
    mocked_config.testing_split_ratio = 0.1
    return mocked_config


@pytest.fixture
def invalid_rl_model_config() -> MagicMock:
    mocked_config = MagicMock()
    mocked_config.training_split_ratio = 0.7
    mocked_config.validating_split_ratio = 0.2
    mocked_config.testing_split_ratio = 0.5
    return mocked_config


@pytest.fixture
def split_service(rl_model_config: MagicMock) -> SplitService:
    return SplitService(rl_model_config)


@pytest.fixture
def mocked_df() -> pd.DataFrame:
    data = {"col1": range(100), "col2": range(100, 200)}
    return pd.DataFrame(data)


def test_split_dataset(split_service: SplitService, mocked_df: pd.DataFrame) -> None:
    # When
    result = split_service.split_dataset(mocked_df)

    # Then
    assert isinstance(result, SplitDataSetContainer)
    assert len(result.training_data) == 70
    assert len(result.validation_data) == 20
    assert len(result.test_data) == 10


def test_invalid_ratios(
    invalid_rl_model_config: MagicMock, mocked_df: pd.DataFrame
) -> None:
    # Given
    svc = SplitService(invalid_rl_model_config)

    # When/Then
    with pytest.raises(AssertionError):
        svc.split_dataset(mocked_df)
