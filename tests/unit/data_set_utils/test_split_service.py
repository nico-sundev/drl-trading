from unittest.mock import MagicMock

import pandas as pd
import pytest

from ai_trading.data_set_utils.split_service import SplitService
from ai_trading.model.split_dataset_container import SplitDataSetContainer


@pytest.fixture
def rl_model_config() -> MagicMock:
    mocked_config = MagicMock()
    mocked_config.train_ratio = 0.7
    mocked_config.val_ratio = 0.2
    mocked_config.test_ratio = 0.1
    return mocked_config


@pytest.fixture
def invalid_rl_model_config() -> MagicMock:
    mocked_config = MagicMock()
    mocked_config.train_ratio = 0.7
    mocked_config.val_ratio = 0.2
    mocked_config.test_ratio = 0.5
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
    # Attempt to create SplitService with invalid config
    # Given
    svc = SplitService(invalid_rl_model_config)

    with pytest.raises(AssertionError):
        svc.split_dataset(mocked_df)
