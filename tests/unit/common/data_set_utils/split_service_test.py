from unittest.mock import MagicMock

import pandas as pd
import pytest

from drl_trading_framework.common.model.split_dataset_container import (
    SplitDataSetContainer,
)
from drl_trading_framework.preprocess.data_set_utils.split_service import SplitService


@pytest.fixture
def rl_model_config() -> MagicMock:
    """Create a mock RL model config with valid split ratios."""
    mocked_config = MagicMock()
    mocked_config.training_split_ratio = 0.7
    mocked_config.validating_split_ratio = 0.2
    mocked_config.testing_split_ratio = 0.1
    return mocked_config


@pytest.fixture
def invalid_rl_model_config() -> MagicMock:
    """Create a mock RL model config with invalid split ratios."""
    mocked_config = MagicMock()
    mocked_config.training_split_ratio = 0.7
    mocked_config.validating_split_ratio = 0.2
    mocked_config.testing_split_ratio = 0.5
    return mocked_config


@pytest.fixture
def split_service(rl_model_config: MagicMock) -> SplitService:
    """Create a split service with valid config."""
    return SplitService(rl_model_config)


@pytest.fixture
def mocked_df() -> pd.DataFrame:
    """Create a test DataFrame."""
    data = {"col1": range(100), "col2": range(100, 200)}
    return pd.DataFrame(data)


def test_split_dataset_with_config(
    split_service: SplitService, mocked_df: pd.DataFrame
) -> None:
    """Test splitting dataset using ratios from config."""
    # Given
    # Split service initialized with config

    # When
    result = split_service.split_dataset(mocked_df)

    # Then
    assert isinstance(result, SplitDataSetContainer)
    assert len(result.training_data) == 70
    assert len(result.validation_data) == 20
    assert len(result.test_data) == 10


def test_split_dataset_with_explicit_ratios(mocked_df: pd.DataFrame) -> None:
    """Test splitting dataset using explicit ratios."""
    # Given
    svc = SplitService()  # No config

    # When
    result = svc.split_dataset(
        mocked_df, training_ratio=0.6, validation_ratio=0.3, testing_ratio=0.1
    )

    # Then
    assert isinstance(result, SplitDataSetContainer)
    assert len(result.training_data) == 60
    assert len(result.validation_data) == 30
    assert len(result.test_data) == 10


def test_invalid_ratios(
    invalid_rl_model_config: MagicMock, mocked_df: pd.DataFrame
) -> None:
    """Test handling of invalid ratios that don't sum to 1.0."""
    # Given
    svc = SplitService(invalid_rl_model_config)

    # When/Then
    with pytest.raises(ValueError):
        svc.split_dataset(mocked_df)


def test_missing_ratios(mocked_df: pd.DataFrame) -> None:
    """Test handling when no ratios are provided."""
    # Given
    svc = SplitService()  # No config

    # When/Then
    with pytest.raises(ValueError):
        svc.split_dataset(mocked_df)  # No explicit ratios
