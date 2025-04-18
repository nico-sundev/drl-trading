from unittest import mock

import pytest
from pandas import DataFrame

from ai_trading.preprocess.feature.feature_aggregator import FeatureAggregator


@pytest.fixture
def dummy_df():
    return DataFrame({"Time": [1, 2, 3], "price": [100, 101, 102]})


@pytest.fixture
def mock_param_set():
    return mock.Mock()


@pytest.fixture
def mock_feature_definition(mock_param_set):
    feature_def = mock.Mock()
    feature_def.name = "rsi"
    feature_def.parsed_parameter_sets = [mock_param_set]
    return feature_def


@pytest.fixture(autouse=True)
def mock_config(mock_feature_definition):
    config = mock.Mock()
    config.feature_definitions = [mock_feature_definition]
    return config


@pytest.fixture
def mock_feature_df():
    return DataFrame({"Time": [1, 2, 3], "rsi_14": [30, 50, 70]})


@pytest.fixture(autouse=True)
def mock_feature_class_instance(mock_feature_df, mock_param_set):
    instance = mock.Mock()
    instance.compute.return_value = mock_feature_df
    return instance


@pytest.fixture(autouse=True)
def mock_registry(mock_feature_class_instance):
    mock_class = mock.Mock(return_value=mock_feature_class_instance)
    registry = mock.Mock()
    registry.feature_class_map = {"rsi": mock_class}
    return registry


def test_feature_aggregator_compute(
    dummy_df, mock_config, mock_registry, mock_feature_df
):
    aggregator = FeatureAggregator(dummy_df, mock_config, mock_registry)
    result_df = aggregator.compute()

    # Assertions
    assert isinstance(result_df, DataFrame)
    assert "Time" in result_df.columns
    assert "rsi_14" in result_df.columns
    assert len(result_df) == 3
