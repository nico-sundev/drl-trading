from pandas import DataFrame
import pytest

from ai_trading.preprocess.feature_engineering.feature_engineering_service import FeatureEngineeringService
from .test_config import asset_price_mock_data

@pytest.fixture
def feature_engineering_service():
    return FeatureEngineeringService(asset_price_mock_data(), timeframe_postfix="test")


def test_all_feature(feature_engineering_service: FeatureEngineeringService):
    features = feature_engineering_service.compute_features()

    assert len(features) > 0


def test_rsi_calculation(feature_engineering_service: FeatureEngineeringService):
    rsi = feature_engineering_service.calculate_rsi(14)

    assert len(rsi) > 0
    assert "rsi_14test" in rsi.columns 


def test_roc_calculation(feature_engineering_service: FeatureEngineeringService):
    roc = feature_engineering_service.calculate_roc(3)

    assert len(roc) > 0
    assert "roc_3test" in roc.columns 

def test_macd_signals_calculation(feature_engineering_service: FeatureEngineeringService):
    macd_signals = feature_engineering_service.calculate_macd_signals(3, 9, 4)

    assert len(macd_signals) > 0
    assert "macd_trendtest" in macd_signals.columns 
    assert "macd_cross_bullishtest" in macd_signals.columns
    assert "macd_cross_bearishtest" in macd_signals.columns
