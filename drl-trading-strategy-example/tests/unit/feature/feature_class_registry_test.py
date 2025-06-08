"""
Unit tests for feature_class_registry._extract_key_from_class method
with the feature_type decorator.
"""

from unittest.mock import MagicMock

import pytest
from drl_trading_common.base.base_feature import BaseFeature
from drl_trading_common.interfaces.indicator.technical_indicator_facade_interface import (
    TechnicalIndicatorFacadeInterface,
)
from drl_trading_strategy.decorators import feature_type
from drl_trading_strategy.enum.feature_type_enum import FeatureTypeEnum
from drl_trading_strategy.feature.feature_class_registry import FeatureClassRegistry
from pandas import DataFrame


class TestFeatureClassRegistryExtractKey:
    """Test cases for _extract_key_from_class method with decorator support."""

    @pytest.fixture
    def registry(self):
        """Create a FeatureClassRegistry instance for testing."""
        return FeatureClassRegistry()

    @pytest.fixture
    def mock_indicator_service(self):
        """Create a mock indicator service for testing."""
        return MagicMock(spec=TechnicalIndicatorFacadeInterface)

    def test_extract_key_from_decorated_class(self, registry):
        """Test that _extract_key_from_class works with @feature_type decorated classes."""

        @feature_type(FeatureTypeEnum.RSI)
        class TestDecoratedFeature(BaseFeature):
            def compute(self) -> DataFrame:
                return DataFrame()

            def get_sub_features_names(self) -> list[str]:
                return ["test_decorated"]

            def get_feature_name(self) -> str:
                return "test_decorated"

        # Test the extraction
        result = registry._extract_key_from_class(TestDecoratedFeature)

        assert result == FeatureTypeEnum.RSI
        assert isinstance(result, FeatureTypeEnum)

    def test_extract_key_from_traditional_class(self, registry):
        """Test that _extract_key_from_class still works with traditional static method classes."""

        class TestTraditionalFeature(BaseFeature):
            def compute(self) -> DataFrame:
                return DataFrame()

            def get_sub_features_names(self) -> list[str]:
                return ["test_traditional"]

            def get_feature_name(self) -> str:
                return "test_traditional"

            @staticmethod
            def get_feature_type() -> FeatureTypeEnum:
                return FeatureTypeEnum.RSI

        # Test the extraction
        result = registry._extract_key_from_class(TestTraditionalFeature)

        assert result == FeatureTypeEnum.RSI
        assert isinstance(result, FeatureTypeEnum)

    def test_decorator_vs_traditional_consistency(self, registry):
        """Test that decorated and traditional approaches return the same result."""

        @feature_type(FeatureTypeEnum.RSI)
        class DecoratedFeature(BaseFeature):
            def compute(self) -> DataFrame:
                return DataFrame()

            def get_sub_features_names(self) -> list[str]:
                return ["decorated"]

            def get_feature_name(self) -> str:
                return "decorated"

        class TraditionalFeature(BaseFeature):
            def compute(self) -> DataFrame:
                return DataFrame()

            def get_sub_features_names(self) -> list[str]:
                return ["traditional"]

            def get_feature_name(self) -> str:
                return "traditional"

            @staticmethod
            def get_feature_type() -> FeatureTypeEnum:
                return FeatureTypeEnum.RSI

        decorated_result = registry._extract_key_from_class(DecoratedFeature)
        traditional_result = registry._extract_key_from_class(TraditionalFeature)

        assert decorated_result == traditional_result
        assert decorated_result == FeatureTypeEnum.RSI

    def test_decorated_class_has_expected_attributes(self):
        """Test that the decorator properly sets class attributes."""

        @feature_type(FeatureTypeEnum.RSI)
        class TestFeature(BaseFeature):
            def compute(self) -> DataFrame:
                return DataFrame()

            def get_sub_features_names(self) -> list[str]:
                return ["test"]

            def get_feature_name(self) -> str:
                return "test"

        # Test that the decorator added the expected attributes/methods
        assert hasattr(TestFeature, '_feature_type')
        assert TestFeature._feature_type == FeatureTypeEnum.RSI

        assert hasattr(TestFeature, 'get_feature_type')
        assert callable(TestFeature.get_feature_type)
        assert TestFeature.get_feature_type() == FeatureTypeEnum.RSI

    def test_extract_key_fallback_to_class_name(self, registry):
        """Test that _extract_key_from_class falls back to class name extraction when no method exists."""

        class RsiFeature(BaseFeature):  # Note: class name should be parsed as "rsi"
            def compute(self) -> DataFrame:
                return DataFrame()

            def get_sub_features_names(self) -> list[str]:
                return ["fallback"]

            def get_feature_name(self) -> str:
                return "fallback"

            # No get_feature_type method - should fall back to name parsing

        result = registry._extract_key_from_class(RsiFeature)

        # Should extract "rsi" from "RsiFeature" class name
        assert result == FeatureTypeEnum.RSI

    def test_extract_key_handles_get_feature_type_exceptions(self, registry, caplog):
        """Test that _extract_key_from_class handles exceptions in get_feature_type gracefully."""

        class BrokenFeature(BaseFeature):
            def compute(self) -> DataFrame:
                return DataFrame()

            def get_sub_features_names(self) -> list[str]:
                return ["broken"]

            def get_feature_name(self) -> str:
                return "broken"

            @staticmethod
            def get_feature_type() -> FeatureTypeEnum:
                raise RuntimeError("Simulated error")

        # Should fall back to class name parsing and log a warning
        result = registry._extract_key_from_class(BrokenFeature)

        # Should extract "broken" from "BrokenFeature" but since it's not in enum,
        # this will raise ValueError from FeatureTypeConverter
        with pytest.raises(ValueError):
            registry._extract_key_from_class(BrokenFeature)
