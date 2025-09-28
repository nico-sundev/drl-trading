"""Unit tests for the FeatureValidator class."""

import logging
from typing import Dict, List
from unittest.mock import Mock

import pytest

from drl_trading_common.config.feature_config import FeatureDefinition
from drl_trading_core.core.service.feature_manager import FeatureManager
from drl_trading_preprocess.core.service.feature_validator import FeatureValidator


class TestFeatureValidator:
    """Test cases for FeatureValidator class."""

    @pytest.fixture
    def feature_manager_mock(self) -> Mock:
        """Create mock for FeatureManager."""
        return Mock(spec=FeatureManager)

    @pytest.fixture
    def feature_validator(self, feature_manager_mock: Mock) -> FeatureValidator:
        """Create FeatureValidator instance with mocked dependencies."""
        return FeatureValidator(feature_manager=feature_manager_mock)

    @pytest.fixture
    def sample_feature_definitions(self) -> List[FeatureDefinition]:
        """Create sample feature definitions for testing."""
        feature1 = Mock(spec=FeatureDefinition)
        feature1.name = "rsi_14"
        feature1.feature_class = "RSIFeature"
        feature1.parameters = {"period": 14}

        feature2 = Mock(spec=FeatureDefinition)
        feature2.name = "sma_20"
        feature2.feature_class = "SMAFeature"
        feature2.parameters = {"period": 20}

        feature3 = Mock(spec=FeatureDefinition)
        feature3.name = "bollinger_bands"
        feature3.feature_class = "BollingerBandsFeature"
        feature3.parameters = {"period": 20, "std_dev": 2}

        return [feature1, feature2, feature3]

    def test_initialization(self, feature_manager_mock: Mock) -> None:
        """Test FeatureValidator initialization."""
        # Given/When
        validator = FeatureValidator(feature_manager=feature_manager_mock)

        # Then
        assert validator.feature_manager is feature_manager_mock

    def test_validate_definitions_all_supported(
        self,
        feature_validator: FeatureValidator,
        feature_manager_mock: Mock,
        sample_feature_definitions: List[FeatureDefinition],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test validate_definitions when all features are supported."""
        # Given
        expected_validation_results = {
            "rsi_14": True,
            "sma_20": True,
            "bollinger_bands": True,
        }
        feature_manager_mock.validate_feature_definitions.return_value = (
            expected_validation_results
        )

        with caplog.at_level(logging.DEBUG):
            # When
            result = feature_validator.validate_definitions(sample_feature_definitions)

        # Then
        assert result == expected_validation_results
        feature_manager_mock.validate_feature_definitions.assert_called_once_with(
            sample_feature_definitions
        )

        # Check logging
        assert "Validating 3 feature definitions" in caplog.text
        assert (
            "Feature validation completed: 3/3 features supported" in caplog.text
        )

    def test_validate_definitions_some_unsupported(
        self,
        feature_validator: FeatureValidator,
        feature_manager_mock: Mock,
        sample_feature_definitions: List[FeatureDefinition],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test validate_definitions when some features are not supported."""
        # Given
        expected_validation_results = {
            "rsi_14": True,
            "sma_20": False,
            "bollinger_bands": True,
        }
        feature_manager_mock.validate_feature_definitions.return_value = (
            expected_validation_results
        )

        with caplog.at_level(logging.DEBUG):
            # When
            result = feature_validator.validate_definitions(sample_feature_definitions)

        # Then
        assert result == expected_validation_results
        feature_manager_mock.validate_feature_definitions.assert_called_once_with(
            sample_feature_definitions
        )

        # Check logging
        assert "Validating 3 feature definitions" in caplog.text
        assert (
            "Feature validation completed: 2/3 features supported" in caplog.text
        )
        assert "Unsupported features detected: ['sma_20']" in caplog.text

    def test_validate_definitions_none_supported(
        self,
        feature_validator: FeatureValidator,
        feature_manager_mock: Mock,
        sample_feature_definitions: List[FeatureDefinition],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test validate_definitions when no features are supported."""
        # Given
        expected_validation_results = {
            "rsi_14": False,
            "sma_20": False,
            "bollinger_bands": False,
        }
        feature_manager_mock.validate_feature_definitions.return_value = (
            expected_validation_results
        )

        with caplog.at_level(logging.DEBUG):
            # When
            result = feature_validator.validate_definitions(sample_feature_definitions)

        # Then
        assert result == expected_validation_results
        feature_manager_mock.validate_feature_definitions.assert_called_once_with(
            sample_feature_definitions
        )

        # Check logging
        assert "Validating 3 feature definitions" in caplog.text
        assert (
            "Feature validation completed: 0/3 features supported" in caplog.text
        )
        assert (
            "Unsupported features detected: ['rsi_14', 'sma_20', 'bollinger_bands']"
            in caplog.text
        )

    def test_validate_definitions_empty_list(
        self,
        feature_validator: FeatureValidator,
        feature_manager_mock: Mock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test validate_definitions with empty feature list."""
        # Given
        empty_feature_definitions: List[FeatureDefinition] = []
        expected_validation_results: Dict[str, bool] = {}
        feature_manager_mock.validate_feature_definitions.return_value = (
            expected_validation_results
        )

        with caplog.at_level(logging.DEBUG):
            # When
            result = feature_validator.validate_definitions(empty_feature_definitions)

        # Then
        assert result == expected_validation_results
        feature_manager_mock.validate_feature_definitions.assert_called_once_with(
            empty_feature_definitions
        )

        # Check logging
        assert "Validating 0 feature definitions" in caplog.text
        assert "Feature validation completed: 0/0 features supported" in caplog.text

    def test_validate_definitions_single_feature_supported(
        self,
        feature_validator: FeatureValidator,
        feature_manager_mock: Mock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test validate_definitions with single supported feature."""
        # Given
        single_feature = Mock(spec=FeatureDefinition)
        single_feature.name = "macd"
        single_feature.feature_class = "MACDFeature"
        single_feature.parameters = {"fast_period": 12, "slow_period": 26}

        expected_validation_results = {"macd": True}
        feature_manager_mock.validate_feature_definitions.return_value = (
            expected_validation_results
        )

        with caplog.at_level(logging.DEBUG):
            # When
            result = feature_validator.validate_definitions([single_feature])

        # Then
        assert result == expected_validation_results
        feature_manager_mock.validate_feature_definitions.assert_called_once_with(
            [single_feature]
        )

        # Check logging
        assert "Validating 1 feature definitions" in caplog.text
        assert "Feature validation completed: 1/1 features supported" in caplog.text

    def test_validate_definitions_single_feature_unsupported(
        self,
        feature_validator: FeatureValidator,
        feature_manager_mock: Mock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test validate_definitions with single unsupported feature."""
        # Given
        single_feature = Mock(spec=FeatureDefinition)
        single_feature.name = "custom_indicator"
        single_feature.feature_class = "CustomIndicatorFeature"
        single_feature.parameters = {"threshold": 0.5}

        expected_validation_results = {"custom_indicator": False}
        feature_manager_mock.validate_feature_definitions.return_value = (
            expected_validation_results
        )

        with caplog.at_level(logging.DEBUG):
            # When
            result = feature_validator.validate_definitions([single_feature])

        # Then
        assert result == expected_validation_results
        feature_manager_mock.validate_feature_definitions.assert_called_once_with(
            [single_feature]
        )

        # Check logging
        assert "Validating 1 feature definitions" in caplog.text
        assert "Feature validation completed: 0/1 features supported" in caplog.text
        assert (
            "Unsupported features detected: ['custom_indicator']" in caplog.text
        )

    @pytest.mark.parametrize(
        "validation_results,expected_supported_count,expected_unsupported",
        [
            (
                {"feat1": True, "feat2": True, "feat3": True, "feat4": True},
                4,
                [],
            ),
            (
                {"feat1": True, "feat2": False, "feat3": True, "feat4": False},
                2,
                ["feat2", "feat4"],
            ),
            (
                {"feat1": False, "feat2": False, "feat3": False},
                0,
                ["feat1", "feat2", "feat3"],
            ),
        ],
    )
    def test_validate_definitions_parametrized_scenarios(
        self,
        feature_validator: FeatureValidator,
        feature_manager_mock: Mock,
        validation_results: Dict[str, bool],
        expected_supported_count: int,
        expected_unsupported: List[str],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test validate_definitions with various parametrized scenarios."""
        # Given
        mock_features = []
        for feature_name in validation_results.keys():
            mock_feature = Mock(spec=FeatureDefinition)
            mock_feature.name = feature_name
            mock_features.append(mock_feature)

        feature_manager_mock.validate_feature_definitions.return_value = (
            validation_results
        )

        with caplog.at_level(logging.DEBUG):
            # When
            result = feature_validator.validate_definitions(mock_features)  # type: ignore[arg-type]

        # Then
        assert result == validation_results
        feature_manager_mock.validate_feature_definitions.assert_called_once_with(
            mock_features
        )

        # Check logging
        total_features = len(validation_results)
        assert f"Validating {total_features} feature definitions" in caplog.text
        assert (
            f"Feature validation completed: {expected_supported_count}/{total_features} features supported"
            in caplog.text
        )

        if expected_unsupported:
            assert f"Unsupported features detected: {expected_unsupported}" in caplog.text
