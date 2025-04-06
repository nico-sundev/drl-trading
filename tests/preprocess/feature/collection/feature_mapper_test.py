from unittest import mock
from ai_trading.preprocess.feature.collection.base_feature import BaseFeature
from ai_trading.preprocess.feature.collection.feature_mapper import discover_feature_classes

# Mocked Feature classes for testing
class MacdFeature(BaseFeature):
    pass

class RsiFeature(BaseFeature):
    pass

# Test the discover_feature_classes function
def test_discover_feature_classes():
    # Mock the parts of the function that interact with the filesystem and modules.
    with mock.patch("pkgutil.iter_modules") as mock_iter_modules, \
         mock.patch("importlib.import_module") as mock_import_module, \
         mock.patch("inspect.getmembers") as mock_getmembers:

        # Mock the package object to simulate the __path__ attribute
        mock_package = mock.Mock()
        mock_package.__path__ = ["path_to_features"]

        # Define what the mocked iter_modules will return
        mock_iter_modules.return_value = [
            ("", "feature_mapper_test.py", False),
        ]

        # Mock the import_module to simulate the module loading
        mock_import_module.return_value = mock_package

        # Define the mock for inspect.getmembers to return mock classes
        mock_getmembers.return_value = [
            ("MacdFeature", MacdFeature),
            ("RsiFeature", RsiFeature),
        ]

        # Call the function we are testing
        feature_map = discover_feature_classes()

        # Validate the results
        assert "macd" in feature_map
        assert "rsi" in feature_map
        assert feature_map["macd"] is MacdFeature
        assert feature_map["rsi"] is RsiFeature
        assert len(feature_map) == 2  # Make sure only 2 features are found

        # Verify that we didn't include BaseFeature
        assert "base" not in feature_map  # BaseFeature shouldn't be included
