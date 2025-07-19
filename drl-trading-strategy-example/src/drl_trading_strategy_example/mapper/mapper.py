from drl_trading_strategy_example.enum.feature_type_enum import FeatureTypeEnum
from drl_trading_strategy_example.enum.indicator_type_enum import IndicatorTypeEnum


class TypeMapper:

    @staticmethod
    def map_feature_to_indicator(feature_type: FeatureTypeEnum) -> IndicatorTypeEnum:
        """
        Maps a feature type to its corresponding indicator type.

        Args:
            feature_type: The feature type to map.

        Returns:
            The corresponding indicator type.

        Raises:
            ValueError: If the feature type does not have a corresponding indicator type.
        """
        mapping = {
            FeatureTypeEnum.RSI: IndicatorTypeEnum.RSI
            # Add more mappings as needed
        }

        if feature_type not in mapping:
            raise ValueError(f"No corresponding indicator type for feature type {feature_type}")

        return mapping[feature_type]
