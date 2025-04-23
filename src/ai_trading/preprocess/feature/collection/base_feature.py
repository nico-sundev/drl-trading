from abc import ABC, abstractmethod

from pandas import DataFrame

from ai_trading.config.base_parameter_set_config import BaseParameterSetConfig


class BaseFeature(ABC):

    def __init__(
        self, source: DataFrame, config: BaseParameterSetConfig, postfix: str = ""
    ) -> None:
        self.df_source = source
        self.config = config
        self.postfix = postfix

    @abstractmethod
    def compute(self) -> DataFrame:
        pass

    @abstractmethod
    def get_sub_features_names(self) -> list[str]:
        """Get the names of the sub-features.
        This method should be implemented by subclasses to return the names of the sub-features.

        Returns:
            list[str]: A list of sub-feature names.
        """
        pass

    def get_feature_name(self) -> str:
        """Extract the feature name from the class name.

        For example, if the class name is 'RsiFeature', this method will return 'Rsi'.

        Returns:
            str: The base feature name without the 'Feature' suffix
        """
        class_name = self.__class__.__name__
        if class_name.endswith("Feature"):
            return class_name[:-7]  # Remove "Feature" suffix
        return class_name  # Return original name if it doesn't end with "Feature"
