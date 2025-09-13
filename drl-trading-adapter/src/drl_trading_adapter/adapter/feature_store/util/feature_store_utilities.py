from drl_trading_common.enum.feature_role_enum import FeatureRoleEnum
from drl_trading_common.model.feature_config_version_info import FeatureConfigVersionInfo

def get_feature_service_name(
    feature_service_role: FeatureRoleEnum,
    symbol: str,
    feature_version_info: FeatureConfigVersionInfo,
) -> str:
    """
    Generate a feature service name based on the base service name, symbol, and feature version info.

    Args:
        base_service_name (str): The base name of the feature service.
        symbol (str): The symbol for which the service is being generated.
        feature_version_info (FeatureConfigVersionInfo): The version info of the feature configuration.

    Returns:
        str: The generated feature service name.
    """
    return f"{feature_service_role.value}_{symbol}_{feature_version_info.semver}-{feature_version_info.hash}"
