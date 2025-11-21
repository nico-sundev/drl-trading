from drl_trading_core.common.model.feature_service_metadata import FeatureServiceMetadata
from drl_trading_core.common.model.feature_view_metadata import FeatureViewMetadata

def get_feature_service_name(
    request: FeatureServiceMetadata
) -> str:
    """
    Generate a feature service name based on the base service name, symbol, and feature version info.

    Args:
        base_service_name (str): The base name of the feature service.
        symbol (str): The symbol for which the service is being generated.
        feature_version_info (FeatureConfigVersionInfo): The version info of the feature configuration.
        timeframe (Timeframe): The timeframe for which the service is being generated.

    Returns:
        str: The generated feature service name.
    """
    # Sanitize timeframe value to be SQL-safe (replace hyphens with underscores)
    sanitized_timeframe = request.timeframe.value.replace("-", "_")
    return f"{request.feature_service_role.value}_{request.symbol}_{sanitized_timeframe}_{request.feature_version_info.semver}_{request.feature_version_info.hash}"

def get_feature_view_name(
    base_feature_view_name: str,
    request: FeatureViewMetadata
) -> str:
    """Generate a feature view name based on the base feature view name, symbol, and timeframe.

    Args:
        base_feature_view_name (str): The base name of the feature view.
        symbol (str): The symbol for which the view is being generated.
        timeframe (Timeframe): The timeframe for which the view is being generated.

    Returns:
        str: The generated feature view name.
    """
    # Sanitize timeframe value to be SQL-safe (replace hyphens with underscores)
    sanitized_timeframe = request.timeframe.value.replace("-", "_")
    return f"{base_feature_view_name}_{request.symbol}_{sanitized_timeframe}"
