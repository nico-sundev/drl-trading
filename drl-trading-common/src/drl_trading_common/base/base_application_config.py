
from drl_trading_common.base.base_schema import BaseSchema


class BaseApplicationConfig(BaseSchema):
    """Base configuration that all services inherit."""
    app_name: str
    version: str = "1.0.0"

    # Common infrastructure settings that all services need
    stage: str = "local"  # local | cicd | prod
