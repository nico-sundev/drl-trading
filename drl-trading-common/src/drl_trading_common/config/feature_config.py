from typing import Any, Dict, List, Optional

from drl_trading_common.enum.offline_repo_strategy_enum import OfflineRepoStrategyEnum
from drl_trading_common.adapter.model.timeframe import Timeframe
from pydantic import Field

from ..base.base_parameter_set_config import BaseParameterSetConfig
from ..base.base_schema import BaseSchema


class LocalRepoConfig(BaseSchema):
    """Configuration for local filesystem-based offline feature repository."""
    repo_path: str


class S3RepoConfig(BaseSchema):
    """Configuration for S3-based offline feature repository."""
    bucket_name: str = "drl-trading-features"
    prefix: str = "features"
    endpoint_url: Optional[str] = None  # Optional for custom S3-compatible services
    region: str = "us-east-1"
    access_key_id: Optional[str] = None  # Optional, can use AWS credentials chain
    secret_access_key: Optional[str] = None  # Optional, can use AWS credentials chain


class FeatureStoreConfig(BaseSchema):
    cache_enabled: bool
    entity_name: str
    ttl_days: int
    online_enabled: bool = False
    service_name: str
    service_version: str

    # Feast configuration directory (where feature_store.yaml is stored)
    config_directory: str

    # Strategy selection for offline repository
    offline_repo_strategy: OfflineRepoStrategyEnum = OfflineRepoStrategyEnum.LOCAL

    # Repository-specific configurations
    local_repo_config: Optional[LocalRepoConfig] = None
    s3_repo_config: Optional[S3RepoConfig] = None

class FeatureDefinition(BaseSchema):
    """Feature definition configuration.

    Uses string-based names to avoid circular dependencies between common library
    and strategy-specific enums. The strategy layer can provide type-safe conversion
    to enums when needed.

    Features can be configured with parameter sets (for configurable features like RSI)
    or without any parameters (for simple features like close price).
    """
    name: str  # Feature type identifier (e.g., "rsi", "macd", "close_price")
    enabled: bool
    derivatives: List[int]
    parameter_sets: List[Dict[str, Any]] = Field(default_factory=list)  # raw input from JSON, can be empty
    parsed_parameter_sets: Dict[str, BaseParameterSetConfig] = Field(default_factory=dict)

class FeaturesConfig(BaseSchema):
    dataset_definitions: Dict[str, List[Timeframe]]
    feature_definitions: List[FeatureDefinition]
