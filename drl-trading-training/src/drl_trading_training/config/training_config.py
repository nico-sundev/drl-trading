"""Service-specific configuration for training service."""
from typing import Any, Dict, List

from drl_trading_common.base.base_application_config import BaseApplicationConfig
from drl_trading_common.config.context_feature_config import ContextFeatureConfig
from drl_trading_common.config.environment_config import EnvironmentConfig
from drl_trading_common.config.feature_config import FeaturesConfig

from drl_trading_training.config.experiment_tracking_config import (
    ExperimentTrackingConfig,
)


class TrainingConfig(BaseApplicationConfig):
    """Configuration for training service, including RL model and dataset settings."""
    # Core configuration categories from common
    features_config: FeaturesConfig
    environment_config: EnvironmentConfig
    context_feature_config: ContextFeatureConfig

    # Training-specific configuration
    experiment_tracking: ExperimentTrackingConfig

    # Dataset configuration
    datasets: Dict[str, Any] = {
        "input_path": "data/raw",
        "split_ratios": {
            "train": 0.8,
            "validation": 0.1,
            "test": 0.1
        },
        "symbols": ["EURUSD"],
        "timeframes": ["H1", "H4"],
        "base_timeframe": "H1",
        "limit": 10000
    }

    # Agents to train
    agents: List[str] = ["PPO"]

    # Agent comparison and ensemble configuration
    ensemble_config: Dict[str, Any] = {
        "enabled": False,
        "voting_scheme": "majority",  # majority, weighted, mean
        "threshold": 0.1
    }

    # Output configuration
    output_config: Dict[str, Any] = {
        "model_save_path": "experiments/models",
        "results_save_path": "experiments/results",
        "save_format": "pickle",  # pickle, joblib, onnx
        "create_visualization": True,
        "publish_metrics": False
    }
