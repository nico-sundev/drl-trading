"""MLflow integration configuration for experiment tracking."""
from typing import Any, Dict, List, Optional

from drl_trading_common.base.base_schema import BaseSchema


class MLflowExperimentConfig(BaseSchema):
    """Configuration for MLflow experiment tracking."""
    experiment_name: str
    tracking_uri: Optional[str] = None  # Uses default if not provided
    registry_uri: Optional[str] = None
    artifact_location: Optional[str] = None

    # Tags applied to all runs in this experiment
    default_tags: Dict[str, str] = {}

    # Run naming strategy
    run_name_pattern: str = "{experiment_name}_{timestamp}"

    # Metrics to log (can be customized by strategy)
    metrics_to_log: List[str] = [
        "reward",
        "episode_length",
        "episode_return",
        "cumulative_profit"
    ]

    # Parameters to log (can be customized by strategy)
    params_to_log: List[str] = [
        "learning_rate",
        "batch_size",
        "total_timesteps"
    ]

    # Artifact logging configuration
    log_model_checkpoint: bool = True
    log_model_final: bool = True
    log_environment: bool = True
    log_source_code: bool = True
    log_config: bool = True
    log_metrics: bool = True


class HyperParameterConfig(BaseSchema):
    """Configuration for hyperparameter values that vary per experiment."""
    # RL Training Parameters
    learning_rate: float = 3e-4
    batch_size: int = 64
    buffer_size: int = 10000
    gamma: float = 0.99

    # Training process configuration
    total_timesteps: int = 10000
    eval_frequency: int = 1000

    # Architecture
    policy_type: str = "MlpPolicy"
    hidden_size: List[int] = [64, 64]

    # Custom parameters (strategy-specific)
    custom: Dict[str, Any] = {}


class ExperimentTrackingConfig(BaseSchema):
    """Complete configuration for experiment tracking."""
    enabled: bool = True
    mlflow: MLflowExperimentConfig
    hyperparameters: HyperParameterConfig = HyperParameterConfig()
