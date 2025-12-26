"""Service-specific configuration for training service."""
from pydantic import Field
from drl_trading_common.base.base_application_config import BaseApplicationConfig
from drl_trading_common.base.base_schema import BaseSchema
from drl_trading_common.config.infrastructure_config import InfrastructureConfig
from drl_trading_training.infrastructure.config.experiment_tracking_config import ExperimentTrackingConfig


class DatasetConfig(BaseSchema):
    """Dataset configuration for training."""
    input_path: str = "data/raw"
    train_split: float = 0.8
    validation_split: float = 0.1
    test_split: float = 0.1
    symbols: list[str] = ["EURUSD"]
    timeframes: list[str] = ["H1", "H4"]
    base_timeframe: str = "H1"
    limit: int = 10000


class AgentConfig(BaseSchema):
    """RL agent configuration."""
    algorithms: list[str] = ["PPO"]
    hyperparameters: dict[str, float] = {
        "learning_rate": 0.0003,
        "gamma": 0.99,
        "batch_size": 64,
        "n_epochs": 10
    }


class EnsembleConfig(BaseSchema):
    """Ensemble configuration for multiple agents."""
    enabled: bool = False
    voting_scheme: str = "majority"  # majority | weighted | mean
    threshold: float = 0.1


class OutputConfig(BaseSchema):
    """Output configuration for trained models."""
    model_save_path: str = "experiments/models"
    results_save_path: str = "experiments/results"
    save_format: str = "pickle"  # pickle | joblib | onnx
    create_visualization: bool = True
    publish_metrics: bool = False


class TrainingConfig(BaseApplicationConfig):
    """Configuration for training service - focused on ML model training."""
    app_name: str = "drl-trading-training"
    infrastructure: InfrastructureConfig = Field(default_factory=InfrastructureConfig)
    experiment_tracking: ExperimentTrackingConfig = Field(default_factory=ExperimentTrackingConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    ensemble: EnsembleConfig = Field(default_factory=EnsembleConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
