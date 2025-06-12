"""Service-specific configuration for inference service."""
from drl_trading_common.base.base_application_config import BaseApplicationConfig
from drl_trading_common.config.infrastructure_config import InfrastructureConfig


class InferenceConfig(BaseApplicationConfig):
    """Configuration for inference service - only what it needs."""
    infrastructure: InfrastructureConfig

    # Model configuration
    model_config: dict = {
        "model_path": "/app/models/latest",
        "model_format": "onnx",  # or "pickle", "joblib"
        "batch_size": 1,
        "prediction_timeout": 5.0  # seconds
    }

    # Real-time processing configuration
    processing_config: dict = {
        "feature_buffer_size": 1000,
        "prediction_frequency": "1s",  # How often to generate predictions
        "symbols": ["EURUSD"],  # Only symbols this service handles
        "timeframes": ["H1"]    # Only timeframes needed for inference
    }

    # Message bus configuration (only inference routing)
    message_routing: dict = {
        "input_topic": "market_data",
        "output_topic": "trading_signals",
        "error_topic": "inference_errors"
    }
