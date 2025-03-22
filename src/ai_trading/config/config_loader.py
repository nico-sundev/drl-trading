import json
from ai_trading.preprocess.feature.feature_config import FeatureConfig, MACDConfig

class ConfigLoader:
    @staticmethod
    def from_json(json_path: str) -> FeatureConfig:
        """Load config from JSON file."""
        with open(json_path, "r") as file:
            data = json.load(file)
        
        return FeatureConfig(
            macd=MACDConfig(**data["macd"]),
            rsi_lengths=data["rsi_lengths"],
            roc_lengths=data["roc_lengths"]
        )
