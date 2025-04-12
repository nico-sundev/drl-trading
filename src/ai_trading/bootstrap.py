# Create the environment and train the agents
import os
from ai_trading.config.config_loader import ConfigLoader
from ai_trading.data_import.data_import_manager import DataImportManager
from ai_trading.data_import.local.csv_data_import_service import CsvDataImportService
from ai_trading.model.split_dataset_container import SplitDataSetContainer
from ai_trading.data_set_utils.split_service import SplitService
from ai_trading.preprocess.feature.feature_class_registry import FeatureClassRegistry
from ai_trading.preprocess.preprocess_service import PreprocessService
from ai_trading.trading_env import StockTradingEnv
from ai_trading.train_and_test import (
    create_env_and_train_agents,
    test_and_visualize_agents,
)
from stable_baselines3.common.vec_env import DummyVecEnv

# Initialize the config
config = ConfigLoader.get_config(
    os.path.join(os.path.dirname(__file__), "../../config/applicationConfig.json")
)

# Load the datasets
csv_import_svc = CsvDataImportService(config.local_data_import_config.datasets)
data_load_manager = DataImportManager(csv_import_svc)
raw_asset_price_datasets = data_load_manager.get_data(
    config.local_data_import_config.limit
)

# Initialize the feature class registry
feature_class_registry = FeatureClassRegistry()

# Preprocess the asset price datasets
preprocess_svc = PreprocessService(
    datasets=raw_asset_price_datasets,
    features_config=config.features_config,
    feature_class_registry=None,
)
preprocessed_data = preprocess_svc.preprocess_data()

# Split the preprocessed data into training, validation, and test sets
data_set_prep_svc = SplitService(config.features_config)
data_sets: SplitDataSetContainer = data_set_prep_svc.split_dataset(
    df=preprocessed_data,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
)

# Create the environment and train the agents
(
    train_env,
    val_env,
    agents,
) = create_env_and_train_agents(
    data_sets.training_data,
    data_sets.validation_data,
    config.rl_model_config.total_timesteps,
    config.rl_model_config.agent_threshold,
    config.rl_model_config.agents,
)

n_tests = 1000

test_and_visualize_agents(train_env, agents, data_sets.training_data, n_tests=n_tests)

test_env = DummyVecEnv([lambda: StockTradingEnv(data_sets.test_data)])
test_and_visualize_agents(test_env, agents, data_sets.test_data, n_tests=n_tests)
