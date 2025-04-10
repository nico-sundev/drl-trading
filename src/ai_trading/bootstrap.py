# Create the environment and train the agents
from ai_trading.model.preprocessed_dataset_container import PreprocessedDataSetContainer
from ai_trading.preprocess.data_set_prep_service import DataSetPrepService
from ai_trading.config.feature_config import FeaturesConfig
from ai_trading.trading_env import StockTradingEnv
from ai_trading.train_and_test import (
    create_env_and_train_agents,
    test_and_visualize_agents,
)
from stable_baselines3.common.vec_env import DummyVecEnv

# -----------------------------------------------------------------------------
feature_config: FeaturesConfig
data_set_prep_svc = DataSetPrepService(feature_config)
data_sets: PreprocessedDataSetContainer = data_set_prep_svc.get_data_sets()
threshold = 0.1
total_timesteps = 10000
(
    train_env,
    val_env,
    ppo_agent,
    a2c_agent,
    ddpg_agent,
    sac_agent,
    td3_agent,
    ensemble_agent,
) = create_env_and_train_agents(
    data_sets.training_data, data_sets.validation_data, total_timesteps, threshold
)

n_tests = 1000
agents = {
    "PPO Agent": ppo_agent,
    "A2C Agent": a2c_agent,
    "DDPG Agent": ddpg_agent,
    "SAC Agent": sac_agent,
    "TD3 Agent": td3_agent,
    "Ensemble Agent": ensemble_agent,
}

test_and_visualize_agents(train_env, agents, data_sets.training_data, n_tests=n_tests)

test_env = DummyVecEnv([lambda: StockTradingEnv(data_sets.test_data)])
test_and_visualize_agents(test_env, agents, data_sets.test_data, n_tests=n_tests)
