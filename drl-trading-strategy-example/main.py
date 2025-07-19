from drl_trading_core import bootstrap_agent_training, bootstrap_inference

from src.drl_trading_strategy_example import MyCustomTradingEnv

CONFIG_PATH = (
    "c:/Users/nico-/Documents/git/ai_trading/drl-trading-strategy-example/app_config.json"
)


def run_training():
    print("Starting agent training...")
    bootstrap_agent_training(env_class=MyCustomTradingEnv, config_path=CONFIG_PATH)
    print("Agent training finished.")


def run_inference():
    print("Starting inference...")
    # Note: bootstrap_inference() is not fully implemented in the framework yet
    try:
        bootstrap_inference()  # This will currently raise NotImplementedError
    except NotImplementedError as e:
        print(f"Inference Error: {e}")
    print("Inference finished.")


if __name__ == "__main__":
    # Example: Run training
    run_training()

    # Example: Run inference (will show NotImplementedError)
    # run_inference()
