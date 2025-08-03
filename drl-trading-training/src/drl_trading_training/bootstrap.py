"""Bootstrap entrypoint for training service."""
from drl_trading_training.infrastructure.bootstrap.training_bootstrap import TrainingBootstrap


def main() -> None:
    """Main entrypoint for training service."""
    bootstrap = TrainingBootstrap()
    bootstrap.start()


if __name__ == "__main__":
    main()
