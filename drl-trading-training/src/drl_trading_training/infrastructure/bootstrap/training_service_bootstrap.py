"""Bootstrap entrypoint for training service."""
from drl_trading_training.infrastructure.bootstrap.training_bootstrap import bootstrap_training_service


def main() -> None:
    """Main entrypoint for training service."""
    bootstrap_training_service()


if __name__ == "__main__":
    main()
