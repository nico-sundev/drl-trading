"""Bootstrap entrypoint for preprocess service."""
from drl_trading_preprocess.infrastructure.bootstrap.preprocess_bootstrap import bootstrap_preprocess_service


def main() -> None:
    """Main entrypoint for preprocess service."""
    bootstrap_preprocess_service()


if __name__ == "__main__":
    main()
