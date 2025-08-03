"""Bootstrap entrypoint for preprocess service."""
from drl_trading_preprocess.infrastructure.bootstrap.preprocess_bootstrap import PreprocessBootstrap


def main() -> None:
    """Main entrypoint for preprocess service."""
    bootstrap = PreprocessBootstrap()
    bootstrap.start()


if __name__ == "__main__":
    main()
