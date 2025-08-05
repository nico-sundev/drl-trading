"""Bootstrap entrypoint for execution service."""
from drl_trading_execution.infrastructure.bootstrap.execution_bootstrap import bootstrap_execution_service


def main() -> None:
    """Main entrypoint for execution service."""
    bootstrap_execution_service()


if __name__ == "__main__":
    main()
