"""Bootstrap entrypoint for execution service."""
from drl_trading_execution.infrastructure.bootstrap.execution_bootstrap import ExecutionBootstrap


def main() -> None:
    """Main entrypoint for execution service."""
    bootstrap = ExecutionBootstrap()
    bootstrap.start()


if __name__ == "__main__":
    main()
