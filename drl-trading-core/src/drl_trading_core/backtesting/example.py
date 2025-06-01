from typing import Any

import pandas as pd

from drl_trading_core.backtesting.backtest_validator import (
    BacktestValidator,
    BacktestValidatorInterface,
)
from drl_trading_core.backtesting.strategy.strategy_interface import (
    StrategyInterface,
)
from drl_trading_core.backtesting.validation.algorithms.config.ftmo_account_config import (
    AccountSize,
    FTMOAccountConfig,
)
from drl_trading_core.backtesting.validation.algorithms.config.monte_carlo_config import (
    MonteCarloConfig,
)
from drl_trading_core.backtesting.validation.algorithms.config.statistical_significance_config import (
    StatisticalSignificanceConfig,
)
from drl_trading_core.backtesting.validation.container.overall_status import (
    OverallStatus,
)


class ExampleStrategy(StrategyInterface):

    def get_equity_curve(self) -> pd.Series:
        return pd.Series([])

    def get_trade_log(self) -> pd.DataFrame:
        return pd.DataFrame([])

    def get_trades(self, include_open: bool = False) -> list:
        return []


my_strategy = ExampleStrategy()


# Configure FTMO account parameters
ftmo_config = FTMOAccountConfig.from_account_size(
    AccountSize.FTMO_100K,
    timezone="Europe/London",
    custom_rules={"news_trading_allowed": False},
)

# Configure statistical validation
monte_carlo_config = MonteCarloConfig(
    num_simulations=5000,
    confidence_level=95.0,
    simulation_method="block_bootstrap",
    block_length=10,
    min_acceptable_win_rate=60.0,
)

statistical_config = StatisticalSignificanceConfig(
    significance_level=0.05,
    tests={"t_test", "bootstrap", "wilcoxon", "white_reality_check"},
    correction_method="holm",
    null_hypothesis="zero_mean",
)

validations: list[dict[str, Any]] = [
    {"name": "MonteCarloValidation", "config": monte_carlo_config},
    {"name": "StatisticalSignificanceValidation", "config": statistical_config},
    {"name": "FTMOMaxDailyLossValidation", "config": ftmo_config},
    {"name": "FTMOStopLossUsageValidation", "config": ftmo_config},
    {"name": "FTMOMaxDrawdownValidation", "config": ftmo_config},
    {"name": "FTMOTradingDaysValidation", "config": ftmo_config},
    {"name": "FTMOProfitTargetValidation", "config": ftmo_config},
    {"name": "FTMOMaxPositionSizeValidation", "config": ftmo_config},
    {"name": "SharpeRatioValidation", "config": {"min_sharpe": 1.5}},
    {"name": "MaxDrawdownValidation", "config": {"max_drawdown_pct": 7.5}},
]

# Run validation against a strategy
backtester: BacktestValidatorInterface = BacktestValidator()
validation_results = backtester.validate(my_strategy, validations)

# Take action based on results
if validation_results.overall_status == OverallStatus.PASS:
    print("Strategy passes all FTMO compliance checks!")
else:
    for failed in validation_results.failed_algorithms:
        print(f"Failed FTMO check: {failed}")


# Access specific results
monte_carlo_result = validation_results.get_result_by_name("MonteCarloValidation")
significance_result = validation_results.get_result_by_name(
    "StatisticalSignificanceValidation"
)

# The explanation field contains detailed statistical analysis
if monte_carlo_result is not None:
    print(monte_carlo_result.explanation)
else:
    print("MonteCarloValidation result not found.")
