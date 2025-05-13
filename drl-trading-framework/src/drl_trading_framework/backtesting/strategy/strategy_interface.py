from abc import ABC, abstractmethod
from typing import Any, Dict, List

import pandas as pd

from drl_trading_framework.backtesting.strategy.trade import Trade


# --- Strategy Interface ---
class StrategyInterface(ABC):
    @abstractmethod
    def get_equity_curve(self) -> pd.Series:
        pass

    @abstractmethod
    def get_trade_log(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_trades(self, include_open: bool = False) -> List[Trade]:
        """
        Get a list of trade objects representing the strategy's trading history.

        Args:
            include_open: Whether to include currently open trades in the result.

        Returns:
            List of Trade objects containing trade details.
        """
        pass

    def get_account_metrics(self) -> Dict[str, float]:
        return {}

    def get_config(self) -> Dict[str, Any]:
        return {}
