# --- FTMO Account Configuration ---
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, TypedDict


class AccountSize(Enum):
    """
    Enumeration of standard FTMO account sizes.
    """

    FTMO_10K = 10000
    FTMO_25K = 25000
    FTMO_50K = 50000
    FTMO_100K = 100000
    FTMO_200K = 200000
    FTMO_300K = 300000
    FTMO_400K = 400000


class FTMOAccountRules(TypedDict, total=False):
    """TypedDict for FTMO account rules to ensure type safety."""

    account_size: float
    currency: str
    max_daily_loss_percent: float
    max_total_loss_percent: float
    min_trading_days: int
    profit_target_percent: float
    max_leverage: float
    timezone: str
    max_position_size: Optional[float]
    require_stop_loss: bool
    enforce_weekend_holding: bool
    max_open_trades: Optional[int]
    custom_rules: Dict[str, Any]


@dataclass(frozen=True)
class FTMOAccountConfig:
    """
    Configuration for FTMO account rules validation.

    This class defines all the relevant FTMO trading rules that need to be
    validated for compliance. The default values represent the standard
    FTMO Challenge/Verification parameters.

    Attributes:
        account_size: Size of the FTMO account in USD/EUR/GBP.
        currency: Account base currency: USD, EUR, or GBP.
        max_daily_loss_percent: Maximum daily loss allowed, as percentage of account size.
        max_total_loss_percent: Maximum total loss (drawdown) allowed, as percentage of account size.
        min_trading_days: Minimum required trading days for the challenge/verification phase.
        profit_target_percent: Profit target as percentage of account size.
        max_leverage: Maximum allowed leverage.
        timezone: Timezone used for daily boundary calculations (e.g., "UTC").
        max_position_size: Maximum position size in lots/contracts.
        require_stop_loss: Whether every trade must have a stop loss.
        enforce_weekend_holding: Whether to allow holding positions over weekends.
        max_open_trades: Maximum number of open positions allowed simultaneously.
        custom_rules: Additional custom rules as key-value pairs.
    """

    account_size: float = 100000.0
    currency: str = "USD"
    max_daily_loss_percent: float = 5.0
    max_total_loss_percent: float = 10.0
    min_trading_days: int = 10
    profit_target_percent: float = 10.0
    max_leverage: float = 30.0
    timezone: str = "UTC"
    max_position_size: Optional[float] = None
    require_stop_loss: bool = True
    enforce_weekend_holding: bool = True
    max_open_trades: Optional[int] = None
    custom_rules: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_account_size(cls, size: AccountSize, **kwargs: Any) -> "FTMOAccountConfig":
        """
        Create an FTMOAccountConfig with predefined rules based on account size.

        Args:
            size: The account size enum value
            **kwargs: Optional overrides for specific configuration values

        Returns:
            FTMOAccountConfig with rules appropriate for the specified account size
        """
        account_size = float(size.value)

        # Create rules with proper typing
        rules: FTMOAccountRules = {
            "account_size": account_size,
            "max_daily_loss_percent": 5.0,  # Fixed at 5% for all account sizes
            "max_total_loss_percent": 10.0,  # Fixed at 10% for all account sizes
        }

        # Add rules that vary by account size
        if account_size <= 100000:
            rules["profit_target_percent"] = 10.0
        else:
            rules["profit_target_percent"] = 8.0

        # Scale position size with account size (10 lots per 100K)
        rules["max_position_size"] = account_size / 100000 * 10

        # Scale max open trades with account size, between 5-10
        rules["max_open_trades"] = min(10, max(5, int(account_size / 20000)))

        # Create the config instance with unpacked rules
        return cls(**rules)
