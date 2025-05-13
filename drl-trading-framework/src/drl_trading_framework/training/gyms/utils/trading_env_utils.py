import random
from enum import Enum
from typing import List, Optional

import numpy as np


class TradingDirection(Enum):
    LONG = 1
    SHORT = -1
    NONE = 0


class TradingEnvUtils:
    @staticmethod
    def calculate_number_contracts(
        direction: TradingDirection,
        open_position_percentage: float,
        current_price: float,
        balance: float,
        fee: float,
        slippage: float = 0.0,
        leverage: float = 1.0,
    ) -> float:
        """Calculate the number of contracts to open based on the open position percentage and current price.

        Args:
            direction (TradingDirection): Direction of the trade (LONG or SHORT)
            open_position_percentage (float): Percentage of the balance to open a new position (0-100)
            current_price (float): Current price of the asset
            balance (float): Current account balance
            fee (float): Trading fee as a percentage (0.01 = 1%)
            slippage (float, optional): Slippage as a percentage (0.01 = 1%). Defaults to 0.0
            leverage (float, optional): Leverage multiplier. Defaults to 1.0

        Returns:
            float: Number of contracts to open
        """
        total_position_value = (balance * open_position_percentage / 100.0) * leverage
        total_position_value_excluding_fee = total_position_value * (
            1.0 - fee - slippage
        )
        return float(
            (total_position_value_excluding_fee / current_price) * direction.value
        )

    @staticmethod
    def calculate_liquidation_price(
        position_open_price: float,
        position_size: float,
        leverage: float = 1.0,
    ) -> float:
        """Calculate the liquidation price for a position based on the open price, leverage, and trade direction.

        Args:
            position_open_price (float): The price at which the position was opened
            position_size (float): The size of the position (positive for long, negative for short)
            leverage (float, optional): The leverage used for the position. Defaults to 1.0

        Returns:
            float: The calculated liquidation price

        Raises:
            ValueError: If leverage <= 1 or if no active position
        """
        if leverage <= 1:
            raise ValueError(
                "Leverage must be greater than 1 to calculate liquidation price."
            )

        if position_size == 0:
            raise ValueError("No active position to calculate liquidation price.")

        # Adjust the formula to account for non-linearity at higher leverage
        adjustment_factor = 1 - (1 / (leverage**1.1))  # Exponential adjustment

        if position_size > 0:  # Long trade
            return float(position_open_price * adjustment_factor)
        else:  # Short trade
            return float(position_open_price / adjustment_factor)

    @staticmethod
    def calculate_close_fee(
        current_price: float,
        position_size: float,
        fee: float,
        slippage: float = 0.0,
        percentage: float = 1.0,
    ) -> float:
        """Calculate the closing fee for a position.

        Args:
            current_price (float): Current price of the asset
            position_size (float): Size of the position
            fee (float): Trading fee as a percentage (0.01 = 1%)
            slippage (float, optional): Slippage as a percentage (0.01 = 1%). Defaults to 0.0
            percentage (float, optional): Percentage of position being closed (0-1). Defaults to 1.0

        Returns:
            float: The calculated closing fee
        """
        return float(abs(current_price * position_size * percentage) * (fee + slippage))

    @staticmethod
    def calculate_pnl(
        current_price: float,
        position_open_price: float,
        position_size: float,
        fee: float,
        slippage: float = 0.0,
    ) -> float:
        """Calculate profit/loss for a position.

        Args:
            current_price (float): Current price of the asset
            position_open_price (float): Price at which position was opened
            position_size (float): Size of the position (positive for long, negative for short)
            fee (float): Trading fee as a percentage (0.01 = 1%)
            slippage (float, optional): Slippage as a percentage (0.01 = 1%). Defaults to 0.0

        Returns:
            float: The calculated profit/loss
        """
        price_difference = current_price - position_open_price
        raw_pnl = price_difference * position_size
        fees = TradingEnvUtils.calculate_close_fee(
            current_price, position_size, fee, slippage
        )
        return float(raw_pnl - fees)

    @staticmethod
    def get_position_direction(position_state: int) -> TradingDirection:
        """Convert position state to TradingDirection enum.

        Args:
            position_state (int): Position state (1 for long, -1 for short, 0 for none)

        Returns:
            TradingDirection: Corresponding trading direction enum value
        """
        if position_state == 1:
            return TradingDirection.LONG
        elif position_state == -1:
            return TradingDirection.SHORT
        return TradingDirection.NONE

    @staticmethod
    def has_active_position(position_state: int) -> bool:
        """Check if there is an active position.

        Args:
            position_state (int): Position state (1 for long, -1 for short, 0 for none)

        Returns:
            bool: True if there is an active position, False otherwise
        """
        return bool(position_state != 0)

    @staticmethod
    def calculate_dynamic_slippage(
        direction: TradingDirection,
        current_price: float,
        atr: float,
        slippage_atr_based: float,
        slippage_against_trade_probability: float,
    ) -> float:
        """Calculate dynamic slippage based on ATR and trade direction.

        This method calculates slippage as a percentage of the price (e.g., 0.01 = 1%).
        The slippage is:
        1. Based on ATR relative to current price to reflect market volatility
        2. Works against the trade with slippage_against_trade_probability (default 60%)
        3. Used both for position entry price adjustment and fee calculations

        Args:
            direction (TradingDirection): Direction of the trade (LONG or SHORT)
            current_price (float): Current price of the asset
            atr (float): Average True Range value at the current step
            slippage_atr_based (float): Base ATR multiplier for slippage calculation
            slippage_against_trade_probability (float): Probability (0-1) that slippage works against the trade

        Returns:
            float: Calculated slippage as a percentage of price
        """
        # Base slippage as a percentage (ATR relative to price)
        base_slippage = (atr / current_price) * slippage_atr_based

        # Determine if slippage works against the trade
        against_trade = random.random() < slippage_against_trade_probability

        # For long positions: negative slippage means price goes up before entry
        # For short positions: positive slippage means price goes down before entry
        if against_trade:
            return float(
                base_slippage if direction == TradingDirection.LONG else -base_slippage
            )
        else:
            return float(
                -base_slippage if direction == TradingDirection.LONG else base_slippage
            )

    @staticmethod
    def calculate_price_std(
        price_history: List[float],
    ) -> float:
        """Calculate the standard deviation of price movement during a period.

        Args:
            price_history (List[float]): List of historical prices

        Returns:
            float: Standard deviation of price returns during the period
        """
        if len(price_history) <= 1:
            return 0.0

        # Convert to numpy array for calculation
        prices = np.asarray(price_history)

        # Calculate returns
        returns = np.diff(prices) / prices[:-1]

        # Calculate standard deviation of returns
        return float(np.std(returns) if len(returns) > 1 else 0.0)

    @staticmethod
    def calculate_risk_adjusted_pnl(
        raw_pnl: float,
        price_std: float,
        leverage: float,
        atr: Optional[float] = None,
    ) -> float:
        """Calculate a Sharpe ratio-like risk-adjusted PnL.

        This method adjusts the raw PnL by the risk taken:
        1. Adjusts PnL by risk (volatility) - higher volatility means higher risk
        2. Incorporates leverage as an additional risk multiplier
        3. Optionally normalizes by ATR to make it volatility-agnostic across securities

        Args:
            raw_pnl (float): Raw profit and loss value
            price_std (float): Standard deviation of price returns
            leverage (float): Current leverage of the position
            atr (Optional[float]): ATR value for normalization across securities

        Returns:
            float: Risk-adjusted PnL
        """
        # Avoid division by zero
        if price_std < 0.0001:
            return raw_pnl

        # Calculate risk-adjusted return (like Sharpe ratio)
        # - For positive PnL: higher volatility = lower reward
        # - For negative PnL: higher volatility = higher penalty
        risk_multiplier = leverage  # Higher leverage = higher risk

        # Normalize by ATR to make it volatility-agnostic across securities if provided
        normalized_pnl = raw_pnl / atr if atr and atr > 0 else raw_pnl
        risk_adjusted_pnl = normalized_pnl / (price_std * risk_multiplier)

        return float(risk_adjusted_pnl)
