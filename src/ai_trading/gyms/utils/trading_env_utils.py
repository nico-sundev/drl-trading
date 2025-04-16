from enum import Enum
import numpy as np
from typing import Union


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
            open_position_percentage (float): Percentage of the balance to open a new position
            current_price (float): Current price of the asset
            balance (float): Current account balance
            fee (float): Trading fee percentage
            slippage (float, optional): Slippage percentage. Defaults to 0.0
            leverage (float, optional): Leverage multiplier. Defaults to 1.0

        Returns:
            float: Number of contracts to open
        """
        total_position_value = (balance * open_position_percentage / 100.0) * leverage
        total_position_value_excluding_fee = total_position_value * (1.0 - fee - slippage)
        return (total_position_value_excluding_fee / current_price) * direction.value

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
            raise ValueError("Leverage must be greater than 1 to calculate liquidation price.")

        if position_size == 0:
            raise ValueError("No active position to calculate liquidation price.")

        # Adjust the formula to account for non-linearity at higher leverage
        adjustment_factor = 1 - (1 / (leverage**1.1))  # Exponential adjustment

        if position_size > 0:  # Long trade
            return position_open_price * adjustment_factor
        else:  # Short trade
            return position_open_price / adjustment_factor

    @staticmethod
    def calculate_close_fee(
        current_price: float,
        position_size: float,
        fee: float,
        slippage: float = 0.0,
        percentage: float = 1.0
    ) -> float:
        """Calculate the closing fee for a position.

        Args:
            current_price (float): Current price of the asset
            position_size (float): Size of the position
            fee (float): Trading fee percentage
            slippage (float, optional): Slippage percentage. Defaults to 0.0
            percentage (float, optional): Percentage of position being closed. Defaults to 1.0

        Returns:
            float: The calculated closing fee
        """
        return abs(current_price * position_size * percentage) * (fee + slippage)

    @staticmethod
    def calculate_pnl(
        current_price: float,
        position_open_price: float,
        position_size: float,
        fee: float,
        slippage: float = 0.0
    ) -> float:
        """Calculate profit/loss for a position.

        Args:
            current_price (float): Current price of the asset
            position_open_price (float): Price at which position was opened
            position_size (float): Size of the position (positive for long, negative for short)
            fee (float): Trading fee percentage
            slippage (float, optional): Slippage percentage. Defaults to 0.0

        Returns:
            float: The calculated profit/loss
        """
        price_difference = current_price - position_open_price
        raw_pnl = price_difference * position_size
        fees = TradingEnvUtils.calculate_close_fee(current_price, position_size, fee, slippage)
        return raw_pnl - fees