import random

import numpy as np
import pytest

from ai_trading.gyms.utils.trading_env_utils import TradingDirection, TradingEnvUtils


class TestTradingEnvUtils:
    # Calculate Number Contracts Tests
    def test_calculate_number_contracts_long(self) -> None:
        # Given
        direction = TradingDirection.LONG
        open_position_percentage = 50.0
        current_price = 100.0
        balance = 1000.0
        fee = 0.01
        slippage = 0.001
        leverage = 2.0

        # When
        result = TradingEnvUtils.calculate_number_contracts(
            direction,
            open_position_percentage,
            current_price,
            balance,
            fee,
            slippage,
            leverage,
        )

        # Then
        expected = ((1000.0 * 0.5 * 2.0) * (1.0 - 0.011)) / 100.0
        assert np.isclose(result, expected)
        assert result > 0  # Should be positive for long positions

    def test_calculate_number_contracts_short(self) -> None:
        # Given
        direction = TradingDirection.SHORT
        open_position_percentage = 50.0
        current_price = 100.0
        balance = 1000.0
        fee = 0.01
        slippage = 0.001
        leverage = 2.0

        # When
        result = TradingEnvUtils.calculate_number_contracts(
            direction,
            open_position_percentage,
            current_price,
            balance,
            fee,
            slippage,
            leverage,
        )

        # Then
        expected = -((1000.0 * 0.5 * 2.0) * (1.0 - 0.011)) / 100.0
        assert np.isclose(result, expected)
        assert result < 0  # Should be negative for short positions

    # Liquidation Price Tests
    def test_calculate_liquidation_price_long(self) -> None:
        # Given
        position_open_price = 100.0
        position_size = 1.0
        leverage = 5.0

        # When
        result = TradingEnvUtils.calculate_liquidation_price(
            position_open_price, position_size, leverage
        )

        # Then
        assert (
            result < position_open_price
        )  # Liquidation price should be lower for longs
        assert result > 0  # Should always be positive

    def test_calculate_liquidation_price_short(self) -> None:
        # Given
        position_open_price = 100.0
        position_size = -1.0
        leverage = 5.0

        # When
        result = TradingEnvUtils.calculate_liquidation_price(
            position_open_price, position_size, leverage
        )

        # Then
        assert (
            result > position_open_price
        )  # Liquidation price should be higher for shorts
        assert result > 0  # Should always be positive

    def test_calculate_liquidation_price_invalid_leverage(self) -> None:
        # Given
        position_open_price = 100.0
        position_size = 1.0
        leverage = 1.0

        # When/Then
        with pytest.raises(ValueError, match="Leverage must be greater than 1"):
            TradingEnvUtils.calculate_liquidation_price(
                position_open_price, position_size, leverage
            )

    def test_calculate_liquidation_price_no_position(self) -> None:
        # Given
        position_open_price = 100.0
        position_size = 0.0
        leverage = 5.0

        # When/Then
        with pytest.raises(ValueError, match="No active position"):
            TradingEnvUtils.calculate_liquidation_price(
                position_open_price, position_size, leverage
            )

    # Close Fee Tests
    def test_calculate_close_fee_full_close(self) -> None:
        # Given
        current_price = 100.0
        position_size = 2.0
        fee = 0.01
        slippage = 0.001

        # When
        result = TradingEnvUtils.calculate_close_fee(
            current_price, position_size, fee, slippage
        )

        # Then
        expected = abs(100.0 * 2.0) * (0.01 + 0.001)
        assert np.isclose(result, expected)

    def test_calculate_close_fee_partial_close(self) -> None:
        # Given
        current_price = 100.0
        position_size = 2.0
        fee = 0.01
        slippage = 0.001
        percentage = 0.5

        # When
        result = TradingEnvUtils.calculate_close_fee(
            current_price, position_size, fee, slippage, percentage
        )

        # Then
        expected = abs(100.0 * 2.0 * 0.5) * (0.01 + 0.001)
        assert np.isclose(result, expected)

    # PNL Tests
    def test_calculate_pnl_profitable_long(self) -> None:
        # Given
        current_price = 110.0
        position_open_price = 100.0
        position_size = 2.0
        fee = 0.01
        slippage = 0.001

        # When
        result = TradingEnvUtils.calculate_pnl(
            current_price, position_open_price, position_size, fee, slippage
        )

        # Then
        raw_pnl = (110.0 - 100.0) * 2.0
        fees = abs(110.0 * 2.0) * (0.01 + 0.001)
        expected = raw_pnl - fees
        assert np.isclose(result, expected)
        assert result > 0  # Should be profitable

    def test_calculate_pnl_losing_short(self) -> None:
        # Given
        current_price = 110.0
        position_open_price = 100.0
        position_size = -2.0
        fee = 0.01
        slippage = 0.001

        # When
        result = TradingEnvUtils.calculate_pnl(
            current_price, position_open_price, position_size, fee, slippage
        )

        # Then
        raw_pnl = (110.0 - 100.0) * -2.0
        fees = abs(110.0 * -2.0) * (0.01 + 0.001)
        expected = raw_pnl - fees
        assert np.isclose(result, expected)
        assert result < 0  # Should be unprofitable

    # Position Direction Tests
    def test_get_position_direction_long(self) -> None:
        # Given
        position_state = 1

        # When
        result = TradingEnvUtils.get_position_direction(position_state)

        # Then
        assert result == TradingDirection.LONG

    def test_get_position_direction_short(self) -> None:
        # Given
        position_state = -1

        # When
        result = TradingEnvUtils.get_position_direction(position_state)

        # Then
        assert result == TradingDirection.SHORT

    def test_get_position_direction_none(self) -> None:
        # Given
        position_state = 0

        # When
        result = TradingEnvUtils.get_position_direction(position_state)

        # Then
        assert result == TradingDirection.NONE

    # Position State Tests
    def test_has_active_position(self) -> None:
        # Given
        position_states = [-1, 0, 1]

        # When/Then
        assert TradingEnvUtils.has_active_position(-1) is True
        assert TradingEnvUtils.has_active_position(0) is False
        assert TradingEnvUtils.has_active_position(1) is True

    # Dynamic Slippage Tests
    def test_calculate_dynamic_slippage_returns_correct_magnitude(self) -> None:
        # Given
        direction = TradingDirection.LONG
        current_price = 100.0
        atr = 2.0
        slippage_atr_based = 0.01
        slippage_against_trade_probability = 0.5
        # Set seed for consistent test
        random.seed(42)

        # When
        result = TradingEnvUtils.calculate_dynamic_slippage(
            direction,
            current_price,
            atr,
            slippage_atr_based,
            slippage_against_trade_probability,
        )

        # Then
        # Expected slippage magnitude = (atr / current_price) * slippage_atr_based = (2 / 100) * 0.01 = 0.0002
        assert abs(result) == pytest.approx(0.0002, abs=1e-6)

    def test_calculate_dynamic_slippage_direction_based_on_probability(self) -> None:
        # Given
        direction = TradingDirection.LONG
        current_price = 100.0
        atr = 2.0
        slippage_atr_based = 0.01

        # Test with 100% against-trade probability
        slippage_against_trade_probability = 1.0

        # When
        result = TradingEnvUtils.calculate_dynamic_slippage(
            direction,
            current_price,
            atr,
            slippage_atr_based,
            slippage_against_trade_probability,
        )

        # Then
        # For LONG with 100% against-trade, slippage should be positive (price goes up before entry)
        assert result > 0

        # Given
        # Test with 0% against-trade probability
        slippage_against_trade_probability = 0.0

        # When
        result = TradingEnvUtils.calculate_dynamic_slippage(
            direction,
            current_price,
            atr,
            slippage_atr_based,
            slippage_against_trade_probability,
        )

        # Then
        # For LONG with 0% against-trade, slippage should be negative (price goes down before entry)
        assert result < 0

    def test_calculate_dynamic_slippage_opposite_for_short_positions(self) -> None:
        # Given
        direction = TradingDirection.SHORT
        current_price = 100.0
        atr = 2.0
        slippage_atr_based = 0.01
        slippage_against_trade_probability = 1.0

        # When
        result = TradingEnvUtils.calculate_dynamic_slippage(
            direction,
            current_price,
            atr,
            slippage_atr_based,
            slippage_against_trade_probability,
        )

        # Then
        # For SHORT with 100% against-trade, slippage should be negative (price goes down before entry)
        assert result < 0

    # Price Standard Deviation Tests
    def test_calculate_price_std_with_known_values(self) -> None:
        # Given
        prices = [100.0, 102.0, 101.0, 104.0, 103.0]

        # When
        result = TradingEnvUtils.calculate_price_std(prices)

        # Then
        # Calculate expected standard deviation of returns manually
        expected_returns = [
            (102.0 - 100.0) / 100.0,
            (101.0 - 102.0) / 102.0,
            (104.0 - 101.0) / 101.0,
            (103.0 - 104.0) / 104.0,
        ]
        expected_std = np.std(expected_returns)
        assert result == pytest.approx(expected_std)

    def test_calculate_price_std_with_single_price(self) -> None:
        # Given
        prices = [100.0]

        # When
        result = TradingEnvUtils.calculate_price_std(prices)

        # Then
        assert result == 0.0

    def test_calculate_price_std_with_empty_list(self) -> None:
        # Given
        prices = []

        # When
        result = TradingEnvUtils.calculate_price_std(prices)

        # Then
        assert result == 0.0

    # Risk-Adjusted PNL Tests
    def test_calculate_risk_adjusted_pnl_with_low_volatility(self) -> None:
        # Given
        raw_pnl = 100.0
        price_std = 0.01
        leverage = 1.0

        # When
        result = TradingEnvUtils.calculate_risk_adjusted_pnl(
            raw_pnl, price_std, leverage
        )

        # Then
        # With low volatility, risk-adjusted PNL should be high
        assert result > raw_pnl

    def test_calculate_risk_adjusted_pnl_with_high_volatility(self) -> None:
        # Given
        raw_pnl = 100.0
        price_std = 1.1
        leverage = 1.0

        # When
        result = TradingEnvUtils.calculate_risk_adjusted_pnl(
            raw_pnl, price_std, leverage
        )

        # Then
        # With high volatility, risk-adjusted PNL should be lower
        assert result < raw_pnl

    def test_calculate_risk_adjusted_pnl_with_leverage(self) -> None:
        # Given
        raw_pnl = 100.0
        price_std = 0.05
        leverage_1 = 1.0
        leverage_5 = 5.0

        # When
        result_1 = TradingEnvUtils.calculate_risk_adjusted_pnl(
            raw_pnl, price_std, leverage_1
        )
        result_5 = TradingEnvUtils.calculate_risk_adjusted_pnl(
            raw_pnl, price_std, leverage_5
        )

        # Then
        # Higher leverage should result in lower risk-adjusted PNL
        assert result_1 > result_5

    def test_calculate_risk_adjusted_pnl_with_atr_normalization(self) -> None:
        # Given
        raw_pnl = 100.0
        price_std = 0.05
        leverage = 1.0
        atr = 2.0

        # When
        result_without_atr = TradingEnvUtils.calculate_risk_adjusted_pnl(
            raw_pnl, price_std, leverage
        )
        result_with_atr = TradingEnvUtils.calculate_risk_adjusted_pnl(
            raw_pnl, price_std, leverage, atr
        )

        # Then
        # With ATR normalization, risk-adjusted PNL should be different
        assert result_without_atr != result_with_atr
        assert result_with_atr == pytest.approx(result_without_atr / atr)
