import pytest
import numpy as np
from ai_trading.utils.trading_env_utils import TradingEnvUtils, TradingDirection


class TestTradingEnvUtils:
    # Calculate Number Contracts Tests
    def test_calculate_number_contracts_long(self):
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
            direction, open_position_percentage, current_price, balance, fee, slippage, leverage
        )

        # Then
        expected = ((1000.0 * 0.5 * 2.0) * (1.0 - 0.011)) / 100.0
        assert np.isclose(result, expected)
        assert result > 0  # Should be positive for long positions

    def test_calculate_number_contracts_short(self):
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
            direction, open_position_percentage, current_price, balance, fee, slippage, leverage
        )

        # Then
        expected = -((1000.0 * 0.5 * 2.0) * (1.0 - 0.011)) / 100.0
        assert np.isclose(result, expected)
        assert result < 0  # Should be negative for short positions

    # Liquidation Price Tests
    def test_calculate_liquidation_price_long(self):
        # Given
        position_open_price = 100.0
        position_size = 1.0
        leverage = 5.0

        # When
        result = TradingEnvUtils.calculate_liquidation_price(
            position_open_price, position_size, leverage
        )

        # Then
        assert result < position_open_price  # Liquidation price should be lower for longs
        assert result > 0  # Should always be positive

    def test_calculate_liquidation_price_short(self):
        # Given
        position_open_price = 100.0
        position_size = -1.0
        leverage = 5.0

        # When
        result = TradingEnvUtils.calculate_liquidation_price(
            position_open_price, position_size, leverage
        )

        # Then
        assert result > position_open_price  # Liquidation price should be higher for shorts
        assert result > 0  # Should always be positive

    def test_calculate_liquidation_price_invalid_leverage(self):
        # Given
        position_open_price = 100.0
        position_size = 1.0
        leverage = 1.0

        # When/Then
        with pytest.raises(ValueError, match="Leverage must be greater than 1"):
            TradingEnvUtils.calculate_liquidation_price(
                position_open_price, position_size, leverage
            )

    def test_calculate_liquidation_price_no_position(self):
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
    def test_calculate_close_fee_full_close(self):
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

    def test_calculate_close_fee_partial_close(self):
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
    def test_calculate_pnl_profitable_long(self):
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

    def test_calculate_pnl_losing_short(self):
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