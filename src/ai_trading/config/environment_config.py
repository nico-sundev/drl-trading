from ai_trading.config.base_schema import BaseSchema


class EnvironmentConfig(BaseSchema):
    fee: float
    slippage_atr_based: float  # Base ATR multiplier for slippage calculation
    slippage_against_trade_probability: (
        float  # Probability (0-1) that slippage works against the trade
    )
    start_balance: float
    max_daily_drawdown: float
    max_alltime_drawdown: float
    max_percentage_open_position: float
    min_percentage_open_position: float
    in_money_factor: float  # Reward multiplier for profitable trades
    out_of_money_factor: float  # Penalty multiplier for losing trades
