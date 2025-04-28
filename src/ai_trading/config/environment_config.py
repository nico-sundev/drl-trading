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
    max_time_in_trade: (
        int  # Maximum optimal time in position before applying sharp penalty
    )
    optimal_exit_time: int  # Ideal time to exit for bonus reward
    variance_penalty_weight: float  # Penalty weight for high volatility trades
    atr_penalty_weight: float  # Penalty weight for high ATR at entry (risk)
