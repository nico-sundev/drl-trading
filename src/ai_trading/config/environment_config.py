from ai_trading.config.base_schema import BaseSchema

class EnvironmentConfig(BaseSchema):
    fee: float
    slippage_atr_based: float
    start_balance: float
    max_daily_drawdown: float
    max_alltime_drawdown: float
    max_percentage_open_position: float
    min_percentage_open_position: float