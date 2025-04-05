from ai_trading.config.base_schema import BaseSchema

class EnvironmentConfig(BaseSchema):
    fee: float
    slippageAtrBased: float 