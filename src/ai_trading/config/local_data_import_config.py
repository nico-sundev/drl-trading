from ai_trading.config.base_schema import BaseSchema

class LocalDataImportConfig(BaseSchema):
    datasets: dict[str, str]