from abc import ABC, abstractmethod
from typing import Any, Tuple

import pandas as pd
from injector import inject
from confluent_kafka import Producer

from drl_trading_ingest.core.port.market_data_repo_interface import (
    TimescaleRepoInterface,
)

TOPIC_BATCH = "ready.rawdata.batch"

class IngestionServiceInterface(ABC):
    """
    Interface for the ingestion service.
    """
    @abstractmethod
    def batch_ingest(self, data: dict) -> Tuple[dict, int]:
        """
        Store timeseries data to the database.
        """
        pass

@inject
class IngestionService(IngestionServiceInterface):

    def __init__(self, db_repo: TimescaleRepoInterface, producer: Producer):
        self.db_repo = db_repo
        self.producer = producer

    def batch_ingest(self, data: dict) -> Tuple[dict, int]:
        filename = data.get("filename")
        symbol = data.get("symbol")
        timeframe = data.get("timeframe")

        if not filename or not symbol or not timeframe:
            return {"error": "filename, symbol, and timeframe are required"}, 400

        def delivery_report(err: str, msg: Any) -> None:
            if err is not None:
                print(f"Delivery failed for message: {err}")
            else:
                print(f"Message delivered to {msg.topic()} [{msg.partition()}]")

        try:
            df = pd.read_csv(filename, parse_dates=["timestamp"])
            self.db_repo.save_market_data(symbol, timeframe, df)
            # Send message using confluent-kafka API with delivery report
            self.producer.produce(TOPIC_BATCH, value=b"", callback=delivery_report)
            self.producer.flush()
            return {"status": "ok"}, 200
        except Exception as e:
            return {"error": str(e)}, 500
