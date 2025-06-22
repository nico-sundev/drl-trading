from abc import ABC, abstractmethod
from typing import Tuple

import pandas as pd
from injector import inject
from kafka import KafkaProducer

from drl_trading_ingest.core.port.timescale_repo_interface import (
    TimescaleRepoInterface,
)

TOPIC_BATCH = "ready.rawdata.batch"

class IngestionServiceInterface(ABC):
    """
    Interface for the ingestion service.
    """
    @abstractmethod
    def batch_ingest(self, data) -> Tuple[dict, int]:
        """
        Store timeseries data to the database.
        """
        pass

@inject
class IngestionService(IngestionServiceInterface):

    def __init__(self, db_repo: TimescaleRepoInterface, producer: KafkaProducer):
        self.db_repo = db_repo
        self.producer = producer

    def batch_ingest(self, data) -> Tuple[dict, int]:
        filename = data.get("filename")
        symbol = data.get("symbol")

        if not filename or not symbol:
            return {"error": "filename and symbol are required"}, 400

        try:
            df = pd.read_csv(filename, parse_dates=["timestamp"])
            self.db_repo.store_timeseries_to_db(symbol, df)
            self.producer.send(TOPIC_BATCH, b"")
            self.producer.flush()
            return {"status": "ok"}, 200
        except Exception as e:
            return {"error": str(e)}, 500
