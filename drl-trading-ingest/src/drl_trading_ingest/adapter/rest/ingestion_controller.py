from abc import ABC, abstractmethod

from flask import jsonify, request
from flask.views import MethodView
from injector import inject

from drl_trading_ingest.core.service.ingestion_service import IngestionService


class IngestionControllerInterface(ABC):
    """
    Interface for the ingestion controller.
    """

    @abstractmethod
    def ingest_batch(self):
        """
        Store timeseries data to the database.
        """
        ...


@inject
class IngestionController(MethodView, IngestionControllerInterface):
    """
    Controller for handling data ingestion requests.
    """

    def __init__(self, ingestion_service: IngestionService):
        self.ingestion_service = ingestion_service

    def ingest_batch(self):
        data = request.get_json()
        return jsonify(self.ingestion_service.batch_ingest(data))
