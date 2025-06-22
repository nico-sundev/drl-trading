# routes.py
from flask import Flask
from injector import Injector

from drl_trading_ingest.adapter.rest.ingestion_controller import (
    IngestionControllerInterface,
)


def register_routes(app: Flask, injector: Injector):
    app.add_url_rule(
        "/api/ingest/batch",
        view_func=injector.get(IngestionControllerInterface).as_view("ingest_batch")
    )
