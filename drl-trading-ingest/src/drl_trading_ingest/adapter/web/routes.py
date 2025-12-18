from flask import Flask
from injector import Injector

from drl_trading_ingest.adapter.rest.preprocessing_controller import (
    PreprocessingControllerInterface,
)


def register_routes(app: Flask, injector: Injector) -> None:
    app.add_url_rule(
        "/api/v1/preprocessing/requests",
        view_func=injector.get(PreprocessingControllerInterface).as_view("submit_preprocessing_request"),
        methods=["POST"]
    )
