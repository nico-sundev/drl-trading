from flask import Flask
from injector import Injector

from drl_trading_ingest.infrastructure.ingest_module import IngestModule
from drl_trading_ingest.infrastructure.routes import register_routes


def create_app():
    app = Flask(__name__)

    injector = Injector([IngestModule()])

    register_routes(app, injector)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
