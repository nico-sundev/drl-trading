import logging
from abc import ABC, abstractmethod
from typing import Any, Tuple

import pandas as pd
from injector import inject
from confluent_kafka import Producer

from drl_trading_common.logging.trading_log_context import TradingLogContext
from drl_trading_common.model.trading_context import TradingContext
from drl_trading_common.model.trading_event_payload import TradingEventPayload
from drl_trading_ingest.core.port.market_data_repo_interface import (
    TimescaleRepoInterface,
)

TOPIC_BATCH = "ready.rawdata.batch"

# ServiceLogger is configured at bootstrap level - individual classes use standard loggers
logger = logging.getLogger("drl-trading-ingest")

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
        """
        Store timeseries data to the database.

        Args:
            data: Dictionary containing filename, symbol, timeframe

        Returns:
            Tuple of (response_dict, status_code)
        """
        filename = data.get("filename")
        symbol = data.get("symbol")
        timeframe = data.get("timeframe")

        if not filename or not symbol or not timeframe:
            logger.error("Invalid batch ingest request", extra={
                'filename': filename,
                'symbol': symbol,
                'timeframe': timeframe,
                'error': 'Missing required fields'
            })
            return {"error": "filename, symbol, and timeframe are required"}, 400

        # Create trading context for this ingestion operation
        trading_context = TradingContext.create_initial_context(
            symbol=symbol,
            timeframe=timeframe,
            filename=filename
        )

        # Set logging context for all operations in this request
        TradingLogContext.from_trading_context(trading_context)

        try:
            logger.info("Starting batch market data ingestion", extra={
                'filename': filename,
                'operation': 'batch_ingest'
            })

            # Read and process the CSV data
            df = pd.read_csv(filename, parse_dates=["timestamp"])

            logger.debug("CSV data loaded", extra={
                'rows': len(df),
                'columns': list(df.columns)
            })

            # Save to database
            self.db_repo.save_market_data(symbol, timeframe, df)

            logger.info("Market data saved to database", extra={
                'rows_saved': len(df)
            })

            # Create trading event payload for Kafka
            event_payload = TradingEventPayload.create_market_data_payload(
                symbol=symbol,
                price=float(df['close'].iloc[-1]) if 'close' in df.columns else 0.0,
                volume=float(df['volume'].iloc[-1]) if 'volume' in df.columns else 0.0,
                timestamp=trading_context.timestamp,
                batch_size=len(df),
                timeframe=timeframe
            )

            # Send Kafka message with structured payload
            self.producer.produce(
                TOPIC_BATCH,
                value=event_payload.json(),
                callback=self._delivery_report
            )

            self.producer.flush()

            logger.info("Batch ingestion completed successfully", extra={
                'kafka_topic': TOPIC_BATCH,
                'records_processed': len(df)
            })

            return {"status": "ok", "records_processed": len(df)}, 200

        except Exception as e:
            logger.error("Batch ingestion failed", extra={
                'error_type': type(e).__name__,
                'error_message': str(e)
            }, exc_info=True)
            return {"error": str(e)}, 500
        finally:
            # Clear logging context when done
            TradingLogContext.clear()

    def _delivery_report(self, err: str, msg: Any) -> None:
        """
        Kafka delivery report callback with structured logging.

        Args:
            err: Error message if delivery failed
            msg: Kafka message object if delivery succeeded
        """
        if err is not None:
            logger.error("Kafka message delivery failed", extra={
                'kafka_error': str(err)
            })
        else:
            logger.debug("Kafka message delivered", extra={
                'topic': msg.topic(),
                'partition': msg.partition(),
                'offset': msg.offset()
            })
