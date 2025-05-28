"""RabbitMQ transport for production mode."""

import json
import logging
import threading
import time
import uuid
from typing import Callable, Dict, Optional

import pika

from .transport_interface import Message, TransportInterface

logger = logging.getLogger(__name__)


class RabbitMQTransport(TransportInterface):
    """RabbitMQ transport for production distributed deployment."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5672,
        username: str = "guest",
        password: str = "guest",
        virtual_host: str = "/",
    ):
        self.connection_params = pika.ConnectionParameters(
            host=host,
            port=port,
            virtual_host=virtual_host,
            credentials=pika.PlainCredentials(username, password),
        )
        self.connection: Optional[pika.BlockingConnection] = None
        self.channel: Optional[pika.channel.Channel] = None
        self._subscribers: Dict[str, Callable[[Message], None]] = {}
        self._reply_queues: Dict[str, str] = {}
        self._running = False
        self._consumer_thread: Optional[threading.Thread] = None

    def _connect(self):
        """Establish connection to RabbitMQ."""
        if not self.connection or self.connection.is_closed:
            self.connection = pika.BlockingConnection(self.connection_params)
            self.channel = self.connection.channel()

            # Declare exchanges for different message types
            self.channel.exchange_declare(
                exchange="trading.events", exchange_type="topic", durable=True
            )
            self.channel.exchange_declare(
                exchange="trading.rpc", exchange_type="direct", durable=True
            )

    def publish(self, message: Message) -> None:
        """Publish message to RabbitMQ with idempotence support."""
        self._connect()

        # Add idempotence headers
        headers = {
            "message_id": message.correlation_id or str(uuid.uuid4()),
            "timestamp": str(int(time.time() * 1000)),
        }

        body = json.dumps(message.payload)

        self.channel.basic_publish(
            exchange="trading.events",
            routing_key=message.topic,
            body=body,
            properties=pika.BasicProperties(
                headers=headers,
                delivery_mode=2,  # Persistent messages
                correlation_id=message.correlation_id,
                reply_to=message.reply_to,
            ),
        )
        logger.debug(f"Published message to topic: {message.topic}")

    def subscribe(self, topic: str, handler: Callable[[Message], None]) -> None:
        """Subscribe to topic with RabbitMQ queue."""
        self._connect()
        self._subscribers[topic] = handler

        # Create queue for this topic
        queue_name = f"trading.{topic}"
        self.channel.queue_declare(queue=queue_name, durable=True)
        self.channel.queue_bind(
            exchange="trading.events", queue=queue_name, routing_key=topic
        )

        def message_callback(ch, method, properties, body):
            try:
                payload = json.loads(body)
                message = Message(
                    topic=topic,
                    payload=payload,
                    correlation_id=properties.correlation_id,
                    reply_to=properties.reply_to,
                )
                handler(message)

                # Acknowledge message only after successful processing
                ch.basic_ack(delivery_tag=method.delivery_tag)

            except Exception as e:
                logger.error(f"Error processing message: {e}")
                # Reject and requeue for retry
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

        self.channel.basic_consume(
            queue=queue_name, on_message_callback=message_callback
        )
        logger.info(f"Subscribed to topic: {topic}")

    def request_reply(self, message: Message, timeout_seconds: int = 30) -> Message:
        """RPC pattern with RabbitMQ."""
        self._connect()

        correlation_id = str(uuid.uuid4())

        # Create temporary reply queue
        result = self.channel.queue_declare(queue="", exclusive=True)
        reply_queue = result.method.queue

        response = None

        def on_response(ch, method, properties, body):
            nonlocal response
            if properties.correlation_id == correlation_id:
                response = Message(
                    topic="reply",
                    payload=json.loads(body),
                    correlation_id=properties.correlation_id,
                )

        self.channel.basic_consume(
            queue=reply_queue, on_message_callback=on_response, auto_ack=True
        )

        # Send request
        message.correlation_id = correlation_id
        message.reply_to = reply_queue
        self.publish(message)

        # Wait for response
        self.connection.process_data_events(time_limit=timeout_seconds)

        if response is None:
            raise TimeoutError(f"Request timed out after {timeout_seconds} seconds")

        return response

    def start(self) -> None:
        """Start consuming messages in separate thread."""
        self._running = True
        self._consumer_thread = threading.Thread(target=self._consume_messages)
        self._consumer_thread.start()
        logger.info("RabbitMQ transport started")

    def stop(self) -> None:
        """Stop transport and close connections."""
        self._running = False
        if self._consumer_thread:
            self._consumer_thread.join()
        if self.connection and not self.connection.is_closed:
            self.connection.close()
        logger.info("RabbitMQ transport stopped")

    def _consume_messages(self):
        """Consumer loop running in separate thread."""
        try:
            while self._running:
                self.connection.process_data_events(time_limit=1)
        except Exception as e:
            logger.error(f"Error in consumer thread: {e}")
