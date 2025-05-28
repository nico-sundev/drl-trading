"""In-memory transport for training mode (single process)."""

import logging
import queue
import threading
import uuid
from typing import Callable, Dict

from .transport_interface import Message, TransportInterface

logger = logging.getLogger(__name__)


class InMemoryTransport(TransportInterface):
    """In-memory transport for direct function calls in training mode."""

    def __init__(self):
        self._subscribers: Dict[str, list[Callable[[Message], None]]] = {}
        self._reply_queues: Dict[str, queue.Queue] = {}
        self._lock = threading.Lock()
        self._running = False

    def publish(self, message: Message) -> None:
        """Publish message directly to subscribers (synchronous)."""
        logger.debug(f"Publishing message to topic: {message.topic}")

        with self._lock:
            if message.topic in self._subscribers:
                for handler in self._subscribers[message.topic]:
                    try:
                        handler(message)
                    except Exception as e:
                        logger.error(f"Error in message handler: {e}")

    def subscribe(self, topic: str, handler: Callable[[Message], None]) -> None:
        """Subscribe to a topic."""
        with self._lock:
            if topic not in self._subscribers:
                self._subscribers[topic] = []
            self._subscribers[topic].append(handler)
        logger.info(f"Subscribed to topic: {topic}")

    def request_reply(self, message: Message, timeout_seconds: int = 30) -> Message:
        """Synchronous request-reply pattern."""
        correlation_id = str(uuid.uuid4())
        reply_queue = queue.Queue()

        # Set up temporary reply handler
        reply_topic = f"reply.{correlation_id}"
        self._reply_queues[correlation_id] = reply_queue

        def reply_handler(reply_msg: Message):
            if reply_msg.correlation_id == correlation_id:
                reply_queue.put(reply_msg)

        self.subscribe(reply_topic, reply_handler)

        # Send request with reply info
        message.correlation_id = correlation_id
        message.reply_to = reply_topic
        self.publish(message)

        # Wait for reply
        try:
            reply = reply_queue.get(timeout=timeout_seconds)
            return reply
        except queue.Empty:
            raise TimeoutError(f"Request timed out after {timeout_seconds} seconds")
        finally:
            # Cleanup
            del self._reply_queues[correlation_id]

    def start(self) -> None:
        """Start transport (no-op for in-memory)."""
        self._running = True
        logger.info("In-memory transport started")

    def stop(self) -> None:
        """Stop transport."""
        self._running = False
        self._subscribers.clear()
        self._reply_queues.clear()
        logger.info("In-memory transport stopped")
